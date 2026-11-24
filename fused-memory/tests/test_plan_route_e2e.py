"""The plan route end to end (task 9944).

Two real sockets: ``proxy`` is the ChatGPT-plan proxy and ``api`` stands in for
api.openai.com. A capped plan must stay on the proxy: not one plan-routed call
may fall back onto the API-key route.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _fm_helpers import mock_memory_backends
from _fm_polling import poll_until
from _mock_openai_server import MockOpenAIServer, chat_completion_body, mock_openai_server
from graphiti_core.llm_client.errors import RateLimitError as GraphitiRateLimitError
from graphiti_core.prompts.models import Message
from pydantic import BaseModel
from shared.project_layout import escalation_queue_dir
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.backends.falkor_server_settings import FalkorServerSettings
from fused_memory.backends.graphiti_client import build_llm_client
from fused_memory.backends.sdk_client_lifecycle import close_wrapped_sdk_clients
from fused_memory.config.schema import (
    EmbedderConfig,
    EmbedderProvidersConfig,
    FusedMemoryConfig,
    LLMConfig,
    LLMProvidersConfig,
    OpenAIProviderConfig,
    PlanRouteConfig,
)
from fused_memory.middleware import _folded_escalation
from fused_memory.models.enums import SourceStore
from fused_memory.models.scope import configured_primary_root
from fused_memory.services.memory_service import MemoryService
from fused_memory.services.plan_route import PlanRoute, PlanRouteSite
from fused_memory.services.provider_billing import BillingSource, classify_billing_exhaustion

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_CHAT = '/chat/completions'
_USAGE_LIMIT = {'error': {
    'type': 'usage_limit_reached', 'code': 'usage_limit_reached',
    'message': 'The usage limit has been reached',
}}


class _Item(BaseModel):
    name: str


class _Extraction(BaseModel):
    items: list[_Item]


_EXTRACTION = json.dumps({'items': [{'name': 'Alice'}]})


class _Endpoints:
    def __init__(self, proxy: MockOpenAIServer, api: MockOpenAIServer) -> None:
        self.proxy = proxy
        self.api = api
        proxy.chat_content = _EXTRACTION
        api.chat_content = _EXTRACTION

    def config(self) -> FusedMemoryConfig:
        return FusedMemoryConfig(
            llm=LLMConfig(
                provider='openai', client_class='openai_generic', model='mock-model',
                providers=LLMProvidersConfig(
                    openai=OpenAIProviderConfig(api_key='plan-key', api_url=self.proxy.base_url),
                ),
            ),
            embedder=EmbedderConfig(
                provider='openai', model='mock-embedding-model',
                providers=EmbedderProvidersConfig(
                    openai=OpenAIProviderConfig(api_key='api-key', api_url=self.api.base_url),
                ),
            ),
        )

    def route(self, probe_interval_seconds: float = 60.0) -> PlanRoute:
        return PlanRoute(PlanRouteConfig(
            proxy_url=self.proxy.base_url, probe_interval_seconds=probe_interval_seconds,
        ))

    def cap(self) -> None:
        self.proxy.set_response_sequence(_CHAT, [(429, _USAGE_LIMIT)])

    def uncap(self) -> None:
        self.proxy.set_response(_CHAT, chat_completion_body(_EXTRACTION))

    def api_llm_requests(self) -> list[dict[str, Any]]:
        return self.api.requests_to(_CHAT) + self.api.requests_to('/responses')


@pytest.fixture
def endpoints(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Endpoints]:
    for name in ('OPENAI_BASE_URL', 'OPENAI_API_BASE', 'OPENAI_API_URL'):
        monkeypatch.delenv(name, raising=False)
    with mock_openai_server() as proxy, mock_openai_server() as api:
        yield _Endpoints(proxy, api)


@asynccontextmanager
async def _graphiti_llm(endpoints: _Endpoints, route: PlanRoute) -> AsyncIterator[Any]:
    http_client = route.http_client_for(endpoints.proxy.base_url, site=PlanRouteSite.GRAPHITI)
    assert http_client is not None
    try:
        client = build_llm_client(endpoints.config(), http_client=http_client)
        assert client is not None
        yield client
    finally:
        await http_client.aclose()


async def _extract(client: Any) -> Any:
    return await client.generate_response(
        [Message(role='user', content='Alice knows Bob.')], response_model=_Extraction,
    )


class TestGraphitisLLMRidesThePlanRoute:
    @pytest.mark.asyncio
    async def test_a_gated_client_lands_on_the_proxy(self, endpoints: _Endpoints) -> None:
        async with _graphiti_llm(endpoints, endpoints.route()) as client:
            assert await _extract(client) == {'items': [{'name': 'Alice'}]}

        assert len(endpoints.proxy.requests_to(_CHAT)) == 1
        assert endpoints.api_llm_requests() == []

    @pytest.mark.asyncio
    async def test_a_capped_plan_raises_once_and_never_falls_back(
        self, endpoints: _Endpoints,
    ) -> None:
        endpoints.cap()

        async with _graphiti_llm(endpoints, endpoints.route()) as client:
            with pytest.raises(GraphitiRateLimitError) as first:
                await _extract(client)
            assert len(endpoints.proxy.requests_to(_CHAT)) == 1
            with pytest.raises(GraphitiRateLimitError):
                await _extract(client)

        exhaustion = classify_billing_exhaustion(first.value)
        assert exhaustion is not None
        assert exhaustion.source is BillingSource.PLAN_CAP
        assert len(endpoints.proxy.requests_to(_CHAT)) == 1
        assert endpoints.api_llm_requests() == []


_PROBE_INTERVAL_SECONDS = 1.0
_CAPPED = 'plan_route_capped'


async def _extract_on_the_plan_route(svc: MemoryService, *_args: Any, **_kwargs: Any) -> None:
    """Stands in for graphiti's add_episode: one extraction call on the plan route."""
    provider = svc.config.llm.providers.openai
    assert provider is not None and provider.api_url is not None
    http_client = svc.plan_route.http_client_for(provider.api_url, site=PlanRouteSite.GRAPHITI)
    assert http_client is not None
    try:
        await _extract(build_llm_client(svc.config, http_client=http_client))
    finally:
        await http_client.aclose()


@asynccontextmanager
async def _routed_service(
    config: FusedMemoryConfig, tmp_path: Path, endpoints: _Endpoints,
) -> AsyncIterator[MemoryService]:
    """A MemoryService over a real queue and journal whose graphiti rides the plan route."""
    from fused_memory.services.write_journal import WriteJournal

    routed = endpoints.config()
    config.llm = routed.llm
    config.embedder = routed.embedder
    config.plan_route = PlanRouteConfig(
        proxy_url=endpoints.proxy.base_url, probe_interval_seconds=_PROBE_INTERVAL_SECONDS,
    )
    svc = mock_memory_backends(MemoryService(config))
    svc.graphiti.list_graphs = AsyncMock(return_value=[])
    svc.graphiti.read_server_settings = AsyncMock(return_value=FalkorServerSettings(
        expected_graph_timeout_ms=30000, graph_timeout_ms=30000, appendonly='yes',
    ))
    svc.graphiti.add_episode = lambda *args, **kwargs: _extract_on_the_plan_route(
        svc, *args, **kwargs,
    )
    await svc.initialize()
    journal = WriteJournal(tmp_path / 'wj_plan_route')
    await journal.initialize()
    svc.set_write_journal(journal)
    svc.set_known_projects({'main': str(tmp_path)})
    try:
        yield svc
    finally:
        await svc.close()
        await journal.close()
        injected = svc.graphiti.initialize.await_args
        await close_wrapped_sdk_clients([injected.kwargs.get('llm_client') if injected else None])


def _primary_root(svc: MemoryService) -> Path:
    root = configured_primary_root(svc.config.taskmaster)
    assert root, 'the test config must pin a primary project root'
    return Path(root)


def _capped_records(root: Path) -> list[dict]:
    records = (json.loads(p.read_text()) for p in escalation_queue_dir(root).rglob('esc-*.json'))
    return [r for r in records if r['category'] == _CAPPED]


def _queue_rows(tmp_path: Path) -> list[dict]:
    with contextlib.closing(sqlite3.connect(tmp_path / 'queue' / 'write_queue.db')) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute('SELECT * FROM write_queue')]


async def _counts(svc: MemoryService) -> dict:
    assert svc.durable_queue is not None
    return (await svc.durable_queue.get_stats())['counts']


async def _parked(svc: MemoryService) -> dict | None:
    counts = await _counts(svc)
    return counts if counts.get('parked', 0) >= 1 else None


async def _drained(svc: MemoryService) -> dict | None:
    counts = await _counts(svc)
    return counts if counts.get('completed', 0) >= 1 and not counts.get('parked') else None


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


@pytest.mark.skipif(
    not _folded_escalation.HAS_ESCALATION, reason='escalation package unavailable (minimal env)',
)
class TestACappedPlanThroughMemoryService:
    @pytest.mark.asyncio
    async def test_a_capped_write_parks_behind_one_l2_and_drains_when_the_plan_returns(
        self, mock_config: FusedMemoryConfig, tmp_path: Path, endpoints: _Endpoints,
    ) -> None:
        endpoints.cap()
        async with _routed_service(mock_config, tmp_path, endpoints) as svc:
            response = await svc.add_memory(
                content='Leo decided the plan route parks capped writes',
                category='decisions_and_rationale', project_id='main',
            )

            assert response.stores_queued == [SourceStore.graphiti]
            await poll_until(lambda: _parked(svc), timeout=20.0, message='nothing parked')
            assert endpoints.api_llm_requests() == []

            capped = (await svc.get_status())['plan_route']
            assert capped['state'] == 'capped'
            assert capped['capped_since'] is not None
            assert capped['parked_items'] >= 1
            [parked] = [row for row in _queue_rows(tmp_path) if row['status'] == 'parked']
            assert parked['next_retry_at'] >= (
                _epoch(capped['capped_since']) + _PROBE_INTERVAL_SECONDS - 0.01
            )
            assert parked['attempts'] == 0
            assert (await _counts(svc)).get('dead', 0) == 0
            assert svc.durable_queue is not None
            assert await svc.durable_queue.get_dead_items() == []

            root = _primary_root(svc)
            [page] = await poll_until(
                lambda: _capped_records(root) or None, timeout=20.0,
                message='no plan_route_capped page was filed',
            )
            assert page['level'] == 2
            assert page['severity'] == 'blocking'
            assert capped['escalation_id'] == page['id']

            filed_count = page['dedupe_count']
            [confirmed] = await poll_until(
                lambda: [r for r in _capped_records(root) if r['dedupe_count'] > filed_count]
                or None,
                timeout=20.0, message='a re-probe did not refresh the page',
            )
            assert confirmed['id'] == page['id']
            assert len(_capped_records(root)) == 1

            endpoints.uncap()
            await poll_until(
                lambda: _drained(svc), timeout=20.0, message='the parked write never drained',
            )
            assert (await svc.get_status())['plan_route']['state'] == 'ok'
            assert [r['status'] for r in _capped_records(root)] == ['pending']
            assert endpoints.api_llm_requests() == []
