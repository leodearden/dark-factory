"""The plan route end to end (task 9944).

Two real sockets: ``proxy`` is the ChatGPT-plan proxy and ``api`` stands in for
api.openai.com. A capped plan must stay on the proxy: not one plan-routed call
may fall back onto the API-key route.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import types
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest
from _fm_helpers import mock_memory_backends
from _fm_polling import poll_until
from _mock_openai_server import (
    MockOpenAIServer,
    chat_completion_body,
    mock_openai_server,
    responses_body,
)
from graphiti_core.llm_client.errors import RateLimitError as GraphitiRateLimitError
from graphiti_core.prompts.models import Message
from pydantic import BaseModel
from shared.project_layout import escalation_queue_dir
from shared.testing_fake_clock import FakeClock
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
from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.models.scope import configured_primary_root
from fused_memory.server.write_triage import OUTCOME_JUDGE, OUTCOME_RESTATED, BandDecision
from fused_memory.server.write_triage_judge import judge_write
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


_CLOCK_START = datetime(2026, 11, 24, 12, 0, tzinfo=UTC).timestamp()
_CAPPED = 'plan_route_capped'

_GraphitiStandIn = Callable[[MemoryService], Coroutine[Any, Any, None]]


async def _extract_on_the_plan_route(svc: MemoryService) -> None:
    """Stands in for graphiti's add_episode: one extraction call on the plan route."""
    provider = svc.config.llm.providers.openai
    assert provider is not None and provider.api_url is not None
    http_client = svc.plan_route.http_client_for(provider.api_url, site=PlanRouteSite.GRAPHITI)
    assert http_client is not None
    try:
        await _extract(build_llm_client(svc.config, http_client=http_client))
    finally:
        await http_client.aclose()


async def _extract_off_the_plan_route(svc: MemoryService) -> None:
    """Stands in for a plan call some site forgot to gate: the same proxy, no gated client."""
    provider = svc.config.llm.providers.openai
    assert provider is not None and provider.api_url is not None
    async with openai.AsyncOpenAI(
        base_url=provider.api_url, api_key='plan-key', max_retries=0,
    ) as client:
        await client.chat.completions.create(
            model='mock-model', messages=[{'role': 'user', 'content': 'Alice knows Bob.'}],
        )


@asynccontextmanager
async def _routed_service(
    config: FusedMemoryConfig, tmp_path: Path, endpoints: _Endpoints, *,
    clock: FakeClock, graphiti_stand_in: _GraphitiStandIn,
) -> AsyncIterator[MemoryService]:
    """A MemoryService over a real queue and journal on *clock*, whose graphiti
    calls the plan proxy through *graphiti_stand_in*."""
    from fused_memory.services.write_journal import WriteJournal

    routed = endpoints.config()
    config.llm = routed.llm
    config.embedder = routed.embedder
    config.plan_route = PlanRouteConfig(proxy_url=endpoints.proxy.base_url)
    svc = mock_memory_backends(MemoryService(config, queue_clock=clock))
    graphiti_initialize = AsyncMock()
    svc.graphiti.initialize = graphiti_initialize
    svc.graphiti.list_graphs = AsyncMock(return_value=[])
    svc.graphiti.read_server_settings = AsyncMock(return_value=FalkorServerSettings(
        expected_graph_timeout_ms=30000, graph_timeout_ms=30000, appendonly='yes',
    ))

    async def add_episode(*_args: Any, **_kwargs: Any) -> None:
        await graphiti_stand_in(svc)

    svc.graphiti.add_episode = add_episode
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
        injected = graphiti_initialize.await_args
        await close_wrapped_sdk_clients([injected.kwargs.get('llm_client') if injected else None])


def _interval(svc: MemoryService) -> float:
    return svc.config.plan_route.probe_interval_seconds


def _primary_root(svc: MemoryService) -> Path:
    root = configured_primary_root(svc.config.taskmaster)
    assert root, 'the test config must pin a primary project root'
    return Path(root)


def _capped_records(root: Path) -> list[dict]:
    records = (json.loads(p.read_text()) for p in escalation_queue_dir(root).rglob('esc-*.json'))
    return [r for r in records if r['category'] == _CAPPED]


def _refreshed(root: Path, page: dict) -> list[dict] | None:
    return [r for r in _capped_records(root) if r['dedupe_count'] > page['dedupe_count']] or None


def _queue_rows(tmp_path: Path) -> list[dict]:
    with contextlib.closing(sqlite3.connect(tmp_path / 'queue' / 'write_queue.db')) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute('SELECT * FROM write_queue')]


def _parked_row(tmp_path: Path) -> dict:
    [parked] = [row for row in _queue_rows(tmp_path) if row['status'] == 'parked']
    return parked


def _proxy_chats(endpoints: _Endpoints) -> int:
    return len(endpoints.proxy.requests_to(_CHAT))


async def _counts(svc: MemoryService) -> dict:
    assert svc.durable_queue is not None
    return (await svc.durable_queue.get_stats())['counts']


async def _parked(svc: MemoryService) -> dict | None:
    counts = await _counts(svc)
    return counts if counts.get('parked', 0) >= 1 else None


async def _drained(svc: MemoryService) -> dict | None:
    counts = await _counts(svc)
    return counts if counts.get('completed', 0) >= 1 and not counts.get('parked') else None


async def _recovered_and_drained(svc: MemoryService) -> bool:
    state = (await svc.get_status())['plan_route']['state']
    return state == 'ok' and await _drained(svc) is not None


async def _add_a_graphiti_write(svc: MemoryService) -> None:
    response = await svc.add_memory(
        content='Leo decided the plan route parks capped writes',
        category='decisions_and_rationale', project_id='main',
    )
    assert response.stores_queued == [SourceStore.graphiti]


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


#: The page's count of the one write: in flight while a gated call trips the
#: cap, parked by the time an ungated cap reaches the park policy.
_ONE_WRITE_WAITING = 'parked_or_in_flight_writes=1'


@pytest.mark.skipif(
    not _folded_escalation.HAS_ESCALATION, reason='escalation package unavailable (minimal env)',
)
class TestACappedPlanThroughMemoryService:
    @pytest.mark.asyncio
    async def test_a_capped_write_parks_behind_one_l2_and_drains_when_the_plan_returns(
        self, mock_config: FusedMemoryConfig, tmp_path: Path, endpoints: _Endpoints,
    ) -> None:
        endpoints.cap()
        clock = FakeClock(start=_CLOCK_START)
        async with _routed_service(
            mock_config, tmp_path, endpoints,
            clock=clock, graphiti_stand_in=_extract_on_the_plan_route,
        ) as svc:
            await _add_a_graphiti_write(svc)

            await poll_until(lambda: _parked(svc), message='nothing parked')
            assert endpoints.api_llm_requests() == []

            capped = (await svc.get_status())['plan_route']
            assert capped['state'] == 'capped'
            assert capped['parked_items'] >= 1
            parked = _parked_row(tmp_path)
            assert parked['next_retry_at'] == _epoch(capped['capped_since']) + _interval(svc)
            assert parked['attempts'] == 0
            assert (await _counts(svc)).get('dead', 0) == 0
            assert svc.durable_queue is not None
            assert await svc.durable_queue.get_dead_items() == []
            assert _proxy_chats(endpoints) == 1

            root = _primary_root(svc)
            [page] = await poll_until(
                lambda: _capped_records(root) or None, message='no plan_route_capped page was filed',
            )
            assert page['level'] == 2
            assert page['severity'] == 'blocking'
            assert capped['escalation_id'] == page['id']
            assert _ONE_WRITE_WAITING in page['detail'].splitlines()

            clock.advance(_interval(svc))
            [confirmed] = await poll_until(
                lambda: _refreshed(root, page), message='a re-probe did not refresh the page',
            )
            assert confirmed['id'] == page['id']
            assert _ONE_WRITE_WAITING in confirmed['detail'].splitlines()
            await poll_until(lambda: _parked(svc), message='the failed probe did not re-park')
            assert _parked_row(tmp_path)['next_retry_at'] == clock() + _interval(svc)
            assert _proxy_chats(endpoints) == 2
            assert len(_capped_records(root)) == 1

            endpoints.uncap()
            clock.advance(_interval(svc))
            await poll_until(lambda: _drained(svc), message='the parked write never drained')

            assert (await svc.get_status())['plan_route']['state'] == 'ok'
            assert _proxy_chats(endpoints) == 3
            assert [r['status'] for r in _capped_records(root)] == ['pending']
            assert endpoints.api_llm_requests() == []

    @pytest.mark.asyncio
    async def test_an_ungated_cap_reprobes_once_per_period_and_clears_when_its_write_lands(
        self, mock_config: FusedMemoryConfig, tmp_path: Path, endpoints: _Endpoints,
    ) -> None:
        endpoints.cap()
        clock = FakeClock(start=_CLOCK_START)
        async with _routed_service(
            mock_config, tmp_path, endpoints,
            clock=clock, graphiti_stand_in=_extract_off_the_plan_route,
        ) as svc:
            await _add_a_graphiti_write(svc)

            await poll_until(lambda: _parked(svc), message='nothing parked')
            root = _primary_root(svc)
            [page] = await poll_until(
                lambda: _capped_records(root) or None, message='no plan_route_capped page was filed',
            )
            assert page['level'] == 2
            parked = _parked_row(tmp_path)
            assert parked['next_retry_at'] == _CLOCK_START + _interval(svc)
            detail = page['detail'].splitlines()
            assert _ONE_WRITE_WAITING in detail
            assert f"sites_affected=durable_queue:{parked['operation']}:1" in detail
            assert (await svc.get_status())['plan_route']['state'] == 'capped'

            clock.advance(_interval(svc))
            await poll_until(
                lambda: _refreshed(root, page), message='a due ungated cap did not refresh the page',
            )

            assert _parked_row(tmp_path)['next_retry_at'] == clock() + _interval(svc)
            assert _proxy_chats(endpoints) == 2

            endpoints.uncap()
            clock.advance(_interval(svc))
            await poll_until(
                lambda: _recovered_and_drained(svc),
                message='the landed write did not drain the queue and clear the route',
            )

            assert _proxy_chats(endpoints) == 3
            assert [r['status'] for r in _capped_records(root)] == ['pending']
            assert endpoints.api_llm_requests() == []


_RESPONSES = '/responses'
_RESTATES = '{"verdict": "restates", "candidate_id": "m1"}'


def _judge_config(endpoints: _Endpoints) -> FusedMemoryConfig:
    """The judge on the plan proxy, the main LLM arm on the API-key endpoint.

    Built as tests/server/test_write_triage_judge.py::_real_config builds its
    config: the write_triage leaves are assigned after construction.
    """
    cfg = FusedMemoryConfig(
        llm=LLMConfig(
            provider='openai', model='gpt-4o-mini',
            providers=LLMProvidersConfig(
                openai=OpenAIProviderConfig(api_key='sk-main', api_url=endpoints.api.base_url),
            ),
        ),
    )
    object.__setattr__(cfg.write_triage, 'judge_provider', 'openai')
    object.__setattr__(cfg.write_triage, 'judge_api_url', endpoints.proxy.base_url)
    object.__setattr__(cfg.write_triage, 'judge_api_key', 'sk-plan-proxy')
    return cfg


async def _judge(memory_service: object) -> Any:
    candidate = MemoryResult(
        id='m1', content='some procedural content',
        category=MemoryCategory.procedural_knowledge, source_store=SourceStore.mem0,
        relevance_score=0.5, metadata={'store_rank': 1, 'store_score': 0.80},
    )
    return await judge_write(
        memory_service=memory_service, content='c', project_id='p',
        decision=BandDecision(OUTCOME_JUDGE, 'm1', 0.80, 0.95, 0.70), candidates=[candidate],
    )


class TestTheWriteTriageJudgeRidesThePlanRoute:
    @pytest.mark.asyncio
    async def test_a_gated_judge_lands_on_the_proxy(self, endpoints: _Endpoints) -> None:
        endpoints.proxy.set_response(_RESPONSES, responses_body(_RESTATES))
        service = types.SimpleNamespace(config=_judge_config(endpoints), plan_route=endpoints.route())

        verdict = await _judge(service)

        assert (verdict.outcome, verdict.candidate_id) == (OUTCOME_RESTATED, 'm1')
        assert len(endpoints.proxy.requests_to(_RESPONSES)) == 1
        assert endpoints.api.requests == []

    @pytest.mark.asyncio
    async def test_a_capped_plan_fails_the_judge_once_and_never_falls_back(
        self, endpoints: _Endpoints,
    ) -> None:
        endpoints.proxy.set_response_sequence(_RESPONSES, [(429, _USAGE_LIMIT)])
        route = endpoints.route()
        service = types.SimpleNamespace(config=_judge_config(endpoints), plan_route=route)

        with pytest.raises(openai.RateLimitError) as first:
            await _judge(service)
        with pytest.raises(openai.RateLimitError) as second:
            await _judge(service)

        for failure in (first.value, second.value):
            exhaustion = classify_billing_exhaustion(failure)
            assert exhaustion is not None
            assert exhaustion.source is BillingSource.PLAN_CAP
        assert len(endpoints.proxy.requests_to(_RESPONSES)) == 1
        assert endpoints.api.requests == []
        status = route.state().to_status(parked_items=0, escalation_id=None)
        assert status['judge_fail_opens_since_latch'] == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize('plan_route', ['absent', 'not_a_plan_route'])
    async def test_a_service_without_a_plan_route_judges_ungated(
        self, endpoints: _Endpoints, plan_route: str,
    ) -> None:
        endpoints.proxy.set_response_sequence(_RESPONSES, [(429, _USAGE_LIMIT)])
        cfg = _judge_config(endpoints)
        service = (
            types.SimpleNamespace(config=cfg) if plan_route == 'absent'
            else types.SimpleNamespace(config=cfg, plan_route=MagicMock())
        )

        for _ in range(2):
            with pytest.raises(openai.RateLimitError):
                await _judge(service)

        assert len(endpoints.proxy.requests_to(_RESPONSES)) == 2
        endpoints.proxy.set_response(_RESPONSES, responses_body(_RESTATES))
        assert (await _judge(service)).outcome == OUTCOME_RESTATED
