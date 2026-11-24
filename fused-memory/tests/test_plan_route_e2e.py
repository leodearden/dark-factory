"""The plan route end to end (task 9944).

Two real sockets: ``proxy`` is the ChatGPT-plan proxy and ``api`` stands in for
api.openai.com. A capped plan must stay on the proxy: not one plan-routed call
may fall back onto the API-key route.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from _mock_openai_server import MockOpenAIServer, mock_openai_server
from graphiti_core.llm_client.errors import RateLimitError as GraphitiRateLimitError
from graphiti_core.prompts.models import Message
from pydantic import BaseModel

from fused_memory.backends.graphiti_client import build_llm_client
from fused_memory.config.schema import (
    EmbedderConfig,
    EmbedderProvidersConfig,
    FusedMemoryConfig,
    LLMConfig,
    LLMProvidersConfig,
    OpenAIProviderConfig,
    PlanRouteConfig,
)
from fused_memory.services.plan_route import PlanRoute, PlanRouteSite
from fused_memory.services.provider_billing import BillingSource, classify_billing_exhaustion

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
