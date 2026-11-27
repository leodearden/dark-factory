"""The extraction category's path from a write to graphiti's LLM client (task 9943).

A write's category is passed explicitly from MemoryService to
GraphitiBackend.add_episode, which scopes it over graphiti_core's add_episode,
where CategoryRoutedLLMClient reads it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _fm_helpers import mock_memory_backends
from _fm_polling import poll_until
from _mock_openai_server import mock_openai_server, responses_body
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.prompts.models import Message
from pydantic import BaseModel

from fused_memory.backends.falkor_server_settings import FalkorServerSettings
from fused_memory.backends.graphiti_client import GraphitiBackend
from fused_memory.backends.llm_clients import (
    CategoryRoutedLLMClient,
    extraction_category_scope,
    wrapped_llm_clients,
)
from fused_memory.backends.sdk_client_lifecycle import close_wrapped_sdk_clients
from fused_memory.config.schema import ExtractionRouteConfig, FusedMemoryConfig, PlanRouteConfig
from fused_memory.models.enums import MemoryCategory
from fused_memory.services.memory_service import MemoryService

_DECISIONS = MemoryCategory.decisions_and_rationale


class _RecordingSdk:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _RecordingClient(LLMClient):
    """Answers with its own name; holds a recording SDK client."""

    def __init__(self, name: str) -> None:
        super().__init__(GraphitiLLMConfig(model=f'{name}-model'))
        self.name = name
        self.client = _RecordingSdk()
        self.prompt_names: list[str | None] = []

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
    ) -> dict[str, Any]:
        self.prompt_names.append(prompt_name)
        return {'client': self.name}

    async def _generate_response(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError('generate_response is overridden')


class _ExtractingGraphiti:
    """Stands in for graphiti_core's client: its add_episode makes one extract_edges call."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def add_episode(self, **_kwargs: Any) -> str:
        await self.llm.generate_response(
            [Message(role='user', content='x')], prompt_name='extract_edges.edge',
        )
        return 'episode-result'


@pytest.fixture
def clients() -> tuple[_RecordingClient, _RecordingClient, CategoryRoutedLLMClient]:
    default, route = _RecordingClient('default'), _RecordingClient('route')
    return default, route, CategoryRoutedLLMClient(default, {_DECISIONS: route})


class TestTheBackendScopesTheCategoryOverGraphitisAddEpisode:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('category, answered_by', [
        (_DECISIONS, 'route'),
        (MemoryCategory.entities_and_relations, 'default'),
        (None, 'default'),
    ])
    async def test_the_category_picks_the_extraction_client(
        self, mock_config, make_backend_over_fake_graphiti, clients,
        category: MemoryCategory | None, answered_by: str,
    ) -> None:
        default, route, router = clients
        backend = make_backend_over_fake_graphiti(mock_config, _ExtractingGraphiti(router))

        result = await backend.add_episode(
            name='n', content='c', group_id='test', extraction_category=category,
        )

        assert result == 'episode-result'
        called = {'route': route, 'default': default}[answered_by]
        uncalled = default if called is route else route
        assert called.prompt_names == ['extract_edges.edge']
        assert uncalled.prompt_names == []

    @pytest.mark.asyncio
    async def test_the_category_is_optional(
        self, mock_config, make_backend_over_fake_graphiti, clients,
    ) -> None:
        default, route, router = clients
        backend = make_backend_over_fake_graphiti(mock_config, _ExtractingGraphiti(router))

        await backend.add_episode(name='n', content='c', group_id='test')

        assert (default.prompt_names, route.prompt_names) == (['extract_edges.edge'], [])


class TestCloseReleasesEveryRoutesSdkClient:
    @pytest.mark.asyncio
    async def test_the_route_and_default_sdk_clients_are_closed(self, mock_config, clients) -> None:
        default, route, router = clients
        driver = AsyncMock()
        backend = GraphitiBackend.with_driver(
            mock_config, driver, registered_graph_ids=(), llm_client=router,
        )

        await backend.close()

        assert (default.client.closed, route.client.closed) == (True, True)


@dataclass(frozen=True)
class _QueuedService:
    svc: MemoryService
    graphiti_initialize: AsyncMock
    graphiti_add_episode: AsyncMock

    def injected_llm_client(self) -> Any:
        injected = self.graphiti_initialize.await_args
        return injected.kwargs.get('llm_client') if injected else None

    async def landed_extraction_category(self) -> MemoryCategory | None:
        await poll_until(lambda: self.graphiti_add_episode.await_count, message='no graphiti write landed')
        assert self.graphiti_add_episode.await_args is not None
        return self.graphiti_add_episode.await_args.kwargs['extraction_category']


@asynccontextmanager
async def _queued_service(config: FusedMemoryConfig, tmp_path: Path) -> AsyncIterator[_QueuedService]:
    """A MemoryService over a real durable queue whose graphiti backend is an AsyncMock."""
    svc = mock_memory_backends(MemoryService(config))
    queued = _QueuedService(svc, AsyncMock(), AsyncMock(return_value=None))
    svc.graphiti.initialize = queued.graphiti_initialize
    svc.graphiti.add_episode = queued.graphiti_add_episode
    svc.graphiti.list_graphs = AsyncMock(return_value=[])
    svc.graphiti.read_server_settings = AsyncMock(return_value=FalkorServerSettings(
        expected_graph_timeout_ms=30000, graph_timeout_ms=30000, appendonly='yes',
    ))
    await svc.initialize()
    svc.set_known_projects({'main': str(tmp_path)})
    try:
        yield queued
    finally:
        await svc.close()
        await close_wrapped_sdk_clients(wrapped_llm_clients(queued.injected_llm_client()))


class TestMemoryServicePassesTheWritesCategory:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('category', [_DECISIONS, MemoryCategory.entities_and_relations])
    async def test_an_add_memory_write_carries_its_category(
        self, mock_config, tmp_path: Path, category: MemoryCategory,
    ) -> None:
        async with _queued_service(mock_config, tmp_path) as queued:
            await queued.svc.add_memory(
                content='Leo decided the plan route parks capped writes',
                category=category.value, project_id='main',
            )

            assert await queued.landed_extraction_category() is category

    @pytest.mark.asyncio
    async def test_an_add_episode_write_carries_none(self, mock_config, tmp_path: Path) -> None:
        async with _queued_service(mock_config, tmp_path) as queued:
            await queued.svc.add_episode(content='Alice met Bob.', project_id='main')

            assert await queued.landed_extraction_category() is None


class _Edge(BaseModel):
    fact: str


def _route_decisions_through(config: FusedMemoryConfig, proxy_url: str) -> None:
    config.plan_route = PlanRouteConfig(proxy_url=proxy_url)
    config.llm.extraction_routes = {
        'decisions_and_rationale': ExtractionRouteConfig(
            model='gpt-6.1-sol', reasoning_effort='low', api_url=proxy_url, api_key='plan-key',
        ),
    }


class TestInitializeBuildsTheGatedRoute:
    @pytest.mark.asyncio
    async def test_the_injected_client_routes_decisions_to_the_plan_proxy(
        self, mock_config, tmp_path: Path, monkeypatch,
    ) -> None:
        for name in ('OPENAI_BASE_URL', 'OPENAI_API_BASE', 'OPENAI_API_URL'):
            monkeypatch.delenv(name, raising=False)
        with mock_openai_server() as proxy:
            proxy.set_response('/responses', responses_body(json.dumps({'fact': 'f'})))
            _route_decisions_through(mock_config, proxy.base_url)

            async with _queued_service(mock_config, tmp_path) as queued:
                client = queued.injected_llm_client()
                assert isinstance(client, CategoryRoutedLLMClient)
                with extraction_category_scope(_DECISIONS):
                    await client.generate_response(
                        [Message(role='user', content='x')], response_model=_Edge,
                        prompt_name='extract_edges.edge',
                    )

            [sent] = proxy.requests_to('/responses')
            assert sent['json_body']['model'] == 'gpt-6.1-sol'

    @pytest.mark.asyncio
    async def test_startup_logs_the_active_route_once_and_nothing_off(
        self, mock_config, tmp_path: Path, monkeypatch, caplog,
    ) -> None:
        for name in ('OPENAI_BASE_URL', 'OPENAI_API_BASE', 'OPENAI_API_URL'):
            monkeypatch.delenv(name, raising=False)
        with mock_openai_server() as proxy:
            _route_decisions_through(mock_config, proxy.base_url)

            with caplog.at_level(logging.DEBUG):
                async with _queued_service(mock_config, tmp_path):
                    pass

        messages = [(r.levelno, r.getMessage()) for r in caplog.records]
        [(level, line)] = [
            (level, message) for level, message in messages
            if message.startswith('Graphiti LLM:') and 'decisions_and_rationale' in message
        ]
        assert level == logging.INFO
        assert 'gpt-6.1-sol' in line
        assert 'effort low' in line
        assert proxy.base_url in line
        assert 'plan-key' not in line
        assert [m for lv, m in messages if lv >= logging.WARNING and 'decisions_and_rationale' in m] == []
        assert [m for _, m in messages if 'is OFF' in m] == []
