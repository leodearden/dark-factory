"""Fused-memory owners release the async SDK clients they hold (task 4417).

An ``AsyncOpenAI`` / ``AsyncAnthropic`` that is dropped without ``close()`` is
finalised at an arbitrary later point, where it can resurrect an ``aclose()``
task onto an unrelated event loop (the mechanism is documented once, at
tests/_fm_helpers.py::reap_leaked_async_httpx_clients). These tests drive each
owner through its public ``close()`` and assert that nothing it built is left
open.
"""

import logging
import types
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from _fm_helpers import (
    _leaked_async_httpx_clients,
    reap_leaked_async_httpx_clients,
    track_async_httpx_clients,
)
from _mock_openai_server import mock_openai_server
from _sdk_client_stubs import completion_client
from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.llm_client.anthropic_client import AnthropicClient

import fused_memory.backends.graphiti_client as graphiti_client_module
from fused_memory.backends.graphiti_client import (
    GraphitiBackend,
    build_embedder,
    build_llm_client,
)
from fused_memory.backends.llm_clients import TokenRecordingOpenAIGenericClient
from fused_memory.backends.sdk_client_lifecycle import close_wrapped_sdk_clients
from fused_memory.maintenance.reindex import ReindexResult, run_reindex
from fused_memory.middleware.task_curator import TaskCurator
from fused_memory.routing.classifier import WriteClassifier
from fused_memory.routing.router import ReadRouter

_LIFECYCLE_LOGGER = 'fused_memory.backends.sdk_client_lifecycle'


def _real_wrappers() -> list:
    return [
        OpenAIClient(config=LLMConfig(api_key='test-key')),
        OpenAIEmbedder(config=OpenAIEmbedderConfig(api_key='test-key')),
        OpenAIRerankerClient(config=LLMConfig(api_key='test-key')),
        AnthropicClient(config=LLMConfig(api_key='test-key')),
    ]


def _awaitable_sdk() -> types.SimpleNamespace:
    return types.SimpleNamespace(close=AsyncMock())


class TestCloseWrappedSdkClients:
    @pytest.mark.asyncio
    async def test_closes_the_sdk_client_behind_each_graphiti_wrapper(self):
        wrappers = _real_wrappers()
        assert [w.client.is_closed() for w in wrappers] == [False] * len(wrappers)

        await close_wrapped_sdk_clients(wrappers)

        assert [w.client.is_closed() for w in wrappers] == [True] * len(wrappers)

    @pytest.mark.asyncio
    async def test_closes_the_client_behind_the_usage_capturing_delegate(self):
        wrapper = TokenRecordingOpenAIGenericClient(config=LLMConfig(api_key='test-key'))
        assert wrapper.client.is_closed() is False

        await close_wrapped_sdk_clients([wrapper])

        assert wrapper.client.is_closed() is True

    @pytest.mark.asyncio
    async def test_a_client_shared_by_two_wrappers_is_closed_once(self):
        sdk = _awaitable_sdk()

        await close_wrapped_sdk_clients(
            [types.SimpleNamespace(client=sdk), types.SimpleNamespace(client=sdk)],
        )

        assert sdk.close.await_count == 1

    @pytest.mark.asyncio
    async def test_entries_it_cannot_close_are_skipped_without_raising(self):
        sync_close_sdk = MagicMock()
        later = OpenAIClient(config=LLMConfig(api_key='test-key'))

        await close_wrapped_sdk_clients([
            None,
            object(),
            types.SimpleNamespace(client=sync_close_sdk),
            later,
        ])

        sync_close_sdk.close.assert_not_called()
        assert later.client.is_closed() is True

    @pytest.mark.asyncio
    async def test_a_failing_close_is_logged_and_the_sweep_continues(self, caplog):
        failing_sdk = types.SimpleNamespace(
            close=AsyncMock(side_effect=RuntimeError('Event loop is closed')),
        )
        later_sdk = _awaitable_sdk()

        with caplog.at_level(logging.WARNING, logger=_LIFECYCLE_LOGGER):
            await close_wrapped_sdk_clients([
                types.SimpleNamespace(client=failing_sdk),
                types.SimpleNamespace(client=later_sdk),
            ])

        later_sdk.close.assert_awaited_once()
        warnings = [
            r for r in caplog.records
            if r.name == _LIFECYCLE_LOGGER and r.levelno == logging.WARNING
        ]
        assert len(warnings) == 1
        assert type(failing_sdk).__name__ in warnings[0].getMessage()


async def _start_census() -> None:
    track_async_httpx_clients()
    await reap_leaked_async_httpx_clients()


@pytest.fixture
def async_httpx_clients_built(monkeypatch) -> list:
    """Every ``httpx.AsyncClient`` constructed during the test, held by strong reference.

    The census above sees only clients still open, so it cannot tell "built
    nothing" from "built one and closed it"; this list can.
    """
    built: list = []
    original_init = httpx.AsyncClient.__init__

    def recording_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        built.append(self)

    monkeypatch.setattr(httpx.AsyncClient, '__init__', recording_init)
    return built


@pytest.fixture
def offline_graphiti(monkeypatch):
    """A real Graphiti over a spec'd driver double: no FalkorDB, no preflight, no extraction."""
    def driver_factory(**_kwargs):
        driver = MagicMock(spec=FalkorDriver)
        driver.clone.return_value = MagicMock(spec=FalkorDriver)
        return driver

    monkeypatch.setattr(graphiti_client_module, '_MultiTenantFalkorDriver', driver_factory)
    monkeypatch.setattr(graphiti_client_module, 'check_openai_responses_api', lambda: None)
    monkeypatch.setattr(Graphiti, 'add_episode', AsyncMock())


async def _write_one_episode(backend: GraphitiBackend, group_id: str) -> None:
    await backend.add_episode(name='e', content='c', group_id=group_id)
    Graphiti.add_episode.assert_awaited_once()
    assert Graphiti.add_episode.await_args.kwargs['group_id'] == group_id


class TestGraphitiBackendCloseReleasesSdkClients:
    @pytest.mark.asyncio
    async def test_configured_sub_clients_are_shared_and_released(
        self, mock_config, offline_graphiti,
    ):
        await _start_census()
        backend = GraphitiBackend(mock_config)
        await backend.initialize(skip_maintenance=True)
        await _write_one_episode(backend, 'g1')

        built = _leaked_async_httpx_clients()
        assert len(built) == 3

        await backend.close()

        assert _leaked_async_httpx_clients() == []

    @pytest.mark.asyncio
    async def test_clients_graphiti_mints_for_itself_are_released(
        self, mock_config, offline_graphiti, monkeypatch,
    ):
        mock_config.llm.providers.openai.api_key = ''
        mock_config.embedder.providers.openai.api_key = ''
        monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
        await _start_census()
        backend = GraphitiBackend(mock_config)
        await backend.initialize(skip_maintenance=True)
        await _write_one_episode(backend, 'g1')

        built = _leaked_async_httpx_clients()
        assert len(built) == 5

        await backend.close()

        assert _leaked_async_httpx_clients() == []

    @pytest.mark.asyncio
    async def test_injected_clients_are_owned_and_released(
        self, mock_config, offline_graphiti,
    ):
        await _start_census()
        llm = build_llm_client(mock_config)
        emb = build_embedder(mock_config)
        assert isinstance(llm, OpenAIClient)
        assert emb is not None
        backend = GraphitiBackend(mock_config)
        await backend.initialize(skip_maintenance=True, llm_client=llm, embedder=emb)
        await _write_one_episode(backend, 'g1')

        built = _leaked_async_httpx_clients()
        assert len(built) == 3

        await backend.close()

        assert _leaked_async_httpx_clients() == []
        assert llm.client.is_closed() is True
        assert emb.client.is_closed() is True

    @pytest.mark.asyncio
    async def test_close_before_initialize_builds_nothing(
        self, mock_config, async_httpx_clients_built,
    ):
        await GraphitiBackend(mock_config).close()

        assert async_httpx_clients_built == []


class TestTaskCuratorCloseReleasesEmbedder:
    @pytest.mark.asyncio
    async def test_the_lazily_built_embedder_is_released(self, mock_config):
        await _start_census()
        curator = TaskCurator(config=mock_config)
        await curator._get_embedder()

        built = _leaked_async_httpx_clients()
        assert len(built) == 1

        await curator.close()

        assert _leaked_async_httpx_clients() == []

    @pytest.mark.asyncio
    async def test_close_without_an_embedder_builds_nothing_and_a_second_close_is_safe(
        self, mock_config, async_httpx_clients_built,
    ):
        await TaskCurator(config=mock_config).close()
        assert async_httpx_clients_built == []

        curator = TaskCurator(config=mock_config)
        await curator._get_embedder()
        await curator.close()
        await curator.close()


_ROUTER_ANSWER = '{"query_type":"broad","stores":["graphiti","mem0"],"primary_store":"graphiti"}'
_CLASSIFIER_ANSWER = '{"primary":"observations_and_summaries","confidence":0.9,"reasoning":"x"}'

_ROUTING_OWNERS = [
    pytest.param(ReadRouter, lambda owner: owner.route('anything'), _ROUTER_ANSWER, id='router'),
    pytest.param(
        WriteClassifier, lambda owner: owner.classify('anything'), _CLASSIFIER_ANSWER,
        id='classifier',
    ),
]


class TestRoutingClientsClose:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(('owner_cls', 'ask', 'answer'), _ROUTING_OWNERS)
    async def test_the_pooled_client_is_released(
        self, mock_config, monkeypatch, owner_cls, ask, answer,
    ):
        mock_config.routing.use_heuristics = False
        mock_config.routing.llm_fallback = True
        owner = owner_cls(mock_config)
        with mock_openai_server() as server:
            monkeypatch.setenv('OPENAI_BASE_URL', server.base_url)
            server.chat_content = answer
            await _start_census()
            await ask(owner)

            assert len(server.requests_to('/chat/completions')) == 1
            built = _leaked_async_httpx_clients()
            assert len(built) == 1

            await owner.close()

        assert _leaked_async_httpx_clients() == []
        await owner.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('owner_cls', [ReadRouter, WriteClassifier])
    async def test_close_on_a_never_used_owner_builds_nothing(
        self, mock_config, async_httpx_clients_built, owner_cls,
    ):
        owner = owner_cls(mock_config)
        await owner.close()
        await owner.close()

        assert async_httpx_clients_built == []

    @pytest.mark.asyncio
    async def test_an_injected_classifier_client_is_owned_and_released(self, mock_config):
        client = completion_client('{}')

        await WriteClassifier(mock_config, openai_client=client).close()

        client.close.assert_awaited_once()


@pytest.fixture
def reindex_manager_cls(mock_config, make_fake_maintenance_service):
    """``run_reindex`` over a fake service and manager; its OpenAIEmbedder stays real."""
    service = AsyncMock()
    service.durable_queue = MagicMock()
    service.graphiti = MagicMock()
    with (
        patch(
            'fused_memory.maintenance.reindex.maintenance_service',
            side_effect=make_fake_maintenance_service(mock_config, service),
        ),
        patch('fused_memory.maintenance.reindex.ReindexManager') as manager_cls,
    ):
        yield manager_cls


class TestRunReindexClosesEmbedder:
    @pytest.mark.asyncio
    async def test_the_embedder_is_released_after_the_run(self, reindex_manager_cls):
        reindex_manager_cls.return_value.reindex_and_replay = AsyncMock(
            return_value={'reindex_result': ReindexResult(), 'replay_count': 0},
        )
        await _start_census()

        await run_reindex()

        embedder = reindex_manager_cls.call_args.kwargs['embedder']
        assert embedder.client.is_closed() is True
        assert _leaked_async_httpx_clients() == []

    @pytest.mark.asyncio
    async def test_the_embedder_is_released_when_the_run_fails(self, reindex_manager_cls):
        reindex_manager_cls.return_value.reindex_and_replay = AsyncMock(
            side_effect=RuntimeError('boom'),
        )
        await _start_census()

        with pytest.raises(RuntimeError, match='boom'):
            await run_reindex()

        embedder = reindex_manager_cls.call_args.kwargs['embedder']
        assert embedder.client.is_closed() is True
        assert _leaked_async_httpx_clients() == []
