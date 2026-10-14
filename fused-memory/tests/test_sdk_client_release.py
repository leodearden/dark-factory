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
from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import (
    _leaked_async_httpx_clients,
    reap_leaked_async_httpx_clients,
    track_async_httpx_clients,
)
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
def offline_falkor(monkeypatch):
    """A real Graphiti over a spec'd driver double: no FalkorDB, no preflight request."""
    def driver_factory(**_kwargs):
        driver = MagicMock(spec=FalkorDriver)
        driver.clone.return_value = MagicMock(spec=FalkorDriver)
        return driver

    monkeypatch.setattr(graphiti_client_module, '_MultiTenantFalkorDriver', driver_factory)
    monkeypatch.setattr(graphiti_client_module, 'check_openai_responses_api', lambda: None)


class TestGraphitiBackendCloseReleasesSdkClients:
    @pytest.mark.asyncio
    async def test_configured_sub_clients_are_shared_and_released(
        self, mock_config, offline_falkor,
    ):
        await _start_census()
        backend = GraphitiBackend(mock_config)
        await backend.initialize(skip_maintenance=True)
        backend._client_for('g1')

        built = _leaked_async_httpx_clients()
        assert len(built) == 3

        await backend.close()

        assert _leaked_async_httpx_clients() == []

    @pytest.mark.asyncio
    async def test_clients_graphiti_mints_for_itself_are_released(
        self, mock_config, offline_falkor, monkeypatch,
    ):
        mock_config.llm.providers.openai.api_key = ''
        mock_config.embedder.providers.openai.api_key = ''
        monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
        await _start_census()
        backend = GraphitiBackend(mock_config)
        await backend.initialize(skip_maintenance=True)
        backend._client_for('g1')

        built = _leaked_async_httpx_clients()
        assert len(built) == 5

        await backend.close()

        assert _leaked_async_httpx_clients() == []

    @pytest.mark.asyncio
    async def test_injected_clients_are_owned_and_released(
        self, mock_config, offline_falkor,
    ):
        await _start_census()
        llm = build_llm_client(mock_config)
        emb = build_embedder(mock_config)
        backend = GraphitiBackend(mock_config)
        await backend.initialize(skip_maintenance=True, llm_client=llm, embedder=emb)
        backend._client_for('g1')

        built = _leaked_async_httpx_clients()
        assert len(built) == 3

        await backend.close()

        assert _leaked_async_httpx_clients() == []
        assert llm.client.is_closed() is True
        assert emb.client.is_closed() is True

    @pytest.mark.asyncio
    async def test_close_before_initialize_is_a_no_op(self, mock_config):
        await GraphitiBackend(mock_config).close()
