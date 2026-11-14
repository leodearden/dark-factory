"""GraphitiBackend.with_driver: a backend wired over a given driver and sub-clients, without initialize()."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from _graph_access_fake import FakeGraphAccess, HangingRawGraph, with_backend_budgets
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder import EmbedderClient
from graphiti_core.llm_client import LLMClient

from fused_memory.backends.graphiti_client import GraphitiBackend


class TestWithDriver:
    @pytest.mark.asyncio
    async def test_graph_reads_run_over_the_given_driver(
        self, mock_config, make_graph_driver, make_graph_mock,
    ):
        driver = make_graph_driver({'g': make_graph_mock([[7]])})
        backend = GraphitiBackend.with_driver(mock_config, driver, registered_graph_ids=())

        assert await backend.node_count('g') == 7

    def test_per_group_clients_carry_the_given_sub_clients(self, mock_config, make_graph_driver):
        llm = MagicMock(spec=LLMClient)
        emb = MagicMock(spec=EmbedderClient)
        ce = MagicMock(spec=CrossEncoderClient)
        backend = GraphitiBackend.with_driver(
            mock_config,
            make_graph_driver({}),
            registered_graph_ids=(),
            llm_client=llm,
            embedder=emb,
            cross_encoder=ce,
        )

        client = backend._client_for('g')

        assert client.llm_client is llm
        assert client.embedder is emb
        assert client.cross_encoder is ce

    def test_construction_touches_no_graph(self, mock_config, make_graph_driver):
        driver = make_graph_driver({})

        GraphitiBackend.with_driver(mock_config, driver, registered_graph_ids=())

        assert driver.client.mock_calls == []

    @pytest.mark.asyncio
    async def test_close_releases_the_given_driver(self, mock_config, make_graph_driver):
        driver = make_graph_driver({})
        driver.client.aclose = AsyncMock()
        backend = GraphitiBackend.with_driver(mock_config, driver, registered_graph_ids=())

        await backend.close()

        driver.client.aclose.assert_awaited()

    def test_registry_is_the_given_one_canonicalized(self, mock_config, make_graph_driver):
        backend = GraphitiBackend.with_driver(
            mock_config, make_graph_driver({}), registered_graph_ids={'A-b'},
        )

        assert backend.registered_graph_ids == frozenset({'a_b'})


class TestGraphAccessSeam:
    def wired_backend(self, config, driver, graph_access=None) -> GraphitiBackend:
        return GraphitiBackend.with_driver(
            config, driver, registered_graph_ids=(), graph_access=graph_access,
        )

    def test_the_injected_provider_is_the_backends_provider(
        self, mock_config, make_graph_driver,
    ):
        fake = FakeGraphAccess({})

        backend = self.wired_backend(mock_config, make_graph_driver({}), fake)

        assert backend.graph_access is fake

    @pytest.mark.asyncio
    async def test_a_raw_read_answers_from_the_injected_providers_graph(
        self, mock_config, make_graph_driver, make_graph_mock,
    ):
        fake = FakeGraphAccess({'g': make_graph_mock([[7]])})

        backend = self.wired_backend(mock_config, make_graph_driver({}), fake)

        assert await backend.node_count('g') == 7

    @pytest.mark.asyncio
    async def test_a_group_the_provider_does_not_serve_raises_naming_it(
        self, mock_config, make_graph_driver, make_graph_mock,
    ):
        fake = FakeGraphAccess({'g': make_graph_mock([[7]])})
        backend = self.wired_backend(mock_config, make_graph_driver({}), fake)

        with pytest.raises(KeyError, match="'other'"):
            await backend.node_count('other')

    def test_the_provider_is_read_only(self, mock_config, make_graph_driver):
        backend = self.wired_backend(mock_config, make_graph_driver({}), FakeGraphAccess({}))
        attribute = 'graph_access'

        with pytest.raises(AttributeError):
            setattr(backend, attribute, FakeGraphAccess({}))

    @pytest.mark.asyncio
    async def test_a_closed_backend_no_longer_serves_the_injected_provider(
        self, mock_config, make_graph_driver,
    ):
        backend = self.wired_backend(mock_config, make_graph_driver({}), FakeGraphAccess({}))

        await backend.close()

        with pytest.raises(RuntimeError, match='not initialized'):
            _ = backend.graph_access

    @pytest.mark.asyncio
    async def test_the_default_provider_carries_the_configured_read_budget(
        self, mock_config, make_graph_driver,
    ):
        config = with_backend_budgets(mock_config, read_s=0.05, write_s=0.08)
        backend = self.wired_backend(config, make_graph_driver({'g': HangingRawGraph()}))

        with pytest.raises(TimeoutError) as raised:
            await asyncio.wait_for(backend.node_count('g'), timeout=5.0)

        assert str(raised.value) == 'Graphiti node_count timed out after 0.05s'

    def test_an_uninitialized_backend_has_no_provider(self, mock_config):
        with pytest.raises(RuntimeError, match='not initialized'):
            _ = GraphitiBackend(mock_config).graph_access

