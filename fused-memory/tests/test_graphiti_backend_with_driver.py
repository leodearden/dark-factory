"""GraphitiBackend.with_driver: a backend wired over a given driver and sub-clients, without initialize()."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
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
