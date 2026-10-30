"""GraphitiBackend.with_driver: a backend wired over a given driver and sub-clients, without initialize()."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder import EmbedderClient
from graphiti_core.llm_client import LLMClient

from fused_memory.backends.graphiti_client import GraphitiBackend, _MultiTenantFalkorDriver


def _driver_over(falkor: MagicMock) -> _MultiTenantFalkorDriver:
    return _MultiTenantFalkorDriver(falkor_db=falkor, database='__base__')


class TestWithDriver:
    @pytest.mark.asyncio
    async def test_graph_reads_run_over_the_given_driver(self, mock_config, make_graph_mock):
        falkor = MagicMock()
        falkor.select_graph.return_value = make_graph_mock([[7]])
        backend = GraphitiBackend.with_driver(
            mock_config, _driver_over(falkor), registered_graph_ids=(),
        )

        assert await backend.node_count('g') == 7
        falkor.select_graph.assert_called_once_with('g')

    def test_per_group_clients_carry_the_given_sub_clients(self, mock_config):
        llm = MagicMock(spec=LLMClient)
        emb = MagicMock(spec=EmbedderClient)
        ce = MagicMock(spec=CrossEncoderClient)
        backend = GraphitiBackend.with_driver(
            mock_config,
            _driver_over(MagicMock()),
            registered_graph_ids=(),
            llm_client=llm,
            embedder=emb,
            cross_encoder=ce,
        )

        client = backend._client_for('g')

        assert client.llm_client is llm
        assert client.embedder is emb
        assert client.cross_encoder is ce
        assert client.driver._database == 'g'

    def test_construction_touches_no_graph(self, mock_config):
        falkor = MagicMock()

        GraphitiBackend.with_driver(mock_config, _driver_over(falkor), registered_graph_ids=())

        assert falkor.mock_calls == []

    @pytest.mark.asyncio
    async def test_close_releases_the_given_driver(self, mock_config):
        falkor = MagicMock()
        falkor.aclose = AsyncMock()
        backend = GraphitiBackend.with_driver(
            mock_config, _driver_over(falkor), registered_graph_ids=(),
        )

        await backend.close()

        falkor.aclose.assert_awaited()

    def test_registry_is_the_given_one_canonicalized(self, mock_config):
        backend = GraphitiBackend.with_driver(
            mock_config, _driver_over(MagicMock()), registered_graph_ids={'A-b'},
        )

        assert backend.registered_graph_ids == frozenset({'a_b'})
