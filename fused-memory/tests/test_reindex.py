"""Tests for reindex maintenance: GraphitiBackend stale-embedding queries and ReindexManager."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.backends.graphiti_client import GraphitiBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backend(config) -> GraphitiBackend:
    """Build a GraphitiBackend with a mock client attached."""
    backend = GraphitiBackend(config)
    # inject a mock Graphiti client
    mock_client = MagicMock()
    backend.client = mock_client
    return backend


def _make_graph_mock(rows: list[list]) -> MagicMock:
    """Return a mock whose .query() is an AsyncMock returning rows."""
    result = MagicMock()
    result.result_set = rows
    graph_mock = MagicMock()
    graph_mock.query = AsyncMock(return_value=result)
    return graph_mock


# ---------------------------------------------------------------------------
# step-1: query_stale_node_embeddings / query_stale_edge_embeddings
# ---------------------------------------------------------------------------

class TestQueryStaleNodeEmbeddings:
    """GraphitiBackend.query_stale_node_embeddings(expected_dim) returns stale nodes."""

    @pytest.mark.asyncio
    async def test_returns_stale_nodes(self, mock_config):
        backend = _make_backend(mock_config)
        rows = [
            ['uuid-1', 'Node A', 1024],
            ['uuid-2', 'Node B', 512],
        ]
        graph = _make_graph_mock(rows)
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            result = await backend.query_stale_node_embeddings(expected_dim=1536)
        assert len(result) == 2
        assert result[0] == ('uuid-1', 'Node A', 1024)
        assert result[1] == ('uuid-2', 'Node B', 512)

    @pytest.mark.asyncio
    async def test_returns_empty_when_all_match(self, mock_config):
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            result = await backend.query_stale_node_embeddings(expected_dim=1536)
        assert result == []

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        backend = GraphitiBackend(mock_config)  # client is None
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.query_stale_node_embeddings(expected_dim=1536)

    @pytest.mark.asyncio
    async def test_passes_dim_param_to_query(self, mock_config):
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            await backend.query_stale_node_embeddings(expected_dim=768)
        call_kwargs = graph.query.call_args
        assert call_kwargs is not None
        # check dim is passed as a parameter
        args, kwargs = call_kwargs
        cypher_args = args[1] if len(args) > 1 else kwargs.get('params', {})
        assert cypher_args.get('dim') == 768


class TestQueryStaleEdgeEmbeddings:
    """GraphitiBackend.query_stale_edge_embeddings(expected_dim) returns stale edges."""

    @pytest.mark.asyncio
    async def test_returns_stale_edges(self, mock_config):
        backend = _make_backend(mock_config)
        rows = [
            ['edge-uuid-1', 'edge name', 1024],
        ]
        graph = _make_graph_mock(rows)
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            result = await backend.query_stale_edge_embeddings(expected_dim=1536)
        assert len(result) == 1
        assert result[0] == ('edge-uuid-1', 'edge name', 1024)

    @pytest.mark.asyncio
    async def test_returns_empty_when_all_match(self, mock_config):
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            result = await backend.query_stale_edge_embeddings(expected_dim=1536)
        assert result == []

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        backend = GraphitiBackend(mock_config)
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.query_stale_edge_embeddings(expected_dim=1536)

    @pytest.mark.asyncio
    async def test_passes_dim_param_to_query(self, mock_config):
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            await backend.query_stale_edge_embeddings(expected_dim=768)
        call_kwargs = graph.query.call_args
        assert call_kwargs is not None
        args, kwargs = call_kwargs
        cypher_args = args[1] if len(args) > 1 else kwargs.get('params', {})
        assert cypher_args.get('dim') == 768
