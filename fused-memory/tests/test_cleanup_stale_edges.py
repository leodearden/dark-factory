"""Tests for cleanup_stale_edges maintenance: GraphitiBackend time-range queries and CleanupManager."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.backends.graphiti_client import GraphitiBackend


# ---------------------------------------------------------------------------
# Helpers (mirrored from test_reindex.py)
# ---------------------------------------------------------------------------

def _make_backend(config) -> GraphitiBackend:
    """Build a GraphitiBackend with a mock client attached."""
    backend = GraphitiBackend(config)
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
# step-1: GraphitiBackend.query_edges_by_time_range
# ---------------------------------------------------------------------------

class TestQueryEdgesByTimeRange:
    """GraphitiBackend.query_edges_by_time_range(start, end) returns matching edges."""

    @pytest.mark.asyncio
    async def test_returns_matching_edges(self, mock_config):
        backend = _make_backend(mock_config)
        rows = [
            ['uuid-1', 'Alice is related to Bob', 'is_related', '2026-03-22T17:51:00', '2026-03-22T18:00:00'],
            ['uuid-2', 'Bob knows Carol', 'knows', '2026-03-22T18:00:00', None],
        ]
        graph = _make_graph_mock(rows)
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            result = await backend.query_edges_by_time_range(
                start='2026-03-22T17:50:00',
                end='2026-03-22T18:15:00',
            )
        assert len(result) == 2
        assert result[0]['uuid'] == 'uuid-1'
        assert result[0]['fact'] == 'Alice is related to Bob'
        assert result[0]['name'] == 'is_related'
        assert result[0]['valid_at'] == '2026-03-22T17:51:00'
        assert result[0]['invalid_at'] == '2026-03-22T18:00:00'
        assert result[1]['uuid'] == 'uuid-2'

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_matches(self, mock_config):
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            result = await backend.query_edges_by_time_range(
                start='2026-03-22T17:50:00',
                end='2026-03-22T18:15:00',
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        backend = GraphitiBackend(mock_config)  # client is None
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.query_edges_by_time_range(
                start='2026-03-22T17:50:00',
                end='2026-03-22T18:15:00',
            )

    @pytest.mark.asyncio
    async def test_passes_time_params_to_query(self, mock_config):
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        start = '2026-03-22T17:50:00'
        end = '2026-03-22T18:15:00'
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            await backend.query_edges_by_time_range(start=start, end=end)
        call_args = graph.query.call_args
        assert call_args is not None
        args, kwargs = call_args
        cypher_params = args[1] if len(args) > 1 else kwargs.get('params', {})
        assert cypher_params.get('start') == start
        assert cypher_params.get('end') == end


# ---------------------------------------------------------------------------
# step-3: GraphitiBackend.bulk_remove_edges
# ---------------------------------------------------------------------------

class TestBulkRemoveEdges:
    """GraphitiBackend.bulk_remove_edges(uuids) deletes edges and returns count."""

    @pytest.mark.asyncio
    async def test_deletes_matching_edges(self, mock_config):
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        uuids = ['uuid-1', 'uuid-2', 'uuid-3']
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            count = await backend.bulk_remove_edges(uuids)
        assert count == 3
        graph.query.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handles_empty_uuid_list(self, mock_config):
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            count = await backend.bulk_remove_edges([])
        assert count == 0
        # Should not query at all for empty list
        graph.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        backend = GraphitiBackend(mock_config)  # client is None
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.bulk_remove_edges(['uuid-1'])

    @pytest.mark.asyncio
    async def test_passes_uuid_list_to_query(self, mock_config):
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        uuids = ['uuid-a', 'uuid-b']
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            await backend.bulk_remove_edges(uuids)
        call_args = graph.query.call_args
        assert call_args is not None
        args, kwargs = call_args
        cypher_params = args[1] if len(args) > 1 else kwargs.get('params', {})
        assert cypher_params.get('uuids') == uuids
