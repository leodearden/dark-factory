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
        # Two calls: pre-count returns [[3]], DELETE returns empty
        pre_count_result = MagicMock()
        pre_count_result.result_set = [[3]]
        delete_result = MagicMock()
        delete_result.result_set = []
        graph_mock = MagicMock()
        graph_mock.query = AsyncMock(side_effect=[pre_count_result, delete_result])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph_mock)
        uuids = ['uuid-1', 'uuid-2', 'uuid-3']
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            count = await backend.bulk_remove_edges(uuids)
        assert count == 3
        assert graph_mock.query.await_count == 2

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
        # Two calls: pre-count then DELETE
        pre_count_result = MagicMock()
        pre_count_result.result_set = [[2]]
        delete_result = MagicMock()
        delete_result.result_set = []
        graph_mock = MagicMock()
        graph_mock.query = AsyncMock(side_effect=[pre_count_result, delete_result])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph_mock)
        uuids = ['uuid-a', 'uuid-b']
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            await backend.bulk_remove_edges(uuids)
        # Verify the DELETE call (second call) passes uuids as params
        all_calls = graph_mock.query.call_args_list
        assert len(all_calls) == 2
        delete_call_args = all_calls[1]
        args, kwargs = delete_call_args
        cypher_params = args[1] if len(args) > 1 else kwargs.get('params', {})
        assert cypher_params.get('uuids') == uuids

    @pytest.mark.asyncio
    async def test_returns_actual_deletion_count_not_input_length(self, mock_config):
        """bulk_remove_edges returns the actual count of matched edges, not len(uuids).

        Pass 3 UUIDs but simulate only 2 existing in FalkorDB via pre-count query.
        The return value must be 2, not 3.
        """
        backend = _make_backend(mock_config)
        cast_target = MagicMock()

        # Pre-count result: 2 of 3 UUIDs exist as edges
        pre_count_result = MagicMock()
        pre_count_result.result_set = [[2]]

        # DELETE result: empty (no RETURN clause)
        delete_result = MagicMock()
        delete_result.result_set = []

        graph_mock = MagicMock()
        graph_mock.query = AsyncMock(side_effect=[pre_count_result, delete_result])
        cast_target._get_graph = MagicMock(return_value=graph_mock)

        uuids = ['uuid-1', 'uuid-2', 'uuid-3']
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            count = await backend.bulk_remove_edges(uuids)

        # Must return 2 (actual matched count), not 3 (input length)
        assert count == 2
        assert graph_mock.query.await_count == 2


# ---------------------------------------------------------------------------
# step-5: CleanupManager and CleanupResult
# ---------------------------------------------------------------------------

class TestCleanupManager:
    """CleanupManager orchestrates find_stale_edges + bulk_remove_edges."""

    @pytest.mark.asyncio
    async def test_find_stale_edges_delegates_to_backend(self, mock_config):
        from fused_memory.maintenance.cleanup_stale_edges import CleanupManager

        backend = _make_backend(mock_config)
        backend.query_edges_by_time_range = AsyncMock(return_value=[
            {'uuid': 'u1', 'fact': 'f1', 'name': 'n1', 'valid_at': '2026-03-22T17:51:00', 'invalid_at': None},
        ])
        manager = CleanupManager(backend)
        result = await manager.find_stale_edges(
            start='2026-03-22T17:50:00', end='2026-03-22T18:15:00'
        )
        backend.query_edges_by_time_range.assert_awaited_once_with(
            start='2026-03-22T17:50:00', end='2026-03-22T18:15:00'
        )
        assert len(result) == 1
        assert result[0]['uuid'] == 'u1'

    @pytest.mark.asyncio
    async def test_cleanup_queries_and_deletes(self, mock_config):
        from fused_memory.maintenance.cleanup_stale_edges import CleanupManager, CleanupResult

        edge_details = [
            {'uuid': 'u1', 'fact': 'f1', 'name': 'n1', 'valid_at': '2026-03-22T17:51:00', 'invalid_at': None},
            {'uuid': 'u2', 'fact': 'f2', 'name': 'n2', 'valid_at': '2026-03-22T18:00:00', 'invalid_at': None},
        ]
        backend = _make_backend(mock_config)
        backend.query_edges_by_time_range = AsyncMock(return_value=edge_details)
        backend.bulk_remove_edges = AsyncMock(return_value=2)
        manager = CleanupManager(backend)
        result = await manager.cleanup(
            start='2026-03-22T17:50:00', end='2026-03-22T18:15:00'
        )
        assert isinstance(result, CleanupResult)
        assert result.edges_found == 2
        assert result.edges_deleted == 2
        backend.bulk_remove_edges.assert_awaited_once_with(['u1', 'u2'])

    @pytest.mark.asyncio
    async def test_dry_run_queries_but_does_not_delete(self, mock_config):
        from fused_memory.maintenance.cleanup_stale_edges import CleanupManager, CleanupResult

        edge_details = [
            {'uuid': 'u1', 'fact': 'f1', 'name': 'n1', 'valid_at': '2026-03-22T17:51:00', 'invalid_at': None},
        ]
        backend = _make_backend(mock_config)
        backend.query_edges_by_time_range = AsyncMock(return_value=edge_details)
        backend.bulk_remove_edges = AsyncMock(return_value=0)
        manager = CleanupManager(backend)
        result = await manager.cleanup(
            start='2026-03-22T17:50:00', end='2026-03-22T18:15:00', dry_run=True
        )
        assert isinstance(result, CleanupResult)
        assert result.edges_found == 1
        assert result.edges_deleted == 0
        backend.bulk_remove_edges.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_empty_result(self, mock_config):
        from fused_memory.maintenance.cleanup_stale_edges import CleanupManager, CleanupResult

        backend = _make_backend(mock_config)
        backend.query_edges_by_time_range = AsyncMock(return_value=[])
        backend.bulk_remove_edges = AsyncMock(return_value=0)
        manager = CleanupManager(backend)
        result = await manager.cleanup(
            start='2026-03-22T17:50:00', end='2026-03-22T18:15:00'
        )
        assert isinstance(result, CleanupResult)
        assert result.edges_found == 0
        assert result.edges_deleted == 0
        backend.bulk_remove_edges.assert_not_called()


# ---------------------------------------------------------------------------
# step-7: run_cleanup async entrypoint
# ---------------------------------------------------------------------------

class TestRunCleanup:
    """run_cleanup() loads config, initializes MemoryService, runs cleanup, closes."""

    @pytest.mark.asyncio
    async def test_loads_config_and_runs_cleanup(self):
        """run_cleanup() initializes the MemoryService and calls cleanup with correct args."""
        from fused_memory.maintenance.cleanup_stale_edges import CleanupResult, run_cleanup

        mock_cfg = MagicMock()
        mock_service = AsyncMock()
        mock_service.graphiti = MagicMock()

        mock_result = CleanupResult(edges_found=2, edges_deleted=2)

        with (
            patch('fused_memory.maintenance.cleanup_stale_edges.FusedMemoryConfig', return_value=mock_cfg),
            patch('fused_memory.maintenance.cleanup_stale_edges.MemoryService', return_value=mock_service),
            patch('fused_memory.maintenance.cleanup_stale_edges.CleanupManager') as mock_mgr_cls,
        ):
            mock_mgr = MagicMock()
            mock_mgr.cleanup = AsyncMock(return_value=mock_result)
            mock_mgr_cls.return_value = mock_mgr

            result = await run_cleanup(
                start='2026-03-22T17:50:00',
                end='2026-03-22T18:15:00',
            )

        mock_service.initialize.assert_awaited_once()
        mock_mgr.cleanup.assert_awaited_once_with(
            start='2026-03-22T17:50:00',
            end='2026-03-22T18:15:00',
            dry_run=False,
        )
        assert result == mock_result

    @pytest.mark.asyncio
    async def test_closes_service_on_success(self):
        """run_cleanup() calls service.close() in finally block after success."""
        from fused_memory.maintenance.cleanup_stale_edges import CleanupResult, run_cleanup

        mock_cfg = MagicMock()
        mock_service = AsyncMock()
        mock_service.graphiti = MagicMock()

        mock_result = CleanupResult()

        with (
            patch('fused_memory.maintenance.cleanup_stale_edges.FusedMemoryConfig', return_value=mock_cfg),
            patch('fused_memory.maintenance.cleanup_stale_edges.MemoryService', return_value=mock_service),
            patch('fused_memory.maintenance.cleanup_stale_edges.CleanupManager') as mock_mgr_cls,
        ):
            mock_mgr = MagicMock()
            mock_mgr.cleanup = AsyncMock(return_value=mock_result)
            mock_mgr_cls.return_value = mock_mgr

            await run_cleanup()

        mock_service.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_closes_service_on_error(self):
        """run_cleanup() calls service.close() in finally block even when cleanup raises."""
        from fused_memory.maintenance.cleanup_stale_edges import run_cleanup

        mock_cfg = MagicMock()
        mock_service = AsyncMock()
        mock_service.graphiti = MagicMock()

        with (
            patch('fused_memory.maintenance.cleanup_stale_edges.FusedMemoryConfig', return_value=mock_cfg),
            patch('fused_memory.maintenance.cleanup_stale_edges.MemoryService', return_value=mock_service),
            patch('fused_memory.maintenance.cleanup_stale_edges.CleanupManager') as mock_mgr_cls,
        ):
            mock_mgr = MagicMock()
            mock_mgr.cleanup = AsyncMock(side_effect=RuntimeError('falkordb unavailable'))
            mock_mgr_cls.return_value = mock_mgr

            with pytest.raises(RuntimeError, match='falkordb unavailable'):
                await run_cleanup()

        mock_service.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_passes_dry_run_flag(self):
        """run_cleanup(dry_run=True) passes dry_run=True to cleanup."""
        from fused_memory.maintenance.cleanup_stale_edges import CleanupResult, run_cleanup

        mock_cfg = MagicMock()
        mock_service = AsyncMock()
        mock_service.graphiti = MagicMock()

        mock_result = CleanupResult(edges_found=3, edges_deleted=0)

        with (
            patch('fused_memory.maintenance.cleanup_stale_edges.FusedMemoryConfig', return_value=mock_cfg),
            patch('fused_memory.maintenance.cleanup_stale_edges.MemoryService', return_value=mock_service),
            patch('fused_memory.maintenance.cleanup_stale_edges.CleanupManager') as mock_mgr_cls,
        ):
            mock_mgr = MagicMock()
            mock_mgr.cleanup = AsyncMock(return_value=mock_result)
            mock_mgr_cls.return_value = mock_mgr

            await run_cleanup(dry_run=True)

        call_kwargs = mock_mgr.cleanup.call_args
        assert call_kwargs is not None
        _, kwargs = call_kwargs
        assert kwargs.get('dry_run') is True
