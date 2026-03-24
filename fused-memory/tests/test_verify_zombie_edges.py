"""Tests for verify_zombie_edges maintenance: GraphitiBackend.check_edges_by_uuid,
ZombieEdgeVerifier, and run_verify_zombie_edges entrypoint."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.backends.graphiti_client import GraphitiBackend

# ---------------------------------------------------------------------------
# Helpers (mirrored from test_cleanup_stale_edges.py)
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
# step-1: GraphitiBackend.check_edges_by_uuid
# ---------------------------------------------------------------------------


class TestCheckEdgesByUuid:
    """GraphitiBackend.check_edges_by_uuid(uuids) returns found UUIDs (read-only)."""

    @pytest.mark.asyncio
    async def test_returns_found_uuids(self, mock_config):
        """Returns list of UUIDs that exist as edges in FalkorDB."""
        backend = _make_backend(mock_config)
        # FalkorDB returns two rows: each row is [uuid_string]
        rows = [['abc12345'], ['def67890']]
        graph = _make_graph_mock(rows)
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            result = await backend.check_edges_by_uuid(['abc12345', 'def67890', 'zzz00000'])
        assert result == ['abc12345', 'def67890']

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_matches(self, mock_config):
        """Returns empty list when no UUIDs match any edge."""
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            result = await backend.check_edges_by_uuid(['notexist1', 'notexist2'])
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_for_empty_input(self, mock_config):
        """Returns empty list immediately without querying FalkorDB for empty input."""
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            result = await backend.check_edges_by_uuid([])
        assert result == []
        graph.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        """Raises RuntimeError if the backend client has not been initialized."""
        backend = GraphitiBackend(mock_config)  # client is None
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.check_edges_by_uuid(['abc12345'])

    @pytest.mark.asyncio
    async def test_passes_uuid_list_to_query(self, mock_config):
        """Passes the uuids list as $uuids parameter to the Cypher query."""
        backend = _make_backend(mock_config)
        uuids = ['aaa11111', 'bbb22222']
        rows = [['aaa11111']]
        graph = _make_graph_mock(rows)
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            await backend.check_edges_by_uuid(uuids)
        call_args = graph.query.call_args
        assert call_args is not None
        args, kwargs = call_args
        cypher_params = args[1] if len(args) > 1 else kwargs.get('params', {})
        assert cypher_params.get('uuids') == uuids


# ---------------------------------------------------------------------------
# step-3: ZombieEdgeVerifier.verify
# ---------------------------------------------------------------------------


class TestZombieEdgeVerifierVerify:
    """ZombieEdgeVerifier.verify(uuids) returns VerifyResult with found/missing lists."""

    @pytest.mark.asyncio
    async def test_returns_verify_result_with_found_and_missing(self, mock_config):
        """verify() partitions input UUIDs into found (exist) and missing (absent)."""
        from fused_memory.maintenance.verify_zombie_edges import VerifyResult, ZombieEdgeVerifier

        backend = _make_backend(mock_config)
        backend.check_edges_by_uuid = AsyncMock(return_value=['uuid-a', 'uuid-b'])
        verifier = ZombieEdgeVerifier(backend=backend)

        result = await verifier.verify(['uuid-a', 'uuid-b', 'uuid-c'])

        assert isinstance(result, VerifyResult)
        assert result.found == ['uuid-a', 'uuid-b']
        assert result.missing == ['uuid-c']
        assert result.deleted == 0

    @pytest.mark.asyncio
    async def test_handles_all_found(self, mock_config):
        """verify() handles case where all UUIDs exist as edges."""
        from fused_memory.maintenance.verify_zombie_edges import ZombieEdgeVerifier

        backend = _make_backend(mock_config)
        backend.check_edges_by_uuid = AsyncMock(return_value=['x1', 'x2', 'x3'])
        verifier = ZombieEdgeVerifier(backend=backend)

        result = await verifier.verify(['x1', 'x2', 'x3'])

        assert result.found == ['x1', 'x2', 'x3']
        assert result.missing == []

    @pytest.mark.asyncio
    async def test_handles_all_missing(self, mock_config):
        """verify() handles case where no UUIDs exist as edges."""
        from fused_memory.maintenance.verify_zombie_edges import ZombieEdgeVerifier

        backend = _make_backend(mock_config)
        backend.check_edges_by_uuid = AsyncMock(return_value=[])
        verifier = ZombieEdgeVerifier(backend=backend)

        result = await verifier.verify(['y1', 'y2'])

        assert result.found == []
        assert result.missing == ['y1', 'y2']

    @pytest.mark.asyncio
    async def test_handles_empty_input(self, mock_config):
        """verify() handles empty UUID list gracefully."""
        from fused_memory.maintenance.verify_zombie_edges import VerifyResult, ZombieEdgeVerifier

        backend = _make_backend(mock_config)
        backend.check_edges_by_uuid = AsyncMock(return_value=[])
        verifier = ZombieEdgeVerifier(backend=backend)

        result = await verifier.verify([])

        assert isinstance(result, VerifyResult)
        assert result.found == []
        assert result.missing == []
        assert result.deleted == 0


# ---------------------------------------------------------------------------
# step-5: ZombieEdgeVerifier.cleanup
# ---------------------------------------------------------------------------


class TestZombieEdgeVerifierCleanup:
    """ZombieEdgeVerifier.cleanup(uuids, dry_run) verifies then conditionally deletes."""

    @pytest.mark.asyncio
    async def test_dry_run_verifies_only_no_deletion(self, mock_config):
        """dry_run=True calls verify but does NOT call bulk_remove_edges."""
        from fused_memory.maintenance.verify_zombie_edges import VerifyResult, ZombieEdgeVerifier

        backend = _make_backend(mock_config)
        backend.check_edges_by_uuid = AsyncMock(return_value=['a1', 'a2'])
        backend.bulk_remove_edges = AsyncMock(return_value=0)
        verifier = ZombieEdgeVerifier(backend=backend)

        result = await verifier.cleanup(['a1', 'a2', 'a3'], dry_run=True)

        assert isinstance(result, VerifyResult)
        assert result.found == ['a1', 'a2']
        assert result.missing == ['a3']
        assert result.deleted == 0
        backend.bulk_remove_edges.assert_not_called()

    @pytest.mark.asyncio
    async def test_live_run_deletes_found_edges(self, mock_config):
        """dry_run=False calls bulk_remove_edges with found UUIDs."""
        from fused_memory.maintenance.verify_zombie_edges import VerifyResult, ZombieEdgeVerifier

        backend = _make_backend(mock_config)
        backend.check_edges_by_uuid = AsyncMock(return_value=['b1', 'b2'])
        backend.bulk_remove_edges = AsyncMock(return_value=2)
        verifier = ZombieEdgeVerifier(backend=backend)

        result = await verifier.cleanup(['b1', 'b2', 'b3'], dry_run=False)

        assert isinstance(result, VerifyResult)
        assert result.found == ['b1', 'b2']
        assert result.missing == ['b3']
        assert result.deleted == 2
        backend.bulk_remove_edges.assert_awaited_once_with(['b1', 'b2'])

    @pytest.mark.asyncio
    async def test_skips_delete_when_no_edges_exist(self, mock_config):
        """cleanup() skips bulk_remove_edges when verify finds no existing edges."""
        from fused_memory.maintenance.verify_zombie_edges import ZombieEdgeVerifier

        backend = _make_backend(mock_config)
        backend.check_edges_by_uuid = AsyncMock(return_value=[])
        backend.bulk_remove_edges = AsyncMock(return_value=0)
        verifier = ZombieEdgeVerifier(backend=backend)

        result = await verifier.cleanup(['c1', 'c2'], dry_run=False)

        assert result.found == []
        assert result.missing == ['c1', 'c2']
        assert result.deleted == 0
        backend.bulk_remove_edges.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_accurate_deleted_count(self, mock_config):
        """deleted field reflects actual return value from bulk_remove_edges."""
        from fused_memory.maintenance.verify_zombie_edges import ZombieEdgeVerifier

        backend = _make_backend(mock_config)
        # 3 UUIDs exist, but bulk_remove_edges returns 2 (one may have been concurrently removed)
        backend.check_edges_by_uuid = AsyncMock(return_value=['d1', 'd2', 'd3'])
        backend.bulk_remove_edges = AsyncMock(return_value=2)
        verifier = ZombieEdgeVerifier(backend=backend)

        result = await verifier.cleanup(['d1', 'd2', 'd3'], dry_run=False)

        assert result.deleted == 2


# ---------------------------------------------------------------------------
# step-7: run_verify_zombie_edges async entrypoint
# ---------------------------------------------------------------------------


class TestRunVerifyZombieEdges:
    """run_verify_zombie_edges() loads config, initializes service, runs cleanup, closes."""

    @pytest.mark.asyncio
    async def test_loads_config_and_runs_cleanup(self):
        """run_verify_zombie_edges() initializes MemoryService and calls cleanup."""
        from fused_memory.maintenance.verify_zombie_edges import (
            VerifyResult,
            run_verify_zombie_edges,
        )

        mock_cfg = MagicMock()
        mock_service = AsyncMock()
        mock_service.graphiti = MagicMock()

        mock_result = VerifyResult(found=['uuid-1'], missing=[], deleted=1)

        with (
            patch('fused_memory.maintenance.verify_zombie_edges.FusedMemoryConfig', return_value=mock_cfg),
            patch('fused_memory.maintenance.verify_zombie_edges.MemoryService', return_value=mock_service),
            patch('fused_memory.maintenance.verify_zombie_edges.ZombieEdgeVerifier') as mock_verifier_cls,
        ):
            mock_verifier = MagicMock()
            mock_verifier.cleanup = AsyncMock(return_value=mock_result)
            mock_verifier_cls.return_value = mock_verifier

            result = await run_verify_zombie_edges(uuids=['uuid-1', 'uuid-2'])

        mock_service.initialize.assert_awaited_once()
        mock_verifier.cleanup.assert_awaited_once_with(
            uuids=['uuid-1', 'uuid-2'],
            dry_run=False,
        )
        assert result == mock_result

    @pytest.mark.asyncio
    async def test_closes_service_on_success(self):
        """run_verify_zombie_edges() calls service.close() in finally block after success."""
        from fused_memory.maintenance.verify_zombie_edges import (
            VerifyResult,
            run_verify_zombie_edges,
        )

        mock_cfg = MagicMock()
        mock_service = AsyncMock()
        mock_service.graphiti = MagicMock()
        mock_result = VerifyResult()

        with (
            patch('fused_memory.maintenance.verify_zombie_edges.FusedMemoryConfig', return_value=mock_cfg),
            patch('fused_memory.maintenance.verify_zombie_edges.MemoryService', return_value=mock_service),
            patch('fused_memory.maintenance.verify_zombie_edges.ZombieEdgeVerifier') as mock_verifier_cls,
        ):
            mock_verifier = MagicMock()
            mock_verifier.cleanup = AsyncMock(return_value=mock_result)
            mock_verifier_cls.return_value = mock_verifier

            await run_verify_zombie_edges()

        mock_service.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_closes_service_on_error(self):
        """run_verify_zombie_edges() calls service.close() even when cleanup raises."""
        from fused_memory.maintenance.verify_zombie_edges import run_verify_zombie_edges

        mock_cfg = MagicMock()
        mock_service = AsyncMock()
        mock_service.graphiti = MagicMock()

        with (
            patch('fused_memory.maintenance.verify_zombie_edges.FusedMemoryConfig', return_value=mock_cfg),
            patch('fused_memory.maintenance.verify_zombie_edges.MemoryService', return_value=mock_service),
            patch('fused_memory.maintenance.verify_zombie_edges.ZombieEdgeVerifier') as mock_verifier_cls,
        ):
            mock_verifier = MagicMock()
            mock_verifier.cleanup = AsyncMock(side_effect=RuntimeError('falkordb unavailable'))
            mock_verifier_cls.return_value = mock_verifier

            with pytest.raises(RuntimeError, match='falkordb unavailable'):
                await run_verify_zombie_edges()

        mock_service.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_passes_dry_run_flag(self):
        """run_verify_zombie_edges(dry_run=True) passes dry_run=True to cleanup."""
        from fused_memory.maintenance.verify_zombie_edges import (
            VerifyResult,
            run_verify_zombie_edges,
        )

        mock_cfg = MagicMock()
        mock_service = AsyncMock()
        mock_service.graphiti = MagicMock()
        mock_result = VerifyResult(found=['e1'], missing=[], deleted=0)

        with (
            patch('fused_memory.maintenance.verify_zombie_edges.FusedMemoryConfig', return_value=mock_cfg),
            patch('fused_memory.maintenance.verify_zombie_edges.MemoryService', return_value=mock_service),
            patch('fused_memory.maintenance.verify_zombie_edges.ZombieEdgeVerifier') as mock_verifier_cls,
        ):
            mock_verifier = MagicMock()
            mock_verifier.cleanup = AsyncMock(return_value=mock_result)
            mock_verifier_cls.return_value = mock_verifier

            await run_verify_zombie_edges(dry_run=True)

        call_kwargs = mock_verifier.cleanup.call_args
        assert call_kwargs is not None
        _, kwargs = call_kwargs
        assert kwargs.get('dry_run') is True
