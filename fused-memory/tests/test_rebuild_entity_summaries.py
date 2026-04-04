"""Tests for rebuild_entity_summaries across backends, service, MCP tool, and CLI.

Covers:
- GraphitiBackend.list_entity_nodes()           (step 1)
- GraphitiBackend.detect_stale_summaries()      (step 2)
- GraphitiBackend.rebuild_entity_summaries()    (step 3)
- MemoryService.rebuild_entity_summaries()      (step 4)
- MCP tool rebuild_entity_summaries             (step 5)
- DISALLOW_MEMORY_WRITES list                   (step 6)
- RebuildSummariesManager / run_rebuild_summaries (step 7)
- GraphitiBackend.get_all_valid_edges()         (task-423)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.backends.graphiti_client import GraphitiBackend

# ---------------------------------------------------------------------------
# step-1: GraphitiBackend.list_entity_nodes
# ---------------------------------------------------------------------------

class TestListEntityNodes:
    """GraphitiBackend.list_entity_nodes() returns all Entity nodes."""

    @pytest.mark.asyncio
    async def test_returns_entity_nodes(self, mock_config, make_backend, make_graph_mock):
        """Returns list of dicts with uuid/name/summary keys."""
        backend = make_backend(mock_config)
        rows = [
            ['uuid-1', 'Alice', 'Alice knows Bob'],
            ['uuid-2', 'Bob', 'Bob lives in London'],
        ]
        graph = make_graph_mock(rows)
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.list_entity_nodes(group_id='test')
        assert len(result) == 2
        assert result[0] == {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'Alice knows Bob'}
        assert result[1] == {'uuid': 'uuid-2', 'name': 'Bob', 'summary': 'Bob lives in London'}

    @pytest.mark.asyncio
    async def test_empty_graph_returns_empty_list(self, mock_config, make_backend, make_graph_mock):
        """Returns empty list when no Entity nodes exist."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.list_entity_nodes(group_id='test')
        assert result == []

    @pytest.mark.asyncio
    async def test_null_summary_defaults_to_empty_string(self, mock_config, make_backend, make_graph_mock):
        """Nodes with NULL summary field return empty string."""
        backend = make_backend(mock_config)
        rows = [['uuid-1', 'Alice', None]]
        graph = make_graph_mock(rows)
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.list_entity_nodes(group_id='test')
        assert result[0]['summary'] == ''

    @pytest.mark.asyncio
    async def test_uses_ro_query(self, mock_config, make_backend, make_graph_mock):
        """Uses ro_query (read-only) for the list operation."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.list_entity_nodes(group_id='test')
        graph.ro_query.assert_awaited_once()
        graph.query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        """Raises RuntimeError when client is not initialized."""
        backend = GraphitiBackend(mock_config)
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.list_entity_nodes(group_id='test')


# ---------------------------------------------------------------------------
# step-2: GraphitiBackend.detect_stale_summaries
# ---------------------------------------------------------------------------

class TestDetectStaleSummaries:
    """GraphitiBackend.detect_stale_summaries() flags entities with stale summaries."""

    @pytest.mark.asyncio
    async def test_clean_entity_not_flagged(self, mock_config, make_backend):
        """Entity whose summary exactly matches deduped valid edge facts is not returned."""
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'Alice knows Bob'},
        ])
        backend.get_valid_edges_for_node = AsyncMock(return_value=[
            {'uuid': 'e1', 'fact': 'Alice knows Bob', 'name': 'knows'},
        ])
        result = await backend.detect_stale_summaries(group_id='test')
        assert result == []

    @pytest.mark.asyncio
    async def test_duplicate_lines_detected(self, mock_config, make_backend):
        """Entity with duplicate summary lines is flagged with correct duplicate_count."""
        backend = make_backend(mock_config)
        # Summary has 'A\nA\nB' — 'A' appears twice → duplicate_count=1
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'factA\nfactA\nfactB'},
        ])
        backend.get_valid_edges_for_node = AsyncMock(return_value=[
            {'uuid': 'e1', 'fact': 'factA', 'name': 'edge1'},
            {'uuid': 'e2', 'fact': 'factB', 'name': 'edge2'},
        ])
        result = await backend.detect_stale_summaries(group_id='test')
        assert len(result) == 1
        assert result[0]['uuid'] == 'uuid-1'
        assert result[0]['duplicate_count'] == 1

    @pytest.mark.asyncio
    async def test_stale_lines_detected(self, mock_config, make_backend):
        """Entity with lines not in any valid edge fact is flagged with stale_line_count."""
        backend = make_backend(mock_config)
        # 'old stale fact' is in summary but NOT a valid edge
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'current fact\nold stale fact'},
        ])
        backend.get_valid_edges_for_node = AsyncMock(return_value=[
            {'uuid': 'e1', 'fact': 'current fact', 'name': 'edge1'},
        ])
        result = await backend.detect_stale_summaries(group_id='test')
        assert len(result) == 1
        assert result[0]['stale_line_count'] == 1
        assert result[0]['valid_fact_count'] == 1

    @pytest.mark.asyncio
    async def test_mixed_staleness(self, mock_config, make_backend):
        """Entity with both duplicates and stale lines reports both counts."""
        backend = make_backend(mock_config)
        # 'factA' duplicated (1 extra), 'old' is stale
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'factA\nfactA\nold'},
        ])
        backend.get_valid_edges_for_node = AsyncMock(return_value=[
            {'uuid': 'e1', 'fact': 'factA', 'name': 'edge1'},
        ])
        result = await backend.detect_stale_summaries(group_id='test')
        assert len(result) == 1
        assert result[0]['duplicate_count'] == 1
        # 'factA' is in valid_fact_set so it's not counted as stale even when duplicated;
        # only 'old' (not backed by any valid edge) is stale.
        assert result[0]['stale_line_count'] == 1

    @pytest.mark.asyncio
    async def test_empty_summary_not_flagged(self, mock_config, make_backend):
        """Entity with empty summary is not flagged as stale."""
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': ''},
        ])
        backend.get_valid_edges_for_node = AsyncMock(return_value=[])
        result = await backend.detect_stale_summaries(group_id='test')
        assert result == []
        # get_valid_edges_for_node should not have been called for empty-summary node
        backend.get_valid_edges_for_node.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_entities_returns_empty(self, mock_config, make_backend):
        """Empty graph returns empty stale list."""
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[])
        result = await backend.detect_stale_summaries(group_id='test')
        assert result == []

    @pytest.mark.asyncio
    async def test_result_includes_summary_line_count(self, mock_config, make_backend):
        """Stale entity dict includes summary_line_count."""
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'lineA\nlineB\nlineC'},
        ])
        backend.get_valid_edges_for_node = AsyncMock(return_value=[
            {'uuid': 'e1', 'fact': 'lineA', 'name': 'edge1'},
        ])
        result = await backend.detect_stale_summaries(group_id='test')
        assert result[0]['summary_line_count'] == 3

    @pytest.mark.asyncio
    async def test_continues_on_edge_fetch_error(self, mock_config, make_backend):
        """When get_valid_edges_for_node raises for one entity, processing continues for others."""
        backend = make_backend(mock_config)
        # Three entities: entity 1 is clean, entity 2 errors out, entity 3 is stale
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'good fact'},
            {'uuid': 'uuid-2', 'name': 'Bob', 'summary': 'some summary'},
            {'uuid': 'uuid-3', 'name': 'Carol', 'summary': 'factA\nfactA'},
        ])
        backend.get_valid_edges_for_node = AsyncMock(side_effect=[
            [{'uuid': 'e1', 'fact': 'good fact', 'name': 'edge1'}],  # entity 1 clean
            RuntimeError('FalkorDB timeout'),                          # entity 2 errors
            [{'uuid': 'e3', 'fact': 'factA', 'name': 'edge3'}],      # entity 3 stale (dup)
        ])
        result = await backend.detect_stale_summaries(group_id='test')
        # Method must NOT raise; only entity 3 should be in the stale list
        assert len(result) == 1
        assert result[0]['uuid'] == 'uuid-3'

    @pytest.mark.asyncio
    async def test_edge_fetch_error_logs_warning(self, mock_config, make_backend, caplog):
        """A warning-level log is emitted containing the failed entity uuid and exception text."""
        import logging
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'good fact'},
            {'uuid': 'uuid-2', 'name': 'Bob', 'summary': 'some summary'},
            {'uuid': 'uuid-3', 'name': 'Carol', 'summary': 'factA\nfactA'},
        ])
        backend.get_valid_edges_for_node = AsyncMock(side_effect=[
            [{'uuid': 'e1', 'fact': 'good fact', 'name': 'edge1'}],
            RuntimeError('FalkorDB timeout'),
            [{'uuid': 'e3', 'fact': 'factA', 'name': 'edge3'}],
        ])
        with caplog.at_level(logging.WARNING):
            await backend.detect_stale_summaries(group_id='test')
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any('uuid-2' in msg and 'FalkorDB timeout' in msg for msg in warning_messages)


# ---------------------------------------------------------------------------
# step-3: GraphitiBackend.rebuild_entity_summaries
# ---------------------------------------------------------------------------

class TestRebuildEntitySummaries:
    """GraphitiBackend.rebuild_entity_summaries() batch-rebuilds stale entities."""

    @pytest.mark.asyncio
    async def test_rebuilds_only_stale_entities(self, mock_config, make_backend):
        """Only calls refresh_entity_summary for entities flagged by detect_stale_summaries."""
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'stale'},
            {'uuid': 'uuid-2', 'name': 'Bob', 'summary': 'clean'},
        ])
        backend.detect_stale_summaries = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'duplicate_count': 0, 'stale_line_count': 1,
             'valid_fact_count': 0, 'summary_line_count': 1},
        ])
        backend.refresh_entity_summary = AsyncMock(return_value={
            'uuid': 'uuid-1', 'name': 'Alice', 'old_summary': 'stale',
            'new_summary': 'current fact', 'edge_count': 1,
        })
        result = await backend.rebuild_entity_summaries(group_id='test')
        assert result['total_entities'] == 2
        assert result['stale_entities'] == 1
        assert result['rebuilt'] == 1
        backend.refresh_entity_summary.assert_awaited_once_with('uuid-1', group_id='test')

    @pytest.mark.asyncio
    async def test_force_rebuilds_all(self, mock_config, make_backend):
        """With force=True, rebuilds all entities regardless of staleness."""
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'ok'},
            {'uuid': 'uuid-2', 'name': 'Bob', 'summary': 'also ok'},
        ])
        backend.refresh_entity_summary = AsyncMock(return_value={
            'uuid': 'uuid-1', 'name': 'Alice', 'old_summary': 'ok',
            'new_summary': 'ok', 'edge_count': 1,
        })
        result = await backend.rebuild_entity_summaries(group_id='test', force=True)
        assert result['total_entities'] == 2
        assert result['stale_entities'] == 2
        assert result['rebuilt'] == 2
        assert backend.refresh_entity_summary.await_count == 2

    @pytest.mark.asyncio
    async def test_returns_aggregate_result(self, mock_config, make_backend):
        """Returns dict with total_entities, stale_entities, rebuilt, skipped, errors, details."""
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[])
        result = await backend.rebuild_entity_summaries(group_id='test')
        assert set(result.keys()) == {'total_entities', 'stale_entities', 'rebuilt', 'skipped', 'errors', 'details'}

    @pytest.mark.asyncio
    async def test_partial_failure_continues(self, mock_config, make_backend):
        """If one entity's refresh fails, continues with remaining and reports error in results."""
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'stale1'},
            {'uuid': 'uuid-2', 'name': 'Bob', 'summary': 'stale2'},
        ])
        backend.detect_stale_summaries = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'duplicate_count': 0, 'stale_line_count': 1,
             'valid_fact_count': 0, 'summary_line_count': 1},
            {'uuid': 'uuid-2', 'name': 'Bob', 'duplicate_count': 0, 'stale_line_count': 1,
             'valid_fact_count': 0, 'summary_line_count': 1},
        ])
        backend.refresh_entity_summary = AsyncMock(side_effect=[
            RuntimeError('FalkorDB timeout'),
            {'uuid': 'uuid-2', 'name': 'Bob', 'old_summary': 'stale2',
             'new_summary': 'new fact', 'edge_count': 1},
        ])
        result = await backend.rebuild_entity_summaries(group_id='test')
        assert result['errors'] == 1
        assert result['rebuilt'] == 1
        assert len(result['details']) == 2
        error_detail = next(d for d in result['details'] if d['status'] == 'error')
        assert 'FalkorDB timeout' in error_detail['error']

    @pytest.mark.asyncio
    async def test_empty_graph_returns_zero_counts(self, mock_config, make_backend):
        """No entities means all counts are 0."""
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[])
        result = await backend.rebuild_entity_summaries(group_id='test')
        assert result['total_entities'] == 0
        assert result['stale_entities'] == 0
        assert result['rebuilt'] == 0
        assert result['skipped'] == 0
        assert result['errors'] == 0
        assert result['details'] == []

    @pytest.mark.asyncio
    async def test_dry_run_returns_stale_without_rebuilding(self, mock_config, make_backend):
        """With dry_run=True, detects stale entities but does not call refresh_entity_summary."""
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'stale'},
        ])
        backend.detect_stale_summaries = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'duplicate_count': 0, 'stale_line_count': 1,
             'valid_fact_count': 0, 'summary_line_count': 1},
        ])
        backend.refresh_entity_summary = AsyncMock()
        result = await backend.rebuild_entity_summaries(group_id='test', dry_run=True)
        assert result['stale_entities'] == 1
        assert result['rebuilt'] == 0
        assert result['skipped'] == 1
        backend.refresh_entity_summary.assert_not_awaited()
        assert result['details'][0]['status'] == 'skipped_dry_run'


# ---------------------------------------------------------------------------
# step-4: MemoryService.rebuild_entity_summaries
# ---------------------------------------------------------------------------

class TestMemoryServiceRebuildEntitySummaries:
    """MemoryService.rebuild_entity_summaries() delegates to graphiti backend."""

    @pytest.fixture
    def service(self, mock_config):
        """MemoryService with mocked backends (no real DB needed)."""
        from fused_memory.services.memory_service import MemoryService
        svc = MemoryService(mock_config)
        svc.graphiti = MagicMock()
        svc.graphiti.rebuild_entity_summaries = AsyncMock(return_value={
            'total_entities': 5,
            'stale_entities': 2,
            'rebuilt': 2,
            'skipped': 0,
            'errors': 0,
            'details': [],
        })
        svc.mem0 = MagicMock()
        svc.durable_queue = MagicMock()
        svc.durable_queue.enqueue = AsyncMock(return_value=1)
        return svc

    @pytest.mark.asyncio
    async def test_delegates_to_backend(self, service):
        """Calls graphiti.rebuild_entity_summaries with correct group_id, force, dry_run."""
        result = await service.rebuild_entity_summaries(
            project_id='dark_factory',
            force=False,
            dry_run=False,
        )
        service.graphiti.rebuild_entity_summaries.assert_awaited_once_with(
            group_id='dark_factory', force=False, dry_run=False
        )
        assert result['total_entities'] == 5

    @pytest.mark.asyncio
    async def test_journal_logs_on_success(self, service):
        """Write journal records operation with params and result_summary."""
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        service.set_write_journal(mock_journal)
        await service.rebuild_entity_summaries(
            project_id='dark_factory',
            agent_id='test-agent',
        )
        mock_journal.log_write_op.assert_awaited_once()
        call_kwargs = mock_journal.log_write_op.call_args[1]
        assert call_kwargs.get('operation') == 'rebuild_entity_summaries'
        assert call_kwargs.get('project_id') == 'dark_factory'
        assert call_kwargs.get('success') is True

    @pytest.mark.asyncio
    async def test_journal_logs_on_failure(self, service):
        """Backend exception is re-raised; journal records success=False with error message."""
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        service.set_write_journal(mock_journal)
        service.graphiti.rebuild_entity_summaries = AsyncMock(
            side_effect=RuntimeError('FalkorDB unavailable')
        )
        with pytest.raises(RuntimeError, match='FalkorDB unavailable'):
            await service.rebuild_entity_summaries(project_id='dark_factory')
        mock_journal.log_write_op.assert_awaited_once()
        call_kwargs = mock_journal.log_write_op.call_args[1]
        assert call_kwargs.get('success') is False
        assert 'FalkorDB unavailable' in call_kwargs.get('error', '')

    @pytest.mark.asyncio
    async def test_journal_failure_does_not_mask_success(self, service):
        """If journal.log_write_op raises, the successful result is still returned."""
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock(side_effect=RuntimeError('journal full'))
        service.set_write_journal(mock_journal)
        result = await service.rebuild_entity_summaries(project_id='dark_factory')
        # Should NOT raise — journal failure must not mask the successful operation
        assert result['total_entities'] == 5


# ---------------------------------------------------------------------------
# step-5: MCP tool rebuild_entity_summaries
# ---------------------------------------------------------------------------

class TestRebuildEntitySummariesMcpTool:
    """MCP tool rebuild_entity_summaries is registered and delegates correctly."""

    @pytest.fixture
    def mock_service(self):
        """Mock MemoryService for tool registration."""
        svc = AsyncMock()
        svc.rebuild_entity_summaries = AsyncMock(return_value={
            'total_entities': 3,
            'stale_entities': 1,
            'rebuilt': 1,
            'skipped': 0,
            'errors': 0,
            'details': [],
        })
        return svc

    @pytest.fixture
    def mcp_server(self, mock_service):
        """MCP server with mock memory service."""
        from fused_memory.server.tools import create_mcp_server
        return create_mcp_server(mock_service)

    @pytest.mark.asyncio
    async def test_tool_registered(self, mcp_server):
        """Tool appears in MCP server's tool list."""
        tool_names = [t.name for t in await mcp_server.list_tools()]
        assert 'rebuild_entity_summaries' in tool_names

    @pytest.mark.asyncio
    async def test_delegates_to_service(self, mcp_server, mock_service):
        """Calls memory_service.rebuild_entity_summaries with correct args."""
        await mcp_server._tool_manager.call_tool(
            'rebuild_entity_summaries',
            {'project_id': 'dark_factory', 'force': False, 'dry_run': True},
        )
        mock_service.rebuild_entity_summaries.assert_awaited_once()
        call_kwargs = mock_service.rebuild_entity_summaries.call_args[1]
        assert call_kwargs.get('project_id') == 'dark_factory'
        assert call_kwargs.get('dry_run') is True

    @pytest.mark.asyncio
    async def test_invalid_project_id_returns_error(self, mcp_server, mock_service):
        """Returns validation error dict for invalid project_id."""
        import json
        result = await mcp_server._tool_manager.call_tool(
            'rebuild_entity_summaries',
            {'project_id': ''},
        )
        if isinstance(result, list):
            content = result[0].text if hasattr(result[0], 'text') else str(result[0])
            parsed = json.loads(content)
        else:
            parsed = result
        assert 'error' in parsed
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.rebuild_entity_summaries.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_exception_returns_error_dict(self, mcp_server, mock_service):
        """Backend exception returns {error, error_type} dict, not raw exception."""
        import json
        mock_service.rebuild_entity_summaries = AsyncMock(
            side_effect=RuntimeError('FalkorDB connection failed')
        )
        result = await mcp_server._tool_manager.call_tool(
            'rebuild_entity_summaries',
            {'project_id': 'dark_factory'},
        )
        if isinstance(result, list):
            content = result[0].text if hasattr(result[0], 'text') else str(result[0])
            parsed = json.loads(content)
        else:
            parsed = result
        assert 'error' in parsed
        assert 'FalkorDB connection failed' in parsed['error']


# ---------------------------------------------------------------------------
# step-6: DISALLOW_MEMORY_WRITES
# ---------------------------------------------------------------------------

class TestDisallowListForRebuildEntitySummaries:
    """rebuild_entity_summaries must be in DISALLOW_MEMORY_WRITES (not in STAGE1_DISALLOWED)."""

    def test_in_disallow_memory_writes(self):
        """'mcp__fused-memory__rebuild_entity_summaries' must be in DISALLOW_MEMORY_WRITES
        so Stage 3 (read-only) cannot call it."""
        from fused_memory.reconciliation.cli_stage_runner import DISALLOW_MEMORY_WRITES
        assert 'mcp__fused-memory__rebuild_entity_summaries' in DISALLOW_MEMORY_WRITES

    def test_not_in_stage1_disallowed(self):
        """Stage 1 must be able to call rebuild_entity_summaries (not in STAGE1_DISALLOWED)."""
        from fused_memory.reconciliation.cli_stage_runner import STAGE1_DISALLOWED
        assert 'mcp__fused-memory__rebuild_entity_summaries' not in STAGE1_DISALLOWED

    def test_in_stage3_disallowed(self):
        """Stage 3 must NOT be able to call rebuild_entity_summaries."""
        from fused_memory.reconciliation.cli_stage_runner import STAGE3_DISALLOWED
        assert 'mcp__fused-memory__rebuild_entity_summaries' in STAGE3_DISALLOWED


# ---------------------------------------------------------------------------
# step-7: Maintenance CLI — RebuildSummariesManager + run_rebuild_summaries
# ---------------------------------------------------------------------------

class TestRebuildSummariesManager:
    """RebuildSummariesManager.run() delegates to backend.rebuild_entity_summaries."""

    @pytest.mark.asyncio
    async def test_manager_delegates_to_backend(self, mock_config):
        """RebuildSummariesManager.run() calls backend.rebuild_entity_summaries."""
        from fused_memory.maintenance.rebuild_summaries import RebuildSummariesManager
        mock_backend = MagicMock()
        mock_backend.rebuild_entity_summaries = AsyncMock(return_value={
            'total_entities': 4,
            'stale_entities': 2,
            'rebuilt': 2,
            'skipped': 0,
            'errors': 0,
            'details': [],
        })
        manager = RebuildSummariesManager(backend=mock_backend, group_id='test_project')
        result = await manager.run()
        mock_backend.rebuild_entity_summaries.assert_awaited_once_with(
            group_id='test_project', force=False, dry_run=False
        )
        assert result.total_entities == 4
        assert result.rebuilt == 2

    @pytest.mark.asyncio
    async def test_manager_passes_force_and_dry_run(self, mock_config):
        """force and dry_run params are forwarded correctly to the backend."""
        from fused_memory.maintenance.rebuild_summaries import RebuildSummariesManager
        mock_backend = MagicMock()
        mock_backend.rebuild_entity_summaries = AsyncMock(return_value={
            'total_entities': 2,
            'stale_entities': 2,
            'rebuilt': 0,
            'skipped': 2,
            'errors': 0,
            'details': [],
        })
        manager = RebuildSummariesManager(backend=mock_backend, group_id='test_project')
        result = await manager.run(force=True, dry_run=True)
        mock_backend.rebuild_entity_summaries.assert_awaited_once_with(
            group_id='test_project', force=True, dry_run=True
        )
        assert result.skipped == 2

    @pytest.mark.asyncio
    async def test_run_entrypoint_uses_maintenance_service(self, make_fake_maintenance_service):
        """run_rebuild_summaries() uses maintenance_service context manager."""
        from fused_memory.maintenance.rebuild_summaries import run_rebuild_summaries
        mock_cfg = MagicMock()
        mock_service = AsyncMock()
        mock_service.graphiti = MagicMock()
        mock_service.graphiti.rebuild_entity_summaries = AsyncMock(return_value={
            'total_entities': 0,
            'stale_entities': 0,
            'rebuilt': 0,
            'skipped': 0,
            'errors': 0,
            'details': [],
        })
        fake_ctx = make_fake_maintenance_service(mock_cfg, mock_service)
        with patch(
            'fused_memory.maintenance.rebuild_summaries.maintenance_service',
            side_effect=fake_ctx,
        ):
            result = await run_rebuild_summaries(config_path='/fake/config.yaml', group_id='test')
        assert result.total_entities == 0

    @pytest.mark.asyncio
    async def test_run_entrypoint_returns_result(self, make_fake_maintenance_service):
        """Returns the RebuildResult from the manager."""
        from fused_memory.maintenance.rebuild_summaries import RebuildResult, run_rebuild_summaries
        mock_cfg = MagicMock()
        mock_service = AsyncMock()
        mock_service.graphiti = MagicMock()
        mock_service.graphiti.rebuild_entity_summaries = AsyncMock(return_value={
            'total_entities': 7,
            'stale_entities': 3,
            'rebuilt': 3,
            'skipped': 0,
            'errors': 0,
            'details': [{'uuid': 'u1', 'name': 'Alice', 'status': 'rebuilt',
                         'old_summary': 'old', 'new_summary': 'new', 'edge_count': 2}],
        })
        fake_ctx = make_fake_maintenance_service(mock_cfg, mock_service)
        with patch(
            'fused_memory.maintenance.rebuild_summaries.maintenance_service',
            side_effect=fake_ctx,
        ):
            result = await run_rebuild_summaries(group_id='test')
        assert isinstance(result, RebuildResult)
        assert result.total_entities == 7
        assert result.stale_entities == 3
        assert result.rebuilt == 3
        assert len(result.details) == 1


# ---------------------------------------------------------------------------
# task-423: GraphitiBackend.get_all_valid_edges
# ---------------------------------------------------------------------------

class TestGetAllValidEdges:
    """GraphitiBackend.get_all_valid_edges() returns all valid edges grouped by entity UUID."""

    @pytest.mark.asyncio
    async def test_groups_edges_by_entity_uuid(self, mock_config, make_backend, make_graph_mock):
        """Returns dict grouping edges by entity UUID with {uuid, fact, name} shape."""
        backend = make_backend(mock_config)
        rows = [
            ['entity-1', 'edge-a', 'Alice knows Bob', 'knows'],
            ['entity-1', 'edge-b', 'Alice lives in Paris', 'lives_in'],
            ['entity-2', 'edge-c', 'Bob works at Acme', 'works_at'],
        ]
        graph = make_graph_mock(rows)
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.get_all_valid_edges(group_id='test')
        assert set(result.keys()) == {'entity-1', 'entity-2'}
        assert len(result['entity-1']) == 2
        assert len(result['entity-2']) == 1
        assert result['entity-1'][0] == {'uuid': 'edge-a', 'fact': 'Alice knows Bob', 'name': 'knows'}
        assert result['entity-1'][1] == {'uuid': 'edge-b', 'fact': 'Alice lives in Paris', 'name': 'lives_in'}
        assert result['entity-2'][0] == {'uuid': 'edge-c', 'fact': 'Bob works at Acme', 'name': 'works_at'}
