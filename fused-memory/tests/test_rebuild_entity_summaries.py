"""Tests for rebuild_entity_summaries across backends, service, MCP tool, and CLI.

Covers:
- GraphitiBackend.list_entity_nodes()           (step 1)
- GraphitiBackend.detect_stale_summaries()      (step 2)
- GraphitiBackend.rebuild_entity_summaries()    (step 3)
- MemoryService.rebuild_entity_summaries()      (step 4)
- MCP tool rebuild_entity_summaries             (step 5)
- DISALLOW_MEMORY_WRITES list                   (step 6)
- RebuildSummariesManager / run_rebuild_summaries (step 7)
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
        backend.get_all_valid_edges = AsyncMock(return_value={
            'uuid-1': [{'uuid': 'e1', 'fact': 'Alice knows Bob', 'name': 'knows'}],
        })
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
        backend.get_all_valid_edges = AsyncMock(return_value={
            'uuid-1': [
                {'uuid': 'e1', 'fact': 'factA', 'name': 'edge1'},
                {'uuid': 'e2', 'fact': 'factB', 'name': 'edge2'},
            ],
        })
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
        backend.get_all_valid_edges = AsyncMock(return_value={
            'uuid-1': [{'uuid': 'e1', 'fact': 'current fact', 'name': 'edge1'}],
        })
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
        backend.get_all_valid_edges = AsyncMock(return_value={
            'uuid-1': [{'uuid': 'e1', 'fact': 'factA', 'name': 'edge1'}],
        })
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
        backend.get_all_valid_edges = AsyncMock(return_value={})
        result = await backend.detect_stale_summaries(group_id='test')
        assert result == []
        # get_all_valid_edges still called once (before the loop), even if no edges
        backend.get_all_valid_edges.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_entities_returns_empty(self, mock_config, make_backend):
        """Empty graph returns empty stale list."""
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[])
        backend.get_all_valid_edges = AsyncMock(return_value={})
        result = await backend.detect_stale_summaries(group_id='test')
        assert result == []

    @pytest.mark.asyncio
    async def test_result_includes_summary_line_count(self, mock_config, make_backend):
        """Stale entity dict includes summary_line_count."""
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'lineA\nlineB\nlineC'},
        ])
        backend.get_all_valid_edges = AsyncMock(return_value={
            'uuid-1': [{'uuid': 'e1', 'fact': 'lineA', 'name': 'edge1'}],
        })
        result = await backend.detect_stale_summaries(group_id='test')
        assert result[0]['summary_line_count'] == 3


# ---------------------------------------------------------------------------
# step-3: GraphitiBackend.rebuild_entity_summaries
# ---------------------------------------------------------------------------

class TestRebuildEntitySummaries:
    """GraphitiBackend.rebuild_entity_summaries() batch-rebuilds stale entities."""

    @pytest.mark.asyncio
    async def test_rebuilds_only_stale_entities(self, mock_config, make_backend):
        """Only rebuilds entities flagged by _detect_stale_summaries_with_edges."""
        backend = make_backend(mock_config)
        # _detect_stale_summaries_with_edges returns (stale_list, all_edges_dict, total_count)
        backend._detect_stale_summaries_with_edges = AsyncMock(return_value=(
            [{'uuid': 'uuid-1', 'name': 'Alice', 'duplicate_count': 0,
              'stale_line_count': 1, 'valid_fact_count': 0, 'summary_line_count': 1}],
            {'uuid-1': [{'uuid': 'e1', 'fact': 'current fact', 'name': 'edge1'}]},
            2,
        ))
        backend.get_node_text = AsyncMock(return_value=('Alice', 'stale'))
        backend.update_node_summary = AsyncMock()
        result = await backend.rebuild_entity_summaries(group_id='test')
        assert result['total_entities'] == 2
        assert result['stale_entities'] == 1
        assert result['rebuilt'] == 1
        backend.update_node_summary.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_force_rebuilds_all(self, mock_config, make_backend):
        """With force=True, rebuilds all entities using get_all_valid_edges (no refresh_entity_summary)."""
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'ok'},
            {'uuid': 'uuid-2', 'name': 'Bob', 'summary': 'also ok'},
        ])
        backend.get_all_valid_edges = AsyncMock(return_value={
            'uuid-1': [{'uuid': 'e1', 'fact': 'fact1', 'name': 'edge1'}],
            'uuid-2': [{'uuid': 'e2', 'fact': 'fact2', 'name': 'edge2'}],
        })
        backend.get_node_text = AsyncMock(side_effect=[('Alice', 'ok'), ('Bob', 'also ok')])
        backend.update_node_summary = AsyncMock()
        result = await backend.rebuild_entity_summaries(group_id='test', force=True)
        assert result['total_entities'] == 2
        assert result['stale_entities'] == 2
        assert result['rebuilt'] == 2
        assert backend.update_node_summary.await_count == 2

    @pytest.mark.asyncio
    async def test_returns_aggregate_result(self, mock_config, make_backend):
        """Returns dict with total_entities, stale_entities, rebuilt, skipped, errors, details."""
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[])
        backend.get_all_valid_edges = AsyncMock(return_value={})
        result = await backend.rebuild_entity_summaries(group_id='test')
        assert set(result.keys()) == {'total_entities', 'stale_entities', 'rebuilt', 'skipped', 'errors', 'details'}

    @pytest.mark.asyncio
    async def test_partial_failure_continues(self, mock_config, make_backend):
        """If one entity's rebuild fails, continues with remaining and reports error in results."""
        backend = make_backend(mock_config)
        backend._detect_stale_summaries_with_edges = AsyncMock(return_value=(
            [
                {'uuid': 'uuid-1', 'name': 'Alice', 'duplicate_count': 0,
                 'stale_line_count': 1, 'valid_fact_count': 0, 'summary_line_count': 1},
                {'uuid': 'uuid-2', 'name': 'Bob', 'duplicate_count': 0,
                 'stale_line_count': 1, 'valid_fact_count': 0, 'summary_line_count': 1},
            ],
            {
                'uuid-1': [{'uuid': 'e1', 'fact': 'current1', 'name': 'edge1'}],
                'uuid-2': [{'uuid': 'e2', 'fact': 'current2', 'name': 'edge2'}],
            },
            2,
        ))
        # Entity 1 fails at get_node_text; entity 2 succeeds
        backend.get_node_text = AsyncMock(side_effect=[
            RuntimeError('FalkorDB timeout'),
            ('Bob', 'stale2'),
        ])
        backend.update_node_summary = AsyncMock()
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
        backend.get_all_valid_edges = AsyncMock(return_value={})
        result = await backend.rebuild_entity_summaries(group_id='test')
        assert result['total_entities'] == 0
        assert result['stale_entities'] == 0
        assert result['rebuilt'] == 0
        assert result['skipped'] == 0
        assert result['errors'] == 0
        assert result['details'] == []

    @pytest.mark.asyncio
    async def test_dry_run_returns_stale_without_rebuilding(self, mock_config, make_backend):
        """With dry_run=True, detects stale entities but does not call _rebuild_entity_from_edges."""
        backend = make_backend(mock_config)
        backend._detect_stale_summaries_with_edges = AsyncMock(return_value=(
            [{'uuid': 'uuid-1', 'name': 'Alice', 'duplicate_count': 0,
              'stale_line_count': 1, 'valid_fact_count': 0, 'summary_line_count': 1}],
            {'uuid-1': [{'uuid': 'e1', 'fact': 'current fact', 'name': 'edge1'}]},
            1,
        ))
        backend.update_node_summary = AsyncMock()
        result = await backend.rebuild_entity_summaries(group_id='test', dry_run=True)
        assert result['stale_entities'] == 1
        assert result['rebuilt'] == 0
        assert result['skipped'] == 1
        backend.update_node_summary.assert_not_awaited()
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
# N+1 fix step-5: GraphitiBackend.get_all_valid_edges (bulk fetch)
# ---------------------------------------------------------------------------

class TestGetAllValidEdges:
    """GraphitiBackend.get_all_valid_edges() bulk-fetches all valid edges, grouped by entity."""

    @pytest.mark.asyncio
    async def test_groups_edges_by_entity_uuid(self, mock_config, make_backend, make_graph_mock):
        """Returns dict keyed by entity uuid; each value is a list of edge dicts."""
        backend = make_backend(mock_config)
        rows = [
            ['node-1', 'e1', 'factA', 'edge1'],
            ['node-1', 'e2', 'factB', 'edge2'],
            ['node-2', 'e3', 'factC', 'edge3'],
        ]
        graph = make_graph_mock(rows)
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.get_all_valid_edges(group_id='test')
        assert set(result.keys()) == {'node-1', 'node-2'}
        assert len(result['node-1']) == 2
        assert {'uuid': 'e1', 'fact': 'factA', 'name': 'edge1'} in result['node-1']
        assert {'uuid': 'e2', 'fact': 'factB', 'name': 'edge2'} in result['node-1']
        assert result['node-2'] == [{'uuid': 'e3', 'fact': 'factC', 'name': 'edge3'}]

    @pytest.mark.asyncio
    async def test_empty_graph_returns_empty_dict(self, mock_config, make_backend, make_graph_mock):
        """Returns empty dict when no valid edges exist."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.get_all_valid_edges(group_id='test')
        assert result == {}

    @pytest.mark.asyncio
    async def test_uses_ro_query_not_query(self, mock_config, make_backend, make_graph_mock):
        """Uses ro_query (read-only) — graph.query must NOT be called."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.get_all_valid_edges(group_id='test')
        graph.ro_query.assert_awaited_once()
        graph.query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cypher_filters_invalid_at_is_null(self, mock_config, make_backend, make_graph_mock):
        """Cypher query includes 'invalid_at IS NULL' to exclude invalidated edges."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.get_all_valid_edges(group_id='test')
        call_args = graph.ro_query.call_args
        cypher = call_args[0][0] if call_args[0] else call_args[1].get('q', '')
        assert 'invalid_at IS NULL' in cypher

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        """Raises RuntimeError when backend not initialized."""
        backend = GraphitiBackend(mock_config)
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.get_all_valid_edges(group_id='test')


# ---------------------------------------------------------------------------
# N+1 fix step-7: detect_stale_summaries uses bulk get_all_valid_edges
# ---------------------------------------------------------------------------

class TestDetectStaleSummariesBulk:
    """detect_stale_summaries uses get_all_valid_edges (one query) not N per-entity queries."""

    @pytest.mark.asyncio
    async def test_calls_get_all_valid_edges_once_not_per_entity(self, mock_config, make_backend):
        """detect_stale_summaries calls get_all_valid_edges exactly once (not N times)."""
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'factA'},
            {'uuid': 'uuid-2', 'name': 'Bob', 'summary': 'factB'},
        ])
        backend.get_all_valid_edges = AsyncMock(return_value={
            'uuid-1': [{'uuid': 'e1', 'fact': 'factA', 'name': 'edge1'}],
            'uuid-2': [{'uuid': 'e2', 'fact': 'factB', 'name': 'edge2'}],
        })
        backend.get_valid_edges_for_node = AsyncMock()
        await backend.detect_stale_summaries(group_id='test')
        backend.get_all_valid_edges.assert_awaited_once()
        backend.get_valid_edges_for_node.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_correctly_identifies_stale_with_bulk_edges(self, mock_config, make_backend):
        """Stale entity is returned; clean entity is not — using bulk-fetched edges."""
        backend = make_backend(mock_config)
        # uuid-1 is stale (summary has old fact), uuid-2 is clean
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'old fact'},
            {'uuid': 'uuid-2', 'name': 'Bob', 'summary': 'current fact'},
        ])
        backend.get_all_valid_edges = AsyncMock(return_value={
            'uuid-1': [{'uuid': 'e1', 'fact': 'current fact', 'name': 'edge1'}],
            'uuid-2': [{'uuid': 'e2', 'fact': 'current fact', 'name': 'edge2'}],
        })
        result = await backend.detect_stale_summaries(group_id='test')
        assert len(result) == 1
        assert result[0]['uuid'] == 'uuid-1'

    @pytest.mark.asyncio
    async def test_empty_summary_entities_skipped_with_bulk_path(self, mock_config, make_backend):
        """Empty-summary entity is skipped even with bulk edges data source."""
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': ''},
        ])
        backend.get_all_valid_edges = AsyncMock(return_value={
            'uuid-1': [{'uuid': 'e1', 'fact': 'some fact', 'name': 'edge1'}],
        })
        result = await backend.detect_stale_summaries(group_id='test')
        assert result == []


# ---------------------------------------------------------------------------
# N+1 fix step-9: rebuild_entity_summaries parallel + no re-fetch
# ---------------------------------------------------------------------------

class TestRebuildEntitySummariesParallel:
    """rebuild_entity_summaries uses _rebuild_entity_from_edges (no re-fetch) + asyncio.gather."""

    @pytest.mark.asyncio
    async def test_non_force_does_not_call_get_valid_edges_for_node(self, mock_config, make_backend):
        """Non-force path: get_valid_edges_for_node is never called (edges come from bulk fetch)."""
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'stale1'},
            {'uuid': 'uuid-2', 'name': 'Bob', 'summary': 'stale2'},
        ])
        backend.get_all_valid_edges = AsyncMock(return_value={
            'uuid-1': [{'uuid': 'e1', 'fact': 'current1', 'name': 'edge1'}],
            'uuid-2': [{'uuid': 'e2', 'fact': 'current2', 'name': 'edge2'}],
        })
        backend.get_node_text = AsyncMock(side_effect=[
            ('Alice', 'stale1'),
            ('Bob', 'stale2'),
        ])
        backend.update_node_summary = AsyncMock()
        backend.get_valid_edges_for_node = AsyncMock()

        result = await backend.rebuild_entity_summaries(group_id='test')

        backend.get_valid_edges_for_node.assert_not_awaited()
        assert result['rebuilt'] == 2

    @pytest.mark.asyncio
    async def test_force_calls_get_all_valid_edges_once_not_refresh_entity_summary(
        self, mock_config, make_backend
    ):
        """Force path: get_all_valid_edges called once; refresh_entity_summary never called."""
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'ok'},
            {'uuid': 'uuid-2', 'name': 'Bob', 'summary': 'also ok'},
        ])
        backend.get_all_valid_edges = AsyncMock(return_value={
            'uuid-1': [{'uuid': 'e1', 'fact': 'fact1', 'name': 'edge1'}],
            'uuid-2': [{'uuid': 'e2', 'fact': 'fact2', 'name': 'edge2'}],
        })
        backend.get_node_text = AsyncMock(side_effect=[
            ('Alice', 'ok'),
            ('Bob', 'also ok'),
        ])
        backend.update_node_summary = AsyncMock()
        backend.refresh_entity_summary = AsyncMock()

        result = await backend.rebuild_entity_summaries(group_id='test', force=True)

        backend.get_all_valid_edges.assert_awaited_once()
        backend.refresh_entity_summary.assert_not_awaited()
        assert result['total_entities'] == 2
        assert result['rebuilt'] == 2

    @pytest.mark.asyncio
    async def test_concurrent_all_five_entities_processed(self, mock_config, make_backend):
        """With 5 target entities, all 5 get update_node_summary calls (parallel processing)."""
        n = 5
        entities = [
            {'uuid': f'uuid-{i}', 'name': f'Entity{i}', 'summary': f'stale{i}'}
            for i in range(n)
        ]
        edges = {
            f'uuid-{i}': [{'uuid': f'e{i}', 'fact': f'current{i}', 'name': 'edge'}]
            for i in range(n)
        }
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=entities)
        backend.get_all_valid_edges = AsyncMock(return_value=edges)
        backend.get_node_text = AsyncMock(side_effect=[
            (f'Entity{i}', f'stale{i}') for i in range(n)
        ])
        backend.update_node_summary = AsyncMock()

        result = await backend.rebuild_entity_summaries(group_id='test')

        assert backend.update_node_summary.await_count == n
        assert result['rebuilt'] == n
        assert result['errors'] == 0

    @pytest.mark.asyncio
    async def test_partial_failure_in_update_does_not_cancel_others(
        self, mock_config, make_backend
    ):
        """If update_node_summary fails for one entity, others still complete.

        asyncio.gather with return_exceptions=True ensures partial failures are
        captured rather than propagated, so the gather completes for all entities.
        """
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'stale1'},
            {'uuid': 'uuid-2', 'name': 'Bob', 'summary': 'stale2'},
        ])
        backend.get_all_valid_edges = AsyncMock(return_value={
            'uuid-1': [{'uuid': 'e1', 'fact': 'current1', 'name': 'edge1'}],
            'uuid-2': [{'uuid': 'e2', 'fact': 'current2', 'name': 'edge2'}],
        })
        backend.get_node_text = AsyncMock(side_effect=[
            ('Alice', 'stale1'),
            ('Bob', 'stale2'),
        ])
        # First entity's write fails, second succeeds
        backend.update_node_summary = AsyncMock(side_effect=[
            RuntimeError('FalkorDB timeout'),
            None,
        ])

        result = await backend.rebuild_entity_summaries(group_id='test')

        assert result['rebuilt'] == 1
        assert result['errors'] == 1


# ---------------------------------------------------------------------------
# task-432 step-1: _rebuild_entity_from_edges accepts old_summary kwarg
# ---------------------------------------------------------------------------

class TestRebuildEntityFromEdgesOldSummary:
    """_rebuild_entity_from_edges accepts old_summary kwarg and skips get_node_text."""

    @pytest.mark.asyncio
    async def test_rebuild_entity_from_edges_uses_passed_old_summary(
        self, mock_config, make_backend
    ):
        """When old_summary kwarg is provided, it is used in the result dict and
        get_node_text is NOT called."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock()
        backend.update_node_summary = AsyncMock()

        edges = [{'uuid': 'e1', 'fact': 'current fact', 'name': 'edge1'}]
        result = await backend._rebuild_entity_from_edges(
            'uuid-1', 'Alice', edges, group_id='test', old_summary='prior summary'
        )

        assert result['old_summary'] == 'prior summary'
        assert result['new_summary'] == 'current fact'
        assert result['uuid'] == 'uuid-1'
        assert result['name'] == 'Alice'
        backend.get_node_text.assert_not_awaited()
