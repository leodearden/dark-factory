"""Tests for delete_entity across backends, service, and MCP tool.

Covers:
- GraphitiBackend.get_connected_entity_uuids()
- GraphitiBackend.delete_entity()
- MemoryService.delete_entity()
- MCP tool delete_entity in tools.py
- DISALLOW_MEMORY_WRITES list in cli_stage_runner.py
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import extract_cypher, extract_params

from fused_memory.backends.graphiti_client import (
    ActiveEdgesError,
    GraphitiBackend,
    NodeNotFoundError,
)

# ---------------------------------------------------------------------------
# step-1: GraphitiBackend.get_connected_entity_uuids
# ---------------------------------------------------------------------------

class TestGetConnectedEntityUuids:
    """GraphitiBackend.get_connected_entity_uuids(uuid, *, group_id) returns neighbour UUIDs."""

    @pytest.mark.asyncio
    async def test_returns_neighbour_uuids(self, mock_config, make_backend, make_graph_mock):
        """Parses distinct neighbour UUID strings from ro_query result_set rows."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[['uuid-a'], ['uuid-b']])
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.get_connected_entity_uuids('node-uuid', group_id='test')
        assert result == ['uuid-a', 'uuid-b']

    @pytest.mark.asyncio
    async def test_uses_ro_query_not_query(self, mock_config, make_backend, make_graph_mock):
        """Uses ro_query (read-only) and NOT the write query path."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.get_connected_entity_uuids('node-uuid', group_id='test')
        graph.ro_query.assert_awaited_once()
        graph.query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_passes_uuid_as_param(self, mock_config, make_backend, make_graph_mock):
        """Passes the target uuid as a Cypher parameter."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[])
        backend._driver._get_graph = MagicMock(return_value=graph)
        target = 'my-node-uuid'
        await backend.get_connected_entity_uuids(target, group_id='test')
        params = extract_params(graph.ro_query.call_args)
        assert params.get('uuid') == target

    @pytest.mark.asyncio
    async def test_cypher_filters_invalid_and_excludes_self(self, mock_config, make_backend, make_graph_mock):
        """Cypher matches RELATES_TO, filters invalid_at IS NULL, and excludes self (m.uuid <> $uuid)."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.get_connected_entity_uuids('node-uuid', group_id='test')
        cypher = extract_cypher(graph.ro_query.call_args)
        assert 'RELATES_TO' in cypher
        assert 'invalid_at IS NULL' in cypher
        assert 'm.uuid <> $uuid' in cypher or "m.uuid <>" in cypher

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        """Raises RuntimeError when constructed uninitialized (no client)."""
        backend = GraphitiBackend(mock_config)
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.get_connected_entity_uuids('node-uuid', group_id='test')

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_neighbours(self, mock_config, make_backend, make_graph_mock):
        """Returns empty list when no neighbours found."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[])
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.get_connected_entity_uuids('isolated-uuid', group_id='test')
        assert result == []


# ---------------------------------------------------------------------------
# step-3: GraphitiBackend.delete_entity
# ---------------------------------------------------------------------------

class TestDeleteEntityBackend:
    """GraphitiBackend.delete_entity(uuid, *, group_id, force=False) orchestrates deletion."""

    @pytest.fixture
    def backend_with_mocks(self, mock_config, make_backend):
        """GraphitiBackend with sub-methods mocked for orchestration testing."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(return_value=('NodeName', 'some summary'))
        backend.get_valid_edges_for_node = AsyncMock(return_value=[])
        backend.get_connected_entity_uuids = AsyncMock(return_value=[])
        backend.delete_entity_node = AsyncMock()
        backend.refresh_entity_summary = AsyncMock(return_value={
            'uuid': 'target-uuid',
            'name': 'Neighbour',
            'old_summary': 'old',
            'new_summary': 'new',
            'edge_count': 1,
        })
        return backend

    @pytest.mark.asyncio
    async def test_calls_get_node_text_to_validate(self, backend_with_mocks):
        """Calls get_node_text to validate existence before proceeding."""
        backend = backend_with_mocks
        await backend.delete_entity(  'target-uuid', group_id='test')
        backend.get_node_text.assert_awaited_once()
        assert backend.get_node_text.call_args[0][0] == 'target-uuid'

    @pytest.mark.asyncio
    async def test_raises_node_not_found_when_missing(self, mock_config, make_backend):
        """Re-raises NodeNotFoundError from get_node_text when node is missing."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(side_effect=NodeNotFoundError('missing'))
        backend.delete_entity_node = AsyncMock()
        with pytest.raises(NodeNotFoundError):
            await backend.delete_entity('missing-uuid', group_id='test')
        backend.delete_entity_node.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_happy_path_no_edges(self, backend_with_mocks):
        """HAPPY PATH: no active edges → deletes node, returns audit dict."""
        backend = backend_with_mocks
        result = await backend.delete_entity('target-uuid', group_id='test')
        backend.delete_entity_node.assert_awaited_once_with('target-uuid', group_id='test')
        assert result['deleted_uuid'] == 'target-uuid'
        assert result['deleted_name'] == 'NodeName'
        assert result['active_edge_count'] == 0
        assert result['forced'] is False
        assert result['connected_refreshed'] == []

    @pytest.mark.asyncio
    async def test_guard_raises_active_edges_error_when_force_false(self, mock_config, make_backend):
        """GUARD: active edges + force=False raises ActiveEdgesError, delete NOT called."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(return_value=('NodeName', ''))
        # Return 2 active edges
        backend.get_valid_edges_for_node = AsyncMock(return_value=[
            {'uuid': 'e1', 'fact': 'fact1', 'name': 'e1'},
            {'uuid': 'e2', 'fact': 'fact2', 'name': 'e2'},
        ])
        backend.delete_entity_node = AsyncMock()
        with pytest.raises(ActiveEdgesError):
            await backend.delete_entity('target-uuid', group_id='test', force=False)
        backend.delete_entity_node.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_force_true_deletes_with_active_edges(self, mock_config, make_backend):
        """FORCE: active edges + force=True proceeds to delete and refreshes neighbours."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(return_value=('NodeName', ''))
        backend.get_valid_edges_for_node = AsyncMock(return_value=[
            {'uuid': 'e1', 'fact': 'fact1', 'name': 'e1'},
            {'uuid': 'e2', 'fact': 'fact2', 'name': 'e2'},
        ])
        backend.get_connected_entity_uuids = AsyncMock(return_value=['nbr-a', 'nbr-b'])
        backend.delete_entity_node = AsyncMock()
        backend.refresh_entity_summary = AsyncMock(return_value={
            'uuid': 'nbr-a', 'name': 'N', 'old_summary': '', 'new_summary': '', 'edge_count': 0
        })
        result = await backend.delete_entity('target-uuid', group_id='test', force=True)
        backend.delete_entity_node.assert_awaited_once_with('target-uuid', group_id='test')
        assert backend.refresh_entity_summary.await_count == 2
        assert result['forced'] is True
        assert result['active_edge_count'] == 2
        assert result['connected_refreshed'] == ['nbr-a', 'nbr-b']

    @pytest.mark.asyncio
    async def test_order_collect_before_delete_before_refresh(self, mock_config, make_backend):
        """ORDER: get_connected_entity_uuids called BEFORE delete_entity_node, which is BEFORE refresh."""
        backend = make_backend(mock_config)
        call_order: list[str] = []

        backend.get_node_text = AsyncMock(return_value=('N', ''))
        backend.get_valid_edges_for_node = AsyncMock(return_value=[])
        backend.get_connected_entity_uuids = AsyncMock(
            side_effect=lambda *a, **kw: call_order.append('collect') or ['nbr-x']
        )
        backend.delete_entity_node = AsyncMock(
            side_effect=lambda *a, **kw: call_order.append('delete')
        )
        backend.refresh_entity_summary = AsyncMock(
            side_effect=lambda *a, **kw: call_order.append('refresh') or {}
        )
        await backend.delete_entity('target-uuid', group_id='test')
        assert call_order == ['collect', 'delete', 'refresh']


# ---------------------------------------------------------------------------
# step-5: MemoryService.delete_entity
# ---------------------------------------------------------------------------

class TestMemoryServiceDeleteEntity:
    """MemoryService.delete_entity() delegates to graphiti backend with journaling."""

    @pytest.fixture
    def service(self, mock_config):
        """MemoryService with mocked graphiti backend."""
        from fused_memory.services.memory_service import MemoryService
        svc = MemoryService(mock_config)
        svc.graphiti = MagicMock()
        svc.graphiti.delete_entity = AsyncMock(return_value={
            'deleted_uuid': 'e-uuid',
            'deleted_name': 'N',
            'active_edge_count': 0,
            'forced': False,
            'connected_refreshed': [],
        })
        svc.mem0 = MagicMock()
        svc.durable_queue = MagicMock()
        svc.durable_queue.enqueue = AsyncMock(return_value=1)
        return svc

    @pytest.mark.asyncio
    async def test_delegates_to_graphiti_backend(self, service):
        """Calls graphiti.delete_entity with entity_uuid, group_id, and force."""
        result = await service.delete_entity(
            entity_uuid='e-uuid',
            project_id='dark_factory',
        )
        service.graphiti.delete_entity.assert_awaited_once_with(
            'e-uuid', group_id='dark_factory', force=False,
        )
        assert result['deleted_uuid'] == 'e-uuid'

    @pytest.mark.asyncio
    async def test_force_threaded_through(self, service):
        """force=True is passed through to the backend call."""
        await service.delete_entity(
            entity_uuid='e-uuid',
            project_id='dark_factory',
            force=True,
        )
        call_kwargs = service.graphiti.delete_entity.call_args[1]
        assert call_kwargs.get('force') is True

    @pytest.mark.asyncio
    async def test_logs_via_write_journal_when_present(self, service):
        """Logs write op via write journal when journal is set."""
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        service.set_write_journal(mock_journal)
        await service.delete_entity(
            entity_uuid='e-uuid',
            project_id='dark_factory',
            agent_id='test-agent',
        )
        mock_journal.log_write_op.assert_awaited_once()
        call_kwargs = mock_journal.log_write_op.call_args[1]
        assert call_kwargs.get('operation') == 'delete_entity'
        assert call_kwargs.get('project_id') == 'dark_factory'
        params = call_kwargs.get('params', {})
        assert params.get('entity_uuid') == 'e-uuid'
        assert 'force' in params

    @pytest.mark.asyncio
    async def test_journal_failure_does_not_mask_success(self, service):
        """Journal failure does not prevent returning the successful result."""
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock(side_effect=RuntimeError('journal down'))
        service.set_write_journal(mock_journal)
        result = await service.delete_entity(
            entity_uuid='e-uuid',
            project_id='dark_factory',
        )
        assert result['deleted_uuid'] == 'e-uuid'

    @pytest.mark.asyncio
    async def test_backend_failure_is_journaled_with_success_false(self, service):
        """When backend raises, journal is still called with success=False."""
        service.graphiti.delete_entity = AsyncMock(
            side_effect=ValueError('node not found')
        )
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        service.set_write_journal(mock_journal)
        with pytest.raises(ValueError, match='node not found'):
            await service.delete_entity(
                entity_uuid='e-uuid',
                project_id='dark_factory',
            )
        mock_journal.log_write_op.assert_awaited_once()
        call_kwargs = mock_journal.log_write_op.call_args[1]
        assert call_kwargs.get('success') is False

    @pytest.mark.asyncio
    async def test_works_without_journal(self, service):
        """Works correctly when no write journal is set."""
        service._write_journal = None
        result = await service.delete_entity(
            entity_uuid='e-uuid',
            project_id='dark_factory',
        )
        assert result['deleted_uuid'] == 'e-uuid'


# ---------------------------------------------------------------------------
# step-7: MCP tool delete_entity
# ---------------------------------------------------------------------------

class TestDeleteEntityMcpTool:
    """MCP tool delete_entity is registered and delegates correctly."""

    @pytest.fixture
    def mock_service(self):
        """Mock MemoryService for tool registration."""
        svc = AsyncMock()
        svc.delete_entity = AsyncMock(return_value={
            'deleted_uuid': 'e-uuid',
            'deleted_name': 'N',
            'active_edge_count': 0,
            'forced': False,
            'connected_refreshed': [],
        })
        return svc

    @pytest.fixture
    def mcp_server(self, mock_service):
        """MCP server with mock memory service."""
        from fused_memory.server.tools import create_mcp_server
        return create_mcp_server(mock_service)

    @pytest.mark.asyncio
    async def test_tool_is_registered(self, mcp_server):
        """delete_entity is registered as an MCP tool."""
        tool_names = [t.name for t in await mcp_server.list_tools()]
        assert 'delete_entity' in tool_names

    @pytest.mark.asyncio
    async def test_delegates_to_memory_service(self, mcp_server, mock_service):
        """Tool calls memory_service.delete_entity with entity_uuid and project_id."""
        await mcp_server._tool_manager.call_tool(
            'delete_entity',
            {
                'entity_uuid': 'e-uuid',
                'project_id': 'dark_factory',
            },
        )
        mock_service.delete_entity.assert_awaited_once()
        call_kwargs = mock_service.delete_entity.call_args[1]
        assert call_kwargs.get('entity_uuid') == 'e-uuid'
        assert call_kwargs.get('project_id') == 'dark_factory'

    @pytest.mark.asyncio
    async def test_force_defaults_to_false(self, mcp_server, mock_service):
        """force defaults to False when omitted by the caller."""
        await mcp_server._tool_manager.call_tool(
            'delete_entity',
            {'entity_uuid': 'e-uuid', 'project_id': 'dark_factory'},
        )
        call_kwargs = mock_service.delete_entity.call_args[1]
        assert call_kwargs.get('force') is False

    @pytest.mark.asyncio
    async def test_force_true_passed_through(self, mcp_server, mock_service):
        """force=True is passed through to the service."""
        await mcp_server._tool_manager.call_tool(
            'delete_entity',
            {'entity_uuid': 'e-uuid', 'project_id': 'dark_factory', 'force': True},
        )
        call_kwargs = mock_service.delete_entity.call_args[1]
        assert call_kwargs.get('force') is True

    @pytest.mark.asyncio
    async def test_empty_project_id_returns_error(self, mcp_server, mock_service):
        """Empty project_id returns validation error dict and service NOT called."""
        import json
        result = await mcp_server._tool_manager.call_tool(
            'delete_entity',
            {'entity_uuid': 'e-uuid', 'project_id': ''},
        )
        if isinstance(result, list):
            content = result[0].text if hasattr(result[0], 'text') else str(result[0])
            parsed = json.loads(content)
        else:
            parsed = result
        assert 'error' in parsed
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.delete_entity.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_entity_uuid_returns_error(self, mcp_server, mock_service):
        """Empty entity_uuid returns validation error dict and service NOT called."""
        import json
        result = await mcp_server._tool_manager.call_tool(
            'delete_entity',
            {'entity_uuid': '', 'project_id': 'dark_factory'},
        )
        if isinstance(result, list):
            content = result[0].text if hasattr(result[0], 'text') else str(result[0])
            parsed = json.loads(content)
        else:
            parsed = result
        assert 'error' in parsed
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.delete_entity.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_whitespace_entity_uuid_returns_error(self, mcp_server, mock_service):
        """Whitespace-only entity_uuid returns validation error dict."""
        import json
        result = await mcp_server._tool_manager.call_tool(
            'delete_entity',
            {'entity_uuid': '   ', 'project_id': 'dark_factory'},
        )
        if isinstance(result, list):
            content = result[0].text if hasattr(result[0], 'text') else str(result[0])
            parsed = json.loads(content)
        else:
            parsed = result
        assert 'error' in parsed
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.delete_entity.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_exception_returns_error_dict(self, mcp_server, mock_service):
        """Exception from memory_service returns error dict (not raised)."""
        import json
        mock_service.delete_entity = AsyncMock(
            side_effect=RuntimeError('FalkorDB connection failed')
        )
        result = await mcp_server._tool_manager.call_tool(
            'delete_entity',
            {'entity_uuid': 'e-uuid', 'project_id': 'dark_factory'},
        )
        if isinstance(result, list):
            content = result[0].text if hasattr(result[0], 'text') else str(result[0])
            parsed = json.loads(content)
        else:
            parsed = result
        assert 'error' in parsed
        assert 'FalkorDB connection failed' in parsed['error']


# ---------------------------------------------------------------------------
# step-9: DISALLOW_MEMORY_WRITES list in cli_stage_runner.py
# ---------------------------------------------------------------------------

class TestDisallowListForDeleteEntity:
    """delete_entity must be in DISALLOW_MEMORY_WRITES but NOT in STAGE1_DISALLOWED."""

    def test_delete_entity_in_disallow_memory_writes(self):
        """'mcp__fused-memory__delete_entity' must be in DISALLOW_MEMORY_WRITES
        so Stage 3 (read-only) cannot call it."""
        from fused_memory.reconciliation.cli_stage_runner import DISALLOW_MEMORY_WRITES
        assert 'mcp__fused-memory__delete_entity' in DISALLOW_MEMORY_WRITES

    def test_delete_entity_not_in_stage1_disallowed(self):
        """Stage 1 must be able to call delete_entity (not in STAGE1_DISALLOWED)."""
        from fused_memory.reconciliation.cli_stage_runner import STAGE1_DISALLOWED
        assert 'mcp__fused-memory__delete_entity' not in STAGE1_DISALLOWED

    def test_delete_entity_in_stage3_disallowed(self):
        """Stage 3 must NOT be able to call delete_entity (in STAGE3_DISALLOWED
        via DISALLOW_MEMORY_WRITES)."""
        from fused_memory.reconciliation.cli_stage_runner import STAGE3_DISALLOWED
        assert 'mcp__fused-memory__delete_entity' in STAGE3_DISALLOWED
