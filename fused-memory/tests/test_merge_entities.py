"""Tests for merge_entities across backends, service, and MCP tool.

Covers:
- GraphitiBackend.redirect_node_edges()
- GraphitiBackend.delete_entity_node()
- GraphitiBackend.merge_entities()
- MemoryService.merge_entities()
- MCP tool merge_entities in tools.py
- DISALLOW_MEMORY_WRITES list in cli_stage_runner.py
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from conftest import extract_cypher, extract_params

from fused_memory.backends.graphiti_client import GraphitiBackend, NodeNotFoundError

# ---------------------------------------------------------------------------
# step-1: GraphitiBackend.redirect_node_edges
# ---------------------------------------------------------------------------

class TestRedirectNodeEdges:
    """GraphitiBackend.redirect_node_edges(deprecated_uuid, surviving_uuid) redirects edges."""

    @pytest.mark.asyncio
    async def test_redirects_outgoing_edges(self, mock_config, make_backend, make_graph_mock):
        """Creates new edges from surviving→target and deletes old edges from deprecated→target."""
        backend = make_backend(mock_config)
        # Simulate query calls: first (inter-node delete) returns empty,
        # second (outgoing redirect) returns empty, third (incoming) returns empty
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.redirect_node_edges('dep-uuid', 'sur-uuid', group_id='test')
        # Should have called graph.query at least 3 times (inter-node, outgoing, incoming)
        assert graph.query.await_count >= 3
        assert 'outgoing_redirected' in result
        assert 'incoming_redirected' in result
        assert 'inter_node_deleted' in result

    @pytest.mark.asyncio
    async def test_deletes_inter_node_edges(self, mock_config, make_backend, make_graph_mock):
        """Edges between deprecated and surviving nodes are removed before redirect."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        dep_uuid = 'dep-abc'
        sur_uuid = 'sur-xyz'
        await backend.redirect_node_edges(dep_uuid, sur_uuid, group_id='test')
        # First query call should target inter-node edges
        first_call = graph.query.call_args_list[0]
        params = extract_params(first_call)
        # Both UUIDs should appear in params
        assert dep_uuid in params.values() or any(
            dep_uuid in str(v) for v in params.values()
        )

    @pytest.mark.asyncio
    async def test_returns_count_dict(self, mock_config, make_backend, make_graph_mock):
        """Returns dict with outgoing_redirected, incoming_redirected, inter_node_deleted counts."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.redirect_node_edges('dep-uuid', 'sur-uuid', group_id='test')
        assert isinstance(result['outgoing_redirected'], int)
        assert isinstance(result['incoming_redirected'], int)
        assert isinstance(result['inter_node_deleted'], int)

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        """Raises RuntimeError when client is not initialized."""
        backend = GraphitiBackend(mock_config)  # client is None
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.redirect_node_edges('dep-uuid', 'sur-uuid', group_id='test')

    @pytest.mark.asyncio
    async def test_handles_empty_edge_sets(self, mock_config, make_backend, make_graph_mock):
        """When no edges exist, returns zeros for all counts."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.redirect_node_edges('dep-uuid', 'sur-uuid', group_id='test')
        assert result['outgoing_redirected'] == 0
        assert result['incoming_redirected'] == 0
        assert result['inter_node_deleted'] == 0


# ---------------------------------------------------------------------------
# step-3: GraphitiBackend.delete_entity_node
# ---------------------------------------------------------------------------

class TestDeleteEntityNode:
    """GraphitiBackend.delete_entity_node(uuid) removes an entity node."""

    @pytest.mark.asyncio
    async def test_executes_detach_delete(self, mock_config, make_backend, make_graph_mock):
        """Issues DETACH DELETE Cypher for the given UUID."""
        backend = make_backend(mock_config)
        # Pre-check uses ro_query; DETACH DELETE uses query
        graph = make_graph_mock(ro_rows=[['NodeName', 'some summary']], q_rows=[])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.delete_entity_node('node-uuid-1', group_id='test')
        assert graph.ro_query.await_count == 1
        assert graph.query.await_count == 1
        # The single query call should contain DETACH DELETE
        cypher = extract_cypher(graph.query.call_args)
        assert 'DETACH DELETE' in cypher

    @pytest.mark.asyncio
    async def test_passes_uuid_as_param(self, mock_config, make_backend, make_graph_mock):
        """Passes node UUID as parameter to both the pre-check and delete Cypher calls."""
        backend = make_backend(mock_config)
        node_uuid = 'my-node-uuid'
        graph = make_graph_mock(ro_rows=[['Name', '']], q_rows=[])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.delete_entity_node(node_uuid, group_id='test')
        # Pre-check (ro_query) should pass uuid param
        ro_params = extract_params(graph.ro_query.call_args)
        assert ro_params.get('uuid') == node_uuid
        # Delete (query) should also pass uuid param
        q_params = extract_params(graph.query.call_args)
        assert q_params.get('uuid') == node_uuid
        # Exactly-once routing contract: each slot called exactly once
        graph.ro_query.assert_awaited_once()
        graph.query.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_node_not_found_when_missing(self, mock_config, make_backend, make_graph_mock):
        """Raises NodeNotFoundError when pre-check returns no rows."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        with pytest.raises(NodeNotFoundError):
            await backend.delete_entity_node('missing-uuid', group_id='test')
        graph.ro_query.assert_awaited_once()  # pre-check ran exactly once
        graph.query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        """Raises RuntimeError when client is not initialized."""
        backend = GraphitiBackend(mock_config)  # client is None
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.delete_entity_node('node-uuid-1', group_id='test')


# ---------------------------------------------------------------------------
# step-5: GraphitiBackend.merge_entities
# ---------------------------------------------------------------------------

class TestMergeEntities:
    """GraphitiBackend.merge_entities(deprecated_uuid, surviving_uuid) merges two nodes."""

    @pytest.fixture
    def backend_with_mocks(self, mock_config, make_backend):
        """GraphitiBackend with sub-methods mocked for orchestration testing."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(side_effect=[
            ('DeprecatedName', 'old dep summary'),   # first call: deprecated node
            ('SurvivingName', 'old sur summary'),     # second call: surviving node
        ])
        backend.redirect_node_edges = AsyncMock(return_value={
            'outgoing_redirected': 2,
            'incoming_redirected': 1,
            'inter_node_deleted': 0,
        })
        backend.delete_entity_node = AsyncMock()
        backend.refresh_entity_summary = AsyncMock(return_value={
            'uuid': 'sur-uuid',
            'name': 'SurvivingName',
            'old_summary': 'old sur summary',
            'new_summary': 'SurvivingName knows Foo\nDeprecatedName knows Bar',
            'edge_count': 3,
        })
        return backend

    @pytest.mark.asyncio
    async def test_validates_both_nodes_exist(self, backend_with_mocks):
        """Calls get_node_text for both UUIDs to validate existence."""
        backend = backend_with_mocks
        await backend.merge_entities('dep-uuid', 'sur-uuid', group_id='test')
        assert backend.get_node_text.await_count == 2
        calls = [c[0][0] for c in backend.get_node_text.call_args_list]
        assert 'dep-uuid' in calls
        assert 'sur-uuid' in calls

    @pytest.mark.asyncio
    async def test_raises_node_not_found_for_deprecated(self, mock_config, make_backend):
        """Raises NodeNotFoundError when deprecated node doesn't exist."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(side_effect=NodeNotFoundError('not found'))
        with pytest.raises(NodeNotFoundError):
            await backend.merge_entities('missing-dep', 'sur-uuid', group_id='test')

    @pytest.mark.asyncio
    async def test_raises_node_not_found_for_surviving(self, mock_config, make_backend):
        """Raises NodeNotFoundError when surviving node doesn't exist."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(side_effect=[
            ('DepName', ''),           # deprecated exists
            NodeNotFoundError('not found'),  # surviving missing
        ])
        with pytest.raises(NodeNotFoundError):
            await backend.merge_entities('dep-uuid', 'missing-sur', group_id='test')

    @pytest.mark.asyncio
    async def test_calls_in_correct_order(self, backend_with_mocks):
        """Calls redirect_node_edges, then delete_entity_node, then refresh_entity_summary."""
        backend = backend_with_mocks
        call_order = []
        backend.redirect_node_edges = AsyncMock(
            side_effect=lambda *a, **kw: call_order.append('redirect') or {
                'outgoing_redirected': 0, 'incoming_redirected': 0, 'inter_node_deleted': 0
            }
        )
        backend.delete_entity_node = AsyncMock(
            side_effect=lambda *a, **kw: call_order.append('delete')
        )
        backend.refresh_entity_summary = AsyncMock(
            side_effect=lambda *a, **kw: call_order.append('refresh') or {
                'uuid': 'sur-uuid', 'name': 'S', 'old_summary': '', 'new_summary': '', 'edge_count': 0
            }
        )
        await backend.merge_entities('dep-uuid', 'sur-uuid', group_id='test')
        assert call_order == ['redirect', 'delete', 'refresh']

    @pytest.mark.asyncio
    async def test_returns_audit_dict(self, backend_with_mocks):
        """Returns audit dict with expected keys."""
        backend = backend_with_mocks
        result = await backend.merge_entities('dep-uuid', 'sur-uuid', group_id='test')
        assert result['surviving_uuid'] == 'sur-uuid'
        assert result['deprecated_uuid'] == 'dep-uuid'
        assert result['surviving_name'] == 'SurvivingName'
        assert result['deprecated_name'] == 'DeprecatedName'
        assert 'edges_redirected' in result
        assert isinstance(result['edges_redirected'], dict)
        assert 'surviving_summary' in result

    @pytest.mark.asyncio
    async def test_merge_with_zero_edges_succeeds(self, mock_config, make_backend):
        """When deprecated node has zero edges, merge still succeeds."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(side_effect=[
            ('DepName', ''),
            ('SurName', 'existing summary'),
        ])
        backend.redirect_node_edges = AsyncMock(return_value={
            'outgoing_redirected': 0,
            'incoming_redirected': 0,
            'inter_node_deleted': 0,
        })
        backend.delete_entity_node = AsyncMock()
        backend.refresh_entity_summary = AsyncMock(return_value={
            'uuid': 'sur-uuid', 'name': 'SurName',
            'old_summary': 'existing summary', 'new_summary': 'existing summary', 'edge_count': 1,
        })
        result = await backend.merge_entities('dep-uuid', 'sur-uuid', group_id='test')
        backend.delete_entity_node.assert_awaited_once_with('dep-uuid', group_id='test')
        backend.refresh_entity_summary.assert_awaited_once_with('sur-uuid', group_id='test')
        assert result['surviving_uuid'] == 'sur-uuid'


# ---------------------------------------------------------------------------
# step-7: MemoryService.merge_entities
# ---------------------------------------------------------------------------

class TestMemoryServiceMergeEntities:
    """MemoryService.merge_entities() delegates to graphiti backend with journaling."""

    @pytest.fixture
    def service(self, mock_config):
        """MemoryService with mocked graphiti backend."""
        from fused_memory.services.memory_service import MemoryService
        svc = MemoryService(mock_config)
        svc.graphiti = MagicMock()
        svc.graphiti.merge_entities = AsyncMock(return_value={
            'surviving_uuid': 'sur-uuid',
            'surviving_name': 'SurvivingName',
            'deprecated_uuid': 'dep-uuid',
            'deprecated_name': 'DeprecatedName',
            'edges_redirected': {'outgoing_redirected': 2, 'incoming_redirected': 1, 'inter_node_deleted': 0},
            'surviving_summary': {'before': 'old', 'after': 'new', 'edge_count': 3},
        })
        svc.mem0 = MagicMock()
        svc.durable_queue = MagicMock()
        svc.durable_queue.enqueue = AsyncMock(return_value=1)
        return svc

    @pytest.mark.asyncio
    async def test_delegates_to_graphiti_backend(self, service):
        """Calls graphiti.merge_entities with correct UUIDs."""
        result = await service.merge_entities(
            deprecated_uuid='dep-uuid',
            surviving_uuid='sur-uuid',
            project_id='dark_factory',
        )
        service.graphiti.merge_entities.assert_awaited_once_with('dep-uuid', 'sur-uuid', group_id='dark_factory')
        assert result['surviving_uuid'] == 'sur-uuid'

    @pytest.mark.asyncio
    async def test_logs_via_write_journal_when_present(self, service):
        """Logs write op via write journal when journal is set."""
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        service.set_write_journal(mock_journal)
        await service.merge_entities(
            deprecated_uuid='dep-uuid',
            surviving_uuid='sur-uuid',
            project_id='dark_factory',
            agent_id='test-agent',
        )
        mock_journal.log_write_op.assert_awaited_once()
        call_kwargs = mock_journal.log_write_op.call_args[1]
        assert call_kwargs.get('operation') == 'merge_entities'
        assert call_kwargs.get('project_id') == 'dark_factory'

    @pytest.mark.asyncio
    async def test_journal_params_contain_both_uuids(self, service):
        """Journal params dict contains both deprecated_uuid and surviving_uuid."""
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        service.set_write_journal(mock_journal)
        await service.merge_entities(
            deprecated_uuid='dep-uuid',
            surviving_uuid='sur-uuid',
            project_id='dark_factory',
        )
        call_kwargs = mock_journal.log_write_op.call_args[1]
        params = call_kwargs.get('params', {})
        assert params.get('deprecated_uuid') == 'dep-uuid'
        assert params.get('surviving_uuid') == 'sur-uuid'

    @pytest.mark.asyncio
    async def test_journal_failure_does_not_mask_success(self, service):
        """Journal failure does not prevent returning successful result."""
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock(side_effect=RuntimeError('journal down'))
        service.set_write_journal(mock_journal)
        result = await service.merge_entities(
            deprecated_uuid='dep-uuid',
            surviving_uuid='sur-uuid',
            project_id='dark_factory',
        )
        assert result['surviving_uuid'] == 'sur-uuid'

    @pytest.mark.asyncio
    async def test_backend_failure_is_journaled(self, service):
        """When backend raises, journal is still called with success=False."""
        service.graphiti.merge_entities = AsyncMock(
            side_effect=ValueError('node not found')
        )
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        service.set_write_journal(mock_journal)
        with pytest.raises(ValueError, match='node not found'):
            await service.merge_entities(
                deprecated_uuid='dep-uuid',
                surviving_uuid='sur-uuid',
                project_id='dark_factory',
            )
        mock_journal.log_write_op.assert_awaited_once()
        call_kwargs = mock_journal.log_write_op.call_args[1]
        assert call_kwargs.get('success') is False

    @pytest.mark.asyncio
    async def test_works_without_journal(self, service):
        """Works correctly when no write journal is set."""
        service._write_journal = None
        result = await service.merge_entities(
            deprecated_uuid='dep-uuid',
            surviving_uuid='sur-uuid',
            project_id='dark_factory',
        )
        assert result['surviving_uuid'] == 'sur-uuid'


# ---------------------------------------------------------------------------
# step-9: MCP tool merge_entities
# ---------------------------------------------------------------------------

class TestMergeEntitiesMcpTool:
    """MCP tool merge_entities is registered and delegates correctly."""

    @pytest.fixture
    def mock_service(self):
        """Mock MemoryService for tool registration."""
        svc = AsyncMock()
        svc.merge_entities = AsyncMock(return_value={
            'surviving_uuid': 'sur-uuid',
            'surviving_name': 'SurvivingName',
            'deprecated_uuid': 'dep-uuid',
            'deprecated_name': 'DeprecatedName',
            'edges_redirected': {'outgoing_redirected': 2, 'incoming_redirected': 1, 'inter_node_deleted': 0},
            'surviving_summary': {'before': 'old', 'after': 'new', 'edge_count': 3},
        })
        return svc

    @pytest.fixture
    def mcp_server(self, mock_service):
        """MCP server with mock memory service."""
        from fused_memory.server.tools import create_mcp_server
        return create_mcp_server(mock_service)

    @pytest.mark.asyncio
    async def test_tool_is_registered(self, mcp_server):
        """merge_entities is registered as an MCP tool."""
        tool_names = [t.name for t in await mcp_server.list_tools()]
        assert 'merge_entities' in tool_names

    @pytest.mark.asyncio
    async def test_delegates_to_memory_service(self, mcp_server, mock_service):
        """Tool calls memory_service.merge_entities with correct UUIDs."""
        await mcp_server._tool_manager.call_tool(
            'merge_entities',
            {
                'deprecated_uuid': 'dep-uuid',
                'surviving_uuid': 'sur-uuid',
                'project_id': 'dark_factory',
            },
        )
        mock_service.merge_entities.assert_awaited_once()
        call_kwargs = mock_service.merge_entities.call_args[1]
        assert call_kwargs.get('deprecated_uuid') == 'dep-uuid'
        assert call_kwargs.get('surviving_uuid') == 'sur-uuid'
        assert call_kwargs.get('project_id') == 'dark_factory'

    @pytest.mark.asyncio
    async def test_empty_project_id_returns_error(self, mcp_server, mock_service):
        """Empty project_id returns validation error dict."""
        import json
        result = await mcp_server._tool_manager.call_tool(
            'merge_entities',
            {'deprecated_uuid': 'dep-uuid', 'surviving_uuid': 'sur-uuid', 'project_id': ''},
        )
        if isinstance(result, list):
            content = result[0].text if hasattr(result[0], 'text') else str(result[0])
            parsed = json.loads(content)
        else:
            parsed = result
        assert 'error' in parsed
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.merge_entities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_deprecated_uuid_returns_error(self, mcp_server, mock_service):
        """Empty deprecated_uuid returns validation error dict."""
        import json
        result = await mcp_server._tool_manager.call_tool(
            'merge_entities',
            {'deprecated_uuid': '', 'surviving_uuid': 'sur-uuid', 'project_id': 'dark_factory'},
        )
        if isinstance(result, list):
            content = result[0].text if hasattr(result[0], 'text') else str(result[0])
            parsed = json.loads(content)
        else:
            parsed = result
        assert 'error' in parsed
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.merge_entities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_surviving_uuid_returns_error(self, mcp_server, mock_service):
        """Empty surviving_uuid returns validation error dict."""
        import json
        result = await mcp_server._tool_manager.call_tool(
            'merge_entities',
            {'deprecated_uuid': 'dep-uuid', 'surviving_uuid': '', 'project_id': 'dark_factory'},
        )
        if isinstance(result, list):
            content = result[0].text if hasattr(result[0], 'text') else str(result[0])
            parsed = json.loads(content)
        else:
            parsed = result
        assert 'error' in parsed
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.merge_entities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_same_uuid_returns_error(self, mcp_server, mock_service):
        """Same UUID for both params returns validation error."""
        import json
        result = await mcp_server._tool_manager.call_tool(
            'merge_entities',
            {'deprecated_uuid': 'same-uuid', 'surviving_uuid': 'same-uuid', 'project_id': 'dark_factory'},
        )
        if isinstance(result, list):
            content = result[0].text if hasattr(result[0], 'text') else str(result[0])
            parsed = json.loads(content)
        else:
            parsed = result
        assert 'error' in parsed
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.merge_entities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_exception_returns_error_dict(self, mcp_server, mock_service):
        """Exception from memory_service returns error dict (not raised)."""
        import json
        mock_service.merge_entities = AsyncMock(
            side_effect=RuntimeError('FalkorDB connection failed')
        )
        result = await mcp_server._tool_manager.call_tool(
            'merge_entities',
            {'deprecated_uuid': 'dep-uuid', 'surviving_uuid': 'sur-uuid', 'project_id': 'dark_factory'},
        )
        if isinstance(result, list):
            content = result[0].text if hasattr(result[0], 'text') else str(result[0])
            parsed = json.loads(content)
        else:
            parsed = result
        assert 'error' in parsed
        assert 'FalkorDB connection failed' in parsed['error']


# ---------------------------------------------------------------------------
# step-11: DISALLOW_MEMORY_WRITES list in cli_stage_runner.py
# ---------------------------------------------------------------------------

class TestDisallowListForMergeEntities:
    """merge_entities must be in DISALLOW_MEMORY_WRITES but NOT in STAGE1_DISALLOWED."""

    def test_merge_entities_in_disallow_memory_writes(self):
        """'mcp__fused-memory__merge_entities' must be in DISALLOW_MEMORY_WRITES
        so Stage 3 (read-only) cannot call it."""
        from fused_memory.reconciliation.cli_stage_runner import DISALLOW_MEMORY_WRITES
        assert 'mcp__fused-memory__merge_entities' in DISALLOW_MEMORY_WRITES

    def test_merge_entities_not_in_stage1_disallowed(self):
        """Stage 1 must be able to call merge_entities (not in STAGE1_DISALLOWED)."""
        from fused_memory.reconciliation.cli_stage_runner import STAGE1_DISALLOWED
        assert 'mcp__fused-memory__merge_entities' not in STAGE1_DISALLOWED

    def test_merge_entities_in_stage3_disallowed(self):
        """Stage 3 must NOT be able to call merge_entities (in STAGE3_DISALLOWED
        via DISALLOW_MEMORY_WRITES)."""
        from fused_memory.reconciliation.cli_stage_runner import STAGE3_DISALLOWED
        assert 'mcp__fused-memory__merge_entities' in STAGE3_DISALLOWED
