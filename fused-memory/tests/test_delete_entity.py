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
        assert result['refresh_errors'] == []

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
        assert result['refresh_errors'] == []

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

    @pytest.mark.asyncio
    async def test_partial_refresh_failure_best_effort(self, mock_config, make_backend):
        """PARTIAL FAILURE: node deleted, one neighbour refresh raises.

        The exception must NOT propagate (node is already gone — irreversible).
        Remaining neighbours must still be refreshed.
        Audit dict must split outcome into connected_refreshed and refresh_errors.
        """
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(return_value=('NodeName', ''))
        backend.get_valid_edges_for_node = AsyncMock(return_value=[])
        backend.get_connected_entity_uuids = AsyncMock(return_value=['nbr-ok', 'nbr-fail', 'nbr-ok2'])
        backend.delete_entity_node = AsyncMock()

        # First and third succeed; second raises
        call_count = [0]
        async def selective_refresh(nbr_uuid, *, group_id):
            call_count[0] += 1
            if nbr_uuid == 'nbr-fail':
                raise RuntimeError('transient graph write error')
            return {'uuid': nbr_uuid, 'name': 'N', 'old_summary': '', 'new_summary': '', 'edge_count': 0}
        backend.refresh_entity_summary = selective_refresh

        result = await backend.delete_entity('target-uuid', group_id='test')

        # The delete itself must succeed (not raise)
        backend.delete_entity_node.assert_awaited_once_with('target-uuid', group_id='test')

        # All three refresh attempts were made (best-effort, not short-circuited)
        assert call_count[0] == 3

        # Audit dict reflects partial outcome
        assert result['connected_refreshed'] == ['nbr-ok', 'nbr-ok2']
        assert result['refresh_errors'] == ['nbr-fail']

        # Core delete fields still correct
        assert result['deleted_uuid'] == 'target-uuid'
        assert result['deleted_name'] == 'NodeName'


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


# ---------------------------------------------------------------------------
# step-25: GraphitiBackend.count_foreign_relationships
# ---------------------------------------------------------------------------

class TestCountForeignRelationships:
    """``count_foreign_relationships(node_uuid, *, group_id, episode_uuid='') -> int``.

    THE QUESTION THIS PRIMITIVE ANSWERS, and why no existing one could.
    ``get_valid_edges_for_node`` answers "does this node still carry live
    facts?" — it is typed to ``RELATES_TO`` and filtered to
    ``invalid_at IS NULL``.  Every other count query in ``graphiti_client.py``
    is likewise typed to ``RELATES_TO``.  So a node's INVALIDATED temporal
    history and its Episodic ``MENTIONS`` provenance links are invisible to
    every primitive this backend has — while ``delete_entity_node`` issues a
    bare ``MATCH (n:Entity {uuid: $uuid}) DETACH DELETE n`` that destroys all
    of them.

    A guard that authorises a destructive operation must test EXACTLY what
    that operation destroys.  This one therefore counts every relationship of
    any type and any validity, minus the single link the caller can prove it
    minted itself: this episode's own ``MENTIONS``.
    """

    @pytest.mark.asyncio
    async def test_uses_ro_query_not_query(self, mock_config, make_backend, make_graph_mock):
        """READ-ONLY, and must stay so: this primitive is a guard.  A guard
        that can write is a guard that can cause the damage it exists to
        prevent."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[[0]])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.count_foreign_relationships('n-3129', group_id='test')
        graph.ro_query.assert_awaited_once()
        graph.query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_match_is_untyped_and_undirected(self, mock_config, make_backend, make_graph_mock):
        """``MATCH (n:Entity {uuid: $uuid})-[r]-(m)`` — no relationship type,
        no direction.  DETACH DELETE respects neither, so neither may this."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[[0]])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.count_foreign_relationships('n-3129', group_id='test')
        cypher = extract_cypher(graph.ro_query.call_args)
        assert '(n:Entity {uuid: $uuid})' in cypher
        assert '-[r]-' in cypher
        assert '->' not in cypher and '<-' not in cypher, (
            'a directed match would miss half the relationships DETACH DELETE '
            'destroys'
        )

    @pytest.mark.asyncio
    async def test_cypher_names_neither_RELATES_TO_nor_invalid_at(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """THESE TWO OMISSIONS ARE THE POINT OF THE PRIMITIVE.

        Re-adding either would silently reintroduce the deletion bug: typing
        the match to ``RELATES_TO`` re-blinds it to ``MENTIONS``, and filtering
        ``invalid_at IS NULL`` re-blinds it to the temporal history DETACH
        DELETE destroys.  Both edits look like harmless tidying next to the
        sibling queries in the same file, which is exactly why this asserts on
        their ABSENCE.
        """
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[[0]])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.count_foreign_relationships('n-3129', group_id='test')
        cypher = extract_cypher(graph.ro_query.call_args)
        assert 'RELATES_TO' not in cypher
        assert 'invalid_at' not in cypher

    @pytest.mark.asyncio
    async def test_passes_the_node_uuid_as_a_param(self, mock_config, make_backend, make_graph_mock):
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[[0]])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.count_foreign_relationships('n-3129', group_id='test')
        assert extract_params(graph.ro_query.call_args).get('uuid') == 'n-3129'

    # -- episode_uuid='' : the fail-closed default counts EVERYTHING ---------

    @pytest.mark.asyncio
    async def test_default_applies_no_exclusion(self, mock_config, make_backend, make_graph_mock):
        """``episode_uuid=''`` means "we could not establish which episode this
        is", so nothing is excluded and the predicate is strictly
        conservative."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[[0]])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.count_foreign_relationships('n-3129', group_id='test')
        cypher = extract_cypher(graph.ro_query.call_args)
        assert 'MENTIONS' not in cypher
        assert 'WHERE' not in cypher

    @pytest.mark.asyncio
    async def test_default_counts_an_invalidated_relates_to(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """A node whose every fact was superseded or TTL-invalidated still has
        that history in the graph, and DETACH DELETE would destroy it.
        ``get_valid_edges_for_node`` reports it as empty; this reports 1."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[[1]])
        backend._driver._get_graph = MagicMock(return_value=graph)
        assert await backend.count_foreign_relationships('n-3129', group_id='test') == 1

    @pytest.mark.asyncio
    async def test_default_counts_an_episodic_mentions(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """``(ep:Episodic)-[:MENTIONS]->(n:Entity)`` links are real,
        load-bearing provenance (``maintenance/cross_graph_move.py`` recreates
        them precisely because losing them loses provenance) and are invisible
        to every RELATES_TO-typed query in the backend."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[[1]])
        backend._driver._get_graph = MagicMock(return_value=graph)
        assert await backend.count_foreign_relationships('n-3129', group_id='test') == 1

    @pytest.mark.asyncio
    async def test_a_node_with_no_relationships_at_all_returns_zero(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """The pattern REQUIRES a relationship, so a node with none produces no
        rows — the empty result_set IS the zero path, not merely a defensive
        one."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[])
        backend._driver._get_graph = MagicMock(return_value=graph)
        assert await backend.count_foreign_relationships('n-3129', group_id='test') == 0

    @pytest.mark.asyncio
    async def test_an_absent_result_set_returns_zero(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """``result_set = None`` degrades the same way as an empty one."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[])
        result = MagicMock()
        result.result_set = None
        graph.ro_query = AsyncMock(return_value=result)
        backend._driver._get_graph = MagicMock(return_value=graph)
        assert await backend.count_foreign_relationships('n-3129', group_id='test') == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize('malformed', [[[]], [[None]], [['not-a-number']]])
    async def test_a_present_but_unreadable_row_raises_rather_than_reading_zero(
        self, mock_config, make_backend, make_graph_mock, malformed,
    ):
        """THE DEFENSIVE ZERO MUST NOT MASK A MALFORMED RESPONSE.

        ``count(r)`` has no grouping key, so against a real server it always
        yields exactly one readable row.  A row that is PRESENT but unreadable
        is a broken response, and quietly reading it as ``0`` would AUTHORISE a
        deletion on the strength of a response nobody understood.  Raising is
        the safe direction: the single caller wraps this in a per-candidate
        try/except, so a raise becomes a logged skip — a refusal to delete.
        """
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[])
        result = MagicMock()
        result.result_set = malformed
        graph.ro_query = AsyncMock(return_value=result)
        backend._driver._get_graph = MagicMock(return_value=graph)
        with pytest.raises((IndexError, TypeError, ValueError)):
            await backend.count_foreign_relationships('n-3129', group_id='test')

    # -- episode_uuid='ep-1' : exclude only THIS episode's own MENTIONS ------

    @pytest.mark.asyncio
    async def test_a_non_empty_episode_uuid_excludes_that_episodes_mentions(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """The exclusion is NARROW by construction: it names the relationship
        TYPE and the specific episode node, so any invalidated RELATES_TO, and
        any MENTIONS from a DIFFERENT episode, still count."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[[0]])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.count_foreign_relationships(
            'n-3129', group_id='test', episode_uuid='ep-1',
        )
        cypher = extract_cypher(graph.ro_query.call_args)
        assert 'WHERE' in cypher
        assert "type(r) = 'MENTIONS'" in cypher
        assert 'm.uuid = $episode_uuid' in cypher
        assert 'NOT (' in cypher
        assert extract_params(graph.ro_query.call_args).get('episode_uuid') == 'ep-1'

    @pytest.mark.asyncio
    async def test_a_node_whose_only_link_is_this_episodes_mentions_returns_zero(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """THE EXCLUSION IS LIVE, NOT DEAD CODE.  graphiti_core mints the
        mis-resolved node AND its MENTIONS link from the same episodic node in
        the same ``add_episode``, so a strict zero-degree predicate could never
        fire on the very phantom the cleanup exists to remove."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[[0]])
        backend._driver._get_graph = MagicMock(return_value=graph)
        count = await backend.count_foreign_relationships(
            'n-3129', group_id='test', episode_uuid='ep-1',
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_a_mentions_from_a_different_episode_still_counts(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """Any MENTIONS from an episode OTHER than the one in flight proves
        pre-existing provenance and must refuse the delete."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[[1]])
        backend._driver._get_graph = MagicMock(return_value=graph)
        count = await backend.count_foreign_relationships(
            'n-3129', group_id='test', episode_uuid='ep-1',
        )
        assert count == 1

    # -- the deliberate self-loop double-count ------------------------------

    @pytest.mark.asyncio
    async def test_a_self_loop_double_counts_and_is_deliberately_not_deduped(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """A single A->A relationship matches the undirected pattern twice and
        returns 2, NOT 1 — the OPPOSITE choice from
        ``get_valid_edges_for_node``, which dedupes on ``e.uuid`` for exactly
        this reason.

        The asymmetry is correct because the two values are consumed
        differently: that one's list is enumerated, while this one is consumed
        only as ``== 0``.  Inflation can therefore only ever REFUSE a delete,
        which is the safe direction.  A test pinning 1 here would be pinning a
        dedup this guard must not have.
        """
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[[2]])
        backend._driver._get_graph = MagicMock(return_value=graph)
        assert await backend.count_foreign_relationships('n-3129', group_id='test') == 2

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        backend = GraphitiBackend(mock_config)
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.count_foreign_relationships('n-3129', group_id='test')

    @pytest.mark.asyncio
    async def test_positional_misuse_is_rejected(self, mock_config, make_backend):
        """Keyword-only-ness pinned EXECUTABLY, not by reading the signature
        object: a caller that tries to pass the identity positionally is
        refused outright rather than silently binding it to ``group_id``."""
        backend = make_backend(mock_config)
        with pytest.raises(TypeError):
            await backend.count_foreign_relationships('n-3129', 'test', 'ep-1')
