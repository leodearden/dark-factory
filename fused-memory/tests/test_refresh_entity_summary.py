"""Tests for refresh_entity_summary across backends, service, and MCP tool.

Covers:
- GraphitiBackend.get_valid_edges_for_node()
- GraphitiBackend.update_node_summary()
- GraphitiBackend.refresh_entity_summary()
- MemoryService.refresh_entity_summary()
- MCP tool refresh_entity_summary in tools.py
- DISALLOW_MEMORY_WRITES list in cli_stage_runner.py
- STAGE1_SYSTEM_PROMPT in prompts/stage1.py
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from conftest import assert_ro_query_only, extract_cypher, extract_params

from fused_memory.backends.graphiti_client import (
    AmbiguousEntityError,
    GraphitiBackend,
    NodeNotFoundError,
)
from fused_memory.server.tools import create_mcp_server

# ---------------------------------------------------------------------------
# step-1 (task-309): GraphitiBackend.resolve_entity_by_name
# ---------------------------------------------------------------------------

class TestResolveEntityByName:
    """GraphitiBackend.resolve_entity_by_name(name, group_id) resolves name→UUID."""

    @pytest.mark.asyncio
    async def test_returns_uuid_on_exact_name_match(self, mock_config, make_backend, make_graph_mock):
        """Returns the single UUID when exactly one entity matches the name."""
        backend = make_backend(mock_config)
        rows = [['uuid-alice', 'Alice']]
        graph = make_graph_mock(rows)
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.resolve_entity_by_name('Alice', group_id='test')
        assert result == 'uuid-alice'

    @pytest.mark.asyncio
    async def test_raises_node_not_found_when_no_match(self, mock_config, make_backend, make_graph_mock):
        """Raises NodeNotFoundError when no entity has the given name."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        with pytest.raises(NodeNotFoundError):
            await backend.resolve_entity_by_name('NonExistent', group_id='test')

    @pytest.mark.asyncio
    async def test_raises_ambiguous_error_when_multiple_match(
        self, mock_config, make_backend, make_graph_mock
    ):
        """Raises AmbiguousEntityError when multiple entities share the same name."""
        backend = make_backend(mock_config)
        rows = [['uuid-1', 'Alice'], ['uuid-2', 'Alice']]
        graph = make_graph_mock(rows)
        backend._driver._get_graph = MagicMock(return_value=graph)
        with pytest.raises(AmbiguousEntityError) as exc_info:
            await backend.resolve_entity_by_name('Alice', group_id='test')
        # Error message should include both UUIDs so callers can disambiguate
        assert 'uuid-1' in str(exc_info.value)
        assert 'uuid-2' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        """Raises RuntimeError when backend client is not initialized."""
        backend = GraphitiBackend(mock_config)  # _driver is None
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.resolve_entity_by_name('Alice', group_id='test')

    @pytest.mark.asyncio
    async def test_passes_name_as_cypher_parameter(self, mock_config, make_backend, make_graph_mock):
        """Passes the name as a Cypher parameter (not interpolated into query string)."""
        backend = make_backend(mock_config)
        rows = [['uuid-alice', 'Alice']]
        graph = make_graph_mock(rows)
        backend._driver._get_graph = MagicMock(return_value=graph)
        entity_name = 'Alice'
        await backend.resolve_entity_by_name(entity_name, group_id='test')
        call_args = graph.ro_query.call_args
        assert call_args is not None
        cypher_params = extract_params(call_args)
        assert cypher_params.get('name') == entity_name
        graph.query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_uses_ro_query_not_query(self, mock_config, make_backend, make_graph_mock):
        """resolve_entity_by_name uses ro_query (read-only path) and never calls graph.query."""
        backend = make_backend(mock_config)
        await assert_ro_query_only(backend, make_graph_mock, [['uuid-alice', 'Alice']], 'resolve_entity_by_name', 'Alice', group_id='test')


# ---------------------------------------------------------------------------
# step-1 (original): GraphitiBackend.get_valid_edges_for_node
# ---------------------------------------------------------------------------

class TestGetValidEdgesForNode:
    """GraphitiBackend.get_valid_edges_for_node(node_uuid) returns valid edges."""

    @pytest.mark.asyncio
    async def test_returns_valid_edges(self, mock_config, make_backend, make_graph_mock):
        """Returns list of dicts with uuid/fact/name keys for matching edges."""
        backend = make_backend(mock_config)
        rows = [
            ['edge-1', 'Alice knows Bob', 'knows'],
            ['edge-2', 'Alice works at Acme', 'works_at'],
        ]
        graph = make_graph_mock(rows)
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.get_valid_edges_for_node('node-uuid-1', group_id='test')
        assert len(result) == 2
        assert result[0]['uuid'] == 'edge-1'
        assert result[0]['fact'] == 'Alice knows Bob'
        assert result[0]['name'] == 'knows'
        assert result[1]['uuid'] == 'edge-2'
        assert result[1]['fact'] == 'Alice works at Acme'
        assert result[1]['name'] == 'works_at'

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_valid_edges(self, mock_config, make_backend, make_graph_mock):
        """Returns empty list when no valid edges remain."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.get_valid_edges_for_node('node-uuid-1', group_id='test')
        assert result == []

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        """Raises RuntimeError when client is not initialized."""
        backend = GraphitiBackend(mock_config)  # client is None
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.get_valid_edges_for_node('node-uuid-1', group_id='test')

    @pytest.mark.asyncio
    async def test_null_fact_defaults_to_empty_string(self, mock_config, make_backend, make_graph_mock):
        """Rows with NULL fact field (row[1] is None) yield fact='' in result dict."""
        backend = make_backend(mock_config)
        rows = [['edge-1', None, 'knows']]
        graph = make_graph_mock(rows)
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.get_valid_edges_for_node('node-uuid-1', group_id='test')
        assert len(result) == 1
        assert result[0]['fact'] == ''

    @pytest.mark.asyncio
    async def test_null_name_defaults_to_empty_string(self, mock_config, make_backend, make_graph_mock):
        """Rows with NULL name field (row[2] is None) yield name='' in result dict."""
        backend = make_backend(mock_config)
        rows = [['edge-1', 'Alice knows Bob', None]]
        graph = make_graph_mock(rows)
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.get_valid_edges_for_node('node-uuid-1', group_id='test')
        assert len(result) == 1
        assert result[0]['name'] == ''

    @pytest.mark.asyncio
    async def test_passes_uuid_to_query(self, mock_config, make_backend, make_graph_mock):
        """Passes node UUID as parameter to the Cypher query."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        node_uuid = 'my-node-uuid'
        await backend.get_valid_edges_for_node(node_uuid, group_id='test')
        call_args = graph.ro_query.call_args
        assert call_args is not None, "graph.ro_query was not called"
        cypher_params = extract_params(call_args)
        assert cypher_params.get('uuid') == node_uuid

    @pytest.mark.asyncio
    async def test_filters_invalid_at_null(self, mock_config, make_backend, make_graph_mock):
        """Query uses WHERE e.invalid_at IS NULL to filter active edges only."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.get_valid_edges_for_node('node-uuid-1', group_id='test')
        call_args = graph.ro_query.call_args
        assert call_args is not None, "graph.ro_query was not called"
        cypher = extract_cypher(call_args)
        assert 'invalid_at IS NULL' in cypher, f"Cypher must filter by invalid_at IS NULL: {cypher}"

    @pytest.mark.asyncio
    async def test_uses_ro_query_not_query(self, mock_config, make_backend, make_graph_mock):
        """Uses ro_query (read-only) — graph.query must NOT be called."""
        backend = make_backend(mock_config)
        await assert_ro_query_only(backend, make_graph_mock, [], 'get_valid_edges_for_node', 'node-uuid-1', group_id='test')


# ---------------------------------------------------------------------------
# GraphitiBackend.get_all_valid_edges (bulk fetch)
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
    async def test_cypher_uses_return_distinct(self, mock_config, make_backend, make_graph_mock):
        """Cypher query passed to ro_query contains 'RETURN DISTINCT'."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.get_all_valid_edges(group_id='test')
        cypher = graph.ro_query.call_args.args[0]
        assert 'RETURN DISTINCT' in cypher

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
    async def test_null_fact_defaults_to_empty_string(self, mock_config, make_backend, make_graph_mock):
        """Row with None fact returns fact=''."""
        backend = make_backend(mock_config)
        rows = [['node-1', 'e1', None, 'edge1']]
        graph = make_graph_mock(rows)
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.get_all_valid_edges(group_id='test')
        assert result['node-1'][0]['fact'] == ''

    @pytest.mark.asyncio
    async def test_null_name_defaults_to_empty_string(self, mock_config, make_backend, make_graph_mock):
        """Row with None name returns name=''."""
        backend = make_backend(mock_config)
        rows = [['node-1', 'e1', 'factA', None]]
        graph = make_graph_mock(rows)
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.get_all_valid_edges(group_id='test')
        assert result['node-1'][0]['name'] == ''

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
# GraphitiBackend._edge_dict: edge dict normalisation helper
# ---------------------------------------------------------------------------

class TestEdgeDict:
    """GraphitiBackend._edge_dict(uuid, fact, name) returns a normalised edge dict."""

    def test_returns_dict_with_correct_keys(self):
        """Returns dict with keys uuid, fact, name for normal (non-None) values."""
        result = GraphitiBackend._edge_dict('e-1', 'Alice knows Bob', 'knows')
        assert result == {'uuid': 'e-1', 'fact': 'Alice knows Bob', 'name': 'knows'}

    def test_none_fact_coerced_to_empty_string(self):
        """None fact is coerced to '' in the returned dict."""
        result = GraphitiBackend._edge_dict('e-1', None, 'knows')
        assert result == {'uuid': 'e-1', 'fact': '', 'name': 'knows'}

    def test_none_name_coerced_to_empty_string(self):
        """None name is coerced to '' in the returned dict."""
        result = GraphitiBackend._edge_dict('e-1', 'Alice knows Bob', None)
        assert result == {'uuid': 'e-1', 'fact': 'Alice knows Bob', 'name': ''}

    def test_both_none_coerced_to_empty_strings(self):
        """Both fact and name None are coerced to empty strings."""
        result = GraphitiBackend._edge_dict('e-null', None, None)
        assert result == {'uuid': 'e-null', 'fact': '', 'name': ''}


# ---------------------------------------------------------------------------
# step-3: GraphitiBackend.update_node_summary
# ---------------------------------------------------------------------------

class TestUpdateNodeSummary:
    """GraphitiBackend.update_node_summary(uuid, summary) sets summary on Entity node."""

    @pytest.mark.asyncio
    async def test_sets_summary_on_node(self, mock_config, make_backend, make_graph_mock):
        """Issues Cypher SET n.summary = $summary for the given Entity node UUID."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.update_node_summary('node-uuid-1', 'Alice knows Bob.\nAlice works at Acme.', group_id='test')
        graph.query.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_passes_uuid_and_summary_as_params(self, mock_config, make_backend, make_graph_mock):
        """Query receives uuid and summary as Cypher parameters."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        node_uuid = 'my-node-uuid'
        summary = 'Alice knows Bob.'
        await backend.update_node_summary(node_uuid, summary, group_id='test')
        call_args = graph.query.call_args
        assert call_args is not None, "graph.query was not called"
        cypher_params = extract_params(call_args)
        assert cypher_params.get('uuid') == node_uuid
        assert cypher_params.get('summary') == summary

    @pytest.mark.asyncio
    async def test_cypher_sets_summary_property(self, mock_config, make_backend, make_graph_mock):
        """Cypher query contains SET n.summary = $summary."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.update_node_summary('node-uuid-1', 'some summary', group_id='test')
        call_args = graph.query.call_args
        assert call_args is not None, "graph.query was not called"
        cypher = extract_cypher(call_args)
        assert 'SET n.summary' in cypher, f"Cypher must SET n.summary: {cypher}"

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        """Raises RuntimeError when client is not initialized."""
        backend = GraphitiBackend(mock_config)
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.update_node_summary('node-uuid-1', 'summary text', group_id='test')


# ---------------------------------------------------------------------------
# step-5: GraphitiBackend.refresh_entity_summary
# ---------------------------------------------------------------------------

class TestRefreshEntitySummary:
    """GraphitiBackend.refresh_entity_summary(node_uuid) regenerates entity summary."""

    @pytest.mark.asyncio
    async def test_returns_result_dict(self, mock_config, make_backend):
        """Returns dict with uuid, name, old_summary, new_summary, edge_count keys."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(return_value=('Alice', 'old summary'))
        backend.get_valid_edges_for_node = AsyncMock(return_value=[
            {'uuid': 'e1', 'fact': 'Alice knows Bob', 'name': 'knows'},
        ])
        backend.update_node_summary = AsyncMock()
        result = await backend.refresh_entity_summary('node-uuid-1', group_id='test')
        assert result['uuid'] == 'node-uuid-1'
        assert result['name'] == 'Alice'
        assert result['old_summary'] == 'old summary'
        assert result['edge_count'] == 1
        assert isinstance(result['new_summary'], str)

    @pytest.mark.asyncio
    async def test_deduplicates_facts(self, mock_config, make_backend):
        """Duplicate facts appear only once in the new summary."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(return_value=('Alice', ''))
        # Two edges with the same fact
        backend.get_valid_edges_for_node = AsyncMock(return_value=[
            {'uuid': 'e1', 'fact': 'Alice knows Bob', 'name': 'knows'},
            {'uuid': 'e2', 'fact': 'Alice knows Bob', 'name': 'knows'},
            {'uuid': 'e3', 'fact': 'Alice works at Acme', 'name': 'works_at'},
        ])
        backend.update_node_summary = AsyncMock()
        result = await backend.refresh_entity_summary('node-uuid-1', group_id='test')
        # 'Alice knows Bob' should appear exactly once
        assert result['new_summary'].count('Alice knows Bob') == 1
        assert 'Alice works at Acme' in result['new_summary']

    @pytest.mark.asyncio
    async def test_calls_update_node_summary_with_new_summary(self, mock_config, make_backend):
        """Calls update_node_summary with the new deduped summary."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(return_value=('Alice', 'old'))
        backend.get_valid_edges_for_node = AsyncMock(return_value=[
            {'uuid': 'e1', 'fact': 'Alice knows Bob', 'name': 'knows'},
        ])
        backend.update_node_summary = AsyncMock()
        result = await backend.refresh_entity_summary('node-uuid-1', group_id='test')
        backend.update_node_summary.assert_awaited_once_with('node-uuid-1', result['new_summary'], group_id='test')

    @pytest.mark.asyncio
    async def test_empty_edges_produces_empty_summary(self, mock_config, make_backend):
        """When zero valid edges remain, new_summary is empty string and update is called."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(return_value=('Alice', 'old summary'))
        backend.get_valid_edges_for_node = AsyncMock(return_value=[])
        backend.update_node_summary = AsyncMock()
        result = await backend.refresh_entity_summary('node-uuid-1', group_id='test')
        assert result['new_summary'] == ''
        assert result['edge_count'] == 0
        backend.update_node_summary.assert_awaited_once_with('node-uuid-1', '', group_id='test')

    @pytest.mark.asyncio
    async def test_old_summary_returned_in_result(self, mock_config, make_backend):
        """The old summary from get_node_text is preserved in result for auditing."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(return_value=('Bob', 'prior summary text'))
        backend.get_valid_edges_for_node = AsyncMock(return_value=[
            {'uuid': 'e1', 'fact': 'Bob lives in London', 'name': 'lives_in'},
        ])
        backend.update_node_summary = AsyncMock()
        result = await backend.refresh_entity_summary('node-uuid-2', group_id='test')
        assert result['old_summary'] == 'prior summary text'

    @pytest.mark.asyncio
    async def test_refresh_entity_summary_with_empty_old_summary_and_name(
        self, mock_config, make_backend
    ):
        """Empty-string old_summary is a valid production value (list_entity_nodes normalizes
        NULL→'') and must not trigger the guard that raises ValueError.  When name and
        old_summary are both provided, get_node_text is skipped entirely."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock()
        backend.get_valid_edges_for_node = AsyncMock(return_value=[
            {'uuid': 'e1', 'fact': 'Alice knows Bob', 'name': 'knows'},
        ])
        backend.update_node_summary = AsyncMock()
        result = await backend.refresh_entity_summary(
            'node-uuid-1', group_id='test', name='Alice', old_summary=''
        )
        backend.get_node_text.assert_not_awaited()
        assert result['old_summary'] == ''
        assert result['name'] == 'Alice'
        assert result['uuid'] == 'node-uuid-1'
        assert result['new_summary'] == 'Alice knows Bob'
        assert result['edge_count'] == 1
        backend.update_node_summary.assert_awaited_once_with('node-uuid-1', 'Alice knows Bob', group_id='test')


# ---------------------------------------------------------------------------
# step-7: MemoryService.refresh_entity_summary
# ---------------------------------------------------------------------------

class TestMemoryServiceRefreshEntitySummary:
    """MemoryService.refresh_entity_summary() delegates to graphiti backend."""

    @pytest.fixture
    def service(self, mock_config):
        """MemoryService with mocked backends (no real DB needed)."""
        from fused_memory.services.memory_service import MemoryService
        svc = MemoryService(mock_config)
        svc.graphiti = MagicMock()
        svc.graphiti.refresh_entity_summary = AsyncMock(return_value={
            'uuid': 'node-1',
            'name': 'Alice',
            'old_summary': 'old',
            'new_summary': 'Alice knows Bob',
            'edge_count': 1,
        })
        svc.mem0 = MagicMock()
        svc.durable_queue = MagicMock()
        svc.durable_queue.enqueue = AsyncMock(return_value=1)
        return svc

    @pytest.mark.asyncio
    async def test_delegates_to_graphiti_backend(self, service):
        """Calls graphiti.refresh_entity_summary with entity_uuid."""
        result = await service.refresh_entity_summary(
            entity_uuid='node-1',
            project_id='dark_factory',
        )
        service.graphiti.refresh_entity_summary.assert_awaited_once_with('node-1', group_id='dark_factory')
        assert result['uuid'] == 'node-1'
        assert result['edge_count'] == 1

    @pytest.mark.asyncio
    async def test_logs_via_write_journal_when_present(self, service):
        """Logs write op via write journal when journal is set."""
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        service.set_write_journal(mock_journal)
        await service.refresh_entity_summary(
            entity_uuid='node-1',
            project_id='dark_factory',
            agent_id='test-agent',
        )
        mock_journal.log_write_op.assert_awaited_once()
        call_kwargs = mock_journal.log_write_op.call_args[1]
        assert call_kwargs.get('operation') == 'refresh_entity_summary'
        assert call_kwargs.get('project_id') == 'dark_factory'

    @pytest.mark.asyncio
    async def test_works_without_journal(self, service):
        """Works correctly when no write journal is set (journal=None)."""
        # Ensure no journal is set
        service._write_journal = None
        result = await service.refresh_entity_summary(
            entity_uuid='node-1',
            project_id='dark_factory',
        )
        assert result['uuid'] == 'node-1'

    @pytest.mark.asyncio
    async def test_returns_backend_result(self, service):
        """Returns the result dict from the backend unchanged."""
        result = await service.refresh_entity_summary(
            entity_uuid='node-1',
            project_id='test_project',
        )
        assert 'uuid' in result
        assert 'name' in result
        assert 'old_summary' in result
        assert 'new_summary' in result
        assert 'edge_count' in result

    # -- step-3 (task-309): entity_name support --

    @pytest.mark.asyncio
    async def test_resolves_entity_name_to_uuid(self, service):
        """Resolves entity_name to UUID via graphiti.resolve_entity_by_name, then refreshes."""
        service.graphiti.resolve_entity_by_name = AsyncMock(return_value='node-1')
        result = await service.refresh_entity_summary(
            entity_name='Alice',
            project_id='dark_factory',
        )
        service.graphiti.resolve_entity_by_name.assert_awaited_once_with('Alice', group_id='dark_factory')
        service.graphiti.refresh_entity_summary.assert_awaited_once_with('node-1', group_id='dark_factory')
        assert result['uuid'] == 'node-1'

    @pytest.mark.asyncio
    async def test_raises_value_error_when_neither_provided(self, service):
        """Raises ValueError when neither entity_uuid nor entity_name is provided."""
        with pytest.raises(ValueError, match='entity_uuid.*entity_name|entity_name.*entity_uuid'):
            await service.refresh_entity_summary(
                project_id='dark_factory',
            )

    @pytest.mark.asyncio
    async def test_prefers_uuid_when_both_provided(self, service):
        """Uses entity_uuid directly and skips resolve when both params are given."""
        service.graphiti.resolve_entity_by_name = AsyncMock(return_value='other-node')
        await service.refresh_entity_summary(
            entity_uuid='node-1',
            entity_name='Alice',
            project_id='dark_factory',
        )
        service.graphiti.resolve_entity_by_name.assert_not_awaited()
        service.graphiti.refresh_entity_summary.assert_awaited_once_with('node-1', group_id='dark_factory')

    @pytest.mark.asyncio
    async def test_journal_includes_entity_name_when_used(self, service):
        """Journal params include entity_name when resolution path is taken."""
        service.graphiti.resolve_entity_by_name = AsyncMock(return_value='node-1')
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        service.set_write_journal(mock_journal)
        await service.refresh_entity_summary(
            entity_name='Alice',
            project_id='dark_factory',
            agent_id='test-agent',
        )
        mock_journal.log_write_op.assert_awaited_once()
        call_kwargs = mock_journal.log_write_op.call_args[1]
        assert call_kwargs['params'].get('entity_name') == 'Alice'


# ---------------------------------------------------------------------------
# step-9: MCP tool refresh_entity_summary
# ---------------------------------------------------------------------------

class TestRefreshEntitySummaryMcpTool:
    """MCP tool refresh_entity_summary is registered and delegates correctly."""

    @pytest.fixture
    def mock_service(self):
        """Mock MemoryService for tool registration."""
        svc = AsyncMock()
        svc.refresh_entity_summary = AsyncMock(return_value={
            'uuid': 'node-1',
            'name': 'Alice',
            'old_summary': 'old text',
            'new_summary': 'Alice knows Bob',
            'edge_count': 1,
        })
        return svc

    @pytest.fixture
    def mcp_server(self, mock_service):
        """MCP server with mock memory service."""
        from fused_memory.server.tools import create_mcp_server
        return create_mcp_server(mock_service)

    @pytest.mark.asyncio
    async def test_tool_is_registered(self, mcp_server):
        """refresh_entity_summary is registered as an MCP tool."""
        tool_names = [t.name for t in await mcp_server.list_tools()]
        assert 'refresh_entity_summary' in tool_names

    @pytest.mark.asyncio
    async def test_delegates_to_memory_service(self, mcp_server, mock_service):
        """Tool calls memory_service.refresh_entity_summary with correct entity_uuid."""
        await mcp_server._tool_manager.call_tool(
            'refresh_entity_summary',
            {'entity_uuid': 'node-1', 'project_id': 'dark_factory'},
        )
        mock_service.refresh_entity_summary.assert_awaited_once()
        call_kwargs = mock_service.refresh_entity_summary.call_args[1]
        assert call_kwargs.get('entity_uuid') == 'node-1'
        assert call_kwargs.get('project_id') == 'dark_factory'

    @pytest.mark.asyncio
    async def test_empty_project_id_returns_error(self, mcp_server, mock_service):
        """Empty project_id returns validation error dict."""
        result = await mcp_server._tool_manager.call_tool(
            'refresh_entity_summary',
            {'entity_uuid': 'node-1', 'project_id': ''},
        )
        # Should be a list of TextContent from FastMCP; parse to get result
        import json
        if isinstance(result, list):
            content = result[0].text if hasattr(result[0], 'text') else str(result[0])
            parsed = json.loads(content)
        else:
            parsed = result
        assert 'error' in parsed
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.refresh_entity_summary.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_exception_returns_error_dict(self, mcp_server, mock_service):
        """Exception from memory_service returns error dict (not raised)."""
        mock_service.refresh_entity_summary = AsyncMock(
            side_effect=RuntimeError('FalkorDB connection failed')
        )
        result = await mcp_server._tool_manager.call_tool(
            'refresh_entity_summary',
            {'entity_uuid': 'node-1', 'project_id': 'dark_factory'},
        )
        import json
        if isinstance(result, list):
            content = result[0].text if hasattr(result[0], 'text') else str(result[0])
            parsed = json.loads(content)
        else:
            parsed = result
        assert 'error' in parsed
        assert 'FalkorDB connection failed' in parsed['error']

    # -- step-5 (task-309): entity_name support in the MCP tool --

    @pytest.mark.asyncio
    async def test_tool_accepts_entity_name(self, mcp_server, mock_service):
        """Tool accepts entity_name and passes it to memory_service."""
        await mcp_server._tool_manager.call_tool(
            'refresh_entity_summary',
            {'entity_name': 'Alice', 'project_id': 'dark_factory'},
        )
        mock_service.refresh_entity_summary.assert_awaited_once()
        call_kwargs = mock_service.refresh_entity_summary.call_args[1]
        assert call_kwargs.get('entity_name') == 'Alice'
        assert call_kwargs.get('project_id') == 'dark_factory'

    @pytest.mark.asyncio
    async def test_neither_uuid_nor_name_returns_validation_error(self, mcp_server, mock_service):
        """Returns validation error when neither entity_uuid nor entity_name is provided."""
        import json
        result = await mcp_server._tool_manager.call_tool(
            'refresh_entity_summary',
            {'project_id': 'dark_factory'},
        )
        if isinstance(result, list):
            content = result[0].text if hasattr(result[0], 'text') else str(result[0])
            parsed = json.loads(content)
        else:
            parsed = result
        assert 'error' in parsed
        mock_service.refresh_entity_summary.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_entity_uuid_alone_still_works(self, mcp_server, mock_service):
        """Backward compat: entity_uuid alone still calls the service correctly."""
        await mcp_server._tool_manager.call_tool(
            'refresh_entity_summary',
            {'entity_uuid': 'node-1', 'project_id': 'dark_factory'},
        )
        mock_service.refresh_entity_summary.assert_awaited_once()
        call_kwargs = mock_service.refresh_entity_summary.call_args[1]
        assert call_kwargs.get('entity_uuid') == 'node-1'

    @pytest.mark.asyncio
    async def test_entity_name_alone_resolves_correctly(self, mcp_server, mock_service):
        """entity_name alone is passed through; service does the resolution."""
        await mcp_server._tool_manager.call_tool(
            'refresh_entity_summary',
            {'entity_name': 'Alice', 'project_id': 'dark_factory'},
        )
        call_kwargs = mock_service.refresh_entity_summary.call_args[1]
        assert call_kwargs.get('entity_name') == 'Alice'
        assert call_kwargs.get('entity_uuid') is None


# ---------------------------------------------------------------------------
# step-7 (task-309): FUSED_MEMORY_INSTRUCTIONS and tool docstring
# ---------------------------------------------------------------------------

class TestFusedMemoryInstructionsEntityName:
    """FUSED_MEMORY_INSTRUCTIONS mentions entity name-based lookup for refresh_entity_summary."""

    def test_instructions_mention_entity_name_lookup(self):
        """FUSED_MEMORY_INSTRUCTIONS describes name-based lookup for refresh_entity_summary."""
        from fused_memory.server.tools import FUSED_MEMORY_INSTRUCTIONS
        # The refresh_entity_summary entry specifically should mention UUID or name acceptance
        assert 'entity_name' in FUSED_MEMORY_INSTRUCTIONS or (
            'name' in FUSED_MEMORY_INSTRUCTIONS
            and 'refresh_entity_summary: ' in FUSED_MEMORY_INSTRUCTIONS
        )
        # Specifically: the instructions must say the tool accepts a name (not just UUID)
        refresh_line = next(
            (line for line in FUSED_MEMORY_INSTRUCTIONS.splitlines()
             if 'refresh_entity_summary' in line and ('uuid' in line.lower() or 'name' in line.lower())),
            None,
        )
        assert refresh_line is not None, (
            'refresh_entity_summary line in FUSED_MEMORY_INSTRUCTIONS should mention uuid or name'
        )

    def test_tool_docstring_mentions_both_uuid_and_name(self):
        """The refresh_entity_summary tool docstring describes both entity_uuid and entity_name."""
        svc = AsyncMock()
        server = create_mcp_server(svc)

        import asyncio
        tools = asyncio.run(server.list_tools())
        tool = next(t for t in tools if t.name == 'refresh_entity_summary')
        desc = tool.description or ''
        assert 'entity_uuid' in desc
        assert 'entity_name' in desc


# ---------------------------------------------------------------------------
# step-11: DISALLOW_MEMORY_WRITES list in cli_stage_runner.py
# ---------------------------------------------------------------------------

class TestDisallowListForRefreshEntitySummary:
    """refresh_entity_summary must be in DISALLOW_MEMORY_WRITES (not in STAGE1_DISALLOWED)."""

    def test_refresh_entity_summary_in_disallow_memory_writes(self):
        """'mcp__fused-memory__refresh_entity_summary' must be in DISALLOW_MEMORY_WRITES
        so Stage 3 (read-only) cannot call it."""
        from fused_memory.reconciliation.cli_stage_runner import DISALLOW_MEMORY_WRITES
        assert 'mcp__fused-memory__refresh_entity_summary' in DISALLOW_MEMORY_WRITES

    def test_refresh_entity_summary_not_in_stage1_disallowed(self):
        """Stage 1 must be able to call refresh_entity_summary (not in STAGE1_DISALLOWED)."""
        from fused_memory.reconciliation.cli_stage_runner import STAGE1_DISALLOWED
        assert 'mcp__fused-memory__refresh_entity_summary' not in STAGE1_DISALLOWED

    def test_refresh_entity_summary_in_stage3_disallowed(self):
        """Stage 3 must NOT be able to call refresh_entity_summary (in STAGE3_DISALLOWED
        via DISALLOW_MEMORY_WRITES)."""
        from fused_memory.reconciliation.cli_stage_runner import STAGE3_DISALLOWED
        assert 'mcp__fused-memory__refresh_entity_summary' in STAGE3_DISALLOWED


# ---------------------------------------------------------------------------
# step-13: STAGE1_SYSTEM_PROMPT mentions refresh_entity_summary
# ---------------------------------------------------------------------------

class TestStage1PromptMentionsRefreshEntitySummary:
    """STAGE1_SYSTEM_PROMPT must list the tool and guide its usage."""

    def test_prompt_lists_refresh_entity_summary_tool(self):
        """STAGE1_SYSTEM_PROMPT lists 'refresh_entity_summary' in Available Tools section."""
        from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
        assert 'refresh_entity_summary' in STAGE1_SYSTEM_PROMPT

    def test_prompt_instructs_calling_after_edge_deletion(self):
        """STAGE1_SYSTEM_PROMPT instructs agent to call refresh after deleting edges."""
        from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
        # The prompt should mention refreshing summary after deleting edges
        prompt_lower = STAGE1_SYSTEM_PROMPT.lower()
        assert 'after delet' in prompt_lower or 'after you delet' in prompt_lower


# ---------------------------------------------------------------------------
# step-15: partial_failure_misreported fix in MemoryService.refresh_entity_summary
# ---------------------------------------------------------------------------

class TestMemoryServiceRefreshEntitySummaryJournalFix:
    """Verify try/finally journal pattern: journal failure cannot mask success,
    and backend failure is always journaled."""

    @pytest.fixture
    def service_with_journal(self, mock_config):
        """MemoryService with mocked backends and a write journal."""
        from fused_memory.services.memory_service import MemoryService
        svc = MemoryService(mock_config)
        svc.graphiti = MagicMock()
        svc.mem0 = MagicMock()
        svc.durable_queue = MagicMock()
        svc.durable_queue.enqueue = AsyncMock(return_value=1)
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        svc.set_write_journal(mock_journal)
        return svc, mock_journal

    @pytest.mark.asyncio
    async def test_journal_failure_does_not_mask_successful_refresh(
        self, service_with_journal
    ):
        """When graphiti succeeds but journal.log_write_op raises RuntimeError,
        the result is still returned — journal failure cannot produce a false negative."""
        svc, mock_journal = service_with_journal
        expected_result = {
            'uuid': 'node-1',
            'name': 'Alice',
            'old_summary': 'old',
            'new_summary': 'Alice knows Bob',
            'edge_count': 1,
        }
        svc.graphiti.refresh_entity_summary = AsyncMock(return_value=expected_result)
        mock_journal.log_write_op.side_effect = RuntimeError('journal db is full')

        # Should NOT raise — journal failure must not mask the successful operation
        result = await svc.refresh_entity_summary(
            entity_uuid='node-1',
            project_id='dark_factory',
        )
        assert result == expected_result

    @pytest.mark.asyncio
    async def test_graphiti_failure_still_logs_journal(self, service_with_journal):
        """When graphiti.refresh_entity_summary raises ValueError, the journal is
        still called (even on failure) and the ValueError propagates to the caller."""
        svc, mock_journal = service_with_journal
        svc.graphiti.refresh_entity_summary = AsyncMock(
            side_effect=ValueError('node not found')
        )

        with pytest.raises(ValueError, match='node not found'):
            await svc.refresh_entity_summary(
                entity_uuid='node-1',
                project_id='dark_factory',
            )

        # Journal must have been called even though graphiti failed
        mock_journal.log_write_op.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_journal_logs_success_false_on_backend_error(
        self, service_with_journal
    ):
        """When graphiti raises, journal is called with success=False and
        a non-empty error string containing the exception message."""
        svc, mock_journal = service_with_journal
        svc.graphiti.refresh_entity_summary = AsyncMock(
            side_effect=ValueError('FalkorDB timeout')
        )

        with pytest.raises(ValueError):
            await svc.refresh_entity_summary(
                entity_uuid='node-1',
                project_id='dark_factory',
            )

        mock_journal.log_write_op.assert_awaited_once()
        call_kwargs = mock_journal.log_write_op.call_args[1]
        assert call_kwargs.get('success') is False
        assert call_kwargs.get('error') is not None
        assert 'FalkorDB timeout' in call_kwargs['error']
