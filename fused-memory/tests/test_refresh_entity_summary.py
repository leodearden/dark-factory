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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.backends.graphiti_client import GraphitiBackend


# ---------------------------------------------------------------------------
# step-1: GraphitiBackend.get_valid_edges_for_node
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
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            result = await backend.get_valid_edges_for_node('node-uuid-1')
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
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            result = await backend.get_valid_edges_for_node('node-uuid-1')
        assert result == []

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        """Raises RuntimeError when client is not initialized."""
        backend = GraphitiBackend(mock_config)  # client is None
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.get_valid_edges_for_node('node-uuid-1')

    @pytest.mark.asyncio
    async def test_passes_uuid_to_query(self, mock_config, make_backend, make_graph_mock):
        """Passes node UUID as parameter to the Cypher query."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        node_uuid = 'my-node-uuid'
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            await backend.get_valid_edges_for_node(node_uuid)
        call_args = graph.query.call_args
        assert call_args is not None
        args, kwargs = call_args
        cypher_params = args[1] if len(args) > 1 else kwargs.get('params', {})
        assert cypher_params.get('uuid') == node_uuid

    @pytest.mark.asyncio
    async def test_filters_invalid_at_null(self, mock_config, make_backend, make_graph_mock):
        """Query uses WHERE e.invalid_at IS NULL to filter active edges only."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            await backend.get_valid_edges_for_node('node-uuid-1')
        call_args = graph.query.call_args
        args, kwargs = call_args
        cypher = args[0] if args else kwargs.get('query', '')
        assert 'invalid_at IS NULL' in cypher


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
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            await backend.update_node_summary('node-uuid-1', 'Alice knows Bob.\nAlice works at Acme.')
        graph.query.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_passes_uuid_and_summary_as_params(self, mock_config, make_backend, make_graph_mock):
        """Query receives uuid and summary as Cypher parameters."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        node_uuid = 'my-node-uuid'
        summary = 'Alice knows Bob.'
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            await backend.update_node_summary(node_uuid, summary)
        call_args = graph.query.call_args
        args, kwargs = call_args
        cypher_params = args[1] if len(args) > 1 else kwargs.get('params', {})
        assert cypher_params.get('uuid') == node_uuid
        assert cypher_params.get('summary') == summary

    @pytest.mark.asyncio
    async def test_cypher_sets_summary_property(self, mock_config, make_backend, make_graph_mock):
        """Cypher query contains SET n.summary = $summary."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            await backend.update_node_summary('node-uuid-1', 'some summary')
        call_args = graph.query.call_args
        args, kwargs = call_args
        cypher = args[0] if args else kwargs.get('query', '')
        assert 'SET n.summary' in cypher

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        """Raises RuntimeError when client is not initialized."""
        backend = GraphitiBackend(mock_config)
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.update_node_summary('node-uuid-1', 'summary text')


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
        result = await backend.refresh_entity_summary('node-uuid-1')
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
        result = await backend.refresh_entity_summary('node-uuid-1')
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
        result = await backend.refresh_entity_summary('node-uuid-1')
        backend.update_node_summary.assert_awaited_once_with('node-uuid-1', result['new_summary'])

    @pytest.mark.asyncio
    async def test_empty_edges_produces_empty_summary(self, mock_config, make_backend):
        """When zero valid edges remain, new_summary is empty string and update is called."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(return_value=('Alice', 'old summary'))
        backend.get_valid_edges_for_node = AsyncMock(return_value=[])
        backend.update_node_summary = AsyncMock()
        result = await backend.refresh_entity_summary('node-uuid-1')
        assert result['new_summary'] == ''
        assert result['edge_count'] == 0
        backend.update_node_summary.assert_awaited_once_with('node-uuid-1', '')

    @pytest.mark.asyncio
    async def test_old_summary_returned_in_result(self, mock_config, make_backend):
        """The old summary from get_node_text is preserved in result for auditing."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(return_value=('Bob', 'prior summary text'))
        backend.get_valid_edges_for_node = AsyncMock(return_value=[
            {'uuid': 'e1', 'fact': 'Bob lives in London', 'name': 'lives_in'},
        ])
        backend.update_node_summary = AsyncMock()
        result = await backend.refresh_entity_summary('node-uuid-2')
        assert result['old_summary'] == 'prior summary text'


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
        service.graphiti.refresh_entity_summary.assert_awaited_once_with('node-1')
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
        result = await mcp_server._tool_manager.call_tool(
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
