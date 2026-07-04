"""Tests for set_entity_summary across backend, service, and MCP tool layers.

set_entity_summary is a direct summary-overwrite escape hatch: it writes an
explicit string verbatim (including '' to clear), bypassing edge-derivation
entirely.  This is distinct from refresh_entity_summary, which regenerates
the summary from the current valid-edge set — see test_refresh_entity_summary.py.

Covers:
- GraphitiBackend.set_entity_summary()
- MemoryService.set_entity_summary()
- MCP tool set_entity_summary in tools.py
- DISALLOW_MEMORY_WRITES list in cli_stage_runner.py
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.backends.graphiti_client import GraphitiBackend, NodeNotFoundError

# ---------------------------------------------------------------------------
# step-1: GraphitiBackend.set_entity_summary
# ---------------------------------------------------------------------------


class TestGraphitiBackendSetEntitySummary:
    """GraphitiBackend.set_entity_summary(node_uuid, summary, *, group_id) overwrites
    an Entity node's summary verbatim, bypassing edge-derivation entirely."""

    @pytest.mark.asyncio
    async def test_returns_result_dict(self, mock_config, make_backend):
        """Returns dict with uuid, name, old_summary, new_summary; new_summary is the
        exact string passed in (not an edge-derived join)."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(return_value=('Alice', 'stale narrative text'))
        backend.update_node_summary = AsyncMock()
        backend.get_valid_edges_for_node = AsyncMock()

        result = await backend.set_entity_summary('node-uuid-1', 'Corrected summary.', group_id='test')

        assert result == {
            'uuid': 'node-uuid-1',
            'name': 'Alice',
            'old_summary': 'stale narrative text',
            'new_summary': 'Corrected summary.',
        }

    @pytest.mark.asyncio
    async def test_calls_update_node_summary_with_explicit_text(self, mock_config, make_backend):
        """Calls update_node_summary(node_uuid, summary, group_id=...) exactly once with
        the explicit text — and never touches get_valid_edges_for_node (no edge-derived join)."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(return_value=('Alice', 'old'))
        backend.update_node_summary = AsyncMock()
        backend.get_valid_edges_for_node = AsyncMock()

        await backend.set_entity_summary('node-uuid-1', 'Exact text.', group_id='test')

        backend.update_node_summary.assert_awaited_once_with(
            'node-uuid-1', 'Exact text.', group_id='test'
        )
        backend.get_valid_edges_for_node.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_string_summary_forwarded_verbatim(self, mock_config, make_backend):
        """An empty-string summary is accepted and forwarded verbatim — the clear case."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(return_value=('Alice', 'stale text to clear'))
        backend.update_node_summary = AsyncMock()
        backend.get_valid_edges_for_node = AsyncMock()

        result = await backend.set_entity_summary('node-uuid-1', '', group_id='test')

        assert result['new_summary'] == ''
        backend.update_node_summary.assert_awaited_once_with('node-uuid-1', '', group_id='test')
        backend.get_valid_edges_for_node.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_node_not_found_propagates(self, mock_config, make_backend):
        """NodeNotFoundError from get_node_text propagates — a missing node is not a
        silent no-op."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(side_effect=NodeNotFoundError('Entity node not found: missing-uuid'))
        backend.update_node_summary = AsyncMock()

        with pytest.raises(NodeNotFoundError):
            await backend.set_entity_summary('missing-uuid', 'text', group_id='test')

        backend.update_node_summary.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        """Raises RuntimeError when backend has no driver (not initialized)."""
        backend = GraphitiBackend(mock_config)  # _driver is None
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.set_entity_summary('node-uuid-1', 'text', group_id='test')


# ---------------------------------------------------------------------------
# step-3: MemoryService.set_entity_summary
# ---------------------------------------------------------------------------


class TestMemoryServiceSetEntitySummary:
    """MemoryService.set_entity_summary() delegates to the graphiti backend.

    Mirrors TestMemoryServiceRefreshEntitySummary + TestMemoryServiceRefreshEntitySummaryJournalFix
    (test_refresh_entity_summary.py) — same identifier-resolution, validation, and
    try/finally journal-logging contracts, applied to the direct-overwrite tool.
    """

    @pytest.fixture
    def service(self, mock_config):
        """MemoryService with mocked backends (no real DB needed)."""
        from fused_memory.services.memory_service import MemoryService
        svc = MemoryService(mock_config)
        svc.graphiti = MagicMock()
        svc.graphiti.set_entity_summary = AsyncMock(
            return_value={
                'uuid': 'node-1',
                'name': 'Alice',
                'old_summary': 'old',
                'new_summary': 'Corrected summary.',
            }
        )
        svc.mem0 = MagicMock()
        svc.durable_queue = MagicMock()
        svc.durable_queue.enqueue = AsyncMock(return_value=1)
        return svc

    @pytest.mark.asyncio
    async def test_delegates_to_graphiti_backend(self, service):
        """(a) Delegates to graphiti.set_entity_summary(uuid, summary, group_id=...) and
        returns the backend dict unchanged."""
        result = await service.set_entity_summary(
            entity_uuid='node-1',
            summary='Corrected summary.',
            project_id='dark_factory',
        )
        service.graphiti.set_entity_summary.assert_awaited_once_with(
            'node-1', 'Corrected summary.', group_id='dark_factory'
        )
        assert result == {
            'uuid': 'node-1',
            'name': 'Alice',
            'old_summary': 'old',
            'new_summary': 'Corrected summary.',
        }

    @pytest.mark.asyncio
    async def test_resolves_entity_name_to_uuid(self, service):
        """(b) entity_name resolves via resolve_entity_by_name, then delegates with the
        resolved uuid."""
        service.graphiti.resolve_entity_by_name = AsyncMock(return_value='node-1')
        result = await service.set_entity_summary(
            entity_name='Alice',
            summary='Corrected summary.',
            project_id='dark_factory',
        )
        service.graphiti.resolve_entity_by_name.assert_awaited_once_with('Alice', group_id='dark_factory')
        service.graphiti.set_entity_summary.assert_awaited_once_with(
            'node-1', 'Corrected summary.', group_id='dark_factory'
        )
        assert result['uuid'] == 'node-1'

    @pytest.mark.asyncio
    async def test_prefers_uuid_when_both_provided(self, service):
        """(c) entity_uuid takes precedence when both are supplied — resolve is never awaited."""
        service.graphiti.resolve_entity_by_name = AsyncMock(return_value='other-node')
        await service.set_entity_summary(
            entity_uuid='node-1',
            entity_name='Alice',
            summary='Corrected summary.',
            project_id='dark_factory',
        )
        service.graphiti.resolve_entity_by_name.assert_not_awaited()
        service.graphiti.set_entity_summary.assert_awaited_once_with(
            'node-1', 'Corrected summary.', group_id='dark_factory'
        )

    @pytest.mark.asyncio
    async def test_raises_value_error_when_neither_identifier_provided(self, service):
        """(d) Raises ValueError when neither entity_uuid nor entity_name is provided."""
        with pytest.raises(ValueError, match='entity_uuid.*entity_name|entity_name.*entity_uuid'):
            await service.set_entity_summary(summary='Corrected summary.', project_id='dark_factory')

    @pytest.mark.asyncio
    async def test_raises_value_error_when_summary_is_none(self, service):
        """(e) Raises ValueError when summary is None (missing/omitted)."""
        with pytest.raises(ValueError, match='summary'):
            await service.set_entity_summary(entity_uuid='node-1', project_id='dark_factory')

    @pytest.mark.asyncio
    async def test_empty_string_summary_is_allowed(self, service):
        """(f) An empty-string summary is a valid input — delegates without raising."""
        service.graphiti.set_entity_summary = AsyncMock(
            return_value={'uuid': 'node-1', 'name': 'Alice', 'old_summary': 'stale', 'new_summary': ''}
        )
        result = await service.set_entity_summary(
            entity_uuid='node-1', summary='', project_id='dark_factory'
        )
        service.graphiti.set_entity_summary.assert_awaited_once_with('node-1', '', group_id='dark_factory')
        assert result['new_summary'] == ''

    @pytest.mark.asyncio
    async def test_journal_logs_operation_and_entity_name_param(self, service):
        """(g) log_write_op is awaited with operation='set_entity_summary', project_id set,
        and params include entity_name when the name-resolution path is used."""
        service.graphiti.resolve_entity_by_name = AsyncMock(return_value='node-1')
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        service.set_write_journal(mock_journal)
        await service.set_entity_summary(
            entity_name='Alice',
            summary='Corrected summary.',
            project_id='dark_factory',
            agent_id='test-agent',
        )
        mock_journal.log_write_op.assert_awaited_once()
        call_kwargs = mock_journal.log_write_op.call_args[1]
        assert call_kwargs.get('operation') == 'set_entity_summary'
        assert call_kwargs.get('project_id') == 'dark_factory'
        assert call_kwargs['params'].get('entity_name') == 'Alice'

    @pytest.mark.asyncio
    async def test_journal_failure_does_not_mask_successful_result(self, service):
        """(h) A journal.log_write_op failure must not mask an otherwise-successful result."""
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock(side_effect=RuntimeError('journal db is full'))
        service.set_write_journal(mock_journal)
        expected = {
            'uuid': 'node-1',
            'name': 'Alice',
            'old_summary': 'old',
            'new_summary': 'Corrected summary.',
        }
        service.graphiti.set_entity_summary = AsyncMock(return_value=expected)

        result = await service.set_entity_summary(
            entity_uuid='node-1', summary='Corrected summary.', project_id='dark_factory'
        )
        assert result == expected

    @pytest.mark.asyncio
    async def test_backend_failure_still_journals_success_false(self, service):
        """(h) A backend exception still journals success=False with the error text, and
        the exception still propagates to the caller."""
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        service.set_write_journal(mock_journal)
        service.graphiti.set_entity_summary = AsyncMock(side_effect=ValueError('FalkorDB timeout'))

        with pytest.raises(ValueError, match='FalkorDB timeout'):
            await service.set_entity_summary(
                entity_uuid='node-1', summary='Corrected summary.', project_id='dark_factory'
            )

        mock_journal.log_write_op.assert_awaited_once()
        call_kwargs = mock_journal.log_write_op.call_args[1]
        assert call_kwargs.get('success') is False
        assert call_kwargs.get('error') is not None
        assert 'FalkorDB timeout' in call_kwargs['error']


# ---------------------------------------------------------------------------
# step-5: MCP tool set_entity_summary
# ---------------------------------------------------------------------------


class TestSetEntitySummaryMcpTool:
    """MCP tool set_entity_summary is registered and delegates correctly.

    Mirrors TestRefreshEntitySummaryMcpTool (test_refresh_entity_summary.py).
    """

    @pytest.fixture
    def mock_service(self):
        """Mock MemoryService for tool registration."""
        svc = AsyncMock()
        svc.set_entity_summary = AsyncMock(
            return_value={
                'uuid': 'node-1',
                'name': 'Alice',
                'old_summary': 'old text',
                'new_summary': 'Corrected summary.',
            }
        )
        return svc

    @pytest.fixture
    def mcp_server(self, mock_service):
        """MCP server with mock memory service."""
        from fused_memory.server.tools import create_mcp_server
        return create_mcp_server(mock_service)

    @staticmethod
    def _parse(result):
        """Parse FastMCP TextContent list results into a dict, as the refresh tests do."""
        import json
        if isinstance(result, list):
            content = result[0].text if hasattr(result[0], 'text') else str(result[0])
            return json.loads(content)
        return result

    @pytest.mark.asyncio
    async def test_tool_is_registered(self, mcp_server):
        """(a) 'set_entity_summary' is registered as an MCP tool."""
        tool_names = [t.name for t in await mcp_server.list_tools()]
        assert 'set_entity_summary' in tool_names

    @pytest.mark.asyncio
    async def test_delegates_to_memory_service(self, mcp_server, mock_service):
        """(b) Calling with {entity_uuid, summary, project_id} delegates to
        memory_service.set_entity_summary with entity_uuid/summary/project_id kwargs."""
        await mcp_server._tool_manager.call_tool(
            'set_entity_summary',
            {'entity_uuid': 'node-1', 'summary': 'Corrected summary.', 'project_id': 'dark_factory'},
        )
        mock_service.set_entity_summary.assert_awaited_once()
        call_kwargs = mock_service.set_entity_summary.call_args[1]
        assert call_kwargs.get('entity_uuid') == 'node-1'
        assert call_kwargs.get('summary') == 'Corrected summary.'
        assert call_kwargs.get('project_id') == 'dark_factory'

    @pytest.mark.asyncio
    async def test_tool_accepts_entity_name(self, mcp_server, mock_service):
        """(c) entity_name is accepted and passed through."""
        await mcp_server._tool_manager.call_tool(
            'set_entity_summary',
            {'entity_name': 'Alice', 'summary': 'Corrected summary.', 'project_id': 'dark_factory'},
        )
        mock_service.set_entity_summary.assert_awaited_once()
        call_kwargs = mock_service.set_entity_summary.call_args[1]
        assert call_kwargs.get('entity_name') == 'Alice'
        assert call_kwargs.get('project_id') == 'dark_factory'

    @pytest.mark.asyncio
    async def test_empty_project_id_returns_error(self, mcp_server, mock_service):
        """(d) Empty project_id returns a ValidationError dict; service is not awaited."""
        result = await mcp_server._tool_manager.call_tool(
            'set_entity_summary',
            {'entity_uuid': 'node-1', 'summary': 'Corrected summary.', 'project_id': ''},
        )
        parsed = self._parse(result)
        assert 'error' in parsed
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.set_entity_summary.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_summary_returns_error(self, mcp_server, mock_service):
        """(e) Missing summary returns a ValidationError dict; service is not awaited."""
        result = await mcp_server._tool_manager.call_tool(
            'set_entity_summary',
            {'entity_uuid': 'node-1', 'project_id': 'dark_factory'},
        )
        parsed = self._parse(result)
        assert 'error' in parsed
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.set_entity_summary.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_neither_uuid_nor_name_returns_validation_error(self, mcp_server, mock_service):
        """(f) Neither entity_uuid nor entity_name returns an error dict; service is not awaited."""
        result = await mcp_server._tool_manager.call_tool(
            'set_entity_summary',
            {'summary': 'Corrected summary.', 'project_id': 'dark_factory'},
        )
        parsed = self._parse(result)
        assert 'error' in parsed
        mock_service.set_entity_summary.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_exception_returns_error_dict(self, mcp_server, mock_service):
        """(g) A service exception is caught and returned as {error, error_type} (not raised)."""
        mock_service.set_entity_summary = AsyncMock(
            side_effect=RuntimeError('FalkorDB connection failed')
        )
        result = await mcp_server._tool_manager.call_tool(
            'set_entity_summary',
            {'entity_uuid': 'node-1', 'summary': 'Corrected summary.', 'project_id': 'dark_factory'},
        )
        parsed = self._parse(result)
        assert 'error' in parsed
        assert 'FalkorDB connection failed' in parsed['error']

    @pytest.mark.asyncio
    async def test_tool_description_mentions_uuid_and_name(self, mcp_server):
        """(h) The tool description mentions both entity_uuid and entity_name."""
        tools = await mcp_server.list_tools()
        tool = next(t for t in tools if t.name == 'set_entity_summary')
        desc = tool.description or ''
        assert 'entity_uuid' in desc
        assert 'entity_name' in desc


# ---------------------------------------------------------------------------
# step-7: DISALLOW_MEMORY_WRITES list in cli_stage_runner.py
# ---------------------------------------------------------------------------


class TestDisallowListForSetEntitySummary:
    """set_entity_summary must be in DISALLOW_MEMORY_WRITES (not in STAGE1_DISALLOWED).

    Mirrors TestDisallowListForRefreshEntitySummary (test_refresh_entity_summary.py).
    """

    def test_set_entity_summary_in_disallow_memory_writes(self):
        """'mcp__fused-memory__set_entity_summary' must be in DISALLOW_MEMORY_WRITES
        so Stage 3 (read-only) cannot call it."""
        from fused_memory.reconciliation.cli_stage_runner import DISALLOW_MEMORY_WRITES
        assert 'mcp__fused-memory__set_entity_summary' in DISALLOW_MEMORY_WRITES

    def test_set_entity_summary_not_in_stage1_disallowed(self):
        """Stage 1 must be able to call set_entity_summary (not in STAGE1_DISALLOWED)."""
        from fused_memory.reconciliation.cli_stage_runner import STAGE1_DISALLOWED
        assert 'mcp__fused-memory__set_entity_summary' not in STAGE1_DISALLOWED

    def test_set_entity_summary_in_stage3_disallowed(self):
        """Stage 3 must NOT be able to call set_entity_summary (in STAGE3_DISALLOWED via
        DISALLOW_MEMORY_WRITES)."""
        from fused_memory.reconciliation.cli_stage_runner import STAGE3_DISALLOWED
        assert 'mcp__fused-memory__set_entity_summary' in STAGE3_DISALLOWED
