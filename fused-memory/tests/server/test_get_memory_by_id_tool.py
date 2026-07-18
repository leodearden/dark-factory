"""Behaviour tests for the get_memory_by_id MCP tool (task 2765).

Direct Mem0/Qdrant point-id lookup (raw content + metadata), non-semantic —
bypasses BOTH semantic search ranking and metadata-equality filtering. Read-only,
Mem0-only, not in any DISALLOW_* list (auto-allowed in Stage 1/Stage 3).

Load-bearing no-silent-fail contract: a GENUINE not-found returns
``{'found': False}`` (a valid answer), while a Qdrant timeout/backend error
surfaces as ``{'error', 'error_type'}`` with ``found`` ABSENT — so the
reconciliation consumer can tell "memory genuinely absent/folded" apart from
"backend timed out".

Mirrors tests/server/test_get_memories_by_metadata_tool.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from fused_memory.backends.mem0_client import Mem0Backend
from fused_memory.server.tools import create_mcp_server
from fused_memory.services.memory_service import MemoryService

_PROJECT_ID = 'dark_factory'
_MEMORY_ID = '77a3f6bc-0000-0000-0000-000000000000'
_SAMPLE_METADATA = {
    'data': 'the raw memory content',
    'category': 'observations_and_summaries',
    'agent_id': 'x',
}


class TestGetMemoryByIdTool:
    """Behaviour tests for mcp__fused-memory__get_memory_by_id."""

    @pytest.mark.asyncio
    async def test_happy_path_found(self):
        """Found: tool returns {'found':True, ids, content, metadata} and calls
        the service exactly once with the right project_id and memory_id."""
        mock_service = AsyncMock()
        mock_service.get_memory_by_id = AsyncMock(
            return_value={'id': _MEMORY_ID, 'content': 'the raw memory content', 'metadata': _SAMPLE_METADATA}
        )
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'get_memory_by_id',
            {'project_id': _PROJECT_ID, 'memory_id': _MEMORY_ID},
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert result == {
            'found': True,
            'memory_id': _MEMORY_ID,
            'project_id': _PROJECT_ID,
            'content': 'the raw memory content',
            'metadata': _SAMPLE_METADATA,
        }, f'Unexpected result: {result!r}'
        assert 'error' not in result, f'Unexpected error in result: {result!r}'

        mock_service.get_memory_by_id.assert_called_once_with(
            project_id=_PROJECT_ID,
            memory_id=_MEMORY_ID,
        )

    @pytest.mark.asyncio
    async def test_not_found_returns_found_false_not_error(self):
        """Genuine miss: service returns None → {'found':False, ...} and NOT an error."""
        mock_service = AsyncMock()
        mock_service.get_memory_by_id = AsyncMock(return_value=None)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'get_memory_by_id',
            {'project_id': _PROJECT_ID, 'memory_id': _MEMORY_ID},
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert result == {
            'found': False,
            'memory_id': _MEMORY_ID,
            'project_id': _PROJECT_ID,
        }, f'Unexpected result: {result!r}'
        assert 'error' not in result, (
            f'A genuine not-found must not be an error dict: {result!r}'
        )

    @pytest.mark.asyncio
    async def test_invalid_project_id_returns_validation_error(self):
        """Invalid project_id (unsafe chars) returns a validation error dict and
        does NOT call the service."""
        mock_service = AsyncMock()
        mock_service.get_memory_by_id = AsyncMock(return_value=None)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'get_memory_by_id',
            {'project_id': 'bad id!', 'memory_id': _MEMORY_ID},
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert 'error' in result, f'Expected error key in result: {result!r}'
        assert result.get('error_type') == 'ValidationError', (
            f"Expected error_type='ValidationError', got: {result!r}"
        )
        mock_service.get_memory_by_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_service_exception_caught(self):
        """A service exception is caught and returned as {'error','error_type'}."""
        mock_service = AsyncMock()
        mock_service.get_memory_by_id = AsyncMock(side_effect=ValueError('boom'))
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'get_memory_by_id',
            {'project_id': _PROJECT_ID, 'memory_id': _MEMORY_ID},
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert 'error' in result, f'Expected error key in result: {result!r}'
        assert result.get('error_type') == 'ValueError', (
            f"Expected error_type='ValueError', got: {result!r}"
        )
        assert 'boom' in result.get('error', ''), (
            f'Expected original error message in result: {result!r}'
        )

    @pytest.mark.asyncio
    async def test_timeout_surfaces_as_error_not_found_false(self, mock_config):
        """A Qdrant retrieve timeout surfaces as {'error','error_type':'TimeoutError'}
        — it must NOT masquerade as a genuine {'found': False} miss (no-silent-fail
        invariant; the whole reason this tool distinguishes miss from timeout).

        Drives a REAL MemoryService + REAL Mem0Backend (only the underlying async
        Qdrant client is patched) so the full get_point_by_id → get_memory_by_id →
        @mcp_tool_errors chain is genuinely exercised.
        """
        svc = MemoryService(mock_config)
        svc.mem0 = Mem0Backend(mock_config)
        mock_client = AsyncMock()
        mock_client.retrieve = AsyncMock(side_effect=TimeoutError('qdrant retrieve timed out'))
        server = create_mcp_server(svc)

        with patch.object(svc.mem0, '_get_async_qdrant', AsyncMock(return_value=mock_client)):
            result = await server._tool_manager.call_tool(
                'get_memory_by_id',
                {'project_id': _PROJECT_ID, 'memory_id': _MEMORY_ID},
            )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert 'error' in result, f'Expected error key in result: {result!r}'
        assert result.get('error_type') == 'TimeoutError', (
            f"Expected error_type='TimeoutError', got: {result!r}"
        )
        assert result.get('found') is None, (
            f'A timeout must NOT masquerade as found:False: {result!r}'
        )

    @pytest.mark.asyncio
    async def test_tool_registered(self):
        """The get_memory_by_id tool is registered on the MCP server."""
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)

        tool_names = [t.name for t in await server.list_tools()]
        assert 'get_memory_by_id' in tool_names, (
            f'get_memory_by_id not registered; tools: {tool_names!r}'
        )
