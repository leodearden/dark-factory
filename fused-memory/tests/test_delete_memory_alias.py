"""Tests for the delete_memory MCP tool's `id` alias for `memory_id`.

delete_memory's response (and search results) return the memory under the
key `id`, but the tool itself required `memory_id`. Agents already send
`{'id': ...}` back (see orchestrator test_agent_loop.py:990), so the tool
must accept `id` as an alias for `memory_id` — matching the
refresh_entity_summary `name`/`entity_name` alias pattern in
test_refresh_entity_summary.py (TestRefreshEntitySummaryMcpTool).
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from fused_memory.server.tools import create_mcp_server


def _parse_result(result):
    """Parse a FastMCP call_tool result (list of TextContent) into a dict."""
    if isinstance(result, list):
        content = result[0].text if hasattr(result[0], 'text') else str(result[0])
        return json.loads(content)
    return result


class TestDeleteMemoryIdAlias:
    """MCP tool delete_memory accepts `id` as an alias for `memory_id`."""

    @pytest.fixture
    def mock_service(self):
        """Mock MemoryService for tool registration."""
        svc = AsyncMock()
        svc.delete_memory = AsyncMock(
            return_value={'status': 'deleted', 'store': 'mem0', 'id': '00000000-0000-4000-8000-00000000000a'}
        )
        return svc

    @pytest.fixture
    def mcp_server(self, mock_service):
        """MCP server with mock memory service."""
        return create_mcp_server(mock_service)

    @pytest.mark.asyncio
    async def test_id_alias_delegates_with_memory_id(self, mcp_server, mock_service):
        """Calling with `id` delegates to memory_service.delete_memory(memory_id=...)."""
        await mcp_server._tool_manager.call_tool(
            'delete_memory',
            {'id': '00000000-0000-4000-8000-00000000000a', 'store': 'mem0', 'project_id': 'dark_factory'},
        )
        mock_service.delete_memory.assert_awaited_once()
        call_kwargs = mock_service.delete_memory.call_args[1]
        assert call_kwargs.get('memory_id') == '00000000-0000-4000-8000-00000000000a'
        assert call_kwargs.get('store') == 'mem0'
        assert call_kwargs.get('project_id') == 'dark_factory'

    @pytest.mark.asyncio
    async def test_canonical_memory_id_still_works(self, mcp_server, mock_service):
        """Backward compat: memory_id alone still calls the service correctly."""
        await mcp_server._tool_manager.call_tool(
            'delete_memory',
            {'memory_id': '00000000-0000-4000-8000-00000000000a', 'store': 'mem0', 'project_id': 'dark_factory'},
        )
        mock_service.delete_memory.assert_awaited_once()
        call_kwargs = mock_service.delete_memory.call_args[1]
        assert call_kwargs.get('memory_id') == '00000000-0000-4000-8000-00000000000a'

    @pytest.mark.asyncio
    async def test_neither_id_nor_memory_id_returns_validation_error(
        self, mcp_server, mock_service
    ):
        """Returns a ValidationError dict when neither memory_id nor id is provided."""
        result = await mcp_server._tool_manager.call_tool(
            'delete_memory',
            {'store': 'mem0', 'project_id': 'dark_factory'},
        )
        parsed = _parse_result(result)
        assert 'error' in parsed
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_conflicting_id_and_memory_id_returns_validation_error(
        self, mcp_server, mock_service
    ):
        """Returns a ValidationError dict when id and memory_id disagree."""
        result = await mcp_server._tool_manager.call_tool(
            'delete_memory',
            {'id': 'a', 'memory_id': 'b', 'store': 'mem0', 'project_id': 'dark_factory'},
        )
        parsed = _parse_result(result)
        assert 'error' in parsed
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_equal_id_and_memory_id_delegates_with_memory_id(
        self, mcp_server, mock_service
    ):
        """Supplying both id and memory_id with the SAME value is valid, not a conflict.

        The docstring's 'Exactly one must be provided; supplying both with
        conflicting values is an error' only rejects disagreement — matching
        values should resolve and delegate normally rather than error out.
        """
        await mcp_server._tool_manager.call_tool(
            'delete_memory',
            {'id': '00000000-0000-4000-8000-00000000000a', 'memory_id': '00000000-0000-4000-8000-00000000000a', 'store': 'mem0', 'project_id': 'dark_factory'},
        )
        mock_service.delete_memory.assert_awaited_once()
        call_kwargs = mock_service.delete_memory.call_args[1]
        assert call_kwargs.get('memory_id') == '00000000-0000-4000-8000-00000000000a'
