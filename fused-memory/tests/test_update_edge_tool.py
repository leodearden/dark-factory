"""Tests for the update_edge MCP tool — clear_invalid_at support.

Step 5: RED tests (fail before step-6 implementation).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest


def _parse_tool_result(result):
    """Extract the dict from a FastMCP TextContent result or pass-through dict."""
    if isinstance(result, list):
        content = result[0].text if hasattr(result[0], 'text') else str(result[0])
        return json.loads(content)
    return result


class TestUpdateEdgeMcpToolClearInvalidAt:
    """MCP tool update_edge must accept clear_invalid_at and enforce constraints."""

    @pytest.fixture
    def mock_service(self):
        svc = MagicMock()
        svc.update_edge = AsyncMock(
            return_value={
                'status': 'updated',
                'store': 'graphiti',
                'uuid': 'e-1',
                'fact': 'unchanged',
                'refreshed_nodes': [],
                'verified': True,
            }
        )
        return svc

    @pytest.fixture
    def mcp_server(self, mock_service):
        from fused_memory.server.tools import create_mcp_server
        return create_mcp_server(mock_service)

    @pytest.mark.asyncio
    async def test_clear_invalid_at_delegates_to_service(self, mcp_server, mock_service):
        """clear_invalid_at=True alone calls service.update_edge with clear_invalid_at=True."""
        await mcp_server._tool_manager.call_tool(
            'update_edge',
            {'edge_uuid': 'e-1', 'project_id': 'dark_factory', 'clear_invalid_at': True},
        )
        mock_service.update_edge.assert_awaited_once()
        kwargs = mock_service.update_edge.call_args[1]
        assert kwargs.get('clear_invalid_at') is True
        assert kwargs.get('fact') is None
        assert kwargs.get('invalid_at') is None

    @pytest.mark.asyncio
    async def test_invalid_at_and_clear_invalid_at_is_contradiction(self, mcp_server, mock_service):
        """Passing both invalid_at and clear_invalid_at=True returns a ValidationError."""
        result = await mcp_server._tool_manager.call_tool(
            'update_edge',
            {
                'edge_uuid': 'e-1',
                'project_id': 'dark_factory',
                'invalid_at': '2026-01-01T00:00:00+00:00',
                'clear_invalid_at': True,
            },
        )
        parsed = _parse_tool_result(result)
        assert 'error' in parsed, f'Expected error dict, got {parsed}'
        assert parsed.get('error_type') == 'ValidationError', (
            f'Expected ValidationError, got {parsed}'
        )
        mock_service.update_edge.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_all_absent_returns_validation_error(self, mcp_server, mock_service):
        """No fact, no invalid_at, clear_invalid_at=False returns existing ValidationError."""
        result = await mcp_server._tool_manager.call_tool(
            'update_edge',
            {'edge_uuid': 'e-1', 'project_id': 'dark_factory'},
        )
        parsed = _parse_tool_result(result)
        assert 'error' in parsed, f'Expected error, got {parsed}'
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.update_edge.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fact_and_clear_invalid_at_together_is_accepted(self, mcp_server, mock_service):
        """fact + clear_invalid_at=True is a valid combination (update text and un-supersede)."""
        result = await mcp_server._tool_manager.call_tool(
            'update_edge',
            {
                'edge_uuid': 'e-1',
                'project_id': 'dark_factory',
                'fact': 'new fact text',
                'clear_invalid_at': True,
            },
        )
        parsed = _parse_tool_result(result)
        assert 'error' not in parsed or parsed.get('error_type') != 'ValidationError', (
            f'fact + clear_invalid_at should be accepted, but got error: {parsed}'
        )
        mock_service.update_edge.assert_awaited_once()
        kwargs = mock_service.update_edge.call_args[1]
        assert kwargs.get('clear_invalid_at') is True
        assert kwargs.get('fact') == 'new fact text'
