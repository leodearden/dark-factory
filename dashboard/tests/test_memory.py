"""Tests for dashboard.data.memory — MCP-based memory health metrics."""

from __future__ import annotations

import json

import httpx
import pytest


def _make_mcp_response(inner_dict: dict) -> httpx.Response:
    """Build a mock MCP JSON-RPC response wrapping *inner_dict*."""
    body = {
        'jsonrpc': '2.0',
        'id': 1,
        'result': {
            'content': [
                {'type': 'text', 'text': json.dumps(inner_dict)},
            ]
        },
    }
    return httpx.Response(200, json=body)


class TestMcpToolCall:
    """Tests for the low-level mcp_tool_call function."""

    async def test_successful_call(self):
        """Valid MCP response is parsed and inner dict returned."""
        from dashboard.data.memory import mcp_tool_call

        expected = {'graphiti': {'connected': True}, 'mem0': {'connected': True}}

        def handler(request: httpx.Request) -> httpx.Response:
            return _make_mcp_response(expected)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await mcp_tool_call(
                client, 'http://localhost:8000', 'get_status', {'project_id': 'dark_factory'}
            )

        assert result == expected

    async def test_timeout_propagates(self):
        """httpx.TimeoutException from the transport propagates to caller."""
        from dashboard.data.memory import mcp_tool_call

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException('timed out')

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(httpx.TimeoutException):
                await mcp_tool_call(
                    client, 'http://localhost:8000', 'get_status', {}
                )

    async def test_non_200_raises_value_error(self):
        """Non-200 HTTP status raises ValueError."""
        from dashboard.data.memory import mcp_tool_call

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text='Internal Server Error')

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ValueError, match='non-200'):
                await mcp_tool_call(
                    client, 'http://localhost:8000', 'get_status', {}
                )
