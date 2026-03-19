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


# --- Helpers for higher-level function tests ---

_STATUS_PAYLOAD = {
    'graphiti': {'connected': True, 'node_count': 42},
    'mem0': {'connected': True, 'memory_count': 5},
    'queue': {'counts': {'pending': 1, 'completed': 8}, 'oldest_pending_age_seconds': 1.2},
}


class TestGetMemoryStatus:
    """Tests for get_memory_status."""

    async def test_successful_status(self, dashboard_config):
        """Returns the parsed status dict from a successful MCP response."""
        from dashboard.data.memory import get_memory_status

        def handler(request: httpx.Request) -> httpx.Response:
            return _make_mcp_response(_STATUS_PAYLOAD)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await get_memory_status(client, dashboard_config)

        assert result == _STATUS_PAYLOAD
        assert result['graphiti']['node_count'] == 42
        assert result['mem0']['memory_count'] == 5

    async def test_connect_error_returns_offline(self, dashboard_config):
        """ConnectError returns {offline: True, error: ...}."""
        from dashboard.data.memory import get_memory_status

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError('connection refused')

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await get_memory_status(client, dashboard_config)

        assert result['offline'] is True
        assert 'error' in result

    async def test_timeout_returns_offline(self, dashboard_config):
        """TimeoutException returns {offline: True, error: ...}."""
        from dashboard.data.memory import get_memory_status

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException('timed out')

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await get_memory_status(client, dashboard_config)

        assert result['offline'] is True
        assert 'error' in result
