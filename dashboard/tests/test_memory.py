"""Tests for dashboard.data.memory — MCP-based memory health metrics."""

from __future__ import annotations

import json
import logging

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
                await mcp_tool_call(client, 'http://localhost:8000', 'get_status', {})

    async def test_non_200_raises_value_error(self):
        """Non-200 HTTP status raises ValueError."""
        from dashboard.data.memory import mcp_tool_call

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text='Internal Server Error')

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ValueError, match='non-200'):
                await mcp_tool_call(client, 'http://localhost:8000', 'get_status', {})

    async def test_307_redirect_followed_with_follow_redirects(self):
        """Regression: 307 from /mcp/ to /mcp is followed when follow_redirects=True.

        Documents the original bug: the fused-memory MCP server (Starlette with
        redirect_slashes=True) redirects POST /mcp/ -> POST /mcp via 307.
        With follow_redirects=False (old default), mcp_tool_call raised
        ValueError('MCP call get_status returned non-200 status: 307').

        This test verifies the safety net: when a client has follow_redirects=True,
        the 307 is transparently followed and the call succeeds.
        """
        from dashboard.data.memory import mcp_tool_call

        expected = {'graphiti': {'connected': True}, 'mem0': {'connected': True}}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == '/mcp/':
                # Simulate Starlette's redirect_slashes=True behaviour
                return httpx.Response(
                    307,
                    headers={'Location': 'http://localhost:8000/mcp'},
                )
            # Canonical path: return valid MCP response
            return _make_mcp_response(expected)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
            result = await mcp_tool_call(
                client, 'http://localhost:8000', 'get_status', {}
            )

        assert result == expected

    async def test_posts_to_correct_url_path(self):
        """mcp_tool_call posts to '{base_url}/mcp' (no trailing slash)."""
        from dashboard.data.memory import mcp_tool_call

        captured_path: list[str] = []
        expected = {'ok': True}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_path.append(request.url.path)
            return _make_mcp_response(expected)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await mcp_tool_call(
                client, 'http://localhost:8000', 'get_status', {}
            )

        assert result == expected
        assert len(captured_path) == 1, 'handler should be called exactly once'
        assert captured_path[0] == '/mcp', (
            f'Expected path /mcp (no trailing slash), got {captured_path[0]!r}'
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

    async def test_non_200_response_returns_offline(self, dashboard_config):
        """Non-200 HTTP status (ValueError from mcp_tool_call) returns offline fallback."""
        from dashboard.data.memory import get_memory_status

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text='Internal Server Error')

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await get_memory_status(client, dashboard_config)

        assert result['offline'] is True
        assert 'error' in result


_QUEUE_STATS_PAYLOAD = {
    'counts': {'pending': 3, 'in_flight': 1, 'retry': 0, 'completed': 10, 'dead': 0},
    'oldest_pending_age_seconds': 5.5,
}


class TestGetQueueStats:
    """Tests for get_queue_stats."""

    async def test_successful_stats(self, dashboard_config):
        """Returns the parsed queue stats dict from a successful MCP response."""
        from dashboard.data.memory import get_queue_stats

        def handler(request: httpx.Request) -> httpx.Response:
            return _make_mcp_response(_QUEUE_STATS_PAYLOAD)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await get_queue_stats(client, dashboard_config)

        assert result == _QUEUE_STATS_PAYLOAD
        assert result['counts']['pending'] == 3
        assert result['oldest_pending_age_seconds'] == 5.5

    async def test_connect_error_returns_offline(self, dashboard_config):
        """ConnectError returns {offline: True, error: ...}."""
        from dashboard.data.memory import get_queue_stats

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError('connection refused')

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await get_queue_stats(client, dashboard_config)

        assert result['offline'] is True
        assert 'error' in result

    async def test_timeout_returns_offline(self, dashboard_config):
        """TimeoutException returns {offline: True, error: ...}."""
        from dashboard.data.memory import get_queue_stats

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException('timed out')

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await get_queue_stats(client, dashboard_config)

        assert result['offline'] is True
        assert 'error' in result

    async def test_non_200_response_returns_offline(self, dashboard_config):
        """Non-200 HTTP status (ValueError from mcp_tool_call) returns offline fallback."""
        from dashboard.data.memory import get_queue_stats

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text='Internal Server Error')

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await get_queue_stats(client, dashboard_config)

        assert result['offline'] is True
        assert 'error' in result


class TestMalformedResponse:
    """Tests for mcp_tool_call with malformed MCP responses."""

    async def test_missing_content_key(self):
        """Response with no result.content path returns empty dict."""
        from dashboard.data.memory import mcp_tool_call

        def handler(request: httpx.Request) -> httpx.Response:
            # Valid JSON but missing the result.content path
            return httpx.Response(200, json={'jsonrpc': '2.0', 'id': 1, 'result': {}})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await mcp_tool_call(client, 'http://localhost:8000', 'get_status', {})

        assert result == {}

    async def test_invalid_inner_json(self):
        """Response where text field contains non-JSON returns empty dict."""
        from dashboard.data.memory import mcp_tool_call

        def handler(request: httpx.Request) -> httpx.Response:
            body = {
                'jsonrpc': '2.0',
                'id': 1,
                'result': {
                    'content': [{'type': 'text', 'text': 'not valid json!!!'}],
                },
            }
            return httpx.Response(200, json=body)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await mcp_tool_call(client, 'http://localhost:8000', 'get_status', {})

        assert result == {}

    async def test_empty_content_array(self):
        """Response with empty content array returns empty dict."""
        from dashboard.data.memory import mcp_tool_call

        def handler(request: httpx.Request) -> httpx.Response:
            body = {
                'jsonrpc': '2.0',
                'id': 1,
                'result': {'content': []},
            }
            return httpx.Response(200, json=body)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await mcp_tool_call(client, 'http://localhost:8000', 'get_status', {})

        assert result == {}

    async def test_non_json_body_returns_empty_dict(self):
        """HTTP 200 with non-JSON body (e.g. HTML from reverse proxy) returns empty dict."""
        from dashboard.data.memory import mcp_tool_call

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text='<html>Bad Gateway</html>')

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await mcp_tool_call(client, 'http://localhost:8000', 'get_status', {})

        assert result == {}


class TestMcpToolCallLogging:
    """Tests that mcp_tool_call emits WARNING-level logs on parse failures."""

    async def test_non_json_body_logs_warning(self, caplog):
        """HTTP 200 with non-JSON body triggers resp.json() failure and logs a WARNING."""
        from dashboard.data.memory import mcp_tool_call

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text='<html>Bad Gateway</html>')

        transport = httpx.MockTransport(handler)
        with caplog.at_level(logging.WARNING, logger='dashboard.data.memory'):
            async with httpx.AsyncClient(transport=transport) as client:
                result = await mcp_tool_call(client, 'http://localhost:8000', 'get_status', {})

        assert result == {}
        assert any(
            r.levelno == logging.WARNING and 'dashboard.data.memory' in r.name
            for r in caplog.records
        ), f'Expected WARNING log from dashboard.data.memory, got: {caplog.records}'

    async def test_invalid_inner_json_logs_warning(self, caplog):
        """Valid outer JSON but inner text is not JSON triggers json.loads failure and logs a WARNING."""
        from dashboard.data.memory import mcp_tool_call

        def handler(request: httpx.Request) -> httpx.Response:
            body = {
                'jsonrpc': '2.0',
                'id': 1,
                'result': {
                    'content': [{'type': 'text', 'text': 'not valid json!!!'}],
                },
            }
            return httpx.Response(200, json=body)

        transport = httpx.MockTransport(handler)
        with caplog.at_level(logging.WARNING, logger='dashboard.data.memory'):
            async with httpx.AsyncClient(transport=transport) as client:
                result = await mcp_tool_call(client, 'http://localhost:8000', 'get_status', {})

        assert result == {}
        assert any(
            r.levelno == logging.WARNING and 'dashboard.data.memory' in r.name
            for r in caplog.records
        ), f'Expected WARNING log from dashboard.data.memory, got: {caplog.records}'
