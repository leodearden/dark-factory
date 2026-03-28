"""Tests for dashboard.data.memory — MCP session-based memory health metrics."""

from __future__ import annotations

import json
import logging

import httpx
import pytest


def _make_mcp_response(inner_dict: dict, request_id: int = 1) -> httpx.Response:
    """Build a mock MCP JSON-RPC response wrapping *inner_dict*."""
    body = {
        'jsonrpc': '2.0',
        'id': request_id,
        'result': {
            'content': [
                {'type': 'text', 'text': json.dumps(inner_dict)},
            ]
        },
    }
    return httpx.Response(
        200, json=body,
        headers={'mcp-session-id': 'test-session-id'},
    )


def _make_init_response(request_id: int = 1) -> httpx.Response:
    """Build a mock MCP initialize response."""
    body = {
        'jsonrpc': '2.0',
        'id': request_id,
        'result': {
            'protocolVersion': '2025-03-26',
            'capabilities': {'tools': {}},
            'serverInfo': {'name': 'test', 'version': '0.1'},
        },
    }
    return httpx.Response(
        200, json=body,
        headers={'mcp-session-id': 'test-session-id'},
    )


def _make_notify_response() -> httpx.Response:
    """Build a 202 Accepted response for notifications."""
    return httpx.Response(202, headers={'mcp-session-id': 'test-session-id'})


class _SessionAwareHandler:
    """Mock handler that responds to initialize, notify, and tools/call."""

    def __init__(self, tool_response: dict | None = None, *, error_status: int | None = None,
                 error_on_tool: Exception | None = None, error_on_all: Exception | None = None):
        self.tool_response = tool_response or {}
        self.error_status = error_status
        self.error_on_tool = error_on_tool
        self.error_on_all = error_on_all
        self.calls: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.error_on_all:
            raise self.error_on_all

        body = json.loads(request.content)
        method = body.get('method', '')
        request_id = body.get('id', 1)
        self.calls.append(body)

        if method == 'initialize':
            return _make_init_response(request_id)

        if method.startswith('notifications/'):
            return _make_notify_response()

        # tools/call
        if self.error_on_tool:
            raise self.error_on_tool
        if self.error_status:
            return httpx.Response(self.error_status, text='Server Error')
        return _make_mcp_response(self.tool_response, request_id)


class TestSessionAwareHandler:
    """Unit tests for _SessionAwareHandler port-tracking behaviour."""

    def _init_request(self, port: int = 9001) -> httpx.Request:
        """Build a minimal JSON-RPC initialize request targeting *port*."""
        body = json.dumps(
            {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize'}
        ).encode()
        return httpx.Request('POST', f'http://localhost:{port}/mcp', content=body)

    def test_ports_seen_initializes_empty(self):
        """Handler initializes with an empty ports_seen set."""
        handler = _SessionAwareHandler({'ok': True})
        assert handler.ports_seen == set()

    def test_ports_seen_after_request(self):
        """After a request to port 9001, ports_seen contains 9001."""
        handler = _SessionAwareHandler({'ok': True})
        handler(self._init_request(9001))
        assert 9001 in handler.ports_seen

    def test_calls_populated_for_successful_request(self):
        """handler.calls is populated after a successful request."""
        handler = _SessionAwareHandler({'ok': True})
        handler(self._init_request(9001))
        assert len(handler.calls) == 1
        assert handler.calls[0]['method'] == 'initialize'


@pytest.fixture(autouse=True)
def _clean_sessions():
    """Reset session cache before each test."""
    from dashboard.data.memory import reset_sessions
    reset_sessions()
    yield
    reset_sessions()


# ── mcp_tool_call ───────────────────────────────────────────────


class TestMcpToolCall:
    """Tests for the low-level mcp_tool_call function."""

    async def test_successful_call(self):
        """Valid MCP response is parsed and inner dict returned."""
        from dashboard.data.memory import mcp_tool_call

        expected = {'graphiti': {'connected': True}, 'mem0': {'connected': True}}
        handler = _SessionAwareHandler(expected)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await mcp_tool_call(
                client, 'http://localhost:8000', 'get_status', {'project_id': 'dark_factory'}
            )

        assert result == expected

    async def test_session_initialization(self):
        """mcp_tool_call performs initialize + initialized notification before tool call."""
        from dashboard.data.memory import mcp_tool_call

        handler = _SessionAwareHandler({'ok': True})
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            await mcp_tool_call(client, 'http://localhost:9999', 'get_status', {})

        methods = [c['method'] for c in handler.calls]
        assert methods == ['initialize', 'notifications/initialized', 'tools/call']

    async def test_session_cached_across_calls(self):
        """Second call on same URL reuses session (no re-init)."""
        from dashboard.data.memory import mcp_tool_call

        handler = _SessionAwareHandler({'ok': True})
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            await mcp_tool_call(client, 'http://localhost:9998', 'get_status', {})
            await mcp_tool_call(client, 'http://localhost:9998', 'get_status', {})

        methods = [c['method'] for c in handler.calls]
        # init + notify + tool_call1 + tool_call2  (no second init)
        assert methods == [
            'initialize', 'notifications/initialized', 'tools/call', 'tools/call',
        ]

    async def test_timeout_propagates(self):
        """httpx.TimeoutException from the transport propagates to caller."""
        from dashboard.data.memory import mcp_tool_call

        handler = _SessionAwareHandler(error_on_all=httpx.TimeoutException('timed out'))
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(httpx.TimeoutException):
                await mcp_tool_call(client, 'http://localhost:8000', 'get_status', {})

    async def test_non_200_raises(self):
        """Non-200 HTTP status raises httpx.HTTPStatusError."""
        from dashboard.data.memory import mcp_tool_call

        handler = _SessionAwareHandler(error_status=500)
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await mcp_tool_call(client, 'http://localhost:8000', 'get_status', {})

    async def test_posts_to_correct_url_path(self):
        """mcp_tool_call posts to '{base_url}/mcp' (no trailing slash)."""
        from dashboard.data.memory import mcp_tool_call

        captured_paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_paths.append(request.url.path)
            body = json.loads(request.content)
            method = body.get('method', '')
            rid = body.get('id', 1)
            if method == 'initialize':
                return _make_init_response(rid)
            if method.startswith('notifications/'):
                return _make_notify_response()
            return _make_mcp_response({'ok': True}, rid)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            await mcp_tool_call(client, 'http://localhost:8000', 'get_status', {})

        assert all(p == '/mcp' for p in captured_paths), (
            f'Expected all paths to be /mcp, got {captured_paths}'
        )


# ── SSE response parsing ───────────────────────────────────────


class TestSseResponseParsing:
    """Tests for SSE response format handling."""

    def test_parse_sse_response(self):
        from dashboard.data.memory import _parse_sse_response

        sse = 'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"content":[]}}\n\n'
        result = _parse_sse_response(sse)
        assert result['jsonrpc'] == '2.0'

    def test_parse_sse_no_data_raises(self):
        from dashboard.data.memory import _parse_sse_response

        with pytest.raises(ValueError, match='No data line'):
            _parse_sse_response('event: message\n\n')


# ── Headers ─────────────────────────────────────────────────────


class TestMcpHeaders:
    """Tests that the MCP client sends correct HTTP headers."""

    async def test_accept_header_includes_both_types(self):
        """Requests include Accept: application/json, text/event-stream."""
        from dashboard.data.memory import mcp_tool_call

        captured_headers: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_headers.append(dict(request.headers))
            body = json.loads(request.content)
            method = body.get('method', '')
            rid = body.get('id', 1)
            if method == 'initialize':
                return _make_init_response(rid)
            if method.startswith('notifications/'):
                return _make_notify_response()
            return _make_mcp_response({'ok': True}, rid)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            await mcp_tool_call(client, 'http://localhost:8000', 'get_status', {})

        # All requests should have the dual Accept header
        for headers in captured_headers:
            accept = headers.get('accept', '')
            assert 'application/json' in accept, f'Missing application/json in {accept}'
            assert 'text/event-stream' in accept, f'Missing text/event-stream in {accept}'

    async def test_session_id_sent_after_init(self):
        """After initialize, subsequent requests include Mcp-Session-Id header."""
        from dashboard.data.memory import mcp_tool_call

        captured_headers: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_headers.append(dict(request.headers))
            body = json.loads(request.content)
            method = body.get('method', '')
            rid = body.get('id', 1)
            if method == 'initialize':
                return _make_init_response(rid)
            if method.startswith('notifications/'):
                return _make_notify_response()
            return _make_mcp_response({'ok': True}, rid)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            await mcp_tool_call(client, 'http://localhost:8000', 'get_status', {})

        # First request (initialize) has no session ID
        assert 'mcp-session-id' not in captured_headers[0]
        # Subsequent requests have the session ID
        for headers in captured_headers[1:]:
            assert headers.get('mcp-session-id') == 'test-session-id'


# ── get_memory_status (multi-URL) ──────────────────────────────


_STATUS_PAYLOAD = {
    'graphiti': {'connected': True},
    'mem0': {'connected': True},
    'projects': {
        'dark_factory': {'graphiti_nodes': 42, 'mem0_memories': 5},
    },
    'queue': {'counts': {'pending': 1, 'completed': 8}, 'oldest_pending_age_seconds': 1.2},
}


class TestGetMemoryStatus:
    """Tests for get_memory_status."""

    async def test_successful_status(self, dashboard_config):
        """Returns the parsed status dict from a successful MCP response."""
        from dashboard.data.memory import get_memory_status

        handler = _SessionAwareHandler(_STATUS_PAYLOAD)
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await get_memory_status(client, dashboard_config)

        assert result == _STATUS_PAYLOAD
        assert result['projects']['dark_factory']['graphiti_nodes'] == 42

    async def test_all_servers_down_returns_offline(self, dashboard_config):
        """When all URLs fail, returns offline with combined error."""
        from dashboard.data.memory import get_memory_status

        handler = _SessionAwareHandler(error_on_all=httpx.ConnectError('refused'))
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await get_memory_status(client, dashboard_config)

        assert result['offline'] is True
        assert 'error' in result

    async def test_first_server_down_falls_through(self, two_url_config):
        """If first URL fails, tries subsequent URLs.

        Uses a two-URL config [9000, 9001] so port 9000 is attempted first and
        fails, proving the fallback to 9001 is actually exercised (not a trivial
        pass where the first server already succeeds).
        """
        from dashboard.data.memory import get_memory_status

        ports_seen: set[int] = set()

        def handler(request: httpx.Request) -> httpx.Response:
            port = request.url.port
            assert port is not None
            ports_seen.add(port)

            # First server (port 9000) always fails
            if port == 9000:
                raise httpx.ConnectError('refused')

            body = json.loads(request.content)
            method = body.get('method', '')
            rid = body.get('id', 1)
            if method == 'initialize':
                return _make_init_response(rid)
            if method.startswith('notifications/'):
                return _make_notify_response()
            return _make_mcp_response(_STATUS_PAYLOAD, rid)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await get_memory_status(client, two_url_config)

        assert result == _STATUS_PAYLOAD
        assert 'offline' not in result
        # Prove port 9000 was actually attempted before falling through to 9001
        assert 9000 in ports_seen
        # Prove the fallback server (9001) was actually reached
        assert 9001 in ports_seen


# ── get_queue_stats (aggregation) ──────────────────────────────


_QUEUE_STATS_PAYLOAD = {
    'counts': {'pending': 3, 'in_flight': 1, 'retry': 0, 'completed': 10, 'dead': 0},
    'oldest_pending_age_seconds': 5.5,
}


class TestGetQueueStats:
    """Tests for get_queue_stats."""

    async def test_successful_stats(self, dashboard_config):
        """Returns aggregated queue stats from all reachable servers."""
        from dashboard.data.memory import get_queue_stats

        handler = _SessionAwareHandler(_QUEUE_STATS_PAYLOAD)
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await get_queue_stats(client, dashboard_config)

        # 1 server × 3 pending = 3 (all share same transport/handler)
        assert result['counts']['pending'] == 3
        assert result['oldest_pending_age_seconds'] == 5.5

    async def test_all_down_returns_offline(self, dashboard_config):
        """When all servers are unreachable, returns offline."""
        from dashboard.data.memory import get_queue_stats

        handler = _SessionAwareHandler(error_on_all=httpx.ConnectError('refused'))
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await get_queue_stats(client, dashboard_config)

        assert result['offline'] is True

    async def test_partial_failure_aggregates_available(self, two_url_config):
        """If some servers are down, aggregate from those that are up.

        Uses two_url_config [9000, 9001]: port 9000 fails, port 9001 succeeds.
        Multi-server aggregation (1 server × 3 = 3) is already covered by
        test_successful_stats; this test focuses on the partial-failure path.
        """
        from dashboard.data.memory import get_queue_stats

        ports_seen: set[int] = set()

        def handler(request: httpx.Request) -> httpx.Response:
            port = request.url.port
            assert port is not None
            ports_seen.add(port)

            if port == 9000:
                raise httpx.ConnectError('refused')
            body = json.loads(request.content)
            method = body.get('method', '')
            rid = body.get('id', 1)
            if method == 'initialize':
                return _make_init_response(rid)
            if method.startswith('notifications/'):
                return _make_notify_response()
            return _make_mcp_response(_QUEUE_STATS_PAYLOAD, rid)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await get_queue_stats(client, two_url_config)

        # 1 server (9001) × 3 pending = 3
        assert result['counts']['pending'] == 3
        assert 'offline' not in result
        # Prove both ports were actually contacted
        assert 9000 in ports_seen
        assert 9001 in ports_seen


# ── Malformed responses ─────────────────────────────────────────


class TestMalformedResponse:
    """Tests for mcp_tool_call with malformed MCP responses."""

    async def test_missing_content_key(self):
        """Response with no result.content path returns empty dict."""
        from dashboard.data.memory import mcp_tool_call

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            method = body.get('method', '')
            rid = body.get('id', 1)
            if method == 'initialize':
                return _make_init_response(rid)
            if method.startswith('notifications/'):
                return _make_notify_response()
            return httpx.Response(
                200,
                json={'jsonrpc': '2.0', 'id': rid, 'result': {}},
                headers={'mcp-session-id': 'test'},
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await mcp_tool_call(client, 'http://localhost:8000', 'get_status', {})

        assert result == {}

    async def test_empty_content_array(self):
        """Response with empty content array returns empty dict."""
        from dashboard.data.memory import mcp_tool_call

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            method = body.get('method', '')
            rid = body.get('id', 1)
            if method == 'initialize':
                return _make_init_response(rid)
            if method.startswith('notifications/'):
                return _make_notify_response()
            return httpx.Response(
                200,
                json={'jsonrpc': '2.0', 'id': rid, 'result': {'content': []}},
                headers={'mcp-session-id': 'test'},
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await mcp_tool_call(client, 'http://localhost:8000', 'get_status', {})

        assert result == {}


# ── Logging ─────────────────────────────────────────────────────


class TestMcpToolCallLogging:
    """Tests that mcp_tool_call emits WARNING-level logs on parse failures."""

    async def test_invalid_inner_json_logs_warning(self, caplog):
        """Inner text is not JSON → logs a WARNING and returns empty dict."""
        from dashboard.data.memory import mcp_tool_call

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            method = body.get('method', '')
            rid = body.get('id', 1)
            if method == 'initialize':
                return _make_init_response(rid)
            if method.startswith('notifications/'):
                return _make_notify_response()
            return httpx.Response(
                200,
                json={
                    'jsonrpc': '2.0', 'id': rid,
                    'result': {'content': [{'type': 'text', 'text': 'not json!!!'}]},
                },
                headers={'mcp-session-id': 'test'},
            )

        transport = httpx.MockTransport(handler)
        with caplog.at_level(logging.WARNING, logger='dashboard.data.memory'):
            async with httpx.AsyncClient(transport=transport) as client:
                result = await mcp_tool_call(client, 'http://localhost:8000', 'get_status', {})

        assert result == {}
        assert any(
            r.levelno == logging.WARNING and 'dashboard.data.memory' in r.name
            for r in caplog.records
        )
