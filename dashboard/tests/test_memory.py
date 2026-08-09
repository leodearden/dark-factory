"""Tests for dashboard.data.memory — MCP session-based memory health metrics."""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from dashboard.data.memory import get_curator_state


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


def _cold_session_responses(
    inner: dict, url: str = 'http://localhost:8000',
) -> list[httpx.Response]:
    """The three responses a COLD McpSession consumes, in post order.

    ``mcp_tool_call`` against a cold session issues ``initialize``, then
    ``notifications/initialized``, then ``tools/call`` — three HTTP posts.
    An AsyncMock client bypasses MockTransport, which is what normally
    attaches ``.request``, so each response needs it set by hand or
    ``raise_for_status()`` raises RuntimeError even on a 200.
    """
    responses = [
        _make_init_response(),
        _make_notify_response(),
        _make_mcp_response(inner),
    ]
    for resp in responses:
        resp.request = httpx.Request('POST', f'{url.rstrip("/")}/mcp')
    return responses


class _SessionAwareHandler:
    """Mock handler that responds to initialize, notify, and tools/call."""

    def __init__(self, tool_response: dict | None = None, *, error_status: int | None = None,
                 error_on_tool: Exception | None = None, error_on_all: Exception | None = None,
                 fail_port: int | None = None):
        self.tool_response = tool_response or {}
        self.error_status = error_status
        self.error_on_tool = error_on_tool
        self.error_on_all = error_on_all
        self.fail_port = fail_port
        self.calls: list[dict] = []
        self.ports_seen: set[int] = set()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Dispatch a mock HTTP request.

        `ports_seen` records every attempted port, including those that raise
        ConnectError via `fail_port`. `calls` only records requests that pass
        all pre-dispatch guards (fail_port check, error_on_all check, and body
        parsing); it is therefore empty when an early-exit path fires.
        """
        port = request.url.port
        assert port is not None, f'Request to {request.url} has no port'
        self.ports_seen.add(port)

        if self.fail_port is not None and port == self.fail_port:
            raise httpx.ConnectError('refused')

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

    def test_fail_port_raises_connect_error(self):
        """Request to fail_port raises httpx.ConnectError."""
        handler = _SessionAwareHandler({'ok': True}, fail_port=9000)
        with pytest.raises(httpx.ConnectError):
            handler(self._init_request(9000))

    def test_fail_port_records_port_before_error(self):
        """Port is recorded in ports_seen even when ConnectError is raised."""
        handler = _SessionAwareHandler({'ok': True}, fail_port=9000)
        with pytest.raises(httpx.ConnectError):
            handler(self._init_request(9000))
        assert 9000 in handler.ports_seen
        assert len(handler.calls) == 0

    def test_fail_port_does_not_affect_other_ports(self):
        """Requests to ports other than fail_port succeed normally."""
        handler = _SessionAwareHandler({'ok': True}, fail_port=9000)
        response = handler(self._init_request(9001))
        assert response.status_code == 200
        assert 9001 in handler.ports_seen
        assert 9000 not in handler.ports_seen

    def test_portless_url_raises_assertion_error(self):
        """Request to a URL without an explicit port raises AssertionError."""
        handler = _SessionAwareHandler({'ok': True})
        body = json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize'}).encode()
        request = httpx.Request('POST', 'http://localhost/mcp', content=body)
        with pytest.raises(AssertionError):
            handler(request)


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


# ── client_op_id injection (twin of orchestrator McpSession) ────


class TestDashboardClientOpIdInjection:
    """The dashboard McpSession twin injects a client_op_id for mutating task
    tools, mirroring the orchestrator (task 2712). Inert today (the dashboard
    issues only reads) but keeps the two twins from diverging and is write-safe
    by construction if the dashboard ever gains a mutating call.
    """

    async def test_injects_client_op_id_for_mutating_tool(self):
        from dashboard.data.memory import McpSession

        session = McpSession('http://localhost:8000')
        resp = _make_mcp_response({'ok': True})
        # MockTransport normally attaches the request; set it here since we
        # bypass the transport and return the response from a mocked client.
        resp.request = httpx.Request('POST', 'http://localhost:8000/mcp')
        mock_client = AsyncMock()
        mock_client.post.return_value = resp

        await session._raw_call(
            mock_client, 'tools/call',
            {'name': 'update_task', 'arguments': {'id': '1'}},
        )

        posted = mock_client.post.call_args.kwargs['json']
        op_id = posted['params']['arguments'].get('client_op_id')
        assert isinstance(op_id, str) and op_id, (
            'a mutating tool call must get a non-empty client_op_id injected'
        )

    async def test_no_injection_for_read_tool(self):
        from dashboard.data.memory import McpSession

        session = McpSession('http://localhost:8000')
        resp = _make_mcp_response({'ok': True})
        resp.request = httpx.Request('POST', 'http://localhost:8000/mcp')
        mock_client = AsyncMock()
        mock_client.post.return_value = resp

        await session._raw_call(
            mock_client, 'tools/call',
            {'name': 'get_status', 'arguments': {}},
        )

        posted = mock_client.post.call_args.kwargs['json']
        assert 'client_op_id' not in posted['params']['arguments'], (
            'read tools must not get a client_op_id injected'
        )


# ── per-call timeout budget threading ──────────────────────────


class TestMcpToolCallTimeoutBudget:
    """A caller's per-call budget must reach EVERY post of the flow.

    ``timeout=`` on ``client.post`` bounds connect/read/write *and pool
    acquisition*. Without threading, a caller working to a 2.0s budget still
    waited up to httpx's 10s default for a pool slot on each of the three
    posts a cold session performs — the incoherence this closes. The
    ``notifications/initialized`` post is the subtlest of the three: it
    hardcoded 10 with no parameter at all.
    """

    async def test_budget_reaches_every_post_of_a_cold_session(self):
        from dashboard.data.memory import mcp_tool_call

        mock_client = AsyncMock()
        mock_client.post.side_effect = _cold_session_responses({'ok': True})

        result = await mcp_tool_call(
            mock_client, 'http://localhost:8000', 'get_status', {}, timeout=2.0,
        )

        assert result == {'ok': True}
        posts = mock_client.post.call_args_list
        assert len(posts) == 3, (
            f'cold session should post initialize + notify + tools/call, '
            f'got {len(posts)} posts'
        )
        timeouts = [c.kwargs['timeout'] for c in posts]
        assert timeouts == [2.0, 2.0, 2.0], (
            f'the caller budget must reach every post (including the notify, '
            f'which hardcoded 10), got {timeouts}'
        )

    async def test_default_timeout_is_unchanged_at_ten(self):
        """Guard: the default must stay 10 — nothing is silently raised."""
        from dashboard.data.memory import mcp_tool_call

        mock_client = AsyncMock()
        mock_client.post.side_effect = _cold_session_responses({'ok': True})

        await mcp_tool_call(mock_client, 'http://localhost:8000', 'get_status', {})

        timeouts = [c.kwargs['timeout'] for c in mock_client.post.call_args_list]
        assert timeouts == [10, 10, 10], (
            f'omitting timeout must keep the pre-existing 10s default on every '
            f'post, got {timeouts}'
        )


class TestAggregateTimeoutBudget:
    """The two multi-URL aggregates must thread a budget to every post.

    ``metrics.py`` wraps ``get_memory_status`` and ``get_queue_stats`` in
    ``asyncio.wait_for``, but those wrappers bound the *aggregate*: both
    functions walk N fused-memory URLs (``get_memory_status`` short-circuits
    on the first success, ``get_queue_stats`` visits all of them), so the
    outer bound alone left each individual post free to wait httpx's 10s
    default — including for a pool slot. The two layers are complementary,
    not redundant, so both must be present.
    """

    async def test_get_memory_status_threads_budget_across_failover(
        self, two_url_config,
    ):
        from dashboard.data.memory import get_memory_status

        url_a, url_b = two_url_config.fused_memory_urls
        mock_client = AsyncMock()
        # First URL's initialize post fails outright → fall through to the
        # second, which serves a full cold-session sequence.
        mock_client.post.side_effect = [
            httpx.ConnectError('refused'),
            *_cold_session_responses({'graphiti': {'connected': True}}, url_b),
        ]

        result = await get_memory_status(mock_client, two_url_config, timeout=3.0)

        assert result.get('offline') is not True, f'expected failover success: {result}'
        posts = mock_client.post.call_args_list
        assert len(posts) == 4, (
            f'expected 1 failed post on {url_a} + 3 cold-session posts on '
            f'{url_b}, got {len(posts)}'
        )
        timeouts = [c.kwargs['timeout'] for c in posts]
        assert timeouts == [3.0] * 4, (
            f'the budget must reach every post of every URL tried, got {timeouts}'
        )

    async def test_get_memory_status_default_timeout_is_ten(self, two_url_config):
        """Guard: omitting the budget keeps the pre-existing 10s default."""
        from dashboard.data.memory import get_memory_status

        _url_a, url_b = two_url_config.fused_memory_urls
        mock_client = AsyncMock()
        mock_client.post.side_effect = _cold_session_responses({'ok': True}, url_b)

        await get_memory_status(mock_client, two_url_config)

        timeouts = [c.kwargs['timeout'] for c in mock_client.post.call_args_list]
        assert timeouts == [10, 10, 10], f'default must stay 10, got {timeouts}'

    async def test_get_queue_stats_threads_budget_to_every_url(self, two_url_config):
        from dashboard.data.memory import get_queue_stats

        url_a, url_b = two_url_config.fused_memory_urls
        stats = {'counts': {'graphiti': 1}, 'oldest_pending_age_seconds': 2.0}
        mock_client = AsyncMock()
        # get_queue_stats visits ALL urls — two cold sessions, six posts.
        mock_client.post.side_effect = [
            *_cold_session_responses(stats, url_a),
            *_cold_session_responses(stats, url_b),
        ]

        result = await get_queue_stats(mock_client, two_url_config, timeout=3.0)

        assert result.get('offline') is not True, f'expected success: {result}'
        assert result['counts'] == {'graphiti': 2}, 'both urls must be summed'
        timeouts = [c.kwargs['timeout'] for c in mock_client.post.call_args_list]
        assert timeouts == [3.0] * 6, (
            f'the budget must reach every post of all N urls, got {timeouts}'
        )

    async def test_get_queue_stats_default_timeout_is_ten(self, two_url_config):
        """Guard: omitting the budget keeps the pre-existing 10s default."""
        from dashboard.data.memory import get_queue_stats

        url_a, url_b = two_url_config.fused_memory_urls
        stats = {'counts': {'graphiti': 1}, 'oldest_pending_age_seconds': 2.0}
        mock_client = AsyncMock()
        mock_client.post.side_effect = [
            *_cold_session_responses(stats, url_a),
            *_cold_session_responses(stats, url_b),
        ]

        await get_queue_stats(mock_client, two_url_config)

        timeouts = [c.kwargs['timeout'] for c in mock_client.post.call_args_list]
        assert timeouts == [10] * 6, f'default must stay 10, got {timeouts}'


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

        handler = _SessionAwareHandler(_STATUS_PAYLOAD, fail_port=9000)
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await get_memory_status(client, two_url_config)

        assert result == _STATUS_PAYLOAD
        assert 'offline' not in result
        # Prove port 9000 was actually attempted before falling through to 9001
        assert 9000 in handler.ports_seen
        # Prove the fallback server (9001) was actually reached
        assert 9001 in handler.ports_seen


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

        handler = _SessionAwareHandler(_QUEUE_STATS_PAYLOAD, fail_port=9000)
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await get_queue_stats(client, two_url_config)

        # 1 server (9001) × 3 pending = 3
        assert result['counts']['pending'] == 3
        assert 'offline' not in result
        # Prove both ports were actually contacted
        assert 9000 in handler.ports_seen
        assert 9001 in handler.ports_seen


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


# ── invalidate_session ──────────────────────────────────────────


class TestInvalidateSession:
    """Unit tests for the public invalidate_session helper.

    Each test relies on the module-scoped _clean_sessions autouse fixture
    (defined above) to guarantee a clean _sessions dict before and after.
    """

    def test_removes_cached_entry(self):
        """invalidate_session removes an existing entry keyed to the base URL."""
        from dashboard.data.memory import _get_session, _sessions, invalidate_session

        _get_session('http://x:8000')
        assert 'http://x:8000' in _sessions

        invalidate_session('http://x:8000')
        assert 'http://x:8000' not in _sessions

    def test_trailing_slash_normalized(self):
        """invalidate_session with trailing slash removes entry keyed without slash."""
        from dashboard.data.memory import _get_session, _sessions, invalidate_session

        # _get_session strips trailing slashes, so the key has none.
        _get_session('http://x:8000')
        assert 'http://x:8000' in _sessions

        # Caller passes URL with trailing slash — should still work.
        invalidate_session('http://x:8000/')
        assert 'http://x:8000' not in _sessions

    def test_unknown_url_is_no_op(self):
        """invalidate_session with a URL not in the cache does not raise."""
        from dashboard.data.memory import invalidate_session

        # Must not raise — idempotent by design.
        invalidate_session('http://unknown:9999')


# ── get_curator_state ───────────────────────────────────────────


def _make_test_config(*, fused_memory_urls: list[str], tmp_path=None):
    """Build a minimal DashboardConfig for get_curator_state tests."""
    from pathlib import Path

    from dashboard.config import DashboardConfig

    root = tmp_path or Path('/tmp/test-curator-state')
    return DashboardConfig(
        project_root=root,
        fused_memory_urls=fused_memory_urls,
    )


class TestGetCuratorState:
    """Tests for the get_curator_state helper in memory.py."""

    @pytest.mark.asyncio
    async def test_get_curator_state_returns_state_from_mcp(self):
        """Helper calls get_curator_state tool and returns the payload on success."""
        mcp_payload = {
            'paused': True,
            'paused_reason': 'all capped',
            'soonest_open_at': '2026-06-01T00:00:00+00:00',
            'account_count': 2,
        }

        config = _make_test_config(fused_memory_urls=['http://localhost:18765'])

        with patch('dashboard.data.memory.mcp_tool_call', new=AsyncMock(return_value=mcp_payload)):
            async with httpx.AsyncClient() as client:
                result = await get_curator_state(client, config)

        assert result == mcp_payload, f'Unexpected result: {result!r}'

    @pytest.mark.asyncio
    async def test_get_curator_state_returns_offline_dict_when_all_urls_fail(self):
        """Helper returns offline dict and calls mcp_tool_call once per URL on failure."""
        config = _make_test_config(
            fused_memory_urls=['http://localhost:18765', 'http://localhost:18766'],
        )

        with patch(
            'dashboard.data.memory.mcp_tool_call',
            new=AsyncMock(side_effect=httpx.ConnectError('refused')),
        ) as mock_call:
            async with httpx.AsyncClient() as client:
                result = await get_curator_state(client, config)

        assert result.get('offline') is True, f'Expected offline=True, got: {result!r}'
        assert 'error' in result, f'Expected error key, got: {result!r}'
        assert 'refused' in result['error'], f'Expected "refused" in error: {result["error"]!r}'
        assert mock_call.call_count == 2, (
            f'Expected mcp_tool_call called twice (once per URL), '
            f'got {mock_call.call_count}'
        )
