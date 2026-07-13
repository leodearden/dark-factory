"""Tests for scripts/cgl_eta_scheduler_gate.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution -- mirrors the pattern in
test_investigate_cross_graph_duplication.py / test_migrate_cross_graph_leak.py.

Exercises McpClient's MCP streamable-HTTP session handshake against an
injected httpx.MockTransport that faithfully emulates a STATEFUL escalation
server's session contract (empirically reproduced against the installed
FastMCP 3.2.2 http_app()): `initialize` must carry NO session id, the server
assigns one via the `mcp-session-id` response header, and every subsequent
request must echo that id back or get HTTP 404 "Session not found". No live
server or orchestrator harness required.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import httpx
import pytest

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'cgl_eta_scheduler_gate.py'

SERVER_SID = 'srv-sid-123'


def _load_module() -> types.ModuleType:
    """Load cgl_eta_scheduler_gate.py from its file path.

    The module is registered in sys.modules under its name so that
    reflection-based decorators work correctly.
    """
    mod_name = 'cgl_eta_scheduler_gate'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()


def _session_not_found() -> httpx.Response:
    """The exact 404 body a stateful FastMCP 3.2.2 server returns for an
    initialize-with-unknown-session-id or any request with a missing/unknown
    session id."""
    return httpx.Response(
        404,
        json={
            'jsonrpc': '2.0', 'id': 'server-error',
            'error': {'code': -32600, 'message': 'Session not found'},
        },
    )


def _make_stateful_handler(calls: list[dict], tool_call_response=None):
    """Build a MockTransport handler emulating a STATEFUL MCP server.

    Faithful to the empirically-observed FastMCP 3.2.2 contract:
      - 'initialize' carrying an 'mcp-session-id' header -> 404 "Session not found"
      - 'initialize' with no session header -> 200 + assigns SERVER_SID via the
        'mcp-session-id' response header
      - any other request with a missing/mismatching sid -> 404 "Session not found"
      - 'notifications/initialized' with the right sid -> 202 empty body
      - 'tools/call' with the right sid -> 200 JSON-RPC result, unless
        *tool_call_response* is given, in which case it is called with the
        parsed request body and its return value is used as the response
        (lets callers exercise error / SSE / malformed response branches
        without duplicating the session-handshake plumbing).

    Records every request (path, headers, parsed body) to *calls*.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        method = body.get('method', '')
        sid = request.headers.get('mcp-session-id')
        calls.append({'path': request.url.path, 'headers': dict(request.headers), 'body': body})

        if method == 'initialize':
            if sid:
                return _session_not_found()
            return httpx.Response(
                200,
                json={
                    'jsonrpc': '2.0', 'id': body.get('id', 1),
                    'result': {
                        'protocolVersion': '2024-11-05',
                        'capabilities': {},
                        'serverInfo': {'name': 'escalation', 'version': '1.0'},
                    },
                },
                headers={'mcp-session-id': SERVER_SID},
            )

        if sid != SERVER_SID:
            return _session_not_found()

        if method == 'notifications/initialized':
            return httpx.Response(202)

        if method == 'tools/call':
            if tool_call_response is not None:
                return tool_call_response(body)
            return httpx.Response(
                200,
                json={
                    'jsonrpc': '2.0', 'id': body.get('id'),
                    'result': {'structuredContent': {'resumed': True, 'was_paused': False}},
                },
            )

        raise AssertionError(f'unexpected method in mock handler: {method}')

    return handler


class TestSessionHandshake:
    """McpClient must perform the correct MCP streamable-HTTP handshake
    against a STATEFUL server: no session id on initialize, then read+reuse
    the server-assigned mcp-session-id on every later request."""

    @pytest.mark.asyncio
    async def test_handshake_reads_and_reuses_server_session_id(self):
        calls: list[dict] = []
        transport = httpx.MockTransport(_make_stateful_handler(calls))

        client = _mod.McpClient('http://127.0.0.1:8102', transport=transport)
        async with client:
            res = await client.call_tool('resume_scheduler', {'reason': 'x'})

        assert res == {'resumed': True, 'was_paused': False}

        init_calls = [c for c in calls if c['body'].get('method') == 'initialize']
        assert len(init_calls) == 1
        assert not init_calls[0]['headers'].get('mcp-session-id')

        notify_calls = [
            c for c in calls if c['body'].get('method') == 'notifications/initialized'
        ]
        assert len(notify_calls) == 1
        assert notify_calls[0]['headers'].get('mcp-session-id') == SERVER_SID

        tool_calls = [c for c in calls if c['body'].get('method') == 'tools/call']
        assert len(tool_calls) == 1
        assert tool_calls[0]['headers'].get('mcp-session-id') == SERVER_SID

    @pytest.mark.asyncio
    async def test_initialize_with_invented_session_id_is_rejected(self):
        """Regression pin for the original bug: a client that (incorrectly)
        invents and sends its own session id on `initialize` must be
        rejected by a stateful server with 404 "Session not found", and
        that failure must surface as `raise_for_status()` raising. The
        fixed McpClient never does this (see the test above) -- this test
        forces the pre-fix behavior by seeding `_session_id` before the
        handshake, so the mock's 404-on-initialize-with-sid branch is
        actually exercised and its consequence is pinned.
        """
        calls: list[dict] = []
        transport = httpx.MockTransport(_make_stateful_handler(calls))

        client = _mod.McpClient('http://127.0.0.1:8102', transport=transport)
        client._session_id = 'invented-client-uuid'  # simulate the pre-fix bug

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            async with client:
                pass

        assert exc_info.value.response.status_code == 404
        init_calls = [c for c in calls if c['body'].get('method') == 'initialize']
        assert len(init_calls) == 1
        assert init_calls[0]['headers'].get('mcp-session-id') == 'invented-client-uuid'


class TestCanonicalPath:
    """McpClient must POST to the canonical '/mcp' path (no trailing slash),
    matching the orchestrator's McpSession and the dashboard client, and
    avoiding an extra 307-redirect round trip."""

    @pytest.mark.asyncio
    async def test_posts_to_canonical_mcp_path_without_trailing_slash(self):
        calls: list[dict] = []
        transport = httpx.MockTransport(_make_stateful_handler(calls))

        client = _mod.McpClient('http://127.0.0.1:8102', transport=transport)
        async with client:
            await client.call_tool('halt_scheduler', {'reason': 'x'})

        paths = [c['path'] for c in calls]
        assert paths, 'expected at least one request to be recorded'
        assert all(p == '/mcp' for p in paths), (
            f'Expected all paths to be /mcp, got {paths}'
        )


class TestSseResponse:
    """FastMCP streamable-HTTP servers routinely reply with
    text/event-stream instead of application/json; `_post` must parse the
    `data:` line of an SSE response and raise if none is present."""

    @pytest.mark.asyncio
    async def test_call_tool_parses_sse_response(self):
        def tool_call_response(body: dict) -> httpx.Response:
            payload = {
                'jsonrpc': '2.0', 'id': body.get('id'),
                'result': {'structuredContent': {'resumed': True, 'was_paused': False}},
            }
            return httpx.Response(
                200,
                content=f'event: message\ndata: {json.dumps(payload)}\n\n',
                headers={'content-type': 'text/event-stream'},
            )

        transport = httpx.MockTransport(
            _make_stateful_handler([], tool_call_response=tool_call_response)
        )
        client = _mod.McpClient('http://127.0.0.1:8102', transport=transport)
        async with client:
            res = await client.call_tool('resume_scheduler', {'reason': 'x'})

        assert res == {'resumed': True, 'was_paused': False}

    @pytest.mark.asyncio
    async def test_raises_on_sse_response_without_data_line(self):
        def tool_call_response(_body: dict) -> httpx.Response:
            return httpx.Response(
                200,
                content='event: message\n\n',
                headers={'content-type': 'text/event-stream'},
            )

        transport = httpx.MockTransport(
            _make_stateful_handler([], tool_call_response=tool_call_response)
        )
        client = _mod.McpClient('http://127.0.0.1:8102', transport=transport)
        async with client:
            with pytest.raises(RuntimeError, match='no SSE data line'):
                await client.call_tool('resume_scheduler', {'reason': 'x'})


class TestCallToolErrorHandling:
    """A JSON-RPC error result (still HTTP 200) must surface as a
    RuntimeError from `call_tool`, not a silently-wrong return value."""

    @pytest.mark.asyncio
    async def test_call_tool_raises_runtime_error_on_json_rpc_error(self):
        def tool_call_response(body: dict) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    'jsonrpc': '2.0', 'id': body.get('id'),
                    'error': {'code': -32000, 'message': 'boom'},
                },
            )

        transport = httpx.MockTransport(
            _make_stateful_handler([], tool_call_response=tool_call_response)
        )
        client = _mod.McpClient('http://127.0.0.1:8102', transport=transport)
        async with client:
            with pytest.raises(RuntimeError, match='resume_scheduler failed'):
                await client.call_tool('resume_scheduler', {'reason': 'x'})


class TestCallToolTextContentFallback:
    """When a tool result carries no `structuredContent`, `call_tool` must
    fall back to the `content` list: JSON-decode a `type == 'text'` entry's
    `text`, or wrap it as `{'_raw': ...}` if it isn't valid JSON. Real
    FastMCP tool responses can take this shape instead of returning
    `structuredContent` directly, so this fallback branch must be exercised
    on its own (not just implied by the structuredContent-only mock)."""

    @pytest.mark.asyncio
    async def test_call_tool_parses_json_text_content(self):
        def tool_call_response(body: dict) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    'jsonrpc': '2.0', 'id': body.get('id'),
                    'result': {
                        'content': [
                            {'type': 'text', 'text': json.dumps({'resumed': True, 'was_paused': False})},
                        ],
                    },
                },
            )

        transport = httpx.MockTransport(
            _make_stateful_handler([], tool_call_response=tool_call_response)
        )
        client = _mod.McpClient('http://127.0.0.1:8102', transport=transport)
        async with client:
            res = await client.call_tool('resume_scheduler', {'reason': 'x'})

        assert res == {'resumed': True, 'was_paused': False}

    @pytest.mark.asyncio
    async def test_call_tool_wraps_non_json_text_content_as_raw(self):
        def tool_call_response(body: dict) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    'jsonrpc': '2.0', 'id': body.get('id'),
                    'result': {
                        'content': [
                            {'type': 'text', 'text': 'not json'},
                        ],
                    },
                },
            )

        transport = httpx.MockTransport(
            _make_stateful_handler([], tool_call_response=tool_call_response)
        )
        client = _mod.McpClient('http://127.0.0.1:8102', transport=transport)
        async with client:
            res = await client.call_tool('resume_scheduler', {'reason': 'x'})

        assert res == {'_raw': 'not json'}


class TestOneBestEffort:
    """`_one` is the wrapper's best-effort, non-fatal entry point: a halt/
    resume miss (JSON-RPC error, unreachable/erroring server) must never
    raise out of `_one` -- it must be swallowed and reported as `False`,
    since a wrapper `trap` calls `resume` unconditionally on any exit path.
    """

    @pytest.mark.asyncio
    async def test_one_returns_false_on_json_rpc_tool_error(self, monkeypatch):
        real_mcp_client = _mod.McpClient  # capture before patching to avoid recursion

        def tool_call_response(body: dict) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    'jsonrpc': '2.0', 'id': body.get('id'),
                    'error': {'code': -32000, 'message': 'boom'},
                },
            )

        transport = httpx.MockTransport(
            _make_stateful_handler([], tool_call_response=tool_call_response)
        )
        monkeypatch.setattr(
            _mod, 'McpClient', lambda url, *a, **kw: real_mcp_client(url, transport=transport)
        )

        result = await _mod._one('dark_factory', 'http://127.0.0.1:8102', 'resume_scheduler', 'x')

        assert result is False

    @pytest.mark.asyncio
    async def test_one_returns_false_on_unreachable_transport(self, monkeypatch):
        real_mcp_client = _mod.McpClient  # capture before patching to avoid recursion

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError('connection refused', request=request)

        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(
            _mod, 'McpClient', lambda url, *a, **kw: real_mcp_client(url, transport=transport)
        )

        result = await _mod._one('dark_factory', 'http://127.0.0.1:8102', 'resume_scheduler', 'x')

        assert result is False

    @pytest.mark.asyncio
    async def test_one_returns_true_on_clean_resume(self, monkeypatch):
        """Happy path: a clean `resume_scheduler` apply reads the `resumed`
        key from the result and returns True."""
        real_mcp_client = _mod.McpClient  # capture before patching to avoid recursion

        # Default handler's tools/call response is {'resumed': True, 'was_paused': False}.
        transport = httpx.MockTransport(_make_stateful_handler([]))
        monkeypatch.setattr(
            _mod, 'McpClient', lambda url, *a, **kw: real_mcp_client(url, transport=transport)
        )

        result = await _mod._one('dark_factory', 'http://127.0.0.1:8102', 'resume_scheduler', 'x')

        assert result is True

    @pytest.mark.asyncio
    async def test_one_returns_true_on_clean_halt(self, monkeypatch):
        """Happy path: a clean `halt_scheduler` apply reads the `halted`
        key (not `resumed`) from the result and returns True."""
        real_mcp_client = _mod.McpClient  # capture before patching to avoid recursion

        def tool_call_response(body: dict) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    'jsonrpc': '2.0', 'id': body.get('id'),
                    'result': {'structuredContent': {'halted': True, 'was_paused': False}},
                },
            )

        transport = httpx.MockTransport(
            _make_stateful_handler([], tool_call_response=tool_call_response)
        )
        monkeypatch.setattr(
            _mod, 'McpClient', lambda url, *a, **kw: real_mcp_client(url, transport=transport)
        )

        result = await _mod._one('dark_factory', 'http://127.0.0.1:8102', 'halt_scheduler', 'x')

        assert result is True
