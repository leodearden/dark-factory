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


def _make_stateful_handler(calls: list[dict]):
    """Build a MockTransport handler emulating a STATEFUL MCP server.

    Faithful to the empirically-observed FastMCP 3.2.2 contract:
      - 'initialize' carrying an 'mcp-session-id' header -> 404 "Session not found"
      - 'initialize' with no session header -> 200 + assigns SERVER_SID via the
        'mcp-session-id' response header
      - any other request with a missing/mismatching sid -> 404 "Session not found"
      - 'notifications/initialized' with the right sid -> 202 empty body
      - 'tools/call' with the right sid -> 200 JSON-RPC result

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
