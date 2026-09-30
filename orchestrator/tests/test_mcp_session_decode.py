"""McpSession decodes and addresses through shared.mcp_post (task 4819).

Task 4819 made ``McpSession`` consume ``shared.mcp_post``'s ``MCP_POST_HEADERS``,
``mcp_endpoint_url`` and ``decode_mcp_response_body``.  These tests drive the
session end to end through its public ``call_tool`` against a fake server, so
the shared primitives' coverage (``shared/tests/test_mcp_post.py``) provably
reaches the lifecycle path.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
from _mcp_transport_harness import RecordingClientFactory, RecordingMcpServer
from shared.mcp_post import MCP_POST_HEADERS

from orchestrator.mcp_lifecycle import McpSession

BASE_URL = 'http://memory.test:8002'
SSE = {'content-type': 'text/event-stream'}
RESULT = {'jsonrpc': '2.0', 'id': 2, 'result': {'content': [{'type': 'text', 'text': 'ok'}]}}


def _sse_body(payload: dict, *, prefix: str = 'data: ') -> bytes:
    return f'event: message\n{prefix}{json.dumps(payload)}\n\n'.encode()


class _ScriptedSessionServer(RecordingMcpServer):
    """The live-server fidelity of the harness, plus a scripted session handshake."""

    def __init__(self, *, tool_status: int = 200, tool_headers: dict, tool_body: bytes):
        super().__init__()
        self._tool_status = tool_status
        self._tool_headers = tool_headers
        self._tool_body = tool_body

    def handler(self, request: httpx.Request) -> httpx.Response:
        triaged = super().handler(request)
        if triaged.status_code != 200:
            return triaged
        method = json.loads(request.content).get('method')
        if method == 'initialize':
            return httpx.Response(
                200,
                json={'jsonrpc': '2.0', 'id': 1, 'result': {}},
                headers={'mcp-session-id': 's-4819'},
            )
        if method == 'notifications/initialized':
            return httpx.Response(202)
        return httpx.Response(
            self._tool_status, headers=self._tool_headers, content=self._tool_body,
        )

    def delivered_methods(self) -> list[str]:
        return [body.get('method') for _path, body in self.delivered]


async def _call_tool(server: RecordingMcpServer, base_url: str = BASE_URL) -> dict:
    with patch('httpx.AsyncClient', RecordingClientFactory(server)):
        return await McpSession(base_url).call_tool('get_status', {})


@pytest.mark.asyncio
@pytest.mark.parametrize('prefix', ['data: ', 'data:'], ids=['spaced', 'tight'])
async def test_call_tool_decodes_an_sse_framed_result(prefix):
    """An SSE-framed tools/call answer decodes to its JSON-RPC payload."""
    server = _ScriptedSessionServer(tool_headers=SSE, tool_body=_sse_body(RESULT, prefix=prefix))

    assert await _call_tool(server) == RESULT


@pytest.mark.asyncio
async def test_call_tool_decodes_an_sse_body_mislabelled_as_json():
    """An SSE body labelled application/json falls back to SSE instead of raising."""
    server = _ScriptedSessionServer(
        tool_headers={'content-type': 'application/json'}, tool_body=_sse_body(RESULT),
    )

    assert await _call_tool(server) == RESULT


@pytest.mark.asyncio
async def test_call_tool_raises_value_error_on_an_sse_body_with_no_data_line():
    """McpSession is not fire-and-forget: an undecodable answer propagates."""
    server = _ScriptedSessionServer(
        tool_headers=SSE, tool_body=b'event: ping\nretry: 3000\n\n',
    )

    with pytest.raises(ValueError):
        await _call_tool(server)


@pytest.mark.asyncio
async def test_every_session_request_sends_the_canonical_headers_to_the_slashless_endpoint():
    """Handshake, notification and tool call all hit /mcp with the canonical Accept."""
    server = _ScriptedSessionServer(tool_headers=SSE, tool_body=_sse_body(RESULT))
    factory = RecordingClientFactory(server)

    with patch('httpx.AsyncClient', factory):
        await McpSession(f'{BASE_URL}/').call_tool('get_status', {})

    assert server.redirected == []
    assert [path for path, _accept in server.seen] == ['/mcp'] * len(server.seen)
    assert [accept for _path, accept in server.seen] == (
        [MCP_POST_HEADERS['Accept']] * len(server.seen)
    )
    server.assert_every_request_accepted_json()
    factory.assert_follows_redirects()
    assert server.delivered_methods() == [
        'initialize', 'notifications/initialized', 'tools/call',
    ]
