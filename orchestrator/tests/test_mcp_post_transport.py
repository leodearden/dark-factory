"""Transport-level regression for the five raw MCP POSTs (task 4023).

WHY THESE TESTS LOOK DIFFERENT FROM THE ONES THAT MISSED THE BUG.  The
pre-existing coverage mocks ``httpx.AsyncClient.post`` and asserts the payload
SHAPE — it pins that a call was made, never where it landed.  That is exactly
why ~3 months of silently-discarded memory writes were invisible: a
payload-shaped mock cannot tell a delivered POST from one absorbed by a
redirect.

So these tests drive a REAL ``httpx.AsyncClient`` through an
``httpx.MockTransport`` whose handler reproduces the live fused-memory server,
and assert the resulting STATE: which path the server actually received a
``tools/call`` on.  ``MockTransport`` keeps httpx's real redirect semantics, so
``follow_redirects`` genuinely matters, while needing no socket.

The handler reproduces BOTH live rejections, measured by curl against
127.0.0.1:8002 while writing this (see escalation esc-4023-2):

  * ``POST /mcp/``            -> 307, Location: /mcp   (trailing slash)
  * ``POST /mcp`` w/o Accept  -> 406 "Client must accept application/json"
  * ``POST /mcp`` w/ Accept   -> 200 + a real JSON-RPC result

Both rejections must be reproduced, because fixing only the slash moves the
failure from a silent 307 no-op to a silent 406 no-op — still a lost write.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from orchestrator.workflow import TaskWorkflow

#: Captured BEFORE any patching so the factory below can build a real client
#: without recursing into its own patch.
_REAL_ASYNC_CLIENT = httpx.AsyncClient

MCP_PATH = '/mcp'
MCP_PATH_WITH_SLASH = '/mcp/'


class RecordingMcpServer:
    """``httpx.MockTransport`` handler reproducing the live fused-memory server."""

    def __init__(self):
        #: (path, json_body) for every request the server actually PROCESSED.
        self.delivered: list[tuple[str, dict]] = []
        #: Requests refused with a 307 because of the trailing slash.
        self.redirected: list[str] = []
        #: Requests refused with a 406 because the Accept header was missing.
        self.not_acceptable: list[str] = []
        #: (path, Accept) for EVERY request that arrived, including the ones
        #: refused above.  Recorded pre-triage so the Accept assertion stays
        #: non-vacuous even while the slash defect still absorbs the request.
        self.seen: list[tuple[str, str]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.seen.append((path, request.headers.get('accept', '')))

        if path == MCP_PATH_WITH_SLASH:
            self.redirected.append(path)
            return httpx.Response(
                307,
                headers={'Location': str(request.url.copy_with(path=MCP_PATH))},
            )

        if path != MCP_PATH:
            return httpx.Response(404, text=f'no route for {path}')

        # The live server refuses a POST that does not accept application/json.
        if 'application/json' not in request.headers.get('accept', ''):
            self.not_acceptable.append(path)
            return httpx.Response(
                406,
                json={
                    'jsonrpc': '2.0',
                    'id': 'server-error',
                    'error': {
                        'code': -32600,
                        'message': 'Not Acceptable: Client must accept application/json',
                    },
                },
            )

        body = json.loads(request.content)
        self.delivered.append((path, body))
        return httpx.Response(
            200,
            json={'jsonrpc': '2.0', 'id': 1, 'result': {'content': []}},
        )

    # -- assertions helpers -------------------------------------------------

    def tool_calls(self, tool_name: str) -> list[dict]:
        """Bodies of every DELIVERED ``tools/call`` for *tool_name*."""
        return [
            body for _path, body in self.delivered
            if body.get('method') == 'tools/call'
            and body.get('params', {}).get('name') == tool_name
        ]

    def assert_every_request_accepted_json(self):
        """Every request that ARRIVED must have declared it accepts JSON.

        Checks ``seen`` rather than ``not_acceptable`` so this stays a real
        assertion even before the slash fix lands: a request absorbed by the
        307 never reaches the 406 branch, so an empty ``not_acceptable`` would
        pass vacuously.
        """
        assert self.seen, 'no request reached the server at all'
        bad = [(p, a) for p, a in self.seen if 'application/json' not in a]
        assert bad == [], (
            'every MCP POST must send Accept: application/json (the live server '
            f'answers 406 without it); these did not: {bad!r}'
        )


class RecordingClientFactory:
    """Stands in for ``httpx.AsyncClient``: records ctor kwargs, injects transport.

    Uses the same ``patch('httpx.AsyncClient', lambda *a, **k: ...)``
    constructor seam that ``test_workflow_completion_memory.py`` already relies
    on — widened to RECORD the kwargs, which is what makes the
    ``follow_redirects=True`` assertion possible at all.
    """

    def __init__(self, server: RecordingMcpServer):
        self._server = server
        #: One dict per ``httpx.AsyncClient(...)`` construction.
        self.ctor_kwargs: list[dict] = []

    def __call__(self, *args, **kwargs):
        self.ctor_kwargs.append(dict(kwargs))
        kwargs.setdefault('transport', httpx.MockTransport(self._server.handler))
        return _REAL_ASYNC_CLIENT(*args, **kwargs)

    def assert_follows_redirects(self):
        assert self.ctor_kwargs, 'no httpx.AsyncClient was constructed at all'
        for kwargs in self.ctor_kwargs:
            assert kwargs.get('follow_redirects') is True, (
                'the MCP client must be constructed with follow_redirects=True '
                f'(defence in depth against a re-introduced slash); got {kwargs!r}'
            )


@pytest.fixture
def server():
    return RecordingMcpServer()


@pytest.fixture
def client_factory(server):
    return RecordingClientFactory(server)


def make_workflow(*, tmp_path, task_id='4023', plan=None, mcp_url='http://memory.test:8002'):
    """Minimal TaskWorkflow carrying only what the memory writers read.

    ``object.__new__`` skips __init__ (same lightweight pattern as
    test_workflow_completion_memory.py) so no worktree/git/scheduler wiring is
    needed.
    """
    wf = object.__new__(TaskWorkflow)
    wf.mcp = MagicMock()
    wf.mcp.url = mcp_url
    wf.config = MagicMock()
    wf.config.fused_memory.project_id = 'dark_factory'
    wf.task = {'id': task_id, 'title': 'a task', 'description': 'does a thing'}
    wf.plan = plan if plan is not None else {
        'analysis': 'because reasons',
        'design_decisions': [],
        'steps': [{'status': 'done'}, {'status': 'pending'}],
    }
    wf.modules = ['orchestrator/src/orchestrator']
    wf.task_id = task_id
    wf.worktree = tmp_path
    return wf


# ---------------------------------------------------------------------------
# _write_completion_to_memory (step-5 RED / step-6 GREEN)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_completion_write_actually_reaches_the_mcp_endpoint(
    tmp_path, server, client_factory,
):
    """THE regression: the add_memory tools/call must land at /mcp.

    Under the pre-fix code the client targets /mcp/ with follow_redirects
    unset, so the server records ZERO delivered tools/call — while the HTTP
    exchange itself reports a perfectly successful 307.
    """
    wf = make_workflow(tmp_path=tmp_path)

    with patch('httpx.AsyncClient', client_factory):
        await wf._write_completion_to_memory()

    add_memory_calls = server.tool_calls('add_memory')
    assert len(add_memory_calls) == 1, (
        'expected exactly one add_memory tools/call DELIVERED at /mcp; got '
        f'{len(add_memory_calls)}. redirected={server.redirected!r} '
        f'not_acceptable={server.not_acceptable!r}'
    )
    assert all(path == MCP_PATH for path, _ in server.delivered)


@pytest.mark.asyncio
async def test_completion_write_never_targets_the_redirecting_slash_path(
    tmp_path, server, client_factory,
):
    """No request may hit /mcp/ at all — the slash is the load-bearing defect."""
    wf = make_workflow(tmp_path=tmp_path)

    with patch('httpx.AsyncClient', client_factory):
        await wf._write_completion_to_memory()

    assert server.redirected == [], (
        f'POSTed to the redirecting {MCP_PATH_WITH_SLASH} path: {server.redirected!r}'
    )


@pytest.mark.asyncio
async def test_completion_write_client_is_constructed_with_follow_redirects(
    tmp_path, server, client_factory,
):
    """Task's explicit ask (a): the client must opt into following redirects."""
    wf = make_workflow(tmp_path=tmp_path)

    with patch('httpx.AsyncClient', client_factory):
        await wf._write_completion_to_memory()

    client_factory.assert_follows_redirects()


@pytest.mark.asyncio
async def test_completion_write_sends_the_mcp_accept_header(
    tmp_path, server, client_factory,
):
    """The fourth part of the fix (esc-4023-2), measured against the live server.

    Without ``Accept: application/json``, a POST that reaches /mcp is refused
    406 — so dropping the slash alone would merely relocate the silent loss.
    """
    wf = make_workflow(tmp_path=tmp_path)

    with patch('httpx.AsyncClient', client_factory):
        await wf._write_completion_to_memory()

    server.assert_every_request_accepted_json()
    assert server.not_acceptable == [], (
        'the POST was refused 406 for a missing/!json Accept header — the write '
        'still did not land'
    )
