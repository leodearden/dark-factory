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


# ---------------------------------------------------------------------------
# _write_decisions_to_memory (step-7 RED / step-8 GREEN)
# ---------------------------------------------------------------------------


DECISIONS = [
    {'decision': 'use a shared primitive', 'rationale': 'five copies is how the drift arose'},
    {'decision': 'anchor on the URL form', 'rationale': 'bare /mcp/ also matches prose'},
    {'decision': 'warn, never raise', 'rationale': 'every call site is fire-and-forget'},
]


def _workflow_with_decisions(tmp_path, decisions=None):
    return make_workflow(
        tmp_path=tmp_path,
        plan={
            'analysis': 'because reasons',
            'design_decisions': DECISIONS if decisions is None else decisions,
            'steps': [{'status': 'done'}],
        },
    )


@pytest.mark.asyncio
async def test_decisions_write_delivers_every_decision_to_the_mcp_endpoint(
    tmp_path, server, client_factory,
):
    """One delivered add_memory per design decision, all landing at /mcp."""
    wf = _workflow_with_decisions(tmp_path)

    with patch('httpx.AsyncClient', client_factory):
        await wf._write_decisions_to_memory()

    calls = server.tool_calls('add_memory')
    assert len(calls) == len(DECISIONS), (
        f'expected {len(DECISIONS)} delivered add_memory calls, got {len(calls)}. '
        f'redirected={server.redirected!r} not_acceptable={server.not_acceptable!r}'
    )
    assert all(path == MCP_PATH for path, _ in server.delivered)
    assert server.redirected == []
    # The decision text must survive the trip, not merely the envelope.
    delivered_text = ' '.join(c['params']['arguments']['content'] for c in calls)
    for decision in DECISIONS:
        assert decision['decision'] in delivered_text


@pytest.mark.asyncio
async def test_decisions_write_uses_the_patched_client_seam(
    tmp_path, server, client_factory,
):
    """The inline ``__import__('httpx')`` site is reached by the same patch seam.

    ``__import__('httpx').AsyncClient`` resolves the attribute at call time, so
    ``patch('httpx.AsyncClient', ...)`` does intercept it — pinned here so a
    future rewrite of that import cannot quietly escape this file's coverage.
    """
    wf = _workflow_with_decisions(tmp_path)

    with patch('httpx.AsyncClient', client_factory):
        await wf._write_decisions_to_memory()

    assert client_factory.ctor_kwargs, (
        'the decisions writer constructed no client through the patched seam — '
        'its httpx import no longer goes through httpx.AsyncClient'
    )


@pytest.mark.asyncio
async def test_decisions_write_client_follows_redirects_and_accepts_json(
    tmp_path, server, client_factory,
):
    wf = _workflow_with_decisions(tmp_path)

    with patch('httpx.AsyncClient', client_factory):
        await wf._write_decisions_to_memory()

    client_factory.assert_follows_redirects()
    server.assert_every_request_accepted_json()


# ---------------------------------------------------------------------------
# _write_suggestions_to_memory (step-7 RED / step-8 GREEN)
# ---------------------------------------------------------------------------


def _reviews_with(n):
    """A ``reviews``-shaped stand-in carrying *n* suggestion dicts."""
    reviews = MagicMock()
    reviews.suggestions = [
        {'category': f'cat-{i}', 'description': f'suggestion number {i}'}
        for i in range(n)
    ]
    return reviews


@pytest.mark.asyncio
async def test_suggestions_write_delivers_to_the_mcp_endpoint(
    tmp_path, server, client_factory,
):
    wf = make_workflow(tmp_path=tmp_path)

    with patch('httpx.AsyncClient', client_factory):
        await wf._write_suggestions_to_memory(_reviews_with(3))

    calls = server.tool_calls('add_memory')
    assert len(calls) == 3, (
        f'expected 3 delivered add_memory calls, got {len(calls)}. '
        f'redirected={server.redirected!r} not_acceptable={server.not_acceptable!r}'
    )
    assert all(path == MCP_PATH for path, _ in server.delivered)
    assert server.redirected == []


@pytest.mark.asyncio
async def test_suggestions_write_caps_at_five_delivered_calls(
    tmp_path, server, client_factory,
):
    """The documented cap is 5 — and it must be a cap on DELIVERED writes."""
    wf = make_workflow(tmp_path=tmp_path)

    with patch('httpx.AsyncClient', client_factory):
        await wf._write_suggestions_to_memory(_reviews_with(9))

    assert len(server.tool_calls('add_memory')) == 5


@pytest.mark.asyncio
async def test_suggestions_write_client_follows_redirects_and_accepts_json(
    tmp_path, server, client_factory,
):
    wf = make_workflow(tmp_path=tmp_path)

    with patch('httpx.AsyncClient', client_factory):
        await wf._write_suggestions_to_memory(_reviews_with(2))

    client_factory.assert_follows_redirects()
    server.assert_every_request_accepted_json()


# ---------------------------------------------------------------------------
# MergeWorker._post_submit_tasks — the main-health mirror (step-11 / step-12)
# ---------------------------------------------------------------------------
#
# Placed beside the workflow cases deliberately: merge_queue.py's method is a
# documented COPY of TaskWorkflow._post_submit_tasks, and copies that are
# tested apart are exactly how these five sites drifted from the already-correct
# mcp_lifecycle.py pattern in the first place.


def _make_merge_worker(tmp_path, mcp_url='http://memory.test:8002'):
    import asyncio

    from orchestrator.git_ops import GitOps
    from orchestrator.merge_queue import SpeculativeMergeWorker

    # project_root must be wired: SpeculativeMergeWorker.__init__ reads it and
    # it is an instance attribute, so a spec-only mock does not carry it (see
    # test_merge_queue_auto_heal.py:_make_mock_git_ops).
    git_ops = MagicMock(spec=GitOps)
    git_ops.project_root = tmp_path

    worker = SpeculativeMergeWorker(git_ops=git_ops, queue=asyncio.Queue())
    worker._mcp = MagicMock()
    worker._mcp.url = mcp_url
    return worker


FIX_TASK_ARGS = {
    'title': 'main is red',
    'description': 'auto-heal fix task',
    'priority': 'high',
}


@pytest.mark.asyncio
async def test_merge_worker_submit_lands_at_the_mcp_endpoint(tmp_path, server, client_factory):
    """The main-health auto-heal fix task must actually reach the curator."""
    worker = _make_merge_worker(tmp_path)

    with patch('httpx.AsyncClient', client_factory):
        await worker._post_submit_tasks([FIX_TASK_ARGS])

    calls = server.tool_calls('submit_task')
    assert len(calls) == 1, (
        f'expected 1 submit_task delivered at {MCP_PATH}, got {len(calls)}. '
        f'redirected={server.redirected!r} not_acceptable={server.not_acceptable!r}'
    )
    assert calls[0]['params']['arguments'] == FIX_TASK_ARGS
    assert server.redirected == []


@pytest.mark.asyncio
async def test_merge_worker_client_follows_redirects_and_accepts_json(
    tmp_path, server, client_factory,
):
    worker = _make_merge_worker(tmp_path)

    with patch('httpx.AsyncClient', client_factory):
        await worker._post_submit_tasks([FIX_TASK_ARGS])

    client_factory.assert_follows_redirects()
    server.assert_every_request_accepted_json()


@pytest.mark.asyncio
async def test_merge_worker_submit_is_still_none_safe(tmp_path, server, client_factory):
    """The documented ``self._mcp is None`` guard must survive the fix."""
    worker = _make_merge_worker(tmp_path)
    worker._mcp = None

    with patch('httpx.AsyncClient', client_factory):
        await worker._post_submit_tasks([FIX_TASK_ARGS])

    assert server.seen == [], 'a None _mcp must produce no HTTP traffic at all'
