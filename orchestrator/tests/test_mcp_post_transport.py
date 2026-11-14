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
``tools/call`` on.  The handler and the recording client factory live in
``_mcp_transport_harness`` — a ``_``-prefixed sibling, because
``test_suggestion_triage.py`` drives the same fake server and test modules must
not import each other.  See that module's docstring for what the handler
reproduces and why.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from _mcp_transport_harness import (
    MCP_PATH,
    MCP_PATH_WITH_SLASH,
    RecordingClientFactory,
    RecordingMcpServer,
)

from orchestrator.workflow import TaskWorkflow


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
    """The decisions writer's client construction is reached by the patch seam.

    ``_write_decisions_to_memory`` obtains its client through
    ``shared.mcp_post.open_mcp_client``, which resolves ``AsyncClient`` as a
    MODULE ATTRIBUTE at call time — so ``patch('httpx.AsyncClient', ...)``
    intercepts it.  Pinned here because that is a property of HOW the import is
    written, not of what it imports: this site has already been rewritten once
    (it used to spell it ``__import__('httpx').AsyncClient``), and a future
    rewrite that bound ``AsyncClient`` at import time — ``from httpx import
    AsyncClient`` at module scope — would silently escape every assertion in
    this file while every test kept passing.
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


def _make_merge_worker(tmp_path, mcp_url: str | None = 'http://memory.test:8002'):
    """A bare merge worker whose MCP handle points at *mcp_url* (None: no handle)."""
    import asyncio

    from orchestrator.git_ops import GitOps
    from orchestrator.merge_lane.worker import SpeculativeMergeWorker

    # project_root must be wired: SpeculativeMergeWorker.__init__ reads it and
    # it is an instance attribute, so a spec-only mock does not carry it (see
    # test_merge_queue_auto_heal.py:_make_mock_git_ops).
    git_ops = MagicMock(spec=GitOps)
    git_ops.project_root = tmp_path

    mcp = None
    if mcp_url is not None:
        mcp = MagicMock()
        mcp.url = mcp_url
    return SpeculativeMergeWorker(git_ops=git_ops, queue=asyncio.Queue(), mcp=mcp)


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
    worker = _make_merge_worker(tmp_path, mcp_url=None)

    with patch('httpx.AsyncClient', client_factory):
        await worker._post_submit_tasks([FIX_TASK_ARGS])

    assert server.seen == [], 'a None _mcp must produce no HTTP traffic at all'
