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

import functools
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from _mcp_transport_harness import (
    MCP_PATH,
    MCP_PATH_WITH_SLASH,
    RecordingClientFactory,
    RecordingMcpServer,
)
from _mcp_url_scan import find_trailing_slash_mcp_urls
from _orch_helpers import WHOLE_TREE_SCAN_TEST_TIMEOUT

from orchestrator.workflow import TaskWorkflow

# The repo-wide sweep guard below ast.parses every swept *.py, so this module
# is a member of the whole-tree-scanner family whose ceiling
# test_whole_tree_scan_timeout_guard.py enforces (see
# WHOLE_TREE_SCAN_TEST_TIMEOUT in _orch_helpers.py).
pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)


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
    from orchestrator.merge_queue import SpeculativeMergeWorker

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


# ---------------------------------------------------------------------------
# Repo-wide sweep guard (step-13 RED / step-14 GREEN)
# ---------------------------------------------------------------------------

#: Directories holding raw-POST source.  Scoped, not repo-wide, on purpose:
#: ``*/tests`` are excluded wholesale because FastMCP's
#: ``StreamableHttpTransport(f'{base}/mcp/')`` fixtures legitimately use the
#: slashed mount (see the guard's docstring).  ``escalation/src`` and
#: ``shared/src`` are in the list even though both are clean today —
#: ``escalation/src`` hosts an MCP server, and ``shared/src`` now hosts the
#: primitive the fix routes through, so both are exactly where a sixth site
#: would appear.  Measured while adding them: 0 hits each.
SWEEP_DIRS = (
    'orchestrator/src',
    'fused-memory/src',
    'dashboard/src',
    'escalation/src',
    'shared/src',
    'scripts',
    'fused-memory/scripts',
)

#: Worktree root.  Same expression as ``conftest.py:38`` and restated here per
#: that file's conftest-collision note — resolving from ``__file__`` is what
#: makes the sweep check THIS worktree, the copy the task's verify run gates on.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: LINES that legitimately build a slashed URL and are NOT raw POSTs, keyed
#: ``(relpath, stripped_source_line)``.
#:
#: PER-LINE, NOT PER-FILE.  A whole-file exemption for
#: ``fused_memory/server/main.py`` — a ~1200-line module that IS the MCP
#: server — would make a genuine raw POST added anywhere in it invisible to
#: the guard, which is the opposite of what an exclusion list is for.
#:
#: Keyed on the source TEXT rather than a line number on purpose: the two
#: ``main.py`` lines below moved from :1083/:1149 to :1118/:1188 in a single
#: unrelated rebase during this task, so line-pinned entries would fail the
#: guard on edits that have nothing to do with MCP URLs.  The text is what
#: makes the entry specific; the file alone is not.
#:
#: (An inline ``# mcp-url-sweep: allow <reason>`` marker on each offending
#: source line would be better still — the justification would live next to
#: the code and could not silently widen — but adding those markers means
#: editing ``fused-memory/src/...``, which task 4023 holds no locks for.
#: Filed as follow-up rather than reached for here.)
SWEEP_EXCLUSIONS = {
    # An MCP *config* entry consumed by the Claude CLI's own MCP client, which
    # follows redirects natively.  A different risk class from a raw POST.
    (
        'fused-memory/src/fused_memory/reconciliation/stages/base.py',
        "'url': f'http://127.0.0.1:{self._recon_report_port}/mcp/',",
    ),
    # Display/log strings in ``main.py::run_server``, not URLs ever fetched.
    (
        'fused-memory/src/fused_memory/server/main.py',
        "logger.info(f'  MCP Endpoint: http://{display_host}:{config.server.port}/mcp/')",
    ),
    (
        'fused-memory/src/fused_memory/server/main.py',
        "'  Recon Report Endpoint: http://%s:%d/mcp/',",
    ),
}


#: Co-location tag for the two tests that call ``_sweep_hits``.
#:
#: MEASURED.  The walk reads and ``ast.parse``s 553 files: ~13s of pure work,
#: 20.5s wall on a machine already running the fleet.  Its timeout budget is
#: the module-level ``pytestmark`` above; this tag is the other half.
#:
#: ``xdist_group`` is what makes the ``lru_cache`` actually pay: ``--dist
#: loadgroup`` spreads UNGROUPED tests across workers, so without the tag the
#: two callers land in different processes and each pays a full walk — a
#: per-process cache cannot help.  Tagged, they share one worker and the second
#: call is free (measured 0.000004s).
SWEEP_GROUP = 'mcp_url_sweep'


@functools.lru_cache(maxsize=1)
def _sweep_hits() -> dict[str, tuple[tuple[int, str], ...]]:
    """Return ``{relpath: hits}`` for every swept file that builds a slashed URL.

    Shared by the guard and the stale-exclusion check below so both read the
    same walk — an exclusion list and the sweep it applies to must never be
    able to disagree about what the tree contains.

    CACHED, and returning tuples so the cached value cannot be mutated by
    either caller.  The walk parses 553 files (measured); it ran twice for a
    byte-identical result, and each run is the dominant cost of this whole
    file — against a per-test timeout that only gets tighter under
    ``-n auto`` contention.  Nothing in the tree
    changes between the two calls within a session, so the second is pure
    waste: measured at 0.000004s from cache.
    """
    found: dict[str, tuple[tuple[int, str], ...]] = {}
    for rel_dir in SWEEP_DIRS:
        for path in sorted((REPO_ROOT / rel_dir).rglob('*.py')):
            rel = path.relative_to(REPO_ROOT).as_posix()
            hits = find_trailing_slash_mcp_urls(
                path.read_text(encoding='utf-8'), filename=rel
            )
            if hits:
                found[rel] = tuple(hits)
    return found


@pytest.mark.xdist_group(SWEEP_GROUP)
def test_no_raw_post_builds_a_trailing_slash_mcp_url():
    """No source file may build an MCP URL with a trailing slash.

    Implements the "grep the slash, not the flag" rule: the slash is the
    NECESSARY condition for the silent loss, and ``follow_redirects`` only
    determines whether the loss is fatal.  A future site can reintroduce this
    without touching any of the five methods fixed here, so the guard is
    repo-scoped rather than attached to them.

    DELIBERATELY OUT OF SCOPE: FastMCP ``StreamableHttpTransport(f'{base}/mcp/')``
    uses under ``escalation/tests/`` and ``fused-memory/tests/``.  ``/mcp/`` is
    FastMCP's canonical ASGI mount — ``escalation/tests/conftest.py:201``
    explicitly waits on it to prove the app finished mounting — and that client
    follows redirects natively, so those sites carry no silent-discard risk.
    Rewriting live fixtures to chase a cosmetic match would risk breaking
    readiness gating for zero benefit.

    THE SLASH IS NECESSARY, NOT SUFFICIENT, and this guard only covers the
    necessary half.  A sixth site could drop the slash and still lose its
    payload by omitting the ``Accept`` header (the live server answers 406) or
    ``follow_redirects=True``.  Those three parts are held together by
    ``shared.mcp_post.post_mcp_tool_call`` / ``open_mcp_client``, which apply
    them as a unit; this guard is the backstop for a site that bypasses the
    primitive entirely.

    DETECTION IS BY PARSING, NOT SUBSTRING.  This guard originally matched the
    text ``}/mcp/'``, chosen over a bare ``/mcp/`` because that also matches
    the phrase ``scheduler/mcp/usage_gate/cost_store`` in merge_queue.py
    docstrings (:1163, :2467) and would have been permanently unsatisfiable.
    But the narrow anchor bought that immunity by hard-coding one spelling —
    it required a brace immediately before the slash and a single quote after,
    so double-quoted, plain-literal, percent-format and concatenated URLs all
    passed silently.  ``_mcp_url_scan`` matches the parsed literal's TAIL
    instead, which catches every spelling and is still immune to the prose.

    MEASURED both directions, against real historical source: run over
    ``git show 2633a244a6:<path>`` the detector fires on exactly the 8 pre-fix
    defect sites (workflow.py 15173/15215/15247/15286, merge_queue.py 16031,
    and the three operator scripts); run over this tree it flags only the
    three excluded LINES below.  A guard that cannot fail on the defect it was
    written for is worth nothing.
    """
    offenders = [
        f'{rel}:{lineno}: {text}'
        for rel, hits in sorted(_sweep_hits().items())
        for lineno, text in hits
        if (rel, text) not in SWEEP_EXCLUSIONS
    ]

    assert offenders == [], (
        'these build an MCP URL with a trailing slash; the server 307-redirects '
        'it and a bare httpx client discards the payload silently. Use '
        'shared.mcp_post.post_mcp_tool_call (or mcp_endpoint_url) instead:\n  '
        + '\n  '.join(offenders)
    )


@pytest.mark.xdist_group(SWEEP_GROUP)
def test_every_sweep_exclusion_still_earns_its_place():
    """An exclusion that outlives its offending line must fail, not lurk.

    Each entry in ``SWEEP_EXCLUSIONS`` is a hole in the guard above.  If the
    line that justified one is fixed or reworded, the hole stays open and
    silently covers whatever a future edit puts in its place.  Requiring every
    exclusion to still match a LIVE hit turns that into a failing test the
    moment it goes stale — the entry gets removed instead of masking a
    regression.

    Because the entries are ``(relpath, source_text)`` this also fails when an
    excluded line is merely REWORDED, which a per-file exemption would have
    absorbed silently.  Measured today: all three still hit.
    """
    live = {
        (rel, text)
        for rel, hits in _sweep_hits().items()
        for _lineno, text in hits
    }

    stale = sorted(SWEEP_EXCLUSIONS - live)

    assert stale == [], (
        'these lines no longer build a slashed MCP URL (moved, reworded or '
        'deleted), so excluding them from the sweep guard now only hides future '
        'regressions. Delete the entry from SWEEP_EXCLUSIONS:\n  '
        + '\n  '.join(f'{rel}: {text}' for rel, text in stale)
    )
