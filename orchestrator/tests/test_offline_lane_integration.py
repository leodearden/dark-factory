"""Integration gate for the offline deep-test lane (task 1955, ζ).

Stands up the REAL trigger→worker→warm-worktree→failure-handling chain
end-to-end and runs PRD §8 boundary scenarios B1-B8 for
``offline-deep-test-lane-worker.md``.  No production module is touched by
this task — every scenario below asserts already-landed behaviour from:

* β1 (task 1951) — ``Harness._offline_lane_notifiee`` / ``_note_merge_all`` /
  ``_note_offline_lane`` fan-out (``harness.py``).
* δ  (task 1952) — the persistent ``_offline-deep`` warm worktree
  (``git_ops.py``: ``persistent_offline_deep_worktree_path``,
  ``reset_persistent_offline_deep_worktree``, the ``cleanup_merge_worktree``
  prune exemption).
* β2 (task 1953) — ``OfflineLaneWorker`` (``offline_lane.py``): the
  coalescing ``run()`` loop, the always-from-head ``_run_once`` snapshot,
  and the injectable ``suite_runner`` / ``confirmation_runner`` seams.
* β3 (task 1954) — ``OfflineLaneWorker._handle_red_run`` (confirmation,
  fingerprinting, dedup'd fix-task file/update, L0/L2 escalation staging).

STRATEGY — injected-seam integration harness: the real reify-side heavy
suite run (Part A, cross-project, not yet on reify ``main``) is faked ONLY
at the ``suite_runner`` / ``confirmation_runner`` boundary — every other
component in the chain (the git repo, ``GitOps``, ``Harness`` fan-out,
``OfflineLaneWorker``, ``EscalationQueue``) is real.  See the task's
``.task/plan.json`` analysis for the full rationale.

Scenario map (PRD §8):

  B1 - advance triggers a from-head run
  B2 - coalescing under a burst of advances
  B3 - never-a-gate (C7): prompt return, fail-open, no halt/gate
  B4 - confirmed red files a normal fix task + L0 info escalation
  B5 - dedup: a same-set recurrence updates rather than duplicates
  B6 - flake filtered: fail-then-pass on confirmation is not a red
  B7 - stall promotes to a born-at-L2 escalate_blocker
  B8 - target/worktree ISOLATION (offline-deep path != merge-verify path)

OUT OF SCOPE:

* B8 asserts isolation only — never real reify compile timing, which is a
  reify-build capability owned by reify:4916/4913 (not deliverable here).
* B9 (flip-live-and-atomic) is owned by ε2 (deps ζ + reify:A4).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec
from escalation.queue import EscalationQueue

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.harness import Harness
from orchestrator.offline_lane import OfflineLaneWorker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeout bounds (task 4030) — named constants for the bounded waits below,
# replacing six near-identical inline 30.0 literals (the duplication that let
# a single false citation drift into three near-verbatim docstring copies).
# Modeled on the repo's own constant-plus-derivation-comment convention:
# _orch_helpers.py's CANCEL_SCOPE_BARRIER_TIMEOUT / CANCEL_SCOPE_PURE_UNIT_TIMEOUT
# and test_lane_lock_leak_guard.py's _FOREIGN_HOLDER_* block. The full floor
# (task 3451's measured spawn latency) / ceiling (the pyproject 60s global)
# derivation lives in `_run_lane`'s docstring below; the executable pins are
# `test_lane_bounds_clear_the_measured_floor_and_the_global_ceiling` and
# `test_every_composing_caller_carries_a_timeout_override`.
# ---------------------------------------------------------------------------

# Bound for a single `_run_lane` pass / `_ControllableSuiteRunner.wait_entered`
# call. See `_run_lane`'s docstring for the full derivation.
_LANE_PASS_BOUND_SECS = 30.0

# Bound for `_assert_never_a_gate`'s `_note_offline_lane` promptness check.
_NOTE_OFFLINE_LANE_BOUND_SECS = 0.5

# Bound for `_assert_never_a_gate`'s `_note_merge_all` promptness check.
_NOTE_MERGE_ALL_BOUND_SECS = 15.0

# Task 3451's measured worst-case happy-path subprocess spawn latency (n=3:
# 2.13/3.10/4.71, load-per-core 6.6) -- the FLOOR authority
# _LANE_PASS_BOUND_SECS is sized against. See `_run_lane`'s docstring for the
# full derivation.
_MEASURED_SPAWN_LATENCY_SECS = 4.71

# task 3836's own fraction, reused rather than re-derived: the CEILING a
# marker-less test's worst-case bounded-wait budget must stay under, as a
# fraction of the effective per-test pytest-timeout. The remaining 40% is
# headroom for real-git fixture setup/teardown sharing the same per-test
# budget (pytest-timeout times the whole `pytest_runtest_protocol`, not just
# the call phase, whenever `func_only` is unset -- true everywhere in this
# repo). See `test_lane_bounds_clear_the_measured_floor_and_the_global_ceiling`
# for the full derivation.
_MARKERLESS_CEILING_FRACTION = 0.6

# ---------------------------------------------------------------------------
# Shared end-to-end scaffolding (prerequisite P1)
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    """Create a real git repo with an initial commit (mirrors test_git_ops.py)."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


async def _advance_main(repo: Path, message: str = 'advance') -> str:
    """Commit a trivial, targeted change on main and return the new head SHA.

    Stages ONLY the dedicated counter file (never ``git add -A``) so that a
    real ``.worktrees/_offline-deep`` worktree nested under *repo* (created
    mid-test by
    :meth:`~orchestrator.git_ops.GitOps.reset_persistent_offline_deep_worktree`)
    is never swept into a same-repo commit as an embedded repository.
    """
    counter_path = repo / '_advance_counter.txt'
    n = int(counter_path.read_text()) + 1 if counter_path.exists() else 1
    counter_path.write_text(str(n))
    await _run(['git', 'add', str(counter_path)], cwd=repo)
    await _run(['git', 'commit', '-m', f'{message} {n}'], cwd=repo)
    _, sha, _ = await _run(['git', 'rev-parse', 'main'], cwd=repo)
    return sha.strip()


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        # No remote configured on the tmp repo; never push in this suite.
        push_after_advance=False,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real tmp git repo (initial commit only) — never a MagicMock git_ops."""
    repo_path = tmp_path / 'repo'
    repo_path.mkdir()
    asyncio.run(_setup_repo(repo_path))
    return repo_path


@pytest.fixture
def git_ops(git_config: GitConfig, repo: Path) -> GitOps:
    """Real GitOps bound to the real tmp repo (B1 from-head snapshots, B8 isolation)."""
    return GitOps(git_config, repo)


@pytest.fixture
def harness(git_config: GitConfig, git_ops: GitOps, mock_orch_config) -> Harness:
    """Minimally-constructed real Harness wired to the real tmp-repo GitOps.

    Modeled on ``test_harness_offline_lane_trigger.py``'s ``harness``
    fixture; the delta is that ``h.git_ops`` is replaced wholesale by the
    real tmp-repo-backed :func:`git_ops` fixture (rather than left as
    Harness's own internally-constructed GitOps over an un-git-initialized
    ``tmp_path``), so that real head snapshots / worktree resets are
    actually exercised end-to-end.
    """
    mock_orch_config.git = git_config
    mock_orch_config.project_root = git_ops.project_root
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)

    h.scheduler = MagicMock()
    h.scheduler._dispatched = set()
    h.event_store = MagicMock()
    h.git_ops = git_ops
    h.git_ops.get_merge_diff_files = AsyncMock(return_value=([], None))
    h._service_restart_coordinators = []
    return h


def _build_worker(
    git_ops: GitOps,
    tmp_path: Path,
    *,
    suite_runner=None,
    confirmation_runner=None,
    task_client=None,
    escalation_queue: EscalationQueue | None = None,
    git_overrides: dict | None = None,
) -> OfflineLaneWorker:
    """Build a real OfflineLaneWorker wired for the integration harness.

    The ONLY injection point is the reify-subprocess boundary
    (``suite_runner`` / ``confirmation_runner``) — everything else is real:
    *git_ops* is the shared real tmp-repo GitOps (see the ``git_ops``
    fixture) so the worker's head snapshots / worktree resets land in the
    SAME repo :func:`_advance_main` commits to; *escalation_queue* defaults
    to a real :class:`EscalationQueue` (never a MagicMock) so B4/B5/B7 can
    assert against real on-disk escalation records; *task_client* defaults
    to a fake ``OfflineLaneTaskClient`` (a bare ``AsyncMock`` — its
    ``submit_fix_task`` / ``append_suspect_range`` / ``get_status``
    children auto-vivify as awaitable ``AsyncMock``s, per the β3 unit-test
    convention in ``test_offline_lane.py``) with ``get_status`` pre-set to
    the steady in-flight ``'in-progress'`` state and ``submit_fix_task``
    pre-set to return a stable ``'fix-task-1'`` id (B4/B5/B7 only ever have
    one fingerprint open at a time, so one stable id suffices).
    """
    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.project_root = git_ops.project_root
    config.git = GitConfig(**(git_overrides or {}))

    if task_client is None:
        task_client = AsyncMock()
        task_client.get_status = AsyncMock(return_value='in-progress')
        task_client.submit_fix_task = AsyncMock(return_value='fix-task-1')

    if escalation_queue is None:
        escalation_queue = EscalationQueue(tmp_path / 'escalations')

    return OfflineLaneWorker(
        git_ops,
        config,
        lock_path=tmp_path / 'offline_lane.lock',
        suite_runner=suite_runner,
        confirmation_runner=confirmation_runner,
        task_client=task_client,
        escalation_queue=escalation_queue,
    )


def _wire_lane(harness: Harness, worker: OfflineLaneWorker) -> None:
    """Register the worker's real on_post_merge as the harness's lane notifiee.

    Mirrors the direct-attribute registration ``Harness._start_offline_lane``
    performs in production (``harness.py:5119``), without going through the
    enable-gated launch path itself — these tests drive the worker loop
    under their own control rather than as a live background task.
    """
    harness._offline_lane_notifiee = worker.on_post_merge


async def _drive_advance(
    harness: Harness, repo: Path, task_id: str = 'task-1',
) -> tuple[str, str]:
    """Advance main and await the REAL on_merge_landed fan-out.

    ``harness._note_merge_all`` IS the exact callback the
    ``SpeculativeMergeWorker`` invokes as ``on_merge_landed`` in production
    (see ``harness.py:4979``) — driving it directly here (rather than
    standing up a full merge worker) exercises the real fan-out without
    pulling the whole speculative-merge machinery into this integration
    gate. Returns ``(base_sha, head_sha)`` — the pre- and post-advance main
    SHAs — for the caller's own assertions.
    """
    base_sha = await harness.git_ops.get_main_sha()
    head_sha = await _advance_main(repo)
    await harness._note_merge_all(task_id, base_sha, head_sha)
    return base_sha, head_sha


async def _run_lane(
    worker: OfflineLaneWorker, expected_passes: int, *, timeout: float = _LANE_PASS_BOUND_SECS,
) -> None:
    """Drive worker.run() as a real background task until *expected_passes*
    full passes (``_run_once`` calls) have COMPLETED, then cancel the loop
    task.

    Wraps ``worker._run_once`` itself — NOT ``suite_runner`` alone — with a
    call counter, so ``done`` is only set once a pass, INCLUDING any
    ``_handle_red_run`` confirmation/fix-task/escalation work it triggers,
    has fully returned. Counting at the ``suite_runner`` boundary instead
    would race the red path: every seam this harness injects (a bare
    ``async def ...: return ...``, or an ``AsyncMock``) happens not to
    yield to the event loop, so today ``_handle_red_run`` always finishes
    before the loop next suspends at ``_wake.wait()`` — but that is an
    accident of the fakes, not a guarantee, and cancelling right after
    ``suite_runner`` returns would race ``_handle_red_run`` mid-flight the
    moment a seam awaits real I/O. Restores the original ``_run_once`` in
    the ``finally`` block so repeated calls (``_drive_reds`` drives 2-3
    passes per scenario) never nest counting wrappers.

    Modeled on ``test_offline_lane.py``'s
    ``test_loop_coalesces_to_exactly_one_rerun`` cancel-after-N pattern,
    generalized from a fixed N=1 to an arbitrary pass count (B1's single
    pass via :func:`_run_one_lane_pass`; B2's held-pass-plus-one-coalesced-
    rerun burst uses N=2). Requires ``worker._dirty`` (or the wake event) to
    already be set by a prior trigger, exactly as production wiring would
    leave it.

    ``timeout`` defaults to 30.0, not a tight bound: the clock starts at
    ``asyncio.create_task`` above, so it also covers the real-git test-body
    work the caller does between entering the held pass and releasing it
    (each :func:`_drive_advance` is a git add + commit + rev-parse, i.e. 3+
    subprocess spawns) — not just the lane pass(es) themselves. Task 3451
    measured worst-case single subprocess spawn latency at 4.71s on this
    host under load; a caller can easily need several spawns inside this
    window. Same load-sensitive full-suite-flake class as
    1335/1836/2819/3451/3491 (task 3832); 30s is the ceiling task 3491
    settled on for it. Widening THIS bound alone can never make a broken
    staging pass — it only lengthens how long a genuinely broken SINGLE pass
    takes to fail. That safety property does not compose for free: callers
    chaining N passes (:func:`_drive_reds`) sum this bound N times, so a
    worst case can now approach or exceed this module's pyproject-configured
    60s per-test timeout before this function's own ``wait_for`` ever fires —
    trading a clean, well-located ``TimeoutError`` here for pytest-timeout's
    blunter thread-mode worker kill instead (task 3832 review; see
    ``orchestrator/pyproject.toml``'s ``timeout``/``timeout_method``
    comment). Multi-pass callers whose worst-case sum is at or above that
    ceiling carry their own ``@pytest.mark.timeout`` override — see
    ``test_b5_same_set_recurrence_updates_not_duplicates`` and
    ``test_b7_stall_promotes_to_blocker`` below.
    """
    inner_run_once = worker._run_once
    done = asyncio.Event()
    count = {'n': 0}

    async def _counting_run_once():
        result = await inner_run_once()
        count['n'] += 1
        if count['n'] >= expected_passes:
            done.set()
        return result

    worker._run_once = _counting_run_once
    task = asyncio.create_task(worker.run())
    try:
        await asyncio.wait_for(done.wait(), timeout=timeout)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        worker._run_once = inner_run_once


async def _run_one_lane_pass(worker: OfflineLaneWorker, *, timeout: float = _LANE_PASS_BOUND_SECS) -> None:
    """Drive worker.run() as a real background task for exactly one pass (B1)."""
    await _run_lane(worker, 1, timeout=timeout)


class _ControllableSuiteRunner:
    """A suite_runner whose FIRST call blocks in-flight until released (B2/B3).

    Records every head it is invoked with, in call order. Only the first
    call blocks (on an internal ``asyncio.Event`` gate) — later calls
    return immediately — so a single instance can drive both the held
    initial pass and the subsequent coalesced re-run(s) in one scenario.
    """

    def __init__(self) -> None:
        self.heads: list[str] = []
        self._hold = asyncio.Event()
        self._entered = asyncio.Event()
        self._held_once = False

    async def __call__(self, wt: Path, head: str, threads: int) -> tuple[int, str]:
        self.heads.append(head)
        if not self._held_once:
            self._held_once = True
            self._entered.set()
            await self._hold.wait()
        return (0, '')

    def release(self) -> None:
        """Release the held first call."""
        self._hold.set()

    async def wait_entered(self, timeout: float = _LANE_PASS_BOUND_SECS) -> None:
        """Block until the held first call has actually started (is in-flight).

        ``timeout`` defaults to 30.0 for the same reason as ``_run_lane``'s
        default above (see the derivation at this module's lines 261-271):
        the ``_run_lane`` task created just before this call races the SAME
        clock, and this call is entered only after
        ``OfflineLaneWorker._run_once`` has already done its own real-git
        work (``get_main_sha`` plus a persistent-worktree reset — 3-5
        subprocess spawns) — a tighter bound here would fire before
        ``_run_lane``'s ever could. Same safety property: widening can never
        make a broken staging pass, it only lengthens how long a genuinely
        wedged runner takes to fail.
        """
        await asyncio.wait_for(self._entered.wait(), timeout=timeout)


async def _assert_never_a_gate(
    harness: Harness,
    worker: OfflineLaneWorker,
    runner: _ControllableSuiteRunner,
    repo: Path,
    git_ops: GitOps,
) -> None:
    """Assert the never-a-gate invariant (C7) while a lane run is held in-flight.

    Called with the FIRST pass already confirmed in-flight (the caller
    awaited ``runner.wait_entered()``) and released only after this
    returns, so the bounded ``wait_for`` below proves the synchronous
    on_merge_landed fan-out never blocks on it.

    (1)+(2) ``harness._note_offline_lane`` — the synchronous notifiee call
    ``_note_merge_all`` (the exact ``on_merge_landed`` callback
    ``SpeculativeMergeWorker`` invokes, ``harness.py:4979``) awaits AHEAD of
    its diff fetch — must return promptly. ``_note_merge_all`` itself is
    then exercised immediately after, at a loose 15.0s bound (well outside
    the flake band, but still catching a genuine block), with ``_dirty``
    explicitly reset just before so the ``worker._dirty`` assertion below
    re-proves ITS OWN fan-out sets it, not merely the ``_note_offline_lane``
    call above (task 3832 review).
    (3) A raising notifiee is fail-open: ``_note_offline_lane``'s own
    try/except (``harness.py:5072-5079``) swallows it — the SAME shape
    ``SpeculativeMergeWorker`` independently wraps this exact call in
    (``merge_queue.py:10578-10596``, belt-and-suspenders) — so the merge
    still lands (proven behaviorally below via the service-restart
    coordinator fan-out still running, NOT by log text).
    (4) Neither case ever halts the merge queue or files an escalation —
    the only halt/gate-adjacent side effects reachable from this call path.
    """
    assert harness.get_merge_halt_status() == {'wired': False}

    base_sha = await git_ops.get_main_sha()
    head_sha = await _advance_main(repo)
    # Bounded at 0.5s (task 3832 assessment, NOT the _run_lane flake class):
    # this window covers only ``harness._note_offline_lane`` ->
    # ``worker.on_post_merge`` (harness.py:9333-9349, offline_lane.py:352-
    # 366), a bare enqueue-and-return (sets ``_dirty`` + an event, no
    # subprocess spawn) — exactly the documented "MUST enqueue-and-return
    # promptly ... rather than perform the deep-test run inline" contract
    # (harness.py:9342-9348). This is a genuine promptness assertion (proves
    # the synchronous on_merge_landed fan-out never blocks on the in-flight
    # run); widening it would weaken what it's testing.
    #
    # ``_note_merge_all`` is called separately, right after, at a much
    # looser 15.0s bound rather than left fully unbounded (task 3832
    # review): unlike ``_note_offline_lane`` alone, it unconditionally runs
    # ``git_ops.get_merge_diff_files`` (a real ``git diff`` subprocess spawn)
    # once ``_note_offline_lane`` returns, so bounding IT at 0.5s would sit
    # in the same load-sensitive flake class this task (3832) exists to
    # remove (task 3451 measured 4.71s worst-case single-spawn latency under
    # load). 15.0s sits comfortably outside that flake band (>3x the
    # measured worst-case single spawn) while still catching a genuine block
    # in the production ``on_merge_landed`` callback this call IS
    # (harness.py:4979) — restoring promptness coverage of the real
    # callback that narrowing the FIRST bound to ``_note_offline_lane``
    # alone would otherwise drop entirely. The resulting double-invoke of
    # ``on_post_merge`` is safe — it is idempotent (sets ``_dirty = True``
    # and an already-set ``asyncio.Event``) — so the assertions below still
    # exercise the full fan-out unchanged. ``_dirty`` is explicitly reset
    # just before this call so the assertion below re-proves
    # ``_note_merge_all``'s OWN fan-out sets it, rather than passing
    # vacuously on the strength of the ``_note_offline_lane`` call above.
    await asyncio.wait_for(
        harness._note_offline_lane('task-2', base_sha, head_sha),
        timeout=_NOTE_OFFLINE_LANE_BOUND_SECS,
    )
    worker._dirty = False
    await asyncio.wait_for(
        harness._note_merge_all('task-2', base_sha, head_sha),
        timeout=_NOTE_MERGE_ALL_BOUND_SECS,
    )
    assert worker._dirty is True, (
        'a landed advance during an in-flight run must arm a coalesced re-run'
    )
    assert harness.get_merge_halt_status() == {'wired': False}
    assert cast(EscalationQueue, worker.escalation_queue).get_pending() == []

    async def _raising_notifiee(task_id: str, base: str, head: str) -> None:
        await worker.on_post_merge(task_id, base, head)
        raise RuntimeError('lane boom')

    harness._offline_lane_notifiee = _raising_notifiee
    coord = MagicMock()
    coord.note_merge = AsyncMock()
    harness._service_restart_coordinators = [coord]

    base2 = head_sha
    head2 = await _advance_main(repo)
    await harness._note_merge_all('task-3', base2, head2)  # must not raise

    # Fail-open is proven behaviorally: the merge still landed and ran the
    # service-restart coordinator fan-out despite the raising notifiee.
    coord.note_merge.assert_awaited_once_with(
        'task-3', base2, head2, prefetched_diff=[],
    )
    assert harness.get_merge_halt_status() == {'wired': False}
    assert cast(EscalationQueue, worker.escalation_queue).get_pending() == []


def _inject_red(worker: OfflineLaneWorker, failing_ids: list[str]) -> None:
    """Wire *worker* to simulate one confirmed-red suite pass (B4/B5/B7).

    ``suite_runner`` reports a normal FAILED pass (``rc=1``);
    ``confirmation_runner`` confirms the SAME *failing_ids* still fail in
    isolation — a genuine break, never the B6 flake case (see
    :func:`_inject_flake`). This is the ONLY point this integration harness
    fakes the reify-subprocess boundary for a red pass; everything
    downstream (fingerprinting, dedup, fix-task file/update, escalation
    staging) is the real ``OfflineLaneWorker._handle_red_run`` chain.
    """

    async def _suite_runner(wt: Path, head: str, threads: int) -> tuple[int, str]:
        return (1, 'FAILED (injected)')

    async def _confirmation_runner(wt: Path, head: str) -> list[str]:
        return list(failing_ids)

    worker.suite_runner = _suite_runner
    worker.confirmation_runner = _confirmation_runner


def _inject_flake(worker: OfflineLaneWorker) -> None:
    """Wire *worker* to simulate a flake: fails once, but confirms clean (B6).

    ``suite_runner`` reports a normal FAILED pass (``rc=1``); the isolated
    confirmation re-run finds NOTHING still failing (empty list) —
    intermittent nondeterminism, never the genuine break :func:`_inject_red`
    simulates.
    """

    async def _suite_runner(wt: Path, head: str, threads: int) -> tuple[int, str]:
        return (1, 'FAILED (injected, flake)')

    async def _confirmation_runner(wt: Path, head: str) -> list[str]:
        return []

    worker.suite_runner = _suite_runner
    worker.confirmation_runner = _confirmation_runner


def _submitted_fix_task_arguments(task_client) -> dict:
    """Assert ``submit_fix_task`` was awaited exactly once; return its sole argument block."""
    task_client.submit_fix_task.assert_awaited_once()
    return task_client.submit_fix_task.await_args.args[0]


def _l0_info_escalations(escalation_queue: EscalationQueue, task_id: str) -> list:
    """Pending L0 (agent→steward) info escalations filed for *task_id*."""
    return escalation_queue.get_by_task(task_id, status='pending', level=0)


def _l2_blocker_escalations(escalation_queue: EscalationQueue, task_id: str) -> list:
    """Pending L2 (born-at-L2) blocker escalations filed for *task_id*."""
    return escalation_queue.get_by_task(task_id, status='pending', level=2)


async def _drive_reds(
    harness: Harness, repo: Path, worker: OfflineLaneWorker, failing_ids: list[str], n: int,
) -> None:
    """Drive *n* same-failing-set confirmed-red advances through the real chain.

    Injects the SAME confirmed break once (:func:`_inject_red`) then, for
    *n* iterations, drives a real advance + the real on_merge_landed fan-out
    (:func:`_drive_advance`, mirroring B1-B4) followed by one full pass
    (:func:`_run_one_lane_pass`) — the same-fingerprint repeat-red sequence
    B5's dedup and B7's stall-promotion scenarios both need.

    Each pass sums its own 30s :func:`_run_lane` bound (see that function's
    docstring); a caller whose ``n`` (or count of calls) pushes the total at
    or above the 60s pyproject per-test timeout MUST carry its own
    ``@pytest.mark.timeout`` override (task 3832 review) — see B5/B7 below.
    """
    _inject_red(worker, failing_ids)
    for _ in range(n):
        await _drive_advance(harness, repo)
        await _run_one_lane_pass(worker)


def _terminal_status_task_client(status: str) -> AsyncMock:
    """A fake ``OfflineLaneTaskClient`` whose open fix task already reports a
    terminal non-done *status* (B7 arm 2 — ``'cancelled'`` / ``'deferred'``).

    :meth:`OfflineLaneWorker._maybe_promote_blocker` promotes on this arm
    immediately, regardless of the configured advance-count threshold.
    ``submit_fix_task`` / ``append_suspect_range`` behave as the
    ``_build_worker`` default (see its docstring) — only ``get_status``
    diverges.
    """
    task_client = AsyncMock()
    task_client.get_status = AsyncMock(return_value=status)
    task_client.submit_fix_task = AsyncMock(return_value='fix-task-1')
    return task_client


def _materialized_worktree_names(worktree_base: Path) -> set[str]:
    """Top-level entry names materialized under *worktree_base* so far (B8).

    A before/after diff of this set proves a lane reset touches ONLY its
    own persistent worktree directory, never the other lane's. Returns the
    empty set when *worktree_base* itself has not been created yet (the
    "before" case, prior to any worktree ever being registered).
    """
    if not worktree_base.exists():
        return set()
    return {p.name for p in worktree_base.iterdir()}


# ---------------------------------------------------------------------------
# Timeout-bound invariants (task 4030)
# ---------------------------------------------------------------------------


def test_every_composing_caller_carries_a_timeout_override() -> None:
    """Every test that composes bounded waits PAST a single `_run_lane` pass
    must carry its own `@pytest.mark.timeout` override wide enough to cover
    its own worst-case bounded-wait sum — otherwise it can silently collide
    with this module's pyproject-configured 60s per-test default.

    Makes WORK item 2's open re-verification question ("does EVERY chaining
    caller carry such an override?") executable rather than prose. The table
    below maps each composing test FUNCTION OBJECT — never a string, so a
    rename breaks this test loudly instead of silently dropping a row — to
    its worst-case bounded-wait sum, expressed in terms of this module's
    named timeout constants (never bare literals).

    ``test_b2_coalesces_burst_of_advances_to_one_rerun`` is deliberately
    ABSENT from this table: it starts `_run_lane(expected_passes=2)` via
    `asyncio.create_task` and then awaits `runner.wait_entered()` — those two
    bounded waits race the SAME wall clock (max, not sum), so its worst case
    is a single `_LANE_PASS_BOUND_SECS`, the marker-less budget
    `test_lane_bounds_clear_the_measured_floor_and_the_global_ceiling` pins.

    Correlation is done via `fn.pytestmark` introspection — pytest's own
    applied-marker list for a given test function — rather than the coarse
    `_TIMEOUT_MARKER_RE` text scan `test_lane_lock_leak_guard.py`'s sibling
    invariant (`test_no_foreign_holder_consumer_opts_out_of_the_global_timeout`)
    uses. That invariant only needs to know whether ANY
    `@pytest.mark.timeout` appears ANYWHERE in a file, so a deliberately
    coarse regex is an acceptable, cheap check for it. This invariant is
    finer-grained: it must correlate a marker to a SPECIFIC test and compare
    its VALUE against that test's own computed worst case. `fn.pytestmark`
    gives that directly, with no AST analysis and no false positives from
    this module's own prose mentions of the marker.
    """
    worst_case_secs = {
        test_b5_same_set_recurrence_updates_not_duplicates: 2 * _LANE_PASS_BOUND_SECS,
        test_b7_stall_promotes_to_blocker: 6 * _LANE_PASS_BOUND_SECS,
        test_b3_never_a_gate: (
            _LANE_PASS_BOUND_SECS
            + _NOTE_OFFLINE_LANE_BOUND_SECS
            + _NOTE_MERGE_ALL_BOUND_SECS
        ),
    }
    for fn, worst_case in worst_case_secs.items():
        markers = [m for m in fn.pytestmark if m.name == 'timeout']
        assert markers, (
            f'{fn.__name__} composes bounded waits to a worst case of '
            f'{worst_case}s but carries no @pytest.mark.timeout override. '
            f'Left uncovered, this can silently collide with the 60s '
            f'orchestrator/pyproject.toml per-test default — under '
            f'timeout_method="thread" with --max-worker-restart=0, '
            f'pytest-timeout os._exit()s the xdist worker instead of '
            f"failing cleanly, discarding _run_lane's own well-located "
            f'TimeoutError. Add @pytest.mark.timeout(N) with N > {worst_case}.'
        )
        value = markers[0].args[0] if markers[0].args else markers[0].kwargs.get(
            'timeout', markers[0].kwargs.get('seconds'),
        )
        assert value > worst_case, (
            f'{fn.__name__} carries @pytest.mark.timeout({value}) but its '
            f'own worst-case bounded-wait sum is {worst_case}s — the '
            f'override does not actually clear what it exists to cover. '
            f'Left uncovered, this can silently collide with the 60s '
            f'orchestrator/pyproject.toml per-test default — under '
            f'timeout_method="thread" with --max-worker-restart=0, '
            f'pytest-timeout os._exit()s the xdist worker instead of '
            f"failing cleanly, discarding _run_lane's own well-located "
            f'TimeoutError.'
        )


def test_lane_bounds_clear_the_measured_floor_and_the_global_ceiling(pytestconfig) -> None:
    """`_LANE_PASS_BOUND_SECS` must clear a MEASURED floor and stay under a
    stated fraction of the effective per-test CEILING — the two-sided
    contract task 3836 landed for the sibling foreign-holder constants
    (``test_lane_lock_leak_guard.py``'s
    `test_foreign_holder_bounds_clear_measured_spawn_latency` +
    `test_foreign_holder_bounds_stay_clear_of_the_global_pytest_timeout`),
    adapted here for this module's own lane-pass bound.

    FLOOR — `_LANE_PASS_BOUND_SECS` must be at least 5x
    `_MEASURED_SPAWN_LATENCY_SECS` (task 3451's measured worst-case
    happy-path subprocess spawn latency: n=3, 2.13/3.10/4.71s,
    load-per-core 6.6). The multiplier of 5 is DERIVED, not guessed: it is
    the measured worst-case count of subprocess spawns that occur INSIDE a
    single `_run_lane` window, and that worst case lives in the SIBLING
    infra module (``test_offline_lane_infra_integration.py``'s
    `test_ib6_real_infra_runner_over_stub_classified_set_files_fix_task`):
    `_run_once`'s 3 git spawns (`get_main_sha`, `worktree list`, `worktree
    add`) plus the real infra seam spawning `run_all.sh` twice (the initial
    run and the isolated confirmation re-run). Later passes swap `worktree
    add` for `reset --hard` + `clean -xfd` — 4 spawns, still under 5. This
    module's own tests sit well under that shared worst case (the numeric
    `suite_runner` seam here is always a bare injected callable, never a
    real subprocess); the floor is still asserted here so this module's copy
    of `_LANE_PASS_BOUND_SECS` cannot drift from the infra module's without
    either invariant catching it.

    CEILING — the marker-less worst case (one `_LANE_PASS_BOUND_SECS`, since
    steps 2/4 of task 4030 covered every test that composes past a single
    pass — a premise `test_every_composing_caller_carries_a_timeout_override`
    makes executable rather than prose-only, named here the same way
    `test_no_foreign_holder_consumer_opts_out_of_the_global_timeout` backs
    its own sibling ceiling invariant) must stay at or under
    `_MARKERLESS_CEILING_FRACTION` (0.6, task 3836's own fraction) of the
    EFFECTIVE per-test timeout, resolved via `_effective_per_test_timeout`
    (imported function-locally from `test_lane_lock_leak_guard` — never
    duplicated or re-derived from a bare `tomllib` read; that module is task
    3836's and already correct, and importing it inherits its correctness).
    Why 60%, made concrete rather than left abstract: pytest-timeout 2.4.0
    installs its timer in `pytest_runtest_protocol` whenever `func_only` is
    False (the default; `timeout_func_only` is not set anywhere in this
    repo), so the 60s budget covers FIXTURE SETUP AND TEARDOWN, not just the
    call phase. Every test in both offline-lane modules transitively pulls
    the `repo` fixture, whose `_setup_repo` costs 5 git spawns (`init`,
    `config` x2, `add`, `commit`), and each `_drive_advance` costs 4 more
    (`add`, `commit`, `rev-parse`, plus `_note_merge_all`'s `git diff`) — the
    24s of headroom the 0.6 fraction leaves at a 60s timeout is what pays
    for that out-of-bound work.

    When no timeout is in effect at all, this either fails loudly (this
    module's own `orchestrator/pyproject.toml` is the governing inifile —
    genuine drift, the premise these bounds are sized against no longer
    holds) or skips, naming the governing inifile (an invocation artifact,
    e.g. a root-bound run).
    """
    from test_lane_lock_leak_guard import _effective_per_test_timeout  # noqa: PLC0415

    floor = 5 * _MEASURED_SPAWN_LATENCY_SECS
    assert floor <= _LANE_PASS_BOUND_SECS, (
        f'_LANE_PASS_BOUND_SECS ({_LANE_PASS_BOUND_SECS}) must clear 5x the measured '
        f'worst-case happy-path subprocess spawn latency ({_MEASURED_SPAWN_LATENCY_SECS}s, '
        f'task 3451: n=3 2.13/3.10/4.71, load-per-core 6.6) = {floor}s. The 5-spawn '
        f'multiplier is the measured worst-case count of subprocess spawns inside a '
        f'single _run_lane window (see '
        f'test_ib6_real_infra_runner_over_stub_classified_set_files_fix_task in '
        f'test_offline_lane_infra_integration.py: 3 git spawns from _run_once plus 2 '
        f'real infra run_all.sh spawns).'
    )

    global_timeout = _effective_per_test_timeout(pytestconfig)
    if global_timeout is None:
        orchestrator_pyproject = Path(__file__).resolve().parents[1] / 'pyproject.toml'
        governing_inifile = pytestconfig.inipath
        if governing_inifile is not None and governing_inifile.resolve() == orchestrator_pyproject:
            pytest.fail(
                f'{governing_inifile} is the governing inifile for this run, yet no '
                f'per-test timeout is in effect (checked --timeout, PYTEST_TIMEOUT, and '
                f'[tool.pytest.ini_options].timeout) — this is a genuine regression: the '
                f'pytest-timeout / os._exit() premise _LANE_PASS_BOUND_SECS is sized '
                f'against no longer holds. Either restore the timeout key or re-derive '
                f'this bound without it.'
            )
        pytest.skip(
            f'no per-test timeout is in effect under the governing inifile '
            f'{governing_inifile!r} — this is not {orchestrator_pyproject}, so this is '
            f'an invocation artifact (e.g. a root-bound run), not drift; the '
            f'pytest-timeout ceiling this invariant checks does not apply to this run'
        )

    ceiling = _MARKERLESS_CEILING_FRACTION * global_timeout
    assert ceiling >= _LANE_PASS_BOUND_SECS, (
        f'_LANE_PASS_BOUND_SECS ({_LANE_PASS_BOUND_SECS}s) exceeds '
        f'{_MARKERLESS_CEILING_FRACTION * 100:.0f}% of the effective per-test timeout '
        f'({global_timeout}s, resolved from {pytestconfig.inipath}) = {ceiling}s. Every '
        f'test composing past a single _run_lane pass carries its own '
        f'@pytest.mark.timeout override (enforced by '
        f'test_every_composing_caller_carries_a_timeout_override), so this bounds ONLY '
        f"the marker-less budget — a single _run_lane pass. The remaining "
        f'{(1 - _MARKERLESS_CEILING_FRACTION) * 100:.0f}% is headroom for the real-git '
        f"fixture work (the repo fixture's _setup_repo, and each _drive_advance) that "
        f'shares the same per-test budget, since pytest-timeout installs its timer in '
        f'pytest_runtest_protocol (func_only=False, unset in this repo) and so times '
        f'fixture setup/teardown too, not just the call phase.'
    )


# ---------------------------------------------------------------------------
# B1 — advance triggers a from-head run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_b1_advance_triggers_from_head_run(harness, git_ops, repo, tmp_path, caplog):
    """B1 (PRD §8) — a landed advance triggers a lane run snapshotted from the
    CURRENT main head, never the advisory trigger SHA passed to on_post_merge."""
    calls: list[tuple] = []

    async def _suite_runner(wt, head, threads):
        calls.append((wt, head, threads))
        return (0, '')

    worker = _build_worker(git_ops, tmp_path, suite_runner=_suite_runner)
    _wire_lane(harness, worker)

    with caplog.at_level(logging.INFO):
        base_sha, head_sha = await _drive_advance(harness, repo)
        await _run_one_lane_pass(worker)

    real_head = await git_ops.get_main_sha()
    assert real_head == head_sha
    assert len(calls) == 1
    assert calls[0][1] == real_head, (
        'suite_runner must be invoked with the real from-head snapshot'
    )

    assert 'offline-lane: on_post_merge' in caplog.text
    assert base_sha[:12] in caplog.text
    assert head_sha[:12] in caplog.text


# ---------------------------------------------------------------------------
# B2 — coalescing under a burst of advances
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_b2_coalesces_burst_of_advances_to_one_rerun(harness, git_ops, repo, tmp_path):
    """B2 (PRD §8) — advances landing while a run is in-flight coalesce into
    exactly ONE re-run at the newest head, never a queue of stale re-runs."""
    runner = _ControllableSuiteRunner()
    worker = _build_worker(git_ops, tmp_path, suite_runner=runner)
    _wire_lane(harness, worker)

    _, first_head = await _drive_advance(harness, repo)

    lane_task = asyncio.create_task(_run_lane(worker, expected_passes=2))
    await runner.wait_entered()

    newest_head = first_head
    for _ in range(3):
        _, newest_head = await _drive_advance(harness, repo)

    runner.release()
    await lane_task

    assert runner.heads == [first_head, newest_head], (
        'a burst of advances during an in-flight run must coalesce into '
        'exactly one re-run at the newest head, never a queue of stale runs'
    )


# ---------------------------------------------------------------------------
# B3 — never-a-gate (C7): prompt return, fail-open, no halt/gate
# ---------------------------------------------------------------------------


@pytest.mark.timeout(150)  # task 4030: NOT a _drive_reds chainer, which is why the task-3832
# review's marker sweep missed it -- but it composes anyway: a 30s _run_lane bound raced
# concurrently by wait_entered (max, not sum), then _assert_never_a_gate's 0.5 + 15.0
# SEQUENTIALLY after it = 45.5s bounded, on top of unbounded real-git spawns (repo init,
# _setup_repo, two _drive_advance/_advance_main rounds; task 3451 measured 4.71s worst case
# per spawn). Clear the 60s pyproject default with margin so a genuinely wedged pass fails via
# _run_lane's own TimeoutError, not pytest-timeout's blunter worker kill.
@pytest.mark.asyncio
async def test_b3_never_a_gate(harness, git_ops, repo, tmp_path):
    """B3 (PRD §8, C7) — a merge-landed notification while the lane is
    in-flight must never gate the merge: it returns promptly (never blocks
    the synchronous notifiee call on the in-flight run), sets _dirty for a
    coalesced re-run, is fail-open when the notifiee raises, and never
    files an escalation (the only halt/gate-adjacent side-effect reachable
    from this call path)."""
    runner = _ControllableSuiteRunner()
    worker = _build_worker(git_ops, tmp_path, suite_runner=runner)
    _wire_lane(harness, worker)

    await _drive_advance(harness, repo)
    lane_task = asyncio.create_task(_run_lane(worker, expected_passes=1))
    await runner.wait_entered()

    await _assert_never_a_gate(harness, worker, runner, repo, git_ops)

    runner.release()
    await lane_task


# ---------------------------------------------------------------------------
# B4 — confirmed red files a normal fix task + L0 info escalation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_b4_confirmed_red_files_fix_task_and_info_escalation(harness, git_ops, repo, tmp_path):
    """B4 (PRD §8) — a confirmed-red run files a NORMAL pending fix task
    (never the b3_gate red-main fix-forward path) plus a non-blocking L0
    info escalation, leaving the merge queue itself completely untouched."""
    from orchestrator.workflow import compute_failing_test_set_fingerprint

    failing_ids = ['t::b', 't::a']
    worker = _build_worker(git_ops, tmp_path)
    _inject_red(worker, failing_ids)
    _wire_lane(harness, worker)

    _, head_sha = await _drive_advance(harness, repo)
    await _run_one_lane_pass(worker)

    arguments = _submitted_fix_task_arguments(worker.task_client)
    assert arguments['status'] == 'pending'
    assert arguments['metadata']['merge_lane'] == 'normal', (
        'a confirmed-red fix task must route through the standard '
        'TDD→PR→merge gate, never the b3_gate red-main fix-forward path'
    )
    assert arguments['metadata']['failing_tests'] == sorted(failing_ids)
    assert arguments['metadata']['suspect_ranges'] == [head_sha], (
        'no prior green run is recorded yet, so the suspect range is just the head'
    )

    fp = compute_failing_test_set_fingerprint(failing_ids)
    task_id = worker.open_fix_tasks[fp]
    escalations = _l0_info_escalations(cast(EscalationQueue, worker.escalation_queue), task_id)
    assert len(escalations) == 1
    assert escalations[0].severity == 'info'
    assert escalations[0].agent_role == 'orchestrator-offline-lane'

    # The merge queue itself is left completely untouched by the red path.
    assert harness.get_merge_halt_status() == {'wired': False}


# ---------------------------------------------------------------------------
# B5 — dedup: a same-set recurrence updates rather than duplicates
# ---------------------------------------------------------------------------


@pytest.mark.timeout(150)  # task 3832 review: _drive_reds(n=2) chains 2 30s-bounded
# _run_one_lane_pass calls (60s alone) plus real-git _drive_advance overhead --
# clear the 60s pyproject default with margin so a genuinely wedged pass fails
# via _run_lane's own TimeoutError, not pytest-timeout's blunter worker kill.
@pytest.mark.asyncio
async def test_b5_same_set_recurrence_updates_not_duplicates(harness, git_ops, repo, tmp_path):
    """B5 (PRD §8) — a SECOND red advance with the SAME failing-test set
    updates the existing fix task (appends a suspect range) rather than
    filing a duplicate task or raising a second escalation."""
    from orchestrator.workflow import compute_failing_test_set_fingerprint

    failing_ids = ['t::a', 't::b']
    worker = _build_worker(git_ops, tmp_path)
    _wire_lane(harness, worker)

    await _drive_reds(harness, repo, worker, failing_ids, 2)

    cast(AsyncMock, worker.task_client).submit_fix_task.assert_awaited_once()
    cast(AsyncMock, worker.task_client).append_suspect_range.assert_awaited_once()

    fp = compute_failing_test_set_fingerprint(failing_ids)
    task_id = worker.open_fix_tasks[fp]
    escalations = _l0_info_escalations(cast(EscalationQueue, worker.escalation_queue), task_id)
    assert len(escalations) == 1, (
        'a same-set recurrence must not raise a second L0 escalation'
    )


# ---------------------------------------------------------------------------
# B6 — flake filtered: fail-then-pass on confirmation is not a red
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_b6_flake_filtered_by_confirmation_rerun(harness, git_ops, repo, tmp_path):
    """B6 (PRD §8) — a run that fails, then passes on the isolated
    confirmation re-run, is intermittent nondeterminism, never a genuine
    break: no fix task and no escalation are ever raised."""
    worker = _build_worker(git_ops, tmp_path)
    _wire_lane(harness, worker)
    _inject_flake(worker)

    await _drive_advance(harness, repo)
    await _run_one_lane_pass(worker)

    # Intermittent-nondeterminism handling is proven behaviorally below
    # (no fix task, no open fingerprint, no escalation) rather than by log
    # text.
    cast(AsyncMock, worker.task_client).submit_fix_task.assert_not_awaited()
    assert worker.open_fix_tasks == {}
    assert cast(EscalationQueue, worker.escalation_queue).get_pending() == []


# ---------------------------------------------------------------------------
# B7 — stall promotes to a born-at-L2 escalate_blocker
# ---------------------------------------------------------------------------


@pytest.mark.timeout(300)  # task 3832 review: 4 _drive_reds calls (2+1+2+1 = 6 total
# 30s-bounded _run_one_lane_pass calls -- 180s worst case) across both arms, plus
# real-git overhead -- clear the 60s pyproject default with margin.
@pytest.mark.asyncio
async def test_b7_stall_promotes_to_blocker(harness, git_ops, repo, tmp_path):
    """B7 (PRD §8, C4) — a fix task that stalls promotes the L0 info signal
    to a real, born-at-L2 escalate_blocker via either arm: the confirmed-red
    advance count reaching the configured threshold (arm 1), or the fix task
    itself landing in a terminal non-done status (arm 2, cancelled/deferred).
    Promotion is idempotent per fingerprint and never touches the merge
    queue."""
    from orchestrator.workflow import compute_failing_test_set_fingerprint

    # Arm 1 — N-advances: the confirmed-red count reaches the (small,
    # explicitly configured) threshold.
    failing_a = ['t::stall_a']
    worker_a = _build_worker(
        git_ops, tmp_path / 'arm1',
        git_overrides={'offline_lane_red_advances_before_blocker': 2},
    )
    _wire_lane(harness, worker_a)
    await _drive_reds(harness, repo, worker_a, failing_a, 2)

    fp_a = compute_failing_test_set_fingerprint(failing_a)
    task_a = worker_a.open_fix_tasks[fp_a]
    blockers_a = _l2_blocker_escalations(cast(EscalationQueue, worker_a.escalation_queue), task_a)
    assert len(blockers_a) == 1
    assert blockers_a[0].severity == 'critical'
    assert blockers_a[0].agent_role == 'orchestrator-offline-lane'

    # A further same-set red must not re-promote (idempotent per fingerprint).
    await _drive_reds(harness, repo, worker_a, failing_a, 1)
    assert len(_l2_blocker_escalations(cast(EscalationQueue, worker_a.escalation_queue), task_a)) == 1
    assert harness.get_merge_halt_status() == {'wired': False}

    # Arm 2 — terminal non-done status: promotes immediately regardless of
    # the count (threshold set deliberately high so arm 1 cannot explain it).
    failing_b = ['t::stall_b']
    worker_b = _build_worker(
        git_ops, tmp_path / 'arm2',
        task_client=_terminal_status_task_client('cancelled'),
        git_overrides={'offline_lane_red_advances_before_blocker': 100},
    )
    _wire_lane(harness, worker_b)
    await _drive_reds(harness, repo, worker_b, failing_b, 2)

    fp_b = compute_failing_test_set_fingerprint(failing_b)
    task_b = worker_b.open_fix_tasks[fp_b]
    blockers_b = _l2_blocker_escalations(cast(EscalationQueue, worker_b.escalation_queue), task_b)
    assert len(blockers_b) == 1, (
        'a terminal non-done fix-task status must promote immediately, '
        'regardless of the configured advance-count threshold'
    )

    await _drive_reds(harness, repo, worker_b, failing_b, 1)
    assert len(_l2_blocker_escalations(cast(EscalationQueue, worker_b.escalation_queue), task_b)) == 1
    assert harness.get_merge_halt_status() == {'wired': False}


# ---------------------------------------------------------------------------
# B8 — target/worktree ISOLATION (offline-deep path != merge-verify path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_b8_offline_deep_worktree_isolated_from_merge_lane(git_ops, repo):
    """B8 (PRD §8) — the offline-deep lane's persistent worktree is isolated
    from the merge lane's: distinct fixed paths; a real reset materializes
    and registers ONLY the offline-deep path (the merge lane's _merge-verify
    path is never created or touched by the lane reset); and
    cleanup_merge_worktree is prune-exempt (a no-op) on the offline-deep
    path. Real reify compile timing is NOT asserted here — self-warming
    timing is a reify-build capability (reify:4916/4913) validated at
    production activation, out of ζ's executable scope."""
    assert (
        git_ops.persistent_offline_deep_worktree_path
        != git_ops.persistent_merge_worktree_path
    )

    before = _materialized_worktree_names(git_ops.worktree_base)
    head = await git_ops.get_main_sha()
    offline_wt = await git_ops.reset_persistent_offline_deep_worktree(head)
    after = _materialized_worktree_names(git_ops.worktree_base)

    assert offline_wt == git_ops.persistent_offline_deep_worktree_path
    assert after - before == {git_ops.persistent_offline_deep_worktree_path.name}, (
        'a lane reset must materialize ONLY its own _offline-deep worktree, '
        'never the merge lane _merge-verify path'
    )
    assert not git_ops.persistent_merge_worktree_path.exists(), (
        'the merge lane worktree must never be created by an offline-deep reset'
    )
    assert await git_ops._is_registered_worktree(offline_wt)

    await git_ops.cleanup_merge_worktree(offline_wt)
    assert offline_wt.exists(), (
        'cleanup_merge_worktree must be prune-exempt (a no-op) on the '
        'persistent offline-deep worktree'
    )
    assert await git_ops._is_registered_worktree(offline_wt), (
        'the offline-deep worktree must remain a registered git worktree '
        'after a cleanup_merge_worktree no-op'
    )


# ---------------------------------------------------------------------------
# Incident reproduction (task 5308) — a verify.sh usage dump files NOTHING
# ---------------------------------------------------------------------------


# A faithful reproduction of reify scripts/verify.sh's error+usage() dump
# (reify:5308 incident; twin of the fixture in test_offline_lane.py): an
# em-dash ``verify.sh: ERROR — <msg>`` banner followed by the full usage()
# block (bare ``Usage:``/``Options:`` section headers + invocation/option
# lines). run-offline-deep.sh merges stderr into stdout, so this whole blob
# reaches the numeric confirmation seam's parser.
_VERIFY_USAGE_DUMP = (
    "verify.sh: ERROR — unknown argument '--test-threads=1'\n"
    "\n"
    "scripts/verify.sh — unified verification entrypoint for Reify.\n"
    "\n"
    "Usage:\n"
    "  verify.sh <test|lint|typecheck|all> [options]\n"
    "\n"
    "Options:\n"
    "  --scope <all|rust|gui|infra>  Restrict verification to a subsystem.\n"
    "  --profile <fast|full|both>    Select the verification profile.\n"
    "  -h|--help  Show usage.\n"
)


@pytest.mark.asyncio
async def test_verify_usage_dump_confirmation_files_no_task_or_escalation(git_ops, tmp_path):
    """Incident reproduction (task 5308; reify:5264) — when the DEFAULT numeric
    confirmation seam ingests a reify verify.sh usage/help/error dump (stderr
    merged into stdout), the full red path files ZERO fix tasks and ZERO
    escalations, never one fake fix task built from usage lines.

    Unlike B4 (which injects a fake confirmation_runner), this leaves the
    numeric confirmation seam at its DEFAULT so the REAL
    ``_default_confirmation_run`` parser runs end-to-end — the reify-subprocess
    boundary is faked one level deeper, at ``create_subprocess_exec``, which is
    patched to emit the usage dump with returncode 64. Everything downstream of
    the parse (the empty-``confirmed`` guard, fingerprinting, fix-task filing,
    escalation staging) is the real ``_handle_red_run`` chain, driven directly
    (the confirmed-list → ``_handle_red_run`` corruption flow's entry point).

    RED pre-fix: the seam's inline comprehension turns every non-blank usage
    line into a confirmed test ID, filing a fix task + an L0 info escalation.
    """
    worker = _build_worker(git_ops, tmp_path)  # DEFAULT confirmation seam, real queue + spy client
    head = await git_ops.get_main_sha()
    wt_path = git_ops.persistent_offline_deep_worktree_path

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(_VERIFY_USAGE_DUMP.encode(), b''))
    mock_proc.returncode = 64

    with patch(
        'orchestrator.offline_lane.asyncio.create_subprocess_exec',
        return_value=mock_proc,
    ):
        await worker._handle_red_run(wt_path, head)

    # Zero fix tasks filed, zero fingerprints opened.
    cast(AsyncMock, worker.task_client).submit_fix_task.assert_not_awaited()
    assert worker.open_fix_tasks == {}
    # Zero escalations of ANY kind reached the real queue (no
    # orchestrator-offline-lane info escalation either).
    assert cast(EscalationQueue, worker.escalation_queue).get_pending() == []
