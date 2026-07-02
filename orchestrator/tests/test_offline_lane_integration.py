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
    worker: OfflineLaneWorker, expected_passes: int, *, timeout: float = 5.0,
) -> None:
    """Drive worker.run() as a real background task until *expected_passes*
    suite_runner calls have COMPLETED, then cancel the loop task.

    Wraps whatever ``suite_runner`` is currently on *worker* with a call
    counter — modeled on ``test_offline_lane.py``'s
    ``test_loop_coalesces_to_exactly_one_rerun`` cancel-after-N pattern,
    generalized from a fixed N=1 to an arbitrary pass count (B1's single
    pass via :func:`_run_one_lane_pass`; B2's held-pass-plus-one-coalesced-
    rerun burst uses N=2). Requires ``worker._dirty`` (or the wake event) to
    already be set by a prior trigger, exactly as production wiring would
    leave it.
    """
    inner_runner = worker.suite_runner
    done = asyncio.Event()
    count = {'n': 0}

    async def _counting_runner(wt, head, threads):
        result = await inner_runner(wt, head, threads)
        count['n'] += 1
        if count['n'] >= expected_passes:
            done.set()
        return result

    worker.suite_runner = _counting_runner
    task = asyncio.create_task(worker.run())
    try:
        await asyncio.wait_for(done.wait(), timeout=timeout)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def _run_one_lane_pass(worker: OfflineLaneWorker, *, timeout: float = 5.0) -> None:
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

    async def wait_entered(self, timeout: float = 5.0) -> None:
        """Block until the held first call has actually started (is in-flight)."""
        await asyncio.wait_for(self._entered.wait(), timeout=timeout)


async def _assert_never_a_gate(
    harness: Harness,
    worker: OfflineLaneWorker,
    runner: _ControllableSuiteRunner,
    repo: Path,
    git_ops: GitOps,
    caplog,
) -> None:
    """Assert the never-a-gate invariant (C7) while a lane run is held in-flight.

    Called with the FIRST pass already confirmed in-flight (the caller
    awaited ``runner.wait_entered()``) and released only after this
    returns, so the bounded ``wait_for`` below proves the synchronous
    on_merge_landed fan-out never blocks on it.

    (1)+(2) ``harness._note_merge_all`` — the exact ``on_merge_landed``
    callback ``SpeculativeMergeWorker`` invokes (``harness.py:4979``) —
    must return promptly and must set ``worker._dirty`` (arming a
    coalesced re-run).
    (3) A raising notifiee is fail-open: ``_note_offline_lane``'s own
    try/except (``harness.py:5072-5079``) swallows it — the SAME shape
    ``SpeculativeMergeWorker`` independently wraps this exact call in
    (``merge_queue.py:10578-10596``, belt-and-suspenders) — so the merge
    still lands (the service-restart coordinator fan-out still runs).
    (4) Neither case ever halts the merge queue or files an escalation —
    the only halt/gate-adjacent side effects reachable from this call path.
    """
    assert harness.get_merge_halt_status() == {'wired': False}

    base_sha = await git_ops.get_main_sha()
    head_sha = await _advance_main(repo)
    await asyncio.wait_for(
        harness._note_merge_all('task-2', base_sha, head_sha), timeout=0.5,
    )
    assert worker._dirty is True, (
        'a landed advance during an in-flight run must arm a coalesced re-run'
    )
    assert harness.get_merge_halt_status() == {'wired': False}
    assert worker.escalation_queue.get_pending() == []

    async def _raising_notifiee(task_id: str, base: str, head: str) -> None:
        await worker.on_post_merge(task_id, base, head)
        raise RuntimeError('lane boom')

    harness._offline_lane_notifiee = _raising_notifiee
    coord = MagicMock()
    coord.note_merge = AsyncMock()
    harness._service_restart_coordinators = [coord]

    base2 = head_sha
    head2 = await _advance_main(repo)
    with caplog.at_level(logging.WARNING):
        await harness._note_merge_all('task-3', base2, head2)  # must not raise

    assert 'fail-open' in caplog.text
    coord.note_merge.assert_awaited_once_with(
        'task-3', base2, head2, prefetched_diff=[],
    )
    assert harness.get_merge_halt_status() == {'wired': False}
    assert worker.escalation_queue.get_pending() == []


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


def _submitted_fix_task_arguments(task_client) -> dict:
    """Assert ``submit_fix_task`` was awaited exactly once; return its sole argument block."""
    task_client.submit_fix_task.assert_awaited_once()
    return task_client.submit_fix_task.await_args.args[0]


def _l0_info_escalations(escalation_queue: EscalationQueue, task_id: str) -> list:
    """Pending L0 (agent→steward) info escalations filed for *task_id*."""
    return escalation_queue.get_by_task(task_id, status='pending', level=0)


async def _drive_reds(
    harness: Harness, repo: Path, worker: OfflineLaneWorker, failing_ids: list[str], n: int,
) -> None:
    """Drive *n* same-failing-set confirmed-red advances through the real chain.

    Injects the SAME confirmed break once (:func:`_inject_red`) then, for
    *n* iterations, drives a real advance + the real on_merge_landed fan-out
    (:func:`_drive_advance`, mirroring B1-B4) followed by one full pass
    (:func:`_run_one_lane_pass`) — the same-fingerprint repeat-red sequence
    B5's dedup and B7's stall-promotion scenarios both need.
    """
    _inject_red(worker, failing_ids)
    for _ in range(n):
        await _drive_advance(harness, repo)
        await _run_one_lane_pass(worker)


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


@pytest.mark.asyncio
async def test_b3_never_a_gate(harness, git_ops, repo, tmp_path, caplog):
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

    await _assert_never_a_gate(harness, worker, runner, repo, git_ops, caplog)

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
    escalations = _l0_info_escalations(worker.escalation_queue, task_id)
    assert len(escalations) == 1
    assert escalations[0].severity == 'info'
    assert escalations[0].agent_role == 'orchestrator-offline-lane'

    # The merge queue itself is left completely untouched by the red path.
    assert harness.get_merge_halt_status() == {'wired': False}


# ---------------------------------------------------------------------------
# B5 — dedup: a same-set recurrence updates rather than duplicates
# ---------------------------------------------------------------------------


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

    worker.task_client.submit_fix_task.assert_awaited_once()
    worker.task_client.append_suspect_range.assert_awaited_once()

    fp = compute_failing_test_set_fingerprint(failing_ids)
    task_id = worker.open_fix_tasks[fp]
    escalations = _l0_info_escalations(worker.escalation_queue, task_id)
    assert len(escalations) == 1, (
        'a same-set recurrence must not raise a second L0 escalation'
    )


# ---------------------------------------------------------------------------
# B6 — flake filtered: fail-then-pass on confirmation is not a red
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_b6_flake_filtered_by_confirmation_rerun(harness, git_ops, repo, tmp_path, caplog):
    """B6 (PRD §8) — a run that fails, then passes on the isolated
    confirmation re-run, is intermittent nondeterminism, never a genuine
    break: no fix task and no escalation are ever raised."""
    worker = _build_worker(git_ops, tmp_path)
    _wire_lane(harness, worker)
    _inject_flake(worker)

    with caplog.at_level(logging.INFO):
        await _drive_advance(harness, repo)
        await _run_one_lane_pass(worker)

    assert 'intermittent nondeterminism' in caplog.text
    worker.task_client.submit_fix_task.assert_not_awaited()
    assert worker.open_fix_tasks == {}
    assert worker.escalation_queue.get_pending() == []
