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
from escalation.queue import EscalationQueue

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.harness import Harness
from orchestrator.offline_lane import OfflineLaneWorker

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
    the steady in-flight ``'in-progress'`` state.
    """
    config = MagicMock()
    config.project_root = git_ops.project_root
    config.git = GitConfig(**(git_overrides or {}))

    if task_client is None:
        task_client = AsyncMock()
        task_client.get_status = AsyncMock(return_value='in-progress')

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


async def _run_one_lane_pass(worker: OfflineLaneWorker, *, timeout: float = 5.0) -> None:
    """Drive worker.run() as a real background task for exactly one pass.

    Wraps whatever ``suite_runner`` is currently on *worker* with a call
    counter and cancels the loop task as soon as the FIRST pass completes —
    modeled on ``test_offline_lane.py``'s
    ``test_loop_coalesces_to_exactly_one_rerun`` cancel-after-N pattern.
    Requires ``worker._dirty`` (or the wake event) to already be set by a
    prior trigger, exactly as production wiring would leave it.
    """
    inner_runner = worker.suite_runner
    done = asyncio.Event()
    count = {'n': 0}

    async def _counting_runner(wt, head, threads):
        result = await inner_runner(wt, head, threads)
        count['n'] += 1
        if count['n'] == 1:
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
