"""Tests for warm-lane pool wiring in Harness.__init__ (task 1788, step-17).

Verifies that Harness constructs GitOps with warm_lane_pool_size=
max_concurrent_tasks when git.warm_lane_pool=True, and with size=0
(pool disabled) when the knob is off.

Step-17: RED — Harness constructs GitOps without warm_lane_pool_size.
Step-18: GREEN — Harness passes warm_lane_pool_size from max_concurrent_tasks.
Step-5 (1881): RED — B1 cancelled workflow must free lane via run() finally.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.queue import EscalationQueue

from orchestrator.config import GitConfig, OrchestratorConfig, VerifyRunnerConfig
from orchestrator.harness import Harness
from orchestrator.warm_lane_pool import WarmLanePool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_harness(config: OrchestratorConfig) -> Harness:
    """Construct a Harness with heavy constructors patched out."""
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        return Harness(config)


def _make_config(
    *,
    max_concurrent_tasks: int,
    warm_lane_pool: bool,
    spare_warm_lanes: int = 0,
    tmp_path: Path,
) -> OrchestratorConfig:
    """Build a minimal OrchestratorConfig with the given warm-lane settings."""
    # Create a minimal git repo directory so GitOps doesn't fail on init
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / '.git').mkdir()  # bare minimum to satisfy path checks
    return OrchestratorConfig(
        project_root=repo,
        max_concurrent_tasks=max_concurrent_tasks,
        git=GitConfig(warm_lane_pool=warm_lane_pool, spare_warm_lanes=spare_warm_lanes),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHarnessWarmLaneWiring:
    """Harness sizes the pool from max_concurrent_tasks at startup (PRD D9)."""

    def test_pool_sized_from_max_concurrent_tasks(self, tmp_path: Path):
        """With warm_lane_pool=True and max_concurrent_tasks=7, pool.size == 7."""
        config = _make_config(
            max_concurrent_tasks=7,
            warm_lane_pool=True,
            tmp_path=tmp_path,
        )
        harness = _build_harness(config)

        assert harness.git_ops.warm_lane_pool is not None, (
            'warm_lane_pool should be a WarmLanePool, not None'
        )
        assert isinstance(harness.git_ops.warm_lane_pool, WarmLanePool)
        assert harness.git_ops.warm_lane_pool.size == 7

    def test_pool_none_when_knob_off(self, tmp_path: Path):
        """With warm_lane_pool=False, pool is None regardless of max_concurrent_tasks."""
        config = _make_config(
            max_concurrent_tasks=7,
            warm_lane_pool=False,
            tmp_path=tmp_path,
        )
        harness = _build_harness(config)

        assert harness.git_ops.warm_lane_pool is None, (
            'warm_lane_pool should be None when knob is off'
        )

    def test_pool_size_matches_max_concurrent_tasks_3(self, tmp_path: Path):
        """Pool size tracks max_concurrent_tasks=3 (default-ish value)."""
        config = _make_config(
            max_concurrent_tasks=3,
            warm_lane_pool=True,
            tmp_path=tmp_path,
        )
        harness = _build_harness(config)

        assert harness.git_ops.warm_lane_pool is not None
        assert harness.git_ops.warm_lane_pool.size == 3

    def test_pool_none_when_max_concurrent_tasks_is_zero(self, tmp_path: Path):
        """max_concurrent_tasks=0 → pool size=0 → pool is None (always exhausted)."""
        config = _make_config(
            max_concurrent_tasks=0,
            warm_lane_pool=True,
            tmp_path=tmp_path,
        )
        harness = _build_harness(config)

        # size=0 → GitOps treats it as disabled → None
        assert harness.git_ops.warm_lane_pool is None

    def test_spare_warm_lanes_adds_to_pool_size(self, tmp_path: Path):
        """spare_warm_lanes=3 adds headroom: pool.size == max_concurrent_tasks + spare."""
        config = _make_config(
            max_concurrent_tasks=7,
            warm_lane_pool=True,
            spare_warm_lanes=3,
            tmp_path=tmp_path,
        )
        harness = _build_harness(config)

        assert harness.git_ops.warm_lane_pool is not None
        assert harness.git_ops.warm_lane_pool.size == 10  # 7 + 3

    def test_spare_warm_lanes_zero_is_byte_identical(self, tmp_path: Path):
        """spare_warm_lanes=0 (default) → pool.size == max_concurrent_tasks (no change)."""
        config = _make_config(
            max_concurrent_tasks=5,
            warm_lane_pool=True,
            spare_warm_lanes=0,
            tmp_path=tmp_path,
        )
        harness = _build_harness(config)

        assert harness.git_ops.warm_lane_pool is not None
        assert harness.git_ops.warm_lane_pool.size == 5  # 5 + 0

    def test_spare_warm_lanes_no_effect_when_pool_off(self, tmp_path: Path):
        """spare_warm_lanes=5 has no effect when warm_lane_pool=False → pool is None."""
        config = _make_config(
            max_concurrent_tasks=4,
            warm_lane_pool=False,
            spare_warm_lanes=5,
            tmp_path=tmp_path,
        )
        harness = _build_harness(config)

        assert harness.git_ops.warm_lane_pool is None


# ---------------------------------------------------------------------------
# Helpers for spec-pool tests
# ---------------------------------------------------------------------------


def _make_runner(name: str) -> VerifyRunnerConfig:
    """Build a minimal enabled VerifyRunnerConfig."""
    return VerifyRunnerConfig(
        name=name,
        ssh_host=f'{name}.example.com',
        git_remote=f'remote-{name}',
    )


def _make_spec_config(
    *,
    merge_spec_warm_lane_pool: bool,
    verify_runners: list[VerifyRunnerConfig],
    tmp_path: Path,
) -> OrchestratorConfig:
    """Build OrchestratorConfig with the spec knob + verify_runners for K tests."""
    repo = tmp_path / 'repo'
    repo.mkdir(exist_ok=True)
    (repo / '.git').mkdir(exist_ok=True)
    return OrchestratorConfig(
        project_root=repo,
        max_concurrent_tasks=4,
        git=GitConfig(merge_spec_warm_lane_pool=merge_spec_warm_lane_pool),
        verify_runners=verify_runners,
    )


# ---------------------------------------------------------------------------
# Step-5 RED / Step-6 GREEN — spec pool sized from shared K source
# ---------------------------------------------------------------------------


class TestHarnessSpecPoolWiring:
    """Harness passes merge_spec_warm_lane_pool_size=K to GitOps (step-5 RED, step-6 GREEN).

    K = 1 + len(config.enabled_verify_runners) — the SAME expression as
    speculation_depth passed to SpeculativeMergeWorker — so the spec pool size
    and the worker cap derive from one source and cannot drift.
    """

    def test_spec_pool_sized_from_k_with_runners(self, tmp_path: Path):
        """K=3 (1+2 runners) → spec pool size==3 when knob on."""
        runners = [_make_runner('r1'), _make_runner('r2')]
        config = _make_spec_config(
            merge_spec_warm_lane_pool=True,
            verify_runners=runners,
            tmp_path=tmp_path,
        )
        harness = _build_harness(config)

        assert harness.git_ops.spec_warm_lane_pool is not None, (
            'spec_warm_lane_pool should be a WarmLanePool when knob on and K>0'
        )
        assert isinstance(harness.git_ops.spec_warm_lane_pool, WarmLanePool)
        # K = 1 + len(enabled_verify_runners) = 1 + 2 = 3
        expected_k = 1 + len(config.enabled_verify_runners)
        assert harness.git_ops.spec_warm_lane_pool.size == expected_k, (
            f'spec pool size must equal K={expected_k}'
        )

    def test_spec_pool_k1_no_runners(self, tmp_path: Path):
        """K=1 (no remote runners) → spec pool size==1 when knob on."""
        config = _make_spec_config(
            merge_spec_warm_lane_pool=True,
            verify_runners=[],
            tmp_path=tmp_path,
        )
        harness = _build_harness(config)

        assert harness.git_ops.spec_warm_lane_pool is not None
        # K = 1 + 0 = 1
        assert harness.git_ops.spec_warm_lane_pool.size == 1

    def test_spec_pool_none_when_knob_off(self, tmp_path: Path):
        """spec_warm_lane_pool is None when merge_spec_warm_lane_pool=False."""
        runners = [_make_runner('r1'), _make_runner('r2')]
        config = _make_spec_config(
            merge_spec_warm_lane_pool=False,
            verify_runners=runners,
            tmp_path=tmp_path,
        )
        harness = _build_harness(config)

        assert harness.git_ops.spec_warm_lane_pool is None, (
            'spec_warm_lane_pool must be None when knob off'
        )


# ===========================================================================
# Step-5 (1881): RED — B1: workflow.run() finally releases lane on CANCELLED exit
# ===========================================================================


async def _init_git_repo(repo: Path) -> None:
    from orchestrator.git_ops import _run
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'init'], cwd=repo)


@pytest.mark.asyncio
class TestCancelledWorkflowLaneRelease:
    """B1 (T4): workflow.run() finally must release the warm lane on CANCELLED exit.

    Diff 3, sub-parts (i) and (ii):
    (i) _handle_cancelled_terminal_exit must call self._enter_phase(CANCELLED)
        so that workflow.state == WorkflowState.CANCELLED after the run.
    (ii) run() finally must call git_ops.release_lane_for_terminal_task(self.task_id)
         when self.state in (DONE, CANCELLED) and not _worktree_external.

    RED: neither behaviour is present; state stays PLAN and lane stays ASSIGNED.
    """

    async def test_cancelled_workflow_frees_lane(self, tmp_path: Path):
        """Cancelled terminal exit: (a) state==CANCELLED, (b) lane freed via primitive."""
        from _workflow_helpers import FakeBriefing, FakeMcp, FakeScheduler
        from escalation.queue import EscalationQueue

        from orchestrator.git_ops import GitOps
        from orchestrator.scheduler import TaskAssignment
        from orchestrator.warm_lane_pool import LaneState
        from orchestrator.workflow import TaskWorkflow, WorkflowOutcome, WorkflowState

        # ── Setup: minimal git repo ─────────────────────────────────────────
        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_git_repo(repo)

        # ── GitOps with warm lane pool ──────────────────────────────────────
        config = OrchestratorConfig(
            project_root=repo,
            max_concurrent_tasks=1,
            git=GitConfig(warm_lane_pool=True),
        )
        git_ops = GitOps(config.git, repo, warm_lane_pool_size=1)
        pool = git_ops.warm_lane_pool
        assert pool is not None

        # Pre-assign a lane for task '42' (simulates harness having acquired it
        # before dispatching the workflow; T4 scenario)
        result = await pool.acquire_for('42')
        assert result is not None
        lane = pool.assignment_for('42')
        assert lane is not None, 'setup: lane must be ASSIGNED for task 42'
        assert pool.state(lane) == LaneState.ASSIGNED

        # ── FakeScheduler returning 'cancelled' ─────────────────────────────
        scheduler = FakeScheduler()
        scheduler.statuses['42'] = ['cancelled']  # triggers TerminalExitRejection

        # ── Build workflow ──────────────────────────────────────────────────
        assignment = TaskAssignment(
            task_id='42',
            task={
                'id': '42', 'title': 'Test task', 'status': 'pending',
                'metadata': {}, 'dependencies': [],
            },
            modules=[],
        )
        workflow = TaskWorkflow(
            assignment=assignment,
            config=config,
            git_ops=git_ops,
            scheduler=scheduler,
            briefing=FakeBriefing(),
            mcp=FakeMcp(),
            escalation_queue=EscalationQueue(tmp_path / 'esc'),
            merge_queue=asyncio.Queue(),
            merge_worker=None,
        )

        # ── Run ─────────────────────────────────────────────────────────────
        outcome = await workflow.run()

        # (a) state must be CANCELLED (RED: _handle_cancelled_terminal_exit
        #     doesn't call _enter_phase(CANCELLED))
        assert workflow.state == WorkflowState.CANCELLED, (
            f'Expected WorkflowState.CANCELLED after cancelled run, got {workflow.state!r}'
        )
        # (b) lane must be freed (RED: run() finally doesn't call
        #     release_lane_for_terminal_task)
        assert pool.assignment_for('42') is None, (
            'run() finally must release the lane for a CANCELLED workflow exit'
        )
        assert pool.state(lane) == LaneState.FREE, (
            f'Lane must be FREE after CANCELLED exit, got {pool.state(lane)!r}'
        )
        # Sanity: correct outcome returned
        assert outcome == WorkflowOutcome.CANCELLED


# ===========================================================================
# Step-3 (1913): RED — B1: run() finally skips release on in-flight hard-cancel
# ===========================================================================


@pytest.mark.asyncio
class TestB1InFlightHardCancelSkipsRelease:
    """β (task 1913): B1 (workflow.run() finally) must SKIP release when an
    asyncio.CancelledError is propagating through the finally block.

    B1 condition (workflow.py:2002):
        if self.state in (DONE, CANCELLED) and not self._worktree_external:
            await git_ops.release_lane_for_terminal_task(...)

    β adds a third guard:  and not _hard_cancel
    where _hard_cancel = sys.exc_info()[0] is asyncio.CancelledError (in-flight).

    A normal DONE/authoritative-CANCELLED run() return has exc_info==None →
    release fires (unchanged, regression-guarded by test_cancelled_workflow_frees_lane).
    An in-flight asyncio.CancelledError (process teardown) suppresses the release.
    """

    async def test_b1_run_finally_skips_release_on_inflight_hard_cancel(
        self, tmp_path: Path
    ):
        """B1 guard: in-flight CancelledError prevents release_lane_for_terminal_task.

        Setup: force run() finally to execute with self.state==CANCELLED AND an
        in-flight asyncio.CancelledError by monkeypatching _setup_worktree_and_artifacts
        (the first await in run()'s try block) to set state then raise CancelledError.

        RED: pre-guard B1 evaluates self.state in (DONE, CANCELLED) and not
        _worktree_external → True → calls release_lane_for_terminal_task (spy fires).
        GREEN: _hard_cancel=True (in-flight) → condition is False → spy not called.

        Contrasting normal-return CANCELLED case (release DOES fire, exc_info==None)
        is covered by TestCancelledWorkflowLaneRelease.test_cancelled_workflow_frees_lane,
        guarding that β does not over-suppress the genuine authoritative-cancel path.

        NOTE — intentionally-unreachable state combination:
        In production, state==CANCELLED can only be written by
        _handle_cancelled_terminal_exit (workflow.py:7265), which is never reached
        during a hard-cancel because the asyncio.CancelledError (a BaseException)
        propagates directly through run()'s ``except Exception`` guard.  So the
        combination ``state==CANCELLED AND exc_info()==CancelledError`` cannot
        occur on a real code path — the ``self.state in (DONE, CANCELLED)`` branch
        of the B1 condition is already False for hard-cancels in practice.
        The test nonetheless exercises this combination to lock the defensive
        ``_hard_cancel`` guard, ensuring it does not silently disappear under
        refactoring.  A future reader should treat this as intentional belt-and-
        suspenders for an unreachable production state, not a test of a live path.
        """
        from unittest.mock import AsyncMock, MagicMock

        from _workflow_helpers import FakeBriefing, FakeMcp, FakeScheduler
        from escalation.queue import EscalationQueue

        from orchestrator.git_ops import GitOps
        from orchestrator.scheduler import TaskAssignment
        from orchestrator.workflow import TaskWorkflow, WorkflowState

        # ── Setup: minimal git repo ─────────────────────────────────────────
        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_git_repo(repo)

        config = OrchestratorConfig(
            project_root=repo,
            max_concurrent_tasks=1,
            git=GitConfig(warm_lane_pool=True),
        )
        git_ops = GitOps(config.git, repo, warm_lane_pool_size=1)

        # ── Build TaskWorkflow ──────────────────────────────────────────────
        scheduler = FakeScheduler()
        assignment = TaskAssignment(
            task_id='55',
            task={
                'id': '55', 'title': 'B1 hard-cancel test', 'status': 'pending',
                'metadata': {}, 'dependencies': [],
            },
            modules=[],
        )
        workflow = TaskWorkflow(
            assignment=assignment,
            config=config,
            git_ops=git_ops,
            scheduler=scheduler,
            briefing=FakeBriefing(),
            mcp=FakeMcp(),
            escalation_queue=EscalationQueue(tmp_path / 'esc'),
            merge_queue=asyncio.Queue(),
            merge_worker=None,
        )

        # ── Stub finally-block helpers so they don't crash on a half-set-up workflow
        # _steward=None by default (no stop() call issued) — nothing to do.
        # _maybe_cleanup_done_worktree and _cleanup_config_dir are replaced
        # with no-ops so the finally completes cleanly before we reach the B1 check.
        workflow._maybe_cleanup_done_worktree = AsyncMock()  # type: ignore[method-assign]
        workflow._cleanup_config_dir = MagicMock()  # type: ignore[method-assign]

        # _worktree_external is False by default (line 609) — B1 condition reads it.
        assert not workflow._worktree_external

        # ── Spy on release_lane_for_terminal_task ───────────────────────────
        release_spy = AsyncMock(return_value=False)
        git_ops.release_lane_for_terminal_task = release_spy  # type: ignore[method-assign]

        # ── Force the run() try-block to raise CancelledError with state==CANCELLED
        # Monkeypatch _setup_worktree_and_artifacts (first await in the try block):
        # set workflow.state to CANCELLED THEN raise asyncio.CancelledError so the
        # finally block sees the terminal state AND an in-flight exception.
        async def _fake_setup_raises_cancelled(branch_name: str) -> None:
            workflow.state = WorkflowState.CANCELLED
            raise asyncio.CancelledError('simulated hard-cancel')

        workflow._setup_worktree_and_artifacts = _fake_setup_raises_cancelled  # type: ignore[method-assign]

        # ── Run — expect CancelledError to propagate ────────────────────────
        with pytest.raises(asyncio.CancelledError):
            await workflow.run()

        # B1 β-guard: in-flight CancelledError must suppress the release.
        # RED: pre-guard B1 condition is True (state==CANCELLED, not external) →
        #      release_spy is called (fails assert_not_called).
        # GREEN: _hard_cancel=True → condition is False → spy not called.
        release_spy.assert_not_called()


# ===========================================================================
# Step-7 (1881) / Step-1 (1913): β — B2 synthetic-CANCELLED contract
# ===========================================================================


@pytest.mark.asyncio
class TestHardCancelLaneRelease:
    """β (task 1913): hard-cancel → synthetic report must NOT trigger eager B2 release.

    B2 (harness.py:3929 finally) gates the warm-lane release on
    ``report.outcome in (DONE, CANCELLED)``.  β adds a second guard:
    ``and not report.synthetic_cancel``.

    A synthetic hard-cancel (asyncio.CancelledError → harness.py:3896) is process
    teardown, NOT "work finished and discardable".  Pre-fix B2 fires
    release_lane_for_terminal_task → git branch -D task/<id>, deleting a
    still-unmerged branch.  Post-fix B2 is skipped for synthetic reports; the lane
    stays ASSIGNED and is reclaimable by the periodic terminal-lane reconciler /
    next acquire.  β alone only DELAYS deletion to the reconciler tick — durable
    branch survival across that reclaim is α (task 1912)'s contract, not β's.

    Contrasting case (non-synthetic CANCELLED still releases) is guarded by
    test_nonsynthetic_terminal_report_still_releases_lane (regression against
    over-suppression).
    """

    async def test_synthetic_hard_cancel_retains_branch_and_lane(self, tmp_path: Path):
        """Hard-cancel (asyncio.CancelledError): β contract — branch survives + lane reclaimable.

        RED triggers:
        1. ``report.synthetic_cancel`` AttributeError (field not yet on TaskReport).
        2. Pre-fix B2 fires release_lane_for_terminal_task → git branch -D task/99
           (the real branch is deleted, so git rev-parse --verify fails).

        GREEN: field added + B2 gated by ``and not report.synthetic_cancel``
        → branch retained, lane ASSIGNED (reclaimable, not permanently leaked).
        """
        from orchestrator.git_ops import _run as git_run
        from orchestrator.scheduler import TaskAssignment
        from orchestrator.warm_lane_pool import LaneState
        from orchestrator.workflow import WorkflowOutcome

        # ── Setup: real git repo ────────────────────────────────────────────
        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_git_repo(repo)

        # Create task/99 branch with an unmerged commit so rev-list main..task/99 > 0.
        # Pre-fix B2 fires git branch -D task/99 → ref deleted → RED.
        # Post-fix B2 is skipped → ref survives → GREEN.
        await git_run(['git', 'checkout', '-b', 'task/99'], cwd=repo)
        (repo / 'task99.txt').write_text('work in progress\n')
        await git_run(['git', 'add', '-A'], cwd=repo)
        await git_run(['git', 'commit', '-m', 'wip: task 99 work'], cwd=repo)
        await git_run(['git', 'checkout', 'main'], cwd=repo)

        config = OrchestratorConfig(
            project_root=repo,
            max_concurrent_tasks=1,
            git=GitConfig(warm_lane_pool=True),
        )

        # ── Harness with real git_ops (warm lane pool) ──────────────────────
        harness = _build_harness(config)
        pool = harness.git_ops.warm_lane_pool
        assert pool is not None

        # Pre-assign lane for task '99' (simulates harness having acquired it
        # before dispatching the slot; T5 hard-cancel scenario)
        await pool.acquire_for('99')
        lane = pool.assignment_for('99')
        assert lane is not None, 'setup: lane must be ASSIGNED for task 99'
        assert pool.state(lane) == LaneState.ASSIGNED

        # ── Configure mocked scheduler ──────────────────────────────────────
        # carries_substrate_probe → False so the substrate gate is skipped
        harness.scheduler.carries_substrate_probe.return_value = False  # type: ignore[attr-defined]
        # is_deterministic must return False to avoid the deterministic route
        # (MagicMock default is truthy → routes to _run_deterministic_slot
        #  which raises RuntimeError because _escalation_queue is None).
        harness.scheduler.is_deterministic.return_value = False  # type: ignore[attr-defined]

        # ── Patch TaskWorkflow to simulate a hard-cancel ────────────────────
        # workflow.run() raises CancelledError, exactly as hard_cancel_workflow()
        # produces when the soft-cancel is ignored past the poll limit.
        with patch('orchestrator.harness.TaskWorkflow') as MockWorkflow:
            mock_wf = MagicMock()
            mock_wf.run = AsyncMock(side_effect=asyncio.CancelledError())
            MockWorkflow.return_value = mock_wf

            assignment = TaskAssignment(
                task_id='99',
                task={
                    'id': '99', 'title': 'Hard-cancel test', 'status': 'pending',
                    'metadata': {}, 'dependencies': [],
                },
                modules=[],
            )
            sem = asyncio.Semaphore(1)
            report = await harness._run_slot(assignment, sem)

        # Must have returned the synthetic CANCELLED report from the except clause
        assert report is not None
        assert report.outcome == WorkflowOutcome.CANCELLED

        # β: synthetic_cancel must be True on the hard-cancel report
        # RED: AttributeError (field not yet on TaskReport)
        assert report.synthetic_cancel is True, (
            'hard-cancel must produce a synthetic report with synthetic_cancel=True'
        )

        # B2 β-fix: task/99 branch must SURVIVE.
        # Pre-fix: B2 fires release_lane_for_terminal_task → git branch -D task/99 → RED.
        # Post-fix: B2 gated by `and not report.synthetic_cancel` → skipped → GREEN.
        rc, _, _ = await git_run(
            ['git', 'rev-parse', '--verify', 'refs/heads/task/99'], cwd=repo,
        )
        assert rc == 0, (
            'task/99 branch must survive hard-cancel — '
            'B2 must not delete it when synthetic_cancel=True'
        )

        # Lane is RECLAIMABLE, not permanently leaked:
        # assignment intact → reconciler / next acquire can locate + reclaim.
        assert pool.assignment_for('99') is not None, (
            'lane assignment must survive hard-cancel '
            '(reclaimable by terminal-lane reconciler / next acquire)'
        )
        assert pool.state(lane) == LaneState.ASSIGNED, (
            f'lane must stay ASSIGNED after synthetic hard-cancel, got {pool.state(lane)!r}'
        )

    async def test_nonsynthetic_terminal_report_still_releases_lane(self, tmp_path: Path):
        """Non-synthetic CANCELLED (run() returns normally) still triggers B2 release.

        Regression guard: β must not over-suppress.  The full TaskReport built at
        harness.py:3842 when run() returns normally inherits synthetic_cancel=False
        (the default), so B2's eager release still fires for genuine terminal exits.

        The contrasting normal-return CANCELLED case where release DOES fire is also
        covered by TestCancelledWorkflowLaneRelease.test_cancelled_workflow_frees_lane
        (authoritative-cancel via B1 path).
        """
        from orchestrator.scheduler import TaskAssignment
        from orchestrator.warm_lane_pool import LaneState
        from orchestrator.workflow import WorkflowOutcome

        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_git_repo(repo)

        config = OrchestratorConfig(
            project_root=repo,
            max_concurrent_tasks=1,
            git=GitConfig(warm_lane_pool=True),
        )
        harness = _build_harness(config)
        pool = harness.git_ops.warm_lane_pool
        assert pool is not None

        tid = '88'
        await pool.acquire_for(tid)
        lane = pool.assignment_for(tid)
        assert lane is not None, f'setup: lane must be ASSIGNED for task {tid}'
        assert pool.state(lane) == LaneState.ASSIGNED

        harness.scheduler.carries_substrate_probe.return_value = False  # type: ignore[attr-defined]
        harness.scheduler.is_deterministic.return_value = False  # type: ignore[attr-defined]

        # Patch TaskWorkflow to return CANCELLED normally (not raise).
        # The harness builds the full report at harness.py:3842 with
        # synthetic_cancel defaulting to False.
        with patch('orchestrator.harness.TaskWorkflow') as MockWorkflow:
            mock_wf = MagicMock()
            mock_wf.run = AsyncMock(return_value=WorkflowOutcome.CANCELLED)
            MockWorkflow.return_value = mock_wf

            assignment = TaskAssignment(
                task_id=tid,
                task={
                    'id': tid, 'title': 'Non-synthetic cancel', 'status': 'pending',
                    'metadata': {}, 'dependencies': [],
                },
                modules=[],
            )
            sem = asyncio.Semaphore(1)
            report = await harness._run_slot(assignment, sem)

        # Full report: synthetic_cancel defaults to False
        # RED: AttributeError (field not yet on TaskReport)
        assert report is not None
        assert report.synthetic_cancel is False, (
            'non-synthetic CANCELLED report must have synthetic_cancel=False'
        )
        # B2 must STILL fire for a non-synthetic terminal report (regression guard)
        assert pool.assignment_for(tid) is None, (
            'B2 must release the lane for a non-synthetic terminal CANCELLED report'
        )
        assert pool.state(lane) == LaneState.FREE, (
            f'lane must be FREE after non-synthetic CANCELLED exit, got {pool.state(lane)!r}'
        )

    async def test_reconciler_reclaims_lane_after_synthetic_hard_cancel(self, tmp_path: Path):
        """β 'reclaimable' guarantee: periodic reconciler drives ASSIGNED → FREE.

        After a synthetic hard-cancel (B2 skipped), the lane stays ASSIGNED.
        This test drives _reconcile_terminal_lanes on the REAL primitive and
        asserts only β's non-leak contract: ASSIGNED → FREE (the reconciler
        can reclaim what β left behind).

        β alone only DELAYS deletion to the reconciler tick; it does NOT
        guarantee the branch survives that reconciler reclaim.  Durable branch
        survival across a reconciler reclaim is α (task 1912)'s contract
        (its branch-preservation guard in release_warm_lane).  That contract
        is covered by α's tests, not here.

        Closes the coverage gap flagged by the code reviewer: the 'reclaimable,
        not permanently leaked' half of β's contract was asserted by
        test_synthetic_hard_cancel_retains_branch_and_lane as pool ASSIGNED
        (can be reclaimed), but not as pool FREE (was actually reclaimed).
        """
        from unittest.mock import AsyncMock, MagicMock

        from orchestrator.git_ops import GitOps
        from orchestrator.git_ops import _run as git_run
        from orchestrator.warm_lane_pool import LaneState

        # ── Setup: repo + harness with pool ────────────────────────────────
        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_git_repo(repo)

        config = OrchestratorConfig(
            project_root=repo,
            max_concurrent_tasks=1,
            git=GitConfig(warm_lane_pool=True),
        )
        git_ops = GitOps(config.git, repo, warm_lane_pool_size=1)
        harness = _build_harness(config)
        harness.git_ops = git_ops
        pool = git_ops.warm_lane_pool
        assert pool is not None

        # ── Create task/99 branch beyond main (simulates unmerged work) ───
        # Realistic setup: β skipped B2 so the branch and lane are both
        # alive when the reconciler fires.  Branch survival after the
        # reconciler tick is α's job (task 1912), not β's — no branch
        # assertion here.
        await git_run(['git', 'checkout', '-b', 'task/99'], cwd=repo)
        (repo / 'recon.txt').write_text('reconciler test\n')
        await git_run(['git', 'add', '-A'], cwd=repo)
        await git_run(['git', 'commit', '-m', 'wip: task 99 reconciler test'], cwd=repo)
        await git_run(['git', 'checkout', 'main'], cwd=repo)

        # ── Simulate β post-hard-cancel state: ASSIGNED, branch alive ─────
        result = await pool.acquire_for('99')
        assert result is not None
        lane, _ = result
        assert pool.state(lane) == LaneState.ASSIGNED, (
            'post-cancel pool must be ASSIGNED (β contract)'
        )

        # ── Reconciler sees task/99 as 'cancelled', not live ─────────────
        harness.scheduler.get_statuses = AsyncMock(return_value=({'99': 'cancelled'}, None))
        harness.scheduler._dispatched = set()
        harness.event_store = MagicMock()

        # ── Drive _reconcile_terminal_lanes (real primitive, no mock) ─────
        await harness._reconcile_terminal_lanes()

        # Lane must be FREE (ASSIGNED → FREE via reconciler) — β's non-leak
        # contract.  The real release_lane_for_terminal_task runs here; without
        # α (1912) it will also delete the branch, but that is by design: β only
        # prevents the EAGER deletion at process-teardown time.  Full branch
        # survival requires α (task 1912) to land.
        assert pool.assignment_for('99') is None, (
            'reconciler must reclaim the lane that β left ASSIGNED after synthetic hard-cancel'
        )
        assert pool.state(lane) == LaneState.FREE, (
            f'lane must be FREE after reconciler, got {pool.state(lane)!r}'
        )


# ===========================================================================
# Step-9 (1881): RED — B3: train callbacks + redrive release lane on done-flip
# ===========================================================================


@pytest.mark.asyncio
class TestTrainCallbackLaneRelease:
    """B3 (T6/T7/T8): build_train_callback_factory must accept git_ops and
    release the warm lane when mark_member_done or redrive_member flips the task done.

    Diff 5a: widen build_train_callback_factory(scheduler) → (scheduler, git_ops)
    Diff 5b: after mark_done inside mark_member_done and redrive_member, call
             await git_ops.release_lane_for_terminal_task(mid)

    RED: factory takes only `scheduler`; closures never release the lane.
    """

    async def test_merge_train_member_done_frees_lane(self, tmp_path: Path):
        """(12) mark_member_done fires mark_done AND releases the lane."""
        from _workflow_helpers import FakeScheduler

        from orchestrator.harness import build_train_callback_factory
        from orchestrator.warm_lane_pool import LaneState

        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_git_repo(repo)

        config = OrchestratorConfig(
            project_root=repo,
            max_concurrent_tasks=1,
            git=GitConfig(warm_lane_pool=True),
        )
        from orchestrator.git_ops import GitOps
        git_ops = GitOps(config.git, repo, warm_lane_pool_size=1)
        pool = git_ops.warm_lane_pool
        assert pool is not None

        mid = '3459'
        # Pre-assign lane for the member task
        await pool.acquire_for(mid)
        lane = pool.assignment_for(mid)
        assert lane is not None
        assert pool.state(lane) == LaneState.ASSIGNED

        scheduler = FakeScheduler()
        scheduler.statuses[mid] = ['merge-deferred']  # task exists in scheduler

        # RED: build_train_callback_factory takes only scheduler → TypeError
        factory = build_train_callback_factory(scheduler, git_ops)
        callbacks = factory('train-1')

        await callbacks.mark_member_done(mid, 'abc123')

        # scheduler.mark_done must have been called
        assert scheduler.statuses.get(mid, [])[-1] == 'done', (
            'mark_member_done must flip the task to done'
        )
        # Lane must be freed (RED: closures don't call release_lane_for_terminal_task)
        assert pool.assignment_for(mid) is None, (
            'mark_member_done must release the warm lane after mark_done'
        )
        assert pool.state(lane) == LaneState.FREE

    async def test_merge_deferred_to_done_lane_freed_at_flip(self, tmp_path: Path):
        """(13) Lane stays ASSIGNED while MERGE_DEFERRED, freed only at the done-flip."""
        from _workflow_helpers import FakeScheduler

        from orchestrator.harness import build_train_callback_factory
        from orchestrator.warm_lane_pool import LaneState

        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_git_repo(repo)

        config = OrchestratorConfig(
            project_root=repo,
            max_concurrent_tasks=1,
            git=GitConfig(warm_lane_pool=True),
        )
        from orchestrator.git_ops import GitOps
        git_ops = GitOps(config.git, repo, warm_lane_pool_size=1)
        pool = git_ops.warm_lane_pool
        assert pool is not None

        mid = '7777'
        await pool.acquire_for(mid)
        lane = pool.assignment_for(mid)
        assert lane is not None
        assert pool.state(lane) == LaneState.ASSIGNED

        scheduler = FakeScheduler()
        scheduler.statuses[mid] = ['merge-deferred']

        factory = build_train_callback_factory(scheduler, git_ops)
        callbacks = factory('train-2')

        # Lane is still ASSIGNED before the done-flip
        assert pool.state(lane) == LaneState.ASSIGNED, (
            'lane must stay ASSIGNED while task is MERGE_DEFERRED (not yet done)'
        )

        # Trigger the done-flip
        await callbacks.mark_member_done(mid, 'def456')

        # Lane must be freed exactly at the done-flip (RED: not freed)
        assert pool.assignment_for(mid) is None, (
            'lane must be freed when mark_member_done flips to done'
        )
        assert pool.state(lane) == LaneState.FREE

    async def test_async_merge_queue_done_frees_lane(self, tmp_path: Path):
        """(14) redrive_member(found_on_main=True) fires mark_done AND releases the lane.

        This covers T6: merge_queue has no task-pool release, so B3 (redrive_member
        inside the factory) is the primary release path for async-merge-queue done events.
        """
        from _workflow_helpers import FakeScheduler

        from orchestrator.harness import build_train_callback_factory
        from orchestrator.warm_lane_pool import LaneState

        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_git_repo(repo)

        config = OrchestratorConfig(
            project_root=repo,
            max_concurrent_tasks=1,
            git=GitConfig(warm_lane_pool=True),
        )
        from orchestrator.git_ops import GitOps
        git_ops = GitOps(config.git, repo, warm_lane_pool_size=1)
        pool = git_ops.warm_lane_pool
        assert pool is not None

        mid = '3459'
        await pool.acquire_for(mid)
        lane = pool.assignment_for(mid)
        assert lane is not None
        assert pool.state(lane) == LaneState.ASSIGNED

        scheduler = FakeScheduler()
        scheduler.statuses[mid] = ['merge-deferred']  # member exists

        factory = build_train_callback_factory(scheduler, git_ops)
        callbacks = factory('train-3')

        # found_on_main=True triggers the mark_done path (T6 canonical scenario)
        assert callbacks.redrive_member is not None
        await callbacks.redrive_member(mid, True, 'ghi789')

        # scheduler.mark_done must have been called
        assert scheduler.statuses.get(mid, [])[-1] == 'done', (
            'redrive_member(found_on_main=True) must flip to done'
        )
        # Lane must be freed (RED: redrive_member closure never releases)
        assert pool.assignment_for(mid) is None, (
            'redrive_member must release the warm lane after mark_done(found_on_main)'
        )
        assert pool.state(lane) == LaneState.FREE


# ===========================================================================
# Step-11 (1881): RED — B3/T9: _mark_in_progress_done releases lane via primitive
# ===========================================================================


@pytest.mark.asyncio
class TestMarkInProgressDoneLaneRelease:
    """B3 (T9): _mark_in_progress_done must call release_lane_for_terminal_task(tid)
    after mark_done so that warm lanes whose in-memory assignment map was lost
    (post-restart) are freed via the on-disk plan.json backstop.

    Diff 5c: after mark_done(kind='found_on_main') inside _mark_in_progress_done,
             add await self.git_ops.release_lane_for_terminal_task(tid).

    RED: _mark_in_progress_done only calls the existing cleanup_worktree (which
    resolves via in-memory map — lost post-restart) and never the primitive, so
    the lane stays ASSIGNED forever.
    """

    async def test_reconciler_found_on_main_frees_lane(self, tmp_path: Path):
        """_mark_in_progress_done frees a lane whose in-memory assignment was lost."""
        from unittest.mock import AsyncMock

        from orchestrator.git_ops import GitOps
        from orchestrator.warm_lane_pool import LaneState

        # ── Setup: minimal git repo ─────────────────────────────────────────
        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_git_repo(repo)

        config = OrchestratorConfig(
            project_root=repo,
            max_concurrent_tasks=1,
            git=GitConfig(warm_lane_pool=True),
        )
        git_ops = GitOps(config.git, repo, warm_lane_pool_size=1)
        pool = git_ops.warm_lane_pool
        assert pool is not None

        tid = '3459'

        # Acquire a lane for the task (simulates pre-restart dispatch)
        result = await pool.acquire_for(tid)
        assert result is not None
        lane, _ = result
        assert pool.state(lane) == LaneState.ASSIGNED

        # Simulate post-restart: in-memory assignment map is cleared
        pool._assignments.clear()
        assert pool.assignment_for(tid) is None, 'setup: assignment map must be empty'

        # Write on-disk plan.json backstop so the primitive can still find the lane
        task_dir = lane / '.task'
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / 'plan.json').write_text('{"task_id": "3459"}')

        # ── Build Harness with real git_ops ─────────────────────────────────
        harness = _build_harness(config)
        harness.git_ops = git_ops
        # Scheduler.mark_done must be awaitable
        harness.scheduler.mark_done = AsyncMock()

        # ── Call _mark_in_progress_done ──────────────────────────────────────
        await harness._mark_in_progress_done(
            tid, sha='abc123', note='test-found-on-main', reason='found-on-main',
        )

        # scheduler.mark_done must have been called
        harness.scheduler.mark_done.assert_called_once()
        call_kwargs = harness.scheduler.mark_done.call_args
        assert call_kwargs.kwargs.get('kind') == 'found_on_main' or (
            len(call_kwargs.args) >= 2 and call_kwargs.args[1] == 'found_on_main'
        ), 'mark_done must be called with kind=found_on_main'

        # (RED) Lane must be freed via the on-disk backstop.
        # Fails: _mark_in_progress_done never calls release_lane_for_terminal_task.
        assert pool.assignment_for(tid) is None, (
            '_mark_in_progress_done must release the lane after mark_done '
            '(the in-memory map was lost post-restart; the primitive uses the '
            'on-disk plan.json backstop to locate and free the lane)'
        )
        assert pool.state(lane) == LaneState.FREE, (
            f'Lane must be FREE after found-on-main mark_done, got {pool.state(lane)!r}'
        )


# ===========================================================================
# Step-13 (1881): RED — A: status-keyed reconciler _reconcile_terminal_lanes
# ===========================================================================


@pytest.mark.asyncio
class TestTerminalLaneReconciler:
    """Layer A (invariant backstop): _reconcile_terminal_lanes releases any lane
    whose assigned task is terminal and NOT in scheduler._dispatched.

    Diff 7: add _reconcile_terminal_lanes near _reconcile_stranded_in_progress;
    wire into _stranded_reconcile_loop (after _reconcile_stranded_in_progress)
    and once at startup after _reap_orphan_worktrees.

    RED: _reconcile_terminal_lanes does not exist yet; calls raise AttributeError.
    """

    async def _make_harness_with_pool(
        self, tmp_path: Path, pool_size: int = 1
    ):
        """Build a harness backed by a real GitOps warm lane pool."""
        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_git_repo(repo)
        config = OrchestratorConfig(
            project_root=repo,
            max_concurrent_tasks=pool_size,
            git=GitConfig(warm_lane_pool=True),
        )
        from orchestrator.git_ops import GitOps
        git_ops = GitOps(config.git, repo, warm_lane_pool_size=pool_size)
        harness = _build_harness(config)
        harness.git_ops = git_ops
        return harness, git_ops, git_ops.warm_lane_pool

    async def test_terminal_lane_reconciler_frees_done_task_lane(self, tmp_path: Path):
        """(16) Lane ASSIGNED to a 'done' task (not dispatched) is freed + event emitted."""
        from unittest.mock import AsyncMock, MagicMock

        from orchestrator.event_store import EventType
        from orchestrator.warm_lane_pool import LaneState

        harness, _git_ops, pool = await self._make_harness_with_pool(tmp_path)
        assert pool is not None

        tid = '3459'
        result = await pool.acquire_for(tid)
        assert result is not None
        lane, _ = result
        assert pool.state(lane) == LaneState.ASSIGNED

        # Scheduler returns 'done'; _dispatched is empty (not a live workflow)
        harness.scheduler.get_statuses = AsyncMock(return_value=({tid: 'done'}, None))
        harness.scheduler._dispatched = set()

        # Wire a mock event_store to capture emissions
        harness.event_store = MagicMock()

        # RED: _reconcile_terminal_lanes does not exist yet → AttributeError
        await harness._reconcile_terminal_lanes()

        # Lane must be freed
        assert pool.state(lane) == LaneState.FREE, (
            f'done-task lane must be FREE after reconciler, got {pool.state(lane)!r}'
        )
        # Event must be emitted with reason='terminal-lane-reconciler'
        harness.event_store.emit.assert_called_once()
        call_args = harness.event_store.emit.call_args
        assert call_args.args[0] == EventType.worktree_reaped, (
            'reconciler must emit EventType.worktree_reaped'
        )
        data = call_args.kwargs.get('data', {})
        assert data.get('reason') == 'terminal-lane-reconciler', (
            f"data.reason must be 'terminal-lane-reconciler', got {data!r}"
        )

    async def test_terminal_lane_reconciler_skips_in_progress(self, tmp_path: Path):
        """(17) Lane ASSIGNED to an 'in-progress' task stays ASSIGNED (not terminal)."""
        from unittest.mock import AsyncMock

        from orchestrator.warm_lane_pool import LaneState

        harness, _git_ops, pool = await self._make_harness_with_pool(tmp_path)
        assert pool is not None

        tid = '7777'
        result = await pool.acquire_for(tid)
        assert result is not None
        lane, _ = result

        harness.scheduler.get_statuses = AsyncMock(return_value=({tid: 'in-progress'}, None))
        harness.scheduler._dispatched = set()

        await harness._reconcile_terminal_lanes()

        assert pool.state(lane) == LaneState.ASSIGNED, (
            'in-progress task lane must stay ASSIGNED — not a terminal status'
        )

    async def test_terminal_lane_reconciler_skips_dispatched(self, tmp_path: Path):
        """(18) 'done' task in _dispatched is NOT freed (live-acquire guard)."""
        from unittest.mock import AsyncMock

        from orchestrator.warm_lane_pool import LaneState

        harness, _git_ops, pool = await self._make_harness_with_pool(tmp_path)
        assert pool is not None

        tid = '8888'
        result = await pool.acquire_for(tid)
        assert result is not None
        lane, _ = result

        harness.scheduler.get_statuses = AsyncMock(return_value=({tid: 'done'}, None))
        # Branch IS in _dispatched → guard must prevent release
        harness.scheduler._dispatched = {tid}

        await harness._reconcile_terminal_lanes()

        assert pool.state(lane) == LaneState.ASSIGNED, (
            'done task in _dispatched must not be freed (live-acquire guard)'
        )

    async def test_terminal_lane_reconciler_aborts_on_empty_statuses(self, tmp_path: Path):
        """(19) Empty get_statuses (with or without error) aborts the sweep — never mass-free."""
        from unittest.mock import AsyncMock

        from orchestrator.warm_lane_pool import LaneState

        harness, _git_ops, pool = await self._make_harness_with_pool(tmp_path)
        assert pool is not None

        tid = '9999'
        result = await pool.acquire_for(tid)
        assert result is not None
        lane, _ = result

        harness.scheduler._dispatched = set()

        # Case A: empty with no error
        harness.scheduler.get_statuses = AsyncMock(return_value=({}, None))
        await harness._reconcile_terminal_lanes()
        assert pool.state(lane) == LaneState.ASSIGNED, (
            'empty get_statuses must abort sweep — lane must stay ASSIGNED (never mass-free)'
        )

        # Case B: empty with an error
        harness.scheduler.get_statuses = AsyncMock(
            return_value=({}, Exception('transient DB error'))
        )
        await harness._reconcile_terminal_lanes()
        assert pool.state(lane) == LaneState.ASSIGNED, (
            'errored get_statuses must abort sweep — lane must stay ASSIGNED'
        )

    async def test_terminal_lane_reconciler_cancelled_status_frees(self, tmp_path: Path):
        """(20) Lane ASSIGNED to a 'cancelled' task is freed."""
        from unittest.mock import AsyncMock

        from orchestrator.warm_lane_pool import LaneState

        harness, _git_ops, pool = await self._make_harness_with_pool(tmp_path)
        assert pool is not None

        tid = '1111'
        result = await pool.acquire_for(tid)
        assert result is not None
        lane, _ = result

        harness.scheduler.get_statuses = AsyncMock(return_value=({tid: 'cancelled'}, None))
        harness.scheduler._dispatched = set()

        await harness._reconcile_terminal_lanes()

        assert pool.state(lane) == LaneState.FREE, (
            f'cancelled task lane must be FREE after reconciler, got {pool.state(lane)!r}'
        )


# ===========================================================================
# Step-15 (1881): RED — T10: recovery-sweep must release terminal task lanes
# ===========================================================================


@pytest.mark.asyncio
class TestRecoveryTerminalTaskLaneRelease:
    """T10 (Recovery fix): _recover_crashed_tasks must release lanes whose task
    is already terminal (done/cancelled) instead of restoring the assignment.

    Diff 6: inside the 'if pool is not None and is_lane and recovery_id:' block,
    BEFORE pool.restore_assignment(...), add:
        term_status = await self.scheduler.get_status(recovery_id)
        if term_status in ('done', 'cancelled'):
            await self.git_ops.cleanup_worktree(entry, recovery_id)
            cleaned += 1; continue

    RED: recovery currently restores unconditionally (pool.restore_assignment
    is called regardless of task status).
    """

    async def _make_recovery_setup(
        self, tmp_path: Path, *, task_id: str = '3459'
    ):
        """Build harness with real git repo, GitOps pool, and a lane dir with
        plan.json carrying a completed step — the minimum _recover_crashed_tasks
        needs to reach the restore_assignment branch.

        Returns: (harness, git_ops, pool, lane_entry)
        """
        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_git_repo(repo)
        config = OrchestratorConfig(
            project_root=repo,
            max_concurrent_tasks=1,
            git=GitConfig(warm_lane_pool=True),
            worktree_identity_guard_enabled=False,  # skip identity guard for simplicity
        )
        from orchestrator.git_ops import GitOps
        git_ops = GitOps(config.git, repo, warm_lane_pool_size=1)
        pool = git_ops.warm_lane_pool
        assert pool is not None

        harness = _build_harness(config)
        harness.git_ops = git_ops

        # Create the lane dir and plan.json so _recover_crashed_tasks finds it
        lane_entry = git_ops.worktree_base / '_lane-0'
        lane_entry.mkdir(parents=True, exist_ok=True)
        task_dir = lane_entry / '.task'
        task_dir.mkdir(exist_ok=True)
        plan = {
            'task_id': task_id,
            'steps': [
                {
                    'id': 'step-1',
                    'type': 'test',
                    'description': 'already done step',
                    'status': 'done',
                },
            ],
        }
        (task_dir / 'plan.json').write_text(json.dumps(plan))

        # Registration guard (reify 4655/4947): default to "still registered"
        # so the T10 restore-path tests (this lane is plain mkdir'd, never
        # `git worktree add`ed) keep exercising the positive (non-terminal +
        # registered -> restore) path. The orphaned-lane invariant test
        # overrides this to False.
        git_ops._is_registered_worktree = AsyncMock(return_value=True)

        return harness, git_ops, pool, lane_entry

    async def test_recovery_releases_terminal_task_lane(self, tmp_path: Path):
        """(21) Done task: cleanup_worktree called, restore_assignment NOT called,
        task NOT stored in _recovered_plans."""
        harness, git_ops, pool, lane_entry = await self._make_recovery_setup(tmp_path)

        harness.scheduler.get_status = AsyncMock(return_value='done')

        with patch.object(git_ops, 'cleanup_worktree', new_callable=AsyncMock) as mock_cleanup, \
             patch.object(pool, 'restore_assignment') as mock_restore:
            await harness._recover_crashed_tasks()

        # Terminal task: cleanup must have been called (closes the T10 amplifier)
        mock_cleanup.assert_called_once()
        # restore_assignment must NOT have been called
        mock_restore.assert_not_called()
        # Task must NOT be stored for resumption
        assert '3459' not in harness._recovered_plans, (
            'terminal task lane must not be stored in _recovered_plans'
        )

    async def test_recovery_restores_non_terminal_task_lane(self, tmp_path: Path):
        """(22) Non-terminal task (in-progress): restore_assignment IS called;
        plan IS stored for resumption (regression guard — existing behavior preserved)."""
        harness, git_ops, pool, lane_entry = await self._make_recovery_setup(tmp_path)

        harness.scheduler.get_status = AsyncMock(return_value='in-progress')

        with patch.object(git_ops, 'cleanup_worktree', new_callable=AsyncMock), \
             patch.object(pool, 'restore_assignment') as mock_restore:
            await harness._recover_crashed_tasks()

        # Non-terminal: restore_assignment must be called (existing behaviour)
        mock_restore.assert_called_once()
        # Plan must be stored for resumption
        assert '3459' in harness._recovered_plans, (
            'in-progress task plan must be stored in _recovered_plans'
        )

    async def test_recovery_restores_on_status_read_failure(self, tmp_path: Path):
        """(23) Transient get_status=None: falls through to restore_assignment.

        Safe default: layer A (_reconcile_terminal_lanes) self-heals any
        genuinely-terminal lane on the next reconcile interval.
        """
        harness, git_ops, pool, lane_entry = await self._make_recovery_setup(tmp_path)

        # Transient failure: get_status returns None (not in ('done','cancelled'))
        harness.scheduler.get_status = AsyncMock(return_value=None)

        with patch.object(git_ops, 'cleanup_worktree', new_callable=AsyncMock), \
             patch.object(pool, 'restore_assignment') as mock_restore:
            await harness._recover_crashed_tasks()

        # Fail-safe: restore_assignment must be called (layer A self-heals later)
        mock_restore.assert_called_once()
        assert '3459' in harness._recovered_plans, (
            'task with None status must fall through to restore (safe default; A self-heals)'
        )

    async def test_recovery_skips_pin_for_unregistered_lane(self, tmp_path: Path):
        """(24) reify 4655/4947: an ORPHANED lane (mkdir'd, never `git worktree
        add`ed) must NOT be re-pinned even for a non-terminal task:
        restore_assignment is skipped, cleanup_worktree is not called either,
        the plan is still recovered, and the lane is left FREE so the
        create-once self-heal path repairs it on the next acquire.

        Sibling coverage: test_crash_recovery.py::TestRecoverCrashedTasksWarmLane::
        test_warm_lane_orphaned_registration_not_pinned exercises the same
        invariant with mock-based fixtures; keep both in sync if this
        invariant ever changes.
        """
        harness, git_ops, pool, lane_entry = await self._make_recovery_setup(tmp_path)

        git_ops._is_registered_worktree = AsyncMock(return_value=False)
        harness.scheduler.get_status = AsyncMock(return_value='in-progress')

        with patch.object(git_ops, 'cleanup_worktree', new_callable=AsyncMock) as mock_cleanup, \
             patch.object(pool, 'restore_assignment') as mock_restore:
            await harness._recover_crashed_tasks()

        from orchestrator.warm_lane_pool import LaneState

        # Unregistered lane: restore_assignment must NOT be called
        mock_restore.assert_not_called()
        # Plan is still recovered — independent of the pin
        assert '3459' in harness._recovered_plans, (
            'plan must still be recovered even when the lane is left unpinned'
        )
        # Lane is not torn down either — left for create-once self-heal
        mock_cleanup.assert_not_called()
        # Lane must remain FREE (never pinned)
        assert pool.state(lane_entry) == LaneState.FREE


# ===========================================================================
# Step-17 (1881): RED — disk-backstop opt-in wiring in _mark_in_progress_done
# ===========================================================================


@pytest.mark.asyncio
class TestMarkInProgressDoneDiskBackstopWiring:
    """_mark_in_progress_done must call release_lane_for_terminal_task with
    allow_disk_backstop=True — it is the ONE legitimate disk-backstop caller
    (the lost-map / post-restart T9 path) after the default flips to in-memory-only.

    Diff 18(ii): harness._mark_in_progress_done calls
        await self.git_ops.release_lane_for_terminal_task(tid, allow_disk_backstop=True)

    RED: current _mark_in_progress_done calls without allow_disk_backstop kwarg;
    spy sees a call without the kwarg → assert_called_once_with fails.
    """

    async def test_mark_in_progress_done_opts_into_disk_backstop(
        self, tmp_path: Path,
    ):
        """_mark_in_progress_done must pass allow_disk_backstop=True to the primitive."""
        from unittest.mock import AsyncMock

        from orchestrator.git_ops import GitOps

        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_git_repo(repo)

        config = OrchestratorConfig(
            project_root=repo,
            max_concurrent_tasks=1,
            git=GitConfig(warm_lane_pool=True),
        )
        git_ops = GitOps(config.git, repo, warm_lane_pool_size=1)

        harness = _build_harness(config)
        harness.git_ops = git_ops
        harness.scheduler.mark_done = AsyncMock()

        # Replace the primitive with a spy so we can inspect the kwargs
        release_spy: AsyncMock = AsyncMock(return_value=True)
        git_ops.release_lane_for_terminal_task = release_spy  # type: ignore[method-assign]

        # Also stub cleanup_worktree to avoid real git subprocess calls
        git_ops.cleanup_worktree = AsyncMock()  # type: ignore[method-assign]

        await harness._mark_in_progress_done(
            '3459', sha='abc123', note='found-on-main', reason='found-on-main',
        )

        # Must have been called EXACTLY once with allow_disk_backstop=True
        release_spy.assert_called_once_with('3459', allow_disk_backstop=True)


# ===========================================================================
# Step-5 (1933): RED — Harness._warm_lane_reclaim_candidates + _is_branch_dispatched
# ===========================================================================


@pytest.mark.asyncio
class TestHarnessReclaimCandidateProvider:
    """Tests for Harness._warm_lane_reclaim_candidates (task 1933, step-5).

    The provider mirrors _reconcile_terminal_lanes INVERTED: keep branches
    whose status is known and NOT in {done, cancelled}; abort to empty-set on
    resolver_failed (fail-safe, never reclaim on degraded read).

    All tests RED today: _warm_lane_reclaim_candidates does not exist.
    """

    def _make_harness_with_stub_scheduler(
        self, tmp_path: Path, get_statuses_return: tuple
    ) -> Harness:
        """Build a harness with scheduler.get_statuses stubbed."""
        config = _make_config(
            max_concurrent_tasks=4,
            warm_lane_pool=True,
            tmp_path=tmp_path,
        )
        harness = _build_harness(config)
        harness.scheduler.get_statuses = AsyncMock(return_value=get_statuses_return)
        harness.scheduler._dispatched = set()
        return harness

    async def test_non_terminal_subset_returned(self, tmp_path: Path):
        """(a) blocked + pending returned; done + cancelled dropped."""
        statuses = {'A': 'blocked', 'B': 'done', 'C': 'pending', 'D': 'cancelled'}
        harness = self._make_harness_with_stub_scheduler(tmp_path, (statuses, None))

        result = await harness._warm_lane_reclaim_candidates(['A', 'B', 'C', 'D'])

        assert result == {'A', 'C'}, (
            f'provider must return only non-terminal candidates (blocked, pending), '
            f'got {result!r}'
        )

    async def test_merge_deferred_and_deferred_excluded_from_candidates(
        self, tmp_path: Path
    ):
        """Protected parked statuses must never be reclaim victims (task 2018).

        Mirrors the merge-deferred early-return guard in
        _reconcile_one_stranded (harness.py:2437): a merge-deferred branch is
        train-parked (PRD § 9.8) and a deferred branch is human-parked —
        both have worktrees owned by a non-scheduler party and must survive
        intact. Neither is terminal, so without this exclusion they would
        pass the TERMINAL_STATUSES filter and become eligible reclaim
        victims. A genuinely-active in-progress branch must still be
        returned, guarding against over-exclusion.
        """
        statuses = {'A': 'merge-deferred', 'B': 'in-progress', 'C': 'deferred'}
        harness = self._make_harness_with_stub_scheduler(tmp_path, (statuses, None))

        result = await harness._warm_lane_reclaim_candidates(['A', 'B', 'C'])

        assert 'A' not in result, (
            f'merge-deferred (train-parked) branch must never be a reclaim '
            f'victim; result={result!r}'
        )
        assert 'C' not in result, (
            f'deferred (human-parked) branch must never be a reclaim victim; '
            f'result={result!r}'
        )
        assert result == {'B'}, (
            f'in-progress branch must still be reclaimable (no '
            f'over-exclusion); got {result!r}'
        )

    async def test_unknown_status_dropped(self, tmp_path: Path):
        """(b) Branch not in the statuses dict is silently dropped (unknown = skip)."""
        # 'X' not in returned statuses → dropped
        statuses = {'A': 'in-progress'}
        harness = self._make_harness_with_stub_scheduler(tmp_path, (statuses, None))

        result = await harness._warm_lane_reclaim_candidates(['A', 'X'])

        assert 'X' not in result, (
            f'branch with unknown/missing status must be dropped; result={result!r}'
        )
        assert 'A' in result, (
            f'in-progress branch must be included as non-terminal; result={result!r}'
        )

    async def test_failsafe_empty_on_empty_statuses(self, tmp_path: Path):
        """(c) resolver_failed (empty statuses + no error) → set() (fail-safe)."""
        harness = self._make_harness_with_stub_scheduler(tmp_path, ({}, None))

        result = await harness._warm_lane_reclaim_candidates(['A', 'B'])

        assert result == set(), (
            f'empty get_statuses must trigger fail-safe → set(), got {result!r}'
        )

    async def test_failsafe_empty_on_error(self, tmp_path: Path):
        """(c) resolver_failed (non-empty statuses + error) → set() (fail-safe)."""
        harness = self._make_harness_with_stub_scheduler(
            tmp_path, ({}, Exception('transient DB error'))
        )

        result = await harness._warm_lane_reclaim_candidates(['A', 'B'])

        assert result == set(), (
            f'errored get_statuses must trigger fail-safe → set(), got {result!r}'
        )

    async def test_empty_candidates_returns_empty_without_get_statuses(
        self, tmp_path: Path
    ):
        """(d) Empty candidates → set() without calling get_statuses."""
        config = _make_config(
            max_concurrent_tasks=4,
            warm_lane_pool=True,
            tmp_path=tmp_path,
        )
        harness = _build_harness(config)
        harness.scheduler.get_statuses = AsyncMock()
        harness.scheduler._dispatched = set()

        result = await harness._warm_lane_reclaim_candidates([])

        assert result == set(), (
            f'empty candidates must return set() immediately; got {result!r}'
        )
        harness.scheduler.get_statuses.assert_not_called()


@pytest.mark.asyncio
class TestHarnessIsBranchDispatched:
    """Tests for Harness._is_branch_dispatched (task 1933, step-5).

    Returns True iff branch is in scheduler._dispatched.
    RED today: method does not exist.
    """

    def _build(self, tmp_path: Path) -> Harness:
        config = _make_config(
            max_concurrent_tasks=2,
            warm_lane_pool=True,
            tmp_path=tmp_path,
        )
        harness = _build_harness(config)
        harness.scheduler._dispatched = {'A', 'B'}
        return harness

    async def test_dispatched_branch_returns_true(self, tmp_path: Path):
        """Branch in _dispatched → True."""
        harness = self._build(tmp_path)
        assert harness._is_branch_dispatched('A') is True, (
            'branch in _dispatched must return True'
        )

    async def test_non_dispatched_returns_false(self, tmp_path: Path):
        """Branch NOT in _dispatched → False."""
        harness = self._build(tmp_path)
        assert harness._is_branch_dispatched('Z') is False, (
            'branch not in _dispatched must return False'
        )


# ===========================================================================
# Step-7 (1933): RED — config knob + knob-gated Harness wiring
# ===========================================================================


def _make_reclaim_config(
    *,
    max_concurrent_tasks: int,
    warm_lane_pool: bool,
    warm_lane_reclaim_on_exhaustion: bool,
    tmp_path: Path,
) -> OrchestratorConfig:
    """Build a minimal OrchestratorConfig with reclaim-on-exhaustion knob."""
    repo = tmp_path / 'repo'
    repo.mkdir(exist_ok=True)
    (repo / '.git').mkdir(exist_ok=True)
    return OrchestratorConfig(
        project_root=repo,
        max_concurrent_tasks=max_concurrent_tasks,
        git=GitConfig(
            warm_lane_pool=warm_lane_pool,
            warm_lane_reclaim_on_exhaustion=warm_lane_reclaim_on_exhaustion,
        ),
    )


class TestReclaimOnExhaustionKnobWiring:
    """Config knob + knob-gated wiring tests (task 1933, step-7).

    (a) knob=True → git_ops callbacks are wired.
    (b) knob=False (default) → callbacks stay None (byte-identical).
    (c) GitConfig() default has warm_lane_reclaim_on_exhaustion == False.

    RED today: GitConfig.warm_lane_reclaim_on_exhaustion does not exist
    (ValidationError) and the Harness wiring block is absent.
    """

    def test_default_gitconfig_has_knob_false(self):
        """(c) GitConfig() default: warm_lane_reclaim_on_exhaustion == False."""
        cfg = GitConfig()
        assert cfg.warm_lane_reclaim_on_exhaustion is False, (
            'GitConfig.warm_lane_reclaim_on_exhaustion must default to False '
            '(byte-identical, trivially revertible)'
        )

    def test_callbacks_wired_when_knob_on(self, tmp_path: Path):
        """(a) knob=True → both callbacks installed on git_ops."""
        config = _make_reclaim_config(
            max_concurrent_tasks=4,
            warm_lane_pool=True,
            warm_lane_reclaim_on_exhaustion=True,
            tmp_path=tmp_path,
        )
        harness = _build_harness(config)

        assert harness.git_ops.warm_lane_reclaim_candidate_provider == (
            harness._warm_lane_reclaim_candidates
        ), (
            'warm_lane_reclaim_candidate_provider must be harness._warm_lane_reclaim_candidates '
            'when warm_lane_reclaim_on_exhaustion=True'
        )
        assert harness.git_ops.warm_lane_dispatched_predicate == (
            harness._is_branch_dispatched
        ), (
            'warm_lane_dispatched_predicate must be harness._is_branch_dispatched '
            'when warm_lane_reclaim_on_exhaustion=True'
        )

    def test_callbacks_none_when_knob_off(self, tmp_path: Path):
        """(b) knob=False (default) → both callbacks stay None."""
        config = _make_reclaim_config(
            max_concurrent_tasks=4,
            warm_lane_pool=True,
            warm_lane_reclaim_on_exhaustion=False,
            tmp_path=tmp_path,
        )
        harness = _build_harness(config)

        assert harness.git_ops.warm_lane_reclaim_candidate_provider is None, (
            'warm_lane_reclaim_candidate_provider must be None when knob is off'
        )
        assert harness.git_ops.warm_lane_dispatched_predicate is None, (
            'warm_lane_dispatched_predicate must be None when knob is off (byte-identical)'
        )


# ===========================================================================
# Step-9 (2061): RED — Harness warm-base hard-down wiring + escalation filing
# ===========================================================================


def _make_warm_base_harness(
    tmp_path: Path,
    *,
    warm_lane_pool: bool = True,
) -> tuple[Harness, EscalationQueue]:
    """Build a Harness (via _build_harness) with a real EscalationQueue wired
    on, mirroring test_harness_starvation_watchdog.py::_make_harness_with_queue
    and test_harness_no_landings_breaker.py::_make_harness.
    """
    config = _make_config(
        max_concurrent_tasks=2,
        warm_lane_pool=warm_lane_pool,
        tmp_path=tmp_path,
    )
    harness = _build_harness(config)
    queue = EscalationQueue(tmp_path / 'esc')
    harness._escalation_queue = queue
    return harness, queue


class TestHarnessWarmBaseHardDownWiring:
    """After Harness construction, the probe + all three callbacks must be
    wired onto the scheduler (task 2061, step-9).

    _build_harness patches ``Scheduler`` with a MagicMock, so an ``is not
    None`` check on an unset attribute would be vacuously true (MagicMock
    auto-vivifies child attributes on access).  Equality against the
    harness-bound method is used instead — the same idiom already used by
    TestReclaimOnExhaustionKnobWiring in this file — so this is a genuine
    RED (unset/auto-vivified Mock != bound method) before Harness.__init__
    wires the callables, and GREEN only once it does.
    """

    def test_wires_probe_and_all_callbacks(self, tmp_path: Path):
        harness, _queue = _make_warm_base_harness(tmp_path)

        assert harness.scheduler._warm_base_health_probe == (
            harness._probe_warm_base_health
        ), (
            'Harness must wire scheduler._warm_base_health_probe = '
            'harness._probe_warm_base_health after construction'
        )
        assert harness.scheduler._on_warm_base_warn == (
            harness._file_warm_base_hard_down_notice
        ), (
            'Harness must wire scheduler._on_warm_base_warn = '
            'harness._file_warm_base_hard_down_notice after construction'
        )
        assert harness.scheduler._on_warm_base_promote_l2 == (
            harness._promote_warm_base_hard_down_l2
        ), (
            'Harness must wire scheduler._on_warm_base_promote_l2 = '
            'harness._promote_warm_base_hard_down_l2 after construction'
        )
        assert harness.scheduler._on_warm_base_resolve == (
            harness._resolve_warm_base_hard_down
        ), (
            'Harness must wire scheduler._on_warm_base_resolve = '
            'harness._resolve_warm_base_hard_down after construction'
        )


@pytest.mark.asyncio
class TestProbeWarmBaseHealth:
    """_probe_warm_base_health bridges GitOps._warm_lane_base_resolvable to the
    scheduler's injected-callback string contract (task 2061, step-9).

    RED: Harness._probe_warm_base_health does not exist yet.
    """

    async def test_pool_off_always_ok(self, tmp_path: Path):
        """git_ops.warm_lane_pool is None (knob off) → probe always 'ok'."""
        harness, _queue = _make_warm_base_harness(tmp_path, warm_lane_pool=False)
        assert harness.git_ops.warm_lane_pool is None

        assert await harness._probe_warm_base_health() == 'ok'

    async def test_pool_on_delegates_ok(self, tmp_path: Path):
        from orchestrator.git_ops import WarmBaseHealth

        harness, _queue = _make_warm_base_harness(tmp_path)
        harness.git_ops._warm_lane_base_resolvable = lambda: WarmBaseHealth.OK

        assert await harness._probe_warm_base_health() == 'ok'

    async def test_pool_on_delegates_absent(self, tmp_path: Path):
        from orchestrator.git_ops import WarmBaseHealth

        harness, _queue = _make_warm_base_harness(tmp_path)
        harness.git_ops._warm_lane_base_resolvable = lambda: WarmBaseHealth.ABSENT

        assert await harness._probe_warm_base_health() == 'absent'

    async def test_pool_on_delegates_indeterminate(self, tmp_path: Path):
        from orchestrator.git_ops import WarmBaseHealth

        harness, _queue = _make_warm_base_harness(tmp_path)
        harness.git_ops._warm_lane_base_resolvable = lambda: WarmBaseHealth.INDETERMINATE

        assert await harness._probe_warm_base_health() == 'indeterminate'


@pytest.mark.asyncio
class TestWarmBaseHardDownFilingDedupResolve:
    """_file_warm_base_hard_down_notice / _promote_warm_base_hard_down_l2 /
    _resolve_warm_base_hard_down: global-sentinel filing, dedup, and resolve
    (task 2061, step-9).  Mirrors test_harness_no_landings_breaker.py's
    sentinel/role escalation asserts.

    RED: none of these three Harness methods exist yet.
    """

    async def test_file_notice_creates_one_pending_info_l0(self, tmp_path: Path):
        harness, queue = _make_warm_base_harness(tmp_path)

        await harness._file_warm_base_hard_down_notice(
            summary='WARM_BASE_HARD_DOWN: absent',
            detail='detail',
        )

        pending = [
            e
            for e in queue.get_by_task(
                Harness._WARM_BASE_HARD_DOWN_SENTINEL, status='pending'
            )
            if e.agent_role == Harness._WARM_BASE_HARD_DOWN_ROLE
        ]
        assert len(pending) == 1, f'expected exactly one pending notice; got {pending!r}'
        esc = pending[0]
        assert esc.severity == 'info', f'expected severity="info"; got {esc.severity!r}'
        assert esc.level == 0, f'expected level=0; got {esc.level}'

    async def test_file_notice_dedup_no_duplicate(self, tmp_path: Path):
        harness, queue = _make_warm_base_harness(tmp_path)

        await harness._file_warm_base_hard_down_notice(summary='first', detail='d1')
        await harness._file_warm_base_hard_down_notice(summary='second', detail='d2')

        pending = [
            e
            for e in queue.get_by_task(
                Harness._WARM_BASE_HARD_DOWN_SENTINEL, status='pending'
            )
            if e.agent_role == Harness._WARM_BASE_HARD_DOWN_ROLE and e.level == 0
        ]
        assert len(pending) == 1, (
            f'expected exactly one pending notice after two calls; got {pending!r}'
        )

    async def test_promote_l2_creates_one_pending_critical_l2(self, tmp_path: Path):
        from escalation.models import BORN_AT_L2_SEVERITIES

        harness, queue = _make_warm_base_harness(tmp_path)

        await harness._promote_warm_base_hard_down_l2(
            summary='WARM_BASE_HARD_DOWN: still absent past window',
            detail='detail',
        )

        pending = [
            e
            for e in queue.get_by_task(
                Harness._WARM_BASE_HARD_DOWN_SENTINEL, status='pending'
            )
            if e.agent_role == Harness._WARM_BASE_HARD_DOWN_ROLE and e.level == 2
        ]
        assert len(pending) == 1, f'expected exactly one pending L2; got {pending!r}'
        esc = pending[0]
        assert esc.severity == 'critical', (
            f'expected severity="critical"; got {esc.severity!r}'
        )
        assert esc.severity in BORN_AT_L2_SEVERITIES, (
            'L2 severity must be in BORN_AT_L2_SEVERITIES (born-at-L2)'
        )
        assert esc.category == 'infra_issue', (
            f'expected category="infra_issue"; got {esc.category!r}'
        )

    async def test_promote_l2_dedup_no_duplicate(self, tmp_path: Path):
        harness, queue = _make_warm_base_harness(tmp_path)

        await harness._promote_warm_base_hard_down_l2(summary='first', detail='d1')
        await harness._promote_warm_base_hard_down_l2(summary='second', detail='d2')

        pending = [
            e
            for e in queue.get_by_task(
                Harness._WARM_BASE_HARD_DOWN_SENTINEL, status='pending'
            )
            if e.agent_role == Harness._WARM_BASE_HARD_DOWN_ROLE and e.level == 2
        ]
        assert len(pending) == 1, (
            f'expected exactly one pending L2 after two calls; got {pending!r}'
        )

    async def test_resolve_clears_both_notice_and_l2(self, tmp_path: Path):
        harness, queue = _make_warm_base_harness(tmp_path)

        await harness._file_warm_base_hard_down_notice(summary='notice', detail='d')
        await harness._promote_warm_base_hard_down_l2(summary='l2', detail='d')
        pre = [
            e
            for e in queue.get_by_task(
                Harness._WARM_BASE_HARD_DOWN_SENTINEL, status='pending'
            )
            if e.agent_role == Harness._WARM_BASE_HARD_DOWN_ROLE
        ]
        assert len(pre) == 2, f'setup: expected notice + L2 both pending; got {pre!r}'

        await harness._resolve_warm_base_hard_down()

        remaining = [
            e
            for e in queue.get_by_task(
                Harness._WARM_BASE_HARD_DOWN_SENTINEL, status='pending'
            )
            if e.agent_role == Harness._WARM_BASE_HARD_DOWN_ROLE
        ]
        assert remaining == [], (
            f'expected no pending escalations after resolve; got {remaining!r}'
        )

    async def test_resolve_is_idempotent(self, tmp_path: Path):
        harness, queue = _make_warm_base_harness(tmp_path)

        await harness._file_warm_base_hard_down_notice(summary='notice', detail='d')
        await harness._resolve_warm_base_hard_down()
        # Second call must not raise or duplicate anything.
        await harness._resolve_warm_base_hard_down()

        remaining = [
            e
            for e in queue.get_by_task(
                Harness._WARM_BASE_HARD_DOWN_SENTINEL, status='pending'
            )
            if e.agent_role == Harness._WARM_BASE_HARD_DOWN_ROLE
        ]
        assert remaining == []


def test_seed_rc_76_maps_to_base_absent():
    """rc-76 discriminant wiring (per GUARD): _seed_rc_to_unavailable(76) is
    WarmLaneUnavailable.BASE_ABSENT (task 2061, step-9)."""
    from orchestrator.git_ops import WarmLaneUnavailable, _seed_rc_to_unavailable

    assert _seed_rc_to_unavailable(76) is WarmLaneUnavailable.BASE_ABSENT


# ===========================================================================
# Task 2062: warm-lane restart recovery — reconcile stale branch checkouts
# ===========================================================================
#
# ROOT CAUSE: release_warm_lane only detaches a lane's HEAD; it never removes
# the worktree, so after a restart git still has task/<id> checked out at the
# old lane while the in-memory pool assignment is empty.  Re-dispatch then
# grabs a DIFFERENT lane and `git worktree add` collides with "already used by
# worktree".  These tests build the primitives (lane_branch_checkouts,
# detach_lane_checkout) and the Harness-level reconciler
# (_reconcile_lane_checkouts) that closes the gap, plus the fix for a
# restart-independent mid-run leak in the recycle/reuse release paths.


@pytest.mark.asyncio
class TestLaneBranchCheckouts:
    """GitOps.lane_branch_checkouts() — {bare_id: canonical_lane} built from
    git's authoritative `git worktree list --porcelain` (step-1 RED / step-2 GREEN).
    """

    async def test_maps_pool_lane_branches_to_bare_ids(self, tmp_path: Path):
        """Two pool lanes checked out on task/<id> branches map to {id: lane};
        a non-pool worktree on a non-task branch is filtered out."""
        from orchestrator.git_ops import GitOps
        from orchestrator.git_ops import _run as git_run

        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_git_repo(repo)

        config = OrchestratorConfig(
            project_root=repo,
            max_concurrent_tasks=2,
            git=GitConfig(warm_lane_pool=True),
        )
        git_ops = GitOps(config.git, repo, warm_lane_pool_size=2)
        pool = git_ops.warm_lane_pool
        assert pool is not None

        lane0 = git_ops.worktree_base / '_lane-0'
        lane1 = git_ops.worktree_base / '_lane-1'
        lane0.parent.mkdir(parents=True, exist_ok=True)
        rc, _, err = await git_run(
            ['git', 'worktree', 'add', str(lane0), '-b', 'task/111', 'main'], cwd=repo,
        )
        assert rc == 0, err
        rc, _, err = await git_run(
            ['git', 'worktree', 'add', str(lane1), '-b', 'task/222', 'main'], cwd=repo,
        )
        assert rc == 0, err

        # Non-pool worktree on a non-task branch — must be filtered out.
        side = tmp_path / 'side'
        rc, _, err = await git_run(
            ['git', 'worktree', 'add', str(side), '-b', 'feature/x', 'main'], cwd=repo,
        )
        assert rc == 0, err

        result = await git_ops.lane_branch_checkouts()

        assert result == {
            '111': pool._match_lane(lane0),
            '222': pool._match_lane(lane1),
        }

    async def test_returns_none_when_pool_disabled(self, tmp_path: Path):
        """warm_lane_pool disabled (size=0) -> lane_branch_checkouts() returns None."""
        from orchestrator.git_ops import GitOps

        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_git_repo(repo)

        config = OrchestratorConfig(
            project_root=repo,
            max_concurrent_tasks=2,
            git=GitConfig(warm_lane_pool=False),
        )
        git_ops = GitOps(config.git, repo, warm_lane_pool_size=0)
        assert git_ops.warm_lane_pool is None

        assert await git_ops.lane_branch_checkouts() is None

    async def test_returns_none_on_git_error(self, tmp_path: Path):
        """`git worktree list` failing (non-zero rc) -> None (fail-safe, never mass-mutate)."""
        from orchestrator.git_ops import GitOps

        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_git_repo(repo)

        config = OrchestratorConfig(
            project_root=repo,
            max_concurrent_tasks=2,
            git=GitConfig(warm_lane_pool=True),
        )
        git_ops = GitOps(config.git, repo, warm_lane_pool_size=2)
        assert git_ops.warm_lane_pool is not None

        async def fake_run(cmd, cwd=None):
            if 'worktree' in cmd and 'list' in cmd:
                return (128, '', 'fatal: boom')
            return (0, '', '')

        with patch('orchestrator.git_ops._run', side_effect=fake_run):
            result = await git_ops.lane_branch_checkouts()

        assert result is None


@pytest.mark.asyncio
class TestDetachLaneCheckout:
    """GitOps.detach_lane_checkout() — commit WIP then detach a lane's HEAD,
    WITHOUT removing the worktree or deleting the branch (step-3 RED / step-4 GREEN).
    """

    async def test_commits_wip_and_detaches_retaining_worktree_and_branch(
        self, tmp_path: Path,
    ):
        """WIP on the checked-out branch is committed, HEAD ends up detached,
        and both the branch and the worktree registration survive."""
        from orchestrator.git_ops import GitOps
        from orchestrator.git_ops import _run as git_run

        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_git_repo(repo)

        config = OrchestratorConfig(
            project_root=repo,
            max_concurrent_tasks=1,
            git=GitConfig(warm_lane_pool=True),
        )
        git_ops = GitOps(config.git, repo, warm_lane_pool_size=1)
        assert git_ops.warm_lane_pool is not None

        lane0 = git_ops.worktree_base / '_lane-0'
        lane0.parent.mkdir(parents=True, exist_ok=True)
        rc, _, err = await git_run(
            ['git', 'worktree', 'add', str(lane0), '-b', 'task/777', 'main'], cwd=repo,
        )
        assert rc == 0, err

        # Uncommitted WIP on a tracked file.
        (lane0 / 'README.md').write_text('# Test\nWIP change\n')

        result = await git_ops.detach_lane_checkout(lane0, '777')
        assert result is True

        # HEAD at lane0 is now DETACHED (no symbolic ref).
        rc, _, _ = await git_run(['git', 'symbolic-ref', '-q', 'HEAD'], cwd=lane0)
        assert rc != 0, 'HEAD must be detached after detach_lane_checkout'

        # The branch still exists...
        rc, _, _ = await git_run(['git', 'rev-parse', '--verify', 'task/777'], cwd=repo)
        assert rc == 0, 'task/777 branch must survive detach_lane_checkout'

        # ...and now carries the committed WIP beyond main.
        rc, out, _ = await git_run(
            ['git', 'rev-list', '--count', 'main..task/777'], cwd=repo,
        )
        assert rc == 0 and int(out.strip()) > 0, (
            'WIP must be committed onto task/777 before detaching'
        )

        # The lane is still a registered worktree (not removed).
        rc, out, _ = await git_run(['git', 'worktree', 'list', '--porcelain'], cwd=repo)
        assert rc == 0
        assert str(lane0) in out, 'lane worktree registration must survive detach'

    async def test_detach_failure_logs_error_and_returns_false(
        self, tmp_path: Path, caplog,
    ):
        """`git checkout --detach` failing -> returns False and logs at ERROR."""
        from orchestrator.git_ops import GitOps
        from orchestrator.git_ops import _run as orig_run

        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_git_repo(repo)

        config = OrchestratorConfig(
            project_root=repo,
            max_concurrent_tasks=1,
            git=GitConfig(warm_lane_pool=True),
        )
        git_ops = GitOps(config.git, repo, warm_lane_pool_size=1)
        assert git_ops.warm_lane_pool is not None

        lane0 = git_ops.worktree_base / '_lane-0'
        lane0.parent.mkdir(parents=True, exist_ok=True)
        rc, _, err = await orig_run(
            ['git', 'worktree', 'add', str(lane0), '-b', 'task/777', 'main'], cwd=repo,
        )
        assert rc == 0, err

        async def fake_run(cmd, cwd=None):
            if 'checkout' in cmd and '--detach' in cmd:
                return (128, '', 'fatal: boom')
            return await orig_run(cmd, cwd=cwd)

        with (
            caplog.at_level(logging.ERROR, logger='orchestrator.git_ops'),
            patch('orchestrator.git_ops._run', side_effect=fake_run),
        ):
            result = await git_ops.detach_lane_checkout(lane0, '777')

        assert result is False
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors, 'detach_lane_checkout must log at ERROR on checkout --detach failure'


@pytest.mark.asyncio
class TestReconcileLaneCheckouts:
    """Harness._reconcile_lane_checkouts() — reconcile git's authoritative
    worktree admin against the in-memory pool assignment map at startup,
    closing the gap left by _recover_crashed_tasks' skip branches
    (steps 5/7/9 RED, steps 6/8/10 GREEN).
    """

    async def _make_fixture(self, tmp_path: Path, *, pool_size: int = 6):
        """Harness + real GitOps pool with task/3965 checked out at _lane-5
        (a commit beyond main), while pool._assignments stays EMPTY —
        simulating a restart where _recover_crashed_tasks skipped this id.

        pool_size=6 so _lane-0..4 are FREE and get handed out by acquire_for
        before _lane-5 (the one actually holding task/3965) is ever reached —
        this is exactly the collision setup the fix must reconcile away.
        """
        from orchestrator.git_ops import GitOps
        from orchestrator.git_ops import _run as git_run

        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_git_repo(repo)

        # Pre-create the default warm-lane CoW seed base so the pre-acquire
        # base-health gate sees WarmBaseHealth.OK (mirrors wl_git_repo in
        # test_warm_lane_pool.py), and a no-op seed script so a successful
        # acquire's seed step succeeds.
        default_base = repo / '.worktrees' / '_merge-verify' / 'target'
        default_base.mkdir(parents=True, exist_ok=True)
        (default_base / '.keep').write_text('warm base sentinel\n')
        # Task 2099: mark pool storage present so the create-once discriminator
        # does not refuse the first lane creation below (mirrors wl_git_repo).
        (repo / '.worktrees' / '.pool-root').touch()
        scripts_dir = repo / 'scripts'
        scripts_dir.mkdir(parents=True, exist_ok=True)
        seed_script = scripts_dir / 'seed-warm-lane.sh'
        seed_script.write_text('#!/usr/bin/env bash\nexit 0\n')
        seed_script.chmod(0o755)
        await git_run(['git', 'add', '-A'], cwd=repo)
        await git_run(['git', 'commit', '-m', 'add no-op seed-warm-lane.sh'], cwd=repo)

        config = OrchestratorConfig(
            project_root=repo,
            max_concurrent_tasks=pool_size,
            git=GitConfig(warm_lane_pool=True),
            worktree_identity_guard_enabled=False,
        )
        git_ops = GitOps(config.git, repo, warm_lane_pool_size=pool_size)
        pool = git_ops.warm_lane_pool
        assert pool is not None

        harness = _build_harness(config)
        harness.git_ops = git_ops

        lane5 = git_ops.worktree_base / '_lane-5'
        lane5.parent.mkdir(parents=True, exist_ok=True)
        rc, _, err = await git_run(
            ['git', 'worktree', 'add', str(lane5), '-b', 'task/3965', 'main'], cwd=repo,
        )
        assert rc == 0, err
        (lane5 / 'wip.txt').write_text('unmerged work\n')
        await git_run(['git', 'add', '-A'], cwd=lane5)
        await git_run(['git', 'commit', '-m', 'wip: task 3965 work'], cwd=lane5)

        return harness, git_ops, pool, lane5

    async def test_control_no_reconcile_collides_on_reacquire(
        self, tmp_path: Path, caplog,
    ):
        """CONTROL (documents the bug): with NO reconcile, a fresh acquire grabs
        a different (first-free) lane and collides with the branch already
        checked out at _lane-5."""
        from orchestrator.git_ops import WarmLaneUnavailable

        _harness, git_ops, _pool, _lane5 = await self._make_fixture(tmp_path)

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            result = await git_ops.acquire_warm_lane('3965', 'main')

        assert result is WarmLaneUnavailable.FAULT, (
            f'expected FAULT from the worktree-add collision, got {result!r}'
        )
        assert 'already used by worktree' in caplog.text, (
            f'expected the git collision message in the log; got: {caplog.text!r}'
        )

    async def test_repin_on_live_status_reuses_original_lane(self, tmp_path: Path):
        """RE-PIN: a live (non-terminal) status re-pins the id to its ORIGINAL
        lane, so the next acquire reuses it (no fault) and the WIP survives."""
        from orchestrator.git_ops import WorktreeInfo
        from orchestrator.git_ops import _run as git_run
        from orchestrator.warm_lane_pool import LaneState

        harness, git_ops, pool, lane5 = await self._make_fixture(tmp_path)
        canon = pool._match_lane(lane5)
        assert canon is not None

        harness.scheduler.get_statuses = AsyncMock(
            return_value=({'3965': 'in-progress'}, None),
        )

        await harness._reconcile_lane_checkouts()

        assert pool.assignment_for('3965') == canon, (
            'reconcile must re-pin 3965 to its ORIGINAL lane (_lane-5)'
        )
        assert pool.state(canon) == LaneState.ASSIGNED

        result = await git_ops.acquire_warm_lane('3965', 'main')
        assert isinstance(result, WorktreeInfo), (
            f'expected a reuse WorktreeInfo (lane re-pinned), got {result!r}'
        )
        assert result.path == canon

        rc, out, _ = await git_run(
            ['git', 'rev-list', '--count', 'main..task/3965'], cwd=git_ops.project_root,
        )
        assert rc == 0 and int(out.strip()) > 0, (
            'the pre-existing WIP commit on task/3965 must survive the reuse'
        )

    async def test_degraded_status_read_repins_fail_safe(self, tmp_path: Path):
        """DEGRADED get_statuses (error) -> fail-safe re-pin (never detach on a
        bad read), mirroring test_recovery_restores_on_status_read_failure."""
        from orchestrator.warm_lane_pool import LaneState

        harness, _git_ops, pool, lane5 = await self._make_fixture(tmp_path)
        canon = pool._match_lane(lane5)
        assert canon is not None

        harness.scheduler.get_statuses = AsyncMock(
            return_value=({}, RuntimeError('boom')),
        )

        await harness._reconcile_lane_checkouts()

        assert pool.assignment_for('3965') == canon, (
            'a degraded status read must fail-safe RE-PIN, never detach'
        )
        assert pool.state(canon) == LaneState.ASSIGNED

    async def test_terminal_status_detaches_and_reattach_elsewhere_succeeds(
        self, tmp_path: Path,
    ):
        """TERMINAL status (e.g. 'done') -> DETACH (never re-pin); a fresh
        acquire lands on a different free lane and reattaches the branch —
        no 'already used by worktree' collision, WIP commit preserved."""
        from orchestrator.git_ops import WorktreeInfo
        from orchestrator.git_ops import _run as git_run

        harness, git_ops, pool, lane5 = await self._make_fixture(tmp_path)

        harness.scheduler.get_statuses = AsyncMock(
            return_value=({'3965': 'done'}, None),
        )

        await harness._reconcile_lane_checkouts()

        assert pool.assignment_for('3965') is None, (
            'a terminal status must NOT be re-pinned'
        )
        rc, _, _ = await git_run(['git', 'symbolic-ref', '-q', 'HEAD'], cwd=lane5)
        assert rc != 0, 'terminal checkout must be detached'

        rc, _, _ = await git_run(
            ['git', 'rev-parse', '--verify', 'task/3965'], cwd=git_ops.project_root,
        )
        assert rc == 0, 'task/3965 branch must survive the detach'

        result = await git_ops.acquire_warm_lane('3965', 'main')
        assert isinstance(result, WorktreeInfo), (
            f'expected a reattach WorktreeInfo onto a fresh lane, got {result!r}'
        )

        rc, out, _ = await git_run(
            ['git', 'rev-list', '--count', 'main..task/3965'], cwd=git_ops.project_root,
        )
        assert rc == 0 and int(out.strip()) > 0, (
            'the pre-existing WIP commit on task/3965 must survive the reattach'
        )

    async def test_deleted_id_absent_from_healthy_read_detaches(
        self, tmp_path: Path,
    ):
        """A HEALTHY status read that OMITS the id entirely (task deleted) ->
        DETACH, same as a terminal status; a fresh acquire does not collide."""
        from orchestrator.git_ops import WorktreeInfo
        from orchestrator.git_ops import _run as git_run

        harness, git_ops, pool, lane5 = await self._make_fixture(tmp_path)

        harness.scheduler.get_statuses = AsyncMock(
            return_value=({'999': 'pending'}, None),
        )

        await harness._reconcile_lane_checkouts()

        assert pool.assignment_for('3965') is None, (
            'an id absent from a healthy read must NOT be re-pinned'
        )
        rc, _, _ = await git_run(['git', 'symbolic-ref', '-q', 'HEAD'], cwd=lane5)
        assert rc != 0, 'deleted-id checkout must be detached'

        result = await git_ops.acquire_warm_lane('3965', 'main')
        assert isinstance(result, WorktreeInfo), (
            f'expected a reattach WorktreeInfo onto a fresh lane, got {result!r}'
        )

    async def test_prepinned_same_lane_is_idempotent_noop(self, tmp_path: Path):
        """PRE-PINNED: recovery already re-pinned this id to this EXACT lane
        (simulated via pool.restore_assignment before reconcile runs) ->
        reconcile is a no-op: detach_lane_checkout is never called and the
        lane stays ASSIGNED/attached."""
        from orchestrator.git_ops import _run as git_run
        from orchestrator.warm_lane_pool import LaneState

        harness, git_ops, pool, lane5 = await self._make_fixture(tmp_path)
        canon = pool._match_lane(lane5)
        assert canon is not None
        pool.restore_assignment('3965', canon)

        harness.scheduler.get_statuses = AsyncMock(
            return_value=({'3965': 'in-progress'}, None),
        )
        git_ops.detach_lane_checkout = AsyncMock()

        await harness._reconcile_lane_checkouts()

        git_ops.detach_lane_checkout.assert_not_called()
        assert pool.assignment_for('3965') == canon
        assert pool.state(canon) == LaneState.ASSIGNED

        rc, _, _ = await git_run(['git', 'symbolic-ref', '-q', 'HEAD'], cwd=lane5)
        assert rc == 0, 'HEAD must remain attached for an already-pinned lane (no-op)'

    async def test_dup_checkout_detaches_unpinned_keeping_original_pin(
        self, tmp_path: Path,
    ):
        """DUP-CHECKOUT: recovery pinned this id to a DIFFERENT lane
        (_lane-2); the physically-checked-out duplicate (_lane-5) is
        detached while the original pin is kept untouched."""
        from orchestrator.git_ops import _run as git_run

        harness, git_ops, pool, lane5 = await self._make_fixture(tmp_path)
        lane2 = git_ops.worktree_base / '_lane-2'
        pool.restore_assignment('3965', lane2)

        harness.scheduler.get_statuses = AsyncMock(
            return_value=({'3965': 'in-progress'}, None),
        )

        await harness._reconcile_lane_checkouts()

        assert pool.assignment_for('3965') == lane2, (
            'the original pin (_lane-2) must be kept untouched'
        )
        rc, _, _ = await git_run(['git', 'symbolic-ref', '-q', 'HEAD'], cwd=lane5)
        assert rc != 0, (
            'the un-pinned duplicate checkout (_lane-5) must be detached'
        )

    async def test_never_steal_skips_when_lane_pinned_to_different_id(
        self, tmp_path: Path, caplog,
    ):
        """NEVER-STEAL: _lane-5 is pinned to a DIFFERENT id (4031); reconcile
        must skip the dup checkout entirely rather than stealing the lane
        recovery already assigned to that other id."""
        from orchestrator.git_ops import _run as git_run

        harness, git_ops, pool, lane5 = await self._make_fixture(tmp_path)
        canon = pool._match_lane(lane5)
        assert canon is not None
        pool.restore_assignment('4031', canon)

        harness.scheduler.get_statuses = AsyncMock(
            return_value=({'3965': 'in-progress'}, None),
        )
        git_ops.detach_lane_checkout = AsyncMock()

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            await harness._reconcile_lane_checkouts()

        git_ops.detach_lane_checkout.assert_not_called()
        assert pool.assignment_for('4031') == canon, (
            'the existing pin to a DIFFERENT id must never be stolen'
        )
        assert pool.assignment_for('3965') is None, (
            'the never-steal case must skip entirely — 3965 must NOT be '
            'pinned onto the lane already owned by 4031'
        )
        rc, _, _ = await git_run(['git', 'symbolic-ref', '-q', 'HEAD'], cwd=lane5)
        assert rc == 0, 'never-steal must not detach the lane'
        assert 'never-steal' in caplog.text, (
            f'expected a never-steal warning in the log; got: {caplog.text!r}'
        )

    async def test_git_error_read_makes_zero_pool_mutations(self, tmp_path: Path):
        """GIT-ERROR: lane_branch_checkouts() returns None -> reconcile makes
        ZERO pool mutations (no restore_assignment, no detach_lane_checkout)."""
        harness, git_ops, pool, _lane5 = await self._make_fixture(tmp_path)
        git_ops.lane_branch_checkouts = AsyncMock(return_value=None)
        git_ops.detach_lane_checkout = AsyncMock()
        pool.restore_assignment = MagicMock(wraps=pool.restore_assignment)

        harness.scheduler.get_statuses = AsyncMock(
            return_value=({'3965': 'in-progress'}, None),
        )

        await harness._reconcile_lane_checkouts()

        git_ops.detach_lane_checkout.assert_not_called()
        pool.restore_assignment.assert_not_called()


@pytest.mark.asyncio
class TestMidRunRecycleLeak:
    """Restart-INDEPENDENT leak: a recycled-lane re-seed failure must not
    leave the new task's branch checked out at the (now-FREE) lane
    (step-11 RED / step-12 GREEN — route the bare ``pool.release()`` calls
    in the recycle/reuse fault paths through ``release_warm_lane`` instead,
    which detaches before releasing).
    """

    async def test_reset_in_place_seed_failure_does_not_leak_checkout(
        self, tmp_path: Path,
    ):
        """A recycled FREE lane whose re-seed fails must not leave task/555
        checked out at the lane afterward — else the next create-once
        acquire elsewhere collides with "already used by worktree"."""
        from orchestrator.git_ops import GitOps, WarmLaneUnavailable
        from orchestrator.git_ops import _run as git_run

        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_git_repo(repo)

        # Pre-create the default warm-lane CoW seed base so the pre-acquire
        # base-health gate sees WarmBaseHealth.OK — it runs BEFORE the (here
        # mocked-out) real seed step either way.
        default_base = repo / '.worktrees' / '_merge-verify' / 'target'
        default_base.mkdir(parents=True, exist_ok=True)
        (default_base / '.keep').write_text('warm base sentinel\n')
        # Task 2099: mark pool storage present (mirrors wl_git_repo) — belt
        # and braces even though this test's reset-in-place path does not
        # exercise the create-once discriminator.
        (repo / '.worktrees' / '.pool-root').touch()

        config = OrchestratorConfig(
            project_root=repo,
            max_concurrent_tasks=1,
            git=GitConfig(warm_lane_pool=True),
        )
        git_ops = GitOps(config.git, repo, warm_lane_pool_size=1)
        pool = git_ops.warm_lane_pool
        assert pool is not None

        # _lane-0 is a recycled FREE lane already holding task/555 at the
        # main tip (NO commits beyond main -> _orphan_has_commits is False
        # -> the fresh reset-in-place recycle path is taken, not the γ
        # reattach guard).  Pool state defaults to FREE with no assignment —
        # exactly the "recycled" shape (a leftover checkout the pool doesn't
        # know about yet).
        lane0 = git_ops.worktree_base / '_lane-0'
        lane0.parent.mkdir(parents=True, exist_ok=True)
        rc, _, err = await git_run(
            ['git', 'worktree', 'add', str(lane0), '-b', 'task/555', 'main'], cwd=repo,
        )
        assert rc == 0, err

        git_ops._seed_warm_lane = AsyncMock(return_value=1)

        result = await git_ops.acquire_warm_lane('555', 'main')
        assert result is WarmLaneUnavailable.FAULT, (
            f'expected FAULT from the forced seed failure, got {result!r}'
        )

        checkouts = await git_ops.lane_branch_checkouts()
        assert checkouts is not None
        assert '555' not in checkouts, (
            'task/555 must not still be checked out at _lane-0 after the '
            'failing re-seed — a bare pool.release() without detaching '
            'leaks the checkout, and a subsequent create-once acquire '
            'elsewhere would collide with "already used by worktree"'
        )
        assert pool.assignment_for('555') is None, (
            'the recycled lane must be released/unassigned after the '
            'failing re-seed, not left dangling on a stale pin'
        )
