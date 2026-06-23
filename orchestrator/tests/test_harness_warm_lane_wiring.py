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
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
# Step-7 (1881): RED — B2: _run_slot finally releases lane on hard-cancel
# ===========================================================================


@pytest.mark.asyncio
class TestHardCancelLaneRelease:
    """B2 (T5): harness._run_slot finally must release the warm lane on hard-cancel.

    Hard-cancel = asyncio.CancelledError raised inside the try block of _run_slot
    (harness.py:3402) when hard_cancel_workflow() calls task.cancel(). The slot
    returns a synthetic TaskReport(outcome=CANCELLED) from the except clause and
    the run() finally of the WORKFLOW never executes (B1 cannot cover this path).

    Diff 4: in _run_slot finally, after the events pops, add:
        if report is not None and report.outcome in (DONE, CANCELLED):
            with contextlib.suppress(Exception):
                await self.git_ops.release_lane_for_terminal_task(assignment.task_id)

    RED: finally has no release; lane stays ASSIGNED after _run_slot returns.
    """

    async def test_hard_cancel_frees_lane(self, tmp_path: Path):
        """Hard-cancelled slot (asyncio.CancelledError path) → lane freed in slot finally."""
        from orchestrator.git_ops import GitOps  # noqa: F401
        from orchestrator.scheduler import TaskAssignment
        from orchestrator.warm_lane_pool import LaneState
        from orchestrator.workflow import WorkflowOutcome

        # ── Setup: minimal git repo ─────────────────────────────────────────
        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_git_repo(repo)

        config = OrchestratorConfig(
            project_root=repo,
            max_concurrent_tasks=1,
            git=GitConfig(warm_lane_pool=True),
        )

        # ── Harness with real git_ops (warm lane pool) + mocked scheduler ──
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
        harness.scheduler.carries_substrate_probe.return_value = False

        # ── Patch TaskWorkflow to simulate a hard-cancel ────────────────────
        # workflow.run() raises CancelledError, exactly as if the slot's
        # asyncio.Task was cancelled by hard_cancel_workflow().
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

        # (B2 RED) Lane must be freed by the slot finally.
        # Fails: _run_slot finally currently has no release_lane_for_terminal_task call.
        assert pool.assignment_for('99') is None, (
            '_run_slot finally must release the lane for a hard-cancelled slot'
        )
        assert pool.state(lane) == LaneState.FREE, (
            f'Lane must be FREE after hard-cancel, got {pool.state(lane)!r}'
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
        assert pool.state(lane) == LaneState.ASSIGNED

        scheduler = FakeScheduler()
        scheduler.statuses[mid] = ['merge-deferred']  # member exists

        factory = build_train_callback_factory(scheduler, git_ops)
        callbacks = factory('train-3')

        # found_on_main=True triggers the mark_done path (T6 canonical scenario)
        await callbacks.redrive_member(mid, found_on_main=True, sha='ghi789')

        # scheduler.mark_done must have been called
        assert scheduler.statuses.get(mid, [])[-1] == 'done', (
            'redrive_member(found_on_main=True) must flip to done'
        )
        # Lane must be freed (RED: redrive_member closure never releases)
        assert pool.assignment_for(mid) is None, (
            'redrive_member must release the warm lane after mark_done(found_on_main)'
        )
        assert pool.state(lane) == LaneState.FREE
