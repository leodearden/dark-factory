"""Tests for GitOps.task_verify_lease — the warm-lane consumer-hold (task 3027).

The warm-lane-gc corruption race (reify esc-5236-7 / esc-5275-10, 3+
reproductions): reify's ``warm-lane-gc.sh reclaim`` reseeds a warm lane whose
LIVE consumer is mid-nextest, deleting in-flight test binaries → exit-127
vanished-artifact failures. Nothing held ``<lane_dir>.lock`` for the DURATION
of a bare task-lane nextest run. ``task_verify_lease`` is the DF half: hold the
lane's flock across the task-lane verify so a concurrent reclaim's per-lane
``flock -n`` refuses/queues (reify task 5354 is the paired reify-side
acquire-time guard).

Deliberately DIFFERENT from :meth:`GitOps.merge_verify_lease` in two ways:
  - flock-ONLY: it does NOT write the global merge-verify holder-pgid
    rendezvous (that single global key would be stomped by many concurrent
    task-lane verifies and would spuriously gate reset_persistent_merge_worktree
    / _run_warm_lane_gc_reclaim).
  - fail-OPEN on flock contention (WARNING + proceed), never raising
    MergeVerifyLeaseContended — a task-lane verify must never be blocked or
    aborted by its own lane lease.

This module builds the CM up from the bottom:
  step-1/2 — the happy-path flock hold + release (flock-only)
  step-3/4 — fail-open on a contended flock
  step-5/6 — the two workflow.py task-lane verify sites hold the lease
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify import VerifyResult
from orchestrator.verify_cancel import (
    acquire_merge_verify_flock,
    lane_lock_path,
    read_lock_holder_pgid,
    release_merge_verify_flock,
)

# Reused read-only (cross-module helper, mirroring the suite's convention, e.g.
# test_harness_task_runtime importing from test_task_runtime): the workflow spy
# factory that builds a TaskWorkflow with all collaborators mocked.
from test_workflow_verify_retry import _make_workflow


def _git_config(**overrides) -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        **overrides,
    )


def _git_ops(tmp_path: Path, **config_overrides) -> GitOps:
    """A pure-FS GitOps: these lease tests never shell out to git, so
    project_root need not be a real git repo (mirrors
    test_merge_verify_lease_guard.py::_git_ops)."""
    project_root = tmp_path / 'project'
    project_root.mkdir(exist_ok=True)
    return GitOps(_git_config(**config_overrides), project_root)


@pytest.mark.asyncio
class TestTaskVerifyLeaseHoldsLaneLock:
    """task_verify_lease holds <lane_dir>.lock for the block, flock-only
    (step-1/2)."""

    async def test_lane_flock_held_inside_released_after_and_flockonly(
        self, tmp_path: Path,
    ):
        """(a) FLOCK-PATH PIN: the lease holds lane_lock_path(lane_dir) for the
        duration of the block — a fresh non-blocking (0.0s) acquire on that same
        path from THIS process observes contention (separate open file
        description → None). (b) After the block exits the lock is FREE — the
        same acquire now returns a real fd. (c) FLOCK-ONLY: the merge-verify
        GLOBAL holder-pgid rendezvous (read_lock_holder_pgid(worktree_base)) is
        never written — None both inside and after the block (pins the
        single-purpose merge-verify holder-pgid design decision).
        """
        git_ops = _git_ops(tmp_path)
        lane_dir = git_ops.worktree_base / '3027'
        lock_path = lane_lock_path(lane_dir)

        # Sanity: nothing held before the block.
        assert read_lock_holder_pgid(git_ops.worktree_base) is None

        async with git_ops.task_verify_lease(lane_dir):
            # (a) The per-lane flock is HELD — a fresh bounded-wait-0.0 acquire
            # from this same process contends (returns None).
            contended = acquire_merge_verify_flock(lock_path, 0.0)
            assert contended is None, (
                'the lane flock must be HELD for the duration of the block'
            )
            # (c) flock-only: NO merge-verify holder-pgid may be written.
            assert read_lock_holder_pgid(git_ops.worktree_base) is None, (
                'task_verify_lease must NOT write the merge-verify holder-pgid'
            )

        # (b) After exit: the lock is released — a fresh acquire now succeeds.
        freed = acquire_merge_verify_flock(lock_path, 0.0)
        assert freed is not None and freed >= 0, (
            'the lane flock must be released when the block exits'
        )
        release_merge_verify_flock(freed)

        # (c) still no holder-pgid after exit.
        assert read_lock_holder_pgid(git_ops.worktree_base) is None


@pytest.mark.asyncio
class TestTaskVerifyLeaseFailsOpenOnContention:
    """A contended flock (bounded wait times out → acquire returns None) must
    FAIL OPEN: log a WARNING and still run the verify body, never raise
    (step-3/4).

    Contrast merge_verify_lease's fail-CLOSED MergeVerifyLeaseContended raise:
    a task-lane verify must never be blocked or aborted by its own lane lease,
    and proceeding-without-the-hold is exactly today's baseline (nothing held
    it), so fail-open is strictly non-regressive.
    """

    async def test_contended_flock_warns_and_enters_body_without_raising(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        git_ops = _git_ops(tmp_path)
        lane_dir = git_ops.worktree_base / '3027'
        lock_path = lane_lock_path(lane_dir)

        # Simulate acquire_merge_verify_flock's bounded-wait TIMEOUT (→ None):
        # the racing reseed held the lane lock past _TASK_VERIFY_LEASE_WAIT_SECS.
        monkeypatch.setattr(
            'orchestrator.git_ops.acquire_merge_verify_flock',
            lambda *a, **k: None,
        )

        entered = False
        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            async with git_ops.task_verify_lease(lane_dir):
                entered = True  # body MUST run — fail-open, not fail-closed

        assert entered, (
            'a contended lane flock must fail OPEN — the verify body must still '
            'run (never blocked/aborted by its own lane lease)'
        )
        # A WARNING naming the contended lane lock must have been logged; and no
        # fd-related TypeError may have propagated (release must not be called
        # on a None fd).
        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and str(lock_path) in r.getMessage()
        ]
        assert warnings, (
            f'expected a WARNING naming the contended lane lock {lock_path}; '
            f'got: {[r.getMessage() for r in caplog.records]}'
        )


def _passing_result() -> VerifyResult:
    return VerifyResult(
        passed=True,
        test_output='',
        lint_output='',
        type_output='',
        summary='ok',
        timed_out=False,
    )


def _build_task_workflow(tmp_path: Path):
    """A TaskWorkflow on a REAL pure-FS GitOps + a real (non-git) worktree dir.

    project_root need not be a git repo: this path exercises only
    task_verify_lease's flock (pure-FS) and the PATCHED run_scoped_verification,
    never a real git operation. Reuses test_workflow_verify_retry._make_workflow
    so the TaskWorkflow's collaborators are wired identically to the primary
    verify-retry tests.
    """
    project_root = tmp_path / 'project'
    project_root.mkdir(exist_ok=True)
    config = OrchestratorConfig(project_root=project_root, git=_git_config())
    git_ops = GitOps(config.git, config.project_root)
    worktree = git_ops.worktree_base / '42'
    worktree.mkdir(parents=True, exist_ok=True)
    assignment = TaskAssignment(
        task_id='42',
        task={
            'id': '42', 'title': 'X', 'description': '',
            'status': 'pending', 'metadata': {'files': ['lib']},
            'dependencies': [],
        },
        modules=['lib'],
    )
    workflow, _artifacts = _make_workflow(config, git_ops, assignment, worktree)
    return workflow


@pytest.mark.asyncio
class TestTaskLaneVerifyHoldsLease:
    """The implement-phase task-lane verify runs INSIDE task_verify_lease
    (step-5/6).

    A call-site spy: the patched run_scoped_verification, when invoked, tries a
    fresh non-blocking acquire of the task lane's flock. If the verify call is
    wrapped in the lease (step-6) that acquire CONTENDS (None); if not (RED) it
    succeeds, so the assertion fails.
    """

    async def test_infra_retry_verify_holds_task_lane_flock(self, tmp_path: Path):
        workflow = _build_task_workflow(tmp_path)
        lock_path = lane_lock_path(workflow.worktree)

        def _assert_flock_held(*_args, **_kwargs) -> VerifyResult:
            # The task lane flock must be HELD for the duration of the verify
            # call — a fresh bounded-wait-0.0 acquire from this same process
            # must contend (separate open file description → None).
            fd = acquire_merge_verify_flock(lock_path, 0.0)
            try:
                assert fd is None, (
                    'the task lane flock must be HELD (task_verify_lease) while '
                    'run_scoped_verification runs — a fresh 0.0s acquire must '
                    'contend, but it succeeded (the verify call is not wrapped '
                    'in the lease)'
                )
            finally:
                if fd is not None:
                    release_merge_verify_flock(fd)
            return _passing_result()

        verify_mock = AsyncMock(side_effect=_assert_flock_held)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr('orchestrator.workflow.run_scoped_verification', verify_mock)
            result = await workflow._run_scoped_verification_with_infra_retry(
                verify_attempt=0,
            )

        assert result is not None and result.passed, (
            'the wrapped verify must return the passing VerifyResult (proving '
            'the in-mock held-flock assertion passed)'
        )
        assert verify_mock.await_count == 1
