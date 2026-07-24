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

from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps
from orchestrator.verify_cancel import (
    acquire_merge_verify_flock,
    lane_lock_path,
    read_lock_holder_pgid,
    release_merge_verify_flock,
)


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
