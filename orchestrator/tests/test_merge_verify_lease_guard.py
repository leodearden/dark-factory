"""Tests for the GitOps merge-verify lease primitives (task 2315, BUG 1).

Incident: the only dynamic mutators of a LIVE ``_merge-verify`` worktree
(``reset_persistent_merge_worktree`` and ``_run_warm_lane_gc_reclaim``)
never consulted the EXISTING verify_cancel merge-verify flock +
holder-pgid lease (task 2306) that the host verify-merge CLI already
records — so a verify in flight could still have its worktree clobbered
out from under it.

Fix: REUSE that lease (no new lock). This module builds up the fix from
the bottom:
  step-11/12 — the lease predicate (``_merge_verify_lease_active``) and
               the ``merge_verify_lease()`` async context manager
  step-13/14 — ``reset_persistent_merge_worktree`` refuses a foreign live
               lease (``MergeVerifyLeaseHeld``)
  step-15/16 — ``_run_warm_lane_gc_reclaim`` defers while ANY lease is
               held (including self)
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps
from orchestrator.verify_cancel import read_lock_holder_pgid, write_lock_holder_pgid


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
    """A pure-FS GitOps: these lease predicate/writer tests never shell out
    to git, so project_root need not be a real git repo."""
    project_root = tmp_path / 'project'
    project_root.mkdir(exist_ok=True)
    return GitOps(_git_config(**config_overrides), project_root)


#: A pgid guaranteed to be dead: os.killpg on this must raise ProcessLookupError
#: (Linux pid_max is nowhere near 2**31-1).
_DEAD_PGID = 2**31 - 1


class TestMergeVerifyLeaseActivePredicate:
    """GitOps._merge_verify_lease_active() (step-11/12)."""

    def test_no_holder_file_is_not_active(self, tmp_path: Path):
        git_ops = _git_ops(tmp_path)
        assert git_ops._merge_verify_lease_active() is False

    def test_live_holder_pgid_is_active(self, tmp_path: Path):
        git_ops = _git_ops(tmp_path)
        write_lock_holder_pgid(git_ops.worktree_base, os.getpgrp())
        assert git_ops._merge_verify_lease_active() is True

    def test_dead_holder_pgid_is_not_active(self, tmp_path: Path):
        """A stale lease (holder died without cleanup) must never wedge
        callers — detection is fail-OPEN."""
        git_ops = _git_ops(tmp_path)
        write_lock_holder_pgid(git_ops.worktree_base, _DEAD_PGID)
        assert git_ops._merge_verify_lease_active() is False


@pytest.mark.asyncio
class TestMergeVerifyLeaseContextManager:
    """GitOps.merge_verify_lease() async ctx-mgr (step-11/12)."""

    async def test_lease_active_inside_block_and_cleared_after_exit(
        self, tmp_path: Path,
    ):
        git_ops = _git_ops(tmp_path)
        assert git_ops._merge_verify_lease_active() is False

        async with git_ops.merge_verify_lease():
            assert git_ops._merge_verify_lease_active() is True, (
                'lease must be active for the duration of the block'
            )
            assert read_lock_holder_pgid(git_ops.worktree_base) == os.getpgrp()

        assert git_ops._merge_verify_lease_active() is False, (
            'holder-pgid must be cleared and the flock released on exit'
        )
        assert read_lock_holder_pgid(git_ops.worktree_base) is None
