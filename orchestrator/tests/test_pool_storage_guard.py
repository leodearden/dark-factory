"""Tests for the ``.pool-root`` sentinel guarding destructive worktree_base sweeps.

Incident (task 2099): after a Jul-3 host crash, reify's orchestrator started
BEFORE the warm-lanes XFS mount came up.  ``worktree_base`` (the mountpoint
dir) existed but was EMPTY, so the existing ``worktree_base.exists()`` guards
were satisfied and the orphan-reaper's unconditional ``prune_worktrees()``
tail ran ``git worktree prune`` against a repo whose mount-resident worktree
dirs all APPEARED missing — wiping every registered ``_lane-*`` +
``_merge-verify`` admin entry.

``GitOps.pool_storage_present()`` answers "is the pool storage actually
mounted?" via a ``.pool-root`` sentinel FILE that lives ON the pool storage
itself — substrate-independent (works for plain dirs and real mounts alike,
no config knob).  ``GitOps.mark_pool_storage_present()`` writes it.  It is
written from exactly ONE chokepoint, ``_seed_warm_lane`` on ``rc == 0``
(step-8), because a successful seed proves the mount is present and writable.

This module builds up the guard from the bottom:
  step-1/2  — the predicate + writer themselves
  step-3/4  — prune_worktrees() guard
  step-5/6  — _run_warm_lane_gc_reclaim() guard
  step-7/8  — acquire_warm_lane / acquire_spec_lane create-once discriminator
              + the _seed_warm_lane mark-on-success chokepoint
  step-9/10 — harness escalation helper + resolver (test_pool_storage_guard.py
              continues to grow via later iterations; harness-side tests may
              also live here)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import POOL_ROOT_SENTINEL, GitOps


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, tmp_path: Path) -> GitOps:
    """A GitOps whose worktree_base (tmp_path/project/.worktrees) is NOT
    pre-created — lets tests control base-exists/sentinel-exists independently.

    project_root need not be a real git repo: these predicate/writer tests
    are pure filesystem checks that never shell out to git.
    """
    project_root = tmp_path / 'project'
    project_root.mkdir()
    return GitOps(git_config, project_root)


class TestPoolStoragePresentPredicate:
    """GitOps.pool_storage_present() / mark_pool_storage_present() (step-1/2)."""

    def test_sentinel_constant(self):
        assert POOL_ROOT_SENTINEL == '.pool-root'

    def test_base_missing_is_absent(self, git_ops: GitOps):
        assert not git_ops.worktree_base.exists()
        assert git_ops.pool_storage_present() is False

    def test_base_exists_no_sentinel_is_absent(self, git_ops: GitOps):
        """Simulates an unmounted mountpoint: the dir exists (empty) but the
        sentinel that lives ON the mounted storage was never written."""
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        assert git_ops.worktree_base.exists()
        assert git_ops.pool_storage_present() is False

    def test_mark_then_present_is_true(self, git_ops: GitOps):
        git_ops.mark_pool_storage_present()
        sentinel = git_ops.worktree_base / POOL_ROOT_SENTINEL
        assert sentinel.is_file()
        assert git_ops.pool_storage_present() is True

    def test_mark_creates_missing_base(self, git_ops: GitOps):
        """mark_pool_storage_present() mkdir(parents=True)s worktree_base
        when it does not yet exist (fresh-host / pool-warmup bootstrap)."""
        assert not git_ops.worktree_base.exists()
        git_ops.mark_pool_storage_present()
        assert git_ops.worktree_base.exists()
        assert git_ops.pool_storage_present() is True

    def test_mark_is_idempotent(self, git_ops: GitOps):
        git_ops.mark_pool_storage_present()
        git_ops.mark_pool_storage_present()  # second call: no-op, no raise
        sentinel = git_ops.worktree_base / POOL_ROOT_SENTINEL
        assert sentinel.is_file()
        assert git_ops.pool_storage_present() is True

    def test_present_fail_safe_on_oserror(self, git_ops: GitOps, monkeypatch):
        """A stat that raises OSError must yield False (fail-safe-absent),
        never propagate — an unreadable mount must skip destructive sweeps,
        not crash them."""
        git_ops.mark_pool_storage_present()
        assert git_ops.pool_storage_present() is True

        def _raise(*_args, **_kwargs):
            raise OSError('simulated stat failure')

        monkeypatch.setattr(Path, 'is_file', _raise)
        assert git_ops.pool_storage_present() is False
