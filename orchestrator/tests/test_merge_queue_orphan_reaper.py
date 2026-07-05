"""Tests for SpeculativeMergeWorker.reap_orphaned_merge_worktrees (I6 leak
closure — reap/re-adopt orphaned unregistered ``_merge-<uuid>`` worktrees on
worker startup/recovery, task 2060).

Steps covered:
  step-1  RED   — core reap + headline signal (bare worker, real worktrees)
  step-2  GREEN — implement reap_orphaned_merge_worktrees() reap core
  step-3  RED   — preservation guards (persistent / owned / fresh-within-grace)
  step-4  GREEN — confirm/refine skip ordering + INFO summary log
  step-5  RED   — re-adoption via recovered_branches
  step-6  GREEN — implement re-adoption
  step-7  RED   — fail-open robustness
  step-8  GREEN — implement fail-open guards

This module reuses the bare-worker git_repo/git_config/git_ops fixtures and
_setup_repo from test_merge_queue_resource_audit.py (per-file duplication
convention — see that module's docstring), along with its
_mkdir_worktree/_GRACE/_NOW age-test helper pattern. It imports
orchestrator.merge_queue LOCALLY inside each test so a not-yet-implemented
symbol never breaks collection of the rest of the file during the RED steps.

Unlike test_merge_queue_resource_audit.py's pure-sync worktree_ledger_violations
tests, reap_orphaned_merge_worktrees is async (it awaits
git_ops.cleanup_merge_worktree / git_ops.find_inflight_merge_worktree), so
every test in this module is ``async def`` under ``@pytest.mark.asyncio``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import PERSISTENT_MERGE_WORKTREE_NAME, GitOps, _run

# ---------------------------------------------------------------------------
# Fixtures + helpers (per-file duplication convention — see
# test_merge_queue_resource_audit.py)
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    """Initialise a bare git repository with a single commit on main."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


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
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


def _mkdir_worktree(git_ops: GitOps, name: str, *, mtime: float | None = None) -> Path:
    """Create worktree_base/name (creating worktree_base itself if absent).

    Copied from test_merge_queue_resource_audit.py (per-file duplication
    convention). Used only for cases that don't need a real git admin entry
    (e.g. the persistent ``_merge-verify`` preservation check) — real
    reapable/re-adoptable ephemeral worktrees are made via
    ``git_ops._create_merge_worktree()`` below so the reap primitive
    (``git worktree remove --force``) has a genuine git admin entry to act on.
    """
    git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
    wt = git_ops.worktree_base / name
    wt.mkdir()
    if mtime is not None:
        os.utime(wt, (mtime, mtime))
    return wt


_GRACE = 100.0
_NOW = 1_000_000.0


async def _create_backdated_merge_worktree(git_ops: GitOps, *, age: float) -> Path:
    """Create a real ephemeral ``_merge-<uuid>`` worktree, backdated to age
    seconds before ``_NOW`` (so it reads as aged-past-grace at ``now=_NOW``).
    """
    wt, _ = await git_ops._create_merge_worktree()
    backdated = _NOW - age
    os.utime(wt, (backdated, backdated))
    return wt


async def _registered_worktree_paths(git_ops: GitOps) -> list[str]:
    """Return the ``git worktree list --porcelain`` registered paths."""
    rc, out, _ = await _run(
        ['git', 'worktree', 'list', '--porcelain'], cwd=git_ops.project_root,
    )
    assert rc == 0
    return [
        line[len('worktree '):].strip()
        for line in out.splitlines()
        if line.startswith('worktree ')
    ]


# ---------------------------------------------------------------------------
# step-1 RED / step-2 GREEN: core reap + headline signal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReapOrphanedMergeWorktreesCore:
    """Unit tests for the reap core.

    RED until step-2 GREEN adds reap_orphaned_merge_worktrees to
    merge_queue.py.
    """

    async def test_aged_orphan_reaped_and_ledger_clears(self, git_ops: GitOps) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS = _GRACE
        wt = await _create_backdated_merge_worktree(git_ops, age=_GRACE + 10)

        # Sanity: the detector flags this orphan BEFORE the sweep runs.
        before = worker.worktree_ledger_violations(now=_NOW)
        assert len(before) == 1, f'expected exactly one violation, got: {before!r}'

        report = await worker.reap_orphaned_merge_worktrees(now=_NOW)

        assert not wt.exists(), 'aged orphan must be removed from disk'
        registered = await _registered_worktree_paths(git_ops)
        assert str(wt) not in registered, (
            f'aged orphan must be gone from git worktree list: {registered}'
        )
        assert str(wt.resolve()) in report['reaped']
        assert report['readopted'] == []
        assert worker.worktree_ledger_violations(now=_NOW) == []
        assert worker.snapshot()['resource_audit']['worktree_ledger'] == []

    async def test_fresh_orphan_within_grace_not_reaped(self, git_ops: GitOps) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS = _GRACE
        # Real ephemeral worktree with a FRESH mtime (not backdated) — within
        # grace at now=_NOW.
        wt, _ = await git_ops._create_merge_worktree()

        report = await worker.reap_orphaned_merge_worktrees(now=_NOW)

        assert wt.exists(), 'fresh-within-grace orphan must be preserved'
        assert str(wt.resolve()) not in report['reaped']


# ---------------------------------------------------------------------------
# step-3 RED / step-4 GREEN: preservation guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReapOrphanedMergeWorktreesPreservation:
    """Unit tests asserting the sweep preserves everything it must not touch
    — the persistent ``_merge-verify`` worktree, an already-owned
    (registered) worktree, and a fresh (within-grace) unregistered worktree
    — while still reaping the one genuine aged orphan alongside them in the
    SAME sweep (task 2060 step-3).

    The ``_merge-verify`` and grace-skip preservations already hold
    structurally from step-2 (same discovery/skip logic as
    ``worktree_ledger_violations``); this class is the first to exercise the
    owned-skip explicitly, and the first to combine all three preservation
    guards with a genuine reap in a single sweep.
    """

    async def test_sweep_reaps_only_the_genuine_orphan(self, git_ops: GitOps) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS = _GRACE

        # (1) genuine aged orphan — expected reaped.
        orphan = await _create_backdated_merge_worktree(git_ops, age=_GRACE + 10)

        # (2) persistent _merge-verify, aged past grace — expected preserved.
        verify_wt = _mkdir_worktree(
            git_ops, PERSISTENT_MERGE_WORKTREE_NAME, mtime=_NOW - _GRACE - 10,
        )

        # (3) aged ephemeral, but pre-registered (owned) — expected preserved.
        owned_wt = await _create_backdated_merge_worktree(git_ops, age=_GRACE + 10)
        worker._register_owned_merge_worktree(owned_wt)

        # (4) fresh (within-grace) unregistered — expected preserved.
        fresh_wt, _ = await git_ops._create_merge_worktree()

        report = await worker.reap_orphaned_merge_worktrees(now=_NOW)

        assert str(orphan.resolve()) in report['reaped']
        assert not orphan.exists()

        assert verify_wt.exists(), '_merge-verify must survive the sweep'
        assert str(verify_wt.resolve()) not in report['reaped']

        assert owned_wt.exists(), 'owned/registered worktree must survive the sweep'
        assert str(owned_wt.resolve()) not in report['reaped']

        assert fresh_wt.exists(), 'fresh-within-grace worktree must survive the sweep'
        assert str(fresh_wt.resolve()) not in report['reaped']

        assert report['reaped'] == [str(orphan.resolve())], (
            f'expected only the genuine orphan reaped, got: {report["reaped"]!r}'
        )
