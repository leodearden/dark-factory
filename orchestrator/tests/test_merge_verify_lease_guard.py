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

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run
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


# ---------------------------------------------------------------------------
# Real-git fixtures (mirroring test_git_ops.py's TestPersistentMergeWorktree)
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


async def _get_merge_commit(git_ops: GitOps, branch_name: str, filename: str) -> str:
    """Create a feature branch, commit a file, merge to main, return merge_commit."""
    wt_info = await git_ops.create_worktree(branch_name)
    (wt_info.path / filename).write_text(f'{branch_name} = True\n')
    await git_ops.commit(wt_info.path, f'Add {filename}')
    result = await git_ops.merge_to_main(wt_info.path, branch_name)
    assert result.success and result.merge_commit
    return result.merge_commit


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A real temporary git repository with an initial commit (step-13+)."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


@pytest.fixture
def real_git_ops(git_repo: Path) -> GitOps:
    """A real-git-backed GitOps for tests exercising reset_persistent_merge_worktree
    end to end (unlike the pure-FS ``_git_ops`` helper above, which never shells
    out to git)."""
    return GitOps(_git_config(), git_repo)


@pytest.mark.asyncio
class TestResetPersistentMergeWorktreeLeaseGuard:
    """reset_persistent_merge_worktree refuses to clobber a live foreign
    merge-verify lease (step-13/14).

    Each test first registers the persistent ``_merge-verify`` worktree for
    real (a create-once call), so the call under test takes the
    reset-in-place branch — the one an in-flight verify would actually be
    sitting in when the clobber incident happens.
    """

    async def test_foreign_live_holder_refuses_reset(
        self, real_git_ops: GitOps, monkeypatch: pytest.MonkeyPatch,
    ):
        from orchestrator.git_ops import MergeVerifyLeaseHeld  # noqa: PLC0415

        merge_commit_a = await _get_merge_commit(real_git_ops, 'lease-a1', 'a1.py')
        await real_git_ops.reset_persistent_merge_worktree(merge_commit_a)
        merge_commit_b = await _get_merge_commit(real_git_ops, 'lease-a2', 'a2.py')

        # A genuinely LIVE pgid (our own test process's real pgid) recorded as
        # the holder, so the lease reads as held by a live process. To make
        # it read as FOREIGN (not self), make GitOps see a different "self"
        # pgid at compare time.
        foreign_holder = os.getpgrp()
        write_lock_holder_pgid(real_git_ops.worktree_base, foreign_holder)
        monkeypatch.setattr(os, 'getpgrp', lambda: foreign_holder + 1)

        with patch('orchestrator.git_ops._run', new_callable=AsyncMock) as mock_run:
            with pytest.raises(MergeVerifyLeaseHeld):
                await real_git_ops.reset_persistent_merge_worktree(merge_commit_b)
        mock_run.assert_not_awaited()

        # Worktree must be untouched: HEAD still at merge_commit_a.
        _, head_sha, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=real_git_ops.persistent_merge_worktree_path,
        )
        assert head_sha.strip() == merge_commit_a.strip(), (
            'a foreign live lease holder must block the reset entirely'
        )

    async def test_no_holder_reset_proceeds(self, real_git_ops: GitOps):
        merge_commit_a = await _get_merge_commit(real_git_ops, 'lease-b1', 'b1.py')
        await real_git_ops.reset_persistent_merge_worktree(merge_commit_a)
        merge_commit_b = await _get_merge_commit(real_git_ops, 'lease-b2', 'b2.py')

        assert read_lock_holder_pgid(real_git_ops.worktree_base) is None, (
            'sanity: no holder-pgid file recorded'
        )

        warm_path = await real_git_ops.reset_persistent_merge_worktree(merge_commit_b)

        assert warm_path == real_git_ops.persistent_merge_worktree_path
        _, head_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=warm_path)
        assert head_sha.strip() == merge_commit_b.strip(), (
            'no holder: reset must proceed (HEAD must advance to merge_commit_b)'
        )

    async def test_self_held_lease_reset_proceeds(self, real_git_ops: GitOps):
        merge_commit_a = await _get_merge_commit(real_git_ops, 'lease-c1', 'c1.py')
        await real_git_ops.reset_persistent_merge_worktree(merge_commit_a)
        merge_commit_b = await _get_merge_commit(real_git_ops, 'lease-c2', 'c2.py')

        write_lock_holder_pgid(real_git_ops.worktree_base, os.getpgrp())  # self-held

        warm_path = await real_git_ops.reset_persistent_merge_worktree(merge_commit_b)

        assert warm_path == real_git_ops.persistent_merge_worktree_path
        _, head_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=warm_path)
        assert head_sha.strip() == merge_commit_b.strip(), (
            'self-held lease: reset must proceed (self excluded from the guard)'
        )
