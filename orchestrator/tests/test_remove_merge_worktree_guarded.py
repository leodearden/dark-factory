"""Real-git tests for GitOps.remove_merge_worktree_guarded.

PRD: docs/prds/merge-worktree-lifecycle-integrity.md, task alpha (C1).

remove_merge_worktree_guarded is the lease-enforced removal primitive that
closes the incident's 23s TOCTOU: a non-blocking try-acquire of the tree's
merge-verify flock is HELD across the ``git worktree remove`` (acquire-then-
remove, never check-then-remove). This module pins the full outcome
vocabulary — 'removed', 'skipped_lease_held', 'skipped_persistent',
'not_present', 'failed' — via real git worktrees (no mocking of git itself),
mirroring test_inflight_verify_merge_lease.py's fixture pattern. No test
bodies yet — this scaffolding adds only the shared real-git fixtures and
helpers each subsequent step's test function builds on.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.verify_cancel import (
    acquire_merge_verify_flock,
    lane_lock_path,
    release_merge_verify_flock,
    remove_lock_holder_pgid,
    write_lock_holder_pgid,
)

# ---------------------------------------------------------------------------
# Real-git fixtures (adapted from test_inflight_verify_merge_lease.py:39-76)
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


async def _head_sha(repo: Path) -> str:
    rc, out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0
    return out.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


@pytest.fixture
def git_ops(git_repo: Path) -> GitOps:
    git_config = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )
    return GitOps(git_config, git_repo)


async def _make_ephemeral_worktree(git_ops: GitOps) -> Path:
    """Build a real ephemeral ``_merge-<uuid>`` worktree at the repo's HEAD."""
    return await git_ops.create_throwaway_verify_worktree(await _head_sha(git_ops.project_root))


# ---------------------------------------------------------------------------
# step-1: uncontended removal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRemoveMergeWorktreeGuarded:
    """GitOps.remove_merge_worktree_guarded — lease-enforced removal (C1).

    Acquire-then-remove, never check-then-remove: the primitive holds the
    tree's merge-verify flock across the ``git worktree remove`` so no
    verify can start in the window between the liveness check and the
    delete (the incident's 23s TOCTOU).
    """

    async def test_uncontended_removal_returns_removed(self, git_ops: GitOps):
        """Positive control paired with the lease-held skip test: an
        uncontended tree must actually be removed, not just left alone by an
        inert stub that always reports skipped."""
        wt = await _make_ephemeral_worktree(git_ops)

        outcome = await git_ops.remove_merge_worktree_guarded(wt, reason='t')

        assert outcome == 'removed'
        assert not wt.exists()

    async def test_lease_held_skip_returns_skipped_lease_held_and_warns_once(
        self, git_ops: GitOps, caplog,
    ):
        """A LIVE holder of the tree's merge-verify flock makes removal skip
        rather than force through — the fix for the incident's TOCTOU. The
        skip must be OBSERVED via both the returned outcome and a single
        WARNING naming the holder pgid and the caller's reason, never merely
        inferred from the tree surviving."""
        wt = await _make_ephemeral_worktree(git_ops)
        fd = acquire_merge_verify_flock(lane_lock_path(wt), 5.0)
        assert fd is not None, 'test setup: must be able to acquire the tree lease itself'
        write_lock_holder_pgid(git_ops.worktree_base, os.getpgrp())
        try:
            with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
                outcome = await git_ops.remove_merge_worktree_guarded(wt, reason='sweep')

            assert outcome == 'skipped_lease_held'
            assert wt.exists(), 'a live lease holder must leave the tree intact'

            warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert len(warnings) == 1, (
                f'expected exactly one WARNING, got {len(warnings)}: '
                f'{[r.getMessage() for r in warnings]}'
            )
            message = warnings[0].getMessage()
            assert str(os.getpgrp()) in message, message
            assert 'sweep' in message, message
        finally:
            release_merge_verify_flock(fd)
            remove_lock_holder_pgid(git_ops.worktree_base)

    async def test_dead_holder_pgid_fails_open_and_removes(
        self, git_ops: GitOps, caplog,
    ):
        """Fail-open positive control: a STALE holder-pgid record (no live
        process actually holding the flock) must never wedge removal. The
        kernel auto-releases a crashed holder's flock, so the non-blocking
        acquire simply succeeds — proving the primitive gates on the flock
        itself, never on the (fail-open, best-effort) pgid rendezvous file."""
        wt = await _make_ephemeral_worktree(git_ops)
        write_lock_holder_pgid(git_ops.worktree_base, 999999)
        try:
            with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
                outcome = await git_ops.remove_merge_worktree_guarded(wt, reason='sweep')

            assert outcome == 'removed'
            assert not wt.exists()
            assert not caplog.records, (
                f'a stale holder-pgid record must never block removal or emit a '
                f'skip warning; got: {[r.getMessage() for r in caplog.records]}'
            )
        finally:
            remove_lock_holder_pgid(git_ops.worktree_base)

    async def test_persistent_merge_worktree_path_is_skipped(self, git_ops: GitOps):
        """Persistent lanes are never removed regardless of lease — the
        skip fires before any flock is even touched."""
        p = git_ops.persistent_merge_worktree_path
        p.mkdir(parents=True)

        outcome = await git_ops.remove_merge_worktree_guarded(p, reason='t')

        assert outcome == 'skipped_persistent'
        assert p.exists()

    async def test_persistent_offline_deep_worktree_path_is_skipped(self, git_ops: GitOps):
        """Mirrors test_persistent_merge_worktree_path_is_skipped for the
        second persistent lane (_offline-deep)."""
        p = git_ops.persistent_offline_deep_worktree_path
        p.mkdir(parents=True)

        outcome = await git_ops.remove_merge_worktree_guarded(p, reason='t')

        assert outcome == 'skipped_persistent'
        assert p.exists()

    async def test_never_created_path_returns_not_present(self, git_ops: GitOps):
        """A path that was never created (uncontended flock, nothing to
        remove) must report 'not_present' distinctly from a git-remove
        failure on an existing-but-unregistered directory."""
        p = git_ops.worktree_base / '_merge-never-created'

        outcome = await git_ops.remove_merge_worktree_guarded(p, reason='t')

        assert outcome == 'not_present'

    async def test_non_worktree_directory_returns_failed(self, git_ops: GitOps):
        """Positive control pinning the distinct 'failed' outcome: an
        existing directory that is NOT a registered git worktree makes
        `git worktree remove --force` itself error."""
        p = git_ops.worktree_base / '_merge-plain-dir'
        p.mkdir(parents=True)

        outcome = await git_ops.remove_merge_worktree_guarded(p, reason='t')

        assert outcome == 'failed'

    async def test_removal_unlinks_its_own_lock_file(self, git_ops: GitOps):
        """The removing party unlinks the sibling ``<path>.lock`` on the
        tree-gone outcomes, so an ephemeral removal leaves no orphan
        ``_merge-<uuid>.lock`` behind (regression: the sibling lock file
        matches test_git_ops.py's ``_merge-*`` no-leak glob). Mirrors the
        task-2507 ephemeral_worktree precedent."""
        wt = await _make_ephemeral_worktree(git_ops)
        lock_path = lane_lock_path(wt)

        outcome = await git_ops.remove_merge_worktree_guarded(wt, reason='t')

        assert outcome == 'removed'
        assert not wt.exists()
        assert not lock_path.exists(), (
            f'the removing party must unlink its own lock file; {lock_path} leaked'
        )
        assert not list(git_ops.worktree_base.glob('_merge-*')), (
            'no _merge-* directory OR lock-file orphan may survive an '
            'uncontended removal'
        )

    async def test_lease_held_skip_preserves_holders_lock_file(
        self, git_ops: GitOps,
    ):
        """Safety property: a skipped_lease_held outcome must NEVER unlink the
        lock file — this call did not acquire it, so yanking a live holder's
        lock file would corrupt the serialization the holder relies on."""
        wt = await _make_ephemeral_worktree(git_ops)
        lock_path = lane_lock_path(wt)
        fd = acquire_merge_verify_flock(lock_path, 5.0)
        assert fd is not None, 'test setup: must be able to acquire the tree lease itself'
        write_lock_holder_pgid(git_ops.worktree_base, os.getpgrp())
        try:
            outcome = await git_ops.remove_merge_worktree_guarded(wt, reason='sweep')

            assert outcome == 'skipped_lease_held'
            assert lock_path.exists(), (
                'a lock we did not acquire must be left untouched for its live holder'
            )
        finally:
            release_merge_verify_flock(fd)
            remove_lock_holder_pgid(git_ops.worktree_base)


# ---------------------------------------------------------------------------
# step-9: cleanup_merge_worktree routes through the guarded primitive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCleanupMergeWorktreeRouting:
    """cleanup_merge_worktree delegates to remove_merge_worktree_guarded (C1).

    The only intended behavior change: a lease-held tree is now skipped
    instead of force-removed out from under a live verify (the incident's
    bug). Uncontended and persistent-lane behavior are unchanged.
    """

    async def test_cleanup_skips_tree_under_held_lease(self, git_ops: GitOps, caplog):
        """Pre-routing, cleanup_merge_worktree force-removes unconditionally
        -- this IS the incident's TOCTOU bug. Post-routing it must leave a
        lease-held tree intact and emit exactly one skip WARNING, rather
        than force through."""
        wt = await _make_ephemeral_worktree(git_ops)
        fd = acquire_merge_verify_flock(lane_lock_path(wt), 5.0)
        assert fd is not None, 'test setup: must be able to acquire the tree lease itself'
        write_lock_holder_pgid(git_ops.worktree_base, os.getpgrp())
        try:
            with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
                await git_ops.cleanup_merge_worktree(wt)

            assert wt.exists(), 'a live lease holder must leave the tree intact'
            warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert len(warnings) == 1, (
                f'expected exactly one WARNING, got {len(warnings)}: '
                f'{[r.getMessage() for r in warnings]}'
            )
        finally:
            release_merge_verify_flock(fd)
            remove_lock_holder_pgid(git_ops.worktree_base)

    async def test_cleanup_still_removes_uncontended_tree(self, git_ops: GitOps):
        """Positive control paired with the lease-held skip: an uncontended
        tree must still actually be removed by cleanup_merge_worktree."""
        wt = await _make_ephemeral_worktree(git_ops)

        await git_ops.cleanup_merge_worktree(wt)

        assert not wt.exists()
