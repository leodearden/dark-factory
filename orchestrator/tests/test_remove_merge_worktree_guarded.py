"""Real-git tests for GitOps.remove_merge_worktree_guarded.

PRD: docs/prds/merge-worktree-lifecycle-integrity.md, task alpha (C1).

remove_merge_worktree_guarded is the lease-enforced removal primitive that
closes the incident's 23s TOCTOU: a non-blocking try-acquire of the tree's
merge-verify flock is HELD across the ``git worktree remove`` (acquire-then-
remove, never check-then-remove). This module pins the full outcome
vocabulary — 'removed', 'skipped_lease_held', 'skipped_persistent',
'not_present', 'failed', 'skipped_lock_error' — via real git worktrees (no
mocking of git itself), mirroring test_inflight_verify_merge_lease.py's
fixture pattern.

TestRemoveMergeWorktreeGuarded exercises the primitive directly: uncontended
removal; a lease-held skip that warns exactly once naming the holder pgid
and reason (paired with a dead-holder fail-open positive control proving a
stale pgid record never wedges removal); the persistent-lane exemption for
both persistent worktrees; the not_present vs. failed outcome split; and the
sibling ``<path>.lock`` file unlink-on-removal vs. retain-on-skip contract.
TestCleanupMergeWorktreeRouting pins that cleanup_merge_worktree delegates
to the guarded primitive, so a lease-held tree is skipped rather than
force-removed, while uncontended removal is unchanged.
"""
from __future__ import annotations

import asyncio
import errno
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


def _raise_enospc(*_a, **_k):
    """Simulate acquire_merge_verify_flock's PRE-bounded-wait failure.

    verify_cancel.py does ``path.parent.mkdir(...)`` + ``os.open(path,
    O_RDWR | O_CREAT)`` BEFORE its own try/except, so ENOSPC (a documented
    recurring condition on this lane), EACCES, EMFILE or EROFS raises out
    of the acquire rather than returning the None that means 'contended'.
    """
    raise OSError(errno.ENOSPC, 'No space left on device')


def _raise_emfile_spawn(*_a, **_k):
    """Simulate the SECOND OSError escape: the git removal that cannot spawn.

    ``_run`` raises before the child ever executes in two ways --
    :class:`~orchestrator.git_ops.WorktreeMissing` (a ``FileNotFoundError``,
    hence an ``OSError``) when its ``cwd`` has vanished, and a bare
    ``OSError`` out of ``asyncio.create_subprocess_exec`` under EMFILE/
    ENOMEM. EMFILE is the pointed case: fd exhaustion fails the lane-lock
    open and the subprocess spawn from the SAME root cause, so guarding only
    the former would leave cleanup_merge_worktree's "Never raises" contract
    false under exactly the condition that motivated the guard.
    """
    raise OSError(errno.EMFILE, 'Too many open files')


def _raise_runtime_error(*_a, **_k):
    """A NON-OSError failure, for pinning the narrow ``except OSError``.

    Both guards in remove_merge_worktree_guarded catch ``OSError`` only.
    Widening either to a bare ``except Exception`` -- a very natural refactor
    for a method documented as never raising -- would swallow programmer
    errors (and, for ``BaseException``, ``CancelledError``) in silence. These
    tests fail loudly if that ever happens.
    """
    raise RuntimeError('not an OSError -- must stay loud')


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
        failure on an existing-but-unregistered directory.

        Also pins the SECOND tree-gone unlink. The primitive's docstring
        promises the sibling ``<path>.lock`` unlink for 'removed' AND
        'not_present', but only 'removed' was covered (by
        test_removal_unlinks_its_own_lock_file) -- leaving the branch that
        creates a lock file for a tree that never existed unasserted."""
        p = git_ops.worktree_base / '_merge-never-created'
        lock_path = lane_lock_path(p)

        outcome = await git_ops.remove_merge_worktree_guarded(p, reason='t')

        assert outcome == 'not_present'
        assert not lock_path.exists(), (
            f'not_present is a tree-gone outcome: the flock acquire CREATES '
            f'<path>.lock as a side effect, so the primitive must unlink it '
            f'here exactly as it does on removed; {lock_path} leaked'
        )
        assert not list(git_ops.worktree_base.glob('_merge-*')), (
            'a never-created path must leave no _merge-* dir OR lock-file orphan'
        )

    async def test_non_worktree_directory_returns_failed(self, git_ops: GitOps):
        """Positive control pinning the distinct 'failed' outcome: an
        existing directory that is NOT a registered git worktree makes
        `git worktree remove --force` itself error. The directory must
        survive (the remove failed) and the sibling lock file must be
        retained -- unlink-on-tree-gone only fires for 'removed', never for
        'failed'."""
        p = git_ops.worktree_base / '_merge-plain-dir'
        p.mkdir(parents=True)
        lock_path = lane_lock_path(p)

        outcome = await git_ops.remove_merge_worktree_guarded(p, reason='t')

        assert outcome == 'failed'
        assert p.exists(), 'a failed removal must leave the directory intact'
        assert lock_path.exists(), (
            'the lock file (created as a side effect of acquiring the flock) '
            'must be retained, not unlinked, when the lane survives a failed removal'
        )

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

    async def test_lane_lock_open_error_returns_skipped_lock_error_and_warns_once(
        self, git_ops: GitOps, monkeypatch, caplog,
    ):
        """A lane lock that cannot be OPENED AT ALL must degrade to a distinct
        skip outcome, never propagate. acquire_merge_verify_flock's pre-wait
        ``mkdir`` + ``os.open`` sit outside its own try, so ENOSPC/EACCES/
        EMFILE/EROFS raise instead of returning the None that means
        'contended' -- and this primitive backs cleanup_merge_worktree's
        documented "Never raises" contract.

        The tree must SURVIVE: an unopenable lock never confirmed the lease,
        so a live verify may still hold the flock on an existing lock inode
        while our own open fails. The tree is left for the merge reaper."""
        wt = await _make_ephemeral_worktree(git_ops)
        monkeypatch.setattr(
            'orchestrator.git_ops.acquire_merge_verify_flock', _raise_enospc,
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            outcome = await git_ops.remove_merge_worktree_guarded(wt, reason='sweep')

        assert outcome == 'skipped_lock_error'
        assert wt.exists(), (
            'an unopenable lane lock leaves the lease unconfirmed, so the tree '
            'must survive for the merge reaper'
        )

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, (
            f'expected exactly one WARNING, got {len(warnings)}: '
            f'{[r.getMessage() for r in warnings]}'
        )
        message = warnings[0].getMessage()
        assert 'sweep' in message, message
        assert 'No space left on device' in message, message

    async def test_lane_lock_open_error_never_unlinks_a_lock_it_did_not_acquire(
        self, git_ops: GitOps, monkeypatch,
    ):
        """Safety property mirroring test_lease_held_skip_preserves_holders_lock_file:
        a failed open acquired nothing, so an already-present lock file (which
        may be a live holder's) must be left untouched."""
        wt = await _make_ephemeral_worktree(git_ops)
        lock_path = lane_lock_path(wt)
        lock_path.touch()
        monkeypatch.setattr(
            'orchestrator.git_ops.acquire_merge_verify_flock', _raise_enospc,
        )

        outcome = await git_ops.remove_merge_worktree_guarded(wt, reason='sweep')

        assert outcome == 'skipped_lock_error'
        assert lock_path.exists(), (
            'a lock this call did not acquire must never be yanked -- it may be '
            'a live holder\'s'
        )

    async def test_git_removal_spawn_error_returns_failed_and_warns_once(
        self, git_ops: GitOps, monkeypatch, caplog,
    ):
        """The lane-lock open is not the only OSError escape: on the ACQUIRED-
        lease path the ``git worktree remove`` subprocess can fail to spawn at
        all (EMFILE/ENOMEM from create_subprocess_exec, or WorktreeMissing when
        project_root vanished), which would propagate straight through
        cleanup_merge_worktree's "Never raises" contract.

        The outcome must be 'failed', NOT 'skipped_lock_error': we hold the
        flock here, so the lease acquire already proved there is no live
        holder -- the exact premise that licenses cleanup's force-rmtree
        fallback. 'failed' is also honest, being precisely "git's removal did
        not succeed and the tree survives"."""
        wt = await _make_ephemeral_worktree(git_ops)
        lock_path = lane_lock_path(wt)
        monkeypatch.setattr('orchestrator.git_ops._run', _raise_emfile_spawn)

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            outcome = await git_ops.remove_merge_worktree_guarded(wt, reason='sweep')

        assert outcome == 'failed'
        assert wt.exists(), 'git never ran, so the tree is necessarily still there'
        assert lock_path.exists(), (
            "'failed' retains the sibling lock -- the tree, and thus its lane, "
            'survives (same rule as the non-worktree-directory failed case)'
        )

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, (
            f'expected exactly one WARNING, got {len(warnings)}: '
            f'{[r.getMessage() for r in warnings]}'
        )
        message = warnings[0].getMessage()
        assert 'sweep' in message, message
        assert 'Too many open files' in message, message

    async def test_non_oserror_from_lane_lock_acquire_propagates(
        self, git_ops: GitOps, monkeypatch,
    ):
        """Pins the lock guard's narrow ``except OSError`` in BEHAVIOUR, not
        just in a comment. A non-OSError out of the acquire is a programmer
        error (or, for BaseException, a cancellation) and must stay loud --
        widening the catch to ``except Exception`` must break this test."""
        wt = await _make_ephemeral_worktree(git_ops)
        monkeypatch.setattr(
            'orchestrator.git_ops.acquire_merge_verify_flock', _raise_runtime_error,
        )

        with pytest.raises(RuntimeError, match='must stay loud'):
            await git_ops.remove_merge_worktree_guarded(wt, reason='sweep')

    async def test_non_oserror_from_git_removal_propagates(
        self, git_ops: GitOps, monkeypatch,
    ):
        """Same narrow-catch pin for the second guard, the git-removal spawn.

        Degrading a genuine programmer error to 'failed' would be strictly
        worse than raising: 'failed' routes to cleanup's force-rmtree, so a
        widened catch would silently destroy trees on a code bug."""
        wt = await _make_ephemeral_worktree(git_ops)
        monkeypatch.setattr('orchestrator.git_ops._run', _raise_runtime_error)

        with pytest.raises(RuntimeError, match='must stay loud'):
            await git_ops.remove_merge_worktree_guarded(wt, reason='sweep')


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

    async def test_cleanup_never_raises_when_lane_lock_cannot_be_opened(
        self, git_ops: GitOps, monkeypatch,
    ):
        """Makes cleanup_merge_worktree's "Never raises; idempotent" docstring
        contract actually checkable. An OSError out of the flock acquire (the
        pre-wait mkdir/os.open) must not escape.

        The tree must SURVIVE: the crash-safe ``shutil.rmtree`` fallback fires
        ONLY on 'failed', and that force-removal is sound only because 'failed'
        means the lease acquire SUCCEEDED and proved no live holder. An
        unopenable lock proves nothing about holders, so it must never reach
        that fallback -- a future refactor degrading this path to 'failed'
        reintroduces the 23s TOCTOU C1 exists to close, and must break here."""
        wt = await _make_ephemeral_worktree(git_ops)
        monkeypatch.setattr(
            'orchestrator.git_ops.acquire_merge_verify_flock', _raise_enospc,
        )

        await git_ops.cleanup_merge_worktree(wt)

        assert wt.exists(), (
            'the rmtree fallback must NOT fire on a lock-open error: the lease '
            'was never confirmed, so a live holder cannot be ruled out'
        )

        # The docstring's idempotency half: a re-call must also not raise.
        await git_ops.cleanup_merge_worktree(wt)
        assert wt.exists()

    async def test_cleanup_never_raises_when_the_git_removal_cannot_spawn(
        self, git_ops: GitOps, monkeypatch,
    ):
        """The OTHER half of the "Never raises" contract. Guarding only the
        lane-lock open would leave the contract false under fd exhaustion --
        the very condition (EMFILE) that motivates the lock guard fails the
        subprocess spawn too, so both escapes fire from one root cause.

        Unlike the lock-open path the tree here IS force-removed, and that is
        correct: the primitive held the flock, so the lease acquire proved no
        live holder -- the premise the rmtree fallback is documented to rest
        on. ``_prune_registrations`` swallows its own spawn failure, so
        nothing escapes downstream of the fallback either."""
        wt = await _make_ephemeral_worktree(git_ops)
        monkeypatch.setattr('orchestrator.git_ops._run', _raise_emfile_spawn)

        await git_ops.cleanup_merge_worktree(wt)

        assert not wt.exists(), (
            'the lease was confirmed (we held the flock), so the crash-safe '
            'rmtree fallback is licensed and must clear this unleased tree'
        )
        assert not lane_lock_path(wt).exists(), (
            'cleanup unlinks the lock the primitive retained on failed, once '
            'the lane is gone'
        )

        # Idempotency half: a re-call over the now-absent tree must not raise
        # either (the primitive short-circuits to 'not_present').
        await git_ops.cleanup_merge_worktree(wt)
