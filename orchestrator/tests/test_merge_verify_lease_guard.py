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
import fcntl
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.verify_cancel import (
    lane_lock_path,
    read_lock_holder_pgid,
    write_lock_holder_pgid,
)
from orchestrator.warm_lane_pool import WarmLanePool


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

    async def test_contended_flock_raises_and_records_no_lease(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """A contended flock (bounded wait times out) RAISES
        MergeVerifyLeaseContended and records NO lease.

        The verify must then be DEFERRED/requeued by the caller, never run
        unprotected — this is task 2828's limb-2 fix, replacing the old
        yield-without-a-lease path that let a 1--2h verify run fully exposed
        to a concurrent reseed/thin/gc clobber.
        """
        from orchestrator.git_ops import MergeVerifyLeaseContended  # noqa: PLC0415

        git_ops = _git_ops(tmp_path)
        # Simulate acquire_merge_verify_flock's bounded-wait TIMEOUT (-> None).
        monkeypatch.setattr(
            'orchestrator.git_ops.acquire_merge_verify_flock',
            lambda *a, **k: None,
        )

        entered = False
        with pytest.raises(MergeVerifyLeaseContended):
            async with git_ops.merge_verify_lease():
                entered = True  # body must never run

        assert not entered, 'lease body must NOT run on a contended flock'
        assert read_lock_holder_pgid(git_ops.worktree_base) is None, (
            'no holder-pgid may be recorded when the lease is refused'
        )
        assert git_ops._merge_verify_lease_active() is False

    async def test_flock_acquire_runs_off_the_event_loop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """The (now minutes-long) bounded-wait flock acquire must run OFF the
        event loop via asyncio.to_thread, never inline on it.

        A synchronous minutes-long poll on the loop would freeze the whole
        orchestrator; we prove off-thread execution by asserting the acquire
        ran on a DIFFERENT thread than the event-loop thread.
        """
        import threading  # noqa: PLC0415

        from orchestrator.git_ops import MergeVerifyLeaseContended  # noqa: PLC0415

        git_ops = _git_ops(tmp_path)
        main_thread_id = threading.get_ident()
        seen: dict[str, int] = {}

        def record_thread(*_a, **_k):
            seen['thread_id'] = threading.get_ident()
            return None  # timeout -> contended (keeps this a single-path test)

        monkeypatch.setattr(
            'orchestrator.git_ops.acquire_merge_verify_flock', record_thread,
        )

        with pytest.raises(MergeVerifyLeaseContended):
            async with git_ops.merge_verify_lease():
                pass

        assert seen.get('thread_id') is not None, 'acquire was never invoked'
        assert seen['thread_id'] != main_thread_id, (
            'acquire_merge_verify_flock must run off the event-loop thread '
            '(asyncio.to_thread), not inline on it'
        )


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

        with (
            patch('orchestrator.git_ops._run', new_callable=AsyncMock) as mock_run,
            pytest.raises(MergeVerifyLeaseHeld),
        ):
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


def _write_warm_lane_gc_stub(project_root: Path) -> Path:
    """Write a minimal ``warm-lane-gc.sh`` stub at ``<project_root>/scripts/``.

    Mirrors test_pool_storage_guard.py's ``_write_warm_lane_gc_stub`` —
    existence is all ``_run_warm_lane_gc_reclaim``'s ``script.exists()``
    check cares about; ``_run`` is mocked in these tests so the stub body
    never actually runs.
    """
    scripts_dir = project_root / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script = scripts_dir / 'warm-lane-gc.sh'
    script.write_text('#!/usr/bin/env bash\nexit 0\n')
    script.chmod(0o755)
    return script


@pytest.mark.asyncio
class TestGcReclaimDefersOnMergeVerifyLease:
    """_run_warm_lane_gc_reclaim defers (127, fail-soft) while ANY
    merge-verify lease is held — INCLUDING our own (step-15/16).

    Unlike the reset guard (which excludes self so the normal
    reset-then-verify flow works), the GC cadence must defer to an
    in-process local verify just as much as to a foreign one, so self is
    NOT excluded here.

    Sentinel/base setup uses ``mark_pool_storage_present()`` directly (the
    simplest way to make the BUG-2 ``_reconcile_pool_storage_before_sweep``
    gate pass) so a skip in these tests is attributable ONLY to the lease,
    never to the sentinel.
    """

    async def test_lease_held_defers_script_never_spawned(self, tmp_path: Path):
        git_ops = _git_ops(tmp_path)
        git_ops.mark_pool_storage_present()
        git_ops.warm_lane_pool = WarmLanePool(worktree_base=git_ops.worktree_base, size=1)
        script = _write_warm_lane_gc_stub(git_ops.project_root)
        assert script.exists()

        write_lock_holder_pgid(git_ops.worktree_base, os.getpgrp())  # live lease held

        mock_run = AsyncMock(return_value=(0, '', ''))
        with patch('orchestrator.git_ops._run', mock_run):
            rc = await git_ops._run_warm_lane_gc_reclaim()

        assert rc == 127, f'expected fail-soft defer sentinel while a lease is held, got {rc}'
        mock_run.assert_not_awaited()

    async def test_no_lease_reclaim_proceeds_control(self, tmp_path: Path):
        """Control: with no holder-pgid recorded, the reclaim script IS spawned."""
        git_ops = _git_ops(tmp_path)
        git_ops.mark_pool_storage_present()
        git_ops.warm_lane_pool = WarmLanePool(worktree_base=git_ops.worktree_base, size=1)
        script = _write_warm_lane_gc_stub(git_ops.project_root)

        assert read_lock_holder_pgid(git_ops.worktree_base) is None

        mock_run = AsyncMock(return_value=(0, '', ''))
        with patch('orchestrator.git_ops._run', mock_run):
            rc = await git_ops._run_warm_lane_gc_reclaim()

        assert rc == 0
        mock_run.assert_awaited_once_with(
            [str(script), 'reclaim', '--mount', str(git_ops.worktree_base)],
            cwd=git_ops.project_root,
        )


# ---------------------------------------------------------------------------
# task 2685, step-1 — merge_verify_lease() must hold the SHARED
# <lane_dir>.lock (lane_lock_path(persistent_merge_worktree_path)), not the
# divergent .merge_verify.lock.
#
# Incident: SEED/mutator actors (reify's seed-warm-lane.sh / thin-warm-lane.sh
# / warm-lane-gc.sh, and DF's own GitOps._seed_warm_lane) all take
# `flock -x <lane_dir>.lock`. merge_verify_lease() instead held a DIFFERENT
# file, .merge_verify.lock (same directory, different inode) — so a live
# verify never excluded a concurrent reseed/thin/gc, and the verify's
# working tree could be clobbered mid-run. Both tests below are RED before
# step-2's fix.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMergeVerifyLeaseHoldsLaneLock:
    """merge_verify_lease() holds <lane_dir>.lock, not .merge_verify.lock (task 2685)."""

    async def test_lease_holds_lane_lock_and_pgid_rendezvous(self, tmp_path: Path):
        """(a) FLOCK-PATH PIN: the lease must hold
        lane_lock_path(persistent_merge_worktree_path) (=
        <worktree_base>/_merge-verify.lock for the singleton persistent
        merge lane) — a non-blocking flock re-acquire on that SAME path
        from this test process must fail while the lease is held. The pgid
        rendezvous (read_lock_holder_pgid) must also still hold, unchanged.
        """
        git_ops = _git_ops(tmp_path)
        lock_path = lane_lock_path(git_ops.persistent_merge_worktree_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        async with git_ops.merge_verify_lease():
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(fd)
            assert read_lock_holder_pgid(git_ops.worktree_base) == os.getpgrp(), (
                'pgid rendezvous must still be recorded while the lease is held'
            )

    async def test_lease_excludes_concurrent_seed_warm_lane(
        self, real_git_ops: GitOps, monkeypatch: pytest.MonkeyPatch,
    ):
        """(b) END-TO-END MUTUAL EXCLUSION: a live merge_verify_lease() must
        block a concurrent _seed_warm_lane (the reseed side, real subprocess
        flock) on the SAME <lane_dir>.lock, rather than letting it proceed
        and clobber the tree underneath the verify.
        """
        from orchestrator import git_ops as git_ops_mod  # noqa: PLC0415

        merge_commit = await _get_merge_commit(real_git_ops, 'lease-lock-1', 'll1.py')
        warm_path = await real_git_ops.reset_persistent_merge_worktree(merge_commit)

        scripts_dir = warm_path / 'scripts'
        scripts_dir.mkdir(parents=True, exist_ok=True)
        marker = warm_path / 'seeded.marker'
        script = scripts_dir / 'seed-warm-lane.sh'
        script.write_text(
            f'#!/usr/bin/env bash\nset -euo pipefail\necho seeded > {marker}\nexit 0\n'
        )
        script.chmod(0o755)

        # Shrink the outer <lane_dir>.lock bound so the test doesn't have to
        # wait the real 30s for _seed_warm_lane's lock-wait timeout.
        monkeypatch.setattr(git_ops_mod, '_SEED_WARM_LANE_LOCK_WAIT_SECS', 2)

        async with real_git_ops.merge_verify_lease():
            rc = await asyncio.wait_for(
                real_git_ops._seed_warm_lane(warm_path, '--fresh-checkout'), 10,
            )

        assert rc == 124, (
            f'_seed_warm_lane must time out (124, lane-lock wait) while a '
            f'live merge-verify lease holds the SAME <lane_dir>.lock, got '
            f'rc={rc!r}'
        )
        assert not marker.exists(), (
            'seed-warm-lane.sh must never have run — reseed and the live '
            'verify must mutually exclude on the one <lane_dir>.lock'
        )


# ---------------------------------------------------------------------------
# task 2685, step-3 — reset_persistent_merge_worktree must ALSO hold
# <lane_dir>.lock across its own tree mutation, closing the residual window
# between reset_persistent_merge_worktree returning and merge_verify_lease()
# being acquired by the caller (merge_queue.py's post-reset gap). Without
# this, a reify/DF reseed racing in during that window could still clobber
# the tree between reset and the lease being taken.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestResetHoldsLaneLock:
    """reset_persistent_merge_worktree acquires <lane_dir>.lock across its
    tree-mutating body (task 2685, step-3).

    Mirrors test_git_ops.py::TestSeedWarmLaneTakesLaneLock.
    test_seed_blocks_until_lane_lock_released's block-then-release pattern:
    an external holder of <lane_dir>.lock must genuinely block the reset
    (not be ignored), and the reset must proceed once the lock is released.
    """

    async def test_reset_blocks_until_lane_lock_released(
        self, real_git_ops: GitOps, monkeypatch: pytest.MonkeyPatch,
    ):
        from orchestrator import git_ops as git_ops_mod  # noqa: PLC0415

        # Comfortably longer than the ~0.3s external hold below, but far
        # short of the real 30s default so a regression that never
        # acquires (and so never blocks at all) still fails fast rather
        # than via this test hanging for 30s.  The RESET path has its own
        # constant (task 3003), distinct from _seed_warm_lane's.
        monkeypatch.setattr(git_ops_mod, '_RESET_WARM_LANE_LOCK_WAIT_SECS', 5)

        merge_commit_a = await _get_merge_commit(real_git_ops, 'reset-lock-a1', 'rla1.py')
        await real_git_ops.reset_persistent_merge_worktree(merge_commit_a)  # create-once
        merge_commit_b = await _get_merge_commit(real_git_ops, 'reset-lock-a2', 'rla2.py')

        lock_path = lane_lock_path(real_git_ops.persistent_merge_worktree_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_held = True

        task = asyncio.create_task(
            real_git_ops.reset_persistent_merge_worktree(merge_commit_b)
        )
        try:
            await asyncio.sleep(0.3)

            assert not task.done(), (
                'reset_persistent_merge_worktree must block acquiring '
                '<lane_dir>.lock while an external holder (a reify/DF '
                'actor) holds it'
            )
            _, head_sha, _ = await _run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=real_git_ops.persistent_merge_worktree_path,
            )
            assert head_sha.strip() == merge_commit_a.strip(), (
                'HEAD must not have advanced yet — the reset is still '
                'waiting to acquire <lane_dir>.lock'
            )

            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
            lock_held = False

            warm_path = await asyncio.wait_for(task, 10)

            assert warm_path == real_git_ops.persistent_merge_worktree_path
            _, head_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=warm_path)
            assert head_sha.strip() == merge_commit_b.strip(), (
                'reset must proceed and advance HEAD once the lane lock '
                'is released'
            )
        finally:
            if not task.done():
                task.cancel()
            if lock_held:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)

    async def test_reset_raises_typed_contended_when_lane_lock_held_past_timeout(
        self, real_git_ops: GitOps, monkeypatch: pytest.MonkeyPatch,
    ):
        """Fail-CLOSED: when a foreign holder keeps <lane_dir>.lock held past
        the bounded wait, reset_persistent_merge_worktree must RAISE rather
        than mutate the tree unprotected — this is the central safety
        guarantee step-4 adds (a live reify/DF holder must block the mutation,
        never be silently ignored).

        task 3003 tightens the CLASSIFICATION of that raise: it must be the
        typed :class:`MergeVerifyLeaseContended`, not a bare ``RuntimeError``,
        so the merge worker's requeue seam DEFERS the dispatch instead of the
        generic ``except Exception`` mapping it to a deterministic-reason
        MergeOutcome('blocked') that feeds the anti-thrash ladder.

        NOTE the payload assertions are load-bearing:
        ``MergeVerifyLeaseContended`` IS-A ``RuntimeError``, so the old
        ``pytest.raises(RuntimeError, match='Timed out')`` would keep passing
        after the fix while pinning nothing about the new contract.

        The holder is a genuine ``flock(1)`` SUBPROCESS (task 3081): this test
        has always claimed "a foreign holder", but it used to stage one as a
        second fd in this same process — which at kernel level is precisely the
        B13 SELF-OWNED case, since ``/proc/locks`` reports our own pid.  Once
        that case is detected and typed separately, a same-process fd would no
        longer exercise foreign contention at all, and this test would be
        quietly asserting something other than what it says.
        """
        from orchestrator import git_ops as git_ops_mod  # noqa: PLC0415
        from orchestrator.git_ops import MergeVerifyLeaseContended  # noqa: PLC0415
        from test_lane_lock_leak_guard import (  # noqa: PLC0415
            foreign_lane_lock_holder,
        )

        # Small enough the test doesn't wait the real 30s default, but long
        # enough it can't be satisfied by anything other than a genuine
        # timeout of the bounded wait.  The RESET path has its own constant
        # (task 3003) — patching the seed's would no longer shorten this wait.
        monkeypatch.setattr(git_ops_mod, '_RESET_WARM_LANE_LOCK_WAIT_SECS', 1)

        merge_commit_a = await _get_merge_commit(real_git_ops, 'reset-lock-b1', 'rlb1.py')
        await real_git_ops.reset_persistent_merge_worktree(merge_commit_a)  # create-once
        merge_commit_b = await _get_merge_commit(real_git_ops, 'reset-lock-b2', 'rlb2.py')

        lock_path = lane_lock_path(real_git_ops.persistent_merge_worktree_path)
        with foreign_lane_lock_holder(lock_path):
            with pytest.raises(MergeVerifyLeaseContended) as excinfo:
                await real_git_ops.reset_persistent_merge_worktree(merge_commit_b)

            assert excinfo.value.lock_path == lock_path, (
                'the typed exception must carry the contended lane lock path '
                'so the operator log names the actual inode'
            )
            assert excinfo.value.wait_secs == 1, (
                'the typed exception must carry the RESET path\'s own bounded '
                'wait (the monkeypatched _RESET_WARM_LANE_LOCK_WAIT_SECS), not '
                'the seed constant or a hardcoded default'
            )
            # task 3003 amend (reviewer_comprehensive, error_message_accuracy):
            # the replaced bare RuntimeError named the tree it was protecting
            # ('refusing to mutate {warm_path} unprotected'); the typed raise
            # must keep that context rather than only the lock inode.
            assert (
                excinfo.value.protected_path
                == real_git_ops.persistent_merge_worktree_path
            ), (
                'the typed exception must name the warm worktree the refusal '
                'is protecting, which the lock path alone does not convey'
            )
            _msg = str(excinfo.value)
            assert str(real_git_ops.persistent_merge_worktree_path) in _msg, (
                f'the message must still name the protected tree; got {_msg!r}'
            )
            # …and it must state only what was OBSERVED, never the caller's
            # disposition: this exact raise site is ALSO reached by cli.py's
            # `verify-merge` (via acquire_host_verify_worktree), where it is a
            # TERMINAL bail — telling that operator the dispatch is being
            # 'deferred' and will be retried would be simply false.
            assert 'deferring' not in _msg.lower(), (
                f'the message must be caller-neutral — the merge worker defers '
                f'and logs so itself, but verify-merge exits; got {_msg!r}'
            )

            _, head_sha, _ = await _run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=real_git_ops.persistent_merge_worktree_path,
            )
            assert head_sha.strip() == merge_commit_a.strip(), (
                'a foreign holder that never releases <lane_dir>.lock must '
                'block the reset entirely — the tree must be left '
                'untouched, never mutated unprotected'
            )


# ---------------------------------------------------------------------------
# task 2873, step-1 — merge_verify_lease(lane_dir=X) must flock
# lane_lock_path(X) (the ACTUAL lane), leaving the hardcoded persistent lane
# FREE; the no-arg call still locks the persistent lane (backward compatible).
#
# Incident (reify, 2026-07-20, task 5310, merge_sha 83336a32): the per-land
# REMOTE-green cross-check re-verify (merge_queue.py, DF 2822) runs a full
# local trust-anchor verify in an EPHEMERAL ``_merge-<hash>`` worktree OUTSIDE
# any lane-lock lease, so a concurrent reseed/reclaim of that ephemeral lane
# clobbered the tree mid-compile → ENOENT storm → spurious DIVERGE that
# withheld a good land and quarantined the laptop remote. Parametrizing the
# lease (DF 2873) lets that cross-check flock the ACTUAL merge_wt lane —
# persistent OR ephemeral. Both tests below are RED before step-2's fix:
# merge_verify_lease takes no ``lane_dir`` yet (TypeError on the kwarg).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMergeVerifyLeaseParametrizedLane:
    """merge_verify_lease(lane_dir=X) flocks lane_lock_path(X), NOT the
    hardcoded persistent lane (task 2873)."""

    async def test_lease_locks_given_lane_and_leaves_persistent_free(
        self, tmp_path: Path,
    ):
        """(a) An EPHEMERAL lane_dir: the lease holds lane_lock_path(lane_dir)
        — a non-blocking flock re-acquire on that path raises BlockingIOError
        — WHILE lane_lock_path(persistent_merge_worktree_path) stays FREE
        (acquirable), proving it locked the GIVEN lane and not the hardcoded
        persistent lane. The GLOBAL holder-pgid rendezvous
        (read_lock_holder_pgid(worktree_base)) is still recorded, unchanged.
        """
        git_ops = _git_ops(tmp_path)
        # An ephemeral speculation lane (the incident's _merge-<hash> shape).
        ephemeral_wt = git_ops.worktree_base / '_merge-98c756bc'
        ephemeral_lock = lane_lock_path(ephemeral_wt)
        persistent_lock = lane_lock_path(git_ops.persistent_merge_worktree_path)
        ephemeral_lock.parent.mkdir(parents=True, exist_ok=True)
        persistent_lock.parent.mkdir(parents=True, exist_ok=True)
        assert ephemeral_lock != persistent_lock, (
            'sanity: the ephemeral lane lock must be a DIFFERENT inode than '
            'the persistent lane lock'
        )

        async with git_ops.merge_verify_lease(lane_dir=ephemeral_wt):
            # The GIVEN lane's lock is HELD — a non-blocking re-acquire fails.
            fd_eph = os.open(ephemeral_lock, os.O_RDWR | os.O_CREAT)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(fd_eph, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(fd_eph)

            # The hardcoded persistent lane lock is FREE — acquirable, proving
            # the lease did NOT touch it (it locked the given lane instead).
            fd_persist = os.open(persistent_lock, os.O_RDWR | os.O_CREAT)
            try:
                fcntl.flock(fd_persist, fcntl.LOCK_EX | fcntl.LOCK_NB)  # must NOT raise
                fcntl.flock(fd_persist, fcntl.LOCK_UN)
            finally:
                os.close(fd_persist)

            # The global holder-pgid rendezvous stays keyed to worktree_base.
            assert read_lock_holder_pgid(git_ops.worktree_base) == os.getpgrp(), (
                'the GLOBAL holder-pgid must still be recorded even for an '
                'ephemeral-lane lease'
            )

        # Released on exit — holder-pgid cleared, flock dropped.
        assert read_lock_holder_pgid(git_ops.worktree_base) is None

    async def test_default_lane_dir_still_locks_persistent(self, tmp_path: Path):
        """(b) BACKWARD-COMPAT: no lane_dir arg → holds lane_lock_path(
        persistent_merge_worktree_path), byte-identical to the sole existing
        caller (merge_queue.py:2113) and every existing lease test."""
        git_ops = _git_ops(tmp_path)
        persistent_lock = lane_lock_path(git_ops.persistent_merge_worktree_path)
        persistent_lock.parent.mkdir(parents=True, exist_ok=True)

        async with git_ops.merge_verify_lease():
            fd = os.open(persistent_lock, os.O_RDWR | os.O_CREAT)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(fd)
            assert read_lock_holder_pgid(git_ops.worktree_base) == os.getpgrp()
