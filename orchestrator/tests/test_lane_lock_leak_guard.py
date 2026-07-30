"""Lane-lock leak guard — D8 / B12 + B13 of the warm-lane repatriation PRD.

PRD: ``plans/warm-lane-infra-repatriation-prd.md`` §D8.
Incident anchor: reify ``esc-5548-5`` — a merge-verify lane lock left held by
an orphaned ``asyncio.to_thread`` acquire.  Kernel forensics at the time read
``/proc/locks`` and inode-matched ``FLOCK ADVISORY WRITE 588232 07:1d:4300647613``
against ``_merge-verify.lock``; three tasks then blocked behind one identical
``merge_outcome_signature`` and nothing surfaced until an unattended restart
roughly three hours later.

Two defects live here:

* **B12** — a cancelled lane-lock acquire could ORPHAN the fd.
  :func:`asyncio.to_thread` cannot interrupt its worker thread: once the outer
  ``await`` is cancelled, ``asyncio.futures._copy_future_state`` bails on
  ``dest.cancelled()`` and the thread's return value — the freshly acquired fd
  — is discarded, so ``release_merge_verify_flock`` never runs for it and the
  lane lock stays held until process exit.
* **B13** — nothing DETECTED that state.  A lane lock held by an fd in THIS
  process, with no live in-process span and no live verify, is a self-owned
  leak, not foreign contention, and must be reported as such.

This module owns the new coverage for both.  It reuses the real-git fixtures of
:mod:`test_merge_verify_lease_guard` wholesale (the suite's cross-module-helper
convention) and adds two shared helpers used by every class below:
:func:`foreign_lane_lock_holder` (a GENUINELY foreign holder — same-process fds
are indistinguishable from the leak at kernel level) and :func:`wait_until` (a
bounded async poll, so orphan-release callbacks settle deterministically
instead of behind a fixed sleep).
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

# Real-git fixtures + helpers reused from the sibling lease-guard module (the
# suite's established cross-module-helper convention, cf.
# test_verify_scope_kappa.py / test_coalesce_integration_gate.py).  `git_repo`
# and `real_git_ops` are pytest FIXTURES: importing them binds them as module
# attributes here, which is what makes them requestable by the tests below.
from test_merge_verify_lease_guard import (  # noqa: F401 -- fixtures used by name
    _get_merge_commit,
    _git_config,
    _git_ops,
    _setup_repo,
    git_repo,
    real_git_ops,
)

from orchestrator.git_ops import GitOps, MergeVerifyLeaseContended, _run
from orchestrator.verify_cancel import (
    acquire_merge_verify_flock,
    lane_lock_holder_pids,
    lane_lock_path,
    read_lock_holder_pgid,
    release_merge_verify_flock,
    remove_lock_holder_pgid,
    write_lock_holder_pgid,
)

__all__ = [
    '_get_merge_commit',
    '_git_config',
    '_git_ops',
    '_setup_repo',
    'foreign_lane_lock_holder',
    'git_repo',
    'lane_is_free',
    'lane_lock_path',
    'leaked_lane_lock',
    'real_git_ops',
    'wait_until',
]


#: How long the ``flock(1)`` child holds the lane lock before exiting on its
#: own.  Long enough that no test races it; the contextmanager terminates the
#: child on exit regardless, so this is only a leak backstop.
_FOREIGN_HOLDER_HOLD_SECS = 120

#: Bound on how long :func:`foreign_lane_lock_holder` waits for the child to
#: actually own the lock before failing the test.
_FOREIGN_HOLDER_STARTUP_SECS = 5.0

#: Bound on how long :func:`foreign_lane_lock_holder` waits, on exit, for the
#: kernel to stop attributing the lock to the child.
_FOREIGN_HOLDER_TEARDOWN_SECS = 5.0


@contextlib.contextmanager
def foreign_lane_lock_holder(lock_path: Path) -> Iterator[subprocess.Popen]:
    """Hold *lock_path* from a GENUINELY foreign process, yielding the child.

    Spawns ``flock -x <lock_path> sleep N`` — util-linux ``flock(1)``, already
    exercised by this suite via :meth:`GitOps._seed_warm_lane`, and
    interoperable with ``fcntl.flock(2)`` on the same inode.

    Why a subprocess and not a second fd in this process: at KERNEL level a
    same-process fd is precisely the B13 self-owned case (``/proc/locks``
    reports our own pid as the holder), so it cannot stand in for foreign
    contention once the leak detector exists.  Any test that means "somebody
    else holds this lane" must use this helper.

    Blocks until the child provably owns the lock (a zero-timeout probe
    acquire returns ``None``), failing the test if that has not happened
    within :data:`_FOREIGN_HOLDER_STARTUP_SECS`.

    Teardown kills the child's whole PROCESS GROUP, not just the child.
    Verified empirically: util-linux ``flock(1)`` FORKS the command rather than
    exec'ing it, so ``sleep`` inherits the locked fd; ``child.terminate()``
    alone reaps only the ``flock`` parent and the grandchild keeps the open file
    description — and therefore the flock — alive, with ``/proc/locks`` still
    naming the now-dead ``flock`` pid.  Hence ``start_new_session=True`` (the
    child leads its own group, which also makes it foreign in the strongest
    sense) plus :func:`os.killpg`, and a bounded poll proving the kernel has
    actually dropped the attribution.  Without this, every test that RELEASES
    the foreign holder to hand the lane over would silently be testing a lane
    that is still locked.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    child = subprocess.Popen(
        ['flock', '-x', str(lock_path), 'sleep', str(_FOREIGN_HOLDER_HOLD_SECS)],
        start_new_session=True,
    )

    def _kill_group() -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(child.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            child.wait(timeout=5)

    clean_exit = False
    try:
        deadline = time.monotonic() + _FOREIGN_HOLDER_STARTUP_SECS
        while True:
            probe = acquire_merge_verify_flock(lock_path, 0.0)
            if probe is None:
                break  # the child owns it — contention is observable
            release_merge_verify_flock(probe)
            if time.monotonic() >= deadline:
                _kill_group()
                pytest.fail(
                    f'flock(1) child never took {lock_path} within '
                    f'{_FOREIGN_HOLDER_STARTUP_SECS}s — the foreign-holder '
                    f'staging is broken, so any assertion built on it would '
                    f'be vacuous'
                )
            time.sleep(0.02)
        yield child
        clean_exit = True
    finally:
        _kill_group()
        released = False
        teardown_deadline = time.monotonic() + _FOREIGN_HOLDER_TEARDOWN_SECS
        while True:
            if child.pid not in lane_lock_holder_pids(lock_path):
                released = True
                break
            if time.monotonic() >= teardown_deadline:
                break
            time.sleep(0.02)
        # Only report this when the body itself succeeded, so a genuine test
        # failure is never masked by its own cleanup.
        if clean_exit and not released:
            pytest.fail(
                f'the foreign holder still owns {lock_path} after killing its '
                f'process group — a test that hands the lane over by releasing '
                f'it would be asserting against a still-locked lane'
            )


@contextlib.contextmanager
def leaked_lane_lock(lock_path: Path) -> Iterator[int]:
    """Stage the B13 fault: hold *lock_path* from an fd nothing will release.

    Acquires through :func:`acquire_merge_verify_flock` DIRECTLY rather than
    through ``GitOps._acquire_lane_flock_off_thread``, which is what makes this
    the leak shape rather than a legitimate hold: the fd never enters the
    in-process held-fd registry, and no holder-pgid rendezvous is written —
    exactly the state an orphaned ``asyncio.to_thread`` acquire leaves behind.

    The real orphan's fd is unreachable and stays held until process exit; this
    helper keeps a handle purely so the test process does not accumulate held
    locks.  Nothing the detector can observe differs.
    """
    fd = acquire_merge_verify_flock(lock_path, 0.0)
    if fd is None:
        pytest.fail(
            f'could not take {lock_path} to stage the leak — something else '
            f'already holds it, so the assertion below would test the wrong fault'
        )
    try:
        yield fd
    finally:
        release_merge_verify_flock(fd)


async def wait_until(
    predicate: Callable[[], bool],
    timeout: float = 5.0,
    interval: float = 0.05,
) -> bool:
    """Poll *predicate* until true, returning whether it became true in time.

    Returns ``True`` as soon as *predicate* holds, ``False`` once *timeout*
    elapses — deliberately NOT raising, so a caller can assert on the result
    with its own message.  Used to settle orphan-release callbacks (which run
    on the event loop, after the cancelling coroutine has already resumed)
    without a fixed sleep that would either flake or waste wall clock.
    """
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(interval)


def lane_is_free(lock_path: Path) -> bool:
    """Whether *lock_path* is unheld AND unattributed to this process.

    The PRD's B12 signal verbatim, and BOTH halves are load-bearing: a
    successful fresh acquire alone could be succeeding against a different
    inode than the one we leaked, and ``/proc/locks`` alone is precisely the
    manual forensics reify ``esc-5548-5`` had to do by hand.

    Note the ordering — the kernel attribution is checked BEFORE the probe
    acquire, because the probe itself briefly becomes a holder and would
    otherwise report on itself.
    """
    if os.getpid() in lane_lock_holder_pids(lock_path):
        return False
    probe = acquire_merge_verify_flock(lock_path, 0.0)
    if probe is None:
        return False
    release_merge_verify_flock(probe)
    return True


def test_scaffold_helpers_are_importable(tmp_path: Path):
    """Smoke pin: the shared helpers exist and the foreign holder really holds.

    Guards the scaffold itself — every class in this module builds on
    ``foreign_lane_lock_holder`` observably owning the lock, so a broken helper
    would silently make downstream assertions vacuous rather than red.
    """
    lock_path = lane_lock_path(tmp_path / 'lane')
    with foreign_lane_lock_holder(lock_path):
        assert acquire_merge_verify_flock(lock_path, 0.0) is None
    assert asyncio.run(wait_until(lambda: True, timeout=0.1)) is True


@pytest.mark.asyncio
async def test_scaffold_real_git_fixtures_available(real_git_ops):
    """Smoke pin: the imported real-git fixtures resolve in THIS module.

    Cross-module fixture imports are load-bearing here (they are what makes
    ``real_git_ops`` requestable); pinning them once means a future refactor of
    the sibling module fails loudly here instead of erroring one test at a time.
    """
    commit = await _get_merge_commit(real_git_ops, 'scaffold-lane', 'scaffold.py')
    assert commit


# ---------------------------------------------------------------------------
# Step-3 (B13): a lane lock held by an fd in THIS process, with no live
# in-process span and no live verify, is a SELF-OWNED LEAK — not the foreign
# contention the fail-closed timeout path exists for.
#
# The three layers of the predicate are pinned independently below, because
# each one alone would produce a false positive:
#   (1) kernel — our pid among the FLOCK holders of the lock's inode;
#   (2) registry — no fd registered for that path by an in-process acquire;
#   (3) liveness — _merge_verify_lease_active() False.
# Layer (3) alone cannot carry it: task_verify_lease deliberately never writes
# the rendezvous, so a legitimate live consumer-hold would be libelled a leak —
# and a leak report is a LOUD, human-escalating event.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSelfOwnedLaneLockLeak:
    """B13 — self-owned lane-lock leaks are detected and reported as such."""

    async def test_reset_raises_self_owned_leak_naming_pid_pgid_and_lock(
        self, real_git_ops, monkeypatch: pytest.MonkeyPatch,
    ):
        """CANONICAL B13: the incident's own path, reproduced end to end.

        A leaked lane lock makes the reset's bounded wait time out exactly as
        foreign contention would.  Today that is indistinguishable from "some
        other actor is busy" — which is why reify esc-5548-5 took roughly three
        hours and manual ``/proc/locks`` + ``stat -c %i`` forensics to
        attribute.  The fault must name the culprit itself: our pid, our pgid,
        and the lock inode.

        The tree must still be left untouched — detection changes the
        DIAGNOSIS, never the fail-closed refusal to mutate unprotected.
        """
        from orchestrator import git_ops as git_ops_mod  # noqa: PLC0415
        from orchestrator.git_ops import LaneLockSelfOwnedLeak  # noqa: PLC0415

        monkeypatch.setattr(git_ops_mod, '_RESET_WARM_LANE_LOCK_WAIT_SECS', 1)

        commit_a = await _get_merge_commit(real_git_ops, 'leak-a', 'leak_a.py')
        await real_git_ops.reset_persistent_merge_worktree(commit_a)  # create-once
        commit_b = await _get_merge_commit(real_git_ops, 'leak-b', 'leak_b.py')

        warm_path = real_git_ops.persistent_merge_worktree_path
        lock_path = lane_lock_path(warm_path)

        with leaked_lane_lock(lock_path):
            with pytest.raises(LaneLockSelfOwnedLeak) as excinfo:
                await real_git_ops.reset_persistent_merge_worktree(commit_b)

            msg = str(excinfo.value)
            assert str(os.getpid()) in msg, (
                f'the fault must name the leaking pid — OUR pid — so an '
                f'operator is not left doing the incident\'s manual '
                f'/proc/locks forensics again; got {msg!r}'
            )
            assert str(os.getpgrp()) in msg, (
                f'the fault must name our pgid, the corroborating fact the '
                f'holder-pgid rendezvous would have carried had the acquire '
                f'ever completed; got {msg!r}'
            )
            assert str(lock_path) in msg, (
                f'the fault must name the leaked lock inode; got {msg!r}'
            )

            _, head_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=warm_path)
            assert head_sha.strip() == commit_a.strip(), (
                'detecting the leak must not weaken the fail-closed refusal: '
                'the tree is still never mutated unprotected'
            )

    async def test_leak_type_and_payload_stay_contended_compatible(
        self, real_git_ops, monkeypatch: pytest.MonkeyPatch,
    ):
        """The leak IS-A MergeVerifyLeaseContended, payload and message included.

        Load-bearing, not taxonomy.  Both merge-worker consumers
        (merge_queue.py's cross-check fail-safe and its bounded contended-defer
        arm) are isinstance-based on the parent, so a standalone type would fall
        through to the generic ``except Exception`` → ``MergeOutcome('blocked')``
        with a deterministic reason string — an identical
        ``merge_outcome_signature`` every attempt, tripping the
        ``consecutive_merge_thrash`` ladder into precisely the false-positive
        escalation DF 3003 was chartered to stop.

        The parent's message contract is asserted verbatim (names the protected
        tree, never says "deferring") because DF 3003's own guard test pins
        exactly those two properties — a subclass that broke either would break
        that test from a distance.
        """
        from orchestrator import git_ops as git_ops_mod  # noqa: PLC0415
        from orchestrator.git_ops import LaneLockSelfOwnedLeak  # noqa: PLC0415

        assert issubclass(LaneLockSelfOwnedLeak, MergeVerifyLeaseContended), (
            'a standalone type would miss both isinstance-based merge-worker '
            'handlers and be mapped to a deterministic-reason blocked outcome'
        )

        monkeypatch.setattr(git_ops_mod, '_RESET_WARM_LANE_LOCK_WAIT_SECS', 1)

        commit_a = await _get_merge_commit(real_git_ops, 'leak-p1', 'leak_p1.py')
        await real_git_ops.reset_persistent_merge_worktree(commit_a)
        commit_b = await _get_merge_commit(real_git_ops, 'leak-p2', 'leak_p2.py')

        warm_path = real_git_ops.persistent_merge_worktree_path
        lock_path = lane_lock_path(warm_path)

        with leaked_lane_lock(lock_path), pytest.raises(LaneLockSelfOwnedLeak) as excinfo:
            await real_git_ops.reset_persistent_merge_worktree(commit_b)

        exc = excinfo.value
        # The parent's full payload, forwarded — every existing assertion on a
        # MergeVerifyLeaseContended keeps holding for this subclass.
        assert exc.lock_path == lock_path
        assert exc.wait_secs == 1
        assert exc.operation == 'the warm merge-worktree reset'
        assert exc.protected_path == warm_path
        # …plus the leak facts the parent has no room for.
        assert os.getpid() in exc.holder_pids
        assert exc.self_pid == os.getpid()
        assert exc.self_pgid == os.getpgrp()
        # …and the parent's message contract, verbatim (cf. DF 3003's guard).
        _msg = str(exc)
        assert str(exc.protected_path) in _msg, (
            f'the subclass must keep naming the protected tree; got {_msg!r}'
        )
        assert 'deferring' not in _msg.lower(), (
            f'caller-neutrality is inherited: this raise also reaches cli.py '
            f'verify-merge, where it is a TERMINAL bail; got {_msg!r}'
        )

    async def test_merge_verify_lease_reports_leak_and_records_no_holder_pgid(
        self, real_git_ops, monkeypatch: pytest.MonkeyPatch,
    ):
        """The lease acquire detects the same fault on the same inode.

        Both leases and the reset contend for ONE lock; a leak is a property of
        that inode, not of whichever caller happened to notice.  The rendezvous
        must stay unwritten: the lease never took the lock, so claiming holder
        status would corrupt the very liveness signal layer (3) reads.
        """
        from orchestrator import git_ops as git_ops_mod  # noqa: PLC0415
        from orchestrator.git_ops import LaneLockSelfOwnedLeak  # noqa: PLC0415

        monkeypatch.setattr(git_ops_mod, '_MERGE_VERIFY_LEASE_WAIT_SECS', 1)

        lock_path = lane_lock_path(real_git_ops.persistent_merge_worktree_path)

        with leaked_lane_lock(lock_path), pytest.raises(LaneLockSelfOwnedLeak):
            async with real_git_ops.merge_verify_lease():
                pytest.fail(
                    'the lease body must never run while the lane lock is '
                    'leaked — that is the unprotected 1-2h verify window '
                    'task 2828 closed'
                )

        assert read_lock_holder_pgid(real_git_ops.worktree_base) is None, (
            'a lease that never acquired must not record itself as the holder'
        )

    async def test_foreign_holder_is_contention_not_a_leak(
        self, real_git_ops, monkeypatch: pytest.MonkeyPatch,
    ):
        """NEGATIVE (1): another process holding the lane is not our leak.

        The discriminator is WHOSE pid the kernel reports.  A reify
        ``flock(1)``, a ``verify-merge`` CLI subprocess, or another orchestrator
        must all stay on today's fail-closed contention path — that is DF 3003's
        territory, and misrouting it into a loud leak escalation would be a
        false alarm on entirely healthy contention.

        ``type(...) is`` and not ``isinstance``: the subclass IS-A the parent, so
        an isinstance check here would pass even if every foreign contention
        were misreported as a leak.
        """
        from orchestrator import git_ops as git_ops_mod  # noqa: PLC0415
        from orchestrator.git_ops import LaneLockSelfOwnedLeak  # noqa: PLC0415

        monkeypatch.setattr(git_ops_mod, '_RESET_WARM_LANE_LOCK_WAIT_SECS', 1)

        commit_a = await _get_merge_commit(real_git_ops, 'leak-f1', 'leak_f1.py')
        await real_git_ops.reset_persistent_merge_worktree(commit_a)
        commit_b = await _get_merge_commit(real_git_ops, 'leak-f2', 'leak_f2.py')

        lock_path = lane_lock_path(real_git_ops.persistent_merge_worktree_path)

        with foreign_lane_lock_holder(lock_path) as child:
            holders = lane_lock_holder_pids(lock_path)
            assert os.getpid() not in holders, (
                'staging error: this process must NOT hold the lock, or the '
                'negative below would be vacuous'
            )
            assert child.pid in holders

            with pytest.raises(MergeVerifyLeaseContended) as excinfo:
                await real_git_ops.reset_persistent_merge_worktree(commit_b)

            assert type(excinfo.value) is not LaneLockSelfOwnedLeak
            assert type(excinfo.value) is MergeVerifyLeaseContended
            assert str(child.pid) in str(excinfo.value), (
                f'the timeout must name the kernel-reported holder — the '
                f'incident needed manual /proc/locks + stat forensics to learn '
                f'exactly this; got {str(excinfo.value)!r}'
            )

    async def test_live_in_process_span_is_not_a_leak(self, real_git_ops):
        """NEGATIVE (2): a registered in-process hold is a live span, not a leak.

        Layer (2) of the predicate.  Without it, every legitimate concurrent
        holder in this process — most sharply ``task_verify_lease``, which by
        design never writes the rendezvous layer (3) reads — would be reported
        as a leak.  The A/B is the point: the SAME kernel state, differing only
        in whether the acquire went through the registered seam.
        """
        lock_path = lane_lock_path(real_git_ops.persistent_merge_worktree_path)

        fd = await GitOps._acquire_lane_flock_off_thread(lock_path, 1.0)
        assert fd is not None
        try:
            assert os.getpid() in lane_lock_holder_pids(lock_path), (
                'staging error: the kernel must see us as a holder, or layer '
                '(1) would be what returns None here'
            )
            assert real_git_ops._lane_lock_self_owned_leak(lock_path, 1.0) is None, (
                'an fd taken through the registered acquire seam is a LIVE '
                'span — reporting it as leaked would escalate a healthy hold'
            )
        finally:
            GitOps._release_lane_flock(fd)

        with leaked_lane_lock(lock_path):
            assert real_git_ops._lane_lock_self_owned_leak(lock_path, 1.0) is not None, (
                'identical kernel state, unregistered fd: the registry must be '
                'what discriminates, not something incidental to the A case'
            )

    async def test_live_verify_lease_is_not_a_leak(self, real_git_ops):
        """NEGATIVE (3): a live recorded verify is contention, not a leak.

        Layer (3), and DF 3003's own case: a genuine long verify holding the
        lane past the bounded wait must keep deferring quietly rather than
        raising a human-escalating fault every attempt.
        """
        lock_path = lane_lock_path(real_git_ops.persistent_merge_worktree_path)

        with leaked_lane_lock(lock_path):
            assert real_git_ops._lane_lock_self_owned_leak(lock_path, 1.0) is not None

            write_lock_holder_pgid(real_git_ops.worktree_base, os.getpgrp())
            try:
                assert real_git_ops._merge_verify_lease_active() is True, (
                    'staging error: our own pgid is live, so the rendezvous '
                    'must read as an active lease'
                )
                assert real_git_ops._lane_lock_self_owned_leak(lock_path, 1.0) is None
            finally:
                remove_lock_holder_pgid(real_git_ops.worktree_base)


# ---------------------------------------------------------------------------
# Step-7 (B12): a CANCELLED acquire must never orphan the lane lock.
#
# The fault is structural, not incidental: `asyncio.to_thread` cannot interrupt
# its worker thread, so cancelling the outer await does NOT stop the acquire —
# it only stops anyone from ever seeing the fd it wins.  Concretely,
# `asyncio.futures._copy_future_state` bails when `dest.cancelled()`, so the
# thread's return value is dropped on the floor and `release_merge_verify_flock`
# is never called for it.  The lane then stays locked until process exit, which
# is exactly the state reify esc-5548-5 found by hand in `/proc/locks`.
#
# The fix (step-8) must therefore not "avoid" the late win but TAKE OWNERSHIP of
# it: the acquire's inner future is shielded so it survives the cancellation,
# and a done-callback releases whatever it eventually returns.
# ---------------------------------------------------------------------------


#: Sentinel fd for the deterministic unit cases.  Deliberately NOT a real fd:
#: these cases stub both flock primitives, so nothing may actually touch it, and
#: an accidental real syscall on it would fail loudly rather than silently
#: operating on some unrelated open file.
_SENTINEL_FD = 4242


@pytest.mark.asyncio
class TestCancelledAcquireNeverOrphansTheLaneLock:
    """B12 — the shared acquire seam (`GitOps._acquire_lane_flock_off_thread`).

    Pinned on the SHARED seam rather than on either lease, because that is what
    makes the guarantee indivisible: `merge_verify_lease`, `task_verify_lease`
    and (from step-10) `reset_persistent_merge_worktree` all acquire through
    this one method, so a fix proven here cannot be true of one caller and false
    of another.
    """

    @staticmethod
    def _stub_acquire(
        monkeypatch: pytest.MonkeyPatch,
        gate: threading.Event,
        result: int | None,
    ) -> dict[str, object]:
        """Replace the flock primitives with a gated stub + release recorder.

        Returns a dict of observations: ``running`` (set once the stub is
        executing off-thread), ``returned`` (set once it has produced *result*),
        ``thread_id`` (the thread it ran on) and ``released`` (every fd passed
        to :func:`release_merge_verify_flock`).

        Stubbing is what makes these cases DETERMINISTIC: the cancellation lands
        while the acquire is provably mid-flight, with no dependence on kernel
        lock timing, and the late win happens exactly when the test says so.
        """
        from orchestrator import git_ops as git_ops_mod  # noqa: PLC0415

        obs: dict[str, object] = {
            'running': threading.Event(),
            'returned': threading.Event(),
            'thread_id': None,
            'released': [],
        }

        def _gated_acquire(*_a, **_k):
            obs['thread_id'] = threading.get_ident()
            obs['running'].set()  # type: ignore[union-attr]
            gate.wait(timeout=10)
            obs['returned'].set()  # type: ignore[union-attr]
            return result

        def _record_release(fd) -> None:
            obs['released'].append(fd)  # type: ignore[union-attr]

        monkeypatch.setattr(
            git_ops_mod, 'acquire_merge_verify_flock', _gated_acquire,
        )
        monkeypatch.setattr(
            git_ops_mod, 'release_merge_verify_flock', _record_release,
        )
        return obs

    async def test_late_won_fd_is_released_after_cancellation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """CANONICAL B12: an fd won AFTER cancellation is still released.

        The worker thread is uninterruptible by design, so the only sound
        contract is that ownership of a late win TRANSFERS to the canceller's
        cleanup path.  Today there is no such path: the fd is won, dropped, and
        the lane stays locked for the life of the process.

        The gate is released only AFTER the cancellation has been observed, so
        this is unambiguously the post-cancellation win and not a race the test
        happened to lose.
        """
        gate = threading.Event()
        obs = self._stub_acquire(monkeypatch, gate, _SENTINEL_FD)
        lock_path = lane_lock_path(tmp_path / 'lane')

        task = asyncio.create_task(
            GitOps._acquire_lane_flock_off_thread(lock_path, 5.0)
        )
        assert await wait_until(obs['running'].is_set), (  # type: ignore[union-attr]
            'the acquire never reached the worker thread — cancelling now '
            'would prove nothing about a late win'
        )

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not obs['released'], (
            'nothing may be released before the acquire has produced anything'
        )

        gate.set()  # the acquire wins the lock AFTER its awaiter is gone
        assert await wait_until(lambda: _SENTINEL_FD in obs['released']), (
            f'the late-won fd was ORPHANED: a cancelled acquire dropped a held '
            f'lane lock (reify esc-5548-5 — held until process exit, found only '
            f'by hand in /proc/locks); released={obs["released"]!r}'
        )

    async def test_cancelled_acquire_that_times_out_releases_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """The genuine-timeout-after-cancellation case is a silent no-op.

        A ``None`` return means the bounded wait expired with NOTHING held, so
        the orphan path must neither release nor raise.  Unguarded, it would
        call ``fcntl.flock(None, ...)`` and surface a ``TypeError`` through the
        loop's exception handler — noise attached to a cancellation that is
        entirely normal (`_release_lane_flock`'s existing ``None`` guard is what
        this reuses).
        """
        loop = asyncio.get_running_loop()
        caught: list[dict] = []
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, ctx: caught.append(ctx))
        try:
            gate = threading.Event()
            obs = self._stub_acquire(monkeypatch, gate, None)
            lock_path = lane_lock_path(tmp_path / 'lane')

            task = asyncio.create_task(
                GitOps._acquire_lane_flock_off_thread(lock_path, 5.0)
            )
            assert await wait_until(obs['running'].is_set)  # type: ignore[union-attr]

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            gate.set()
            assert await wait_until(obs['returned'].is_set)  # type: ignore[union-attr]
            # Let the done-callback run to completion on the loop.
            for _ in range(20):
                await asyncio.sleep(0.01)

            assert obs['released'] == [], (
                f'a timed-out acquire holds nothing — releasing anything here '
                f'would be releasing somebody else\'s fd number; '
                f'released={obs["released"]!r}'
            )
            assert caught == [], (
                f'the orphan path must stay silent on a plain timeout, not '
                f'raise through the loop exception handler; got {caught!r}'
            )
        finally:
            loop.set_exception_handler(previous)

    async def test_uncancelled_acquire_keeps_its_fd_and_registers_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """NON-REGRESSION: the normal path is untouched by the orphan guard.

        The failure mode being excluded is a fix that releases the fd on EVERY
        completion rather than only the cancelled one — which would hand the
        caller an fd whose lock had already been dropped, silently reopening the
        unprotected-verify window task 2828 closed.  The registry entry is
        asserted in the same breath because B13's layer (2) reads it: an
        unregistered legitimate hold is libelled a leak.
        """
        from orchestrator.git_ops import (  # noqa: PLC0415
            _lane_lock_held_in_process,
        )

        gate = threading.Event()
        gate.set()  # no cancellation here: let the acquire return immediately
        obs = self._stub_acquire(monkeypatch, gate, _SENTINEL_FD)
        lock_path = lane_lock_path(tmp_path / 'lane')

        fd = await GitOps._acquire_lane_flock_off_thread(lock_path, 5.0)
        try:
            assert fd == _SENTINEL_FD, 'the won fd must reach its caller'
            assert _lane_lock_held_in_process(lock_path) is True, (
                'a legitimate hold must be registered, or B13 layer (2) will '
                'report this very fd as a leak'
            )
            for _ in range(10):
                await asyncio.sleep(0.01)
            assert obs['released'] == [], (
                f'the acquire released the fd behind its caller\'s back — the '
                f'caller would then run its span unprotected; '
                f'released={obs["released"]!r}'
            )
        finally:
            GitOps._release_lane_flock(fd)

        assert obs['released'] == [_SENTINEL_FD]
        assert _lane_lock_held_in_process(lock_path) is False, (
            'the registry must be symmetric — a stale entry would mask a real '
            'leak on this inode forever after'
        )

    async def test_acquire_still_runs_off_the_event_loop_thread(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """The cancellation guard must not drag the acquire back on-loop.

        `acquire_merge_verify_flock` is a synchronous ``time.sleep`` poll whose
        bounded wait is minutes; running it inline would freeze the entire
        orchestrator.  Pinned HERE as well as at
        ``test_merge_verify_lease_guard.py:135`` because step-8 restructures the
        very statement that property depends on, and step-10 routes a third
        caller through it.
        """
        gate = threading.Event()
        gate.set()
        obs = self._stub_acquire(monkeypatch, gate, _SENTINEL_FD)
        lock_path = lane_lock_path(tmp_path / 'lane')

        fd = await GitOps._acquire_lane_flock_off_thread(lock_path, 5.0)
        try:
            assert obs['thread_id'] is not None, 'acquire was never invoked'
            assert obs['thread_id'] != threading.get_ident(), (
                'the bounded-wait acquire ran ON the event-loop thread — a '
                'minutes-long synchronous poll there freezes the orchestrator'
            )
        finally:
            GitOps._release_lane_flock(fd)

    async def test_cancelled_merge_verify_lease_leaves_the_lane_free(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """END-TO-END with REAL flocks, through `merge_verify_lease`.

        The PRD's B12 signal verbatim: after a cancelled lease whose acquire won
        the lock late, the lane must be re-acquirable AND the kernel must no
        longer attribute the lock to this process.  Both halves matter — a fresh
        acquire alone could succeed against a stale fd on a *different* inode,
        and `/proc/locks` alone is exactly the manual forensics the incident had
        to do by hand.

        No stubs on the flock primitives: the win is a genuine kernel
        acquisition, handed over by releasing a genuinely foreign holder after
        the cancellation.
        """
        from orchestrator import git_ops as git_ops_mod  # noqa: PLC0415

        # Bounded well above the ~0.1s poll that actually elapses, so the
        # acquire cannot reach its deadline and return None — which would make
        # the assertions below vacuously true.
        monkeypatch.setattr(git_ops_mod, '_MERGE_VERIFY_LEASE_WAIT_SECS', 30.0)

        git_ops = _git_ops(tmp_path)
        lock_path = lane_lock_path(git_ops.persistent_merge_worktree_path)

        polling = threading.Event()
        outcome: dict[str, int | None] = {}
        real_acquire = git_ops_mod.acquire_merge_verify_flock

        def _watched_acquire(*args, **kwargs):
            polling.set()
            fd = real_acquire(*args, **kwargs)
            outcome['fd'] = fd
            return fd

        monkeypatch.setattr(
            git_ops_mod, 'acquire_merge_verify_flock', _watched_acquire,
        )

        async def _lease() -> None:
            async with git_ops.merge_verify_lease():
                pytest.fail('the lease body must never run: the lane is held')

        with foreign_lane_lock_holder(lock_path):
            task = asyncio.create_task(_lease())
            assert await wait_until(polling.is_set), (
                'the lease never entered its bounded-wait acquire'
            )
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        # The foreign holder is gone: the orphaned worker thread now wins.

        # Ordered BEFORE any probe of our own: a probe fired while the worker is
        # still polling would take the freed lock itself, report "free", and
        # pass this test without the orphan ever having won anything.  Waiting
        # for the worker's own return value removes that race AND proves the
        # premise — a timed-out (`None`) acquire holds nothing, so everything
        # below it would be vacuously true.
        assert await wait_until(lambda: 'fd' in outcome, timeout=10.0), (
            'the orphaned acquire never returned — the lane handover never '
            'happened, so this test would prove nothing about a late win'
        )
        assert outcome['fd'] is not None, (
            'the acquire hit its bounded-wait deadline instead of winning the '
            'freed lock: nothing was ever held, so there is no orphan to detect'
        )

        assert await wait_until(lambda: lane_is_free(lock_path), timeout=10.0), (
            f'the cancelled lease left the lane lock HELD by this process — '
            f'the exact esc-5548-5 state: every later merge-verify on this lane '
            f'blocks behind an fd no code path can reach, until process exit. '
            f'holders={lane_lock_holder_pids(lock_path)!r}, ours={os.getpid()}'
        )
        assert read_lock_holder_pgid(git_ops.worktree_base) is None, (
            'a lease cancelled before it acquired must record no holder-pgid'
        )


# ---------------------------------------------------------------------------
# Step-9 (B12): the RESET path — the incident's own caller.
#
# `reset_persistent_merge_worktree` owns a SECOND, duplicated bare
# `asyncio.to_thread(acquire_merge_verify_flock, ...)`, so the guarantee proved
# on the shared seam above simply does not reach it.  This class asserts the
# behaviour end to end AND pins the structure that makes it hold — a duplicate
# acquire site is exactly how the fix silently regresses, so "the reset acquires
# through the one guarded seam" is itself the invariant worth a test.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCancelledResetNeverOrphansTheLaneLock:
    """B12 on `reset_persistent_merge_worktree` — reify esc-5548-5's own path."""

    @staticmethod
    def _watch_acquire(
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[threading.Event, dict[str, object]]:
        """Observe the REAL bounded-wait acquire without replacing it.

        Returns ``(polling, outcome)``: *polling* is set the moment the acquire
        begins on its worker thread, and *outcome* collects ``fd`` (its return
        value) and ``thread_id``.  Pass-through, not a stub — the lock handover
        below has to be a genuine kernel acquisition or it proves nothing.
        """
        from orchestrator import git_ops as git_ops_mod  # noqa: PLC0415

        polling = threading.Event()
        outcome: dict[str, object] = {}
        real_acquire = git_ops_mod.acquire_merge_verify_flock

        def _watched(*args, **kwargs):
            outcome['thread_id'] = threading.get_ident()
            polling.set()
            fd = real_acquire(*args, **kwargs)
            outcome['fd'] = fd
            return fd

        monkeypatch.setattr(
            git_ops_mod, 'acquire_merge_verify_flock', _watched,
        )
        return polling, outcome

    async def test_cancelled_reset_leaves_the_lane_free_and_the_tree_untouched(
        self, real_git_ops, monkeypatch: pytest.MonkeyPatch,
    ):
        """CANONICAL B12 on the reset: cancel mid-acquire, lane still released.

        This is the incident verbatim.  A merge dispatch is cancelled (task
        timeout, shutdown, requeue) while the reset is polling for a lane lock
        reify's own tooling holds; the poll wins moments later, and the fd goes
        nowhere.  In esc-5548-5 the lane then stayed locked for roughly three
        hours, blocking three tasks behind one `merge_outcome_signature`, until
        an unattended restart cleared it.

        The tree assertion is the other half: a cancelled reset must not have
        mutated the worktree on its way out.
        """
        from orchestrator import git_ops as git_ops_mod  # noqa: PLC0415

        # Long enough that the acquire is provably STILL POLLING when the
        # cancellation lands — the whole point is a win that arrives after its
        # awaiter is gone, not a wait that expires first.
        monkeypatch.setattr(git_ops_mod, '_RESET_WARM_LANE_LOCK_WAIT_SECS', 10)

        commit_a = await _get_merge_commit(real_git_ops, 'b12-a', 'b12_a.py')
        await real_git_ops.reset_persistent_merge_worktree(commit_a)  # create-once
        commit_b = await _get_merge_commit(real_git_ops, 'b12-b', 'b12_b.py')

        warm_path = real_git_ops.persistent_merge_worktree_path
        lock_path = lane_lock_path(warm_path)
        polling, outcome = self._watch_acquire(monkeypatch)

        with foreign_lane_lock_holder(lock_path):
            task = asyncio.create_task(
                real_git_ops.reset_persistent_merge_worktree(commit_b)
            )
            assert await wait_until(polling.is_set), (
                'the reset never entered its bounded-wait acquire — cancelling '
                'now would prove nothing about a late win'
            )
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        # The foreign holder is gone; the orphaned worker thread now wins.

        # Ordered before any probe of ours, which would otherwise take the freed
        # lock itself and report "free" without the orphan ever winning.
        assert await wait_until(lambda: 'fd' in outcome, timeout=10.0), (
            'the orphaned acquire never returned — no handover happened'
        )
        assert outcome['fd'] is not None, (
            'the acquire hit its deadline instead of winning the freed lock: '
            'nothing was held, so there is no orphan to detect here'
        )

        assert await wait_until(lambda: lane_is_free(lock_path), timeout=10.0), (
            f'the cancelled reset left the lane lock HELD by this process — '
            f'esc-5548-5 exactly: every later merge-verify and reset on this '
            f'lane blocks behind an fd no code path can reach, until process '
            f'exit. holders={lane_lock_holder_pids(lock_path)!r}, '
            f'ours={os.getpid()}'
        )

        _, head_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=warm_path)
        assert head_sha.strip() == commit_a.strip(), (
            'a reset cancelled during its acquire never held the lane, so it '
            'must not have touched the tree'
        )

    async def test_reset_acquires_through_the_shared_guarded_seam(
        self, real_git_ops, monkeypatch: pytest.MonkeyPatch,
    ):
        """STRUCTURAL: the reset must not own a second acquire site.

        Behavioural coverage alone cannot hold this line.  The cancellation
        guarantee lives in `GitOps._acquire_lane_flock_off_thread`; a caller
        that hand-rolls its own `asyncio.to_thread(acquire_merge_verify_flock,
        ...)` silently opts out of it, which is precisely the state this task
        found the reset in.  Pinning the seam makes that regression loud instead
        of invisible.
        """
        from orchestrator import git_ops as git_ops_mod  # noqa: PLC0415

        calls: list[tuple[Path, float]] = []
        real_seam = GitOps._acquire_lane_flock_off_thread

        async def _spy(lock_path: Path, wait_secs: float):
            calls.append((lock_path, wait_secs))
            return await real_seam(lock_path, wait_secs)

        monkeypatch.setattr(
            GitOps, '_acquire_lane_flock_off_thread', staticmethod(_spy),
        )

        commit_a = await _get_merge_commit(real_git_ops, 'seam-a', 'seam_a.py')
        await real_git_ops.reset_persistent_merge_worktree(commit_a)
        commit_b = await _get_merge_commit(real_git_ops, 'seam-b', 'seam_b.py')

        lock_path = lane_lock_path(real_git_ops.persistent_merge_worktree_path)
        calls.clear()  # measure the reset-in-place, not the create-once above
        await real_git_ops.reset_persistent_merge_worktree(commit_b)

        assert calls == [(lock_path, git_ops_mod._RESET_WARM_LANE_LOCK_WAIT_SECS)], (
            f'the reset must acquire the lane lock through the ONE guarded '
            f'seam, with its own timeout constant — a duplicate bare '
            f'asyncio.to_thread here silently loses the cancellation '
            f'guarantee; got {calls!r}'
        )

    async def test_reset_acquire_runs_off_the_event_loop_thread(
        self, real_git_ops, monkeypatch: pytest.MonkeyPatch,
    ):
        """The reset's acquire must stay OFF the event loop.

        `test_merge_verify_lease_guard.py:135` pins this for the LEASE acquire
        only; no equivalent existed for the reset.  Converging the two onto one
        seam would otherwise leave the reset's copy of the property untested at
        exactly the moment it is being restructured — and a minutes-long
        synchronous poll on the loop freezes the whole orchestrator.
        """
        polling, outcome = self._watch_acquire(monkeypatch)

        commit = await _get_merge_commit(real_git_ops, 'off-loop', 'off_loop.py')
        await real_git_ops.reset_persistent_merge_worktree(commit)

        assert polling.is_set(), 'the reset never invoked the acquire'
        assert outcome['thread_id'] != threading.get_ident(), (
            'the reset acquired the lane lock ON the event-loop thread — a '
            'bounded wait of minutes there stalls every other coroutine'
        )
