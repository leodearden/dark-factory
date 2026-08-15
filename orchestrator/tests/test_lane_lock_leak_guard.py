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
convention) and adds five shared helpers used by the classes below:

* :func:`foreign_lane_lock_holder` — a GENUINELY foreign holder, since
  same-process fds are indistinguishable from the leak at kernel level;
* :func:`leaked_lane_lock` — the B13 fault staged as an unregistered fd with no
  holder-pgid rendezvous, exactly what an orphaned acquire leaves behind;
* :func:`wait_until` — a bounded async poll, so orphan-release callbacks settle
  deterministically instead of behind a fixed sleep;
* :func:`require_lane_lock_holders` — a bounded-retry STRICT read of the kernel
  lock table, so "I could not tell" is never rendered as "nobody holds it";
* :func:`lane_is_free` — the PRD's B12 signal: unheld AND unattributed to us,
  with a third outcome — an UNKNOWN kernel read fails loudly rather than
  passing a leak assertion vacuously (task 3604).
"""
from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace

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
    lane_lock_holder_pids_strict,
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
    'require_lane_lock_holders',
    'wait_for_lane_lock_holder',
    'wait_until',
]


#: How long the ``flock(1)`` child holds the lane lock before exiting on its
#: own.  Long enough that no test races it; the contextmanager terminates the
#: child on exit regardless, so this is only a leak backstop.
_FOREIGN_HOLDER_HOLD_SECS = 120

#: Bound on how long :func:`foreign_lane_lock_holder` waits for the child to
#: actually own the lock before failing the test — the gate waits for a
#: spawned ``flock(1)`` to acquire the lock.  SHARED derivation with
#: :data:`_FOREIGN_HOLDER_ATTRIBUTION_SECS` immediately below — the two had
#: drifted into near-verbatim copies of this comment before task 3836
#: consolidated them here, which is how the original 30.0-vs-5.0 asymmetry
#: between them arose in the first place.
#:
#: FLOOR — task 3451 measured worst-case happy-path subprocess spawn latency
#: at 4.71s (n=3: 2.13/3.10/4.71, load-per-core 6.6), the same load-sensitive
#: full-suite-flake class as 1335/1836/2819/3451/3491.  12.0 is 2.55x that
#: measured worst case, versus the original 5.0s bound's ~6% headroom —
#: pinned by :func:`test_foreign_holder_bounds_clear_measured_spawn_latency`
#: (now ``>= 12.0`` exactly, not merely ``>= 8.0``).
#:
#: CEILING (task 3836) — the 60s global pytest-timeout
#: (orchestrator/pyproject.toml: timeout_method = "thread",
#: --max-worker-restart=0).  Task 3491 is precedent AGAINST a bound near
#: that ceiling, not for one: it REJECTED a 30s ceiling for this exact
#: collision and landed ``asyncio.wait_for(..., timeout=5)`` instead
#: (test_usage_gate.py:328,790), warning that a ceiling closer to 60s
#: "would trade one flake class for a worse one."  Pinned by
#: ``test_foreign_holder_bounds_stay_clear_of_the_global_pytest_timeout``.
#: Past it, widening does NOT merely lengthen a broken staging's time to
#: fail: no test here, nor in the sibling consumer
#: ``test_merge_verify_lease_guard.py``, opts out with
#: ``@pytest.mark.timeout`` (pinned by
#: ``test_no_foreign_holder_consumer_opts_out_of_the_global_timeout``), so
#: pytest-timeout fires first and, under timeout_method="thread",
#: os._exit()s the xdist worker, discarding this helper's own
#: ``pytest.fail`` diagnostics.  A per-consumer opt-out was considered and
#: REJECTED for exactly this reason: it only protects tests that remember to
#: add it, where a shared, narrower bound protects every consumer, present
#: and future, without relying on that discipline.
#:
#: The bounds must clear the measured spawn latency AND stay narrow enough
#: that the helper can still emit its own diagnostic when they are
#: exhausted.
_FOREIGN_HOLDER_STARTUP_SECS = 12.0

#: Bound on how long :func:`foreign_lane_lock_holder` waits, on exit, for the
#: kernel to stop attributing the lock to the child.  Left UNCHANGED at 5.0:
#: this polls for the DISAPPEARANCE of an attribution after a SIGKILL to an
#: already-running process group, which is not gated on spawn latency and has
#: not been observed to flake.
_FOREIGN_HOLDER_TEARDOWN_SECS = 5.0

#: Bound on how long :func:`wait_for_lane_lock_holder` polls for the kernel
#: to ATTRIBUTE the lock to a specific pid (as opposed to merely being held
#: by somebody).  Same FLOOR/CEILING derivation as
#: :data:`_FOREIGN_HOLDER_STARTUP_SECS` immediately above — see its comment
#: for the measurements, the rejected per-consumer-timeout alternative, and
#: the executable floor/ceiling checks.  Differs only in WHAT is polled for:
#: kernel attribution of the pid, not mere heldness of the lock.
_FOREIGN_HOLDER_ATTRIBUTION_SECS = 12.0

#: Bound on how long :func:`require_lane_lock_holders` retries an UNREADABLE
#: kernel lock table before failing loudly.  Deliberately NOT the 30s
#: spawn-latency class (_FOREIGN_HOLDER_*_SECS): this retries a procfs read,
#: which either succeeds within microseconds or is structurally broken (a
#: deleted lock file, a host with no /proc/locks), so a longer bound only
#: delays a certain failure rather than buying safety margin.  It must also
#: stay well inside the 10.0s `wait_until` timeout at both B12 call sites, or
#: a single poll iteration would consume the whole outer wait.
_LANE_LOCK_STRICT_READ_SECS = 2.0


@contextlib.contextmanager
def foreign_lane_lock_holder(
    lock_path: Path,
) -> Iterator[tuple[subprocess.Popen, list[int]]]:
    """Hold *lock_path* from a GENUINELY foreign process, yielding the child.

    Spawns ``flock -x <lock_path> sleep N`` — util-linux ``flock(1)``, already
    exercised by this suite via :meth:`GitOps._seed_warm_lane`, and
    interoperable with ``fcntl.flock(2)`` on the same inode.

    Why a subprocess and not a second fd in this process: at KERNEL level a
    same-process fd is precisely the B13 self-owned case (``/proc/locks``
    reports our own pid as the holder), so it cannot stand in for foreign
    contention once the leak detector exists.  Any test that means "somebody
    else holds this lane" must use this helper.

    Blocks until TWO facts both hold, failing the test if either does not
    settle in time: (1) the lock is HELD — a zero-timeout probe acquire
    returns ``None`` — within :data:`_FOREIGN_HOLDER_STARTUP_SECS`; and (2)
    ``/proc/locks`` NAMES ``child.pid`` as the FLOCK holder, within
    :data:`_FOREIGN_HOLDER_ATTRIBUTION_SECS`.  Until task 3598, only the first
    half was ever established, even though ``test_foreign_holder_is_contention_not_a_leak``'s
    ``child.pid in holders`` assertion (and ``_lane_lock_holder_facts``'
    raise-time re-read, at the point ``git_ops`` reports the timeout) both
    depend on the second — so every "the foreign holder" assertion downstream
    was racing ``/proc/locks`` settling under load rather than testing
    genuine contention.

    Yields ``(child, holders)``: *child* is the ``flock(1)`` subprocess, and
    *holders* is the SAME kernel-reported holder-pid snapshot fact (2) above
    already polled to settle.  Reuse it rather than re-polling for the
    identical fact via a second :func:`wait_for_lane_lock_holder` call —
    that fact cannot have changed since this gate just proved it and nothing
    releases the lock in between, so a second poll only spends up to another
    full :data:`_FOREIGN_HOLDER_ATTRIBUTION_SECS` for no new information.
    (Task 3836 amendment: this is exactly the redundant poll
    ``test_foreign_holder_is_contention_not_a_leak`` used to perform on its
    own, which made that one test's true worst-case wait stack exceed what
    the ceiling invariant below computed until it was eliminated by
    threading this value through instead.)

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
        holders = wait_for_lane_lock_holder(
            lock_path, child.pid, timeout=_FOREIGN_HOLDER_ATTRIBUTION_SECS,
        )
        if child.pid not in holders:
            _kill_group()
            pytest.fail(
                f'{lock_path} is held, but /proc/locks does not attribute it '
                f'to flock(1) child {child.pid} within '
                f'{_FOREIGN_HOLDER_ATTRIBUTION_SECS}s — so any downstream '
                f'assertion about "the foreign holder" would be racing '
                f'/proc/locks settling rather than testing genuine '
                f'contention; observed holders={holders!r}'
            )
        yield child, holders
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


def wait_for_lane_lock_holder(
    lock_path: Path,
    pid: int,
    timeout: float = _FOREIGN_HOLDER_ATTRIBUTION_SECS,
    interval: float = 0.02,
    read_holders: Callable[[Path], list[int]] | None = None,
) -> list[int]:
    """Poll until *pid* is among *lock_path*'s kernel-reported FLOCK holders.

    Why this exists: :func:`lane_lock_holder_pids` is deliberately fail-safe —
    a missing lock file, an unreadable ``/proc/locks``, and an unparseable row
    all yield ``[]``, the SAME thing "nobody holds it" yields.  A single read
    therefore cannot distinguish "the child does not hold the lock" from "the
    kernel table has not settled yet" or "this particular read was bad".  That
    ambiguity is exactly what produced the observed ``assert 658016 in []``
    under 16-worker xdist load: the child genuinely held the lock, but the
    one-shot read landed on an empty snapshot.

    Returns the snapshot in which *pid* first appeared, so a caller's own
    assertion (e.g. ``assert pid in holders``) reports the real list the
    kernel returned.  On timeout, returns the LAST snapshot read instead of
    raising — mirroring :func:`wait_until`'s deliberate return-don't-raise
    contract — so the caller's assertion still carries the genuine
    kernel-observed evidence rather than a helper-internal traceback.

    *read_holders* defaults to ``None`` and is resolved to the module-global
    :func:`lane_lock_holder_pids` at CALL time (a bare name lookup inside the
    function body, not a def-time default), so a ``monkeypatch.setattr`` on
    this module's ``lane_lock_holder_pids`` attribute is honoured — the same
    module-attribute stubbing seam this suite already uses on
    ``git_ops_mod.acquire_merge_verify_flock`` inside
    ``TestCancelledAcquireNeverOrphansTheLaneLock._stub_acquire``.
    """
    reader = read_holders if read_holders is not None else lane_lock_holder_pids
    deadline = time.monotonic() + timeout
    while True:
        holders = reader(lock_path)
        if pid in holders:
            return holders
        if time.monotonic() >= deadline:
            return holders
        time.sleep(interval)


def require_lane_lock_holders(
    lock_path: Path,
    timeout: float = _LANE_LOCK_STRICT_READ_SECS,
    interval: float = 0.02,
    read_holders: Callable[[Path], list[int]] | None = None,
) -> list[int]:
    """Read *lock_path*'s kernel FLOCK holders, failing loudly if it cannot be read.

    The NEGATIVE-side counterpart of :func:`wait_for_lane_lock_holder`, and the
    contrast is the point.  That helper polls to confirm a POSITIVE attribution
    and, on timeout, RETURNS the last snapshot rather than raising — the
    caller's own ``assert pid in holders`` then carries the kernel-observed
    evidence.  Here the caller is asserting the OPPOSITE (that a lane is FREE),
    and a fail-safe empty read is precisely the answer that SATISFIES it.  So
    an unreadable table must raise: rendered as ``[]`` it would pass a B12 leak
    assertion vacuously, which is strictly worse than the red task 3598 removed
    — a silent green reports a leaked lane as clean.

    Reads through :func:`~orchestrator.verify_cancel.lane_lock_holder_pids_strict`
    (task 3604), which propagates the ``OSError`` from a missing lock file or an
    unreadable ``/proc/locks`` instead of swallowing it.  A read that SUCCEEDS
    and reports no holders is a real answer and is returned immediately — "the
    kernel says nobody holds it" is exactly what the caller is entitled to
    conclude from, and is not retried.

    The bounded retry (:data:`_LANE_LOCK_STRICT_READ_SECS`) is what keeps this
    from trading a silent green for a NEW flake class: a one-off transient
    procfs hiccup — the class of event task 3598 measured under 16-worker xdist
    load — is absorbed, while a structurally unreadable table still fails
    within the bound.

    *read_holders* defaults to ``None`` and is resolved to the module-global
    ``lane_lock_holder_pids_strict`` at CALL time (a bare name lookup inside
    the function body, not a def-time default), so a ``monkeypatch.setattr`` on
    this module's attribute is honoured — the same seam
    :func:`wait_for_lane_lock_holder` documents.
    """
    reader = read_holders if read_holders is not None else lane_lock_holder_pids_strict
    deadline = time.monotonic() + timeout
    while True:
        try:
            return reader(lock_path)
        except OSError as exc:
            last_error = exc
        if time.monotonic() >= deadline:
            # `!s`, deliberately NOT `!r`: repr(OSError) DROPS the filename —
            # repr(OSError(2, 'No such file...', '/proc/locks')) renders as
            # `FileNotFoundError(2, 'No such file...')`, while str() carries
            # `: '/proc/locks'`.  That path is the forensic distinction the
            # strict read exists to preserve, because TWO different reads can
            # fail here and they mean different things — and the headline case
            # for this task (a genuinely-held lane whose LOCK FILE was unlinked)
            # fails on os.stat(lock_path), not on /proc/locks at all.  Blaming
            # the kernel lock table for it would send a human debugging a
            # vacuous-green B12 poll straight past the actual cause.
            pytest.fail(
                f'could not determine the FLOCK holders of {lock_path} within '
                f'{timeout}s — the lock file itself, or the kernel lock table '
                f'(/proc/locks), could not be read; last error: {last_error!s}. '
                f'This is UNKNOWN, not "the lane is free": neither read examined '
                f'a single row, so rendering it as "no holders" would pass a B12 '
                f'leak assertion vacuously while the lane is genuinely held '
                f'(task 3604)'
            )
        time.sleep(interval)


def lane_is_free(
    lock_path: Path,
    strict_read_timeout: float = _LANE_LOCK_STRICT_READ_SECS,
) -> bool:
    """Whether *lock_path* is unheld AND unattributed to this process.

    The PRD's B12 signal verbatim, and BOTH halves are load-bearing: a
    successful fresh acquire alone could be succeeding against a different
    inode than the one we leaked, and ``/proc/locks`` alone is precisely the
    manual forensics reify ``esc-5548-5`` had to do by hand.

    Note the ordering — the kernel attribution is checked BEFORE the probe
    acquire, because the probe itself briefly becomes a holder and would
    otherwise report on itself.

    THREE outcomes, not two (task 3604): free, held, and UNKNOWN — where
    unknown fails loudly via :func:`require_lane_lock_holders` rather than
    returning a bool.  A fail-safe empty read cuts the OPPOSITE way here than
    it does at the positive call sites task 3598 fixed: there a bad read
    produced a spurious RED (``assert <pid> in []``), which is noisy but
    self-announcing; here it produces a silent GREEN — a B12 leak assertion
    passing vacuously — which is strictly worse, because a leaked lane is then
    reported clean and nothing surfaces.

    The measured case that motivated it: hold the lock and unlink the lock
    file, and this returned ``True`` while the lane was genuinely orphaned.
    ``os.stat`` failed so the attribution read yielded ``[]``, and the probe's
    ``O_CREAT`` then minted a FRESH inode and acquired it happily — so both
    halves answered about an inode that was not the one being held.  That is
    why the strict read comes first and why neither it nor the ordering should
    later be "simplified" back: the check must fail while the lock file is
    still absent, before the probe can resurrect the evidence.

    *strict_read_timeout* forwards to :func:`require_lane_lock_holders` and
    defaults to the full transient-absorber bound, which is what both B12 call
    sites get.  It exists for the tests below that stage a DETERMINISTICALLY
    unreadable read — a permanently-unlinked lock file, an injected reader that
    always raises — where the retry is guaranteed to exhaust and the absorber
    budget therefore buys nothing but ~2s of full-suite wall clock per case.
    Do NOT shorten it at a call site staging a REAL race: absorbing a one-off
    procfs hiccup is the only thing keeping this from becoming its own flake
    class.
    """
    if os.getpid() in require_lane_lock_holders(lock_path, timeout=strict_read_timeout):
        return False
    probe = acquire_merge_verify_flock(lock_path, 0.0)
    if probe is None:
        return False
    release_merge_verify_flock(probe)
    return True


# ---------------------------------------------------------------------------
# `wait_for_lane_lock_holder` — the bounded-poll idiom already used above, in
# the startup and teardown polls of `foreign_lane_lock_holder`, factored out
# and given a name so the fixture's own attribution gate (step-4) and the
# flaky test's staging read (step-7) can share ONE settle bound instead of
# each hand-rolling — or, as the flaky assertion did, omitting — one.
#
# Pinned here in isolation against an INJECTED fake reader, so these cases are
# deterministic and touch no real lock; the real-lock fixture behaviour is
# pinned separately, further below, by
# `test_foreign_holder_fixture_survives_a_transient_empty_holder_read`.
# ---------------------------------------------------------------------------


class TestWaitForLaneLockHolder:
    """Unit-pins for the shared bounded-poll helper, `wait_for_lane_lock_holder`."""

    def test_polls_rather_than_reading_once(self):
        """A pid that only appears on the 3rd read must still be found.

        This is the exact shape of the observed flake: a single read (today's
        one-shot ``lane_lock_holder_pids`` call) can land on an empty snapshot
        even though the pid is about to be attributed. The helper must return
        the SNAPSHOT IN WHICH THE PID APPEARED, not a filtered or synthesised
        list — callers assert on the whole list (e.g. the negative check that
        ``os.getpid()`` is NOT among the holders).
        """
        readings = [[], [], [999, 4242]]
        calls: list[Path] = []

        def _reader(path: Path) -> list[int]:
            calls.append(path)
            return readings[len(calls) - 1]

        result = wait_for_lane_lock_holder(
            # A generous timeout costs nothing here: the loop returns the
            # instant the pid appears (3rd read), so this only matters if the
            # helper genuinely stalls. A tight bound (e.g. 1.0s) risks a
            # >1s scheduling stall between reads under xdist load tripping
            # the deadline check before the 3rd read ever happens, which
            # would fail this test on the wrong assertion.
            Path('/irrelevant'), 4242, timeout=30.0, interval=0.001, read_holders=_reader,
        )

        assert result == [999, 4242], (
            f'must return the snapshot the pid actually appeared in, not a '
            f'filtered/synthesised list; got {result!r}'
        )
        assert len(calls) == 3, (
            f'must poll rather than trust a single read — the observed flake '
            f'is exactly a one-shot read landing on an empty snapshot; the '
            f'reader was called {len(calls)} time(s)'
        )

    def test_bounded_and_honest_on_timeout(self):
        """A pid that never appears must return the LAST snapshot, not hang or raise.

        Mirrors ``wait_until``'s deliberate return-don't-raise contract: the
        caller's own ``assert pid in holders`` then reports the real
        kernel-observed list — precisely the diagnostic (``assert 658016 in
        []``) that identified this flake in the first place. A helper that
        raised would replace that evidence with an internal traceback instead.
        """
        def _reader(_path: Path) -> list[int]:
            return [7]

        timeout = 0.1
        start = time.monotonic()
        result = wait_for_lane_lock_holder(
            Path('/irrelevant'), 999, timeout=timeout, interval=0.01, read_holders=_reader,
        )
        elapsed = time.monotonic() - start

        assert result == [7], (
            f'must return the last observed snapshot on timeout so the '
            f'caller\'s own assertion reports what the kernel actually said; '
            f'got {result!r}'
        )
        assert elapsed >= timeout, (
            f'must not return before the bound elapses; elapsed={elapsed!r}'
        )
        assert elapsed < timeout + 5.0, (
            f'must not overrun the bound by more than a comfortable slack '
            f'margin — widened to tolerate scheduling delays under xdist '
            f'load rather than the bound itself; elapsed={elapsed!r}'
        )

    def test_no_cost_on_the_happy_path(self):
        """A pid already attributed on the FIRST read must cost exactly one read.

        The fixture and staging-read call sites this helper is wired into
        (steps 4 and 7) are on the happy path almost always — attribution is
        normally already settled by the time anyone asks. This pins that the
        added safety net is free in that case, so none of this fixture's five
        consumers pays for a race they do not have.
        """
        calls: list[Path] = []

        def _reader(path: Path) -> list[int]:
            calls.append(path)
            return [4242]

        interval = 1.0  # deliberately large: a single extra poll would cost 1s
        result = wait_for_lane_lock_holder(
            Path('/irrelevant'), 4242, timeout=5.0, interval=interval, read_holders=_reader,
        )

        assert result == [4242]
        # `len(calls) == 1` is the real pin: it alone proves no sleep ran
        # (the loop's only sleep follows a non-matching read), so a separate
        # wall-clock upper bound would be redundant — and a genuine one would
        # be a source of flake under xdist load for no added coverage.
        assert len(calls) == 1, (
            f'attribution was already settled on the first read — must not '
            f'poll again; reader was called {len(calls)} time(s)'
        )


# ---------------------------------------------------------------------------
# `require_lane_lock_holders` (task 3604) — the NEGATIVE-side counterpart of
# `wait_for_lane_lock_holder` above.
#
# That helper polls to confirm a POSITIVE attribution and RETURNS the last
# snapshot on timeout, because the caller's own assertion then carries the
# kernel-observed evidence.  Here the caller is asserting the opposite — that
# the lane is FREE — and a fail-safe empty read is precisely the answer that
# SATISFIES it.  So this one must raise instead: an unknown read that returns
# `[]` would pass a B12 leak assertion vacuously.
#
# Pinned here against an INJECTED reader, so these cases are deterministic and
# touch no real lock and no real /proc/locks; the end-to-end real-kernel
# behaviour is pinned separately below by
# `TestLaneIsFreeNeverPassesOnAnUnknownRead`.
# ---------------------------------------------------------------------------


class TestRequireLaneLockHolders:
    """Unit-pins for the bounded-retry strict read, `require_lane_lock_holders`."""

    def test_no_cost_on_the_happy_path(self):
        """A readable table on the FIRST read must cost exactly one read.

        `lane_is_free` is called from inside a `wait_until` poll loop at both
        B12 call sites, so anything this helper spends is paid once per poll.
        Pinning that the strict read is free in the normal case is what keeps
        the fix from taxing every healthy run.
        """
        calls: list[Path] = []

        def _reader(path: Path) -> list[int]:
            calls.append(path)
            return [4242]

        result = require_lane_lock_holders(
            Path('/irrelevant'), timeout=5.0, interval=1.0, read_holders=_reader,
        )

        assert result == [4242]
        # `len(calls) == 1` is the real pin: it alone proves no sleep ran (the
        # loop's only sleep follows a failed read), so a separate wall-clock
        # upper bound would be redundant — and a genuine one would just be a
        # source of flake under xdist load for no added coverage.
        assert len(calls) == 1, (
            f'a readable table on the first read must not be re-read; the '
            f'reader was called {len(calls)} time(s)'
        )

    def test_a_transient_unreadable_read_is_absorbed(self):
        """ONE bad procfs read must be retried, not escalated to a test failure.

        The anti-new-flake pin. Converting the silent green this task removes
        into a hair-trigger red would just trade one bad failure mode for
        another — and a one-off transient bad read is exactly the class of
        event task 3598 measured under 16-worker xdist load.
        """
        calls: list[Path] = []

        def _reader(path: Path) -> list[int]:
            calls.append(path)
            if len(calls) == 1:
                raise OSError(errno.ENOENT, 'No such file or directory', '/proc/locks')
            return [4242]

        result = require_lane_lock_holders(
            Path('/irrelevant'), timeout=5.0, interval=0.001, read_holders=_reader,
        )

        assert result == [4242]
        assert len(calls) == 2, (
            f'the first read raised and must have been retried rather than '
            f'failing the test; the reader was called {len(calls)} time(s)'
        )

    def test_a_persistently_unreadable_table_fails_loudly_and_bounded(self):
        """A table that NEVER reads must fail loudly — never return `[]` — and be bounded.

        Returning `[]` is the whole defect: it is indistinguishable from "the
        lane is free", which is what the caller is trying to establish. The
        failure must also name the underlying error, or the diagnostic is
        strictly worse than the fail-safe `[]` it replaces.

        The FAILING PATH is pinned separately from the strerror, and that is
        not belt-and-braces: `repr(OSError)` drops the filename while `str()`
        keeps it, so a `{...!r}` message names neither the path nor which of
        the helper's two possible reads (`os.stat(lock_path)` or
        `/proc/locks`) actually failed — the first thing a human needs here.
        """
        def _reader(_path: Path) -> list[int]:
            raise OSError(errno.EACCES, 'Permission denied', '/proc/locks')

        timeout = 0.1
        start = time.monotonic()
        with pytest.raises(pytest.fail.Exception) as excinfo:
            require_lane_lock_holders(
                Path('/irrelevant'), timeout=timeout, interval=0.01,
                read_holders=_reader,
            )
        elapsed = time.monotonic() - start

        message = str(excinfo.value)
        assert 'Permission denied' in message, (
            f'the failure must carry the underlying error, or it diagnoses '
            f'less than the fail-safe [] it replaces; got {message!r}'
        )
        assert '/proc/locks' in message, (
            f'the FAILING PATH must survive into the message — render the '
            f'error with !s, not !r, which drops OSError.filename and so '
            f'cannot say WHICH read failed; got {message!r}'
        )

        assert elapsed >= timeout, (
            f'must not give up before the bound elapses — a hair trigger would '
            f'be its own flake class; elapsed={elapsed!r}'
        )
        assert elapsed < timeout + 5.0, (
            f'must not overrun the bound by more than a comfortable slack '
            f'margin — widened to tolerate scheduling delays under xdist '
            f'load rather than the bound itself; elapsed={elapsed!r}'
        )

    def test_a_genuinely_empty_holder_list_is_returned_not_failed(self):
        """"The kernel says nobody holds it" must be RETURNED, not retried or failed.

        The entire distinction this task turns on. Without this pin, the fix
        would make every genuinely-free lane fail — loud, but useless, and it
        would break both B12 consumers, whose whole point is to observe a lane
        becoming free.
        """
        calls: list[Path] = []

        def _reader(path: Path) -> list[int]:
            calls.append(path)
            return []

        result = require_lane_lock_holders(
            Path('/irrelevant'), timeout=5.0, interval=1.0, read_holders=_reader,
        )

        assert result == [], (
            f'a successful read reporting no holders is a real ANSWER, not an '
            f'unknown one; got {result!r}'
        )
        assert len(calls) == 1, (
            f'an empty-but-successful read must not be retried; the reader was '
            f'called {len(calls)} time(s)'
        )


# ---------------------------------------------------------------------------
# `lane_is_free` must never pass on an UNKNOWN read (task 3604).
#
# Measured against the real kernel before this was written, with a leaked fd
# and a real flock:
#
#   A. we genuinely hold it, lock file present     -> False   (correct)
#   B. we genuinely hold it, lock file UNLINKED    -> True    (THE SILENT GREEN)
#   C. we genuinely hold it, lock table unreadable -> False   (silent, not wrong)
#
# B is the defect in its sharpest form and needs no mocks: os.stat fails so the
# attribution read yields [], then the probe's O_CREAT mints a FRESH inode and
# succeeds — so both halves answer about an inode that is NOT the one being
# held.  Wrapped in `wait_until` at the two B12 call sites, one such poll
# short-circuits the wait and the leak assertion passes green while the lane is
# genuinely orphaned.
#
# C does NOT change today's return value (the probe already catches it), which
# is exactly why its case below asserts on a RAISE and not on `is False` — see
# that test's docstring.
# ---------------------------------------------------------------------------


class TestLaneIsFreeNeverPassesOnAnUnknownRead:
    """lane_is_free has THREE outcomes — free, held, and UNKNOWN (loud)."""

    def test_a_held_but_unlinked_lane_fails_loudly_without_recreating_the_lock_file(
        self, tmp_path: Path
    ):
        """THE headline case: a genuinely-held lane whose lock file is gone.

        Staged entirely against a real kernel with no monkeypatching, because a
        mock-free reproduction proves the two halves genuinely answer about
        DIFFERENT inodes rather than merely proving that a stubbed ``[]``
        propagates.

        Measured today: this returns ``True``. ``os.stat`` fails on the
        unlinked path so the attribution read yields ``[]``, and the probe's
        ``O_CREAT`` then mints a fresh inode and acquires it happily. Both
        halves pass vacuously, and the ``wait_until`` wrapper at the two B12
        call sites short-circuits green on the first such poll — a leaked lane
        reported clean, strictly worse than the red task 3598 removed.

        The second assertion — that the lock file is still ABSENT afterwards —
        is non-obvious and load-bearing, and is the reason this stays one test
        rather than two: it pins the ORDERING that makes the first assertion
        reachable at all. Attribution is checked before the probe (as
        ``lane_is_free`` documents), so a correct implementation fails while
        the lock file is still gone. Today's code does not: the probe's
        ``O_CREAT`` recreates the very file whose absence made the answer
        unknown, quietly repairing the evidence a human would need to diagnose
        the leak. Split across two tests these were byte-identical staging and
        the weaker one covered nothing the stronger did not.

        The short *strict_read_timeout*: the unlink is PERMANENT, so the retry
        is guaranteed to exhaust. The absorber bound exists to ride out a
        transient procfs hiccup and buys nothing against a deterministic
        never-resolves staging — it would only spend the default 2s of
        full-suite wall clock waiting for a certain failure.
        """
        lock_path = lane_lock_path(tmp_path / 'lane')
        with leaked_lane_lock(lock_path):
            lock_path.unlink()

            with pytest.raises(pytest.fail.Exception):
                lane_is_free(lock_path, strict_read_timeout=0.1)

            assert not lock_path.exists(), (
                'the probe-acquire ran despite the attribution read being '
                'unknown, recreating the lock file it was asked about — the '
                'strict read must abort first'
            )

    def test_an_unreadable_lock_table_is_not_silently_no_holders(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """An unreadable kernel lock table is UNKNOWN, not "nobody holds it".

        Why this asserts on the RAISE and not on ``lane_is_free(...) is
        False``: measured, with the lane genuinely held, today's return is
        ALREADY ``False`` — the probe-acquire half fails on the
        same-process/different-fd flock conflict and masks the bad read. An
        ``is False`` assertion would therefore be green from the start and pin
        nothing. Asserting the raise is red today for the right reason (DID NOT
        RAISE) and pins the property actually being added.

        On the staging seam: the plan called for
        ``monkeypatch.setattr(verify_cancel_mod, 'PROC_LOCKS_PATH', ...)``, but
        that is INERT and was measured to be so — ``PROC_LOCKS_PATH`` is
        consumed as a DEF-TIME DEFAULT (``locks_path: Path = PROC_LOCKS_PATH``)
        in both reader variants, so rebinding the module attribute after import
        never reaches them and the real ``/proc/locks`` is still read. The
        working injection points are the ``locks_path=`` keyword (which
        ``lane_is_free`` deliberately does not expose) and a module-attribute
        stub on THIS module, which is the seam
        ``require_lane_lock_holders`` resolves by bare name specifically to
        support, and which this file already uses twice above. Recorded here so
        a later reader does not re-attempt the inert one (esc-3604-1).
        """
        def _unreadable(_path: Path) -> list[int]:
            raise OSError(errno.EACCES, 'Permission denied', '/proc/locks')

        monkeypatch.setattr(
            sys.modules[__name__], 'lane_lock_holder_pids_strict', _unreadable,
        )

        lock_path = lane_lock_path(tmp_path / 'lane')
        with leaked_lane_lock(lock_path), pytest.raises(pytest.fail.Exception):
            # Short bound for the same reason as the case above: this reader
            # ALWAYS raises, so the transient-absorber retry is certain to
            # exhaust and the default 2s would be spent waiting for it.
            lane_is_free(lock_path, strict_read_timeout=0.1)

    def test_a_genuinely_free_lane_is_still_reported_free(self, tmp_path: Path):
        """NON-VACUITY CONTROL: a fresh, unheld lane must still read as free.

        Without this and its sibling below, the fix could satisfy every case
        above by simply failing on everything — and both B12 consumers, whose
        entire purpose is to observe a lane BECOMING free, would then be
        loud but useless.

        Staged on an UNTOUCHED REAL LOCK FILE — present, and nobody's — which
        is the state at both B12 call sites (the ``flock(1)`` child and the
        orphaned acquire's ``O_CREAT`` each leave the file behind). The
        distinction matters and is not cosmetic: an ABSENT lock file is not
        "free", it is UNKNOWN, because ``os.stat`` cannot tell a path that was
        never created from one unlinked out from under a still-held fd — which
        is precisely case (a) above. Reading absent as free is the defect this
        class exists to close, so this control must not be re-staged on a
        nonexistent path.
        """
        lock_path = lane_lock_path(tmp_path / 'lane')
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.touch()

        assert lane_is_free(lock_path) is True

    def test_a_genuinely_held_lane_is_still_reported_held(self, tmp_path: Path):
        """NON-VACUITY CONTROL: a leaked lane must still read as held, not unknown.

        The B13 signal itself. A held lane whose lock file is intact is a
        KNOWN answer — ``False`` — and must not be escalated to a loud failure
        just because the strict read now exists.
        """
        lock_path = lane_lock_path(tmp_path / 'lane')
        with leaked_lane_lock(lock_path):
            assert lane_is_free(lock_path) is False


def test_scaffold_helpers_are_importable(tmp_path: Path):
    """Smoke pin: the shared helpers exist and the foreign holder really holds.

    Guards the scaffold itself — every class in this module builds on
    ``foreign_lane_lock_holder`` observably owning the lock, so a broken helper
    would silently make downstream assertions vacuous rather than red.

    ``wait_until`` is smoked on BOTH arms.  The satisfied arm alone is
    tautological — a constantly-true predicate cannot fail, so it pins nothing
    about the timeout it is supposedly exercising.  The timeout arm is the one
    the downstream orphan-settle assertions actually depend on: they read the
    RETURN value, so a helper that raised or looped forever on a predicate that
    never holds would hang the suite instead of failing it.
    """
    lock_path = lane_lock_path(tmp_path / 'lane')
    with foreign_lane_lock_holder(lock_path):
        assert acquire_merge_verify_flock(lock_path, 0.0) is None

    async def _both_arms() -> tuple[bool, bool]:
        return (
            await wait_until(lambda: True, timeout=0.1),
            await wait_until(lambda: False, timeout=0.05, interval=0.01),
        )

    became_true, timed_out = asyncio.run(_both_arms())
    assert became_true is True, 'a predicate that already holds must report True'
    assert timed_out is False, (
        'wait_until must RETURN False once the timeout elapses rather than '
        'raise or spin — the orphan-settle assertions below assert on that '
        'return value, and a non-terminating helper would hang the suite'
    )


def test_foreign_holder_fixture_survives_a_transient_empty_holder_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The fixture's own attribution must survive ONE bad ``/proc/locks`` read.

    Reproduces the observed flake deterministically, against a REAL
    ``flock(1)`` foreign holder: :func:`lane_lock_holder_pids` is wrapped so
    it returns an empty snapshot on its FIRST call — exactly the transient
    ``assert 658016 in []`` observed under 16-worker xdist load — and
    delegates to the real probe on every later call.

    Today, :func:`foreign_lane_lock_holder` never reads
    ``lane_lock_holder_pids`` before yielding — its startup gate checks a
    probe *acquire* instead, and its only read of ``lane_lock_holder_pids``
    is in TEARDOWN, after the child is already dead.  So the injected empty
    snapshot is consumed by THIS test's own single read below, and the
    assertion fails exactly as the reported flake did:
    ``assert <child.pid> in []``.  Once the fixture itself polls for
    attribution (step-4), it is the one that absorbs the bad read, and this
    test's single read lands on a settled, real snapshot instead.
    """
    calls: list[Path] = []
    real_reader = lane_lock_holder_pids

    def _flaky(path: Path) -> list[int]:
        calls.append(path)
        if len(calls) == 1:
            return []
        return real_reader(path)

    monkeypatch.setattr(sys.modules[__name__], 'lane_lock_holder_pids', _flaky)

    lock_path = lane_lock_path(tmp_path / 'lane')
    with foreign_lane_lock_holder(lock_path) as (child, _fixture_holders):
        # Deliberately NOT using _fixture_holders below: this test's whole
        # point is that a single BARE read (not the fixture's own settled
        # snapshot) can legitimately observe the transient empty read.
        # Captured BEFORE the bare read below adds its own call, and before
        # the `with` exits and teardown adds more still — otherwise a
        # post-block count could not tell "the fixture polled" from "the
        # test's own read plus teardown's poll happened to add up to >1",
        # which a fixture that never polls at all would also satisfy.
        calls_at_yield = len(calls)
        holders = lane_lock_holder_pids(lock_path)  # deliberately a bare single read
        assert child.pid in holders, (
            f'the fixture must not yield until the KERNEL names the child as '
            f'the holder, or every "the foreign holder" assertion downstream '
            f'is racing /proc/locks settling — even a single transient empty '
            f'read (exactly what was observed: assert <pid> in []) would '
            f'defeat it; got holders={holders!r}'
        )

    assert calls_at_yield >= 2, (
        f'the fixture must actually have read lane_lock_holder_pids more '
        f'than once BEFORE yielding — absorbing the injected empty read, '
        f'then settling on a real one — or a fixture that merely happened '
        f'to skip the bad read would pass this test vacuously; the fixture '
        f'read it {calls_at_yield} time(s) before yielding'
    )


def test_foreign_holder_fixture_fails_loudly_when_attribution_never_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The attribution gate's FAILURE arm: a kernel that never settles must fail loudly.

    Only the transient-single-bad-read path is pinned above. A regression
    that inverted the gate's ``if child.pid not in holders:`` condition, or
    that raised without killing the process group first, would let every
    downstream "the foreign holder" assertion go back to racing
    ``/proc/locks`` while this whole suite stayed green — precisely the
    failure mode task 3598 exists to prevent.

    Patches ``wait_for_lane_lock_holder`` itself — the exact symbol
    ``foreign_lane_lock_holder``'s attribution gate calls by bare name — to a
    fake that always reports "no holders", rather than patching the
    lower-level ``lane_lock_holder_pids``. The latter would ALSO defeat the
    fixture's teardown poll (which reads the very same function to confirm
    the killed child has genuinely released the lock), racing this test's own
    post-block verification against a real SIGKILL under load — precisely
    the single-shot-read hazard this task exists to fix. Patching the
    higher-level poll leaves the teardown poll's real, bounded kernel reads
    untouched, so the verification below is itself race-free.
    """
    def _never_attributed(*_args, **_kwargs) -> list[int]:
        return []

    monkeypatch.setattr(sys.modules[__name__], 'wait_for_lane_lock_holder', _never_attributed)

    lock_path = lane_lock_path(tmp_path / 'lane')
    with (
        pytest.raises(pytest.fail.Exception, match='does not attribute it'),
        foreign_lane_lock_holder(lock_path),
    ):
        pytest.fail(
            'unreachable: the fixture must fail before yielding when '
            'attribution never settles'
        )

    # The gate's side effect: it must kill the foreign holder's process
    # group before failing, not merely raise and leave it running. A
    # genuinely killed child releases the flock, so a fresh probe-acquire
    # now succeeds; a gate that raised without cleaning up would leave the
    # lock held, and this probe would return None instead.
    probe = acquire_merge_verify_flock(lock_path, 0.0)
    assert probe is not None, (
        'the gate must kill the foreign-holder process group before '
        'failing — a broken gate that raised without cleaning up would '
        'leak the held lane lock past this very test'
    )
    release_merge_verify_flock(probe)


# ---------------------------------------------------------------------------
# `_effective_per_test_timeout` — resolves the per-test timeout pytest-timeout
# will actually enforce for THIS run, mirroring the installed plugin's own
# precedence chain (`pytest_timeout.get_env_settings`) instead of a bare
# `tomllib` read of a single pyproject.toml. The ceiling invariant below
# needs this because which pyproject.toml governs a given run is not fixed
# (task 3836 review amendment; precedent:
# test_marker_registration_drift.py:734-757, which pins that "registered"
# means the EFFECTIVE `getini(...)` set, not a bare TOML read).
#
# Pinned here in isolation against a stub config, so each precedence branch
# is exercised deterministically without spawning a live pytest sub-run —
# same convention as `TestWaitForLaneLockHolder` above.
# ---------------------------------------------------------------------------


def _effective_per_test_timeout(config) -> float | None:
    """The per-test timeout ``pytest-timeout`` will actually enforce for
    *config*'s run, or ``None`` if no timeout is in effect.

    Mirrors the installed plugin's own ``get_env_settings`` precedence
    chain (``getvalue('timeout')`` -> ``PYTEST_TIMEOUT`` env var ->
    ``getini('timeout')``) rather than reading pyproject.toml directly, so
    the result is correct regardless of which pyproject.toml pytest
    resolved as its rootdir inifile, and honours ``--timeout``/``-o
    timeout=``/``PYTEST_TIMEOUT`` overrides the way a bare TOML read
    cannot (task 3836 review amendment).

    The ``hasplugin`` check MUST run first and short-circuit before any
    ``getvalue``/``getini`` call: with the plugin disabled (``-p
    no:timeout``), ``getini('timeout')`` raises ``ValueError: unknown
    configuration value: 'timeout'`` (measured) because the plugin never
    registered the ini option in the first place.

    The ini value is a STRING when present (``'60'``, not ``60``) and its
    unset sentinel is the EMPTY STRING (``''``), not ``None`` (both
    measured) — hence the explicit ``float()`` coercion and the falsiness
    check rather than an ``is not None`` guard.
    """
    if not config.pluginmanager.hasplugin('timeout'):
        return None
    cli_timeout = config.getvalue('timeout')
    if cli_timeout is not None:
        return float(cli_timeout)
    env_timeout = os.environ.get('PYTEST_TIMEOUT')
    if env_timeout:
        return float(env_timeout)
    ini_timeout = config.getini('timeout')
    if ini_timeout:
        return float(ini_timeout)
    return None


def _stub_pytest_config(
    *,
    cli_timeout: float | None,
    ini_timeout: str,
    has_timeout_plugin: bool = True,
) -> SimpleNamespace:
    """A minimal stand-in for ``pytest.Config``, exposing only the surface
    ``_effective_per_test_timeout`` reads: ``getvalue``, ``getini``, and
    ``pluginmanager.hasplugin``.

    When *has_timeout_plugin* is False, ``getvalue``/``getini`` raise
    ``ValueError`` if called at all — mirroring the MEASURED ``-p
    no:timeout`` behaviour, where ``getini('timeout')`` raises exactly
    ``ValueError: unknown configuration value: 'timeout'`` because the
    plugin never registered the ini option. A caller that skips the
    ``hasplugin`` guard therefore surfaces as a test ERROR here, not a
    silently wrong answer.
    """
    def _raise_if_disabled(name: str) -> None:
        if not has_timeout_plugin:
            raise ValueError(f'unknown configuration value: {name!r}')

    def _getvalue(name: str) -> float | None:
        _raise_if_disabled(name)
        return cli_timeout if name == 'timeout' else None

    def _getini(name: str) -> str:
        _raise_if_disabled(name)
        assert name == 'timeout'
        return ini_timeout

    return SimpleNamespace(
        getvalue=_getvalue,
        getini=_getini,
        pluginmanager=SimpleNamespace(
            hasplugin=lambda name: has_timeout_plugin if name == 'timeout' else False
        ),
    )


class TestEffectivePerTestTimeout:
    """Unit-pins for :func:`_effective_per_test_timeout`'s two branches a
    live run cannot exercise on its own, plus a differential check against
    the installed plugin for everything else.

    Task 3836 review amendment: this class used to carry six hand-rolled
    precedence cases (CLI-beats-ini, env-when-no-CLI, CLI-beats-env, str
    coercion, plus the two kept below), each pinned against a stub config
    that encoded the SAME precedence assumptions the helper encodes — so
    none of them could catch the one drift risk that actually matters,
    pytest-timeout changing its own ``get_env_settings`` precedence chain or
    ini registration. Four were deleted; ``test_matches_the_installed_plugins_own_resolution``
    below replaces them with a direct comparison against the plugin's own
    function under the LIVE config, which is both higher-signal (it is
    exactly the drift risk) and cannot be satisfied by a mock written to
    agree with the code under test. Only the two branches a live run cannot
    exercise itself stay as stubs: an unset ini (the EMPTY STRING sentinel,
    not ``None``) and a fully disabled plugin (``-p no:timeout``, under
    which ``getini`` itself raises).
    """

    def test_empty_ini_string_means_no_timeout_in_effect(self, monkeypatch):
        """Measured: a root-bound run (``-c pyproject.toml``). The root
        pyproject declares no ``timeout`` key, so ``getini('timeout')``
        returns the EMPTY STRING, not ``None`` — a naive ``is not None``
        guard would fall through to ``float('')`` and raise ``ValueError``.
        """
        monkeypatch.delenv('PYTEST_TIMEOUT', raising=False)
        config = _stub_pytest_config(cli_timeout=None, ini_timeout='')
        assert _effective_per_test_timeout(config) is None

    def test_disabled_plugin_short_circuits_before_any_config_read(self, monkeypatch):
        """Measured: ``-p no:timeout``. ``getini('timeout')`` raises
        ``ValueError: unknown configuration value: 'timeout'`` in this mode —
        an uncaught crash, not a legible failure. The implementation must
        check ``hasplugin`` FIRST and never reach ``getvalue``/``getini`` at
        all; the stub raises from both if that guard is skipped, so a
        regression here surfaces as this test erroring, not silently
        returning the wrong answer.
        """
        monkeypatch.delenv('PYTEST_TIMEOUT', raising=False)
        config = _stub_pytest_config(
            cli_timeout=None, ini_timeout='60', has_timeout_plugin=False,
        )
        assert _effective_per_test_timeout(config) is None

    def test_matches_the_installed_plugins_own_resolution(self, pytestconfig):
        """Differential pin against the actual drift risk (task 3836 review
        amendment).

        Compares directly against ``pytest_timeout.get_env_settings`` — the
        installed plugin's OWN resolution function — under the live
        ``pytestconfig`` of whatever invocation is actually running this
        suite, instead of reproducing its precedence logic against a mock
        written to agree with it. If the plugin ever changes that chain or
        its ini registration, this is the one test that would actually
        notice; the stub cases above cannot, by construction.
        """
        from pytest_timeout import get_env_settings  # noqa: PLC0415

        assert _effective_per_test_timeout(pytestconfig) == get_env_settings(pytestconfig).timeout


def test_foreign_holder_bounds_clear_measured_spawn_latency():
    """Both foreign-holder bounds must clear a MEASURED worst-case spawn latency.

    Not a guess: task 3451 measured worst-case happy-path subprocess spawn
    latency on this host at 4.71s (n=3: 2.13/3.10/4.71) at load-per-core 6.6.
    Any bound near 5s is within noise of that measured worst case — which is
    exactly how ``_FOREIGN_HOLDER_STARTUP_SECS = 5.0`` produced the flake this
    task fixes. This is the FLOOR half of a two-sided contract (task 3836):
    these bounds are also bounded from above by the global pytest-timeout,
    pinned by the sibling invariant
    ``test_foreign_holder_bounds_stay_clear_of_the_global_pytest_timeout``
    immediately below. Widening buys safety margin for free only up to that
    ceiling — past it, a genuinely broken staging no longer merely takes longer
    to fail: pytest-timeout fires first and, under timeout_method="thread",
    os._exit()s the xdist worker, discarding this helper's own ``pytest.fail``
    diagnostics. Tightening either bound back below the measured worst case
    still buys nothing but another full-suite flake — hence pinning the floor
    as an invariant here, following task 3451's headroom-invariant precedent
    (``test_set_started_grace_floor_clears_measured_happy_path_latency``).

    Task 3836 review amendment: asserts ``>= 12.0`` exactly, not merely
    ``>= 8.0``. The old ``>= 8.0`` floor, combined with the sibling ceiling
    invariant's constraint that the two bounds sum to at most 26.0 (so their
    ``+ teardown(5.0) + kill-wait(5.0)`` stays within the 36.0 ceiling),
    PERMITTED — without pinning — any pair satisfying both ``>= 8.0``
    individually and ``<= 26.0`` jointly: e.g. 8.0/8.0, or an asymmetric
    18.0/8.0, neither of which is the 12.0/12.0 actually chosen. A future
    accidental narrowing of either bound back toward 8.0 (eroding the 2.55x
    measured-latency margin this task exists to establish) would have passed
    both invariants unnoticed. Asserting the actual chosen value directly
    makes the choice itself, not just the region that permits it, an
    invariant.
    """
    bounds = {
        '_FOREIGN_HOLDER_STARTUP_SECS': _FOREIGN_HOLDER_STARTUP_SECS,
        '_FOREIGN_HOLDER_ATTRIBUTION_SECS': _FOREIGN_HOLDER_ATTRIBUTION_SECS,
    }
    assert all(bound >= 12.0 for bound in bounds.values()), (
        f'both foreign-holder bounds must stay at or above 12.0s — the '
        f'value task 3836 chose (2.55x the measured worst-case happy-path '
        f'spawn latency of 4.71s at load-per-core 6.6, task 3451), pinning '
        f'the actual choice rather than merely a >= 8.0 floor that permitted '
        f'e.g. 8.0/8.0 or an asymmetric 18.0/8.0 split without failing; got '
        f'{bounds!r}'
    )


def test_foreign_holder_bounds_stay_clear_of_the_global_pytest_timeout(pytestconfig):
    """The foreign-holder bounds must also clear the global pytest-timeout.

    Mirrors :func:`test_foreign_holder_bounds_clear_measured_spawn_latency`
    immediately above: that test pins the FLOOR (bounds must clear the
    measured spawn latency); this test pins the CEILING (bounds must not
    approach the global per-test timeout). Together they bound the same two
    constants from both sides.

    ``foreign_lane_lock_holder`` runs its startup poll
    (``_FOREIGN_HOLDER_STARTUP_SECS``) then its attribution poll
    (``_FOREIGN_HOLDER_ATTRIBUTION_SECS``) back-to-back, and on the way out
    — via ``finally`` — the ``_kill_group`` kill-wait
    (``child.wait(timeout=5)``) followed by the teardown poll
    (``_FOREIGN_HOLDER_TEARDOWN_SECS``). All four are entered
    UNCONDITIONALLY on every use and run in sequence, not in parallel, so
    their SUM is the WORST-CASE wall clock a single use can spend inside the
    helper — not what it typically costs. Each poll returns the instant its
    condition holds (the startup loop ``break``s the moment the probe
    acquire returns ``None``; ``wait_for_lane_lock_holder`` returns the
    instant the pid appears; ``child.wait`` returns immediately once the
    SIGKILL it waits on lands), so on the happy path this costs
    milliseconds, not seconds — the margin asserted below costs nothing in
    practice; it exists for the day the staging is genuinely broken and
    every poll runs out its deadline.

    Task 3836 amendment: ``test_foreign_holder_is_contention_not_a_leak``
    used to pay a further bounded wait of its own INSIDE the ``with`` block
    — a second, redundant ``wait_for_lane_lock_holder`` call re-checking the
    very attribution the fixture had just established.  That second wait was
    NOT mutually exclusive with the fixture's own entry stack (an earlier
    poll returning successfully-but-slowly does not prevent a later,
    independent poll from also being slow), so it was not covered by
    ``unconditional_stack`` below and that one test's true worst case could
    exceed what this invariant computed.  It has been eliminated —
    :func:`foreign_lane_lock_holder` now yields its settled attribution
    snapshot for reuse — rather than merely adding its bound to the sum, so
    ``unconditional_stack`` genuinely is every consumer's worst case, not
    just a lower bound on it.

    The effective per-test timeout is resolved via
    :func:`_effective_per_test_timeout`, which mirrors pytest-timeout's own
    precedence chain (``--timeout`` CLI flag, then ``PYTEST_TIMEOUT`` env
    var, then the ini ``timeout`` key read through the EFFECTIVE pytest
    config) rather than a bare ``tomllib`` read of a single pyproject.toml
    (task 3836 review amendment) — so this invariant is correct regardless
    of which pyproject.toml pytest resolved as its rootdir inifile, and
    honours ``--timeout``/``-o timeout=``/``PYTEST_TIMEOUT`` overrides the
    way a bare TOML read cannot. A run under a genuinely smaller effective
    timeout (e.g. ``--timeout=30``, where the ceiling becomes 18.0s against
    the 34.0s stack) correctly turns this test RED — that is the invariant
    working, not a bug to be papered over by widening the 60% fraction.
    When no timeout is in effect at all, this test either fails loudly
    (orchestrator/pyproject.toml is the governing inifile — genuine drift)
    or skips, naming the governing inifile (an invocation artifact, e.g. a
    root-bound run whose pyproject.toml declares no ``timeout`` key at
    all).

    Why 60% of the global timeout, not 100%: no test in this module, NOR in
    its sibling consumer ``test_merge_verify_lease_guard.py`` (which imports
    and uses this helper at two of its own call sites), opts out with
    ``@pytest.mark.timeout`` — made executable, not just asserted here in
    prose, by
    ``test_no_foreign_holder_consumer_opts_out_of_the_global_timeout``
    immediately below. So once the unconditional stack collides
    with the global ceiling, pytest-timeout fires FIRST — and under
    ``timeout_method = "thread"`` (with ``--max-worker-restart=0``) it
    os._exit()s the xdist worker rather than failing the test cleanly.
    That destroys the helper's own carefully-authored ``pytest.fail``
    diagnostics (the "foreign-holder staging is broken" / "observed
    holders=..." messages) and replaces them with an opaque worker death
    carrying none of that structured evidence — exceeding the global timeout
    does not merely delay a failure, it destroys the failure's diagnostic
    content. The remaining 40% is headroom for the real-git fixture work
    (``git_repo``/``real_git_ops``, ``_get_merge_commit``,
    ``reset_persistent_merge_worktree``) sharing the same per-test budget.
    """
    global_timeout = _effective_per_test_timeout(pytestconfig)
    if global_timeout is None:
        orchestrator_pyproject = Path(__file__).resolve().parents[1] / 'pyproject.toml'
        governing_inifile = pytestconfig.inipath
        if governing_inifile is not None and governing_inifile.resolve() == orchestrator_pyproject:
            pytest.fail(
                f'{governing_inifile} is the governing inifile for this run, '
                f'yet no per-test timeout is in effect (checked --timeout, '
                f'PYTEST_TIMEOUT, and [tool.pytest.ini_options].timeout) — '
                f'this is a genuine regression: the pytest-timeout / '
                f'os._exit() premise the foreign-holder bounds in this '
                f'module are sized against no longer holds. Either restore '
                f'the timeout key or re-derive these bounds without it.'
            )
        pytest.skip(
            f'no per-test timeout is in effect under the governing inifile '
            f'{governing_inifile!r} — this is not {orchestrator_pyproject}, '
            f'so this is an invocation artifact (e.g. a root-bound run), '
            f'not drift; the pytest-timeout ceiling this invariant checks '
            f'does not apply to this run'
        )

    # The _kill_group kill-wait (child.wait(timeout=5)) runs unconditionally
    # on every exit path, in addition to the three named _FOREIGN_HOLDER_*
    # bounds.
    unconditional_stack = (
        _FOREIGN_HOLDER_STARTUP_SECS
        + _FOREIGN_HOLDER_ATTRIBUTION_SECS
        + _FOREIGN_HOLDER_TEARDOWN_SECS
        + 5.0
    )
    ceiling = 0.6 * global_timeout
    assert unconditional_stack <= ceiling, (
        f'the WORST-CASE unconditional wait stack inside '
        f'foreign_lane_lock_holder is {unconditional_stack}s '
        f'({_FOREIGN_HOLDER_STARTUP_SECS} startup + '
        f'{_FOREIGN_HOLDER_ATTRIBUTION_SECS} attribution + '
        f'{_FOREIGN_HOLDER_TEARDOWN_SECS} teardown + 5.0 kill-wait), which '
        f'exceeds 60% of the effective per-test timeout ({global_timeout}s, '
        f'resolved from {pytestconfig.inipath}) — only {ceiling}s of '
        f'headroom is allowed. No test in this module, nor in the sibling '
        f'consumer test_merge_verify_lease_guard.py, opts out with '
        f'@pytest.mark.timeout, so exceeding the effective timeout does not '
        f'merely delay a failure: pytest-timeout fires first and, under '
        f'timeout_method="thread" with --max-worker-restart=0, os._exit()s '
        f'the xdist worker instead of failing cleanly — discarding the '
        f"helper's own pytest.fail diagnostics and every bit of structured "
        f'evidence they carry. This is a worst-case bound, not a typical '
        f'cost: each poll returns the instant its condition holds, so a '
        f'healthy run spends milliseconds here, not seconds.'
    )


_TIMEOUT_MARKER_RE = re.compile(r'^\s*@pytest\.mark\.timeout\b', re.MULTILINE)


def test_no_foreign_holder_consumer_opts_out_of_the_global_timeout():
    """The ceiling invariant's premise, made executable instead of prose-only.

    ``test_foreign_holder_bounds_stay_clear_of_the_global_pytest_timeout``
    (immediately above) derives its 60% ceiling on the assumption that NO
    test using :func:`foreign_lane_lock_holder` opts out of the global
    per-test timeout with ``@pytest.mark.timeout`` — until now that was only
    ever asserted in prose. ``foreign_lane_lock_holder`` has exactly two
    known consumers: this module, and ``test_merge_verify_lease_guard.py``
    (which imports it by name and uses it at two ``with
    foreign_lane_lock_holder(...)`` call sites). A future
    ``@pytest.mark.timeout(N)`` added to either file would override the
    very timeout the ceiling invariant measures for that test, silently
    invalidating its derivation without turning that invariant red — the
    marker changes what governs the marked test, not the module-wide
    default this invariant reads.

    Deliberately coarse: this fails on ANY applied ``@pytest.mark.timeout``
    marker found in either file, whether or not that specific test actually
    calls ``foreign_lane_lock_holder`` — correlating a marker to a specific
    call site would need AST analysis, and given how rarely either file
    needs a per-test timeout override today (zero), a possible false
    positive here is a cheap price for a check that needs none. A marker
    LINE is identified by its first non-whitespace token being
    ``@pytest.mark.timeout`` — the only way to apply it as a decorator — so
    this cannot mistake this module's own prose references to the marker
    (its docstrings mention it by name, unapplied) for a real opt-out.
    """
    this_module = Path(__file__)
    sibling = this_module.parent / 'test_merge_verify_lease_guard.py'
    assert sibling.is_file(), (
        f'{sibling} must exist — it is a known consumer of '
        f'foreign_lane_lock_holder; if it moved or was renamed, update this '
        f'test and the ceiling invariant\'s docstring to name its new '
        f'location'
    )
    for consumer in (this_module, sibling):
        offending = _TIMEOUT_MARKER_RE.findall(consumer.read_text())
        assert not offending, (
            f'{consumer} applies @pytest.mark.timeout to at least one test '
            f'({len(offending)} marker line(s)) — this overrides the global '
            f'per-test timeout that '
            f'test_foreign_holder_bounds_stay_clear_of_the_global_pytest_timeout '
            f"measures, so that invariant's ceiling derivation no longer "
            f'covers the marked test. If the marked test does not use '
            f'foreign_lane_lock_holder, this is a false positive from a '
            f'deliberately coarse check — narrow the marker or this scan. '
            f'If it does, either remove the opt-out or re-derive that '
            f"test's own headroom against the marker's own timeout value."
        )


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

            exc = excinfo.value
            msg = str(exc)
            # Structured payload FIRST.  A bare `str(pid) in msg` containment
            # check can be satisfied by the wrong field entirely — a pgid of
            # 123 "passes" against a message that merely happens to name pid
            # 1234 — so the field identities are asserted where they are
            # unambiguous, and the message is then checked in DELIMITED form.
            assert exc.self_pid == os.getpid(), (
                f'the fault must name the leaking pid — OUR pid — so an '
                f'operator is not left doing the incident\'s manual '
                f'/proc/locks forensics again; got {exc.self_pid!r}'
            )
            assert exc.self_pgid == os.getpgrp(), (
                f'the fault must carry our pgid, the corroborating fact the '
                f'holder-pgid rendezvous would have carried had the acquire '
                f'ever completed; got {exc.self_pgid!r}'
            )
            assert f'pid={os.getpid()}' in msg, (
                f'the RENDERED message is what reaches the operator, so it '
                f'must state the leaking pid, not merely carry it; got {msg!r}'
            )
            assert f'pgid={os.getpgrp()}' in msg, (
                f'…and our pgid alongside it; got {msg!r}'
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

        with foreign_lane_lock_holder(lock_path) as (child, holders):
            # `holders` is the fixture's OWN settled attribution snapshot —
            # not a fresh read.  Task 3836 amendment: this used to re-poll
            # via its own `wait_for_lane_lock_holder(...,
            # timeout=_FOREIGN_HOLDER_ATTRIBUTION_SECS)` call here, spending
            # up to a second full attribution bound to re-establish a fact
            # the fixture had just proved moments earlier — nothing releases
            # the lock in between, so that second poll could only ever
            # re-confirm the same snapshot, never learn anything new. Reusing
            # it removes that redundant wait (see
            # `foreign_lane_lock_holder`'s docstring) and keeps this test's
            # true worst-case wait stack equal to the helper's own
            # unconditional stack, which is what
            # `test_foreign_holder_bounds_stay_clear_of_the_global_pytest_timeout`
            # assumes of every consumer.
            assert os.getpid() not in holders, (
                'staging error: this process must NOT hold the lock, or the '
                'negative below would be vacuous'
            )
            assert child.pid in holders, (
                f'the kernel-reported holder pid is the discriminator between '
                f'foreign contention and a self-owned leak; the same pid must '
                f'also survive into _lane_lock_holder_facts\' raise-time '
                f're-read for the str(child.pid) in str(excinfo.value) '
                f'assertion below to hold; observed holders={holders!r}'
            )

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
            # Bounded poll, NOT a one-shot `lane_lock_holder_pids` read: this
            # staging assertion was observed failing `assert 1986528 in []`
            # under 24-worker xdist load — the same transient
            # `wait_for_lane_lock_holder` was introduced to absorb (see its
            # docstring's `assert 658016 in []`), just at a call site that was
            # never migrated onto it.
            #
            # Mechanism, measured on this host: `/proc/locks` is a seq_file the
            # kernel serves one PAGE per read(2) regardless of the caller's
            # buffer (a 13062-byte table took 4 reads even for a 1 MiB
            # request), and each read restarts the per-CPU lock-list walk from
            # a POSITIONAL index — so a lock released at an earlier position
            # between chunks shifts every later record down and ours is skipped
            # outright.  Reproduced at 1.54% of reads (144/9337) against a real
            # held flock with 24 concurrent churners, which is exactly why this
            # is green in isolation and red only in a full parallel suite.
            #
            # Nothing is being waited FOR here: we already hold the fd, so the
            # kernel fact is settled the instant `_acquire_lane_flock_off_thread`
            # returns.  Only a lossy READ of that settled fact is being retried
            # past — hence the procfs-read-class bound
            # (:data:`_LANE_LOCK_STRICT_READ_SECS`) rather than the helper's
            # default 12.0s spawn-latency class: no process is spawned here, and
            # a re-read either succeeds in microseconds or is structurally
            # broken, so a wider bound would only delay a certain failure.
            holders = wait_for_lane_lock_holder(
                lock_path, os.getpid(), timeout=_LANE_LOCK_STRICT_READ_SECS,
            )
            assert os.getpid() in holders, (
                f'staging error: the kernel must see us as a holder, or layer '
                f'(1) would be what returns None here; observed '
                f'holders={holders!r}'
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
        assert await wait_until(lambda: _SENTINEL_FD in obs['released']), (  # type: ignore[operator]
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
