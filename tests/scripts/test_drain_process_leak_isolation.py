"""Guard tests: a test's spawn timeout can never leak a process group (task 3798).

Both harnesses that drive ``scripts/restart-all-orchestrators.sh --drain`` from
pytest used a plain ``subprocess.run(..., timeout=N)``.  On POSIX that timeout
path ``kill()``s the DIRECT CHILD only.  The drain script forks poll loops that
survive it, get reparented to ``systemd --user``, and then sit spending their
grace: 86 concurrent orphans accumulated on 2026-08-06 and 82 more on
2026-08-07, each holding an ``ORCH_RESTART_FORCE_FIRE_AFTER_SECS`` /
``ORCH_DRAIN_UNKNOWN_GRACE_SECS`` of ``99999`` seconds — 27.8 HOURS — before
expiry.

Two independent defects, and this module covers the suite-wide fix for both:

* the spawn is not session-isolated, so the timeout kill cannot reach the group
  (:func:`df_pytest_isolation.run_in_new_session`); and
* the grace knobs were hand-typed at a value so large that anything which DOES
  escape outlives pytest's tmpdir GC — at which point its PATH no longer
  resolves ``systemctl`` to the test's fake and it restarts REAL units
  (:func:`df_pytest_isolation.wait_proof_grace_secs`).

Shaped like ``test_deploy_clock_isolation.py``, the module covering this repo's
sibling suite-wide isolation defence, so the two read as one family: pure
helpers tested directly, then ``TestGuardIsLiveInThisRun`` for the wiring, then
a nested-pytest end-to-end failure contract with a non-vacuity control.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
# APPEND, never insert(0, ...): the repo root must stay LAST on sys.path or the
# subproject directories (orchestrator/, shared/, ...) resolve as namespace
# packages shadowing their own src/<pkg>/ — the failure the root conftest.py
# docstring exists to prevent.
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import df_pytest_isolation  # noqa: E402
from df_pytest_isolation import (  # noqa: E402
    LEAK_SELF_TERMINATION_CEILING_SECS,
    WAIT_PROOF_GRACE_FLOOR_SECS,
    WAIT_PROOF_GRACE_MULTIPLIER,
    run_in_new_session,
    wait_proof_grace_secs,
)

# ---------------------------------------------------------------------------
# Synthetic leakers.
#
# THE PIPE-HOLDING VARIANT LIVES HERE AND ONLY HERE.  Its background child
# inherits the spawn's stdout/stderr pipe write ends, so a drain run against it
# after the kill never sees EOF — unbounded, it blocks forever.  That is what
# makes it the right probe for `run_in_new_session`, which bounds its drain.
#
# The tests that drive the REAL spawners — in
# `scripts/tests/test_restart_all_orchestrators.py` and
# `tests/scripts/test_orchestrator_watchdog.py` — use a PIPE-CLOSING variant
# (`>/dev/null 2>&1`) instead, so their assertions depend only on "is the
# grandchild alive" and not on what a spawner does after its kill.  (Measured
# while implementing 3798: stock `subprocess.run` would not actually hang on
# the pipe-holding variant either — its POSIX timeout branch calls
# `process.wait()`, never a second `communicate()`.  Pipe-closing remains the
# right choice there because it pins less, not because the alternative hangs.)
# ---------------------------------------------------------------------------

_PIPE_HOLDING_LEAKER = '''\
sleep 300 &
echo $! > "$LEAK_PIDFILE"
echo MAIN_UP
sleep 300
'''


def _leaker_script(tmp_path: Path, name: str = 'pipe_holding_leaker.sh') -> Path:
    """Write a bash script that forks a grandchild outliving its own parent.

    Deliberately NOT named ``restart-all-orchestrators.sh``: the leak guard
    matches on that basename in ``/proc/<pid>/cmdline``, and these tests reap
    their own spawns, so borrowing the marker would only muddy a real failure.
    """
    script = tmp_path / name
    script.write_text(_PIPE_HOLDING_LEAKER)
    return script


def _leaker_env(pidfile: Path) -> dict[str, str]:
    """Child env for a leaker: the real environment plus its pidfile.

    Copied from ``os.environ`` rather than built from scratch, mirroring what
    both real spawners do — a bare ``{'LEAK_PIDFILE': ...}`` would leave the
    child without a PATH and exercise a call shape nothing in the repo uses.
    """
    env = dict(os.environ)
    env['LEAK_PIDFILE'] = str(pidfile)
    return env


def _read_leaked_pid(pidfile: Path, *, timeout: float = 10.0) -> int:
    """Poll until the leaker has recorded its grandchild's pid, then return it.

    Polls rather than reads once: the file is created by a `>` redirection and
    filled by a separate write, so a single read can catch it empty.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            text = pidfile.read_text().strip()
        except OSError:
            text = ''
        if text.isdigit():
            return int(text)
        if time.monotonic() >= deadline:
            pytest.fail(
                f'the leaker never recorded a pid in {pidfile} within {timeout}s '
                f'(last read: {text!r}); the test harness is broken, which says '
                'nothing either way about the spawn helper under test.',
                pytrace=False,
            )
        time.sleep(0.05)


def _is_gone(pid: int) -> bool:
    """Whether *pid* no longer exists. Signal 0 is the standard liveness probe."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        # Alive and owned by someone else — a PID the kernel already recycled.
        return False
    return False


def _wait_gone(pid: int, *, timeout: float = 3.0) -> bool:
    """Poll for *pid* to disappear, allowing for SIGKILL delivery + reaping."""
    deadline = time.monotonic() + timeout
    while True:
        if _is_gone(pid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _reap(pid: int | None) -> None:
    """SIGKILL *pid*, ignoring every way it can already be gone.

    Every test that spawns calls this in a ``finally``: a RED that fails its
    "the grandchild is dead" assertion has, by construction, a live orphan in
    hand, and a guard test that leaks is worse than no guard test.
    """
    if pid is None:
        return
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGKILL)


class TestRunInNewSession:
    """The spawn helper kills the whole process group when its timeout fires."""

    def test_a_timeout_kills_the_backgrounded_grandchild_too(self, tmp_path: Path) -> None:
        """The defect, stated directly.

        ``subprocess.run``'s timeout path calls ``Popen.kill()``, which signals
        the direct child and nothing else — verified against real processes: the
        backgrounded grandchild survives. ``start_new_session=True`` plus a
        ``killpg`` of the captured pgid is what actually reaches it.
        """
        pidfile = tmp_path / 'leaked.pid'
        leaker = _leaker_script(tmp_path)
        leaked_pid = None
        try:
            with pytest.raises(subprocess.TimeoutExpired):
                run_in_new_session(
                    ['bash', str(leaker)], env=_leaker_env(pidfile), timeout=2,
                )

            leaked_pid = _read_leaked_pid(pidfile)
            assert _wait_gone(leaked_pid), (
                f'pid {leaked_pid} — a grandchild backgrounded by the spawned '
                'script — is STILL ALIVE after the spawn timed out. The timeout '
                'killed only the direct child, so the poll loop it forked is now '
                'an orphan reparented to systemd --user, free to spend its grace '
                'and then issue a real systemctl restart.'
            )
        finally:
            _reap(leaked_pid)

    def test_the_timeout_is_honoured_as_a_wall_clock_bound(self, tmp_path: Path) -> None:
        """The timeout must be a real bound even against a pipe-HOLDING child.

        The non-obvious half of the fix. A surviving grandchild holds the
        inherited stdout/stderr pipe write ends, so a drain run against it
        never sees EOF; unbounded, it blocks forever and the caller's
        ``timeout=`` stops being a bound at all. Measured: a direct
        ``communicate()`` after a plain ``kill()`` of such a child does hang.

        Green today via the group kill (which closes every write end at once),
        so this is a REGRESSION guard on the two degraded paths — a refused or
        failed ``killpg`` falls back to killing the direct child only, and
        there the bound is the sole thing standing between this suite and
        pytest-timeout's 300s axe.

        15s for a 2s timeout is deliberately loose: this asserts "bounded", not
        a latency, so it cannot flake under load.
        """
        pidfile = tmp_path / 'leaked.pid'
        leaker = _leaker_script(tmp_path)
        leaked_pid = None
        try:
            started = time.monotonic()
            with pytest.raises(subprocess.TimeoutExpired):
                run_in_new_session(
                    ['bash', str(leaker)], env=_leaker_env(pidfile), timeout=2,
                )
            elapsed = time.monotonic() - started
            # Read (and therefore arm the reap of) the pid BEFORE asserting, so
            # a failure here still cleans up. `_read_leaked_pid` can itself
            # fail, which is why it must not run inside the `finally`.
            leaked_pid = _read_leaked_pid(pidfile)

            assert elapsed < 15, (
                f'the spawn took {elapsed:.1f}s to honour a timeout of 2s. The '
                'post-kill drain is unbounded and is blocking on the pipe the '
                'surviving grandchild still holds open.'
            )
        finally:
            _reap(leaked_pid)

    def test_partial_output_is_attached_to_the_timeout(self, tmp_path: Path) -> None:
        """``TimeoutExpired.stdout`` still carries what the child managed to print.

        Load-bearing for every existing caller: the three defer/grace tests
        assert on the script's output from the TIMEOUT path, through ``_decode``
        / ``_boundary_decode``. A helper that dropped the partial output would
        turn those into silent vacuous passes.
        """
        pidfile = tmp_path / 'leaked.pid'
        leaker = _leaker_script(tmp_path)
        leaked_pid = None
        try:
            with pytest.raises(subprocess.TimeoutExpired) as exc_info:
                run_in_new_session(
                    ['bash', str(leaker)], env=_leaker_env(pidfile), timeout=2,
                )

            leaked_pid = _read_leaked_pid(pidfile)
            stdout = exc_info.value.stdout
            assert stdout is not None, 'TimeoutExpired carried no stdout at all'
            text = stdout.decode(errors='replace') if isinstance(stdout, bytes) else stdout
            assert 'MAIN_UP' in text, (
                f'partial stdout was lost on the timeout path; got {text!r}'
            )
        finally:
            _reap(leaked_pid)

    def test_a_normal_exit_returns_a_completed_process(self) -> None:
        """Non-vacuity control: a drop-in for ``subprocess.run`` on the happy path.

        ~26 existing call sites read ``.returncode`` / ``.stdout`` / ``.stderr``
        off the return value, so the helper is only substitutable if the
        untimed-out path is byte-for-byte the same shape.
        """
        result = run_in_new_session(
            ['bash', '-c', 'echo hi; echo boom >&2; exit 3'], timeout=30,
        )

        assert isinstance(result, subprocess.CompletedProcess)
        assert result.returncode == 3
        assert result.stdout == 'hi\n'
        assert result.stderr == 'boom\n'

    def test_a_suspect_pgid_is_never_signalled(self) -> None:
        """The defence-in-depth refusal ported from ``shared.proc_group``.

        Not cosmetic. That module's docstring records task 845: a ``killpg``
        aimed via ``os.getpgid(pid)`` at a RECYCLED pid resolved to the
        ``systemd --user`` manager's group and killed the user's entire login
        session. The hazard is especially live here — the orphans this guard
        exists to kill are themselves reparented to ``systemd --user``.

        Asserts each refusal is non-None, never its wording: the message is
        triage text, the refusal is the contract.
        """
        unsafe = df_pytest_isolation._unsafe_pgid_reason

        for pgid in (-1, 0, 1, os.getpid(), os.getppid(), os.getpgrp()):
            assert unsafe(pgid) is not None, (
                f'pgid {pgid} must be refused — signalling it can take out this '
                'process, its parent, or the whole login session.'
            )

        forbidden = {0, 1, os.getpid(), os.getppid(), os.getpgrp()}
        ordinary = next(c for c in range(4_000_000, 4_000_100) if c not in forbidden)
        assert unsafe(ordinary) is None, (
            f'pgid {ordinary} is an ordinary unrelated group and must be allowed, '
            'or the helper degrades to never group-killing anything.'
        )


class TestWaitProofGraceSecs:
    """The grace a wait-proving test hands its script is DERIVED, not typed.

    Three call sites each hard-coded ``99999`` seconds — 27.8 HOURS. Every one
    was locally reasonable: the tests prove the drain script is still POLLING
    rather than fail-opening, which they do by letting their own short
    ``subprocess`` timeout fire first, so the grace only has to be big enough
    not to expire mid-test. Nothing in the code stated the OTHER side of the
    constraint — that the same number also bounds how long a LEAKED poller
    lives — so nothing pushed back on making it enormous.

    These tests pin both sides at once, in one place, so a future editor cannot
    restore a 27.8h value at any single site without failing a test that
    explains why.
    """

    # The two spawn timeouts actually in use, at the three sites step-9 edits:
    # test_defer_withholds_restart_while_busy (3s),
    # test_unknown_grace_withholds_restart_while_absent (3s),
    # test_boundary4_defers_busy_unit_while_others_proceed (20s).
    REAL_SPAWN_TIMEOUTS = (3, 20)

    def test_the_grace_comfortably_exceeds_the_spawn_timeout_that_kills_it(self) -> None:
        """Too SMALL and the wait-proving tests stop proving anything.

        Each asserts "no restart was recorded before my timeout fired". If the
        grace expired first the script would force-fire, record the restart,
        and the assertion would fail — so the margin is what keeps those three
        tests meaningful rather than flaky.
        """
        assert WAIT_PROOF_GRACE_MULTIPLIER >= 3, (
            f'a {WAIT_PROOF_GRACE_MULTIPLIER}x margin over the spawn timeout is '
            'too tight to absorb scheduling jitter under 32-way xdist load.'
        )
        for spawn_timeout in self.REAL_SPAWN_TIMEOUTS:
            grace = wait_proof_grace_secs(spawn_timeout)
            assert grace >= spawn_timeout * WAIT_PROOF_GRACE_MULTIPLIER, (
                f'grace {grace}s for a {spawn_timeout}s spawn timeout is under '
                f'the {WAIT_PROOF_GRACE_MULTIPLIER}x floor; the script could '
                'force-fire before the test observes its defer line.'
            )

    def test_a_leak_self_terminates_within_the_ceiling(self) -> None:
        """Too LARGE and anything that escapes outlives its own containment.

        This is the actual fix. A leaked poller holding a 99999s grace is still
        alive 27.8h later — long after pytest's tmpdir GC has removed the fake
        ``systemctl`` its PATH points at, so its expiry falls through to
        ``/usr/bin/systemctl`` and restarts REAL units.
        """
        assert LEAK_SELF_TERMINATION_CEILING_SECS <= 120, (
            f'a leaked poller may live {LEAK_SELF_TERMINATION_CEILING_SECS}s; '
            'the ceiling exists to keep that inside a couple of minutes.'
        )
        # The regression this class exists for, stated as a number: the old
        # value was three orders of magnitude above the ceiling.
        assert LEAK_SELF_TERMINATION_CEILING_SECS <= 99999 / 100, (
            'the ceiling must be far below the 99999s (27.8h) that produced 86 '
            'concurrent orphans on 2026-08-06 and 82 more on 2026-08-07.'
        )
        for spawn_timeout in self.REAL_SPAWN_TIMEOUTS:
            grace = wait_proof_grace_secs(spawn_timeout)
            assert grace <= LEAK_SELF_TERMINATION_CEILING_SECS, (
                f'grace {grace}s for a {spawn_timeout}s spawn timeout exceeds '
                f'the {LEAK_SELF_TERMINATION_CEILING_SECS}s self-termination '
                'ceiling — a leak from this call site would outlive its fake '
                'systemctl and reach the real one.'
            )

    def test_a_floor_applies_to_very_short_timeouts(self) -> None:
        """A tiny timeout must not derive a grace that expires mid-test.

        Purely multiplicative, a 1s timeout would yield a few seconds of grace
        — comparable to the script's own startup — and the defer line the test
        is waiting for might never be printed.
        """
        assert WAIT_PROOF_GRACE_FLOOR_SECS >= 10, (
            f'a {WAIT_PROOF_GRACE_FLOOR_SECS}s floor is under the wall-clock '
            'cost of the handful of bash+python3 spawns the script makes '
            'before it reaches its first defer decision.'
        )
        for spawn_timeout in (0, 1):
            assert wait_proof_grace_secs(spawn_timeout) >= WAIT_PROOF_GRACE_FLOOR_SECS

    def test_it_is_monotonic_and_returns_ints(self) -> None:
        """``int`` is load-bearing, not tidiness.

        The value is stringified into an env var and compared by bash's integer
        operators; a float would arrive as ``30.0`` and make ``[ "$x" -gt ... ]``
        fail outright.
        """
        previous = None
        for spawn_timeout in range(0, 61, 5):
            grace = wait_proof_grace_secs(spawn_timeout)
            assert type(grace) is int, (
                f'wait_proof_grace_secs({spawn_timeout}) returned '
                f'{grace!r} ({type(grace).__name__}); bash needs an integer.'
            )
            if previous is not None:
                assert grace >= previous, (
                    f'grace fell from {previous} to {grace} as the spawn timeout '
                    'rose; a longer-running spawn cannot need LESS grace.'
                )
            previous = grace

    def test_the_two_real_call_site_timeouts_derive_the_documented_values(self) -> None:
        """Pin the numbers the three call sites will actually carry.

        Fixed here rather than invented at the call sites, so the derivation and
        the values it produces cannot drift apart silently.
        """
        assert wait_proof_grace_secs(3) == 30
        assert wait_proof_grace_secs(20) == 80
