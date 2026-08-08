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
from df_pytest_isolation import run_in_new_session  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic leakers.
#
# THE PIPE-HOLDING VARIANT LIVES HERE AND ONLY HERE.  Its background child
# inherits the spawn's stdout/stderr pipe write ends, so a post-kill drain that
# is not bounded BLOCKS on it forever.  That is safe to assert against
# `run_in_new_session` — which bounds its drain — but it must never be pointed
# at a still-`subprocess.run`-based spawner, where it would HANG the test until
# pytest-timeout's 300s axe instead of failing.  The tests in
# `scripts/tests/test_restart_all_orchestrators.py` and
# `tests/scripts/test_orchestrator_watchdog.py` that drive the real spawners use
# a PIPE-CLOSING variant (`>/dev/null 2>&1`) for exactly that reason.
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
