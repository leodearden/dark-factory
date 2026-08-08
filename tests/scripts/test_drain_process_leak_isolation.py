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
import shutil
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

# Reused, not re-derived. pytest's fixture marker is private and MOVED between
# 8.x and 9.x; `_fixture_marker` already accepts both spellings and fails loudly
# when it finds neither, which is the whole value of the helper — a second copy
# would be a second thing to update the next time pytest moves it. Resolves
# because `tests/scripts/conftest.py` puts this directory on sys.path (pytest's
# --import-mode=importlib deliberately does not), the same mechanism
# `test_skills_module_config_decision.py` uses for its sibling probe. A bare
# import, never an importorskip: if the sibling stops importing, say so loudly.
from test_deploy_clock_isolation import _fixture_marker  # noqa: E402

import df_pytest_isolation  # noqa: E402
from df_pytest_isolation import (  # noqa: E402
    DRAIN_SCRIPT_CMDLINE_MARKER,
    LEAK_SELF_TERMINATION_CEILING_SECS,
    LEAK_TOKEN_ENV,
    WAIT_PROOF_GRACE_FLOOR_SECS,
    WAIT_PROOF_GRACE_MULTIPLIER,
    leaked_drain_process_reason,
    leaked_drain_processes,
    run_in_new_session,
    wait_proof_grace_secs,
)

# NOT `from df_pytest_isolation import _df_no_leaked_drain_processes`. Importing
# a FIXTURE into a test module binds it as a module-scoped fixture that SHADOWS
# the conftest's, so the registration test below would resolve its own import
# and pass with the conftest wiring deleted — precisely the dead defence it
# exists to detect. Reach it through the module instead.
_GUARD_NAME = '_df_no_leaked_drain_processes'

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


# ---------------------------------------------------------------------------
# The in-suite guard's pure core.
#
# Driven against SYNTHETIC /proc trees under tmp_path, so the scan logic is
# testable without leaking a single real process.
# ---------------------------------------------------------------------------

# Built FROM the module's own marker, not from a hardcoded copy of it: that is
# what makes these tests pin "the scan finds what the module says it looks for"
# rather than "the scan finds this string I happened to type twice".
_DRAIN_CMDLINE = (
    b'bash\x00/repo/scripts/' + DRAIN_SCRIPT_CMDLINE_MARKER.encode() + b'\x00--drain\x00'
)
_OTHER_CMDLINE = b'bash\x00/repo/scripts/some-other-script.sh\x00'


def _proc_entry(
    proc_root: Path, pid: int, cmdline: bytes, environ: bytes | None,
) -> None:
    """Write one synthetic ``/proc/<pid>`` entry.

    *environ* of ``None`` writes NO environ file at all — the shape of a pid
    that exited between the directory listing and the read.
    """
    entry = proc_root / str(pid)
    entry.mkdir(parents=True)
    (entry / 'cmdline').write_bytes(cmdline)
    if environ is not None:
        (entry / 'environ').write_bytes(environ)


def _environ_with(token: str, *, extra: bytes = b'PATH=/usr/bin\x00') -> bytes:
    """A NUL-separated environ block carrying *token*, as /proc renders it."""
    return extra + LEAK_TOKEN_ENV.encode() + b'=' + token.encode() + b'\x00'


class TestLeakedDrainProcesses:
    """Which surviving processes count as THIS session's leak."""

    def test_a_process_matching_both_marker_and_token_is_reported(
        self, tmp_path: Path,
    ) -> None:
        _proc_entry(tmp_path, 4242, _DRAIN_CMDLINE, _environ_with('tok'))

        leaks = leaked_drain_processes('tok', proc_root=tmp_path)

        assert [pid for pid, _ in leaks] == [4242]
        assert DRAIN_SCRIPT_CMDLINE_MARKER in leaks[0][1]

    def test_the_drain_marker_alone_is_not_enough(self, tmp_path: Path) -> None:
        """A genuine or concurrent --drain must NEVER fail this suite.

        `merge_verify_breadth: "full"` runs this suite in many worktrees at
        once, and the main checkout is machine-operated (CLAUDE.md) — the
        deployed watchdog and the merge coordinator both run --drain for real.
        A bare `pgrep -f restart-all-orchestrators.sh` would fail those runs
        spuriously, which is why the token is required as well.
        """
        _proc_entry(tmp_path, 11, _DRAIN_CMDLINE, b'PATH=/usr/bin\x00')
        _proc_entry(tmp_path, 12, _DRAIN_CMDLINE, _environ_with('another-session'))

        assert leaked_drain_processes('tok', proc_root=tmp_path) == []

    def test_a_token_that_merely_prefixes_ours_is_not_a_match(
        self, tmp_path: Path,
    ) -> None:
        """Matched as a whole NUL-delimited assignment, not a substring.

        Without the delimiters, session `tok` would match session `tokZZZ`'s
        processes and fail a run for another worktree's leak.
        """
        _proc_entry(tmp_path, 21, _DRAIN_CMDLINE, _environ_with('tokZZZ'))

        assert leaked_drain_processes('tok', proc_root=tmp_path) == []

    def test_the_token_alone_is_not_enough(self, tmp_path: Path) -> None:
        """These two directories hold ~90 test files that spawn subprocesses.

        Scanning for "any surviving descendant" rather than the drain script
        would flake constantly on all of them.
        """
        _proc_entry(tmp_path, 31, _OTHER_CMDLINE, _environ_with('tok'))

        assert leaked_drain_processes('tok', proc_root=tmp_path) == []

    def test_a_vanished_pid_does_not_raise(self, tmp_path: Path) -> None:
        """Scanning /proc races process exit by construction."""
        _proc_entry(tmp_path, 41, _DRAIN_CMDLINE, None)  # no environ file
        _proc_entry(tmp_path, 42, _DRAIN_CMDLINE, _environ_with('tok'))

        assert [pid for pid, _ in leaked_drain_processes('tok', proc_root=tmp_path)] == [42]

    def test_non_numeric_proc_entries_are_skipped(self, tmp_path: Path) -> None:
        """Real /proc holds `self`, `meminfo`, `net`, ... — none of them pids."""
        (tmp_path / 'meminfo').write_bytes(b'MemTotal: 1 kB\n')
        (tmp_path / 'self').mkdir()
        _proc_entry(tmp_path, 51, _DRAIN_CMDLINE, _environ_with('tok'))

        assert [pid for pid, _ in leaked_drain_processes('tok', proc_root=tmp_path)] == [51]

    def test_an_empty_or_missing_token_matches_nothing(self, tmp_path: Path) -> None:
        """A falsy token must fail CLOSED on the match, never open.

        Treating it as "match everything" would turn the guard into a
        suite-wide false-positive generator the first time the fixture failed
        to stamp its token.
        """
        _proc_entry(tmp_path, 61, _DRAIN_CMDLINE, _environ_with('tok'))

        assert leaked_drain_processes('', proc_root=tmp_path) == []
        assert leaked_drain_processes(None, proc_root=tmp_path) == []

    def test_the_real_proc_tree_is_scannable(self) -> None:
        """The default proc_root works on this box.

        The task's own evidence — every sampled orphan identified by its
        `PYTEST_CURRENT_TEST` environ value — already proves environ scanning is
        permitted here; this pins it so a hardened kernel would fail loudly
        rather than silently reporting all-clear forever.
        """
        leaks = leaked_drain_processes('a-token-no-process-can-carry-3798')

        assert isinstance(leaks, list)
        assert leaks == []


class TestLeakedDrainProcessReason:
    """The failure message has to be actionable on its own."""

    def test_no_leaks_is_not_a_violation(self) -> None:
        assert leaked_drain_process_reason([]) is None

    def test_it_names_every_pid_its_cmdline_and_the_remedy(self) -> None:
        """Mirrors deploy_clock_violation_reason's "name the remedy" convention.

        Asserted on the pid, the cmdline and the remedy SYMBOL, never on prose
        wording — the message is triage text and must stay editable.
        """
        reason = leaked_drain_process_reason([
            (4242, f'bash /repo/scripts/{DRAIN_SCRIPT_CMDLINE_MARKER} --drain'),
            (4243, f'bash /repo/scripts/{DRAIN_SCRIPT_CMDLINE_MARKER} --drain'),
        ])

        assert reason is not None
        assert '4242' in reason and '4243' in reason
        assert DRAIN_SCRIPT_CMDLINE_MARKER in reason
        assert '3798' in reason
        assert 'run_in_new_session' in reason


class TestGuardIsLiveInThisRun:
    """The fixture is WIRED, not merely defined.

    Everything above tests pure functions against synthetic ``/proc`` trees and
    ``tmp_path`` leakers; all of it would stay green if the fixture were never
    loaded by any conftest.  This class is the only assertion that the defence
    is actually armed in the process running it — the difference between a wired
    defence and a dead one.
    """

    def test_the_guard_fixture_exists(self) -> None:
        assert hasattr(df_pytest_isolation, _GUARD_NAME), (
            f'df_pytest_isolation defines no {_GUARD_NAME}; the pure scan '
            'helpers above protect nothing on their own.'
        )

    def test_the_guard_fixture_is_session_scoped_and_autouse(self) -> None:
        """Both properties pinned STRUCTURALLY, not by inspection of behaviour.

        A function-scoped or non-autouse variant would still import cleanly and
        keep every test above green while protecting nothing: without
        ``autouse`` nothing would ever request it, and function scope would both
        cost a /proc sweep per test across ~90 files times every xdist worker
        AND miss a leak from a module- or session-scoped fixture, which is
        exactly where expensive subprocess setup tends to live.
        """
        marker = _fixture_marker(getattr(df_pytest_isolation, _GUARD_NAME))

        assert marker.scope == 'session', f'scope is {marker.scope!r}, expected session'
        assert marker.autouse is True, 'the guard must be autouse — nothing requests it'

    def test_the_guard_fixture_is_registered_in_this_run(self, request) -> None:
        """The conftest binding is real WIRING, not a dormant definition.

        pytest only collects fixtures bound into a conftest's namespace, which
        is why df_pytest_isolation's fixtures are imported there under
        ``# noqa: F401 — the binding IS the wiring``. Deleting that import
        breaks nothing visible except this assertion.
        """
        try:
            request.getfixturevalue(_GUARD_NAME)
        except pytest.FixtureLookupError:
            pytest.fail(
                f'{_GUARD_NAME} is not registered for this rootdir. Wire the '
                'test-root conftest to import it from df_pytest_isolation '
                '(`# noqa: F401 — the binding IS the wiring`); without that, '
                'this whole suite runs with no leaked-drain-process guard.',
                pytrace=False,
            )

    def test_the_session_token_is_stamped_into_the_environment(self) -> None:
        """The token is in ``os.environ`` NOW, which is what makes the guard free.

        Both real spawners build their child env from ``dict(os.environ)``, so a
        token stamped there tags every descendant with no per-spawner change —
        the same "the defect class is a spawner that forgets" reasoning as
        3797's clock redirect. Stamped anywhere else (a fixture argument, an
        explicit env key at each call site) the next spawner would silently opt
        out, and `leaked_drain_processes` fails CLOSED on an absent token, so
        the guard would report all-clear forever.
        """
        token = os.environ.get(LEAK_TOKEN_ENV)

        assert token, (
            f'{LEAK_TOKEN_ENV} is unset in this run, so nothing this suite '
            'spawns is attributable to it and the guard can never match.'
        )


# ---------------------------------------------------------------------------
# The guard's FAILURE contract, exercised through a real nested pytest run.
#
# Everything above pins the helpers, the marker and the registration; none of it
# pins what a leak actually DOES to the run. A refactor that warned instead of
# failing, or moved the check into a fixture nobody requests, would leave every
# test in this file green while the defence emitted nothing an exit code could
# carry. Only a nested run can observe "the process exits non-zero even though
# every test passed", because a fixture cannot fail its own session.
# ---------------------------------------------------------------------------

# Minimal ini so the nested run's rootdir is the tmp tree and NOT this repo:
# without it pytest walks up looking for an inifile and would inherit this
# repo's addopts (`--import-mode=importlib -m 'not smoke ...'`).
_NESTED_INI = '[pytest]\n'

_NESTED_CONFTEST = '''\
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from df_pytest_isolation import _df_no_leaked_drain_processes  # noqa: F401
'''

# NAMED for the marker and nothing else. The guard matches this basename in
# /proc/<pid>/cmdline, and a test that reached for the REAL script to earn that
# match would be executing production fleet-restart code — spawning systemctl
# against live units — from inside a unit test. This is a throwaway file the
# test writes into tmp_path that merely happens to share the name.
_LEAKER_NAME = 'restart-all-orchestrators.sh'
_LEAKED_PIDFILE_NAME = 'leaked.pid'

# 45s: long enough to still be alive at the nested session's teardown (~2-4s)
# by a wide margin, short enough that a residue from a BROKEN guard expires on
# its own rather than sitting on the box. The trailing statement keeps bash
# from exec-ing the sleep in its own process and taking the marker with it —
# measured while implementing 3798 (a one-line `sleep` script keeps `bash
# <path>` as its cmdline on this box), and cheap insurance across bash versions.
_NESTED_LEAKER = '''\
sleep 45
echo unreachable-inside-this-test
'''


def _nested_test_source(*, leaks: bool) -> str:
    """Source for the nested test module — which PASSES either way.

    The spawn mirrors both real spawners: ``dict(os.environ)`` for the child
    env, which is what hands it the NESTED session's token. That is also what
    keeps THIS session's guard blind to it — a different token — so the outer
    run cannot fail on the leak its own test deliberately created.
    """
    body = (
        (
            '    proc = subprocess.Popen(\n'
            '        ["bash", str(SCRIPT)],\n'
            '        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,\n'
            '        start_new_session=True, env=dict(os.environ),\n'
            '    )\n'
            '    PIDFILE.write_text(str(proc.pid))\n'
        )
        if leaks
        else '    pass\n'
    )
    return (
        'import os\n'
        'import subprocess\n'
        'from pathlib import Path\n'
        '\n'
        'HERE = Path(__file__).resolve().parent\n'
        f'SCRIPT = HERE / {_LEAKER_NAME!r}\n'
        f'PIDFILE = HERE / {_LEAKED_PIDFILE_NAME!r}\n'
        '\n'
        '\n'
        'def test_a_forgetful_spawner():\n'
        '    """PASSES. The damage is the surviving process, not this result."""\n'
        + body
    )


def _nested_run(root: Path, *, leaks: bool) -> subprocess.CompletedProcess[str]:
    """Run a throwaway pytest session wired to the guard, in its own tmp tree.

    *root* is passed in rather than derived so the caller knows the pidfile path
    BEFORE the run starts, and can therefore reap a leaker even when this call
    raises.

    The 120s cap is under pytest-timeout's own ``--timeout=300`` per-test axe, so
    a wedged nested run reports as this test's failure rather than as the whole
    outer session being killed mid-assertion.
    """
    root.mkdir(parents=True)
    shutil.copy2(Path(df_pytest_isolation.__file__), root / 'df_pytest_isolation.py')
    (root / 'pytest.ini').write_text(_NESTED_INI)
    (root / 'conftest.py').write_text(_NESTED_CONFTEST)
    (root / _LEAKER_NAME).write_text(_NESTED_LEAKER)
    (root / 'test_forgetful.py').write_text(_nested_test_source(leaks=leaks))
    return subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', str(root)],
        cwd=root, capture_output=True, text=True, timeout=120,
    )


def _read_pid_file(pidfile: Path) -> int | None:
    """The pid the nested test recorded, or ``None`` if it never got that far."""
    try:
        text = pidfile.read_text().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def _reap_leaker(pid: int | None) -> None:
    """SIGKILL a nested run's leaker — group and all — after proving it is ours.

    The identity check is not ceremony. ``os.killpg`` aimed at a pid the kernel
    has already RECYCLED is task 845's failure, where it resolved to the
    ``systemd --user`` manager's group and killed the user's entire login
    session (see ``shared/src/shared/proc_group.py``). The nested pytest process
    has already exited by the time we read its pidfile, so this pid is exactly
    as stale as that one was. Matching the marker in ``/proc/<pid>/cmdline``
    first means a recycled pid is skipped rather than signalled.

    The GROUP, not just the pid, because the leaker's ``sleep`` is a child: the
    guard reports only the marker-carrying bash, so killing that alone would
    leave the sleep orphaned for the rest of its 45s.
    """
    if pid is None:
        return
    try:
        cmdline = (Path('/proc') / str(pid) / 'cmdline').read_bytes()
    except OSError:
        return  # already gone — the guard reaps what it reports
    if DRAIN_SCRIPT_CMDLINE_MARKER.encode() not in cmdline:
        return  # a recycled pid: never signal a group we cannot identify
    with contextlib.suppress(OSError):
        os.killpg(pid, signal.SIGKILL)
    _reap(pid)


class TestTheGuardFailsTheRunEndToEnd:
    """A leak must cost the RUN, not merely log something."""

    def test_a_leaked_drain_process_fails_a_session_whose_tests_all_passed(
        self, tmp_path: Path,
    ) -> None:
        root = tmp_path / 'leaking'
        pidfile = root / _LEAKED_PIDFILE_NAME
        try:
            result = _nested_run(root, leaks=True)
            combined = result.stdout + result.stderr
            leaked_pid = _read_pid_file(pidfile)

            assert result.returncode != 0, (
                'a run that leaked a drain process exited 0 — the guard is '
                f'inert. stdout={result.stdout!r}'
            )
            assert '1 passed' in combined, (
                'the nested TEST must still pass, or this proves nothing about '
                f'a green suite being caught. output={combined!r}'
            )
            assert leaked_pid is not None, (
                f'the nested test recorded no pid in {pidfile}; the harness is '
                'broken, which says nothing either way about the guard.'
            )
            assert str(leaked_pid) in combined, (
                f'the guard did not name the leaked pid {leaked_pid}. Triage '
                f'has to be possible from the failure output alone: {combined!r}'
            )
            assert 'run_in_new_session' in combined, combined
        finally:
            _reap_leaker(_read_pid_file(pidfile))

    def test_the_same_harness_without_the_leak_exits_zero(self, tmp_path: Path) -> None:
        """Non-vacuity control: the identical nested tree, minus the spawn.

        Without it the failure above could be any nested-harness breakage — a
        bad ini, an unimportable module, a missing pytest.
        """
        result = _nested_run(tmp_path / 'clean', leaks=False)

        assert result.returncode == 0, (
            f'the control run failed for an unrelated reason: '
            f'stdout={result.stdout!r} stderr={result.stderr!r}'
        )
        assert 'LEAKED' not in result.stdout
