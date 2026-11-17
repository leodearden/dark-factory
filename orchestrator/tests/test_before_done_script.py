"""Tests for orchestrator.before_done_script: the before_done subprocess and its teardown."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import threading
import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _held_loop_clock import held_loop_clock
from _script_survivor import SURVIVOR_SCRIPT, is_alive, kill_survivor, read_survivor_pid
from shared.proc_group import SigkillDispatch
from shared.process_identity import Liveness, capture, liveness

from orchestrator.before_done_script import (
    REAP_GRACE_SECS,
    ProcessTeardown,
    ScriptTimeout,
    run_script,
    terminate_process_tree,
)

# ---------------------------------------------------------------------------
# run_script env merge
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRunScriptEnv:
    """run_script merges a partial env dict over os.environ."""

    async def test_partial_env_inherits_path(self, tmp_path: Path):
        """Passing a partial env dict must NOT drop os.environ — PATH must survive.

        Regression guard for the latent footgun where ``create_subprocess_exec``
        received the raw ``before_done['env']`` dict, which completely replaced the
        process environment (no PATH/HOME/XDG_RUNTIME_DIR → most binaries fail to
        resolve).  The fix merges over ``os.environ`` so the deploy script still runs
        in a sane environment while callers can still override individual variables.
        """
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b'', b''))
        mock_proc.returncode = 0

        with patch('asyncio.create_subprocess_exec', return_value=mock_proc) as mock_exec:
            await run_script(
                Path('/usr/bin/true'),
                [],
                cwd=tmp_path,
                env_overrides={'MY_DEPLOY_VAR': 'hello'},
                timeout_secs=5,
                reap_grace_secs=REAP_GRACE_SECS,
            )

        call_kwargs = mock_exec.call_args.kwargs
        passed_env = call_kwargs['env']

        assert passed_env is not None, (
            'env must not be None when before_done.env is non-empty'
        )
        assert 'PATH' in passed_env, (
            f'child must inherit PATH from os.environ when a partial env dict is '
            f'passed; got env keys: {sorted(passed_env)}'
        )
        assert passed_env['PATH'] == os.environ['PATH'], (
            'inherited PATH must match os.environ[PATH]'
        )
        assert passed_env['MY_DEPLOY_VAR'] == 'hello', (
            'custom override from before_done.env must be present in merged env'
        )


# ---------------------------------------------------------------------------
# Task 2090 — Layer A: whole-process-group kill on subprocess timeout
# ---------------------------------------------------------------------------

def _fake_pgid() -> int:
    """A stand-in pgid that ``shared.proc_group.unsafe_pgid_reason`` will never refuse.

    ``terminate_process_tree`` delegates to
    ``shared/src/shared/proc_group.py::sigkill_process_group``, which
    sanity-checks the pgid against this process's own identifiers (task 3884
    amendment), so a hard-coded literal can collide with
    ``os.getpid()``/``os.getppid()``/``os.getpgrp()`` and make the helper
    refuse-and-fall-back instead of dispatching the killpg these tests assert
    on. That is ~1-in-1.4M with this box's ``pid_max=4194304``, but ~1-in-10k
    on the older 32768 default — a real flake, and a baffling one to debug.
    Deriving a provably-distinct value removes the class outright.
    """
    live = {os.getpid(), os.getppid(), os.getpgrp()}
    candidate = 12345
    while candidate in live:
        candidate += 1
    return candidate


def _wait_for_recorded_pid(
    pidfile: Path, abandon: threading.Event, *, within: float = 30.0,
) -> int | None:
    """Block until *pidfile* holds a complete pid line, or return None once *abandon* is set.

    Runs off-loop because the loop clock may be held.
    """
    deadline = time.monotonic() + within
    while not abandon.is_set():
        try:
            text = pidfile.read_text()
        except FileNotFoundError:
            text = ''
        if text.endswith('\n'):
            return int(text)
        if time.monotonic() >= deadline:
            raise AssertionError(
                f'{pidfile} never held a complete pid line within {within}s: '
                f'the script never recorded its grandchild'
            )
        abandon.wait(0.01)
    return None


RunScriptTask = asyncio.Task[tuple[int, str]]


async def _await_recorded_pid(pidfile: Path, run: RunScriptTask) -> int:
    """The pid the script records for its grandchild, or *run*'s own outcome if it ends first."""
    abandon = threading.Event()
    pid_wait = asyncio.ensure_future(
        asyncio.to_thread(_wait_for_recorded_pid, pidfile, abandon),
    )
    try:
        await asyncio.wait({run, pid_wait}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        abandon.set()
    pid = await pid_wait
    if pid is None or run.done():
        outcome = await run
        raise AssertionError(
            f'run_script returned {outcome!r} before the script recorded its grandchild'
        )
    return pid


async def _tear_down_through_the_budget(run: RunScriptTask, release: Callable[[], None]) -> None:
    """Let a still-pending *run* reach its own timeout teardown, which kills the script's group.

    Cancelling it instead would skip that teardown (run_script catches only
    TimeoutError) and leak the script and its grandchild.
    """
    if run.done():
        return
    release()
    with contextlib.suppress(ScriptTimeout):
        await run


@pytest.mark.asyncio
class TestBeforeDoneSubprocessTimeoutHardening:
    """run_script — Layer A: kill the WHOLE process tree on timeout.

    Guards against the inherited-pipe hang behind task 2087's stranding:
    restart-fused-memory.sh --drain spawns grandchildren (systemctl restart,
    curl, journalctl, sleep, the restarted daemon) that inherit the write end
    of the merged stdout pipe.  Killing only the direct child (pre-2090
    behavior) leaves the tree alive and the pipe open forever.

    Task 4065 / 4252 amendment: the timeout branch RAISES ``ScriptTimeout``
    instead of returning ``(1, '<script timed out after Ns>')`` — see that
    class's docstring for why — and that exception carries only MEASURED
    data: the overrun budget it blew, the script's own exit code when the
    teardown found it already exited, and WHICH SIGNAL that teardown actually
    dispatched (``ProcessTeardown``).  No fabricated exit code, no stand-in
    output, and no inferred kill.  The Layer-A teardown below is unchanged and
    must still happen BEFORE the raise, so the grandchild-is-dead assertion
    stays exactly as it was.

    The teardown disposition is asserted in BOTH directions here, because the
    two shapes are what the escalation text has to tell apart: a script still
    running at teardown entry gets a real whole-group SIGKILL (and the
    grandchild-is-dead assertion corroborates it), while a script that already
    exited gets NO signal at all — its pid may have been recycled — so the
    child it spawned is still running when the escalation is filed.  Each of
    the four ``terminate_process_tree`` branch tests below therefore also
    pins the value it RETURNS, since those are the tests that independently
    prove which branch ran.
    """

    async def test_timeout_kills_whole_process_group(self, tmp_path: Path):
        """On timeout, a backgrounded grandchild must be killed too, not just
        the direct child — and the timeout must surface as ``ScriptTimeout``.

        The raise carries only MEASURED structured data (task 4252): the
        overrun budget, and an ``exit_code`` that is ``None`` here because
        this script's own process was still running when the kill fired.
        (The exited-script case, where a surviving child holds the pipe open
        and a real exit code IS in hand, is pinned by
        ``test_deterministic_runner.py::TestDefaultRunnerInnerTimeoutDeployHonesty``.)

        The 1s budget counts from the recorded, live grandchild, not from the
        spawn: the loop clock is held until then (``_held_loop_clock``), so a
        host stall before the record cannot expire it (task 9612).
        """
        script = tmp_path / 'hang.sh'
        pidfile = tmp_path / 'grandchild.pid'
        script.write_text(
            '#!/bin/sh\n'
            'sleep 60 &\n'
            'echo $! > "$1"\n'
            'sleep 60\n'
        )
        script.chmod(0o755)

        with held_loop_clock() as release:
            # Hang tripwire: if the fix regresses into a real hang, fail loudly
            # instead of stalling the suite.
            run = asyncio.ensure_future(
                asyncio.wait_for(
                    run_script(
                        script,
                        [str(pidfile)],
                        cwd=tmp_path,
                        env_overrides=None,
                        timeout_secs=1,
                        reap_grace_secs=REAP_GRACE_SECS,
                    ),
                    timeout=10,
                ),
            )
            try:
                grandchild_pid = await _await_recorded_pid(pidfile, run)
                grandchild = capture(grandchild_pid)
                assert grandchild is not None, (
                    f'the precondition is a live, recorded grandchild when the budget '
                    f'starts, but recorded pid {grandchild_pid} is not running'
                )
                assert os.getpgid(grandchild_pid) != os.getpgrp(), (
                    f'the precondition is a grandchild in the script\'s own process '
                    f'group, but pid {grandchild_pid} is in pytest\'s group {os.getpgrp()}'
                )
                release()
                with pytest.raises(ScriptTimeout) as excinfo:
                    await run
            finally:
                await _tear_down_through_the_budget(run, release)

        exc = excinfo.value
        assert exc.timeout_secs == 1, (
            f'ScriptTimeout must carry the budget it overran, got {exc.timeout_secs!r}'
        )
        assert exc.exit_code is None, (
            f'the direct child was still running when the timeout fired, so '
            f'it produced no exit code — the teardown signal that killed it '
            f'(a negative returncode) is not one, and must normalize to None: '
            f'{exc.exit_code!r}'
        )
        assert exc.teardown is ProcessTeardown.GROUP_KILLED, (
            f'the script was alive at teardown entry, so the whole-group '
            f'SIGKILL really was dispatched — and the grandchild-is-dead '
            f'assertion below is the INDEPENDENT corroboration that this '
            f'disposition is honest, so the two agree by construction rather '
            f'than by assumption (task 4252): {exc.teardown!r}'
        )
        assert not hasattr(exc, 'rc'), (
            f'a SIGKILLed script produced NO exit code, so the exception must '
            f'not carry a fabricated one for a caller to re-report to a human '
            f'(task 4252), got rc={getattr(exc, "rc", None)!r}'
        )
        assert not hasattr(exc, 'tail'), (
            f'no output was captured either — communicate() was cancelled by '
            f'the timeout — so the old marker string was a placeholder, not '
            f'script output (task 4252), got tail='
            f'{getattr(exc, "tail", None)!r}'
        )
        assert 'no exit code' in str(exc), (
            f'the message must still say what did NOT happen, so a log line '
            f'carrying only str(exc) is readable: {str(exc)!r}'
        )
        assert '1s' in str(exc), (
            f'the message must still name the budget it overran: {str(exc)!r}'
        )

        deadline = time.monotonic() + 3.0
        state = liveness(grandchild)
        while state is not Liveness.DEAD and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
            state = liveness(grandchild)
        assert state is Liveness.DEAD, (
            f'grandchild pid {grandchild_pid} must be dead after timeout — the '
            f'WHOLE process group must be killed, not just the direct child: '
            f'{grandchild} reads {state}'
        )

    async def test_timeout_of_already_exited_script_does_not_signal_the_group(
        self, tmp_path: Path,
    ):
        """The other half of Layer A, measured: when the SCRIPT has already
        exited, the teardown dispatches NO signal and the child it spawned
        SURVIVES — so "exited" and "group killed" are independent facts.

        Mechanism: ``communicate()`` waits for the merged stdout/stderr pipe to
        CLOSE, and the child holding its write end keeps it open past the
        script's own exit, while asyncio's child watcher populates
        ``Process.returncode`` independently.  So ``returncode`` is ALREADY set
        at teardown entry, and ``terminate_process_tree``'s pid-recycling
        short-circuit (task 845/3884) fires: refusing to signal a possibly
        recycled pid beats reaping an orphan.

        All three facts are asserted together because the escalation text is
        built from them, and the defect this pins is text that infers a kill
        from a timeout rather than reporting what was measured (task 4252
        reviewer finding).
        """
        script = tmp_path / 'exits-but-child-lives.sh'
        pidfile = tmp_path / 'survivor.pid'
        script.write_text(SURVIVOR_SCRIPT)
        script.chmod(0o755)

        try:
            # Hang tripwire, as in the sibling test above.
            with pytest.raises(ScriptTimeout) as excinfo:
                await asyncio.wait_for(
                    run_script(
                        script,
                        [str(pidfile)],
                        cwd=tmp_path,
                        env_overrides=None,
                        timeout_secs=1,
                        reap_grace_secs=REAP_GRACE_SECS,
                    ),
                    timeout=10,
                )

            exc = excinfo.value
            assert exc.exit_code == 7, (
                f'the script really did run to completion, and its exit code is '
                f'in hand at the raise site: {exc.exit_code!r}'
            )
            assert exc.teardown is ProcessTeardown.NOT_SIGNALLED, (
                f'the direct child was already reaped at teardown entry, so the '
                f'pid-recycling guard dispatched NOTHING — not a group kill, not '
                f'even a direct kill: {exc.teardown!r}'
            )
            survivor_pid = read_survivor_pid(pidfile)
            assert is_alive(survivor_pid), (
                f'survivor pid {survivor_pid} must still be RUNNING — nothing '
                f'signalled it, and this is the fact the escalation text has to '
                f'report: the process holding the deploy output pipe open is '
                f'still there for an operator to find and kill'
            )
        finally:
            kill_survivor(pidfile)

    async def test_terminate_process_tree_swallows_process_lookup_error(
        self, tmp_path: Path,
    ):
        """``terminate_process_tree`` must swallow ProcessLookupError from
        BOTH the killpg path and the proc.kill() fallback — an already-exited
        process must never propagate out of the timeout-cleanup helper.

        Task 3884: the killpg now targets the pgid FROZEN at spawn (passed in
        by the caller) rather than one re-derived via ``os.getpgid(proc.pid)``
        at kill time, so the dispatch is actually reached here — the swallow
        being asserted is the killpg's own ProcessLookupError, then the
        proc.kill() fallback's.
        """
        pgid = _fake_pgid()
        mock_proc = MagicMock()
        # pid must equal the pgid: unsafe_pgid_reason refuses a mismatch as a
        # corrupted capture, which would short-circuit before the killpg.
        mock_proc.pid = pgid
        # A bare MagicMock's `.returncode` is a truthy Mock (i.e. `is not None`),
        # which the task-3884 reaped-proc short-circuit would read as "already
        # reaped" and skip the signal entirely.  This proc is still running.
        mock_proc.returncode = None
        mock_proc.kill = MagicMock(side_effect=ProcessLookupError('already exited'))
        mock_proc.wait = AsyncMock(return_value=None)

        with patch(
            'os.killpg', side_effect=ProcessLookupError('no such process'),
        ) as mock_killpg:
            result = await terminate_process_tree(mock_proc, pgid, reap_grace_secs=0.05)

        mock_killpg.assert_called_once_with(pgid, signal.SIGKILL)
        mock_proc.kill.assert_called_once()
        assert result is ProcessTeardown.DIRECT_KILLED, (
            f'the killpg raised, so only the DIRECT child was signalled and '
            f'anything it spawned survives — the caller writes operator-facing '
            f'text about what was killed, so it must be able to learn that '
            f'(task 4252 reviewer finding): {result!r}'
        )

    async def test_terminate_process_tree_bounds_reap_when_proc_never_exits(
        self, tmp_path: Path,
    ):
        """The reap following the kill signal must be bounded by
        ``reap_grace_secs`` — a process that never exits after being signaled
        (e.g. stuck in an uninterruptible D-state) must not hang
        ``terminate_process_tree`` forever.

        The existing ProcessLookupError-swallow test's ``mock_proc.wait``
        resolves immediately, so it never exercises the
        ``except TimeoutError: logger.warning(...)`` branch around
        ``asyncio.wait_for(proc.wait(), timeout=reap_grace_secs)`` —
        this test exercises that bound directly.
        """
        pgid = _fake_pgid()
        mock_proc = MagicMock()
        # pid must equal the pgid — see the sibling test.
        mock_proc.pid = pgid
        # Still running (see the sibling test): a bare MagicMock's `.returncode`
        # is truthy, which the task-3884 reaped-proc short-circuit would treat
        # as already-reaped and skip the killpg this test asserts on.
        mock_proc.returncode = None
        # proc.wait() never resolves — simulates an unkillable/D-state process
        # that ignores SIGKILL.
        never_resolves = asyncio.Event()
        mock_proc.wait = AsyncMock(side_effect=never_resolves.wait)

        with patch('os.killpg') as mock_killpg:
            # Hang tripwire: if reap_grace_secs stops bounding the wait, fail
            # loudly instead of stalling the suite.
            result = await asyncio.wait_for(
                terminate_process_tree(mock_proc, pgid, reap_grace_secs=0.05), timeout=5,
            )

        mock_killpg.assert_called_once()
        assert result is ProcessTeardown.GROUP_KILLED, (
            f'the abandoned reap must NOT downgrade the disposition: a whole-group '
            f'signal WAS dispatched, and what the reap proves (or fails to prove) '
            f'is whether anything actually DIED — a separate fact the caller is '
            f'told not to conclude from this value: {result!r}'
        )

    async def test_no_killpg_after_proc_reaped(self, tmp_path: Path):
        """A reaped proc must receive NO killpg — its pid may already be recycled.

        Task 3884 / task 845: freezing the pgid at spawn removes the
        ``os.getpgid`` lookup, but the frozen NUMBER is itself stale once the
        leader has been reaped — the kernel may have recycled that pid onto an
        unrelated group (in the original incidents, the user's ``systemd --user``
        group, which killed the whole login session).  So the helper must
        dispatch no signal at all once ``proc.returncode is not None``, mirroring
        ``shared.proc_group.terminate_process_group``'s step 1.
        """
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.returncode = 0          # already reaped by asyncio's child watcher
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)
        killpg_calls: list[tuple[int, int]] = []
        getpgid_calls: list[int] = []
        with (
            patch('os.killpg', side_effect=lambda pgid, sig: killpg_calls.append((pgid, sig))),
            patch('os.getpgid', side_effect=lambda pid: getpgid_calls.append(pid) or 999),
        ):
            result = await terminate_process_tree(mock_proc, 12345, reap_grace_secs=0.05)
        assert killpg_calls == [], f'must not killpg a reaped proc (task 845); got {killpg_calls}'
        assert getpgid_calls == [], f'must never re-derive the pgid at kill time; got {getpgid_calls}'
        mock_proc.kill.assert_not_called()
        assert result is ProcessTeardown.NOT_SIGNALLED, (
            f'the caller builds operator-facing text claiming a kill, so it must '
            f'be able to learn that NOTHING was signalled (task 4252 reviewer '
            f'finding) — the three assertions above are the independent proof '
            f'that this disposition is the TRUE one: {result!r}'
        )

    @pytest.mark.parametrize('scenario', ['own_process_group', 'pid_mismatch'])
    async def test_refuses_to_killpg_an_unsafe_pgid(
        self, tmp_path: Path, scenario: str,
    ):
        """An unsafe pgid must be refused and degraded to a direct kill().

        The residual defence behind the frozen capture and the reaped-proc
        short-circuit: this helper delegates the signal choice to
        ``shared/src/shared/proc_group.py::sigkill_process_group``, whose
        ``unsafe_pgid_reason`` check is what refuses here.

        ``own_process_group`` is the consequential case, not a synthetic one —
        it is precisely the task-845 outcome. A pgid that resolves to our OWN
        group means the killpg would SIGKILL this test session, the
        orchestrator, and everything else sharing it. ``pid_mismatch`` covers a
        corrupted capture, where the frozen int no longer describes ``proc``.

        Degrading to ``proc.kill()`` rather than doing nothing matches
        ``df_pytest_isolation``'s handling of the same refusal: the timed-out
        child still dies, and only the (unreachable) grandchildren are forgone.
        """
        mock_proc = MagicMock()
        mock_proc.returncode = None          # still running: not the reaped path
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock(return_value=None)

        if scenario == 'own_process_group':
            pgid = os.getpgrp()
            mock_proc.pid = pgid
        else:
            pgid = _fake_pgid()
            mock_proc.pid = pgid + 1         # capture no longer describes proc

        killpg_calls: list[tuple[int, int]] = []
        with patch('os.killpg', side_effect=lambda p, s: killpg_calls.append((p, s))):
            result = await terminate_process_tree(mock_proc, pgid, reap_grace_secs=0.05)

        assert killpg_calls == [], (
            f'{scenario}: refused pgid must never be signalled -- killpg on our '
            f'own group is the task-845 login-session kill; got {killpg_calls}'
        )
        mock_proc.kill.assert_called_once()
        assert result is ProcessTeardown.DIRECT_KILLED, (
            f'{scenario}: only the direct child was signalled, so anything it '
            f'spawned SURVIVES — the caller must be able to learn that instead '
            f'of telling an operator the whole group is gone (task 4252 reviewer '
            f'finding): {result!r}'
        )


class TestProcessTeardownFromDispatch:
    """ProcessTeardown.from_dispatch renders shared.proc_group's SigkillDispatch."""

    @pytest.mark.parametrize(
        ('dispatch', 'expected'),
        [
            (SigkillDispatch.GROUP_KILLED, ProcessTeardown.GROUP_KILLED),
            (SigkillDispatch.DIRECT_KILLED, ProcessTeardown.DIRECT_KILLED),
            (SigkillDispatch.NOT_SIGNALLED, ProcessTeardown.NOT_SIGNALLED),
        ],
    )
    def test_maps_each_dispatch_to_its_namesake(self, dispatch, expected):
        assert ProcessTeardown.from_dispatch(dispatch) is expected

    def test_the_mapping_is_a_bijection(self):
        assert len(SigkillDispatch) == len(ProcessTeardown)
        assert {ProcessTeardown.from_dispatch(d) for d in SigkillDispatch} == set(
            ProcessTeardown
        )



_SCRIPT = Path('/srv/deploy/restart-unit.sh')

# Every measured shape a ScriptTimeout can carry: {exit_code} x {disposition}.
_MEASURED_SHAPES = pytest.mark.parametrize(
    ('exit_code', 'teardown'),
    [(code, teardown) for code in (None, 0) for teardown in ProcessTeardown],
    ids=lambda value: value.name if isinstance(value, ProcessTeardown) else f'rc={value}',
)


class TestScriptTimeoutRendering:
    """ScriptTimeout renders its operator text from the facts it measured.

    The runner's deploy and predicate escalation tests pin the same text
    end-to-end; these pin it at the type that owns it.
    """

    @_MEASURED_SHAPES
    def test_summary_phrase_claims_no_exit_code_only_without_one(
        self, exit_code: int | None, teardown: ProcessTeardown,
    ):
        exc = ScriptTimeout(5, exit_code=exit_code, teardown=teardown)

        expected = (
            'no exit code'
            if exit_code is None
            else 'script exited rc=0, output pipe held open'
        )
        assert exc.summary_phrase == expected

    @_MEASURED_SHAPES
    def test_budget_line_names_the_budget_and_the_dispatched_signal(
        self, exit_code: int | None, teardown: ProcessTeardown,
    ):
        exc = ScriptTimeout(5, exit_code=exit_code, teardown=teardown)

        assert exc.budget_line('Deploy script') == (
            "Deploy script exceeded its own per-script timeout "
            f"(5s = before_done['timeout_secs']) and {teardown.clause}."
        )

    @_MEASURED_SHAPES
    def test_survivor_line_appears_iff_the_group_was_not_killed(
        self, exit_code: int | None, teardown: ProcessTeardown,
    ):
        exc = ScriptTimeout(5, exit_code=exit_code, teardown=teardown)

        line = exc.survivor_line(_SCRIPT)

        assert (line is None) is (teardown is ProcessTeardown.GROUP_KILLED)
        assert line is None or str(_SCRIPT) in line, line
