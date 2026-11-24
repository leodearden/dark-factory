"""Tests for the `orchestrator run` command and its shutdown helpers."""

import asyncio
import contextlib
import io
import logging
import threading
import time
import traceback as traceback_module
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec
from click.testing import CliRunner

import orchestrator.cli as cli_module
import orchestrator.cli_config as cli_config
from orchestrator.cli import (
    SHUTDOWN_WATCHDOG_TIMEOUT_SECS,
    _force_exit_after_delay,
    _make_cancel_handler,
    _parse_duration,
    main,
)
from orchestrator.config import OrchestratorConfig


@pytest.fixture(autouse=True)
def spawn_temp_sweep(monkeypatch) -> MagicMock:
    """No test in this module may sweep the real /tmp through the ``run`` command."""
    sweep = MagicMock(return_value=0)
    monkeypatch.setattr(cli_module, 'sweep_orphaned_spawn_temp_files', sweep, raising=True)
    return sweep


class TestParseDuration:
    def test_hours(self):
        assert _parse_duration("4h") == 14400

    def test_minutes(self):
        assert _parse_duration("30m") == 1800

    def test_seconds(self):
        assert _parse_duration("90s") == 90

    def test_bare_number(self):
        assert _parse_duration("3600") == 3600

    def test_uppercase(self):
        assert _parse_duration("2H") == 7200

    def test_whitespace(self):
        assert _parse_duration("  10m  ") == 600

    def test_invalid(self):
        with pytest.raises(ValueError):
            _parse_duration("abc")


class TestSignalHandlerIdempotence:
    """_make_cancel_handler returns an idempotent SIGTERM/SIGINT callback.

    Rationale: a second signal during shutdown cleanup was observed to
    re-cancel the main task mid-finally, skipping cost_store.close() and
    leaving aiosqlite's non-daemon worker thread alive → interpreter hang.
    """

    def test_first_signal_cancels_main_task(self):
        main_task = MagicMock()
        logger = logging.getLogger('test.cli')
        handler = _make_cancel_handler(main_task, logger)

        handler('SIGTERM')

        main_task.cancel.assert_called_once()

    def test_second_signal_does_not_re_cancel(self, caplog):
        main_task = MagicMock()
        logger = logging.getLogger('orchestrator.cli.test')
        handler = _make_cancel_handler(main_task, logger)

        handler('SIGTERM')
        with caplog.at_level(logging.INFO, logger=logger.name):
            handler('SIGTERM')

        # cancel() must still be called exactly once — the second signal is a no-op
        main_task.cancel.assert_called_once()
        # Second invocation logs at INFO level so operators see it wasn't ignored silently
        info_records = [
            r for r in caplog.records
            if r.levelno == logging.INFO and 'already in progress' in r.message
        ]
        assert len(info_records) == 1
        assert 'SIGTERM' in info_records[0].message

    def test_each_handler_instance_is_independent(self):
        """Two handlers from two _make_cancel_handler calls don't share state."""
        task_a = MagicMock()
        task_b = MagicMock()
        logger = logging.getLogger('test.cli')
        handler_a = _make_cancel_handler(task_a, logger)
        handler_b = _make_cancel_handler(task_b, logger)

        handler_a('SIGTERM')
        handler_b('SIGINT')

        task_a.cancel.assert_called_once()
        task_b.cancel.assert_called_once()


class TestForceExitWatchdog:
    """Tests for _force_exit_after_delay shutdown watchdog helper."""

    def test_fires_after_timeout(self):
        """Watchdog calls os._exit(137) after the timeout elapses."""
        calls: list[int] = []

        def stub(code: int) -> None:
            calls.append(code)

        handle = _force_exit_after_delay(timeout_secs=0.05, _exit=stub)

        # Poll with a deadline instead of a fixed sleep to avoid spurious
        # failures under CI scheduler stalls (GC, container contention, etc.).
        deadline = time.monotonic() + 5.0
        while not calls and time.monotonic() < deadline:
            time.sleep(0.05)

        assert calls == [137], f'expected [137], got {calls}'
        handle.thread.join(timeout=1.0)
        assert not handle.thread.is_alive(), (
            'watchdog thread did not exit after firing os._exit replacement'
        )

    def test_does_not_fire_before_timeout(self):
        """Watchdog does not call os._exit before its timeout elapses.

        This pins the 'never fires on clean exit' guarantee at the unit layer —
        if the process terminates before timeout_secs (daemon thread killed by
        interpreter shutdown on clean exit), the watchdog is still inside its
        `_event.wait` and has not yet reached the os._exit call site.

        Together with test_fires_after_timeout (fires AFTER timeout) and
        test_disarm_prevents_force_exit (never fires after disarm), this closes
        the timing-contract circle: fires exactly once, after the timeout, and
        only if not disarmed.

        The subprocess-level counterpart (`test_shutdown_watchdog_force_exits_on_thread_leak`
        in test_shutdown.py) pins the opposite — fires when a non-daemon thread is leaked.

        Structural fix (same class as test_disarm_prevents_force_exit below):
        this previously armed with timeout_secs=2.0 and used a fixed
        ``time.sleep(0.2)`` as a "10x margin" before checking ``calls == []``.
        No fixed margin is safe under full-suite xdist load — the main thread's
        wakeup from ``sleep(0.2)`` can itself be delayed by scheduler noise, and
        if that delay pushes the check past the 2.0s deadline the watchdog has
        legitimately fired by the time ``calls`` is read (same failure shape as
        test_disarm_prevents_force_exit's observed ``calls == [137]``). Arming
        with an effectively-unbounded timeout_secs instead makes the deadline
        unreachable within the test's lifetime, so ``thread.join(timeout=...)``
        can be used as the "has not fired yet" probe without racing a wall-clock
        deadline: no matter how delayed the join itself is in landing,
        ``is_alive()`` cannot go False before 3600s have elapsed.
        """
        calls: list[int] = []

        def stub(code: int) -> None:
            calls.append(code)

        handle = _force_exit_after_delay(timeout_secs=3600.0, _exit=stub)

        # "Has not fired yet" probe: join blocks for up to 0.2s or until the
        # watchdog thread exits, whichever comes first. Unlike a free-standing
        # sleep, there is no deadline here for scheduler noise to run past —
        # timeout_secs is unreachable within the test's lifetime, so is_alive()
        # cannot go False regardless of how delayed the join is in landing.
        handle.thread.join(timeout=0.2)
        assert handle.thread.is_alive(), (
            'watchdog thread exited before its timeout elapsed — fired early?'
        )
        assert calls == [], (
            f'watchdog fired before timeout elapsed (clean-exit window): {calls}'
        )

        # Cleanup: disarm and join to guarantee thread exits cleanly.
        handle.disarm()
        handle.thread.join(timeout=1.0)
        assert not handle.thread.is_alive(), (
            'watchdog thread did not exit after disarm'
        )

    def test_disarm_prevents_force_exit(self):
        """Calling disarm() before timeout prevents os._exit from being called.

        Structural fix (same class as done tasks 1836/1851/2320/2840/2921/2959/
        3491): the watchdog thread blocks on ``threading.Event.wait(timeout_secs)``
        and ``disarm()`` calls ``_event.set()``, which wakes the thread and makes
        it return WITHOUT ever calling ``_exit_fn`` — regardless of how large
        timeout_secs is, and regardless of whether the event is set before or
        after the thread reaches ``.wait()`` (Event is stateful, not an
        edge-triggered signal, so ordering can't be missed). The correctness
        condition is only "does disarm() land before timeout_secs elapses" — a
        small timeout_secs (previously 0.2s) instead raced the main thread's next
        bytecode (the disarm() call) against the watchdog's internal deadline;
        under full-suite xdist load that arm→disarm gap can exceed hundreds of ms
        and the watchdog fires before disarm() lands (observed: `calls == [137]`
        on a saturated worker). Using an effectively-unbounded timeout_secs makes
        that gap structurally impossible to exceed rather than merely widening a
        still-tight window, and removes the need for a compensating sleep.

        Synchronisation note (load-bearing, keep this ordering): ``calls`` is
        mutated by the watchdog thread and read by the main thread, so it is
        only meaningfully observable after a happens-before edge between the
        two. ``thread.join()`` is that edge. The join below is therefore NOT
        cleanup — it MUST run, and its ``is_alive()`` check MUST be asserted,
        BEFORE the ``calls == []`` assertion. Checking `calls` first would be
        vacuous: right after `disarm()` the watchdog thread has had no chance
        to run at all, so `calls` reads `[]` regardless of whether the
        production guard (`cli.py`'s `fired = not _event.wait(timeout_secs)`)
        is correct or inverted, letting a real regression go undetected. The
        join timeout is deliberately generous (30.0s, not the usual 1.0s used
        for pure cleanup joins elsewhere in this file): once `disarm()` sets
        the Event the thread wakes in microseconds on the happy path, so the
        bound is free there, but on a saturated xdist worker a tight bound
        could itself manufacture a spurious `is_alive()` failure — exactly the
        load-sensitive flake class this task removes. Do not reorder or
        shrink this without re-deriving both properties.
        """
        calls: list[int] = []

        def stub(code: int) -> None:
            calls.append(code)

        handle = _force_exit_after_delay(timeout_secs=3600.0, _exit=stub)
        handle.disarm()

        # Synchronisation point FIRST: join before making any claim about
        # `calls`, which the watchdog thread mutates (see docstring above).
        handle.thread.join(timeout=30.0)
        assert not handle.thread.is_alive(), (
            'watchdog thread did not exit after disarm'
        )
        assert calls == [], f'expected no calls, got {calls}'

    def test_diagnostic_dump_lists_live_threads(self):
        """When the watchdog fires, it writes a diagnostic dump to the stream."""
        calls: list[int] = []

        def stub(code: int) -> None:
            calls.append(code)

        stream = io.StringIO()
        handle = _force_exit_after_delay(timeout_secs=0.05, stream=stream, _exit=stub)

        # Poll with a deadline — stream is written before os._exit is called,
        # so once calls is non-empty the output is already available.
        deadline = time.monotonic() + 5.0
        while not calls and time.monotonic() < deadline:
            time.sleep(0.05)

        output = stream.getvalue()
        # Sentinel header must be present — match the full happy-path string so
        # a silent fall-through into the except branch (which emits a different
        # sentinel) is caught rather than masked by the shorter prefix.
        assert 'SHUTDOWN WATCHDOG FIRED — process hung after asyncio.run() returned' in output, (
            f'sentinel missing from dump:\n{output!r}'
        )
        # The main thread should appear in the dump
        main_thread_name = threading.main_thread().name  # typically 'MainThread'
        assert main_thread_name in output, (
            f'main thread {main_thread_name!r} not in dump:\n{output!r}'
        )
        # At least one stack frame line (traceback.format_stack produces "  File ..." lines)
        assert '  File ' in output, (
            f'no frame lines in dump:\n{output!r}'
        )
        assert calls == [137]
        handle.thread.join(timeout=1.0)
        assert not handle.thread.is_alive(), (
            'watchdog thread did not exit after firing os._exit replacement'
        )

    def test_dump_failure_still_fires_exit(self, monkeypatch):
        """os._exit(137) is called even when the diagnostic dump itself fails.

        During interpreter shutdown, sys._current_frames, traceback internals,
        or sys.stderr may be partially torn down.  The outer try/except Exception
        in the watchdog catches any failure and still reaches _exit_fn — so the
        process always escapes a hang even if the diagnostic output is lost.
        This test locks in the 'fail-open to force-exit' guarantee.
        """
        calls: list[int] = []

        def stub(code: int) -> None:
            calls.append(code)

        # Simulate partial interpreter shutdown where traceback.format_stack is broken.
        def _raise_on_format(*args, **kwargs):
            raise RuntimeError('simulated shutdown tear-down of traceback module')

        monkeypatch.setattr(traceback_module, 'format_stack', _raise_on_format)

        stream = io.StringIO()
        handle = _force_exit_after_delay(timeout_secs=0.05, stream=stream, _exit=stub)

        # Poll with deadline — os._exit must be called even though the dump failed.
        deadline = time.monotonic() + 5.0
        while not calls and time.monotonic() < deadline:
            time.sleep(0.05)

        assert calls == [137], (
            f'expected os._exit(137) even when dump fails, got {calls}'
        )

        output = stream.getvalue()
        assert 'SHUTDOWN WATCHDOG FIRED (diagnostic dump failed)' in output, (
            f'fallback sentinel missing from dump-failure output:\n{output!r}'
        )
        handle.thread.join(timeout=1.0)
        assert not handle.thread.is_alive(), (
            'watchdog thread did not exit after firing os._exit replacement'
        )

    def test_force_exit_returns_handle_with_disarm_and_thread(self):
        """_force_exit_after_delay returns a WatchdogHandle whose thread is a live daemon
        and whose disarm() stops the thread.

        Pins only the behavioral contract of WatchdogHandle:
        - .thread is a daemon (load-bearing — non-daemon threads block interpreter shutdown)
        - .thread is alive immediately after arming (proves thread.start() ran)
        - Calling disarm() stops the thread within a reasonable timeout

        NamedTuple shape (.disarm is callable, .thread is threading.Thread) is guaranteed
        by construction in cli.py and not re-asserted here. Thread name is cosmetic and
        intentionally unpinned — rename-friendly.
        """
        calls: list[int] = []

        def stub(code: int) -> None:
            calls.append(code)

        handle = _force_exit_after_delay(timeout_secs=5.0, _exit=stub)

        assert handle.thread.daemon is True, (
            'handle.thread must be a daemon thread'
        )
        assert handle.thread.is_alive(), (
            'handle.thread must be alive immediately after arming'
        )

        # Cleanup: disarm and join to confirm thread exits cleanly.
        handle.disarm()
        handle.thread.join(timeout=1.0)
        assert not handle.thread.is_alive(), 'thread did not exit after disarm'

    def test_fallback_write_failure_still_fires_exit(self, monkeypatch):
        """os._exit(137) is called even when both the dump AND the fallback write fail.

        The watchdog comment explicitly promises: "Wrapped in its own try/except
        so a stream-write failure still falls through to _exit_fn".  This test
        locks in that doubly-defensive contract by making traceback.format_stack
        *and* stream.write both raise unconditionally.  If the nested try/except
        is ever removed, this test fails before the force-exit guarantee is silently
        dropped.

        Coverage split: this test mocks traceback.format_stack to raise BEFORE
        out.write(''.join(lines)) on cli.py:85 is reached — so only the INNER
        try/except that wraps the fallback-sentinel write is exercised here.  The
        complement case (format_stack intact, the OUTER out.write raises first,
        fallback also fails) is covered by test_outer_write_failure_still_fires_exit.
        Together the pair pins both entry points into the nested try/except.
        """
        calls: list[int] = []

        def stub(code: int) -> None:
            calls.append(code)

        # Simulate partial interpreter shutdown where traceback internals are broken.
        def _raise_on_format(*args, **kwargs):
            raise RuntimeError('simulated shutdown tear-down of traceback module')

        monkeypatch.setattr(traceback_module, 'format_stack', _raise_on_format)

        # A stream whose write() always raises — exercises the inner try/except that
        # wraps the fallback sentinel write.
        class _BrokenStream:
            def write(self, _s):
                raise OSError('simulated torn-down stderr')

            def flush(self):
                raise OSError('simulated torn-down stderr')

        handle = _force_exit_after_delay(timeout_secs=0.05, stream=_BrokenStream(), _exit=stub)

        # Poll with deadline — os._exit must fire even when the fallback write also fails.
        deadline = time.monotonic() + 5.0
        while not calls and time.monotonic() < deadline:
            time.sleep(0.05)

        assert calls == [137], (
            f'expected os._exit(137) even when fallback write fails, got {calls}'
        )
        handle.thread.join(timeout=1.0)
        assert not handle.thread.is_alive(), (
            'watchdog thread did not exit after firing os._exit replacement'
        )

    def test_outer_write_failure_still_fires_exit(self):
        """os._exit(137) is called even when out.write(''.join(lines)) raises with format_stack intact.

        Closes the coverage gap in test_fallback_write_failure_still_fires_exit: that sibling
        test mocks traceback.format_stack to raise FIRST, so `lines` is never built and the
        outer out.write(''.join(lines)) call is never reached.  This test covers the complement:
        format_stack succeeds, `lines` is populated with real frames, but out.write raises
        (outer except catches), then the inner fallback write also raises (_OuterBrokenStream
        makes every write raise), the inner except swallows, and execution falls through to
        _exit_fn(137).

        If the outer try/except wrapping out.write is ever removed, this test will fail before
        the force-exit guarantee is silently dropped — mirroring the sibling test's contract
        for the nested try/except.
        """
        calls: list[int] = []

        def stub(code: int) -> None:
            calls.append(code)

        # Does NOT mock traceback.format_stack — it runs normally, producing real frames.
        # Records every write attempt (payload captured before OSError is raised) so we
        # can assert which sentinel strings the watchdog tried to write, proving exactly
        # which control-flow paths were traversed.
        write_attempts: list[str] = []

        class _OuterBrokenStream:
            def write(self, s):
                write_attempts.append(s)  # record before raising
                raise OSError('outer write broken')

            def flush(self):
                raise OSError('outer flush broken')

        handle = _force_exit_after_delay(timeout_secs=0.05, stream=_OuterBrokenStream(), _exit=stub)

        # Poll with deadline — os._exit must fire even when the outer write fails.
        deadline = time.monotonic() + 5.0
        while not calls and time.monotonic() < deadline:
            time.sleep(0.05)

        assert calls == [137], (
            f'expected os._exit(137) even when outer write fails with format_stack intact, '
            f'got {calls}'
        )
        # Existence checks rather than positional: the distinguishing invariant vs.
        # test_fallback_write_failure_still_fires_exit is that the outer header sentinel
        # can only appear if format_stack ran and outer out.write was reached — which is
        # exactly the path mocked out in the sibling.
        assert any('process hung after asyncio.run()' in s for s in write_attempts), (
            f'outer watchdog-fired header sentinel missing from write attempts '
            f"(proves format_stack ran and outer out.write was reached — "
            f"'process hung after asyncio.run()' appears only in the cli.py:78 "
            f'outer header, not in the cli.py:94 fallback sentinel): {write_attempts!r}'
        )
        assert any('(diagnostic dump failed)' in s for s in write_attempts), (
            f'inner fallback sentinel missing from write attempts '
            f'(proves inner fallback out.write was also exercised): {write_attempts!r}'
        )
        handle.thread.join(timeout=1.0)
        assert not handle.thread.is_alive(), (
            'watchdog thread did not exit after firing os._exit replacement'
        )

    def test_force_exit_watchdog_worker_safety_contract(self):
        """Worker-safety contract: injected _exit is called; global os._exit is not.

        xdist worker-safety regression: under CPU starvation a watchdog daemon
        thread can fire LATE — after the test function's monkeypatch teardown
        has restored the REAL global os._exit. If the watchdog then calls the
        REAL os._exit(137), the xdist worker process is killed abruptly, causing
        ``[gwN] node down: Not properly terminated`` and a KeyError INTERNALERROR.

        With `_exit=stub` injection the closure captures the stub at arm-time.
        Even if the thread fires after the test exits, it calls the stub — a
        harmless no-op — not the REAL global os._exit.

        This test does NOT monkeypatch os._exit at the global level.  The fact
        that this function completes at all proves global os._exit was not called
        (os._exit would have terminated the process).  Verified contract:

        1. The injected stub receives [137] (watchdog fired via injection).
        2. The thread exits after firing (no resource leak).
        3. No 'shutdown-watchdog' thread remains alive (no thread leak).
        """
        calls: list[int] = []

        def stub(code: int) -> None:
            calls.append(code)

        handle = _force_exit_after_delay(timeout_secs=0.05, _exit=stub)

        # Poll with a deadline so spurious CI scheduler stalls don't cause
        # false failures.
        deadline = time.monotonic() + 5.0
        while not calls and time.monotonic() < deadline:
            time.sleep(0.05)

        # (1) Stub was called with 137 — watchdog fired via injection.
        assert calls == [137], f'expected injected stub to receive [137], got {calls}'

        # (2) Thread exits after firing.
        handle.thread.join(timeout=1.0)
        assert not handle.thread.is_alive(), (
            'watchdog thread must exit after firing the injected _exit stub'
        )

        # (3) No 'shutdown-watchdog' thread may remain alive (no thread leak).
        live_watchdog_threads = [
            t for t in threading.enumerate()
            if t.name == 'shutdown-watchdog' and t.is_alive()
        ]
        assert not live_watchdog_threads, (
            f'shutdown-watchdog thread(s) still alive after firing and join: '
            f'{live_watchdog_threads}'
        )


_HANG_GUARD_SECS = 10.0


async def _ignore_cancels_until(release: asyncio.Event) -> None:
    while not release.is_set():
        try:
            await release.wait()
        except asyncio.CancelledError:
            continue


def _shutdown_watchdog_threads() -> set[threading.Thread]:
    return {t for t in threading.enumerate() if t.name == 'shutdown-watchdog'}


class TestRunWithBoundedTeardown:
    """asyncio.run()'s own teardown waits, unbounded, on every task left when
    the main coroutine ends; _run_with_bounded_teardown bounds that wait."""

    @pytest.mark.parametrize('ending', ['returns', 'raises_cancelled'])
    def test_a_leftover_task_that_ignores_its_cancel_is_force_exited(self, ending, capsys):
        box: dict = {}
        exits: list[int] = []
        guard_fired = threading.Event()

        async def main():
            box['loop'] = asyncio.get_running_loop()
            box['release'] = asyncio.Event()
            box['leftover'] = asyncio.create_task(
                _ignore_cancels_until(box['release']), name='immune-leftover',
            )
            await asyncio.sleep(0)
            if ending == 'raises_cancelled':
                raise asyncio.CancelledError()
            return 'report'

        def release() -> None:
            with contextlib.suppress(RuntimeError):
                box['loop'].call_soon_threadsafe(box['release'].set)

        def exit_stub(code: int) -> None:
            exits.append(code)
            release()

        def on_guard() -> None:
            guard_fired.set()
            release()

        guard = threading.Timer(_HANG_GUARD_SECS, on_guard)
        guard.start()
        try:
            if ending == 'raises_cancelled':
                with pytest.raises(asyncio.CancelledError):
                    cli_module._run_with_bounded_teardown(
                        main, teardown_secs=0.2, _exit=exit_stub,
                    )
            else:
                result = cli_module._run_with_bounded_teardown(
                    main, teardown_secs=0.2, _exit=exit_stub,
                )
                assert result == 'report'
        finally:
            guard.cancel()

        assert exits == [137]
        assert not guard_fired.is_set(), 'the hang guard, not the watchdog, ended the teardown'
        assert "asyncio.run()'s teardown" in capsys.readouterr().err

    def test_a_clean_teardown_disarms_the_watchdog(self):
        exits: list[int] = []
        leftovers: list[asyncio.Task] = []

        async def main():
            leftovers.append(asyncio.create_task(asyncio.sleep(3600)))
            return 'report'

        before = _shutdown_watchdog_threads()
        result = cli_module._run_with_bounded_teardown(
            main, teardown_secs=0.5, _exit=exits.append,
        )
        for thread in _shutdown_watchdog_threads() - before:
            thread.join(timeout=5.0)

        assert result == 'report'
        assert exits == [], 'the teardown watchdog stayed armed after asyncio.run returned'


def _wire_run_with_fake_harness(monkeypatch, fake_report):
    """Wire mocks so the real _main coroutine runs and calls harness.run once."""
    # Stub the watchdog so no real daemon timer thread is spawned.
    monkeypatch.setattr(
        cli_module, '_force_exit_after_delay', lambda *a, **k: MagicMock()
    )
    fake_config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    monkeypatch.setattr(cli_config, 'load_config', lambda _path: fake_config)

    fake_harness = MagicMock()
    # AsyncMock so `await harness.run(...)` inside the real _main resolves.
    fake_harness.run = AsyncMock(return_value=fake_report)
    monkeypatch.setattr('orchestrator.harness.Harness', lambda config, **_: fake_harness)
    return fake_harness


class TestRunUntilIdleFlag:
    """The --until-idle flag must thread through to harness.run(until_idle=...).

    Default (flag absent) is run-forever (until_idle=False); the flag opts into
    legacy exit-on-drain (until_idle=True).
    """

    def test_flag_threads_until_idle_true(self, monkeypatch):
        fake_report = MagicMock()
        fake_report.blocked = 0
        fake_report.summary.return_value = 'ok'
        fake_harness = _wire_run_with_fake_harness(monkeypatch, fake_report)

        result = CliRunner().invoke(
            main, ['run', '--config', '/dev/null', '--until-idle']
        )

        assert result.exit_code == 0, result.output
        assert fake_harness.run.call_args.kwargs.get('until_idle') is True, (
            f'--until-idle must pass until_idle=True; '
            f'kwargs={fake_harness.run.call_args.kwargs!r}'
        )

    def test_default_threads_until_idle_false(self, monkeypatch):
        fake_report = MagicMock()
        fake_report.blocked = 0
        fake_report.summary.return_value = 'ok'
        fake_harness = _wire_run_with_fake_harness(monkeypatch, fake_report)

        result = CliRunner().invoke(main, ['run', '--config', '/dev/null'])

        assert result.exit_code == 0, result.output
        assert fake_harness.run.call_args.kwargs.get('until_idle') is False, (
            f'default run must pass until_idle=False (run forever); '
            f'kwargs={fake_harness.run.call_args.kwargs!r}'
        )


class TestRunAppliesProjectSliceWeight:
    """`orchestrator run`, every unit's ExecStart, is the one caller that opts
    into startup step 1c1i (docs/cgroup-slice-topology.md §Slice weight)."""

    def test_run_requests_the_project_slice_weight(self, monkeypatch):
        fake_report = MagicMock()
        fake_report.blocked = 0
        fake_report.summary.return_value = 'ok'
        fake_harness = _wire_run_with_fake_harness(monkeypatch, fake_report)

        result = CliRunner().invoke(main, ['run', '--config', '/dev/null'])

        assert result.exit_code == 0, result.output
        assert fake_harness.run.call_args.kwargs.get('apply_project_slice_weight') is True, (
            f'run must pass apply_project_slice_weight=True; '
            f'kwargs={fake_harness.run.call_args.kwargs!r}'
        )


class TestRunSweepsOrphanedSpawnTempFiles:
    """run() sweeps orphaned agent-spawn temp files once, before building the Harness,
    and a failing sweep never blocks startup."""

    @staticmethod
    def _report() -> MagicMock:
        report = MagicMock()
        report.blocked = 0
        report.summary.return_value = 'ok'
        return report

    def test_sweeps_once_before_the_harness_is_built(self, monkeypatch, spawn_temp_sweep):
        fake_harness = _wire_run_with_fake_harness(monkeypatch, self._report())
        events: list[str] = []
        spawn_temp_sweep.side_effect = lambda *a, **k: events.append('sweep') or 0

        def _recording_harness(config, **_):
            events.append('harness')
            return fake_harness

        monkeypatch.setattr('orchestrator.harness.Harness', _recording_harness)

        result = CliRunner().invoke(main, ['run', '--config', '/dev/null'])

        assert result.exit_code == 0, result.output
        assert events.index('sweep') < events.index('harness'), events
        spawn_temp_sweep.assert_called_once()

    def test_a_failing_sweep_does_not_block_startup(self, monkeypatch, spawn_temp_sweep, caplog):
        fake_harness = _wire_run_with_fake_harness(monkeypatch, self._report())
        spawn_temp_sweep.side_effect = RuntimeError('boom')

        with caplog.at_level(logging.WARNING, logger='orchestrator.cli'):
            result = CliRunner().invoke(main, ['run', '--config', '/dev/null'])

        assert result.exit_code == 0, result.output
        fake_harness.run.assert_awaited()
        warnings = [
            r.getMessage() for r in caplog.records
            if r.name == 'orchestrator.cli' and r.levelno == logging.WARNING
        ]
        assert any('spawn temp files' in m for m in warnings), warnings


class TestRunArmsWatchdog:
    """run() must arm the shutdown watchdog and disarm it on both exit paths."""

    def _fake_watchdog_factory(self, events: list | None = None):
        """Returns (recorder, fake_force_exit_after_delay).

        recorder has:
          .armed_with       – the timeout_secs passed on arming
          .disarm_called    – True once disarm() is called

        Optional `events` list: when provided, the fake appends the string
        ``'arm'`` to `events` on each arming call, enabling ordering assertions
        (e.g., that echo precedes arm).  The echo-side recording is NOT handled
        here — callers that need it (test_report_emitted_before_watchdog_armed)
        wrap click.echo themselves, keeping factory concerns minimal.

        The ``thread`` field of the returned ``WatchdogHandle`` is a
        ``MagicMock(spec=threading.Thread)`` — a type-structural placeholder
        that never spawns a real thread.  The spec constraint means any attempt
        to access an attribute that does not exist on ``threading.Thread``
        raises ``AttributeError`` immediately, so accidental misuse is caught
        loudly.  ``TestRunArmsWatchdog`` callers never call ``.join()`` or
        ``.is_alive()`` on the fake; those are exercised only by
        ``TestForceExitWatchdog`` tests that use the real implementation.
        """
        state = {'armed_with': None, 'disarm_called': False}

        def fake_disarm():
            state['disarm_called'] = True

        def fake_force_exit(timeout_secs, exit_code=137, *, stream=None):
            if events is not None:
                events.append('arm')
            state['armed_with'] = timeout_secs
            # MagicMock(spec=threading.Thread) satisfies the WatchdogHandle.thread
            # type structurally without spawning a background thread.  spec= ensures
            # any accidental access to a non-Thread attribute raises AttributeError.
            fake_thread = MagicMock(spec=threading.Thread)
            return cli_module.WatchdogHandle(disarm=fake_disarm, thread=fake_thread)

        return state, fake_force_exit

    def test_normal_path_arms_watchdog_and_leaves_armed(self, monkeypatch):
        """run() arms the watchdog on normal (non-cancelled) exit and leaves it armed.

        The watchdog must NOT be disarmed — disarming defeats its purpose.
        The whole point is to guard the interpreter-shutdown path that begins
        after sys.exit raises SystemExit: atexit callbacks run, then
        threading._shutdown() joins non-daemon threads. If a non-daemon thread
        is stuck, threading._shutdown() hangs there. The armed daemon watchdog
        fires os._exit(137) after SHUTDOWN_WATCHDOG_TIMEOUT_SECS.

        If shutdown completes cleanly, the daemon watchdog thread is killed
        with the process and fires nothing. Either way, NOT disarming is correct.
        """
        state, fake_force_exit = self._fake_watchdog_factory()
        monkeypatch.setattr(cli_module, '_force_exit_after_delay', fake_force_exit)

        # Fake config so load_config doesn't need a real file
        fake_config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        monkeypatch.setattr(cli_config, 'load_config', lambda _path: fake_config)

        # Fake Harness so we don't need a real one
        fake_harness = MagicMock()
        fake_report = MagicMock()
        fake_report.blocked = 0
        fake_report.summary.return_value = 'all done'
        fake_harness.run = MagicMock()

        monkeypatch.setattr('orchestrator.harness.Harness', lambda config, **_: fake_harness)

        # asyncio.run returns fake_report (bypasses _main entirely).
        # Close the coroutine to avoid "coroutine was never awaited" warnings.
        def fake_asyncio_run(coro):
            coro.close()
            return fake_report

        monkeypatch.setattr(cli_module.asyncio, 'run', fake_asyncio_run)

        runner = CliRunner()
        runner.invoke(main, ['run', '--config', '/dev/null'])

        # Watchdog must have been armed with the module constant
        assert state['armed_with'] == SHUTDOWN_WATCHDOG_TIMEOUT_SECS, (
            f"expected armed_with={SHUTDOWN_WATCHDOG_TIMEOUT_SECS}, got {state['armed_with']}"
        )
        # Watchdog must NOT be disarmed — it must remain armed to guard interpreter shutdown.
        assert not state['disarm_called'], (
            'disarm() was called — watchdog must remain armed to guard interpreter shutdown '
            '(threading._shutdown() joining non-daemon threads after sys.exit)'
        )

    def test_cancelled_path_arms_watchdog_and_leaves_armed(self, monkeypatch):
        """run() arms the watchdog even when asyncio.run raises CancelledError, and leaves it armed.

        Same rationale as test_normal_path_arms_watchdog_and_leaves_armed: the
        watchdog guards the interpreter-shutdown window (atexit + threading._shutdown)
        that begins AFTER sys.exit(130) raises SystemExit. Disarming it would
        defeat its purpose on the cancellation path.
        """
        state, fake_force_exit = self._fake_watchdog_factory()
        monkeypatch.setattr(cli_module, '_force_exit_after_delay', fake_force_exit)

        fake_config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        monkeypatch.setattr(cli_config, 'load_config', lambda _path: fake_config)

        fake_harness = MagicMock()
        monkeypatch.setattr('orchestrator.harness.Harness', lambda config, **_: fake_harness)

        # asyncio.run raises CancelledError to simulate SIGTERM path.
        # Close the coroutine to avoid "coroutine was never awaited" warnings.
        def raise_cancelled(coro):
            coro.close()
            raise asyncio.CancelledError()

        monkeypatch.setattr(cli_module.asyncio, 'run', raise_cancelled)

        runner = CliRunner()
        result = runner.invoke(main, ['run', '--config', '/dev/null'])

        assert result.exit_code == 130, (
            f'expected exit code 130 (SIGINT/SIGTERM), got {result.exit_code}'
        )
        assert state['armed_with'] == SHUTDOWN_WATCHDOG_TIMEOUT_SECS, (
            f"expected armed_with={SHUTDOWN_WATCHDOG_TIMEOUT_SECS}, got {state['armed_with']}"
        )
        # Watchdog must NOT be disarmed — it guards interpreter shutdown after sys.exit(130).
        assert not state['disarm_called'], (
            'disarm() was called on CancelledError path — watchdog must remain armed to guard '
            'interpreter shutdown (threading._shutdown() joining non-daemon threads)'
        )

    def test_watchdog_armed_only_after_asyncio_run_returns(self, monkeypatch):
        """Watchdog must be armed AFTER asyncio.run() returns, not before.

        Arming before asyncio.run would start the 30-second timer at the
        beginning of orchestration; real runs take longer than 30s, so the
        watchdog would fire mid-run and kill the orchestrator during normal
        operation.

        This test captures the state of 'armed_with' INSIDE the fake
        asyncio.run call (i.e., while _main() is still "running") to prove
        the watchdog was not yet armed at that point.
        """
        state, fake_force_exit = self._fake_watchdog_factory()
        monkeypatch.setattr(cli_module, '_force_exit_after_delay', fake_force_exit)

        fake_config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        monkeypatch.setattr(cli_config, 'load_config', lambda _path: fake_config)

        fake_harness = MagicMock()
        fake_report = MagicMock()
        fake_report.blocked = 0
        fake_report.summary.return_value = 'all done'
        monkeypatch.setattr('orchestrator.harness.Harness', lambda config, **_: fake_harness)

        # Capture what 'armed_with' is while asyncio.run is "executing".
        armed_during_asyncio_run: list = []

        def fake_asyncio_run(coro):
            coro.close()
            # At this point asyncio.run is "in progress" — watchdog must NOT be armed yet.
            armed_during_asyncio_run.append(state['armed_with'])
            return fake_report

        monkeypatch.setattr(cli_module.asyncio, 'run', fake_asyncio_run)

        runner = CliRunner()
        runner.invoke(main, ['run', '--config', '/dev/null'])

        # Exactly one call happened (our fake_asyncio_run ran once).
        assert len(armed_during_asyncio_run) == 1, (
            f'fake asyncio.run was called {len(armed_during_asyncio_run)} times'
        )
        # While asyncio.run was "running", the watchdog must NOT have been armed.
        assert armed_during_asyncio_run[0] is None, (
            f'Watchdog was already armed with timeout={armed_during_asyncio_run[0]} '
            f'before/during asyncio.run — this would fire the watchdog during long-running '
            f'orchestration (>30s) and kill the orchestrator mid-run.'
        )
        # But AFTER asyncio.run returns, the watchdog must be armed.
        assert state['armed_with'] == SHUTDOWN_WATCHDOG_TIMEOUT_SECS, (
            f"expected armed_with={SHUTDOWN_WATCHDOG_TIMEOUT_SECS} after asyncio.run returned, "
            f"got {state['armed_with']}"
        )

    def test_report_emitted_before_watchdog_armed(self, monkeypatch):
        """click.echo(report.summary()) must run BEFORE _force_exit_after_delay on the normal path.

        Report formatting and stdout I/O (arbitrary size, arbitrary latency) must NOT be
        covered by the 30-second watchdog timer.  Scope the watchdog to interpreter shutdown
        only (atexit callbacks + threading._shutdown() joining non-daemon threads).
        """
        events: list[str | tuple[str, str | None]] = []
        state, fake_force_exit = self._fake_watchdog_factory(events=events)
        monkeypatch.setattr(cli_module, '_force_exit_after_delay', fake_force_exit)

        original_echo = cli_module.click.echo

        def recording_echo(msg=None, **kwargs):
            # Record every echo call so failures show what actually happened
            # and the test is robust to changes in the fake summary string.
            events.append(('echo', msg))
            return original_echo(msg, **kwargs)

        monkeypatch.setattr(cli_module.click, 'echo', recording_echo)

        fake_config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        monkeypatch.setattr(cli_config, 'load_config', lambda _path: fake_config)

        fake_harness = MagicMock()
        fake_report = MagicMock()
        fake_report.blocked = 0
        fake_report.summary.return_value = 'all done'
        monkeypatch.setattr('orchestrator.harness.Harness', lambda config, **_: fake_harness)

        def fake_asyncio_run(coro):
            coro.close()
            return fake_report

        monkeypatch.setattr(cli_module.asyncio, 'run', fake_asyncio_run)

        CliRunner().invoke(main, ['run', '--config', '/dev/null'])

        summary_str = fake_report.summary.return_value
        echo_indices = [i for i, e in enumerate(events) if e == ('echo', summary_str)]
        assert echo_indices, (
            f"click.echo(report.summary()) was never called — can't check ordering. "
            f"events: {events!r}"
        )
        arm_indices = [i for i, e in enumerate(events) if e == 'arm']
        assert arm_indices, (
            f'_force_exit_after_delay was never called — watchdog not armed at all. '
            f'events: {events!r}'
        )
        echo_idx = echo_indices[0]
        arm_idx = arm_indices[0]
        assert echo_idx < arm_idx, (
            f'click.echo(report.summary()) must run BEFORE _force_exit_after_delay, '
            f'but got order: {events!r}'
        )
        # Arm must still happen (just after user-visible work).
        assert state['armed_with'] == SHUTDOWN_WATCHDOG_TIMEOUT_SECS, (
            f"expected armed_with={SHUTDOWN_WATCHDOG_TIMEOUT_SECS}, got {state['armed_with']}"
        )

    def test_watchdog_armed_before_blocked_exit(self, monkeypatch):
        """_force_exit_after_delay must be called even when report.blocked > 0.

        The blocked-task exit (sys.exit(1)) goes through the same interpreter
        shutdown as the clean exit — atexit callbacks run and
        threading._shutdown() joins non-daemon threads from harness.run().
        Both paths carry identical hang risk, so the watchdog must guard both.

        The arm is placed AFTER click.echo(report.summary()) (not before, so
        user-visible output is not covered) but BEFORE the
        `if report.blocked > 0: sys.exit(1)` branch so the blocked path is
        covered too, matching the CancelledError branch's pattern.
        """
        state, fake_force_exit = self._fake_watchdog_factory()
        monkeypatch.setattr(cli_module, '_force_exit_after_delay', fake_force_exit)

        fake_config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        monkeypatch.setattr(cli_config, 'load_config', lambda _path: fake_config)

        fake_harness = MagicMock()
        fake_report = MagicMock()
        fake_report.blocked = 3  # non-zero forces the sys.exit(1) branch
        fake_report.summary.return_value = 'blocked'
        monkeypatch.setattr('orchestrator.harness.Harness', lambda config, **_: fake_harness)

        def fake_asyncio_run(coro):
            coro.close()
            return fake_report

        monkeypatch.setattr(cli_module.asyncio, 'run', fake_asyncio_run)

        result = CliRunner().invoke(main, ['run', '--config', '/dev/null'])

        assert result.exit_code == 1, (
            f'expected exit_code 1 (blocked tasks), got {result.exit_code}'
        )
        # Watchdog must be armed even on the blocked exit path.
        assert state['armed_with'] == SHUTDOWN_WATCHDOG_TIMEOUT_SECS, (
            f'_force_exit_after_delay was not called on the blocked>0 path — '
            f'watchdog must guard interpreter shutdown on both clean and blocked exits. '
            f'got armed_with={state["armed_with"]!r}'
        )
        # Watchdog must NOT be disarmed — it guards interpreter shutdown after sys.exit(1).
        assert not state['disarm_called'], (
            'disarm() was called on the blocked>0 path — watchdog must remain armed to '
            'guard interpreter shutdown (threading._shutdown() joining non-daemon threads)'
        )


class TestRunCapturesTheDeployedSha:
    """Task 6565 D1: ``run`` brackets the harness import and injects the sha."""

    def _invoke(self, monkeypatch, *, enabled_runners):
        monkeypatch.setattr(cli_module, '_force_exit_after_delay', lambda *a, **k: MagicMock())
        fake_config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        fake_config.enabled_verify_runners = enabled_runners
        monkeypatch.setattr(cli_config, 'load_config', lambda _path: fake_config)
        fake_report = MagicMock(blocked=0)
        fake_report.summary.return_value = 'ok'
        received: dict[str, object] = {}

        def fake_harness(config, **kwargs):
            received.update(kwargs)
            harness = MagicMock()
            harness.run = AsyncMock(return_value=fake_report)
            return harness

        monkeypatch.setattr('orchestrator.harness.Harness', fake_harness)
        result = CliRunner().invoke(cli_module.main, ['run', '--config', '/dev/null'])
        return result, received

    def test_harness_receives_the_checkout_head(self, monkeypatch):
        from orchestrator.deployed_sha import DeployedSha, checkout_head, resolve_code_checkout

        result, received = self._invoke(monkeypatch, enabled_runners=[])

        assert result.exit_code == 0, result.output
        deployed = received['deployed_sha']
        assert isinstance(deployed, DeployedSha)
        checkout = resolve_code_checkout()
        assert checkout is not None
        assert deployed.sha == checkout_head(checkout)

    def test_unreadable_sha_with_an_enabled_remote_runner_refuses_to_start(self, monkeypatch):
        monkeypatch.setattr('orchestrator.deployed_sha.resolve_code_checkout', lambda: None)

        result, received = self._invoke(monkeypatch, enabled_runners=[MagicMock()])

        assert result.exit_code == 1
        assert 'remote verify runner is enabled' in result.output
        assert received == {}

    def test_unreadable_sha_without_remote_runners_starts_with_none(self, monkeypatch):
        monkeypatch.setattr('orchestrator.deployed_sha.resolve_code_checkout', lambda: None)

        result, received = self._invoke(monkeypatch, enabled_runners=[])

        assert result.exit_code == 0, result.output
        assert received == {'deployed_sha': None}
