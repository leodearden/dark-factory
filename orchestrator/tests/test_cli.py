"""Tests for CLI helpers."""

import asyncio
import io
import logging
import threading
import time
import traceback as traceback_module
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

import orchestrator.cli as cli_module
from orchestrator.cli import (
    SHUTDOWN_WATCHDOG_TIMEOUT_SECS,
    _force_exit_after_delay,
    _make_cancel_handler,
    _parse_duration,
    main,
)


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

    def test_fires_after_timeout(self, monkeypatch):
        """Watchdog calls os._exit(137) after the timeout elapses."""
        calls = []
        monkeypatch.setattr('os._exit', lambda code: calls.append(code))

        handle = _force_exit_after_delay(timeout_secs=0.05)

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

    def test_does_not_fire_before_timeout(self, monkeypatch):
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
        """
        calls = []
        monkeypatch.setattr('os._exit', lambda code: calls.append(code))

        handle = _force_exit_after_delay(timeout_secs=2.0)

        # Sleep well within the timeout — 0.2s is 10x margin under any scheduler load.
        time.sleep(0.2)

        assert calls == [], (
            f'watchdog fired before timeout elapsed (clean-exit window): {calls}'
        )

        # Cleanup: disarm and join to guarantee thread exits cleanly.
        handle.disarm()
        handle.thread.join(timeout=1.0)
        assert not handle.thread.is_alive(), (
            'watchdog thread did not exit after disarm'
        )

    def test_disarm_prevents_force_exit(self, monkeypatch):
        """Calling disarm() before timeout prevents os._exit from being called."""
        calls = []
        monkeypatch.setattr('os._exit', lambda code: calls.append(code))

        # Use 0.2s timeout and a 2.0s wait to give ample margin under CI load;
        # disarm() sets the event immediately so the watchdog thread returns
        # without calling os._exit even if the scheduler is delayed.
        handle = _force_exit_after_delay(timeout_secs=0.2)
        handle.disarm()
        time.sleep(2.0)

        assert calls == [], f'expected no calls, got {calls}'
        handle.thread.join(timeout=1.0)
        assert not handle.thread.is_alive(), (
            'watchdog thread did not exit after disarm'
        )

    def test_diagnostic_dump_lists_live_threads(self, monkeypatch):
        """When the watchdog fires, it writes a diagnostic dump to the stream."""
        calls = []
        monkeypatch.setattr('os._exit', lambda code: calls.append(code))

        stream = io.StringIO()
        handle = _force_exit_after_delay(timeout_secs=0.05, stream=stream)

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
        in the watchdog catches any failure and still reaches os._exit — so the
        process always escapes a hang even if the diagnostic output is lost.
        This test locks in the 'fail-open to force-exit' guarantee.
        """
        calls = []
        monkeypatch.setattr('os._exit', lambda code: calls.append(code))

        # Simulate partial interpreter shutdown where traceback.format_stack is broken.
        def _raise_on_format(*args, **kwargs):
            raise RuntimeError('simulated shutdown tear-down of traceback module')

        monkeypatch.setattr(traceback_module, 'format_stack', _raise_on_format)

        stream = io.StringIO()
        handle = _force_exit_after_delay(timeout_secs=0.05, stream=stream)

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

    def test_force_exit_returns_handle_with_disarm_and_thread(self, monkeypatch):
        """_force_exit_after_delay returns a WatchdogHandle with .disarm and .thread.

        The returned handle must have:
        - a callable .disarm attribute
        - a threading.Thread .thread attribute that is daemonised, named
          'shutdown-watchdog', and already alive after arming.
        Calling disarm() must stop the thread within a reasonable timeout.
        """
        calls = []
        monkeypatch.setattr('os._exit', lambda code: calls.append(code))

        handle = _force_exit_after_delay(timeout_secs=5.0)

        assert callable(handle.disarm), (
            f'handle.disarm must be callable, got {handle.disarm!r}'
        )
        assert isinstance(handle.thread, threading.Thread), (
            f'handle.thread must be threading.Thread, got {type(handle.thread)}'
        )
        assert handle.thread.daemon is True, (
            'handle.thread must be a daemon thread'
        )
        assert handle.thread.name == 'shutdown-watchdog', (
            f'expected thread name "shutdown-watchdog", got {handle.thread.name!r}'
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
        so a stream-write failure still falls through to os._exit".  This test
        locks in that doubly-defensive contract by making traceback.format_stack
        *and* stream.write both raise unconditionally.  If the nested try/except
        is ever removed, this test fails before the force-exit guarantee is silently
        dropped.
        """
        calls = []
        monkeypatch.setattr('os._exit', lambda code: calls.append(code))

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

        handle = _force_exit_after_delay(timeout_secs=0.05, stream=_BrokenStream())

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

    def test_outer_write_failure_still_fires_exit(self, monkeypatch):
        """os._exit(137) is called even when out.write(''.join(lines)) raises with format_stack intact.

        Closes the coverage gap in test_fallback_write_failure_still_fires_exit: that sibling
        test mocks traceback.format_stack to raise FIRST, so `lines` is never built and the
        outer out.write(''.join(lines)) call is never reached.  This test covers the complement:
        format_stack succeeds, `lines` is populated with real frames, but out.write raises
        (outer except catches), then the inner fallback write also raises (_OuterBrokenStream
        makes every write raise), the inner except swallows, and execution falls through to
        os._exit(137).

        If the outer try/except wrapping out.write is ever removed, this test will fail before
        the force-exit guarantee is silently dropped — mirroring the sibling test's contract
        for the nested try/except.
        """
        calls = []
        monkeypatch.setattr('os._exit', lambda code: calls.append(code))

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

        handle = _force_exit_after_delay(timeout_secs=0.05, stream=_OuterBrokenStream())

        # Poll with deadline — os._exit must fire even when the outer write fails.
        deadline = time.monotonic() + 5.0
        while not calls and time.monotonic() < deadline:
            time.sleep(0.05)

        assert calls == [137], (
            f'expected os._exit(137) even when outer write fails with format_stack intact, '
            f'got {calls}'
        )
        # Verify the control-flow path: format_stack built real frames so the outer
        # out.write(''.join(lines)) was attempted first (must contain the watchdog-fired
        # header sentinel, proving the outer path was reached), then the inner fallback
        # out.write was attempted second (must contain the dump-failed sentinel).  Both
        # raised — inner except swallowed — os._exit still reached.  This distinguishes
        # the test from its sibling test_fallback_write_failure_still_fires_exit where
        # format_stack raises first (lines is never built, outer out.write is never reached).
        assert len(write_attempts) >= 2, (
            f'expected at least 2 write attempts (outer sentinel + inner fallback), '
            f'got {write_attempts!r}'
        )
        assert 'SHUTDOWN WATCHDOG FIRED — process hung' in write_attempts[0], (
            f'first write attempt must contain the watchdog-fired header sentinel '
            f'(proves format_stack ran and outer out.write was reached), '
            f'got: {write_attempts[0]!r}'
        )
        assert '(diagnostic dump failed)' in write_attempts[1], (
            f'second write attempt must contain the fallback sentinel '
            f'(proves inner fallback out.write was also exercised), '
            f'got: {write_attempts[1]!r}'
        )
        handle.thread.join(timeout=1.0)
        assert not handle.thread.is_alive(), (
            'watchdog thread did not exit after firing os._exit replacement'
        )


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
        fake_config = MagicMock()
        monkeypatch.setattr(cli_module, 'load_config', lambda _path: fake_config)

        # Fake Harness so we don't need a real one
        fake_harness = MagicMock()
        fake_report = MagicMock()
        fake_report.blocked = 0
        fake_report.summary.return_value = 'all done'
        fake_harness.run = MagicMock()

        monkeypatch.setattr('orchestrator.harness.Harness', lambda config: fake_harness)

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

        fake_config = MagicMock()
        monkeypatch.setattr(cli_module, 'load_config', lambda _path: fake_config)

        fake_harness = MagicMock()
        monkeypatch.setattr('orchestrator.harness.Harness', lambda config: fake_harness)

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

        fake_config = MagicMock()
        monkeypatch.setattr(cli_module, 'load_config', lambda _path: fake_config)

        fake_harness = MagicMock()
        fake_report = MagicMock()
        fake_report.blocked = 0
        fake_report.summary.return_value = 'all done'
        monkeypatch.setattr('orchestrator.harness.Harness', lambda config: fake_harness)

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

        fake_config = MagicMock()
        monkeypatch.setattr(cli_module, 'load_config', lambda _path: fake_config)

        fake_harness = MagicMock()
        fake_report = MagicMock()
        fake_report.blocked = 0
        fake_report.summary.return_value = 'all done'
        monkeypatch.setattr('orchestrator.harness.Harness', lambda config: fake_harness)

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

        fake_config = MagicMock()
        monkeypatch.setattr(cli_module, 'load_config', lambda _path: fake_config)

        fake_harness = MagicMock()
        fake_report = MagicMock()
        fake_report.blocked = 3  # non-zero forces the sys.exit(1) branch
        fake_report.summary.return_value = 'blocked'
        monkeypatch.setattr('orchestrator.harness.Harness', lambda config: fake_harness)

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
