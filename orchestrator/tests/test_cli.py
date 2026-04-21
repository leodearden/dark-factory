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

        _force_exit_after_delay(timeout_secs=0.05)

        # Poll with a deadline instead of a fixed sleep to avoid spurious
        # failures under CI scheduler stalls (GC, container contention, etc.).
        deadline = time.monotonic() + 5.0
        while not calls and time.monotonic() < deadline:
            time.sleep(0.05)

        assert calls == [137], f'expected [137], got {calls}'

    def test_disarm_prevents_force_exit(self, monkeypatch):
        """Calling disarm() before timeout prevents os._exit from being called."""
        calls = []
        monkeypatch.setattr('os._exit', lambda code: calls.append(code))

        # Use 0.2s timeout and a 2.0s wait to give ample margin under CI load;
        # disarm() sets the event immediately so the watchdog thread returns
        # without calling os._exit even if the scheduler is delayed.
        disarm = _force_exit_after_delay(timeout_secs=0.2)
        disarm()
        time.sleep(2.0)

        assert calls == [], f'expected no calls, got {calls}'

    def test_diagnostic_dump_lists_live_threads(self, monkeypatch):
        """When the watchdog fires, it writes a diagnostic dump to the stream."""
        calls = []
        monkeypatch.setattr('os._exit', lambda code: calls.append(code))

        stream = io.StringIO()
        _force_exit_after_delay(timeout_secs=0.05, stream=stream)

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
        _force_exit_after_delay(timeout_secs=0.05, stream=stream)

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

        _force_exit_after_delay(timeout_secs=0.05, stream=_BrokenStream())

        # Poll with deadline — os._exit must fire even when the fallback write also fails.
        deadline = time.monotonic() + 5.0
        while not calls and time.monotonic() < deadline:
            time.sleep(0.05)

        assert calls == [137], (
            f'expected os._exit(137) even when fallback write fails, got {calls}'
        )


class TestRunArmsWatchdog:
    """run() must arm the shutdown watchdog and disarm it on both exit paths."""

    def _fake_watchdog_factory(self):
        """Returns (recorder, fake_force_exit_after_delay).

        recorder has:
          .armed_with       – the timeout_secs passed on arming
          .disarm_called    – True once disarm() is called
        """
        state = {'armed_with': None, 'disarm_called': False}

        def fake_disarm():
            state['disarm_called'] = True

        def fake_force_exit(timeout_secs, exit_code=137, *, stream=None):
            state['armed_with'] = timeout_secs
            return fake_disarm

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
        """click.echo(report.summary()) must run BEFORE _force_exit_after_delay on normal path.

        On the current (unfixed) code the arm runs before click.echo(report.summary()), so
        report formatting and stdout I/O are covered by the 30-second timer. Moving the arm
        after all user-visible work limits the watchdog scope to interpreter shutdown only.

        This test FAILS on current code (arm at line 208 precedes echo at line 209).
        """
        events: list[str] = []
        state = {'armed_with': None}

        def recording_force_exit(timeout_secs, exit_code=137, *, stream=None):
            state['armed_with'] = timeout_secs
            events.append('arm')
            return lambda: None

        monkeypatch.setattr(cli_module, '_force_exit_after_delay', recording_force_exit)

        original_echo = cli_module.click.echo

        def recording_echo(msg=None, **kwargs):
            if msg == 'all done':
                events.append('echo_summary')
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

        assert 'echo_summary' in events, (
            "click.echo(report.summary()) was never called — can't check ordering"
        )
        assert 'arm' in events, (
            '_force_exit_after_delay was never called — watchdog not armed at all'
        )
        echo_idx = events.index('echo_summary')
        arm_idx = events.index('arm')
        assert echo_idx < arm_idx, (
            f'click.echo(report.summary()) must run BEFORE _force_exit_after_delay, '
            f'but got order: {events!r}'
        )
        # Arm must still happen (just after user-visible work).
        assert state['armed_with'] == SHUTDOWN_WATCHDOG_TIMEOUT_SECS, (
            f"expected armed_with={SHUTDOWN_WATCHDOG_TIMEOUT_SECS}, got {state['armed_with']}"
        )

    def test_watchdog_not_armed_on_blocked_exit(self, monkeypatch):
        """_force_exit_after_delay must NOT be called when report.blocked > 0.

        When the orchestrator exits via sys.exit(1) (blocked tasks), the arm call
        at the fall-through position is never reached. Only interpreter shutdown
        after a *successful* run is guarded by the watchdog. The CancelledError
        branch arms separately before its own sys.exit(130), so SIGTERM/SIGINT
        shutdown remains covered.

        This test FAILS against the intermediate impl where the arm sits between
        click.echo and sys.exit(1); it passes only when the arm is placed AFTER
        the sys.exit(1) branch (fall-through position).
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
        assert state['armed_with'] is None, (
            f'_force_exit_after_delay was called with timeout={state["armed_with"]} '
            f'on the blocked>0 path — watchdog must NOT be armed before sys.exit(1); '
            f'arm must be placed after the sys.exit(1) branch (fall-through only)'
        )
