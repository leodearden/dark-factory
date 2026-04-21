"""Tests for CLI helpers."""

import asyncio
import io
import logging
import threading
import time
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
        time.sleep(0.3)

        assert calls == [137], f'expected [137], got {calls}'

    def test_disarm_prevents_force_exit(self, monkeypatch):
        """Calling disarm() before timeout prevents os._exit from being called."""
        calls = []
        monkeypatch.setattr('os._exit', lambda code: calls.append(code))

        disarm = _force_exit_after_delay(timeout_secs=0.3)
        disarm()
        time.sleep(0.6)

        assert calls == [], f'expected no calls, got {calls}'

    def test_diagnostic_dump_lists_live_threads(self, monkeypatch):
        """When the watchdog fires, it writes a diagnostic dump to the stream."""
        calls = []
        monkeypatch.setattr('os._exit', lambda code: calls.append(code))

        stream = io.StringIO()
        _force_exit_after_delay(timeout_secs=0.05, stream=stream)
        time.sleep(0.3)

        output = stream.getvalue()
        # Sentinel header must be present
        assert 'SHUTDOWN WATCHDOG FIRED' in output, (
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

    def test_normal_path_arms_and_disarms(self, monkeypatch):
        """run() arms the watchdog and disarms it on normal (non-cancelled) exit."""
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
        # And disarmed before exit
        assert state['disarm_called'], 'disarm() was never called on normal exit path'

    def test_cancelled_path_arms_and_disarms(self, monkeypatch):
        """run() arms the watchdog and disarms it even when asyncio.run raises CancelledError."""
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
        assert state['disarm_called'], 'disarm() was never called on CancelledError exit path'
