"""Tests for CLI helpers."""

import asyncio
import io
import logging
import subprocess
import threading
import time
import traceback as traceback_module
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from _orch_helpers import pydantic_spec
from click.testing import CliRunner

import orchestrator.cli as cli_module
from orchestrator.cli import (
    SHUTDOWN_WATCHDOG_TIMEOUT_SECS,
    _force_exit_after_delay,
    _make_cancel_handler,
    _parse_duration,
    main,
)
from orchestrator.config import OrchestratorConfig


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
        calls = []
        monkeypatch.setattr('os._exit', lambda code: calls.append(code))

        handle = _force_exit_after_delay(timeout_secs=5.0)

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
        so a stream-write failure still falls through to os._exit".  This test
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


class TestRunUntilIdleFlag:
    """The --until-idle flag must thread through to harness.run(until_idle=...).

    Default (flag absent) is run-forever (until_idle=False); the flag opts into
    legacy exit-on-drain (until_idle=True).
    """

    def _setup(self, monkeypatch, fake_report):
        """Wire mocks so the real _main coroutine runs and calls harness.run once."""
        # Stub the watchdog so no real daemon timer thread is spawned.
        monkeypatch.setattr(
            cli_module, '_force_exit_after_delay', lambda *a, **k: MagicMock()
        )
        fake_config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        monkeypatch.setattr(cli_module, 'load_config', lambda _path: fake_config)

        fake_harness = MagicMock()
        # AsyncMock so `await harness.run(...)` inside the real _main resolves.
        from unittest.mock import AsyncMock
        fake_harness.run = AsyncMock(return_value=fake_report)
        monkeypatch.setattr('orchestrator.harness.Harness', lambda config: fake_harness)
        return fake_harness

    def test_flag_threads_until_idle_true(self, monkeypatch):
        fake_report = MagicMock()
        fake_report.blocked = 0
        fake_report.summary.return_value = 'ok'
        fake_harness = self._setup(monkeypatch, fake_report)

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
        fake_harness = self._setup(monkeypatch, fake_report)

        result = CliRunner().invoke(main, ['run', '--config', '/dev/null'])

        assert result.exit_code == 0, result.output
        assert fake_harness.run.call_args.kwargs.get('until_idle') is False, (
            f'default run must pass until_idle=False (run forever); '
            f'kwargs={fake_harness.run.call_args.kwargs!r}'
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
        fake_config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
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

        fake_config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
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

        fake_config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
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

        fake_config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
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

        fake_config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
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


# ---------------------------------------------------------------------------
# Step-7: `orchestrator verify-merge` integration tests (parity + cleanup + errors)
# ---------------------------------------------------------------------------


def _setup_verify_repo(tmp_path: Path):
    """Init a minimal git repo with a mod/test_x.py file, return (repo, head_sha)."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    p = str(repo)
    subprocess.run(['git', 'init', '-b', 'main', p], check=True, capture_output=True)
    subprocess.run(['git', '-C', p, 'config', 'user.name', 'Test User'],
                   check=True, capture_output=True)
    subprocess.run(['git', '-C', p, 'config', 'user.email', 'test@example.com'],
                   check=True, capture_output=True)
    mod_dir = repo / 'mod'
    mod_dir.mkdir()
    (mod_dir / 'test_x.py').write_text('# placeholder\n')
    subprocess.run(['git', '-C', p, 'add', '.'], check=True, capture_output=True)
    subprocess.run(['git', '-C', p, 'commit', '-m', 'initial commit'],
                   check=True, capture_output=True)
    result = subprocess.run(['git', '-C', p, 'rev-parse', 'HEAD'],
                            check=True, capture_output=True, text=True)
    head_sha = result.stdout.strip()
    return repo, head_sha


@pytest.mark.parametrize('test_command,expect_pass', [('true', True), ('false', False)])
def test_verify_merge_cli_wrapper_transparency(tmp_path, monkeypatch, test_command, expect_pass):
    """CLI verify-merge is a transparent wrapper: JSON round-trip is lossless and exit code is clean.

    Both the local baseline and the CLI call run_merge_verify_on_worktree on the same
    SHA with the same spec, so the test validates that the CLI scaffolding (config load,
    spec parse, worktree create/cleanup, JSON serialise) adds no observable difference.
    It does *not* validate parity against the merge-queue dispatch path directly.

    SYNC test (asyncio_mode=auto): the local result is computed via asyncio.run()
    before invoking CliRunner so we never nest event loops.
    """
    from orchestrator.git_ops import GitOps
    from orchestrator.verify_runner import (
        MergeVerifySpec,
        UnscopedTypecheckSpec,
        VerifyCommand,
        result_from_json,
        run_merge_verify_on_worktree,
        spec_to_json,
    )

    repo, head_sha = _setup_verify_repo(tmp_path)
    config = OrchestratorConfig(project_root=repo)
    monkeypatch.setattr(cli_module, 'load_config', lambda _: config)

    # A dummy config file just to satisfy click.Path(exists=True)
    cfg_file = tmp_path / 'dummy.yaml'
    cfg_file.write_text('')

    spec = MergeVerifySpec(
        verify_commands=(VerifyCommand('mod', test_command=test_command),),
        unscoped_typecheck=UnscopedTypecheckSpec(
            commands=(VerifyCommand('mod', type_check_command='true'),),
            block_on_timeout=True,
        ),
        task_files=('mod/test_x.py',),
        verify_env={},
        cold_timeout_secs=300.0,
    )

    # Compute local result in-process (SYNC: use asyncio.run to avoid nested loop)
    async def _local_run():
        git_ops = GitOps(config.git, repo)
        wt, _ = await git_ops._create_merge_worktree(base_sha=head_sha)
        try:
            return await run_merge_verify_on_worktree(wt, config, spec)
        finally:
            await git_ops.cleanup_merge_worktree(wt)

    local = asyncio.run(_local_run())

    # Invoke CLI
    r = CliRunner().invoke(main, [
        'verify-merge',
        '--sha', head_sha,
        '--spec', spec_to_json(spec),
        '--config', str(cfg_file),
    ])

    assert r.exit_code == 0, (
        f'expected exit_code 0, got {r.exit_code}; output={r.output!r}'
    )
    cli_result = result_from_json(r.output)
    assert cli_result == local, (
        f'CLI result != local result: cli={cli_result!r}, local={local!r}'
    )
    assert local.passed is expect_pass


def test_verify_merge_cleanup(tmp_path, monkeypatch):
    """verify-merge must not leak _merge-* worktrees after the run completes."""
    from orchestrator.verify_runner import (
        MergeVerifySpec,
        UnscopedTypecheckSpec,
        VerifyCommand,
        spec_to_json,
    )

    repo, head_sha = _setup_verify_repo(tmp_path)
    config = OrchestratorConfig(project_root=repo)
    monkeypatch.setattr(cli_module, 'load_config', lambda _: config)

    cfg_file = tmp_path / 'dummy.yaml'
    cfg_file.write_text('')

    spec = MergeVerifySpec(
        verify_commands=(VerifyCommand('mod', test_command='true'),),
        unscoped_typecheck=UnscopedTypecheckSpec(
            commands=(VerifyCommand('mod', type_check_command='true'),),
            block_on_timeout=True,
        ),
        task_files=('mod/test_x.py',),
        verify_env={},
        cold_timeout_secs=300.0,
    )

    r = CliRunner().invoke(main, [
        'verify-merge',
        '--sha', head_sha,
        '--spec', spec_to_json(spec),
        '--config', str(cfg_file),
    ])
    assert r.exit_code == 0, f'expected exit_code 0, got {r.exit_code}; output={r.output!r}'

    # No _merge-* directories should remain under .worktrees
    worktrees_dir = repo / '.worktrees'
    leaked = (
        any(p.name.startswith('_merge-') for p in worktrees_dir.iterdir())
        if worktrees_dir.exists()
        else False
    )
    assert not leaked, f'leaked worktree under {worktrees_dir}'

    # git worktree list must not show any _merge-* entry
    wt_list = subprocess.run(
        ['git', '-C', str(repo), 'worktree', 'list', '--porcelain'],
        check=True, capture_output=True, text=True,
    ).stdout
    assert '_merge-' not in wt_list, f'git worktree list shows leaked entry:\n{wt_list}'


@pytest.mark.parametrize('bad_case', ['malformed_spec', 'absent_sha'])
def test_verify_merge_clean_error_contract(tmp_path, monkeypatch, bad_case):
    """Bad requests never emit a non-verdict to stdout and always exit non-zero.

    Case A (malformed --spec): JSON parse error → graceful non-zero, empty stdout.
    Case B (absent SHA): non-existent commit → graceful non-zero, empty stdout.
    Both cases must have a concise message on stderr and no uncaught traceback.
    """
    from orchestrator.verify_runner import (
        MergeVerifySpec,
        UnscopedTypecheckSpec,
        VerifyCommand,
        result_from_json,
        spec_to_json,
    )

    repo, head_sha = _setup_verify_repo(tmp_path)
    config = OrchestratorConfig(project_root=repo)
    monkeypatch.setattr(cli_module, 'load_config', lambda _: config)

    cfg_file = tmp_path / 'dummy.yaml'
    cfg_file.write_text('')

    good_spec = MergeVerifySpec(
        verify_commands=(VerifyCommand('mod', test_command='true'),),
        unscoped_typecheck=UnscopedTypecheckSpec(
            commands=(VerifyCommand('mod', type_check_command='true'),),
            block_on_timeout=True,
        ),
        task_files=('mod/test_x.py',),
        verify_env={},
        cold_timeout_secs=300.0,
    )

    if bad_case == 'malformed_spec':
        invoke_args = [
            'verify-merge',
            '--sha', head_sha,
            '--spec', 'not valid json',
            '--config', str(cfg_file),
        ]
    else:  # absent_sha
        invoke_args = [
            'verify-merge',
            '--sha', '0' * 40,
            '--spec', spec_to_json(good_spec),
            '--config', str(cfg_file),
        ]

    r = CliRunner().invoke(main, invoke_args)

    # Must exit non-zero
    assert r.exit_code != 0, (
        f'expected non-zero exit_code for {bad_case}, got {r.exit_code}'
    )
    # stdout must not contain a parseable VerifyResult
    # (click 8.3.2: r.output mixes stdout+stderr; use result_from_json to prove
    # no valid verdict was emitted)
    with pytest.raises((ValueError, TypeError)):
        result_from_json(r.output)
    # No uncaught traceback — exception must be SystemExit (or None)
    assert r.exception is None or isinstance(r.exception, SystemExit), (
        f'expected SystemExit or None, got {type(r.exception).__name__}: {r.exception}'
    )
    # A concise error message on stderr
    assert r.stderr.strip() != '', (
        f'expected non-empty stderr for {bad_case!r}'
    )


# ---------------------------------------------------------------------------
# Task 1699 step-7 — verify-merge subcommand wiring: acquire_host_verify_worktree
# ---------------------------------------------------------------------------


def test_verify_merge_uses_acquire_host_verify_worktree(tmp_path, monkeypatch):
    """verify-merge must call acquire_host_verify_worktree, NOT _create_merge_worktree.

    Step-7 (RED): the subcommand currently calls _create_merge_worktree;
    _create_merge_worktree.assert_not_called() fails until step-8 swaps it.
    """
    from unittest.mock import AsyncMock, MagicMock

    sha = 'abc1234567890abc1234567890abc1234567890ab'
    fake_wt = tmp_path / '_merge-verify'
    fake_wt.mkdir()

    # --- Build mock GitOps instance ---
    mock_git_ops = MagicMock()
    mock_git_ops.acquire_host_verify_worktree = AsyncMock(return_value=fake_wt)
    mock_git_ops.cleanup_merge_worktree = AsyncMock(return_value=None)
    # _create_merge_worktree is a spy: return a valid 2-tuple so the old code
    # path doesn't crash today, but we assert it's NOT called after step-8.
    mock_git_ops._create_merge_worktree = AsyncMock(return_value=(fake_wt, sha))

    # Patch GitOps class so instantiation returns our mock
    monkeypatch.setattr('orchestrator.git_ops.GitOps', MagicMock(return_value=mock_git_ops))

    # --- Config with persistent_merge_worktree=True ---
    from orchestrator.config import GitConfig, OrchestratorConfig
    git_cfg = GitConfig(persistent_merge_worktree=True)
    fake_config = OrchestratorConfig(project_root=tmp_path, git=git_cfg)
    monkeypatch.setattr(cli_module, 'load_config', lambda _: fake_config)

    # --- Patch verify_runner helpers ---
    known_json = '{"passed": true, "results": []}'
    monkeypatch.setattr(
        'orchestrator.verify_runner.spec_from_json',
        lambda s: MagicMock(),
    )
    monkeypatch.setattr(
        'orchestrator.verify_runner.run_merge_verify_on_worktree',
        AsyncMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        'orchestrator.verify_runner.result_to_json',
        lambda r: known_json,
    )

    # --- Dummy config file to satisfy click.Path(exists=True) ---
    cfg_file = tmp_path / 'config.yaml'
    cfg_file.write_text('')

    r = CliRunner().invoke(main, [
        'verify-merge',
        '--sha', sha,
        '--spec', '{}',
        '--config', str(cfg_file),
    ])

    assert r.exit_code == 0, (
        f'expected exit_code 0, got {r.exit_code}; output={r.output!r}'
    )
    assert known_json in r.output, (
        f'stdout must contain the known JSON; got: {r.output!r}'
    )

    # Core assertion: acquire_host_verify_worktree called; _create_merge_worktree NOT
    mock_git_ops.acquire_host_verify_worktree.assert_awaited_once_with(sha)
    mock_git_ops._create_merge_worktree.assert_not_called()


# ---------------------------------------------------------------------------
# Task 1732 step-9 — verify-merge --request-id pgid lifecycle + back-compat
# ---------------------------------------------------------------------------


def test_verify_merge_request_id_pgid_lifecycle(tmp_path, monkeypatch):
    """verify-merge --request-id writes pgid file during run, removes it on exit.

    Injects start_own_process_group as a spy; mocks all IO so no real git/build
    work happens. Checks file existence mid-run via the mocked
    run_merge_verify_on_worktree coroutine.
    """
    from unittest.mock import AsyncMock, MagicMock

    from orchestrator.verify_cancel import pgid_file

    FAKE_PGID = 77777
    FAKE_REQUEST_ID = 'test-req-1732'
    known_json = '{"passed": true, "results": []}'

    # worktree_base that GitOps would compute
    fake_worktree_base = tmp_path / '.worktrees'

    # --- Mock GitOps ---
    fake_wt = tmp_path / '_merge-verify'
    fake_wt.mkdir()
    mock_git_ops = MagicMock()
    mock_git_ops.worktree_base = fake_worktree_base
    mock_git_ops.acquire_host_verify_worktree = AsyncMock(return_value=fake_wt)
    mock_git_ops.cleanup_merge_worktree = AsyncMock(return_value=None)
    monkeypatch.setattr('orchestrator.git_ops.GitOps', MagicMock(return_value=mock_git_ops))

    # --- Mock config ---
    from orchestrator.config import OrchestratorConfig
    fake_config = OrchestratorConfig(project_root=tmp_path)
    monkeypatch.setattr(cli_module, 'load_config', lambda _: fake_config)

    # --- Spy on start_own_process_group ---
    sopg_calls = []

    def fake_sopg():
        sopg_calls.append(True)
        return FAKE_PGID

    monkeypatch.setattr(cli_module, 'start_own_process_group', fake_sopg)

    # --- Mock verify_runner helpers; capture pgid file existence mid-run ---
    pgf = pgid_file(fake_worktree_base, FAKE_REQUEST_ID)
    file_existed_mid_run = []

    async def fake_run_merge_verify(wt, cfg, spec, merge_sha=None):
        file_existed_mid_run.append(pgf.exists())
        result = MagicMock()
        return result

    monkeypatch.setattr('orchestrator.verify_runner.spec_from_json', lambda s: MagicMock())
    monkeypatch.setattr('orchestrator.verify_runner.run_merge_verify_on_worktree', fake_run_merge_verify)
    monkeypatch.setattr('orchestrator.verify_runner.result_to_json', lambda r: known_json)

    cfg_file = tmp_path / 'config.yaml'
    cfg_file.write_text('')

    sha = 'abc1234567890abc1234567890abc1234567890ab'
    r = CliRunner().invoke(main, [
        'verify-merge',
        '--sha', sha,
        '--spec', '{}',
        '--config', str(cfg_file),
        '--request-id', FAKE_REQUEST_ID,
    ])

    assert r.exit_code == 0, f'expected exit_code 0, got {r.exit_code}; output={r.output!r}'
    assert known_json in r.output

    # start_own_process_group must have been called once
    assert len(sopg_calls) == 1, 'start_own_process_group must be called once with --request-id'

    # pgid file must have existed during the run
    assert file_existed_mid_run == [True], (
        f'pgid file was not present mid-run; file_existed_mid_run={file_existed_mid_run!r}'
    )

    # pgid file must be removed after normal exit
    assert not pgf.exists(), f'pgid file must be removed on normal exit; still present at {pgf}'


def test_verify_merge_no_request_id_back_compat(tmp_path, monkeypatch):
    """Without --request-id, start_own_process_group is NOT called and no pgid file created.

    Verifies today's exact behavior is unchanged (back-compat).
    """
    from unittest.mock import AsyncMock, MagicMock

    # --- Mock GitOps ---
    fake_wt = tmp_path / '_merge-verify'
    fake_wt.mkdir()
    mock_git_ops = MagicMock()
    mock_git_ops.acquire_host_verify_worktree = AsyncMock(return_value=fake_wt)
    mock_git_ops.cleanup_merge_worktree = AsyncMock(return_value=None)
    monkeypatch.setattr('orchestrator.git_ops.GitOps', MagicMock(return_value=mock_git_ops))

    # --- Mock config ---
    from orchestrator.config import OrchestratorConfig
    fake_config = OrchestratorConfig(project_root=tmp_path)
    monkeypatch.setattr(cli_module, 'load_config', lambda _: fake_config)

    # --- Spy on start_own_process_group to ensure it is NOT called ---
    sopg_calls = []

    def fake_sopg():
        sopg_calls.append(True)
        return 99999

    monkeypatch.setattr(cli_module, 'start_own_process_group', fake_sopg)

    # --- Mock verify_runner helpers ---
    known_json = '{"passed": false, "results": []}'
    monkeypatch.setattr('orchestrator.verify_runner.spec_from_json', lambda s: MagicMock())
    monkeypatch.setattr(
        'orchestrator.verify_runner.run_merge_verify_on_worktree',
        AsyncMock(return_value=MagicMock()),
    )
    monkeypatch.setattr('orchestrator.verify_runner.result_to_json', lambda r: known_json)

    cfg_file = tmp_path / 'config.yaml'
    cfg_file.write_text('')

    sha = 'abc1234567890abc1234567890abc1234567890ab'
    r = CliRunner().invoke(main, [
        'verify-merge',
        '--sha', sha,
        '--spec', '{}',
        '--config', str(cfg_file),
    ])

    assert r.exit_code == 0, f'expected exit_code 0, got {r.exit_code}; output={r.output!r}'

    # start_own_process_group must NOT have been called (no --request-id)
    assert sopg_calls == [], (
        'start_own_process_group must NOT be called without --request-id'
    )

    # No .merge_verify_pgids directory should have been created
    from orchestrator.verify_cancel import PGID_DIR_NAME
    pgid_dir_path = tmp_path / '.worktrees' / PGID_DIR_NAME
    assert not pgid_dir_path.exists(), (
        f'.merge_verify_pgids directory created without --request-id: {pgid_dir_path}'
    )


# ---------------------------------------------------------------------------
# Task 1732 step-11 — cancel-verify subcommand (CliRunner)
# ---------------------------------------------------------------------------


def test_cancel_verify_unknown_id_exits_0(tmp_path, monkeypatch):
    """(a) Unknown request id (no pgid file present) -> exit 0 (idempotent)."""
    from unittest.mock import MagicMock

    from orchestrator.config import OrchestratorConfig

    # Minimal config: project_root -> tmp_path (worktree_base = tmp_path/.worktrees)
    fake_config = OrchestratorConfig(project_root=tmp_path)
    monkeypatch.setattr(cli_module, 'load_config', lambda _: fake_config)

    # GitOps mock returning a worktree_base that has no pgid file
    fake_worktree_base = tmp_path / '.worktrees'
    mock_git_ops = MagicMock()
    mock_git_ops.worktree_base = fake_worktree_base
    monkeypatch.setattr('orchestrator.git_ops.GitOps', MagicMock(return_value=mock_git_ops))

    cfg_file = tmp_path / 'config.yaml'
    cfg_file.write_text('')

    r = CliRunner().invoke(main, [
        'cancel-verify',
        '--request-id', 'nonexistent-req',
        '--config', str(cfg_file),
    ])
    assert r.exit_code == 0, (
        f'expected exit_code 0 for unknown id, got {r.exit_code}; output={r.output!r}'
    )


def test_cancel_verify_rc_wiring(tmp_path, monkeypatch):
    """(b) cancel_request spy: CLI passes pgid_file path and exits with exactly that rc."""
    from unittest.mock import MagicMock

    from orchestrator.config import OrchestratorConfig
    from orchestrator.verify_cancel import pgid_file

    FAKE_REQUEST_ID = 'wiring-test-req'
    EXPECTED_RC = 42  # unusual value to prove propagation

    fake_worktree_base = tmp_path / '.worktrees'
    expected_path = pgid_file(fake_worktree_base, FAKE_REQUEST_ID)

    cancel_calls = []

    def fake_cancel_request(path, **kwargs):
        cancel_calls.append(path)
        return EXPECTED_RC

    monkeypatch.setattr(cli_module, 'cancel_request', fake_cancel_request)

    fake_config = OrchestratorConfig(project_root=tmp_path)
    monkeypatch.setattr(cli_module, 'load_config', lambda _: fake_config)

    mock_git_ops = MagicMock()
    mock_git_ops.worktree_base = fake_worktree_base
    monkeypatch.setattr('orchestrator.git_ops.GitOps', MagicMock(return_value=mock_git_ops))

    cfg_file = tmp_path / 'config.yaml'
    cfg_file.write_text('')

    r = CliRunner().invoke(main, [
        'cancel-verify',
        '--request-id', FAKE_REQUEST_ID,
        '--config', str(cfg_file),
    ])

    # RC propagated correctly
    assert r.exit_code == EXPECTED_RC, (
        f'expected exit_code {EXPECTED_RC}, got {r.exit_code}; output={r.output!r}'
    )
    # Path derivation: cancel_request was called with the right pgid_file path
    assert len(cancel_calls) == 1, f'cancel_request called {len(cancel_calls)} times'
    assert cancel_calls[0] == expected_path, (
        f'cancel_request called with wrong path: {cancel_calls[0]} != {expected_path}'
    )


def test_cancel_verify_real_impl_dead_pgid(tmp_path, monkeypatch):
    """(c) Real pgid file with a dead pgid -> exit 0, file removed."""
    from unittest.mock import MagicMock

    from orchestrator.config import OrchestratorConfig
    from orchestrator.verify_cancel import pgid_file, write_pgid_file

    FAKE_REQUEST_ID = 'dead-pgid-req'
    # Use pid 1 (init) as the "dead" pgid — we can't kill it, but with a dead
    # process we instead use a non-existent high pid that raises ProcessLookupError.
    # We'll write a nonsense high pid that definitely doesn't exist.
    NONEXISTENT_PID = 2 ** 22  # well above /proc/sys/kernel/pid_max on most systems

    fake_worktree_base = tmp_path / '.worktrees'
    pgf = pgid_file(fake_worktree_base, FAKE_REQUEST_ID)
    write_pgid_file(pgf, NONEXISTENT_PID)

    fake_config = OrchestratorConfig(project_root=tmp_path)
    monkeypatch.setattr(cli_module, 'load_config', lambda _: fake_config)

    mock_git_ops = MagicMock()
    mock_git_ops.worktree_base = fake_worktree_base
    monkeypatch.setattr('orchestrator.git_ops.GitOps', MagicMock(return_value=mock_git_ops))

    cfg_file = tmp_path / 'config.yaml'
    cfg_file.write_text('')

    r = CliRunner().invoke(main, [
        'cancel-verify',
        '--request-id', FAKE_REQUEST_ID,
        '--config', str(cfg_file),
    ])

    assert r.exit_code == 0, (
        f'expected exit_code 0 for dead pgid, got {r.exit_code}; output={r.output!r}'
    )
    assert not pgf.exists(), f'pgid file must be removed after successful cancel; still at {pgf}'


# ---------------------------------------------------------------------------
# Task 1732 step-13 (part 2) — end-to-end: real subprocess cancel capstone
# ---------------------------------------------------------------------------


def _wait_for_file_cli(path: 'Path', timeout: float = 10.0, interval: float = 0.1) -> bool:
    """Poll until *path* exists or *timeout* expires. Return True if found."""
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(interval)
    return False


@pytest.mark.timeout(30)
def test_verify_merge_cancel_end_to_end(tmp_path, monkeypatch):
    """End-to-end: real subprocess 'verify-merge --request-id X' is killed by cancel-verify.

    Spawns a real 'orchestrator verify-merge --request-id X' subprocess with a
    long-running sleep spec.  Polls for the pgid file that verify-merge writes
    after os.setsid().  Calls 'cancel-verify --request-id X' via CliRunner
    (in-process, monkeypatched load_config).  Asserts:

    * cancel-verify exits 0
    * pgid file is removed by cancel_request
    * verify-merge subprocess exits within seconds (SIGKILL delivered)
    * the recorded pgid group is gone (os.killpg raises ESRCH) after wait()
    """
    import errno
    import os
    import subprocess as subprocess_mod
    import sys

    from orchestrator.config import OrchestratorConfig
    from orchestrator.git_ops import GitOps
    from orchestrator.verify_cancel import pgid_file
    from orchestrator.verify_runner import (
        MergeVerifySpec,
        UnscopedTypecheckSpec,
        VerifyCommand,
        spec_to_json,
    )

    # --- Set up a real git repo ---
    repo, head_sha = _setup_verify_repo(tmp_path)

    # --- Write minimal config YAML (project_root -> repo) ---
    cfg_file = tmp_path / 'config.yaml'
    cfg_file.write_text(f'project_root: {repo}\n')

    # --- Spec with a long-running sleep test command ---
    spec = MergeVerifySpec(
        verify_commands=(VerifyCommand('mod', test_command='sleep 300'),),
        unscoped_typecheck=UnscopedTypecheckSpec(
            commands=(VerifyCommand('mod', type_check_command='true'),),
            block_on_timeout=True,
        ),
        task_files=('mod/test_x.py',),
        verify_env={},
        cold_timeout_secs=300.0,
    )

    REQUEST_ID = 'e2e-cancel-test'

    # --- Derive the expected pgid file path (same derivation as verify-merge uses) ---
    config_obj = OrchestratorConfig(project_root=repo)
    _git_ops = GitOps(config_obj.git, repo)
    pgf = pgid_file(_git_ops.worktree_base, REQUEST_ID)

    # --- Spawn verify-merge as a real subprocess ---
    # Set PYTHONPATH so the subprocess imports from the worktree's src (not the
    # editable-install main-checkout), ensuring it uses the task-branch code.
    worktree_src = str(Path(__file__).parent.parent / 'src')
    env = dict(os.environ)
    existing_pp = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = f'{worktree_src}:{existing_pp}' if existing_pp else worktree_src
    # Remove ORCH_PROJECT_ROOT so it does not override the YAML's project_root.
    # The _isolate_orch_config autouse fixture sets ORCH_PROJECT_ROOT=<pytest
    # tmp_path>, and pydantic-settings env_settings has higher priority than
    # yaml_settings.  Without this removal the subprocess would use the pytest
    # tmp_path as project_root instead of the test's git repo, causing git
    # worktree operations to fail with "not a git repository".
    env.pop('ORCH_PROJECT_ROOT', None)

    child = subprocess_mod.Popen(
        [sys.executable, '-c', 'from orchestrator.cli import main; main()',
         'verify-merge',
         '--sha', head_sha,
         '--spec', spec_to_json(spec),
         '--config', str(cfg_file),
         '--request-id', REQUEST_ID],
        env=env,
        stdout=subprocess_mod.PIPE,
        stderr=subprocess_mod.PIPE,
    )

    pgid_val = None
    try:
        # --- Poll for the pgid file (written before asyncio.run) ---
        if not _wait_for_file_cli(pgf, timeout=15):
            _debug_stdout, _debug_stderr = child.communicate(timeout=5) if child.poll() is not None else (b'', b'')
            pytest.fail(
                f'verify-merge did not write pgid file within 15s '
                f'(subprocess poll={child.poll()!r})\n'
                f'STDOUT: {_debug_stdout.decode()[:2000]!r}\n'
                f'STDERR: {_debug_stderr.decode()[:2000]!r}'
            )

        # Save pgid BEFORE cancel-verify removes the file
        pgid_val = int(pgf.read_text().strip())

        # --- Cancel via CliRunner (in-process, monkeypatched load_config) ---
        monkeypatch.setattr(cli_module, 'load_config', lambda _: config_obj)
        r = CliRunner().invoke(main, [
            'cancel-verify',
            '--request-id', REQUEST_ID,
            '--config', str(cfg_file),
        ])

        assert r.exit_code == 0, (
            f'cancel-verify expected exit 0, got {r.exit_code}; '
            f'output={r.output!r}'
        )
        assert not pgf.exists(), (
            'pgid file must be removed by cancel-verify on success'
        )

        # --- verify-merge subprocess must exit within seconds ---
        try:
            child.wait(timeout=10)
        except subprocess_mod.TimeoutExpired:
            pytest.fail(
                'verify-merge subprocess did not exit within 10s after cancel-verify'
            )

        # --- Process group must be gone after the subprocess is reaped ---
        # child.wait() above reaped the zombie; pgid == pid after setsid, so
        # there should be no processes left in the group.
        try:
            os.killpg(pgid_val, 0)
            pytest.fail(
                f'verify-merge pgid group {pgid_val} still alive after cancel-verify'
            )
        except (ProcessLookupError, OSError) as exc:
            if hasattr(exc, 'errno') and exc.errno == errno.EPERM:
                pytest.fail(f'pgid group {pgid_val}: got EPERM (unexpected)')
            # ESRCH → group is gone — expected
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
