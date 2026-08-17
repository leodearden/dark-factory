"""Tests for the on-loop event-loop-lag heartbeat in ``server/main.py`` (task 3778).

The defect this heartbeat exists to name: the reconciliation payload renderer
fanned ~500 synchronous ``git`` probes out on the event loop thread, wedging it
for 15-43 s at a stretch.  Every component looked individually healthy for days
— the systemd watchdog kept pinging (``_watchdog_thread_loop`` deliberately runs
OFF the loop on its own OS thread), the process was alive, and nothing logged.
The only symptom was ``/health`` timing out.

This heartbeat is that off-loop watchdog's ON-loop complement: it schedules a
callback for T, measures how late it actually ran, and — the point — FIRES a
WARNING when the overshoot crosses a ``ServerConfig``-sourced threshold.  A
queryable field alone is explicitly not what is wanted (INV-4 loud-over-silent);
the whole failure mode was "nobody was told".

Covers:
  * ``_loop_lag_iteration(overshoot_ms, threshold_ms)`` — the pure log-level
    decision, extracted for exactly the reason ``_thread_monitor_iteration``'s
    docstring gives (unit-testable without mocking ``asyncio.sleep``)
  * ``_loop_lag_monitor(threshold_ms, interval=None)`` — the sampling coroutine
  * ``_start_loop_lag_monitor(config)`` — the retained-handle spawn helper
  * ``ServerConfig.loop_lag_warn_ms`` — the operator-tunable threshold

Pattern mirrors tests/test_thread_monitor.py (call the extracted pure iteration
function directly; ``caplog.at_level(..., logger='fused_memory.server.main')``;
assert on rendered substrings).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import time

import pytest
from pydantic import ValidationError

from fused_memory.config.schema import ServerConfig
from fused_memory.server import main as server_main


def _lag_records(caplog) -> list[logging.LogRecord]:
    """Every record this heartbeat emitted, in order."""
    return [r for r in caplog.records if 'loop_lag' in r.getMessage()]


class _StubConfig:
    """Minimal stand-in for FusedMemoryConfig carrying only ``.server``."""

    def __init__(self, server: ServerConfig) -> None:
        self.server = server


# ---------------------------------------------------------------------------
# (a) the pure iteration function: INFO below threshold, WARNING at/above,
#     and the message NAMES the measured lag in ms.
# ---------------------------------------------------------------------------


class TestLoopLagIteration:
    """_loop_lag_iteration(overshoot_ms, threshold_ms) decides the log level."""

    def test_below_threshold_emits_info(self, caplog):
        with caplog.at_level(logging.DEBUG, logger='fused_memory.server.main'):
            server_main._loop_lag_iteration(12.5, 1000.0)
        records = _lag_records(caplog)
        assert len(records) == 1
        assert records[0].levelno == logging.INFO

    def test_at_threshold_emits_warning(self, caplog):
        """At/above, not merely above — a lag exactly at the operator's bar is a hit."""
        with caplog.at_level(logging.DEBUG, logger='fused_memory.server.main'):
            server_main._loop_lag_iteration(1000.0, 1000.0)
        records = _lag_records(caplog)
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING

    def test_above_threshold_emits_warning(self, caplog):
        with caplog.at_level(logging.DEBUG, logger='fused_memory.server.main'):
            server_main._loop_lag_iteration(29_200.0, 1000.0)
        records = _lag_records(caplog)
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING

    def test_message_names_the_measured_lag_in_ms(self, caplog):
        """The number must be IN the message — a queryable field alone is not the ask.

        29_200 ms is the measured render stall from the root-cause report; an
        operator grepping the journal must be able to read the magnitude off
        the line itself.
        """
        with caplog.at_level(logging.DEBUG, logger='fused_memory.server.main'):
            server_main._loop_lag_iteration(29_200.0, 1000.0)
        message = _lag_records(caplog)[0].getMessage()
        assert '29200' in message.replace(',', '').replace('_', ''), message
        assert 'ms' in message

    def test_warning_message_names_the_threshold_that_was_crossed(self, caplog):
        """Without the threshold the reader cannot tell how far over the bar it is."""
        with caplog.at_level(logging.DEBUG, logger='fused_memory.server.main'):
            server_main._loop_lag_iteration(4321.0, 1500.0)
        message = _lag_records(caplog)[0].getMessage()
        assert '1500' in message.replace(',', '').replace('_', ''), message

    def test_is_not_a_coroutine_function(self):
        """Pure and synchronous, so tests need not mock asyncio.sleep."""
        assert not inspect.iscoroutinefunction(server_main._loop_lag_iteration)


# ---------------------------------------------------------------------------
# (b)/(c) the measuring coroutine reports a REAL overshoot, and is quiet when
#         the loop is healthy.
# ---------------------------------------------------------------------------


class TestLoopLagMonitorMeasurement:
    """_loop_lag_monitor measures actual scheduling delay, not a constant."""

    @pytest.mark.asyncio
    async def test_reports_real_overshoot_when_the_loop_is_blocked(self, monkeypatch):
        """THE test that would have named the 15-43 s defect, in seconds.

        Blocks the loop with a synchronous ``time.sleep`` — exactly what ~500
        inline ``subprocess.run`` git probes did — and asserts the heartbeat
        reports a lag of at least that magnitude.  An implementation that
        reported a constant, or that measured wall clock rather than scheduling
        delay, reads RED here.
        """
        blocked_secs = 0.40
        interval = 0.01
        recorded: list[tuple[float, float]] = []
        monkeypatch.setattr(
            server_main, '_loop_lag_iteration',
            lambda overshoot_ms, threshold_ms: recorded.append((overshoot_ms, threshold_ms)),
        )

        task = asyncio.create_task(
            server_main._loop_lag_monitor(threshold_ms=1.0, interval=interval),
        )
        try:
            await asyncio.sleep(0)  # let the monitor take its first timestamp
            time.sleep(blocked_secs)  # <-- the loop is now wedged, on purpose
            await asyncio.sleep(interval * 3)  # let the late timer callback run
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert recorded, 'heartbeat never reported — a blocked loop must be observable'
        worst_ms = max(overshoot for overshoot, _ in recorded)
        floor_ms = (blocked_secs - interval) * 1000 * 0.8
        assert worst_ms >= floor_ms, (
            f'reported lag {worst_ms:.1f}ms is below the {floor_ms:.1f}ms floor implied '
            f'by a {blocked_secs * 1000:.0f}ms block — the probe is not measuring '
            f'scheduling delay'
        )

    @pytest.mark.asyncio
    async def test_idle_loop_reports_near_zero_lag_and_never_warns(self, monkeypatch, caplog):
        """No false alarm in the steady state.

        An idle loop overshoots a timer by microseconds-to-milliseconds.  With
        the shipped threshold (three orders of magnitude above that) the
        heartbeat must stay at INFO — a heartbeat that cries wolf gets muted,
        and then the next 29 s stall goes unreported again.
        """
        interval = 0.01
        recorded: list[tuple[float, float]] = []
        real_iteration = server_main._loop_lag_iteration

        def _spy(overshoot_ms: float, threshold_ms: float) -> None:
            recorded.append((overshoot_ms, threshold_ms))
            real_iteration(overshoot_ms, threshold_ms)

        monkeypatch.setattr(server_main, '_loop_lag_iteration', _spy)
        monkeypatch.setattr(server_main, '_LOOP_LAG_INFO_EVERY', 1)

        threshold_ms = float(ServerConfig().loop_lag_warn_ms)
        with caplog.at_level(logging.DEBUG, logger='fused_memory.server.main'):
            task = asyncio.create_task(
                server_main._loop_lag_monitor(threshold_ms=threshold_ms, interval=interval),
            )
            try:
                await asyncio.sleep(interval * 8)
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        assert recorded, 'an idle heartbeat must still beat'
        worst_ms = max(overshoot for overshoot, _ in recorded)
        assert worst_ms < threshold_ms, f'idle overshoot {worst_ms:.1f}ms crossed the bar'
        warnings = [r for r in _lag_records(caplog) if r.levelno >= logging.WARNING]
        assert warnings == [], f'idle loop must not warn; got {[r.getMessage() for r in warnings]}'

    @pytest.mark.asyncio
    async def test_overshoot_excludes_the_intended_sleep(self, monkeypatch):
        """The reported number is OVERSHOOT, not elapsed time.

        A 10 ms sample interval that returns after 10 ms is a lag of ~0, not of
        10.  Getting this wrong would make every idle sample look like a stall
        the size of the sample interval.
        """
        interval = 0.05
        recorded: list[float] = []
        monkeypatch.setattr(
            server_main, '_loop_lag_iteration',
            lambda overshoot_ms, threshold_ms: recorded.append(overshoot_ms),
        )
        monkeypatch.setattr(server_main, '_LOOP_LAG_INFO_EVERY', 1)

        task = asyncio.create_task(
            server_main._loop_lag_monitor(threshold_ms=10_000.0, interval=interval),
        )
        try:
            await asyncio.sleep(interval * 2.5)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert recorded
        assert min(recorded) < interval * 1000 / 2, (
            f'overshoot {min(recorded):.1f}ms is not small relative to the '
            f'{interval * 1000:.0f}ms interval — the sleep itself is being counted as lag'
        )

    @pytest.mark.asyncio
    async def test_a_raising_iteration_does_not_kill_the_heartbeat(self, monkeypatch):
        """A logging hiccup must not silently end the monitor.

        Same reasoning as _watchdog_thread_loop's guard (task 1731): a dead
        heartbeat is indistinguishable from a healthy one, which is the exact
        failure class this whole task is about.
        """
        interval = 0.01
        calls: list[float] = []

        def _boom(overshoot_ms: float, threshold_ms: float) -> None:
            calls.append(overshoot_ms)
            raise RuntimeError('logging backend exploded')

        monkeypatch.setattr(server_main, '_loop_lag_iteration', _boom)
        monkeypatch.setattr(server_main, '_LOOP_LAG_INFO_EVERY', 1)

        task = asyncio.create_task(
            server_main._loop_lag_monitor(threshold_ms=1.0, interval=interval),
        )
        try:
            await asyncio.sleep(interval * 8)
            assert len(calls) >= 2, 'monitor stopped after the first raise'
            assert not task.done(), 'monitor task died on a logging error'
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


# ---------------------------------------------------------------------------
# (d) retained handle + clean cancellation (the checkpoint_task pattern, NOT
#     _thread_monitor's fire-and-forget).
# ---------------------------------------------------------------------------


class TestLoopLagMonitorLifecycle:
    """The monitor is spawned with a retained handle and cancels cleanly."""

    @pytest.mark.asyncio
    async def test_start_returns_a_named_task_handle(self):
        """Returning the handle is the structural difference from fire-and-forget.

        ``asyncio.create_task(_thread_monitor())`` at main.py:975 keeps no
        reference: the task can be GC'd mid-flight and emits a 'Task was
        destroyed but it is pending' line on shutdown.  That is a pre-existing
        wart, not a pattern to copy — ``checkpoint_task`` (:1038/:1216-1219) is
        the correct in-file precedent.
        """
        config = _StubConfig(ServerConfig())
        task = server_main._start_loop_lag_monitor(config)
        try:
            assert isinstance(task, asyncio.Task)
            assert task.get_name(), 'task must be named so it is identifiable in a task dump'
            assert 'loop_lag' in task.get_name()
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    @pytest.mark.asyncio
    async def test_cancellation_is_clean(self, monkeypatch):
        """cancel() + await must settle as cancelled, raising nothing else."""
        monkeypatch.setattr(server_main, '_LOOP_LAG_INTERVAL', 0.01)
        config = _StubConfig(ServerConfig())
        task = server_main._start_loop_lag_monitor(config)
        await asyncio.sleep(0.02)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_run_server_teardown_cancels_the_monitor(self):
        """Scoped source check on run_server's own body.

        No behavioural equivalent exists: ``run_server`` binds two uvicorn
        ports and constructs every store in the process, so it cannot be driven
        from a unit test.  The check is scoped to a SINGLE function's source
        (like the retained guard in test_tool_errors.py, not a module-wide text
        scan), and the needles are exact code constructs, so unrelated edits
        elsewhere in main.py cannot trip it.
        """
        source = inspect.getsource(server_main.run_server)
        assert '_start_loop_lag_monitor(config)' in source, (
            'run_server must spawn the loop-lag heartbeat'
        )
        assert 'loop_lag_task.cancel()' in source, (
            'run_server teardown must cancel the heartbeat (the checkpoint_task '
            'pattern), not leave it to be killed with the loop'
        )
        assert 'await loop_lag_task' in source, (
            'run_server teardown must await the cancelled heartbeat so it settles'
        )


# ---------------------------------------------------------------------------
# (e) the threshold is operator-tunable via ServerConfig, never a literal.
# ---------------------------------------------------------------------------


class TestLoopLagThresholdIsConfigSourced:
    """ServerConfig.loop_lag_warn_ms drives the heartbeat, not a hardcoded number."""

    def test_default_is_well_clear_of_idle_noise_and_below_the_observed_stall(self):
        """Between the ~12-35 ms idle baseline and the 15-43 s observed stalls."""
        default = ServerConfig().loop_lag_warn_ms
        assert default > 100, 'a threshold near the idle baseline would cry wolf'
        assert default <= 5_000, 'a threshold above ~5s would have missed the 15s /health failures'

    def test_override_round_trips(self):
        assert ServerConfig(loop_lag_warn_ms=250).loop_lag_warn_ms == 250

    def test_rejects_non_int(self):
        with pytest.raises(ValidationError):
            ServerConfig(loop_lag_warn_ms='not-an-int')  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_threshold_moves_when_the_config_value_moves(self, monkeypatch):
        """The value the monitor compares against IS the configured one.

        Asserted by observing the threshold the monitor forwards to the pure
        iteration function for a deliberately non-default config value — a
        module literal would keep reporting 1000 here.
        """
        monkeypatch.setattr(server_main, '_LOOP_LAG_INTERVAL', 0.01)
        monkeypatch.setattr(server_main, '_LOOP_LAG_INFO_EVERY', 1)
        seen: list[float] = []
        monkeypatch.setattr(
            server_main, '_loop_lag_iteration',
            lambda overshoot_ms, threshold_ms: seen.append(threshold_ms),
        )

        config = _StubConfig(ServerConfig(loop_lag_warn_ms=137))
        task = server_main._start_loop_lag_monitor(config)
        try:
            await asyncio.sleep(0.05)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert seen, 'monitor never reported'
        assert set(seen) == {137.0}, f'threshold did not follow the config: {sorted(set(seen))}'

    @pytest.mark.asyncio
    async def test_a_low_configured_threshold_makes_a_small_lag_warn(self, monkeypatch, caplog):
        """End-to-end: config -> monitor -> WARNING, on a real measured overshoot.

        The same 150 ms block is INFO-level noise under the shipped 1000 ms
        default and a WARNING under a 25 ms operator override, which is what
        "tunable" has to mean.
        """
        monkeypatch.setattr(server_main, '_LOOP_LAG_INTERVAL', 0.01)
        config = _StubConfig(ServerConfig(loop_lag_warn_ms=25))

        with caplog.at_level(logging.DEBUG, logger='fused_memory.server.main'):
            task = server_main._start_loop_lag_monitor(config)
            try:
                await asyncio.sleep(0)
                time.sleep(0.15)
                await asyncio.sleep(0.05)
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        warnings = [r for r in _lag_records(caplog) if r.levelno >= logging.WARNING]
        assert warnings, (
            'a 150ms block against a 25ms configured threshold must WARN; '
            f'records={[r.getMessage() for r in _lag_records(caplog)]}'
        )


# ---------------------------------------------------------------------------
# Log-volume discipline: every crossing is loud, the routine heartbeat is not.
# ---------------------------------------------------------------------------


class TestLoopLagReportingCadence:
    """Above-threshold samples always report; below-threshold ones are throttled."""

    @pytest.mark.asyncio
    async def test_below_threshold_samples_are_throttled_to_the_info_cadence(
        self, monkeypatch,
    ):
        """Sampling is frequent (so a stall is actually caught) but INFO is not.

        Sampling every _LOOP_LAG_INTERVAL is what makes a 15-43 s stall
        overlap a deadline and be measured at near-full magnitude; reporting
        every one of those samples at INFO would be thousands of lines a day.
        """
        interval = 0.01
        monkeypatch.setattr(server_main, '_LOOP_LAG_INFO_EVERY', 5)
        reported: list[float] = []
        monkeypatch.setattr(
            server_main, '_loop_lag_iteration',
            lambda overshoot_ms, threshold_ms: reported.append(overshoot_ms),
        )

        task = asyncio.create_task(
            server_main._loop_lag_monitor(threshold_ms=10_000.0, interval=interval),
        )
        try:
            await asyncio.sleep(interval * 10)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # ~10 samples elapsed; at one report per 5 samples that is ~2, and must
        # in any case be strictly fewer than the number of samples taken.
        assert len(reported) <= 4, f'INFO cadence not throttled: {len(reported)} reports'

    @pytest.mark.asyncio
    async def test_above_threshold_samples_are_never_throttled(self, monkeypatch):
        """A crossing is reported on the sample it happens — never deferred."""
        interval = 0.01
        monkeypatch.setattr(server_main, '_LOOP_LAG_INFO_EVERY', 1_000_000)
        reported: list[float] = []
        monkeypatch.setattr(
            server_main, '_loop_lag_iteration',
            lambda overshoot_ms, threshold_ms: reported.append(overshoot_ms),
        )

        task = asyncio.create_task(
            server_main._loop_lag_monitor(threshold_ms=0.0, interval=interval),
        )
        try:
            await asyncio.sleep(interval * 5)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert len(reported) >= 2, (
            'threshold crossings must bypass the INFO throttle; '
            f'got {len(reported)} reports'
        )
