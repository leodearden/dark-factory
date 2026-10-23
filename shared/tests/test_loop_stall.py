"""Behaviour of ``shared.loop_stall.LoopStallMonitor`` on a real event loop."""
from __future__ import annotations

import asyncio
import logging
import threading
import time

import pytest
from shared.loop_stall import LoopStallMonitor

_LOGGER_NAME = 'shared.loop_stall'
_WATCHDOG_NAME = 'loop-stall-watchdog'


def _hold_the_loop_thread(secs: float) -> None:
    time.sleep(secs)


def _stall_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records
        if r.name == _LOGGER_NAME and r.levelno >= logging.WARNING
    ]


def _stack_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in _stall_records(caplog) if hasattr(r, 'loop_stack')]


def _watchdogs() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == _WATCHDOG_NAME and t.is_alive()]


async def _block_once(monitor: LoopStallMonitor, block_secs: float) -> None:
    monitor.start()
    try:
        await asyncio.sleep(0.1)
        _hold_the_loop_thread(block_secs)
        await asyncio.sleep(0.2)
    finally:
        await monitor.stop()


async def test_a_blocked_loop_is_reported_with_the_blocking_frame(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    monitor = LoopStallMonitor(
        warn_after_secs=0.2, beat_interval_secs=0.05, restack_interval_secs=60,
    )

    await _block_once(monitor, 0.8)

    stacks = _stack_records(caplog)
    assert len(stacks) == 1, [r.getMessage() for r in _stall_records(caplog)]
    record = stacks[0]
    assert record.levelno == logging.WARNING
    assert record.thread != threading.get_ident()
    assert '_hold_the_loop_thread' in record.loop_stack


async def test_resume_reports_the_whole_stall(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    monitor = LoopStallMonitor(
        warn_after_secs=0.2, beat_interval_secs=0.05, restack_interval_secs=60,
    )

    await _block_once(monitor, 0.8)

    on_loop = [
        r for r in _stall_records(caplog)
        if r.thread == threading.get_ident() and hasattr(r, 'stall_secs')
    ]
    assert on_loop, [r.getMessage() for r in _stall_records(caplog)]
    assert max(r.stall_secs for r in on_loop) >= 0.5


async def test_a_long_stall_is_restacked(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    monitor = LoopStallMonitor(
        warn_after_secs=0.1, beat_interval_secs=0.05, restack_interval_secs=0.3,
    )

    await _block_once(monitor, 1.0)

    assert len(_stack_records(caplog)) >= 2


async def test_a_responsive_loop_reports_nothing(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    monitor = LoopStallMonitor(
        warn_after_secs=5.0, beat_interval_secs=0.05, restack_interval_secs=60,
    )

    monitor.start()
    try:
        await asyncio.sleep(0.3)
    finally:
        await monitor.stop()

    assert _stall_records(caplog) == []


async def test_stop_ends_the_watchdog() -> None:
    monitor = LoopStallMonitor(
        warn_after_secs=5.0, beat_interval_secs=0.05, restack_interval_secs=60,
    )

    monitor.start()
    try:
        assert len(_watchdogs()) == 1
        monitor.start()
        assert len(_watchdogs()) == 1
    finally:
        await monitor.stop()

    assert _watchdogs() == []
    await monitor.stop()


def test_the_watchdog_ends_with_its_loop(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)

    async def start_and_abandon() -> None:
        LoopStallMonitor(
            warn_after_secs=0.2, beat_interval_secs=0.05, restack_interval_secs=60,
        ).start()
        await asyncio.sleep(0.1)

    asyncio.run(start_and_abandon())

    for thread in [t for t in threading.enumerate() if t.name == _WATCHDOG_NAME]:
        thread.join(2.0)
    assert _watchdogs() == []
    assert _stall_records(caplog) == []


@pytest.mark.parametrize('argument', [
    'warn_after_secs', 'beat_interval_secs', 'restack_interval_secs',
])
@pytest.mark.parametrize('value', [0, -1.5])
def test_non_positive_intervals_are_rejected(argument: str, value: float) -> None:
    with pytest.raises(ValueError) as excinfo:
        LoopStallMonitor(**{argument: value})

    assert argument in str(excinfo.value)
    assert repr(value) in str(excinfo.value)
