"""Tests for the WP-C SqliteWatchdog.

The watchdog is a lightweight observer over :class:`EventQueue` — tests use a
minimal fake queue exposing the same ``stats()`` / ``recent_ops()`` surface
so behavior can be driven deterministically without SQLite in the loop.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from fused_memory.reconciliation.sqlite_watchdog import SqliteWatchdog


class _FakeEventQueue:
    """Test double matching the subset of EventQueue the watchdog consults."""

    def __init__(self) -> None:
        self._stats: dict = {
            'queue_depth': 0,
            'queue_capacity': 100,
            'last_commit_ts': None,
            'events_committed': 0,
            'overflow_drops': 0,
            'dead_letters': 0,
            'retry_in_flight': 0,
            'drainer_running': True,
        }
        self._recent_ops: list[dict] = []

    def set_stats(self, **overrides) -> None:
        self._stats.update(overrides)

    def set_recent_ops(self, ops: list[dict]) -> None:
        self._recent_ops = list(ops)

    def stats(self) -> dict:
        return dict(self._stats)

    def recent_ops(self) -> list[dict]:
        return list(self._recent_ops)


# ── Wedge detection ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_wedge_fires_on_artificial_stall(caplog):
    """Depth > 0 AND no recent commit beyond the stall threshold → ERROR log."""
    fake = _FakeEventQueue()
    # Simulate: drainer committed once, 10s ago; now 5 events sit in queue.
    fake.set_stats(
        queue_depth=5,
        last_commit_ts=time.time() - 10.0,
        retry_in_flight=3,
        events_committed=42,
    )
    fake.set_recent_ops([
        {'event_id': 'e1', 'event_type': 'task_created', 'status': 'retrying', 'attempts': 4},
    ])

    wedge_payloads: list[dict] = []

    async def callback(payload: dict) -> None:
        wedge_payloads.append(payload)

    watchdog = SqliteWatchdog(
        fake,
        check_interval_seconds=0.05,
        stall_threshold_seconds=1.0,
        rearm_after_seconds=60.0,
        wedge_callback=callback,
    )

    caplog.set_level(logging.ERROR, logger='fused_memory.reconciliation.sqlite_watchdog')
    await watchdog.start()
    try:
        # First tick happens after check_interval; stall_threshold is 1s so
        # at ~0.05s we're healthy — give it 1.5s to cross the threshold.
        await asyncio.sleep(1.3)
    finally:
        await watchdog.close()

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert error_records, 'watchdog did not emit ERROR log for wedged drainer'
    msg = error_records[0].getMessage()
    assert 'wedged' in msg
    assert 'queue_depth' in msg.lower() or '5' in msg

    assert wedge_payloads, 'wedge_callback was not invoked'
    payload = wedge_payloads[0]
    # Diagnostic payload must carry the keys operators need to triage.
    for key in (
        'stale_for_seconds', 'queue_depth', 'retry_in_flight',
        'events_committed', 'recent_ops', 'asyncio_task_count',
    ):
        assert key in payload, f'missing diagnostic key: {key}'
    assert payload['queue_depth'] == 5
    assert payload['retry_in_flight'] == 3
    assert payload['events_committed'] == 42
    assert payload['recent_ops'][0]['event_id'] == 'e1'


@pytest.mark.asyncio
async def test_wedge_not_fired_when_drainer_healthy(caplog):
    """Fresh commits → no ERROR log, callback not invoked."""
    fake = _FakeEventQueue()
    # Healthy: depth>0 but committed < 1s ago.
    fake.set_stats(
        queue_depth=3,
        last_commit_ts=time.time(),
        events_committed=100,
    )
    callback_count = 0

    async def callback(payload: dict) -> None:
        nonlocal callback_count
        callback_count += 1

    watchdog = SqliteWatchdog(
        fake,
        check_interval_seconds=0.05,
        stall_threshold_seconds=1.0,
        wedge_callback=callback,
    )

    caplog.set_level(logging.ERROR, logger='fused_memory.reconciliation.sqlite_watchdog')
    await watchdog.start()
    try:
        # Keep last_commit_ts fresh across several ticks.
        for _ in range(6):
            fake.set_stats(last_commit_ts=time.time())
            await asyncio.sleep(0.1)
    finally:
        await watchdog.close()

    assert not [r for r in caplog.records if r.levelno == logging.ERROR], (
        'watchdog emitted ERROR despite healthy drainer'
    )
    assert callback_count == 0, 'wedge_callback fired for a healthy drainer'


@pytest.mark.asyncio
async def test_wedge_not_fired_when_queue_empty(caplog):
    """Stale last_commit is fine if nothing outstanding (idle, not wedged)."""
    fake = _FakeEventQueue()
    # No outstanding work: queue empty and no retries in flight.
    fake.set_stats(
        queue_depth=0,
        retry_in_flight=0,
        last_commit_ts=time.time() - 1000.0,
    )

    watchdog = SqliteWatchdog(
        fake,
        check_interval_seconds=0.05,
        stall_threshold_seconds=0.1,
    )
    caplog.set_level(logging.ERROR, logger='fused_memory.reconciliation.sqlite_watchdog')
    await watchdog.start()
    try:
        await asyncio.sleep(0.4)
    finally:
        await watchdog.close()

    assert not [r for r in caplog.records if r.levelno == logging.ERROR]


@pytest.mark.asyncio
async def test_wedge_fires_when_only_retry_in_flight(caplog):
    """queue_depth=0 but an event is stuck retrying → wedge detected.

    The drainer pops an event off the queue before attempting to commit it;
    if that commit retries indefinitely, ``queue_depth`` goes to 0 but
    ``retry_in_flight`` stays > 0. The watchdog must catch this.
    """
    fake = _FakeEventQueue()
    fake.set_stats(
        queue_depth=0,
        retry_in_flight=1,
        last_commit_ts=time.time() - 100.0,
    )

    wedge_payloads: list[dict] = []

    async def callback(payload: dict) -> None:
        wedge_payloads.append(payload)

    watchdog = SqliteWatchdog(
        fake,
        check_interval_seconds=0.02,
        stall_threshold_seconds=0.05,
        rearm_after_seconds=60.0,
        wedge_callback=callback,
    )

    caplog.set_level(logging.ERROR, logger='fused_memory.reconciliation.sqlite_watchdog')
    await watchdog.start()
    try:
        await asyncio.sleep(0.3)
    finally:
        await watchdog.close()

    assert wedge_payloads, 'watchdog missed retry-only wedge'
    assert wedge_payloads[0]['retry_in_flight'] == 1


# ── Re-arm / rate-limit ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_wedge_does_not_spam(caplog):
    """A persistent wedge produces one ERROR per rearm window, not one per tick."""
    fake = _FakeEventQueue()
    fake.set_stats(
        queue_depth=1,
        last_commit_ts=time.time() - 100.0,
    )

    callback_calls = 0

    async def callback(payload: dict) -> None:
        nonlocal callback_calls
        callback_calls += 1

    watchdog = SqliteWatchdog(
        fake,
        check_interval_seconds=0.02,
        stall_threshold_seconds=0.05,
        rearm_after_seconds=60.0,  # effectively no re-arm in this test window
        wedge_callback=callback,
    )

    caplog.set_level(logging.ERROR, logger='fused_memory.reconciliation.sqlite_watchdog')
    await watchdog.start()
    try:
        # Many ticks across the stall threshold.
        await asyncio.sleep(0.5)
    finally:
        await watchdog.close()

    error_count = sum(
        1 for r in caplog.records if r.levelno == logging.ERROR
    )
    # Without re-arm: exactly one ERROR even though many ticks detected the wedge.
    assert error_count == 1, f'expected 1 ERROR (rate-limited), got {error_count}'
    assert callback_calls == 1, (
        f'expected 1 callback invocation, got {callback_calls}'
    )


@pytest.mark.asyncio
async def test_wedge_rearms_after_recovery(caplog):
    """If the drainer recovers (fresh commit) and wedges again, ERROR re-fires."""
    fake = _FakeEventQueue()
    # Phase 1: wedged.
    fake.set_stats(queue_depth=2, last_commit_ts=time.time() - 100.0)

    callback_count = 0

    async def callback(payload: dict) -> None:
        nonlocal callback_count
        callback_count += 1

    watchdog = SqliteWatchdog(
        fake,
        check_interval_seconds=0.02,
        stall_threshold_seconds=0.05,
        rearm_after_seconds=60.0,
        wedge_callback=callback,
    )

    caplog.set_level(logging.ERROR, logger='fused_memory.reconciliation.sqlite_watchdog')
    await watchdog.start()
    try:
        # Let the first wedge fire.
        await asyncio.sleep(0.2)
        # Phase 2: recover — fresh commit, non-empty queue still fine because
        # the recency anchor is last_commit_ts.
        fake.set_stats(queue_depth=0, last_commit_ts=time.time())
        await asyncio.sleep(0.15)
        # Phase 3: wedge again.
        fake.set_stats(queue_depth=3, last_commit_ts=time.time() - 100.0)
        await asyncio.sleep(0.2)
    finally:
        await watchdog.close()

    error_count = sum(
        1 for r in caplog.records if r.levelno == logging.ERROR
    )
    assert error_count == 2, (
        f'expected 2 ERRORs (wedge → recover → wedge), got {error_count}'
    )
    assert callback_count == 2


# ── Graceful close ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_close_cancels_check_loop():
    """close() cancels the loop task and is idempotent."""
    fake = _FakeEventQueue()
    watchdog = SqliteWatchdog(
        fake,
        check_interval_seconds=0.05,
        stall_threshold_seconds=1.0,
    )
    await watchdog.start()
    # Task is live.
    assert watchdog._task is not None
    assert not watchdog._task.done()

    await watchdog.close()
    assert watchdog._task is None
    assert watchdog._closed is True

    # Idempotent — second close() is a no-op.
    await watchdog.close()


@pytest.mark.asyncio
async def test_close_before_start_is_noop():
    """close() before start() does not raise."""
    watchdog = SqliteWatchdog(_FakeEventQueue())
    await watchdog.close()  # must not raise


@pytest.mark.asyncio
async def test_start_twice_raises():
    """Calling start() twice is a programmer error."""
    watchdog = SqliteWatchdog(_FakeEventQueue())
    await watchdog.start()
    try:
        with pytest.raises(RuntimeError):
            await watchdog.start()
    finally:
        await watchdog.close()


# ── Integration with real EventQueue ─────────────────────────────────────


@pytest.mark.asyncio
async def test_wedge_fires_with_real_event_queue(tmp_path, caplog):
    """End-to-end: real EventQueue whose buffer is locked → watchdog fires.

    Exercises the full stats()/recent_ops() surface against the production
    class so we know the watchdog reads fields that actually exist.
    """
    from unittest.mock import AsyncMock

    import aiosqlite

    from fused_memory.models.reconciliation import (
        EventSource,
        EventType,
        ReconciliationEvent,
    )
    from fused_memory.reconciliation.event_queue import EventQueue

    # Buffer that always raises "database is locked" — the drainer retries
    # indefinitely, so last_commit_ts stays None and queue_depth stays > 0.
    buf = AsyncMock()
    buf.push = AsyncMock(
        side_effect=aiosqlite.OperationalError('database is locked'),
    )

    queue = EventQueue(
        buf,
        dead_letter_path=tmp_path / 'dl.jsonl',
        maxsize=100,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.05,
        shutdown_flush_seconds=0.1,
    )
    await queue.start()

    wedge_payloads: list[dict] = []

    async def callback(payload: dict) -> None:
        wedge_payloads.append(payload)

    watchdog = SqliteWatchdog(
        queue,
        check_interval_seconds=0.05,
        stall_threshold_seconds=0.3,
        rearm_after_seconds=60.0,
        wedge_callback=callback,
    )

    caplog.set_level(logging.ERROR, logger='fused_memory.reconciliation.sqlite_watchdog')
    await watchdog.start()
    try:
        # Enqueue an event — drainer will keep retrying, never commit.
        import uuid
        from datetime import UTC, datetime

        event = ReconciliationEvent(
            id=str(uuid.uuid4()),
            type=EventType.task_created,
            source=EventSource.agent,
            project_id='wp-c-test',
            timestamp=datetime.now(UTC),
            payload={'wp': 'c'},
        )
        queue.enqueue(event)

        # stall_threshold=0.3s; give it ~1s to cross it.
        await asyncio.sleep(1.0)
    finally:
        await watchdog.close()
        queue._shutdown_flush = 0.05
        await queue.close()

    assert wedge_payloads, 'watchdog did not fire against real EventQueue'
    payload = wedge_payloads[0]
    # Drainer pops the event immediately, so queue_depth=0 by the time the
    # watchdog ticks — but retry_in_flight > 0 keeps the wedge signal true.
    assert payload['retry_in_flight'] >= 1
    # recent_ops ring buffer should show retrying attempts for our event.
    assert any(
        op['status'] == 'retrying' for op in payload['recent_ops']
    )
