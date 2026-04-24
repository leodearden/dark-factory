"""Tests for the fire-and-forget event queue (WP-B)."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import aiosqlite
import pytest
import pytest_asyncio

from fused_memory.models.reconciliation import (
    EventSource,
    EventType,
    ReconciliationEvent,
)
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.event_queue import EventQueue


def _make_event(
    project_id: str = 'test-project',
    event_type: EventType = EventType.task_created,
) -> ReconciliationEvent:
    return ReconciliationEvent(
        id=str(uuid.uuid4()),
        type=event_type,
        source=EventSource.agent,
        project_id=project_id,
        timestamp=datetime.now(UTC),
        payload={'test': True},
    )


@pytest_asyncio.fixture
async def real_buffer(tmp_path):
    """Real EventBuffer over a temp SQLite file."""
    buf = EventBuffer(db_path=tmp_path / 'buf.db', buffer_size_threshold=100)
    await buf.initialize()
    yield buf
    await buf.close()


@pytest_asyncio.fixture
async def queue(real_buffer, tmp_path):
    """EventQueue wired to the real buffer with defaults tuned for tests."""
    q = EventQueue(
        real_buffer,
        dead_letter_path=tmp_path / 'dead_letter.jsonl',
        maxsize=100,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.1,
        shutdown_flush_seconds=2.0,
    )
    await q.start()
    yield q
    await q.close()


# ── Happy path ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_enqueue_persists_to_buffer(queue, real_buffer):
    """Enqueued events eventually land in SQLite."""
    for _ in range(5):
        assert queue.enqueue(_make_event()) is True
    # Wait for drainer to catch up.
    await asyncio.wait_for(queue._queue.join(), timeout=1.0)
    stats = await real_buffer.get_buffer_stats('test-project')
    assert stats['size'] == 5


@pytest.mark.asyncio
async def test_enqueue_returns_immediately(queue):
    """enqueue is synchronous and non-blocking."""
    import time
    # Pre-create events so model construction doesn't inflate the timing.
    events = [_make_event() for _ in range(100)]
    t0 = time.perf_counter()
    for event in events:
        queue.enqueue(event)
    elapsed = time.perf_counter() - t0
    # 100 non-blocking puts should take well under 50ms.
    assert elapsed < 0.05, f'enqueue took {elapsed:.3f}s for 100 calls'


# ── Failure injection ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hot_path_immunity_under_sqlite_lock(tmp_path):
    """When buffer.push always raises OperationalError, enqueue still returns fast."""
    buf = AsyncMock()
    buf.push = AsyncMock(side_effect=aiosqlite.OperationalError('database is locked'))
    q = EventQueue(
        buf,
        dead_letter_path=tmp_path / 'dl.jsonl',
        maxsize=1000,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.05,
    )
    await q.start()
    try:
        import time
        t0 = time.perf_counter()
        for _ in range(10):
            assert q.enqueue(_make_event()) is True
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.5, f'enqueue path was not immune to lock: {elapsed:.3f}s'
        # Let drainer retry a few times; nothing should be dead-lettered
        # because OperationalError is retriable.
        await asyncio.sleep(0.2)
        assert q.stats()['dead_letters'] == 0
        assert q.stats()['retry_in_flight'] >= 1
    finally:
        # Forcibly close without waiting for lock to clear
        q._shutdown_flush = 0.05
        await q.close()


@pytest.mark.asyncio
async def test_drainer_recovers_after_transient_failure(tmp_path, real_buffer):
    """Drainer retries OperationalError and eventually succeeds."""
    call_log: list[int] = []
    original_push = real_buffer.push

    async def flaky_push(event):
        call_log.append(1)
        if len(call_log) <= 3:
            raise aiosqlite.OperationalError('transient lock')
        return await original_push(event)

    real_buffer.push = flaky_push

    q = EventQueue(
        real_buffer,
        dead_letter_path=tmp_path / 'dl.jsonl',
        maxsize=100,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.1,
        shutdown_flush_seconds=2.0,
    )
    await q.start()
    try:
        for _ in range(5):
            q.enqueue(_make_event())
        await asyncio.wait_for(q._queue.join(), timeout=2.0)
        stats = await real_buffer.get_buffer_stats('test-project')
        # All 5 eventually landed; push() was called >= 5 + 3 retries = 8 times.
        assert stats['size'] == 5
        assert len(call_log) >= 8
    finally:
        await q.close()


@pytest.mark.asyncio
async def test_non_retriable_error_goes_to_dead_letter(tmp_path):
    """ValueError (or any non-OperationalError) → dead-letter immediately."""
    buf = AsyncMock()
    buf.push = AsyncMock(side_effect=ValueError('schema mismatch'))
    dl = tmp_path / 'dl.jsonl'
    q = EventQueue(
        buf,
        dead_letter_path=dl,
        maxsize=100,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.05,
        shutdown_flush_seconds=2.0,
    )
    await q.start()
    try:
        for _ in range(3):
            q.enqueue(_make_event())
        await asyncio.wait_for(q._queue.join(), timeout=1.0)
        # All 3 should have been dead-lettered after a single failed push.
        assert q.stats()['dead_letters'] == 3
        lines = dl.read_text().strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            record = json.loads(line)
            assert record['reason'] == 'non_retriable'
            assert record['attempts'] == 1
            assert 'event' in record
            assert 'failed_at' in record
    finally:
        await q.close()


# ── Overflow ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_overflow_writes_to_dead_letter(tmp_path):
    """When queue is full, enqueue returns False and writes to dead-letter."""
    buf = AsyncMock()

    # Drainer blocks forever so the queue fills up.
    async def blocking_push(event):
        await asyncio.sleep(1000)

    buf.push = blocking_push

    dl = tmp_path / 'dl.jsonl'
    q = EventQueue(
        buf,
        dead_letter_path=dl,
        maxsize=2,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.05,
        shutdown_flush_seconds=0.1,
    )
    await q.start()
    try:
        # First event gets picked up by the drainer (blocks there).
        # Next 2 fit in the queue (maxsize=2). Subsequent 3 overflow.
        results = [q.enqueue(_make_event()) for _ in range(6)]
        # Pumping depends on scheduler ordering, but at least the last
        # 3 puts should overflow.
        overflowed = sum(1 for r in results if r is False)
        assert overflowed >= 3, f'expected >=3 overflow, got {overflowed}'
        assert q.stats()['overflow_drops'] == overflowed
        # Dead-letter file contains overflow records.
        lines = dl.read_text().strip().splitlines()
        assert len(lines) == overflowed
        assert all(
            json.loads(line)['reason'] == 'overflow_drop' for line in lines
        )
    finally:
        # Close without waiting for the blocked push.
        q._shutdown_flush = 0.05
        await q.close()


# ── Shutdown ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_graceful_shutdown_flushes_within_window(tmp_path, real_buffer):
    """Bounded flush drains everything when drainer can keep up."""
    q = EventQueue(
        real_buffer,
        dead_letter_path=tmp_path / 'dl.jsonl',
        maxsize=100,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.1,
        shutdown_flush_seconds=5.0,
    )
    await q.start()
    for _ in range(50):
        q.enqueue(_make_event())
    await q.close()
    # All events landed; no dead-letter residue.
    stats = await real_buffer.get_buffer_stats('test-project')
    assert stats['size'] == 50
    dl_path = tmp_path / 'dl.jsonl'
    assert not dl_path.exists() or dl_path.read_text() == ''


@pytest.mark.asyncio
async def test_shutdown_timeout_dumps_remainder(tmp_path):
    """When flush window expires, unflushed events go to dead-letter."""
    buf = AsyncMock()
    push_started = asyncio.Event()

    async def slow_push(event):
        push_started.set()
        # Each push takes 100ms — we won't finish 20 in 0.2s.
        await asyncio.sleep(0.1)

    buf.push = slow_push

    dl = tmp_path / 'dl.jsonl'
    q = EventQueue(
        buf,
        dead_letter_path=dl,
        maxsize=100,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.05,
        shutdown_flush_seconds=0.2,
    )
    await q.start()
    for _ in range(20):
        q.enqueue(_make_event())
    # Wait until the drainer has at least started processing.
    await push_started.wait()
    await q.close()
    # Dead-letter should have some (not all) events with reason=shutdown_timeout.
    assert dl.exists()
    lines = dl.read_text().strip().splitlines()
    assert len(lines) >= 1, 'expected dead-lettered residue on timeout'
    reasons = {json.loads(line)['reason'] for line in lines}
    assert 'shutdown_timeout' in reasons


@pytest.mark.asyncio
async def test_enqueue_after_close_diverts_to_dead_letter(tmp_path, real_buffer):
    """Post-close enqueue writes to dead-letter (not lost, not raising)."""
    q = EventQueue(
        real_buffer,
        dead_letter_path=tmp_path / 'dl.jsonl',
        maxsize=100,
        shutdown_flush_seconds=1.0,
    )
    await q.start()
    await q.close()
    assert q.enqueue(_make_event()) is False
    dl = tmp_path / 'dl.jsonl'
    assert dl.exists()
    record = json.loads(dl.read_text().strip().splitlines()[0])
    assert record['reason'] == 'post_close'


# ── Stats surface ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stats_surface_has_expected_keys(queue):
    """stats() is the contract surface for WP-C watchdog + WP-D policy."""
    stats = queue.stats()
    expected = {
        'queue_depth',
        'queue_capacity',
        'last_commit_ts',
        'events_committed',
        'overflow_drops',
        'dead_letters',
        'retry_in_flight',
        'drainer_running',
    }
    assert expected.issubset(stats.keys())
    assert stats['drainer_running'] is True


@pytest.mark.asyncio
async def test_stats_tracks_commits(queue, real_buffer):
    """events_committed and last_commit_ts advance after each successful push."""
    assert queue.stats()['events_committed'] == 0
    assert queue.stats()['last_commit_ts'] is None
    queue.enqueue(_make_event())
    await asyncio.wait_for(queue._queue.join(), timeout=1.0)
    stats = queue.stats()
    assert stats['events_committed'] == 1
    assert stats['last_commit_ts'] is not None


# ── Rotation ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dead_letter_rotation_basic(tmp_path):
    """After file exceeds max_bytes, it is rotated to .jsonl.1 and a fresh .jsonl starts."""
    buf = AsyncMock()
    buf.push = AsyncMock(side_effect=ValueError('non-retriable'))
    dl = tmp_path / 'dead_letter.jsonl'

    # max_bytes=500 — each record is ~300 bytes, so rotation fires after 2 events.
    q = EventQueue(
        buf,
        dead_letter_path=dl,
        maxsize=1000,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.05,
        shutdown_flush_seconds=2.0,
        max_bytes=500,
        keep_rotations=2,
    )
    await q.start()
    try:
        # 3 events: events 1+2 fill the file (~600 bytes), event 3 triggers rotation.
        # After rotation dl.jsonl starts fresh with event 3 only (~300 bytes < 500).
        for _ in range(3):
            q.enqueue(_make_event())
        await asyncio.wait_for(q._queue.join(), timeout=2.0)

        # (a) Current dead_letter.jsonl must exist and be under 500 bytes.
        assert dl.exists(), 'dead_letter.jsonl must exist'
        current_size = dl.stat().st_size
        assert current_size < 500, (
            f'After rotation .jsonl should hold only 1 record (<500 bytes), got {current_size}'
        )

        # (b) At least one rotation must have been made (.jsonl.1 contains the older events).
        rotated = tmp_path / 'dead_letter.jsonl.1'
        assert rotated.exists(), 'dead_letter.jsonl.1 must exist after rotation'
        assert rotated.stat().st_size > 0

    finally:
        await q.close()


@pytest.mark.asyncio
async def test_dead_letter_rotation_drops_beyond_keep(tmp_path):
    """Rotation beyond keep_rotations=2 drops the oldest; .jsonl.3 must never appear."""
    buf = AsyncMock()
    buf.push = AsyncMock(side_effect=ValueError('non-retriable'))
    dl = tmp_path / 'dead_letter.jsonl'

    q = EventQueue(
        buf,
        dead_letter_path=dl,
        maxsize=1000,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.05,
        shutdown_flush_seconds=2.0,
        max_bytes=300,   # very tight — triggers rotation quickly
        keep_rotations=2,
    )
    await q.start()
    try:
        # Enqueue many events to trigger 3+ rotations.
        for _ in range(15):
            q.enqueue(_make_event())
        await asyncio.wait_for(q._queue.join(), timeout=5.0)

        # (c) .jsonl.2 may exist (keep_rotations=2 means keep up to .2).
        #     .jsonl.3 must NOT exist (dropped beyond keep_rotations).
        over_limit = tmp_path / 'dead_letter.jsonl.3'
        assert not over_limit.exists(), (
            'dead_letter.jsonl.3 must not exist when keep_rotations=2'
        )

    finally:
        await q.close()
