"""Tests for the fire-and-forget event queue (WP-B)."""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
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
from fused_memory.reconciliation.event_queue import EventQueue, _iter_lines_reversed


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


# ── Test helpers ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_drain_for_test_drains_enqueued_events(real_buffer, tmp_path):
    """_drain_for_test() waits until every enqueued event has been processed.

    Constructs an EventQueue wired to a real SQLite buffer, enqueues 5 events,
    then awaits q._drain_for_test(timeout=1.0).  After the call returns the
    buffer must contain exactly 5 committed events (stats['size'] == 5).
    """
    q = EventQueue(
        real_buffer,
        dead_letter_path=tmp_path / 'dead_letter.jsonl',
        maxsize=100,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.1,
        shutdown_flush_seconds=2.0,
    )
    await q.start()
    try:
        for _ in range(5):
            q.enqueue(_make_event())
        await q._drain_for_test(timeout=1.0)
        stats = await real_buffer.get_buffer_stats('test-project')
        assert stats['size'] == 5, (
            f"Expected 5 events in buffer after _drain_for_test, got {stats['size']}"
        )
    finally:
        await q.close()


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
        await q._drain_for_test(timeout=2.0)

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
        await q._drain_for_test(timeout=5.0)

        # (c) .jsonl.2 may exist (keep_rotations=2 means keep up to .2).
        #     .jsonl.3 must NOT exist (dropped beyond keep_rotations).
        over_limit = tmp_path / 'dead_letter.jsonl.3'
        assert not over_limit.exists(), (
            'dead_letter.jsonl.3 must not exist when keep_rotations=2'
        )

    finally:
        await q.close()


@pytest.mark.asyncio
async def test_dead_letter_rotation_zero_keep_discards_file(tmp_path):
    """keep_rotations=0: once the byte cap is hit the current file is unlinked.

    No .jsonl.1, .jsonl.2 … siblings should appear — the rotation just
    truncates by deleting the file so subsequent writes start fresh.
    """
    buf = AsyncMock()
    buf.push = AsyncMock(side_effect=ValueError('non-retriable'))
    dl = tmp_path / 'dead_letter.jsonl'

    q = EventQueue(
        buf,
        dead_letter_path=dl,
        maxsize=100,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.05,
        shutdown_flush_seconds=2.0,
        max_bytes=200,  # very tight — triggers rotation after a few records
        keep_rotations=0,
    )
    await q.start()
    try:
        # Enqueue enough events to exceed the 200-byte cap several times over.
        for _ in range(20):
            q.enqueue(_make_event())
        await q._drain_for_test(timeout=5.0)

        # No archived sibling should exist.
        assert not (tmp_path / 'dead_letter.jsonl.1').exists(), (
            '.jsonl.1 must not exist when keep_rotations=0'
        )
        # The current file may or may not exist (it was just written to after
        # last rotation), but it must be under the cap if it does exist.
        if dl.exists():
            assert dl.stat().st_size <= 200 * 5, (
                'dead_letter.jsonl grew unbounded with keep_rotations=0'
            )
    finally:
        await q.close()


@pytest.mark.asyncio
async def test_rotation_keep_zero_purges_orphan_rotations(tmp_path):
    """keep_rotations=0: pre-existing orphan siblings are purged on the next rotation.

    Simulates an operator who previously ran with keep_rotations=5 by pre-creating
    dead_letter.jsonl.1 through .5 on disk.  The current run uses keep_rotations=0.
    After triggering a rotation, all five orphan siblings must be unlinked and no
    new .1 sibling must appear (zero-keep never archives).
    """
    buf = AsyncMock()
    buf.push = AsyncMock(side_effect=ValueError('non-retriable'))
    dl = tmp_path / 'dead_letter.jsonl'

    # Pre-create orphan rotation files simulating a prior run with keep_rotations=5.
    for n in range(1, 6):
        orphan = tmp_path / f'dead_letter.jsonl.{n}'
        orphan.write_text('{"orphan": true}\n', encoding='utf-8')

    q = EventQueue(
        buf,
        dead_letter_path=dl,
        maxsize=100,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.05,
        shutdown_flush_seconds=2.0,
        max_bytes=200,  # very tight — triggers rotation after a few records
        keep_rotations=0,
    )
    await q.start()
    try:
        # Enqueue enough events to trigger at least one _rotate_dead_letter call.
        for _ in range(10):
            q.enqueue(_make_event())
        await q._drain_for_test(timeout=5.0)

        # Rotation must have fired at least once.  With max_bytes=200 and
        # each dead-letter record being ~300 bytes, the first write fills the
        # file past the cap, so every subsequent write triggers a rotation.
        # _dead_letters >= 2 is therefore a direct witness that
        # _rotate_dead_letter ran (the 2nd dead letter can only be written
        # after rotation cleared the file).
        assert q._dead_letters >= 2, (
            f'expected at least 2 dead letters (and thus at least one rotation), '
            f'got {q._dead_letters}'
        )

        # All orphan siblings must have been purged.
        for n in range(1, 6):
            orphan = tmp_path / f'dead_letter.jsonl.{n}'
            assert not orphan.exists(), (
                f'dead_letter.jsonl.{n} must be purged when keep_rotations=0'
            )
    finally:
        await q.close()


@pytest.mark.asyncio
async def test_rotation_purges_orphan_rotations(tmp_path):
    """After keep_rotations is reduced, orphaned rotation files are purged on next rotation.

    Simulates a previous run with keep_rotations=5 by pre-creating dead_letter.jsonl.3,
    .4, and .5 on disk.  The current run uses keep_rotations=2 and max_bytes=300.
    After triggering a rotation (by enqueuing ~5 events), the orphan files beyond
    keep_rotations must be unlinked while retained siblings (.1, .2) are untouched.
    """
    buf = AsyncMock()
    buf.push = AsyncMock(side_effect=ValueError('non-retriable'))
    dl = tmp_path / 'dead_letter.jsonl'

    # Pre-create orphan rotation files from a prior run with a higher keep_rotations.
    orphan_3 = tmp_path / 'dead_letter.jsonl.3'
    orphan_4 = tmp_path / 'dead_letter.jsonl.4'
    orphan_5 = tmp_path / 'dead_letter.jsonl.5'
    for orphan in (orphan_3, orphan_4, orphan_5):
        orphan.write_text('{"orphan": true}\n', encoding='utf-8')

    q = EventQueue(
        buf,
        dead_letter_path=dl,
        maxsize=1000,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.05,
        shutdown_flush_seconds=2.0,
        max_bytes=300,  # tight — triggers rotation after ~1 record (~300 bytes each)
        keep_rotations=2,
    )
    await q.start()
    try:
        # Enqueue enough events to trigger at least one rotation.
        for _ in range(5):
            q.enqueue(_make_event())
        await q._drain_for_test(timeout=5.0)

        # Orphan files beyond keep_rotations must be purged.
        assert not orphan_3.exists(), 'dead_letter.jsonl.3 must be purged after rotation'
        assert not orphan_4.exists(), 'dead_letter.jsonl.4 must be purged after rotation'
        assert not orphan_5.exists(), 'dead_letter.jsonl.5 must be purged after rotation'

        # Retained siblings (index ≤ keep_rotations) must not have been deleted.
        retained_1 = tmp_path / 'dead_letter.jsonl.1'
        retained_2 = tmp_path / 'dead_letter.jsonl.2'
        # With max_bytes=300 and ~5 records of ~300 bytes each, at least one
        # rotation must have fired — so .1 must exist regardless of timing.
        assert retained_1.exists(), 'rotation should have fired at least once given max_bytes=300 and 5 records'
        assert retained_1.stat().st_size > 0, '.jsonl.1 should have event data'
        if retained_2.exists():
            assert retained_2.stat().st_size > 0, '.jsonl.2 should have event data'

    finally:
        await q.close()


# ── read_dead_letters ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_dead_letters_basic_keys(tmp_path):
    """read_dead_letters returns parsed records with required keys."""
    buf = AsyncMock()
    buf.push = AsyncMock(side_effect=ValueError('non-retriable'))
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
        q.enqueue(_make_event())
        await asyncio.wait_for(q._queue.join(), timeout=2.0)

        records = q.read_dead_letters()
        assert len(records) == 1
        rec = records[0]
        assert 'event' in rec
        assert 'reason' in rec
        assert 'attempts' in rec
        assert 'failed_at' in rec
        assert rec['reason'] == 'non_retriable'
        assert rec['attempts'] == 1
    finally:
        await q.close()


@pytest.mark.asyncio
async def test_read_dead_letters_limit(tmp_path):
    """read_dead_letters(limit=N) returns at most N records, newest-first."""
    buf = AsyncMock()
    buf.push = AsyncMock(side_effect=ValueError('non-retriable'))
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
        for _ in range(5):
            q.enqueue(_make_event())
        await asyncio.wait_for(q._queue.join(), timeout=2.0)

        limited = q.read_dead_letters(limit=2)
        assert len(limited) == 2
        # Newest-first: the last 2 written records should be returned first.
        all_records = q.read_dead_letters()
        assert len(all_records) == 5
        assert limited == all_records[:2]
    finally:
        await q.close()


@pytest.mark.asyncio
async def test_read_dead_letters_project_id_filter(tmp_path):
    """read_dead_letters(project_id=X) returns only records matching that project."""
    buf = AsyncMock()
    buf.push = AsyncMock(side_effect=ValueError('non-retriable'))
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
            q.enqueue(_make_event(project_id='proj-a'))
        for _ in range(2):
            q.enqueue(_make_event(project_id='proj-b'))
        await asyncio.wait_for(q._queue.join(), timeout=2.0)

        proj_a_records = q.read_dead_letters(project_id='proj-a')
        assert len(proj_a_records) == 3
        assert all(r['event']['project_id'] == 'proj-a' for r in proj_a_records)

        proj_b_records = q.read_dead_letters(project_id='proj-b')
        assert len(proj_b_records) == 2
        assert all(r['event']['project_id'] == 'proj-b' for r in proj_b_records)
    finally:
        await q.close()


@pytest.mark.asyncio
async def test_read_dead_letters_after_rotation(tmp_path):
    """After rotation, read_dead_letters merges current + rotated files newest-first."""
    buf = AsyncMock()
    buf.push = AsyncMock(side_effect=ValueError('non-retriable'))
    dl = tmp_path / 'dl.jsonl'

    q = EventQueue(
        buf,
        dead_letter_path=dl,
        maxsize=100,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.05,
        shutdown_flush_seconds=2.0,
        max_bytes=500,
        keep_rotations=2,
    )
    await q.start()
    try:
        # 3 events: events 1+2 fill the pre-rotation file, event 3 goes to fresh file.
        for _ in range(3):
            q.enqueue(_make_event())
        await asyncio.wait_for(q._queue.join(), timeout=2.0)

        # Rotation should have happened; .jsonl has event3, .jsonl.1 has events1+2.
        assert (tmp_path / 'dl.jsonl.1').exists(), 'Rotation must have occurred'

        all_records = q.read_dead_letters()
        assert len(all_records) == 3, (
            f'Must return records from current + rotated file; got {len(all_records)}'
        )
    finally:
        await q.close()


@pytest.mark.asyncio
async def test_read_dead_letters_tolerates_malformed_line(tmp_path):
    """A malformed JSON line is skipped; no exception is raised."""
    buf = AsyncMock()
    buf.push = AsyncMock(side_effect=ValueError('non-retriable'))
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
        q.enqueue(_make_event())
        await asyncio.wait_for(q._queue.join(), timeout=2.0)
    finally:
        await q.close()

    # Inject a malformed line into the JSONL file.
    with dl.open('a', encoding='utf-8') as fh:
        fh.write('THIS IS NOT JSON\n')

    # Must not raise, must return the 1 valid record (malformed line skipped).
    records = q.read_dead_letters()
    assert len(records) == 1


@pytest.mark.asyncio
async def test_read_dead_letters_streams_without_loading_whole_files(tmp_path, monkeypatch):
    """Streaming implementation never calls read_text; older rotation files not opened when limit satisfied."""
    buf = AsyncMock()
    buf.push = AsyncMock(side_effect=ValueError('non-retriable'))
    dl = tmp_path / 'dl.jsonl'

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
        for _ in range(10):
            q.enqueue(_make_event())
        await asyncio.wait_for(q._queue.join(), timeout=5.0)
    finally:
        await q.close()

    # Precondition: both dl.jsonl and dl.jsonl.1 must exist so the test
    # exercises the short-circuit optimization across rotation files.
    assert dl.exists(), 'dl.jsonl must exist (precondition)'
    assert (tmp_path / 'dl.jsonl.1').exists(), (
        'dl.jsonl.1 must exist after rotation (precondition)'
    )

    # Track every path opened by read_dead_letters.
    opened_paths: list[str] = []
    _original_open = pathlib.Path.open

    def _tracking_open(self: pathlib.Path, *args, **kwargs):
        opened_paths.append(str(self))
        return _original_open(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, 'open', _tracking_open)

    # Patch read_text to raise — the new implementation must NOT call it.
    # If the old implementation is still in place it will call read_text,
    # the AssertionError is caught by the outer `except Exception` guard,
    # and read_dead_letters returns [] — causing the len==1 assertion below
    # to fail and clearly pinpointing the regression.
    def _raise_on_read_text(self: pathlib.Path, *args, **kwargs):
        raise AssertionError('read_text must not be called; use streaming open()')

    monkeypatch.setattr(pathlib.Path, 'read_text', _raise_on_read_text)

    result = q.read_dead_letters(limit=1)

    # (1) Exactly 1 record returned — implicitly proves read_text was never called.
    assert len(result) == 1, (
        f'Expected 1 record; got {len(result)}. '
        'If 0, read_text was called (raising AssertionError caught by outer guard).'
    )

    # (2) Older rotation files must not have been opened when limit is already
    #     satisfied by the current file.
    rotation_opens = [
        p for p in opened_paths if 'dl.jsonl.1' in p or 'dl.jsonl.2' in p
    ]
    assert rotation_opens == [], (
        f'Older rotation files should not be opened when limit is satisfied '
        f'by the current file, but found: {rotation_opens}'
    )


@pytest.mark.asyncio
async def test_read_dead_letters_tolerates_oserror(tmp_path, monkeypatch, caplog):
    """read_dead_letters never raises when a dead-letter file cannot be opened.

    The OSError handler at event_queue.py:481 catches OSError raised by
    _iter_lines_reversed() (which opens the file via path.open('rb')), emits a
    WARNING containing 'cannot read', and continues.  The result is an empty
    list rather than a raised exception.

    The monkeypatch is selective: only the dead-letter path's open() raises;
    all other Path.open calls (SQLite fixtures, etc.) use the real open.
    """
    buf = AsyncMock()
    buf.push = AsyncMock(side_effect=ValueError('non-retriable'))
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
        q.enqueue(_make_event())
        await asyncio.wait_for(q._queue.join(), timeout=2.0)
    finally:
        await q.close()

    # Confirm the dead-letter file was actually written.
    assert dl.exists(), 'dead-letter file must exist before patching open'

    # Patch Path.open selectively: only fail when opening the dead-letter path.
    _original_open = pathlib.Path.open

    def _failing_open(self: pathlib.Path, *args, **kwargs):
        if str(self) == str(dl):
            raise OSError('simulated unreadable file')
        return _original_open(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, 'open', _failing_open)

    with caplog.at_level(logging.WARNING):
        result = q.read_dead_letters()

    # (a) Returns [] without raising.
    assert result == [], f'expected empty list, got {result!r}'

    # (b) At least one WARNING containing 'cannot read'.
    assert any(
        'cannot read' in r.message for r in caplog.records
    ), f"Expected 'cannot read' warning; got: {[r.message for r in caplog.records]}"


def test_read_dead_letters_cross_file_ordering(tmp_path):
    """Newest-first ordering holds across file boundaries (current file → rotated .1).

    Writes records directly to dl.jsonl and dl.jsonl.1 with known timestamps so
    the expected order is deterministic.  The current file (dl.jsonl) must appear
    before the rotated file in the results, and within each file lines must be
    reversed (last-written = newest = first in results).
    """
    buf = AsyncMock()
    dl = tmp_path / 'dl.jsonl'
    dl_1 = tmp_path / 'dl.jsonl.1'

    # Write 3 records to the rotated file (older timestamps: 2026-01-01).
    with dl_1.open('w', encoding='utf-8') as fh:
        for i in range(3):
            rec = {
                'event': {
                    'id': f'old-{i}',
                    'project_id': 'test-project',
                    'timestamp': f'2026-01-01T00:00:0{i}+00:00',
                },
                'reason': 'test',
                'attempts': 1,
                'failed_at': f'2026-01-01T00:00:0{i}+00:00',
            }
            fh.write(json.dumps(rec) + '\n')

    # Write 2 records to the current file (newer timestamps: 2026-01-02).
    with dl.open('w', encoding='utf-8') as fh:
        for i in range(2):
            rec = {
                'event': {
                    'id': f'new-{i}',
                    'project_id': 'test-project',
                    'timestamp': f'2026-01-02T00:00:0{i}+00:00',
                },
                'reason': 'test',
                'attempts': 1,
                'failed_at': f'2026-01-02T00:00:0{i}+00:00',
            }
            fh.write(json.dumps(rec) + '\n')

    q = EventQueue(
        buf,
        dead_letter_path=dl,
        maxsize=100,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.05,
        shutdown_flush_seconds=2.0,
        keep_rotations=2,
    )

    all_records = q.read_dead_letters()
    assert len(all_records) == 5, (
        f'Expected 5 records (2 from current + 3 from rotated); got {len(all_records)}'
    )

    ids = [r['event']['id'] for r in all_records]
    # Current file (dl.jsonl) comes first, newest-first within the file:
    #   wrote new-0 then new-1  →  reversed: new-1, new-0
    # Rotated file (dl.jsonl.1) comes second, newest-first within the file:
    #   wrote old-0, old-1, old-2  →  reversed: old-2, old-1, old-0
    assert ids == ['new-1', 'new-0', 'old-2', 'old-1', 'old-0'], (
        f'Expected newest-first across file boundaries, got: {ids}'
    )

    # Cross-boundary: the oldest record from the current file must be newer
    # than the newest record from the rotated file.
    boundary_current_oldest = all_records[1]['failed_at']   # 'new-0' → 2026-01-02T00:00:00
    boundary_rotated_newest = all_records[2]['failed_at']   # 'old-2' → 2026-01-01T00:00:02
    assert boundary_current_oldest > boundary_rotated_newest, (
        f'Cross-file boundary ordering violated: '
        f'last current-file record ({boundary_current_oldest}) should be newer than '
        f'first rotated-file record ({boundary_rotated_newest})'
    )


# ── _iter_lines_reversed ───────────────────────────────────────────────


def test_iter_lines_reversed_handles_chunk_boundaries(tmp_path):
    """All three carry-stitching branches execute when the file spans multiple chunks.

    The default chunk_size=8192 means existing tests (dead-letter records ~300 bytes
    each, ≤ 5 per file) never cross a chunk boundary. This test writes 30 lines of
    ~400 bytes each (~12 KiB total) so that every chunk boundary fires:

      - carry-prepend: ``data = chunk + carry`` at each iteration
      - ``reversed(lines[1:])`` emitting multiple complete lines per chunk
      - final ``if carry:`` yielding the first line in the file (no preceding '\\n')

    Asserts all 30 lines are returned newest-first with no loss or duplication.
    """
    lines_written = [f'line-{i:04d}-' + 'x' * 380 for i in range(30)]  # ~400 bytes × 30 ≈ 12 KB
    dl = tmp_path / 'multi_chunk.jsonl'
    with dl.open('w', encoding='utf-8') as fh:
        for line in lines_written:
            fh.write(line + '\n')

    # Precondition: file must exceed the default chunk_size so carry/stitch paths fire.
    assert dl.stat().st_size > 8192, (
        f'file too small ({dl.stat().st_size} bytes) — increase line count or width'
    )

    result = [line for line in _iter_lines_reversed(dl) if line]  # drop empty trailing split

    # All 30 lines yielded — no loss, no duplication.
    assert len(result) == 30, f'expected 30 lines, got {len(result)}'
    assert len(set(result)) == 30, 'duplicate lines in result'
    # Newest-first across chunk boundaries.
    assert result == list(reversed(lines_written)), (
        'Lines not in newest-first order across chunk boundaries'
    )


def test_iter_lines_reversed_max_line_bytes_overflow(tmp_path, caplog):
    """Carry-buffer overflow guard fires when a single line exceeds max_line_bytes.

    The guard at event_queue.py:86-94 logs a WARNING containing 'carry buffer exceeded'
    and yields the accumulated bytes as a malformed fragment rather than raising.
    With the default max_line_bytes=1_048_576 (1 MiB) the branch is unreachable
    in production; this test drives it with chunk_size=50 and max_line_bytes=100.

    Asserts:
      (a) No exception propagates — the generator completes cleanly.
      (b) A WARNING containing 'carry buffer exceeded' is logged.
      (c) No bytes are silently lost — the 500 'X' chars appear across all yields.
    """
    dl = tmp_path / 'huge_line.jsonl'
    # Single line of 500 bytes. With chunk_size=50 and max_line_bytes=100 the
    # carry accumulates until it exceeds 100 bytes, triggering the overflow guard.
    huge_line = 'X' * 500
    with dl.open('w', encoding='utf-8') as fh:
        fh.write(huge_line + '\n')

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.event_queue'):
        results = list(_iter_lines_reversed(dl, chunk_size=50, max_line_bytes=100))

    # (a) No exception — generator completed.
    # (b) Overflow warning logged.
    assert any(
        'carry buffer exceeded' in r.message for r in caplog.records
    ), f"Expected 'carry buffer exceeded' warning; got: {[r.message for r in caplog.records]}"

    # (c) No bytes lost — all 500 'X's appear across the yielded fragments.
    total_x = sum(line.count('X') for line in results)
    assert total_x == 500, f"expected 500 X's across yields, got {total_x}"
