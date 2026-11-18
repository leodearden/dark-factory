"""Tests for the fire-and-forget event queue (WP-B)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import pathlib
import threading
import time
import uuid
from concurrent.futures import Future
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import aiosqlite
import pytest
import pytest_asyncio
from _dead_letter_helpers import (
    hold_fifo,
    journaled_events,
    journaled_ids,
    overflowed_queue,
    parked_queue,
    record_dead_letter_writes,
    seed_dead_letters,
)

from fused_memory.models.reconciliation import (
    EventSource,
    EventType,
    ReconciliationEvent,
)
from fused_memory.reconciliation.dead_letter_writer import DeadLetterWriter
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.event_journal import EventJournal
from fused_memory.reconciliation.event_queue import EventQueue, _iter_lines_reversed


async def _wait_for(
    get_current,
    predicate,
    *,
    timeout: float,
    description: str,
):
    """Poll ``get_current()`` until ``predicate`` accepts its value, or the deadline lapses.

    New waits for a real-EventBuffer drainer commit use ``_signal_after_commits``,
    which has no deadline to race; this poll and its ``_wait_for_buffer_size`` /
    ``_wait_for_drained`` wrappers remain only for their existing call sites.

    Load-tolerant alternative to gating a drainer handoff on
    ``asyncio.wait_for(queue._queue.join(), timeout=<small>)``: that pattern
    ties the pass/fail line to how fast a (possibly starved, under full-suite
    xdist load) event loop happened to schedule the drainer, not to whether
    the drainer actually did its job. Polling the observable outcome keeps the
    guard meaningful — it still fails loudly if the drainer never catches up
    within the (generous) outer deadline — without being sensitive to
    scheduling jitter.

    ``get_current`` may be sync or async (an awaitable return value is
    awaited automatically) — e.g. ``queue.stats`` or
    ``lambda: buffer.get_buffer_stats(project_id)``. Asserts (naming
    ``description``, ``timeout``, and the last observed value) rather than
    returning silently if the deadline lapses, so a future reader can tell
    "the drainer never ran" from "the drainer produced the wrong value".
    """
    async def _read():
        value = get_current()
        if asyncio.iscoroutine(value):
            value = await value
        return value

    deadline = time.monotonic() + timeout
    value = await _read()
    while not predicate(value) and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
        value = await _read()
    assert predicate(value), (
        f'{description}: not satisfied within {timeout}s (last observed: {value!r})'
    )
    return value


async def _wait_for_buffer_size(
    buffer: EventBuffer, project_id: str, expected: int, *, timeout: float = 10.0,
) -> dict:
    """Poll get_buffer_stats until size reaches expected, or the deadline lapses.

    Thin wrapper over ``_wait_for`` — see its docstring for the load-tolerance
    rationale.
    """
    return await _wait_for(
        lambda: buffer.get_buffer_stats(project_id),
        lambda stats: stats['size'] >= expected,
        timeout=timeout,
        description=f"buffer size for project {project_id!r} reaching {expected}",
    )


async def _wait_for_drained(queue, expected: int, *, timeout: float = 10.0) -> dict:
    """Poll until the drainer has committed ``expected`` events and the queue is empty.

    Gates on the queue's OWN accounting rather than on buffer rows because
    ``_events_committed += 1`` and the ``finally: task_done()`` that follows it are
    separated by no await point (event_queue.py::_commit_with_retry / _drain_loop) —
    so observing ``events_committed == expected`` guarantees ``queue.join()`` is already
    satisfied, and a subsequent ``close()`` never consults ``shutdown_flush_seconds``.
    Thin wrapper over ``_wait_for``; see its docstring for the load-tolerance rationale.
    """
    return await _wait_for(
        queue.stats,
        lambda s: s['events_committed'] >= expected and s['queue_depth'] == 0,
        timeout=timeout,
        description=f'drainer committing {expected} event(s) and emptying the queue',
    )


def _signal_after_commits(buffer: EventBuffer, count: int) -> asyncio.Event:
    """Wrap ``buffer.push`` so the returned Event is set once ``count`` pushes have returned.

    The Event is set inside the drainer's own push call, so its synchronous post-push
    bookkeeping (journal mark_processed, events_committed, task_done) has already run
    when a waiter resumes. That relies on no await point after push in
    fused-memory/src/fused_memory/reconciliation/event_queue.py::EventQueue._commit_with_retry
    / _drain_loop. Await ``.wait()`` bare; pytest-timeout bounds a drainer that never
    commits.
    """
    committed = asyncio.Event()
    inner_push = buffer.push
    commits = 0

    async def push(event: ReconciliationEvent) -> None:
        nonlocal commits
        await inner_push(event)
        commits += 1
        if commits == count:
            committed.set()

    buffer.push = push
    return committed


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
    # Wait for drainer to catch up. Polls the outcome (buffer size) under a
    # generous outer deadline rather than gating on how fast a (possibly
    # starved, under full-suite xdist load) event loop scheduled the drainer
    # — see _wait_for_buffer_size.
    stats = await _wait_for_buffer_size(real_buffer, 'test-project', 5)
    assert stats['size'] == 5


@pytest.mark.asyncio
@pytest.mark.xdist_group('event_queue_enqueue_timing')
async def test_enqueue_returns_immediately(queue):
    """enqueue is synchronous and non-blocking."""
    # Pre-create events so model construction doesn't inflate the timing.
    events = [_make_event() for _ in range(100)]
    t0 = time.perf_counter()
    for event in events:
        queue.enqueue(event)
    elapsed = time.perf_counter() - t0
    # 100 non-blocking puts should take well under 50ms in principle, but
    # under full-suite CPU oversubscription (many xdist workers contending
    # for cores) even non-blocking asyncio.Queue.put_nowait calls can be
    # scheduled late. xdist_group pins this test (and its sibling timing
    # test below) to a single worker, so a tighter ceiling than a bare
    # global widen still holds: 0.5s / 100 calls = 5ms/call, 10x the
    # original 0.5ms/call expectation, but still tight enough to catch a
    # meaningful per-call regression while staying well below the failure
    # mode this guards against: enqueue() accidentally awaiting the real
    # write (100 sequential real writes would take single-digit seconds or
    # more under lock contention).
    assert elapsed < 0.5, f'enqueue took {elapsed:.3f}s for 100 calls'


# ── Failure injection ──────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.xdist_group('event_queue_enqueue_timing')
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
        # Widened for load tolerance (see test_enqueue_returns_immediately
        # above, which this is grouped with via xdist_group); still orders
        # of magnitude below the failure mode of enqueue() blocking on the
        # (always-failing) real write for all 10 calls.
        assert elapsed < 2.0, f'enqueue path was not immune to lock: {elapsed:.3f}s'
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
    committed = _signal_after_commits(real_buffer, 5)

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
        await committed.wait()
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
        # Poll the outcome (dead_letters count) rather than gating on how
        # fast a (possibly starved, under full-suite xdist load) event loop
        # scheduled the drainer — see _wait_for.
        stats = await _wait_for(
            q.stats,
            lambda stats: stats['dead_letters'] >= 3,
            timeout=5.0,
            description='dead_letters reaching 3',
        )
        # All 3 should have been dead-lettered after a single failed push.
        assert stats['dead_letters'] == 3
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
        # enqueue hands overflow records to the writer thread without waiting.
        await q.flush_dead_letters()
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
@pytest.mark.parametrize('shutdown_flush_seconds', [2.0, 0.0])
async def test_shutdown_after_drain_leaves_no_dead_letters(
    tmp_path, real_buffer, shutdown_flush_seconds,
):
    """close() dead-letters nothing when the queue is already drained,
    regardless of shutdown_flush_seconds.

    The drain is awaited on the queue's own accounting (_wait_for_drained)
    BEFORE close() runs, so close() only ever performs a no-op flush here —
    shutdown_flush_seconds is never consulted; see _wait_for_drained's
    docstring for why that gate is airtight. Parametrized over 2.0 (the
    file's dominant convention) and 0.0 to prove the outcome is independent
    of the window rather than merely tolerant of a generous one:

    0.0 is not merely "small" — on CPython, asyncio.wait_for with timeout<=0
    does ensure_future(coro) then checks fut.done(), which is False for a
    freshly created coroutine, so it raises TimeoutError WITHOUT EVER READING
    THE CLOCK. The window is therefore "already expired" as a matter of
    logic, not of timing, so this test carries no wall-clock dependency in
    either direction.

    test_graceful_shutdown_close_drains_in_flight_events (below) covers the
    complementary case: close() actually draining events that are still in
    flight when it is called.
    """
    q = EventQueue(
        real_buffer,
        dead_letter_path=tmp_path / 'dl.jsonl',
        maxsize=100,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.1,
        shutdown_flush_seconds=shutdown_flush_seconds,
    )
    await q.start()
    for _ in range(50):
        q.enqueue(_make_event())
    await _wait_for_drained(q, 50)
    await q.close()
    # All events landed; no dead-letter residue.
    stats = await real_buffer.get_buffer_stats('test-project')
    assert stats['size'] == 50
    dl_path = tmp_path / 'dl.jsonl'
    assert not dl_path.exists() or dl_path.read_text() == ''


@pytest.mark.asyncio
async def test_graceful_shutdown_close_drains_in_flight_events(tmp_path, real_buffer):
    """close()'s own bounded flush drains events still in flight when called.

    test_shutdown_after_drain_leaves_no_dead_letters (above) gates the drain
    with _wait_for_drained before calling close(), so by the time close()
    runs the queue is already empty and its
    ``asyncio.wait_for(self._queue.join(), timeout=self._shutdown_flush)``
    only ever performs a no-op flush. This test exercises the success branch
    that leaves uncovered: close() called against a genuinely NON-EMPTY
    queue, which must still drain everything and dead-letter nothing.

    Gates the buffer's push on an ``asyncio.Event`` (not a sleep) so the
    overlap between "close() is awaiting queue.join()" and "events are still
    being committed" is deterministic rather than timing-dependent:
    ``push_started`` confirms the drainer has retrieved the first event and
    is blocked inside push() *before* close() is even invoked, and asyncio's
    FIFO ``call_soon`` scheduling guarantees ``close()``'s
    ``wait_for(queue.join())`` call runs — and suspends, since all 5 events
    are still unfinished — before the gate-release wakes the drainer. So
    close() is proven to observe a genuinely non-empty queue, not one that
    merely happened to drain first.
    """
    push_started = asyncio.Event()
    gate = asyncio.Event()
    original_push = real_buffer.push

    async def gated_push(event):
        push_started.set()
        await gate.wait()
        return await original_push(event)

    real_buffer.push = gated_push

    q = EventQueue(
        real_buffer,
        dead_letter_path=tmp_path / 'dl.jsonl',
        maxsize=100,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.1,
        shutdown_flush_seconds=5.0,
    )
    await q.start()
    for _ in range(5):
        q.enqueue(_make_event())
    # Wait for the drainer to pick up the first event and block inside
    # push() — guarantees all 5 events are still unfinished when close() is
    # invoked below.
    await push_started.wait()
    close_task = asyncio.create_task(q.close())
    # close_task's first step was scheduled above, before gate.set() below
    # schedules the drainer's resume — FIFO call_soon ordering means close()
    # has already entered (and suspended in) wait_for(queue.join()) against
    # a non-empty queue by the time the drainer wakes up and starts
    # committing.
    gate.set()
    await close_task

    stats = await real_buffer.get_buffer_stats('test-project')
    assert stats['size'] == 5
    dl_path = tmp_path / 'dl.jsonl'
    assert not dl_path.exists() or dl_path.read_text() == ''


@pytest.mark.asyncio
async def test_wait_for_drained_fails_loudly_when_drainer_stalls(tmp_path):
    """_wait_for_drained must assert (not return silently) when the drainer
    never catches up — pins the "fail loudly" contract _wait_for's docstring
    states, so a future rewrite of the gate can't silently swallow a genuine
    stall. None of the happy-path tests in this file would catch that
    regression on their own: they only ever exercise _wait_for /
    _wait_for_drained where the predicate becomes true before the deadline,
    so a silent-return rewrite would leave every one of them green.
    """
    buf = AsyncMock()

    # Drainer blocks forever — events_committed can never reach the target.
    async def blocking_push(event):
        await asyncio.sleep(1000)

    buf.push = blocking_push

    q = EventQueue(
        buf,
        dead_letter_path=tmp_path / 'dl.jsonl',
        maxsize=100,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.05,
        shutdown_flush_seconds=0.05,
    )
    await q.start()
    try:
        for _ in range(5):
            q.enqueue(_make_event())
        with pytest.raises(AssertionError):
            await _wait_for_drained(q, 5, timeout=0.5)
    finally:
        # Close without waiting for the blocked push.
        await q.close()


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
    await q.flush_dead_letters()
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
        'dead_letters_pending',
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
    # Poll the outcome (events_committed) rather than gating on how fast a
    # (possibly starved, under full-suite xdist load) event loop scheduled
    # the drainer — see _wait_for.
    stats = await _wait_for(
        queue.stats,
        lambda stats: stats['events_committed'] >= 1,
        timeout=5.0,
        description='events_committed reaching 1',
    )
    assert stats['events_committed'] == 1
    assert stats['last_commit_ts'] is not None


# ── Test helpers ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_drain_for_test_drains_enqueued_events(real_buffer, tmp_path):
    """_drain_for_test() waits until every enqueued event has been processed.

    Constructs an EventQueue wired to a real SQLite buffer, enqueues 5 events,
    then awaits q._drain_for_test() (its default timeout).  After the call
    returns the buffer must contain exactly 5 committed events
    (stats['size'] == 5).

    Relies on _drain_for_test's default 5.0s ceiling, which keeps the guard
    tolerant of a starved event loop under full-suite xdist load while still
    failing loudly if the drainer genuinely never runs.
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
        await q._drain_for_test()
        stats = await real_buffer.get_buffer_stats('test-project')
        assert stats['size'] == 5, (
            f"Expected 5 events in buffer after _drain_for_test, got {stats['size']}"
        )
    finally:
        await q.close()


@pytest.mark.asyncio
async def test_signal_after_commits_counts_only_returned_pushes(real_buffer):
    """The drainer-commit tests await this signal bare, so it must never fire
    before ``count`` pushes have returned with their rows committed."""
    calls: list[int] = []
    set_when_inner_push_returned: list[bool] = []
    original_push = real_buffer.push

    async def fails_first(event):
        calls.append(1)
        if len(calls) == 1:
            raise aiosqlite.OperationalError('transient lock')
        await original_push(event)
        set_when_inner_push_returned.append(committed.is_set())

    real_buffer.push = fails_first
    committed = _signal_after_commits(real_buffer, 2)

    with pytest.raises(aiosqlite.OperationalError):
        await real_buffer.push(_make_event())
    assert not committed.is_set()

    await real_buffer.push(_make_event())
    assert not committed.is_set()

    await real_buffer.push(_make_event())
    assert committed.is_set()
    assert set_when_inner_push_returned == [False, False]
    assert (await real_buffer.get_buffer_stats('test-project'))['size'] == 2


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
        # Enqueue enough events to trigger at least one rotation.
        for _ in range(10):
            q.enqueue(_make_event())
        await q._drain_for_test(timeout=5.0)

        # Rotation must have fired at least once.  With max_bytes=200 and
        # each dead-letter record being ~300 bytes, the first write fills the
        # file past the cap, so every subsequent write triggers a rotation.
        # A dead_letters count >= 2 is therefore a direct witness that a
        # rotation ran (the 2nd dead letter can only be written after
        # rotation cleared the file).
        dead_letters = q.stats()['dead_letters']
        assert dead_letters >= 2, (
            f'expected at least 2 dead letters (and thus at least one rotation), '
            f'got {dead_letters}'
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
async def test_seed_dead_letters_writes_the_writers_record_shape(tmp_path):
    """seed_dead_letters, which reader and replay tests use instead of the
    drainer, writes the same keys and value types as EventQueue's own writer,
    and the same event. A writer-side change therefore fails here rather than
    leaving every seeded reader test green."""
    written = tmp_path / 'written.jsonl'
    seeded = tmp_path / 'seeded.jsonl'
    q = EventQueue(AsyncMock(), dead_letter_path=written, maxsize=1)
    try:
        kept, overflowed = _make_event(), _make_event()
        assert q.enqueue(kept) is True
        assert q.enqueue(overflowed) is False
        await q.flush_dead_letters()
        [real] = [json.loads(line) for line in written.read_text().splitlines()]
    finally:
        await q.close()
    seed_dead_letters(seeded, [overflowed])
    [seed] = [json.loads(line) for line in seeded.read_text().splitlines()]

    assert {k: type(v) for k, v in seed.items()} == {k: type(v) for k, v in real.items()}
    assert seed['event'] == real['event']


def test_read_dead_letters_basic_keys(tmp_path):
    """read_dead_letters returns parsed records with required keys."""
    dl = tmp_path / 'dl.jsonl'
    seed_dead_letters(dl, [_make_event()])
    q = EventQueue(AsyncMock(), dead_letter_path=dl)

    records = q.read_dead_letters()
    assert len(records) == 1
    rec = records[0]
    assert 'event' in rec
    assert 'reason' in rec
    assert 'attempts' in rec
    assert 'failed_at' in rec
    assert rec['reason'] == 'non_retriable'
    assert rec['attempts'] == 1


def test_read_dead_letters_limit(tmp_path):
    """read_dead_letters(limit=N) returns at most N records, newest-first."""
    dl = tmp_path / 'dl.jsonl'
    events = [_make_event() for _ in range(5)]
    seed_dead_letters(dl, events)
    q = EventQueue(AsyncMock(), dead_letter_path=dl)

    limited = q.read_dead_letters(limit=2)
    assert [r['event']['id'] for r in limited] == [e.id for e in reversed(events)][:2]


def test_read_dead_letters_project_id_filter(tmp_path):
    """read_dead_letters(project_id=X) returns only records matching that project."""
    dl = tmp_path / 'dl.jsonl'
    seed_dead_letters(
        dl,
        [_make_event(project_id='proj-a') for _ in range(3)]
        + [_make_event(project_id='proj-b') for _ in range(2)],
    )
    q = EventQueue(AsyncMock(), dead_letter_path=dl)

    proj_a_records = q.read_dead_letters(project_id='proj-a')
    assert len(proj_a_records) == 3
    assert all(r['event']['project_id'] == 'proj-a' for r in proj_a_records)

    proj_b_records = q.read_dead_letters(project_id='proj-b')
    assert len(proj_b_records) == 2
    assert all(r['event']['project_id'] == 'proj-b' for r in proj_b_records)


def test_read_dead_letters_after_rotation(tmp_path):
    """After rotation, read_dead_letters merges current + rotated files newest-first."""
    dl = tmp_path / 'dl.jsonl'
    rotated = [_make_event() for _ in range(2)]
    current = [_make_event()]
    seed_dead_letters(tmp_path / 'dl.jsonl.1', rotated)
    seed_dead_letters(dl, current)
    q = EventQueue(AsyncMock(), dead_letter_path=dl, keep_rotations=2)

    all_records = q.read_dead_letters()
    assert [r['event']['id'] for r in all_records] == (
        [e.id for e in reversed(current)] + [e.id for e in reversed(rotated)]
    )


def test_read_dead_letters_tolerates_malformed_line(tmp_path):
    """A malformed JSON line is skipped; no exception is raised."""
    dl = tmp_path / 'dl.jsonl'
    seed_dead_letters(dl, [_make_event()])
    q = EventQueue(AsyncMock(), dead_letter_path=dl)

    # Inject a malformed line into the JSONL file.
    with dl.open('a', encoding='utf-8') as fh:
        fh.write('THIS IS NOT JSON\n')

    # Must not raise, must return the 1 valid record (malformed line skipped).
    records = q.read_dead_letters()
    assert len(records) == 1


def test_read_dead_letters_streams_without_loading_whole_files(tmp_path, monkeypatch):
    """Streaming implementation never calls read_text; older rotation files not opened when limit satisfied."""
    dl = tmp_path / 'dl.jsonl'
    # Both dl.jsonl and dl.jsonl.1 hold records, so the test exercises the
    # short-circuit optimization across rotation files.
    seed_dead_letters(tmp_path / 'dl.jsonl.1', [_make_event() for _ in range(3)])
    seed_dead_letters(dl, [_make_event() for _ in range(3)])
    q = EventQueue(AsyncMock(), dead_letter_path=dl, keep_rotations=2)

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


def test_read_dead_letters_tolerates_oserror(tmp_path, monkeypatch, caplog):
    """read_dead_letters never raises when a dead-letter file cannot be opened.

    The OSError handler inside `read_dead_letters` catches OSError raised by
    `_iter_lines_reversed()` (which opens the file via path.open('rb')), emits
    a WARNING containing 'cannot read', and continues.  The result is an empty
    list rather than a raised exception.

    The monkeypatch is selective: only the dead-letter path's open() raises;
    all other Path.open calls (SQLite fixtures, etc.) use the real open.
    """
    dl = tmp_path / 'dl.jsonl'
    seed_dead_letters(dl, [_make_event()])
    q = EventQueue(AsyncMock(), dead_letter_path=dl)

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


# ── count_dead_letters ─────────────────────────────────────────────────


def test_count_dead_letters_returns_zero_when_file_missing(tmp_path):
    """count_dead_letters returns 0 when the dead-letter file does not exist."""
    buf = AsyncMock()
    buf.push = AsyncMock(side_effect=ValueError('non-retriable'))
    dl = tmp_path / 'absent.jsonl'

    q = EventQueue(
        buf,
        dead_letter_path=dl,
        maxsize=100,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.05,
        shutdown_flush_seconds=2.0,
    )
    # Do NOT call start()/close() — file is never created.
    assert not dl.exists(), 'Precondition: file must not exist'

    assert q.count_dead_letters() == 0
    assert q.count_dead_letters(project_id='proj-x') == 0


def test_count_dead_letters_matches_read_dead_letters_no_filter(tmp_path):
    """count_dead_letters() == len(read_dead_letters()) when no filter is given."""
    dl = tmp_path / 'dl.jsonl'
    seed_dead_letters(dl, [_make_event() for _ in range(5)])
    q = EventQueue(AsyncMock(), dead_letter_path=dl)

    all_records = q.read_dead_letters()
    assert len(all_records) == 5, f'Expected 5 dead-lettered records, got {len(all_records)}'
    assert q.count_dead_letters() == len(all_records) == 5


def test_count_dead_letters_matches_read_dead_letters_project_id_filter(tmp_path):
    """count_dead_letters(project_id=X) == len(read_dead_letters(project_id=X))."""
    dl = tmp_path / 'dl.jsonl'
    seed_dead_letters(
        dl,
        [_make_event(project_id='proj-a') for _ in range(3)]
        + [_make_event(project_id='proj-b') for _ in range(2)],
    )
    q = EventQueue(AsyncMock(), dead_letter_path=dl)

    proj_a_records = q.read_dead_letters(project_id='proj-a')
    assert len(proj_a_records) == 3
    assert q.count_dead_letters(project_id='proj-a') == len(proj_a_records) == 3

    proj_b_records = q.read_dead_letters(project_id='proj-b')
    assert len(proj_b_records) == 2
    assert q.count_dead_letters(project_id='proj-b') == len(proj_b_records) == 2


def test_count_dead_letters_spans_rotated_files(tmp_path):
    """count_dead_letters spans current + rotated files; equals len(read_dead_letters())."""
    dl = tmp_path / 'dl.jsonl'
    seed_dead_letters(tmp_path / 'dl.jsonl.1', [_make_event() for _ in range(2)])
    seed_dead_letters(dl, [_make_event()])
    q = EventQueue(AsyncMock(), dead_letter_path=dl, keep_rotations=2)

    all_records = q.read_dead_letters()
    assert len(all_records) == 3, (
        f'read_dead_letters must return 3 records across rotated files; got {len(all_records)}'
    )
    assert q.count_dead_letters() == len(all_records) == 3


def test_count_dead_letters_diverges_from_read_dead_letters_on_malformed_line(
    tmp_path, caplog
):
    """count_dead_letters hot-path counts malformed lines; read_dead_letters skips them.

    This is a characterisation / regression-guard test for the divergence documented
    in the count_dead_letters() docstring (task 1162):

    - count_dead_letters() with no project_id: binary forward scan, no json.loads.
      Malformed lines are counted as records (they are non-empty bytes).
    - count_dead_letters(project_id=...) filter path: json.loads per line, skips
      malformed lines with a WARNING — same behaviour as read_dead_letters().
    - read_dead_letters(): skips malformed lines with a WARNING.

    If a future change "fixes" the hot-path to skip malformed lines (or removes the
    skip in read_dead_letters), this test fails and pinpoints the regression.
    """
    dl = tmp_path / 'dl.jsonl'
    seed_dead_letters(dl, [_make_event(project_id='proj-x') for _ in range(2)])
    q = EventQueue(AsyncMock(), dead_letter_path=dl)

    with dl.open('a', encoding='utf-8') as fh:
        fh.write('not-json-at-all\n')

    with caplog.at_level(logging.WARNING):
        valid_records = q.read_dead_letters()
        proj_records = q.read_dead_letters(project_id='proj-x')
        no_filter_count = q.count_dead_letters()
        filtered_count = q.count_dead_letters(project_id='proj-x')

    # read_dead_letters skips malformed lines — 2 valid records.
    assert len(valid_records) == 2, (
        f'read_dead_letters must skip malformed line; got {len(valid_records)} records'
    )

    # count_dead_letters hot-path (no project_id) counts malformed lines — 3 total.
    # This is the documented divergence from read_dead_letters().
    assert no_filter_count == 3, (
        f'count_dead_letters hot-path must count malformed line; got {no_filter_count}'
    )

    # count_dead_letters filter path (project_id given) skips malformed lines — matches read.
    assert filtered_count == len(proj_records) == 2, (
        f'count_dead_letters filter path must skip malformed line; got {filtered_count}'
    )

    # Both code paths must log a WARNING for the malformed line: read_dead_letters skips
    # with a warning, and count_dead_letters filter path (project_id given) also skips
    # with a warning.  Checking both ensures neither path quietly drops the log line.
    read_warns = [
        r for r in caplog.records
        if 'read_dead_letters' in r.message and 'malformed line' in r.message
    ]
    count_warns = [
        r for r in caplog.records
        if 'count_dead_letters' in r.message and 'malformed line' in r.message
    ]
    assert read_warns, 'Expected read_dead_letters to log a WARNING for the malformed line'
    assert count_warns, 'Expected count_dead_letters filter path to log a WARNING for the malformed line'


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

    The `max_line_bytes` overflow guard inside `_iter_lines_reversed` logs a
    WARNING containing 'carry buffer exceeded' and yields the accumulated bytes
    as a malformed fragment rather than raising.
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


# ── Durable-at-enqueue journal (task 2709) ─────────────────────────────────


@pytest.mark.asyncio
async def test_enqueue_is_durable_before_drain(real_buffer, tmp_path):
    """With journal_path set, enqueue() persists the event durably BEFORE the
    drainer runs — the kill-9-between-enqueue-and-drain guarantee.
    """
    ej_path = tmp_path / 'ej.db'
    q = EventQueue(
        real_buffer,
        dead_letter_path=tmp_path / 'dl.jsonl',
        journal_path=ej_path,
        maxsize=100,
    )
    try:
        event = _make_event()
        # NOTE: no q.start() — no drainer. The durable write must have already
        # happened synchronously inside enqueue().
        assert q.enqueue(event) is True
        # The in-memory queue still holds it (dispatch cache over the durable row).
        assert q._queue.qsize() == 1
        # A FRESH EventJournal over the same path sees the durable row — proof
        # the row survived independently of the (never-started) drainer.
        journal = EventJournal(ej_path)
        try:
            recovered = journal.load_unprocessed()
        finally:
            journal.close()
        assert [e.id for e in recovered] == [event.id]
    finally:
        if q._journal is not None:
            q._journal.close()


@pytest.mark.asyncio
async def test_no_journal_path_creates_no_journal(real_buffer, tmp_path):
    """Without journal_path (default None), no journal is created and enqueue
    behaves exactly as today — regression guard for the sync/non-blocking WP-B
    contract (no synchronous FULL commit is added to the hot path).
    """
    q = EventQueue(
        real_buffer,
        dead_letter_path=tmp_path / 'dl.jsonl',
        maxsize=100,
    )
    assert q._journal is None
    assert q.enqueue(_make_event()) is True
    assert q._queue.qsize() == 1
    # No journal db was created — the only *.db under tmp_path is the buffer's
    # own file (from the real_buffer fixture), never a journal db.
    assert {p.name for p in tmp_path.glob('*.db')} == {'buf.db'}


@pytest.mark.asyncio
async def test_drainer_marks_processed_on_commit(real_buffer, tmp_path):
    """After the drainer commits an event to the buffer, its journal row is
    removed — the in-memory queue is a pure dispatch cache over the durable row.
    """
    ej_path = tmp_path / 'ej.db'
    committed = _signal_after_commits(real_buffer, 1)
    q = EventQueue(
        real_buffer,
        dead_letter_path=tmp_path / 'dl.jsonl',
        journal_path=ej_path,
        maxsize=100,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.1,
        shutdown_flush_seconds=2.0,
    )
    await q.start()
    try:
        event = _make_event()
        assert q.enqueue(event) is True
        await committed.wait()
        # Committed to the buffer.
        stats = await real_buffer.get_buffer_stats('test-project')
        assert stats['size'] == 1
        # Journal row removed (drainer marked it processed after the push).
        assert journaled_events(ej_path) == {}
    finally:
        await q.close()


@pytest.mark.asyncio
async def test_drainer_marks_processed_on_dead_letter(tmp_path):
    """When the drainer dead-letters a non-retriable event, its journal row is
    removed too — durability is handed off to the dead-letter JSONL file.
    """
    push_failed = asyncio.Event()

    def fail(event):
        push_failed.set()
        raise ValueError('schema mismatch')

    buf = AsyncMock()
    buf.push = AsyncMock(side_effect=fail)
    ej_path = tmp_path / 'ej.db'
    dl = tmp_path / 'dl.jsonl'
    q = EventQueue(
        buf,
        dead_letter_path=dl,
        journal_path=ej_path,
        maxsize=100,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.05,
        shutdown_flush_seconds=2.0,
    )
    await q.start()
    try:
        event = _make_event()
        assert q.enqueue(event) is True
        await push_failed.wait()
        await q.flush_dead_letters()
        # Dead-lettered (non-retriable push error).
        assert q.stats()['dead_letters'] == 1
        lines = dl.read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])['reason'] == 'non_retriable'
        # Journal row removed — durability handed off to the dead-letter file.
        assert journaled_events(ej_path) == {}
    finally:
        await q.close()


@pytest.mark.asyncio
async def test_startup_recovery_after_kill9(real_buffer, tmp_path):
    """Acceptance: an event durable in the journal but never committed to the
    buffer (a hard kill between enqueue and drain) is present AND processed
    after a restart that shares the same journal_path.
    """
    ej_path = tmp_path / 'ej.db'

    # queue1: enqueue WITHOUT starting the drainer, so the event is durable in
    # the journal but never reaches the buffer — simulating kill -9 between
    # enqueue and drain. queue1's in-memory queue is then abandoned.
    queue1 = EventQueue(
        real_buffer,
        dead_letter_path=tmp_path / 'dl.jsonl',
        journal_path=ej_path,
        maxsize=100,
    )
    event = _make_event()
    assert queue1.enqueue(event) is True
    # Not in the buffer yet — no drainer ran.
    assert (await real_buffer.get_buffer_stats('test-project'))['size'] == 0
    # Release queue1's journal fd (a real kill -9 frees it via OS cleanup); the
    # INSERT was already committed at append, so the row survives.
    if queue1._journal is not None:
        queue1._journal.close()

    # queue2: fresh EventQueue over the SAME buffer + journal_path. start()
    # must recover() the unprocessed row and re-enqueue it for the drainer.
    committed = _signal_after_commits(real_buffer, 1)
    queue2 = EventQueue(
        real_buffer,
        dead_letter_path=tmp_path / 'dl.jsonl',
        journal_path=ej_path,
        maxsize=100,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.1,
        shutdown_flush_seconds=2.0,
    )
    await queue2.start()
    try:
        await committed.wait()
        # (1) present after restart.
        assert (await real_buffer.get_buffer_stats('test-project'))['size'] == 1
        # (2) processed after restart — journal drained.
        assert journaled_events(ej_path) == {}
    finally:
        await queue2.close()


# ── Dead-letter replay (task 2709) ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_replay_dead_letters_reenqueues_and_clears(tmp_path):
    """replay_dead_letters puts every dead-lettered event back in the queue
    unchanged (journaled, observed with the drainer parked so the state is
    settled rather than raced), clears them from the file, and a second call
    finds nothing to replay.

    The drainer's commit of re-enqueued events is covered by
    fused-memory/tests/test_event_queue.py::test_drainer_marks_processed_on_commit.
    """
    dl = tmp_path / 'dl.jsonl'
    ej = tmp_path / 'ej.db'
    async with parked_queue(dl, ej) as q:
        seeded = [_make_event() for _ in range(3)]
        seed_dead_letters(dl, seeded)
        assert len(dl.read_text().strip().splitlines()) == 3

        counts = q.replay_dead_letters()
        assert counts == {'replayed': 3, 'failed': 0}
        assert journaled_events(ej) == {e.id: e for e in seeded}
        # File no longer holds the replayed records.
        assert (dl.read_text().strip() if dl.exists() else '') == ''
        # Second call → nothing to replay (no re-replay).
        assert q.replay_dead_letters() == {'replayed': 0, 'failed': 0}


@pytest.mark.asyncio
async def test_replay_dead_letters_filters_by_project_id(tmp_path):
    """With project_id set, only matching records are replayed; non-matching
    records remain in the dead-letter file."""
    dl = tmp_path / 'dl.jsonl'
    ej = tmp_path / 'ej.db'
    async with parked_queue(dl, ej) as q:
        proj_a = [_make_event(project_id='proj-a') for _ in range(2)]
        seed_dead_letters(dl, proj_a)
        seed_dead_letters(dl, [_make_event(project_id='proj-b') for _ in range(3)])

        counts = q.replay_dead_letters(project_id='proj-a')
        assert counts == {'replayed': 2, 'failed': 0}
        assert journaled_events(ej) == {e.id: e for e in proj_a}
        # Non-matching proj-b records are preserved in the file.
        remaining = q.read_dead_letters()
        assert len(remaining) == 3
        assert all(r['event']['project_id'] == 'proj-b' for r in remaining)


@pytest.mark.asyncio
async def test_replay_dead_letters_preserves_unparseable(tmp_path):
    """A malformed/unparseable dead-letter line counts toward `failed` and is
    preserved (loud-over-silent — never silently dropped)."""
    dl = tmp_path / 'dl.jsonl'
    ej = tmp_path / 'ej.db'
    async with parked_queue(dl, ej) as q:
        good = _make_event()
        seed_dead_letters(dl, [good])
        with dl.open('a', encoding='utf-8') as fh:
            fh.write('{not valid json\n')

        counts = q.replay_dead_letters()
        assert counts == {'replayed': 1, 'failed': 1}
        assert journaled_events(ej) == {good.id: good}
        # The malformed line survives; the good record was replayed out.
        assert dl.read_text().splitlines() == ['{not valid json']


# ── Off-loop dead-letter hand-off (task 6085) ──────────────────────────────


@pytest.mark.asyncio
async def test_enqueue_overflow_returns_before_dead_letter_append_lands(tmp_path):
    """An overflowing enqueue hands its record to the writer thread and returns
    while the append is still held open, so the loop never waits on that I/O.
    """
    held = hold_fifo(tmp_path / 'dl.jsonl')
    e1, e2 = _make_event(), _make_event()
    async with overflowed_queue(held.path, e1, e2) as q:
        assert not held.reader_opened.is_set(), (
            'enqueue() only returned after the FIFO reader opened: the '
            'dead-letter append ran on the event-loop thread'
        )
        assert q.stats()['dead_letters_pending'] == 1

        held.release.set()
        await q.flush_dead_letters()

        records = held.records()
        assert [(r['reason'], r['event']['id']) for r in records] == [('overflow_drop', e2.id)]
        assert q.stats()['dead_letters'] == 1
        assert q.stats()['dead_letters_pending'] == 0


@pytest.mark.asyncio
async def test_overflow_journal_row_survives_until_its_dead_letter_is_written(tmp_path):
    """The overflowed event's journal row is released only once its record is
    on disk, so a kill -9 while the append is pending cannot lose it."""
    held = hold_fifo(tmp_path / 'dl.jsonl')
    ej = tmp_path / 'ej.db'
    e1, e2 = _make_event(), _make_event()
    async with overflowed_queue(held.path, e1, e2, journal_path=ej) as q:
        assert e2.id in journaled_ids(ej)

        held.release.set()
        await q.flush_dead_letters()

        assert [r['event']['id'] for r in held.records()] == [e2.id]
        journaled = journaled_ids(ej)
        assert e1.id in journaled
        assert e2.id not in journaled


@pytest.mark.asyncio
async def test_close_of_a_never_started_queue_closes_its_journal(tmp_path):
    """close() releases the journal's connection even if start() never ran.

    SQLite deletes a WAL database's ``-wal`` file when its last connection
    closes, so a surviving ``-wal`` file means the queue still holds one.
    """
    ej = tmp_path / 'ej.db'
    q = EventQueue(AsyncMock(), dead_letter_path=tmp_path / 'dl.jsonl', journal_path=ej)
    assert q.enqueue(_make_event()) is True
    assert (tmp_path / 'ej.db-wal').exists()

    await q.close()

    assert not (tmp_path / 'ej.db-wal').exists()


@pytest.mark.asyncio
async def test_failed_dead_letter_write_keeps_the_journal_row(tmp_path):
    """A dead-letter write that fails leaves the event's journal row for
    recover() to redeliver, rather than silently losing the event."""
    (tmp_path / 'blocker').write_text('a regular file, not a directory')
    ej = tmp_path / 'ej.db'
    e1, e2 = _make_event(), _make_event()
    async with overflowed_queue(tmp_path / 'blocker' / 'dl.jsonl', e1, e2, journal_path=ej) as q:
        await q.flush_dead_letters()

        assert e2.id in journaled_ids(ej)
        assert q.stats()['dead_letters'] == 0


@pytest.mark.asyncio
async def test_close_leaves_every_handed_off_record_on_disk_and_its_row_released(
    tmp_path, caplog,
):
    """After close(), every record handed to the dead-letter writer is on
    disk with its journal row released; only the event the drainer held when
    close() cancelled it may remain journaled.

    Only set relations and counts are asserted: which event the drainer holds
    when close() cancels it depends on scheduling. Every assertion message
    names each event's store, the queue's stats() and its WARNING+ log
    records, so a red under load shows which branch failed.
    """
    caplog.set_level(logging.WARNING, logger='fused_memory.reconciliation')
    buf = AsyncMock()

    async def blocking_push(event):
        await asyncio.sleep(1000)

    buf.push = blocking_push
    dl = tmp_path / 'dl.jsonl'
    ej = tmp_path / 'ej.db'
    q = EventQueue(
        buf,
        dead_letter_path=dl,
        journal_path=ej,
        maxsize=2,
        shutdown_flush_seconds=0.05,
    )
    await q.start()
    events = [_make_event() for _ in range(6)]
    for event in events:
        q.enqueue(event)
    await q.close()

    stats = q.stats()
    logs = [(r.name, r.levelname, r.getMessage()) for r in caplog.records]
    lines = dl.read_text(encoding='utf-8').splitlines() if dl.exists() else []
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f'unparseable dead-letter line {line!r} stats={stats} logs={logs}',
            ) from exc
    dead_lettered = {r['event']['id'] for r in records}
    journaled = journaled_ids(ej)
    placement = {
        index: sorted(
            store
            for store, ids in (('dead_letter', dead_lettered), ('journal', journaled))
            if event.id in ids
        )
        for index, event in enumerate(events)
    }
    evidence = f'placement={placement} stats={stats} logs={logs}'

    assert all(len(stores) == 1 for stores in placement.values()), evidence
    assert dead_lettered | journaled == {e.id for e in events}, evidence
    assert len(journaled) <= 1, evidence
    assert len(records) == len(dead_lettered), evidence
    assert stats['dead_letters_pending'] == 0, evidence
    assert stats['dead_letters'] == len(records), evidence
    overflow_records = sum(r['reason'] == 'overflow_drop' for r in records)
    assert overflow_records == stats['overflow_drops'] > 0, evidence
    assert {r['reason'] for r in records} <= {'overflow_drop', 'shutdown_timeout'}, evidence


@pytest.mark.asyncio
async def test_flush_dead_letters_returns_only_after_the_journal_row_release_is_scheduled(
    monkeypatch, tmp_path,
):
    """flush_dead_letters(), and so close(), must not return before a written
    record's journal-row release has been queued on the loop.

    This race lets a dead-lettered event stay journaled (the
    test_close_leaves_every_handed_off_record_on_disk_and_its_row_released
    flake). The writer thread's first hop onto the loop is parked to hold
    that ordering open.
    """
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    real_hop = loop.call_soon_threadsafe
    hop_parked = asyncio.Event()
    unpark = threading.Event()
    parked = False

    def park_first_off_loop_hop(callback, *args, **kwargs):
        nonlocal parked
        if threading.get_ident() != loop_thread and not parked:
            parked = True
            real_hop(hop_parked.set)
            unpark.wait()
        return real_hop(callback, *args, **kwargs)

    monkeypatch.setattr(loop, 'call_soon_threadsafe', park_first_off_loop_hop)
    held = hold_fifo(tmp_path / 'dl.jsonl')
    ej = tmp_path / 'ej.db'
    q = EventQueue(AsyncMock(), dead_letter_path=held.path, journal_path=ej, maxsize=1)
    try:
        e1, e2 = _make_event(), _make_event()
        assert q.enqueue(e1) is True
        assert q.enqueue(e2) is False
        held.release.set()
        await hop_parked.wait()

        flush = asyncio.ensure_future(q.flush_dead_letters())
        # Without the ordering guarantee, flush completes within three loop iterations.
        for _ in range(10):
            await asyncio.sleep(0)
        assert not flush.done(), (
            'flush_dead_letters() returned before the writer thread queued the '
            'journal-row release'
        )

        unpark.set()
        await flush
        journaled = journaled_ids(ej)
        assert e2.id not in journaled
        assert e1.id in journaled
    finally:
        unpark.set()
        await q.close()


@pytest.mark.asyncio
async def test_flush_dead_letters_releases_the_row_of_a_write_that_finished_before_the_flush(
    monkeypatch, tmp_path,
):
    """A record already on disk when flush_dead_letters() is called still has
    its journal-row release run before the flush returns.

    Blocking on the write's future holds the loop thread, so the release the
    writer thread queued on the loop has not run when the flush starts.
    """
    writes = record_dead_letter_writes(monkeypatch)
    ej = tmp_path / 'ej.db'
    e1, e2 = _make_event(), _make_event()
    async with overflowed_queue(tmp_path / 'dl.jsonl', e1, e2, journal_path=ej) as q:
        [written] = writes
        assert written.result(timeout=5) is True

        await q.flush_dead_letters()

        journaled = journaled_ids(ej)
        assert e2.id not in journaled
        assert e1.id in journaled


@pytest.mark.asyncio
async def test_close_releases_the_row_of_a_write_that_finished_before_close(
    monkeypatch, tmp_path,
):
    """close() runs the journal-row release of a record already on disk before
    it closes the journal, so recover() cannot redeliver a dead-lettered event."""
    writes = record_dead_letter_writes(monkeypatch)
    ej = tmp_path / 'ej.db'
    e1, e2 = _make_event(), _make_event()
    async with overflowed_queue(tmp_path / 'dl.jsonl', e1, e2, journal_path=ej):
        [written] = writes
        assert written.result(timeout=5) is True

    assert journaled_ids(ej) == {e1.id}


@pytest.mark.asyncio
async def test_a_cancelled_flush_leaves_no_error_on_the_loop_when_its_write_settles(tmp_path):
    """Cancelling flush_dead_letters() while a write is pending leaves nothing
    for the loop's exception handler once that write settles, and a later
    flush still releases the written record's journal row."""
    loop = asyncio.get_running_loop()
    contexts: list[dict] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: contexts.append(context))
    held = hold_fifo(tmp_path / 'dl.jsonl')
    ej = tmp_path / 'ej.db'
    q = EventQueue(AsyncMock(), dead_letter_path=held.path, journal_path=ej, maxsize=1)
    try:
        e1, e2 = _make_event(), _make_event()
        assert q.enqueue(e1) is True
        assert q.enqueue(e2) is False
        flush = asyncio.ensure_future(q.flush_dead_letters())
        await asyncio.sleep(0)
        flush.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await flush

        held.release.set()
        held.records()
        await q.flush_dead_letters()

        assert contexts == []
        assert e2.id not in journaled_ids(ej)
    finally:
        held.release.set()
        await q.close()
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_flush_dead_letters_is_cancelled_when_the_writers_flush_future_is(
    monkeypatch, tmp_path,
):
    """A cancelled writer future cancels the flush awaiting it, as
    asyncio.wrap_future would, rather than leaving that flush to hang."""
    q = EventQueue(AsyncMock(), dead_letter_path=tmp_path / 'dl.jsonl')
    try:
        cancelled: Future[None] = Future()
        cancelled.cancel()
        with monkeypatch.context() as patched:
            patched.setattr(DeadLetterWriter, 'flushed', lambda _writer: cancelled)
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(q.flush_dead_letters(), timeout=5)
    finally:
        await q.close()
