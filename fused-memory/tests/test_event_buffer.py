"""Tests for SQLite-backed reconciliation event buffer."""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from fused_memory.models.reconciliation import (
    EventSource,
    EventType,
    ReconciliationEvent,
)
from fused_memory.reconciliation.event_buffer import EventBuffer


def _make_event(
    project_id: str = 'test-project',
    event_type: EventType = EventType.episode_added,
    timestamp: datetime | None = None,
    agent_id: str | None = None,
) -> ReconciliationEvent:
    return ReconciliationEvent(
        id=str(uuid.uuid4()),
        type=event_type,
        source=EventSource.agent,
        project_id=project_id,
        timestamp=timestamp or datetime.now(timezone.utc),
        payload={'test': True},
        agent_id=agent_id,
    )


@pytest_asyncio.fixture
async def buf(tmp_path):
    b = EventBuffer(
        db_path=tmp_path / 'test.db',
        buffer_size_threshold=5,
        max_staleness_seconds=300,
    )
    await b.initialize()
    yield b
    await b.close()


# ── Original tests (adapted to async SQLite) ──────────────────────────


@pytest.mark.asyncio
async def test_push_and_stats(buf):
    event = _make_event()
    await buf.push(event)
    stats = await buf.get_buffer_stats('test-project')
    assert stats['size'] == 1
    assert stats['oldest_event_age_seconds'] is not None


@pytest.mark.asyncio
async def test_should_trigger_buffer_size(tmp_path):
    buf = EventBuffer(db_path=tmp_path / 'trigger.db', buffer_size_threshold=3, max_staleness_seconds=3600)
    await buf.initialize()
    try:
        for _ in range(3):
            await buf.push(_make_event())
        should, reason = await buf.should_trigger('test-project')
        assert should
        assert 'buffer_size' in reason
    finally:
        await buf.close()


@pytest.mark.asyncio
async def test_should_trigger_staleness(tmp_path):
    buf = EventBuffer(db_path=tmp_path / 'stale.db', buffer_size_threshold=100, max_staleness_seconds=1)
    await buf.initialize()
    try:
        old_time = datetime.now(timezone.utc) - timedelta(seconds=5)
        await buf.push(_make_event(timestamp=old_time))
        should, reason = await buf.should_trigger('test-project')
        assert should
        assert 'max_staleness' in reason
    finally:
        await buf.close()


@pytest.mark.asyncio
async def test_no_trigger_when_empty(buf):
    should, reason = await buf.should_trigger('test-project')
    assert not should
    assert reason == ''


@pytest.mark.asyncio
async def test_no_trigger_when_active_run(tmp_path):
    buf = EventBuffer(db_path=tmp_path / 'locked.db', buffer_size_threshold=1)
    await buf.initialize()
    try:
        await buf.push(_make_event())
        await buf.mark_run_active('test-project')
        should, reason = await buf.should_trigger('test-project')
        assert not should
    finally:
        await buf.close()


@pytest.mark.asyncio
async def test_drain_returns_events_and_clears(buf):
    await buf.push(_make_event())
    await buf.push(_make_event())

    events = await buf.drain('test-project')
    assert len(events) == 2

    # Buffer is now empty (drained events don't count)
    events2 = await buf.drain('test-project')
    assert len(events2) == 0


@pytest.mark.asyncio
async def test_mark_run_active_prevents_double_run(buf):
    assert await buf.mark_run_active('test-project') is True
    assert await buf.mark_run_active('test-project') is False


@pytest.mark.asyncio
async def test_mark_run_complete_allows_new_run(buf):
    await buf.mark_run_active('test-project')
    await buf.mark_run_complete('test-project')
    assert await buf.mark_run_active('test-project') is True


@pytest.mark.asyncio
async def test_separate_project_buffers(tmp_path):
    buf = EventBuffer(db_path=tmp_path / 'sep.db', buffer_size_threshold=2, conditional_trigger_ratio=0.0)
    await buf.initialize()
    try:
        await buf.push(_make_event(project_id='project-a'))
        await buf.push(_make_event(project_id='project-b'))
        await buf.push(_make_event(project_id='project-a'))

        should_a, _ = await buf.should_trigger('project-a')
        should_b, _ = await buf.should_trigger('project-b')
        assert should_a is True
        assert should_b is False
    finally:
        await buf.close()


@pytest.mark.asyncio
async def test_get_active_projects(buf):
    await buf.push(_make_event(project_id='p1'))
    await buf.push(_make_event(project_id='p2'))
    projects = await buf.get_active_projects()
    assert set(projects) == {'p1', 'p2'}


@pytest.mark.asyncio
async def test_empty_project_stats(buf):
    stats = await buf.get_buffer_stats('nonexistent')
    assert stats == {'size': 0, 'oldest_event_age_seconds': None}


# ── New tests: cross-instance, burst detection, quiescence ────────────


@pytest.mark.asyncio
async def test_cross_instance_visibility(tmp_path):
    """Two EventBuffer instances on the same db see each other's events."""
    db_path = tmp_path / 'shared.db'
    buf_a = EventBuffer(db_path=db_path, buffer_size_threshold=5)
    buf_b = EventBuffer(db_path=db_path, buffer_size_threshold=5)
    await buf_a.initialize()
    await buf_b.initialize()
    try:
        await buf_a.push(_make_event())
        await buf_b.push(_make_event())

        stats_a = await buf_a.get_buffer_stats('test-project')
        stats_b = await buf_b.get_buffer_stats('test-project')
        assert stats_a['size'] == 2
        assert stats_b['size'] == 2
    finally:
        await buf_a.close()
        await buf_b.close()


@pytest.mark.asyncio
async def test_burst_detection_enters_bursting(tmp_path):
    """2+ events from the same agent within burst_window → bursting state."""
    buf = EventBuffer(
        db_path=tmp_path / 'burst.db',
        burst_window_seconds=30.0,
    )
    await buf.initialize()
    try:
        now = datetime.now(timezone.utc)
        await buf.push(_make_event(agent_id='agent-1', timestamp=now))
        await buf.push(_make_event(agent_id='agent-1', timestamp=now + timedelta(seconds=5)))

        db = buf._require_db()
        async with db.execute(
            "SELECT state FROM burst_state WHERE agent_id = 'agent-1'"
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row['state'] == 'bursting'
    finally:
        await buf.close()


@pytest.mark.asyncio
async def test_burst_cooldown_exits_bursting(tmp_path):
    """Agent exits bursting after cooldown elapses."""
    buf = EventBuffer(
        db_path=tmp_path / 'cooldown.db',
        burst_window_seconds=30.0,
        burst_cooldown_seconds=150.0,
        buffer_size_threshold=1000,
    )
    await buf.initialize()
    try:
        # Create a burst state with old last_write_at
        old_time = datetime.now(timezone.utc) - timedelta(seconds=200)
        await buf.push(_make_event(agent_id='agent-1', timestamp=old_time))
        await buf.push(_make_event(agent_id='agent-1', timestamp=old_time + timedelta(seconds=1)))

        # _is_quiescent will expire it
        result = await buf._is_quiescent()
        assert result is True

        # Verify state was updated
        db = buf._require_db()
        async with db.execute(
            "SELECT state FROM burst_state WHERE agent_id = 'agent-1'"
        ) as cursor:
            row = await cursor.fetchone()
        assert row['state'] == 'idle'
    finally:
        await buf.close()


@pytest.mark.asyncio
async def test_quiescent_trigger_at_33_percent(tmp_path):
    """165 events (33% of 500) + quiescent → triggers."""
    buf = EventBuffer(
        db_path=tmp_path / 'quiescent.db',
        buffer_size_threshold=500,
        conditional_trigger_ratio=0.33,
        burst_window_seconds=30.0,
        burst_cooldown_seconds=150.0,
    )
    await buf.initialize()
    try:
        for _ in range(165):
            await buf.push(_make_event())

        should, reason = await buf.should_trigger('test-project')
        assert should
        assert 'quiescent' in reason
    finally:
        await buf.close()


@pytest.mark.asyncio
async def test_not_quiescent_when_bursting(tmp_path):
    """165 events + active burst → no conditional trigger."""
    buf = EventBuffer(
        db_path=tmp_path / 'no_q_burst.db',
        buffer_size_threshold=500,
        conditional_trigger_ratio=0.33,
        burst_window_seconds=30.0,
        burst_cooldown_seconds=150.0,
    )
    await buf.initialize()
    try:
        now = datetime.now(timezone.utc)
        # Push events without agent_id (no burst tracking)
        for _ in range(163):
            await buf.push(_make_event())
        # Push 2 rapid events from same agent → bursting
        await buf.push(_make_event(agent_id='agent-1', timestamp=now))
        await buf.push(_make_event(agent_id='agent-1', timestamp=now + timedelta(seconds=1)))

        should, reason = await buf.should_trigger('test-project')
        assert not should
    finally:
        await buf.close()


@pytest.mark.asyncio
async def test_not_quiescent_when_queue_active(tmp_path):
    """165 events + pending queue items → no conditional trigger."""
    async def mock_queue_stats():
        return {'pending': 3, 'retry': 0, 'in_flight': 0}

    buf = EventBuffer(
        db_path=tmp_path / 'no_q_queue.db',
        buffer_size_threshold=500,
        conditional_trigger_ratio=0.33,
        queue_stats_fn=mock_queue_stats,
    )
    await buf.initialize()
    try:
        for _ in range(165):
            await buf.push(_make_event())

        should, reason = await buf.should_trigger('test-project')
        assert not should
    finally:
        await buf.close()


@pytest.mark.asyncio
async def test_cross_instance_lock(tmp_path):
    """Instance B can't lock while A holds it."""
    db_path = tmp_path / 'lock.db'
    buf_a = EventBuffer(db_path=db_path, instance_id='instance-a')
    buf_b = EventBuffer(db_path=db_path, instance_id='instance-b')
    await buf_a.initialize()
    await buf_b.initialize()
    try:
        assert await buf_a.mark_run_active('test-project') is True
        assert await buf_b.mark_run_active('test-project') is False

        await buf_a.mark_run_complete('test-project')
        assert await buf_b.mark_run_active('test-project') is True
    finally:
        await buf_a.close()
        await buf_b.close()


@pytest.mark.asyncio
async def test_stale_lock_recovery(tmp_path):
    """Old heartbeat → lock is auto-broken."""
    db_path = tmp_path / 'stale_lock.db'
    buf_a = EventBuffer(db_path=db_path, instance_id='instance-a', stale_lock_seconds=1)
    buf_b = EventBuffer(db_path=db_path, instance_id='instance-b', stale_lock_seconds=1)
    await buf_a.initialize()
    await buf_b.initialize()
    try:
        assert await buf_a.mark_run_active('test-project') is True

        # Simulate old heartbeat
        db = buf_a._require_db()
        old = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        await db.execute(
            'UPDATE reconciliation_locks SET heartbeat_at = ? WHERE project_id = ?',
            (old, 'test-project'),
        )
        await db.commit()

        # Instance B should be able to break the stale lock
        assert await buf_b.mark_run_active('test-project') is True
    finally:
        await buf_a.close()
        await buf_b.close()


@pytest.mark.asyncio
async def test_agent_id_none_excluded_from_burst(tmp_path):
    """Events without agent_id don't create burst state entries."""
    buf = EventBuffer(db_path=tmp_path / 'no_agent.db')
    await buf.initialize()
    try:
        await buf.push(_make_event())  # agent_id=None
        await buf.push(_make_event())

        db = buf._require_db()
        async with db.execute('SELECT COUNT(*) as cnt FROM burst_state') as cursor:
            row = await cursor.fetchone()
        assert row['cnt'] == 0
    finally:
        await buf.close()


@pytest.mark.asyncio
async def test_cleanup_drained(tmp_path):
    """Drained events older than cutoff get cleaned up."""
    buf = EventBuffer(db_path=tmp_path / 'cleanup.db')
    await buf.initialize()
    try:
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        await buf.push(_make_event(timestamp=old_time))
        await buf.push(_make_event())  # recent

        # Drain all
        await buf.drain('test-project')

        # Cleanup with 1h cutoff
        deleted = await buf.cleanup_drained(max_age_seconds=3600)
        assert deleted == 1

        # Recent drained event should still be there
        db = buf._require_db()
        async with db.execute(
            "SELECT COUNT(*) as cnt FROM event_buffer WHERE status = 'drained'"
        ) as cursor:
            row = await cursor.fetchone()
        assert row['cnt'] == 1
    finally:
        await buf.close()
