"""Tests for SQLite-backed reconciliation event buffer."""

import uuid
from datetime import UTC, datetime, timedelta

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
        timestamp=timestamp or datetime.now(UTC),
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
        old_time = datetime.now(UTC) - timedelta(seconds=5)
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
        now = datetime.now(UTC)
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
        old_time = datetime.now(UTC) - timedelta(seconds=200)
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
        assert row is not None
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
        now = datetime.now(UTC)
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
        return {'counts': {'pending': 3, 'retry': 0, 'in_flight': 0}, 'oldest_pending_age_seconds': None}

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
async def test_not_quiescent_when_queue_has_in_flight(tmp_path):
    """165 events + in-flight queue items → no conditional trigger."""
    async def mock_queue_stats():
        return {'counts': {'in_flight': 1}, 'oldest_pending_age_seconds': None}

    buf = EventBuffer(
        db_path=tmp_path / 'no_q_inflight.db',
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
async def test_not_quiescent_when_queue_has_retries(tmp_path):
    """165 events + retry queue items → no conditional trigger."""
    async def mock_queue_stats():
        return {'counts': {'retry': 2}, 'oldest_pending_age_seconds': 10.0}

    buf = EventBuffer(
        db_path=tmp_path / 'no_q_retry.db',
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
async def test_quiescent_when_queue_truly_empty(tmp_path):
    """165 events + empty queue → conditional trigger fires."""
    async def mock_queue_stats():
        return {'counts': {}, 'oldest_pending_age_seconds': None}

    buf = EventBuffer(
        db_path=tmp_path / 'q_empty.db',
        buffer_size_threshold=500,
        conditional_trigger_ratio=0.33,
        queue_stats_fn=mock_queue_stats,
    )
    await buf.initialize()
    try:
        for _ in range(165):
            await buf.push(_make_event())

        should, reason = await buf.should_trigger('test-project')
        assert should
        assert reason == 'quiescent:165'
    finally:
        await buf.close()


@pytest.mark.asyncio
async def test_quiescent_when_queue_all_completed(tmp_path):
    """165 events + only completed items in queue → conditional trigger fires."""
    async def mock_queue_stats():
        return {'counts': {'completed': 50}, 'oldest_pending_age_seconds': None}

    buf = EventBuffer(
        db_path=tmp_path / 'q_completed.db',
        buffer_size_threshold=500,
        conditional_trigger_ratio=0.33,
        queue_stats_fn=mock_queue_stats,
    )
    await buf.initialize()
    try:
        for _ in range(165):
            await buf.push(_make_event())

        should, reason = await buf.should_trigger('test-project')
        assert should
        assert reason == 'quiescent:165'
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
        old = (datetime.now(UTC) - timedelta(seconds=10)).isoformat()
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
        assert row is not None
        assert row['cnt'] == 0
    finally:
        await buf.close()


@pytest.mark.asyncio
async def test_restore_drained_moves_events_back_to_buffered(buf):
    """Push 3, drain, restore → count should be 3 again."""
    for _ in range(3):
        await buf.push(_make_event())

    await buf.drain('test-project')
    assert (await buf.get_buffer_stats('test-project'))['size'] == 0

    restored = await buf.restore_drained('test-project')
    assert restored == 3
    assert (await buf.get_buffer_stats('test-project'))['size'] == 3


@pytest.mark.asyncio
async def test_restore_drained_returns_zero_when_empty(buf):
    """Restore on empty buffer returns 0."""
    restored = await buf.restore_drained('test-project')
    assert restored == 0


@pytest.mark.asyncio
async def test_restore_drained_only_affects_target_project(tmp_path):
    """Multi-project isolation — restore only affects the specified project."""
    buf = EventBuffer(db_path=tmp_path / 'multi.db', buffer_size_threshold=5)
    await buf.initialize()
    try:
        await buf.push(_make_event(project_id='project-a'))
        await buf.push(_make_event(project_id='project-b'))

        await buf.drain('project-a')
        await buf.drain('project-b')

        restored = await buf.restore_drained('project-a')
        assert restored == 1
        assert (await buf.get_buffer_stats('project-a'))['size'] == 1
        assert (await buf.get_buffer_stats('project-b'))['size'] == 0
    finally:
        await buf.close()


@pytest.mark.asyncio
async def test_expire_stale_bursts_transitions_old_agents(tmp_path):
    """Old burst → idle after cooldown."""
    buf = EventBuffer(
        db_path=tmp_path / 'expire.db',
        burst_window_seconds=30.0,
        burst_cooldown_seconds=150.0,
    )
    await buf.initialize()
    try:
        old_time = datetime.now(UTC) - timedelta(seconds=200)
        await buf.push(_make_event(agent_id='agent-1', timestamp=old_time))
        await buf.push(_make_event(agent_id='agent-1', timestamp=old_time + timedelta(seconds=1)))

        expired = await buf.expire_stale_bursts()
        assert expired == 1

        db = buf._require_db()
        async with db.execute(
            "SELECT state FROM burst_state WHERE agent_id = 'agent-1'"
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row['state'] == 'idle'
    finally:
        await buf.close()


@pytest.mark.asyncio
async def test_expire_stale_bursts_preserves_recent_bursts(tmp_path):
    """Recent burst should remain unchanged."""
    buf = EventBuffer(
        db_path=tmp_path / 'recent_burst.db',
        burst_window_seconds=30.0,
        burst_cooldown_seconds=150.0,
    )
    await buf.initialize()
    try:
        now = datetime.now(UTC)
        await buf.push(_make_event(agent_id='agent-1', timestamp=now))
        await buf.push(_make_event(agent_id='agent-1', timestamp=now + timedelta(seconds=1)))

        expired = await buf.expire_stale_bursts()
        assert expired == 0

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
async def test_should_trigger_expires_bursts_when_buffer_empty(tmp_path):
    """Empty buffer still expires stale bursts via should_trigger."""
    buf = EventBuffer(
        db_path=tmp_path / 'empty_expire.db',
        burst_cooldown_seconds=150.0,
    )
    await buf.initialize()
    try:
        # Create a stale burst state directly
        old_time = datetime.now(UTC) - timedelta(seconds=200)
        db = buf._require_db()
        await db.execute(
            """INSERT INTO burst_state (agent_id, state, last_write_at, burst_started_at)
               VALUES (?, 'bursting', ?, ?)""",
            ('agent-1', old_time.isoformat(), old_time.isoformat()),
        )
        await db.commit()

        # should_trigger with empty buffer
        should, _ = await buf.should_trigger('test-project')
        assert not should

        # But the burst should have been expired
        async with db.execute(
            "SELECT state FROM burst_state WHERE agent_id = 'agent-1'"
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row['state'] == 'idle'
    finally:
        await buf.close()


@pytest.mark.asyncio
async def test_cleanup_drained(tmp_path):
    """Drained events older than cutoff get cleaned up."""
    buf = EventBuffer(db_path=tmp_path / 'cleanup.db')
    await buf.initialize()
    try:
        old_time = datetime.now(UTC) - timedelta(hours=2)
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
        assert row is not None
        assert row['cnt'] == 1
    finally:
        await buf.close()


# ── Deferred writes (cycle fence) ────────────────────────────────────


@pytest.mark.asyncio
async def test_claim_deferred_writes_returns_pending_with_ids(buf):
    """claim_deferred_writes should return pending rows with ids, and second call returns []."""
    await buf.defer_write(
        project_id='test-project',
        content='write-one',
        category='observations_and_summaries',
        metadata={'k': 'v1'},
        agent_id='agent-1',
    )
    await buf.defer_write(
        project_id='test-project',
        content='write-two',
        category='entities_and_relations',
        metadata={'k': 'v2'},
        agent_id=None,
    )

    claimed = await buf.claim_deferred_writes('test-project')
    assert len(claimed) == 2
    expected_keys = {'id', 'content', 'category', 'metadata', 'agent_id'}
    for item in claimed:
        assert set(item.keys()) == expected_keys

    contents = {item['content'] for item in claimed}
    assert contents == {'write-one', 'write-two'}

    item_one = next(i for i in claimed if i['content'] == 'write-one')
    assert item_one['category'] == 'observations_and_summaries'
    assert item_one['metadata'] == {'k': 'v1'}
    assert item_one['agent_id'] == 'agent-1'
    assert item_one['id']  # non-empty

    item_two = next(i for i in claimed if i['content'] == 'write-two')
    assert item_two['agent_id'] is None

    # Second call returns [] — rows are now in-progress (claimed_at IS NOT NULL)
    second = await buf.claim_deferred_writes('test-project')
    assert second == []


@pytest.mark.asyncio
async def test_defer_write_and_pop(buf):
    """Deferred write should be retrievable via pop."""
    write_id = await buf.defer_write(
        project_id='test-project',
        content='Task X completed',
        category='observations_and_summaries',
        metadata={'source': 'targeted_reconciliation', 'task_id': '1'},
        agent_id='targeted-reconciliation',
    )
    assert write_id  # non-empty string

    writes = await buf.pop_deferred_writes('test-project')
    assert len(writes) == 1
    assert writes[0]['content'] == 'Task X completed'
    assert writes[0]['category'] == 'observations_and_summaries'
    assert writes[0]['metadata']['source'] == 'targeted_reconciliation'
    assert writes[0]['agent_id'] == 'targeted-reconciliation'


@pytest.mark.asyncio
async def test_pop_deferred_writes_empty(buf):
    """Pop on empty table returns empty list."""
    writes = await buf.pop_deferred_writes('test-project')
    assert writes == []


@pytest.mark.asyncio
async def test_pop_deferred_writes_clears(buf):
    """Second pop returns empty after first pop consumed all writes."""
    await buf.defer_write('test-project', 'content1', 'cat1', {})
    await buf.defer_write('test-project', 'content2', 'cat2', {})

    first_pop = await buf.pop_deferred_writes('test-project')
    assert len(first_pop) == 2

    second_pop = await buf.pop_deferred_writes('test-project')
    assert second_pop == []


@pytest.mark.asyncio
async def test_pop_deferred_writes_project_isolation(buf):
    """Deferred writes are scoped to project_id."""
    await buf.defer_write('project-a', 'content-a', 'cat', {})
    await buf.defer_write('project-b', 'content-b', 'cat', {})

    writes_a = await buf.pop_deferred_writes('project-a')
    assert len(writes_a) == 1
    assert writes_a[0]['content'] == 'content-a'

    writes_b = await buf.pop_deferred_writes('project-b')
    assert len(writes_b) == 1
    assert writes_b[0]['content'] == 'content-b'


@pytest.mark.asyncio
async def test_is_full_recon_active_no_lock(buf):
    """Returns False when no lock is held."""
    assert await buf.is_full_recon_active('test-project') is False


@pytest.mark.asyncio
async def test_is_full_recon_active_with_lock(buf):
    """Returns True when another instance holds the lock."""
    acquired = await buf.mark_run_active('test-project')
    assert acquired is True
    assert await buf.is_full_recon_active('test-project') is True

    await buf.mark_run_complete('test-project')
    assert await buf.is_full_recon_active('test-project') is False


@pytest.mark.asyncio
async def test_deferred_writes_ordering(buf):
    """Pop returns deferred writes in creation order."""
    await buf.defer_write('test-project', 'first', 'cat', {})
    await buf.defer_write('test-project', 'second', 'cat', {})
    await buf.defer_write('test-project', 'third', 'cat', {})

    writes = await buf.pop_deferred_writes('test-project')
    assert [w['content'] for w in writes] == ['first', 'second', 'third']


# ── Peek and targeted drain ────────────────────────────────────────


@pytest.mark.asyncio
async def test_peek_buffered_does_not_drain(buf):
    """peek_buffered returns events without changing their status."""
    await buf.push(_make_event())
    await buf.push(_make_event())

    peeked = await buf.peek_buffered('test-project', limit=10)
    assert len(peeked) == 2

    # Events should still be buffered
    stats = await buf.get_buffer_stats('test-project')
    assert stats['size'] == 2

    # A second peek returns the same events
    peeked2 = await buf.peek_buffered('test-project', limit=10)
    assert len(peeked2) == 2


@pytest.mark.asyncio
async def test_peek_buffered_respects_limit(buf):
    """peek_buffered only returns up to limit events."""
    for _ in range(5):
        await buf.push(_make_event())

    peeked = await buf.peek_buffered('test-project', limit=3)
    assert len(peeked) == 3


@pytest.mark.asyncio
async def test_peek_buffered_respects_before(buf):
    """peek_buffered with before= filters by timestamp."""
    old = datetime.now(UTC) - timedelta(hours=2)
    recent = datetime.now(UTC)
    cutoff = datetime.now(UTC) - timedelta(hours=1)

    await buf.push(_make_event(timestamp=old))
    await buf.push(_make_event(timestamp=recent))

    peeked = await buf.peek_buffered('test-project', limit=10, before=cutoff)
    assert len(peeked) == 1
    assert peeked[0].timestamp < cutoff


@pytest.mark.asyncio
async def test_peek_buffered_returns_oldest_first(buf):
    """peek_buffered returns events ordered by timestamp ascending."""
    t1 = datetime.now(UTC) - timedelta(minutes=3)
    t2 = datetime.now(UTC) - timedelta(minutes=2)
    t3 = datetime.now(UTC) - timedelta(minutes=1)

    await buf.push(_make_event(timestamp=t3))
    await buf.push(_make_event(timestamp=t1))
    await buf.push(_make_event(timestamp=t2))

    peeked = await buf.peek_buffered('test-project', limit=10)
    assert [e.timestamp for e in peeked] == [t1, t2, t3]


@pytest.mark.asyncio
async def test_drain_by_ids_marks_specific_events(buf):
    """drain_by_ids marks exactly the specified IDs as drained."""
    e1 = _make_event()
    e2 = _make_event()
    e3 = _make_event()
    await buf.push(e1)
    await buf.push(e2)
    await buf.push(e3)

    count = await buf.drain_by_ids('test-project', [e1.id, e3.id])
    assert count == 2

    # e2 should still be buffered
    stats = await buf.get_buffer_stats('test-project')
    assert stats['size'] == 1

    # Drain the remaining one
    remaining = await buf.drain('test-project')
    assert len(remaining) == 1
    assert remaining[0].id == e2.id


@pytest.mark.asyncio
async def test_drain_by_ids_empty_list(buf):
    """drain_by_ids with empty list is a no-op."""
    await buf.push(_make_event())
    count = await buf.drain_by_ids('test-project', [])
    assert count == 0
    assert (await buf.get_buffer_stats('test-project'))['size'] == 1


@pytest.mark.asyncio
async def test_drain_by_ids_ignores_wrong_project(buf):
    """drain_by_ids only affects the specified project."""
    e1 = _make_event(project_id='project-a')
    e2 = _make_event(project_id='project-b')
    await buf.push(e1)
    await buf.push(e2)

    count = await buf.drain_by_ids('project-a', [e1.id, e2.id])
    # e2 belongs to project-b, so only e1 should be drained
    assert count == 1
    assert (await buf.get_buffer_stats('project-a'))['size'] == 0
    assert (await buf.get_buffer_stats('project-b'))['size'] == 1


@pytest.mark.asyncio
async def test_peek_then_drain_by_ids_workflow(buf):
    """peek_buffered → select subset → drain_by_ids is the intended workflow."""
    for _ in range(5):
        await buf.push(_make_event())

    peeked = await buf.peek_buffered('test-project', limit=10)
    assert len(peeked) == 5

    # "Assemble" selects first 3
    selected_ids = [e.id for e in peeked[:3]]
    count = await buf.drain_by_ids('test-project', selected_ids)
    assert count == 3
    assert (await buf.get_buffer_stats('test-project'))['size'] == 2
