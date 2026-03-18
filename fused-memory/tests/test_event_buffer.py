"""Tests for reconciliation event buffer."""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest

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
) -> ReconciliationEvent:
    return ReconciliationEvent(
        id=str(uuid.uuid4()),
        type=event_type,
        source=EventSource.agent,
        project_id=project_id,
        timestamp=timestamp or datetime.now(timezone.utc),
        payload={'test': True},
    )


@pytest.mark.asyncio
async def test_push_and_stats():
    buf = EventBuffer(buffer_size_threshold=5, max_staleness_seconds=300)
    event = _make_event()
    await buf.push(event)
    stats = buf.get_buffer_stats('test-project')
    assert stats['size'] == 1
    assert stats['oldest_event_age_seconds'] is not None


@pytest.mark.asyncio
async def test_should_trigger_buffer_size():
    buf = EventBuffer(buffer_size_threshold=3, max_staleness_seconds=3600)
    for _ in range(3):
        await buf.push(_make_event())
    should, reason = await buf.should_trigger('test-project')
    assert should
    assert 'buffer_size' in reason


@pytest.mark.asyncio
async def test_should_trigger_staleness():
    buf = EventBuffer(buffer_size_threshold=100, max_staleness_seconds=1)
    old_time = datetime.now(timezone.utc) - timedelta(seconds=5)
    await buf.push(_make_event(timestamp=old_time))
    should, reason = await buf.should_trigger('test-project')
    assert should
    assert 'max_staleness' in reason


@pytest.mark.asyncio
async def test_no_trigger_when_empty():
    buf = EventBuffer()
    should, reason = await buf.should_trigger('test-project')
    assert not should
    assert reason == ''


@pytest.mark.asyncio
async def test_no_trigger_when_active_run():
    buf = EventBuffer(buffer_size_threshold=1)
    await buf.push(_make_event())
    await buf.mark_run_active('test-project')
    should, reason = await buf.should_trigger('test-project')
    assert not should


@pytest.mark.asyncio
async def test_drain_returns_events_and_clears():
    buf = EventBuffer()
    await buf.push(_make_event())
    await buf.push(_make_event())

    events = await buf.drain('test-project')
    assert len(events) == 2

    # Buffer is now empty
    events2 = await buf.drain('test-project')
    assert len(events2) == 0


@pytest.mark.asyncio
async def test_mark_run_active_prevents_double_run():
    buf = EventBuffer()
    assert await buf.mark_run_active('test-project') is True
    assert await buf.mark_run_active('test-project') is False


@pytest.mark.asyncio
async def test_mark_run_complete_allows_new_run():
    buf = EventBuffer()
    await buf.mark_run_active('test-project')
    await buf.mark_run_complete('test-project')
    assert await buf.mark_run_active('test-project') is True


@pytest.mark.asyncio
async def test_separate_project_buffers():
    buf = EventBuffer(buffer_size_threshold=2)
    await buf.push(_make_event(project_id='project-a'))
    await buf.push(_make_event(project_id='project-b'))
    await buf.push(_make_event(project_id='project-a'))

    should_a, _ = await buf.should_trigger('project-a')
    should_b, _ = await buf.should_trigger('project-b')
    assert should_a is True
    assert should_b is False


@pytest.mark.asyncio
async def test_get_active_projects():
    buf = EventBuffer()
    await buf.push(_make_event(project_id='p1'))
    await buf.push(_make_event(project_id='p2'))
    projects = buf.get_active_projects()
    assert set(projects) == {'p1', 'p2'}


@pytest.mark.asyncio
async def test_empty_project_stats():
    buf = EventBuffer()
    stats = buf.get_buffer_stats('nonexistent')
    assert stats == {'size': 0, 'oldest_event_age_seconds': None}
