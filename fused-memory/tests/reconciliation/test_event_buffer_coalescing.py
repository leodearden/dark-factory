"""Push-time coalescing through a real EventBuffer (task 4278, recon lever 3).

tests/test_event_buffer.py covers the buffer in general; this file covers only
what folding adds: the schema and its migration, which pushes fold, how a fold
keeps the drain lifecycle's invariants, and how every pushed event stays
counted.  The fold rule itself is tested in test_coalescing.py.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite
import pytest
import pytest_asyncio

from fused_memory.models.reconciliation import (
    EventSource,
    EventType,
    ReconciliationEvent,
)
from fused_memory.reconciliation.coalescing import (
    COALESCED_COUNT_KEY,
    COALESCED_STATUS_PATH_KEY,
)
from fused_memory.reconciliation.event_buffer import EventBuffer

_PROJECT = 'dark_factory'
_T0 = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)


def _at(minutes: float) -> datetime:
    return _T0 + timedelta(minutes=minutes)


def _task_event(
    project_id: str,
    event_type: EventType,
    task_id: str,
    ts: datetime,
    **payload,
) -> ReconciliationEvent:
    """A task event shaped like task_interceptor._make_event's.

    A task_modified event defaults to an update_task payload.
    """
    body: dict = {'task_id': task_id, '_project_root': '/project'}
    if event_type is EventType.task_modified:
        body['operation'] = 'update_task'
    body.update(payload)
    return ReconciliationEvent(
        id=str(uuid.uuid4()),
        type=event_type,
        source=EventSource.agent,
        project_id=project_id,
        timestamp=ts,
        payload=body,
        agent_id=None,
    )


def _raw(db_path: Path, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    """Read the buffer's database file directly, as a second connection would."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


@pytest.fixture
def db_path(tmp_path) -> Path:
    return tmp_path / 'recon.db'


@pytest_asyncio.fixture
async def buf(db_path):
    b = EventBuffer(db_path=db_path, buffer_size_threshold=1000)
    await b.initialize()
    yield b
    await b.close()


# ── schema and migration ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fresh_database_has_the_coalescing_schema(buf, db_path):
    columns = {row['name'] for row in _raw(db_path, 'PRAGMA table_info(event_buffer)')}
    assert 'coalesce_key' in columns

    indexes = {row['name'] for row in _raw(
        db_path, "SELECT name FROM sqlite_master WHERE type = 'index'")}
    assert 'idx_eb_coalesce' in indexes

    folds = _raw(db_path, 'PRAGMA table_info(event_coalesced_hourly)')
    assert [row['name'] for row in folds] == [
        'project_id', 'hour_bucket', 'event_type', 'coalesced_count']
    primary_key = sorted((row['pk'], row['name']) for row in folds if row['pk'])
    assert [name for _, name in primary_key] == ['project_id', 'hour_bucket', 'event_type']


_PRE_COALESCING_EVENT_BUFFER = """
CREATE TABLE event_buffer (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_source TEXT NOT NULL,
    agent_id TEXT,
    timestamp TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'buffered',
    drained_by_run_id TEXT
);
"""


@pytest.mark.asyncio
async def test_pre_coalescing_database_migrates_and_never_folds_legacy_rows(db_path):
    async with aiosqlite.connect(str(db_path)) as db:
        await db.executescript(_PRE_COALESCING_EVENT_BUFFER)
        await db.execute(
            """INSERT INTO event_buffer
               (id, project_id, event_type, event_source, agent_id, timestamp, payload)
               VALUES ('legacy-1', ?, 'task_modified', 'agent', NULL, ?, ?)""",
            (_PROJECT, _at(0).isoformat(),
             '{"task_id": "5", "operation": "update_task", "_project_root": "/project"}'),
        )
        await db.commit()

    b = EventBuffer(db_path=db_path, buffer_size_threshold=1000)
    await b.initialize()
    try:
        columns = {row['name'] for row in _raw(db_path, 'PRAGMA table_info(event_buffer)')}
        assert 'coalesce_key' in columns
        indexes = {row['name'] for row in _raw(
            db_path, "SELECT name FROM sqlite_master WHERE type = 'index'")}
        assert 'idx_eb_coalesce' in indexes
        legacy = _raw(db_path, "SELECT coalesce_key FROM event_buffer WHERE id = 'legacy-1'")
        assert legacy[0]['coalesce_key'] is None

        await b.push(_task_event(_PROJECT, EventType.task_modified, '5', _at(1)))
        assert await b.count_buffered(_PROJECT) == 2
    finally:
        await b.close()


@pytest.mark.asyncio
async def test_initialising_the_same_file_twice_is_harmless(db_path):
    first = EventBuffer(db_path=db_path)
    second = EventBuffer(db_path=db_path)
    await first.initialize()
    try:
        await second.initialize()
        await second.close()
    finally:
        await first.close()


# ── which pushes fold ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_repeat_task_updates_fold_into_the_first_row(buf):
    events = [_task_event(_PROJECT, EventType.task_modified, '644', _at(m)) for m in (0, 1, 2)]
    for event in events:
        await buf.push(event)

    assert await buf.count_buffered(_PROJECT) == 1
    [row] = await buf.peek_buffered(_PROJECT, limit=10)
    assert row.id == events[0].id
    assert row.timestamp == events[0].timestamp
    assert row.payload['task_id'] == '644'
    assert row.payload[COALESCED_COUNT_KEY] == 3


@pytest.mark.asyncio
async def test_updates_to_different_tasks_do_not_fold(buf):
    for task_id in ('1', '2', '3'):
        await buf.push(_task_event(_PROJECT, EventType.task_modified, task_id, _at(0)))
    assert await buf.count_buffered(_PROJECT) == 3


@pytest.mark.asyncio
async def test_the_same_task_id_never_folds_across_projects(buf):
    await buf.push(_task_event('dark_factory', EventType.task_modified, '7', _at(0)))
    await buf.push(_task_event('reify', EventType.task_modified, '7', _at(1)))
    assert await buf.count_buffered('dark_factory') == 1
    assert await buf.count_buffered('reify') == 1


@pytest.mark.asyncio
async def test_dependency_operations_never_fold(buf):
    for depends_on in ('10', '11', '12'):
        await buf.push(_task_event(
            _PROJECT, EventType.task_modified, '644', _at(0),
            operation='add_dependency', depends_on=depends_on,
        ))
    rows = await buf.peek_buffered(_PROJECT, limit=10)
    assert sorted(row.payload['depends_on'] for row in rows) == ['10', '11', '12']


@pytest.mark.asyncio
async def test_an_update_and_a_status_change_for_one_task_stay_separate(buf):
    await buf.push(_task_event(_PROJECT, EventType.task_modified, '644', _at(0)))
    await buf.push(_task_event(
        _PROJECT, EventType.task_status_changed, '644', _at(1),
        old_status='pending', new_status='in-progress',
    ))
    assert await buf.count_buffered(_PROJECT) == 2


@pytest.mark.asyncio
async def test_switching_coalescing_off_keeps_every_row_and_still_stores_the_key(db_path):
    b = EventBuffer(db_path=db_path, coalesce_task_events=False)
    await b.initialize()
    try:
        for minute in (0, 1, 2):
            await b.push(_task_event(_PROJECT, EventType.task_modified, '644', _at(minute)))
        assert await b.count_buffered(_PROJECT) == 3
    finally:
        await b.close()
    keys = [row['coalesce_key'] for row in _raw(db_path, 'SELECT coalesce_key FROM event_buffer')]
    assert keys == ['task_modified:644:update_task'] * 3


@pytest.mark.asyncio
@pytest.mark.parametrize('event_type', [EventType.episode_added, EventType.memory_added])
async def test_memory_events_carrying_a_task_id_never_fold(buf, event_type):
    for minute in (0, 1, 2):
        await buf.push(_task_event(_PROJECT, event_type, '644', _at(minute)))
    assert await buf.count_buffered(_PROJECT) == 3


@pytest.mark.asyncio
async def test_repushing_a_buffered_event_is_still_a_no_op(buf):
    event = _task_event(_PROJECT, EventType.task_modified, '644', _at(0))
    await buf.push(event)
    await buf.push(event)

    assert await buf.count_buffered(_PROJECT) == 1
    [row] = await buf.peek_buffered(_PROJECT, limit=10)
    assert row.payload == event.payload


# ── status changes stay readable ───────────────────────────────────────


def _transition(old: str, new: str, ts: datetime, **extra) -> ReconciliationEvent:
    return _task_event(
        _PROJECT, EventType.task_status_changed, '644', ts,
        old_status=old, new_status=new, **extra,
    )


async def _only_row(buf: EventBuffer) -> ReconciliationEvent:
    assert await buf.count_buffered(_PROJECT) == 1
    [row] = await buf.peek_buffered(_PROJECT, limit=10)
    return row


@pytest.mark.asyncio
async def test_a_status_chain_folds_to_its_start_and_end(buf):
    await buf.push(_transition('pending', 'in-progress', _at(0)))
    await buf.push(_transition('in-progress', 'done', _at(1)))

    payload = (await _only_row(buf)).payload
    assert payload['old_status'] == 'pending'
    assert payload['new_status'] == 'done'
    assert payload[COALESCED_STATUS_PATH_KEY] == ['pending', 'in-progress', 'done']
    assert payload[COALESCED_COUNT_KEY] == 2

    await buf.push(_transition('done', 'review', _at(2)))
    payload = (await _only_row(buf)).payload
    assert payload['new_status'] == 'review'
    assert payload[COALESCED_STATUS_PATH_KEY] == ['pending', 'in-progress', 'done', 'review']


@pytest.mark.asyncio
async def test_a_drained_fold_carries_only_the_newest_transitions_keys(buf):
    await buf.push(_transition('pending', 'done', _at(0), done_provenance={'kind': 'merged'}))
    await buf.push(_transition(
        'done', 'pending', _at(1), reopen_reason='regressed', reopen_from='done'))
    [drained] = await buf.drain(_PROJECT, run_id='r1')
    assert drained.payload['reopen_reason'] == 'regressed'
    assert drained.payload['reopen_from'] == 'done'
    assert 'done_provenance' not in drained.payload


@pytest.mark.asyncio
async def test_a_drained_fold_does_not_carry_a_stale_deferral_entry(buf):
    await buf.push(_transition('pending', 'deferred', _at(0), deferral={'until': '2026-11-01'}))
    await buf.push(_transition('deferred', 'pending', _at(1), deferral_cleared={'by': 'gate'}))
    [drained] = await buf.drain(_PROJECT, run_id='r1')
    assert drained.payload['deferral_cleared'] == {'by': 'gate'}
    assert 'deferral' not in drained.payload


@pytest.mark.asyncio
async def test_an_older_transition_arriving_late_does_not_revert_the_status(buf):
    newer = _transition('in-progress', 'done', _at(2))
    older = _transition('pending', 'in-progress', _at(1))
    await buf.push(newer)
    await buf.push(older)

    row = await _only_row(buf)
    assert row.payload['old_status'] == 'pending'
    assert row.payload['new_status'] == 'done'
    assert row.timestamp == older.timestamp


@pytest.mark.asyncio
async def test_redelivering_a_folded_away_event_changes_nothing(buf, db_path):
    first = _transition('pending', 'in-progress', _at(0))
    second = _transition('in-progress', 'done', _at(1))
    await buf.push(first)
    await buf.push(second)
    await buf.push(second)

    row = await _only_row(buf)
    assert row.payload[COALESCED_COUNT_KEY] == 2
    assert len(_raw(db_path, 'SELECT id FROM event_buffer')) == 1


# ── the drain lifecycle keeps its invariants ───────────────────────────


def _update(task_id: str, ts: datetime) -> ReconciliationEvent:
    return _task_event(_PROJECT, EventType.task_modified, task_id, ts)


def _episode(ts: datetime) -> ReconciliationEvent:
    return _task_event(_PROJECT, EventType.episode_added, '', ts)


@pytest.mark.asyncio
async def test_a_folded_row_drains_in_its_earliest_constituents_slot(buf):
    first_update = _update('A', _at(1))
    for event in (_episode(_at(0)), first_update, _episode(_at(2)), _update('A', _at(3))):
        await buf.push(event)

    drained = await buf.drain(_PROJECT, run_id='r1')
    assert [event.timestamp for event in drained] == [_at(0), _at(1), _at(2)]
    assert drained[1].id == first_update.id
    assert drained[1].payload[COALESCED_COUNT_KEY] == 2


@pytest.mark.asyncio
async def test_folds_cannot_walk_the_oldest_event_past_the_staleness_trigger(db_path):
    b = EventBuffer(db_path=db_path, buffer_size_threshold=1000, max_staleness_seconds=60)
    await b.initialize()
    try:
        now = datetime.now(UTC)
        first = _update('644', now - timedelta(seconds=600))
        await b.push(first)
        for seconds_ago in (5, 1):
            await b.push(_update('644', now - timedelta(seconds=seconds_ago)))

        [row] = await b.peek_buffered(_PROJECT, limit=10)
        assert row.timestamp == first.timestamp
        stats = await b.get_buffer_stats(_PROJECT)
        assert stats['oldest_event_age_seconds'] >= 600
        triggered, reason = await b.should_trigger(_PROJECT)
        assert triggered
        assert reason.startswith('max_staleness:')
    finally:
        await b.close()


@pytest.mark.asyncio
async def test_every_peeked_id_drains_by_id_including_a_folded_row(buf):
    await buf.push(_episode(_at(0)))
    await buf.push(_update('A', _at(1)))
    await buf.push(_update('A', _at(2)))

    peeked = await buf.peek_buffered(_PROJECT, limit=10)
    assert any(COALESCED_COUNT_KEY in event.payload for event in peeked)
    drained = await buf.drain_by_ids(_PROJECT, [event.id for event in peeked], run_id='r2')
    assert drained == len(peeked) == 2


@pytest.mark.asyncio
async def test_a_fold_between_peek_and_drain_is_not_lost(buf):
    await buf.push(_update('A', _at(0)))
    await buf.push(_update('A', _at(1)))
    [peeked] = await buf.peek_buffered(_PROJECT, limit=10)

    await buf.push(_update('A', _at(2)))
    assert await buf.drain_by_ids(_PROJECT, [peeked.id], run_id='r2') == 1
    [drained] = await buf.get_drained_events(_PROJECT, 'r2')
    assert drained.payload[COALESCED_COUNT_KEY] == 3


@pytest.mark.asyncio
async def test_a_push_after_the_drain_starts_a_fresh_row(buf):
    await buf.push(_update('A', _at(0)))
    await buf.push(_update('A', _at(1)))
    [peeked] = await buf.peek_buffered(_PROJECT, limit=10)
    await buf.drain_by_ids(_PROJECT, [peeked.id], run_id='r2')

    await buf.push(_update('A', _at(2)))
    assert await buf.count_buffered(_PROJECT) == 1
    [drained] = await buf.get_drained_events(_PROJECT, 'r2')
    assert drained.payload == peeked.payload


@pytest.mark.asyncio
async def test_a_restored_row_keeps_its_merged_payload_and_folds_again(buf):
    await buf.push(_update('A', _at(0)))
    await buf.push(_update('A', _at(1)))
    [peeked] = await buf.peek_buffered(_PROJECT, limit=10)
    await buf.drain_by_ids(_PROJECT, [peeked.id], run_id='r2')

    assert await buf.restore_drained(_PROJECT, run_id='r2') == 1
    [restored] = await buf.peek_buffered(_PROJECT, limit=10)
    assert restored.payload == peeked.payload

    await buf.push(_update('A', _at(2)))
    [refolded] = await buf.peek_buffered(_PROJECT, limit=10)
    assert refolded.id == peeked.id
    assert refolded.payload[COALESCED_COUNT_KEY] == 3


@pytest.mark.asyncio
async def test_drain_oldest_chunk_drains_a_folded_row_once(buf):
    await buf.push(_update('A', _at(0)))
    await buf.push(_update('A', _at(1)))

    [chunk] = await buf.drain_oldest_chunk(_PROJECT, limit=10)
    assert chunk.payload[COALESCED_COUNT_KEY] == 2
    assert await buf.count_buffered(_PROJECT) == 0
    assert await buf.drain_oldest_chunk(_PROJECT, limit=10) == []
