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
