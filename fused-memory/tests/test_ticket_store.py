"""Tests for the TicketStore SQLite persistence layer (two-phase add_task)."""

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from fused_memory.middleware.ticket_store import TicketStore, _new_ticket_id


@pytest_asyncio.fixture
async def store(tmp_path):
    s = TicketStore(tmp_path / 'tickets.db')
    await s.initialize()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_initialize_creates_schema_and_is_idempotent(tmp_path):
    """initialize() creates the tickets table; calling it twice does not raise."""
    store = TicketStore(tmp_path / 'tickets.db')
    await store.initialize()
    # Second call must be idempotent (CREATE IF NOT EXISTS)
    await store.initialize()

    # Verify the table exists with the expected columns
    db = store._db
    cursor = await db.execute("PRAGMA table_info(tickets)")
    rows = await cursor.fetchall()
    col_names = {row[1] for row in rows}

    expected_columns = {
        'ticket_id',
        'project_id',
        'candidate_json',
        'status',
        'task_id',
        'reason',
        'result_json',
        'created_at',
        'resolved_at',
        'expires_at',
    }
    assert expected_columns == col_names, (
        f"Missing columns: {expected_columns - col_names}; "
        f"Extra columns: {col_names - expected_columns}"
    )
    await store.close()


@pytest.mark.asyncio
async def test_new_ticket_id_has_tkt_prefix_and_sorts_by_time():
    """_new_ticket_id() returns tkt_-prefixed ids that are lexicographically time-ordered."""
    id1 = _new_ticket_id()
    await asyncio.sleep(0.001)  # ensure nanosecond timestamp advances
    id2 = _new_ticket_id()

    # Both must start with tkt_
    assert id1.startswith('tkt_'), f"id1 missing tkt_ prefix: {id1!r}"
    assert id2.startswith('tkt_'), f"id2 missing tkt_ prefix: {id2!r}"

    # Exactly the documented length: 4 (prefix) + 26 (crockford base32 of 16 bytes)
    assert len(id1) == 30, f"Expected length 30, got {len(id1)}"
    assert len(id2) == 30, f"Expected length 30, got {len(id2)}"

    # Later id sorts after the earlier one (monotonic time-ordered)
    assert id1 < id2, f"Expected id1 < id2 but got {id1!r} >= {id2!r}"


@pytest.mark.asyncio
async def test_submit_persists_pending_ticket_and_returns_id(store):
    """submit() inserts a pending row and returns a tkt_-prefixed id."""
    candidate = json.dumps({'title': 'Test Task', 'description': 'Do it'})
    ticket_id = await store.submit(project_id='p', candidate_json=candidate, ttl_seconds=600)

    assert ticket_id.startswith('tkt_')

    # Verify the persisted row
    row = await store.get(ticket_id)
    assert row is not None
    assert row['status'] == 'pending'
    assert row['project_id'] == 'p'
    assert row['candidate_json'] == candidate

    # created_at and expires_at must be set
    created_at = datetime.fromisoformat(row['created_at'])
    expires_at = datetime.fromisoformat(row['expires_at'])
    assert created_at.tzinfo is not None  # timezone-aware
    assert expires_at == created_at + timedelta(seconds=600)

    # Unresolved columns must be NULL
    assert row['task_id'] is None
    assert row['reason'] is None
    assert row['resolved_at'] is None
    assert row['result_json'] is None


@pytest.mark.asyncio
async def test_get_returns_row_or_none(store):
    """get() returns a dict for known tickets and None for unknown ids."""
    candidate = json.dumps({'title': 'T'})
    ticket_id = await store.submit(project_id='proj', candidate_json=candidate)

    row = await store.get(ticket_id)
    assert row is not None
    assert isinstance(row, dict)
    expected_keys = {
        'ticket_id', 'project_id', 'candidate_json', 'status',
        'task_id', 'reason', 'result_json', 'created_at', 'resolved_at', 'expires_at',
    }
    assert expected_keys == set(row.keys())
    assert row['ticket_id'] == ticket_id

    missing = await store.get('tkt_nonexistent_000000000000')
    assert missing is None
