"""Tests for the TicketStore SQLite persistence layer (two-phase add_task)."""

import pytest
import pytest_asyncio

from fused_memory.middleware.ticket_store import TicketStore


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
