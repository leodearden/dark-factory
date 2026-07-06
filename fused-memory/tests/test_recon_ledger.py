"""Tests for ReconLedgerStore — the SQLite control-plane ledger for recon
markers/suppressions/cycle-summaries (task 2219, PRD
plans/recon-reliability-prd.md §8.1, stream W5 foundations phase).

FOUNDATIONS-FIRST: this test module covers only the store itself (schema,
upsert/get_by_identity, list_suppressions, is_suppressed, gc, mark_addressed)
plus the server/main.py build-helper wiring test (step-19). Consumers ι/κ/λ
switch reads to the ledger under a later write-both/read-new cutover.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord, ReconLedgerStore

EXPECTED_COLUMNS = {
    'project_id',
    'record_kind',
    'task_id',
    'flag_type',
    'run_id',
    'payload_json',
    'state',
    'created_at',
    'expires_at',
}

EXPECTED_INDEXES = {
    'ix_recon_ledger_project_kind_state',
    'ix_recon_ledger_project_expires',
}


@pytest_asyncio.fixture
async def store(tmp_path):
    s = ReconLedgerStore(tmp_path / 'reconciliation.db')
    await s.initialize()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_initialize_creates_schema_with_expected_columns_and_indexes(tmp_path):
    """initialize() creates the recon_ledger table + indexes; re-init after close is safe.

    Exercises the init→close→init idempotency pattern (CREATE TABLE/INDEX IF
    NOT EXISTS), mirroring test_ticket_store.py's equivalent schema test.
    """
    s = ReconLedgerStore(tmp_path / 'reconciliation.db')
    await s.initialize()
    # Idempotent re-init: close first (safe reconnect pattern), then re-init.
    await s.close()
    await s.initialize()
    try:
        db = s._db
        assert db is not None

        cursor = await db.execute('PRAGMA table_info(recon_ledger)')
        rows = await cursor.fetchall()
        col_names = {row[1] for row in rows}
        assert col_names == EXPECTED_COLUMNS, (
            f'Missing columns: {EXPECTED_COLUMNS - col_names}; '
            f'Extra columns: {col_names - EXPECTED_COLUMNS}'
        )

        index_cursor = await db.execute('PRAGMA index_list(recon_ledger)')
        index_rows = await index_cursor.fetchall()
        index_names = {row[1] for row in index_rows}
        assert EXPECTED_INDEXES <= index_names, (
            f'Missing indexes: {EXPECTED_INDEXES - index_names}; found: {index_names}'
        )
    finally:
        await s.close()


@pytest.mark.asyncio
async def test_upsert_then_get_by_identity_round_trips(store):
    """upsert() persists a record; get_by_identity() reads it back unchanged."""
    record = ReconLedgerRecord(
        project_id='proj-a',
        record_kind='stage1_flag_marker',
        payload_json='{"flag": "drift"}',
        state='active',
        created_at='2026-07-01T00:00:00+00:00',
        task_id='task-1',
        flag_type='drift_flag',
        run_id='run-1',
        expires_at='2026-07-15T00:00:00+00:00',
    )
    await store.upsert(record)

    fetched = await store.get_by_identity(
        'proj-a',
        'stage1_flag_marker',
        task_id='task-1',
        flag_type='drift_flag',
        run_id='run-1',
    )
    assert fetched == record


@pytest.mark.asyncio
async def test_get_by_identity_none_expires_at_round_trips_to_none(store):
    """A record created with expires_at=None reads back as None, not the string 'None'."""
    record = ReconLedgerRecord(
        project_id='proj-a',
        record_kind='cycle_summary',
        payload_json='{}',
        state='active',
        created_at='2026-07-01T00:00:00+00:00',
    )
    await store.upsert(record)

    fetched = await store.get_by_identity('proj-a', 'cycle_summary')
    assert fetched is not None
    assert fetched.expires_at is None


@pytest.mark.asyncio
async def test_get_by_identity_unknown_identity_returns_none(store):
    """An identity with no matching row returns None rather than raising."""
    fetched = await store.get_by_identity('proj-unknown', 'stage1_flag_marker')
    assert fetched is None


@pytest.mark.asyncio
async def test_double_upsert_same_identity_keeps_one_row_last_write_wins(store):
    """upsert() on the same identity twice must leave exactly one row, with the
    last write's payload_json/state/created_at/expires_at all winning."""
    identity = dict(
        project_id='proj-a',
        record_kind='stage1_flag_marker',
        task_id='task-1',
        flag_type='drift_flag',
        run_id='run-1',
    )
    first = ReconLedgerRecord(
        **identity,
        payload_json='{"v": 1}',
        state='active',
        created_at='2026-07-01T00:00:00+00:00',
        expires_at='2026-07-15T00:00:00+00:00',
    )
    second = ReconLedgerRecord(
        **identity,
        payload_json='{"v": 2}',
        state='addressed',
        created_at='2026-07-02T00:00:00+00:00',
        expires_at='2026-07-20T00:00:00+00:00',
    )
    await store.upsert(first)
    await store.upsert(second)

    db = store._db
    cursor = await db.execute('SELECT COUNT(*) FROM recon_ledger')
    row = await cursor.fetchone()
    assert row[0] == 1

    fetched = await store.get_by_identity(**identity)
    assert fetched == second
