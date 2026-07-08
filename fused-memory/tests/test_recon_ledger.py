"""Tests for ReconLedgerStore — the SQLite control-plane ledger for recon
markers/suppressions/cycle-summaries (task 2219, PRD
plans/recon-reliability-prd.md §8.1, stream W5 foundations phase).

FOUNDATIONS-FIRST: this test module covers only the store itself (schema,
upsert/get_by_identity, list_suppressions, is_suppressed, gc, mark_addressed)
plus the server/main.py build-helper wiring test (step-19). Consumers ι/κ/λ
switch reads to the ledger under a later write-both/read-new cutover.
"""

from __future__ import annotations

import json

import pytest
import pytest_asyncio

from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord, ReconLedgerStore
from fused_memory.server import main as server_main

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


@pytest.mark.asyncio
async def test_list_suppressions_returns_only_active_suppressions_for_project(store):
    """list_suppressions(P) excludes non-suppression records and other projects."""
    suppression_p = ReconLedgerRecord(
        project_id='proj-p',
        record_kind='stage1_flag_suppression',
        payload_json='{}',
        state='active',
        created_at='2026-07-01T00:00:00+00:00',
        task_id='task-1',
        flag_type='drift_flag',
    )
    marker_p = ReconLedgerRecord(
        project_id='proj-p',
        record_kind='stage1_flag_marker',
        payload_json='{}',
        state='active',
        created_at='2026-07-01T00:00:00+00:00',
        task_id='task-2',
        flag_type='drift_flag',
    )
    suppression_q = ReconLedgerRecord(
        project_id='proj-q',
        record_kind='stage1_flag_suppression',
        payload_json='{}',
        state='active',
        created_at='2026-07-01T00:00:00+00:00',
        task_id='task-3',
        flag_type='drift_flag',
    )
    await store.upsert(suppression_p)
    await store.upsert(marker_p)
    await store.upsert(suppression_q)

    results = await store.list_suppressions('proj-p')

    assert results == [suppression_p]


@pytest.mark.asyncio
async def test_is_suppressed_exact_and_blanket_union(store):
    """is_suppressed treats flag_type='' as a blanket suppression covering every flag_type."""
    exact = ReconLedgerRecord(
        project_id='proj-p',
        record_kind='stage1_flag_suppression',
        payload_json='{}',
        state='active',
        created_at='2026-07-01T00:00:00+00:00',
        task_id='T1',
        flag_type='F1',
    )
    blanket = ReconLedgerRecord(
        project_id='proj-p',
        record_kind='stage1_flag_suppression',
        payload_json='{}',
        state='active',
        created_at='2026-07-01T00:00:00+00:00',
        task_id='T2',
        flag_type='',
    )
    await store.upsert(exact)
    await store.upsert(blanket)

    assert await store.is_suppressed('proj-p', 'T1', 'F1') is True
    assert await store.is_suppressed('proj-p', 'T2', 'anything') is True
    assert await store.is_suppressed('proj-p', 'T1', 'F2') is False
    assert await store.is_suppressed('other', 'T1', 'F1') is False


@pytest.mark.asyncio
async def test_gc_deletes_expired_and_terminal_referenced_rows(store):
    """gc() deletes (a) expired rows and (b) marker-kind rows whose task_id is
    terminal; it keeps (c) live rows, (d) never-expire rows, and rows in other
    projects, returning the count of deleted rows (user-observable signal #3)."""
    expired_marker = ReconLedgerRecord(
        project_id='proj-p',
        record_kind='stage1_flag_marker',
        payload_json='{}',
        state='active',
        created_at='2026-06-01T00:00:00+00:00',
        task_id='task-expired',
        flag_type='drift_flag',
        expires_at='2026-06-15T00:00:00+00:00',
    )
    terminal_marker = ReconLedgerRecord(
        project_id='proj-p',
        record_kind='stage2_persistence_marker',
        payload_json='{}',
        state='active',
        created_at='2026-06-01T00:00:00+00:00',
        task_id='task-terminal',
        flag_type='persistence_flag',
        expires_at='2026-12-01T00:00:00+00:00',
    )
    live_marker = ReconLedgerRecord(
        project_id='proj-p',
        record_kind='stage1_flag_marker',
        payload_json='{}',
        state='active',
        created_at='2026-06-01T00:00:00+00:00',
        task_id='task-live',
        flag_type='drift_flag',
        expires_at='2026-12-01T00:00:00+00:00',
    )
    never_expire = ReconLedgerRecord(
        project_id='proj-p',
        record_kind='cycle_summary',
        payload_json='{}',
        state='active',
        created_at='2026-06-01T00:00:00+00:00',
        task_id='task-never',
    )
    other_project_row = ReconLedgerRecord(
        project_id='proj-q',
        record_kind='stage1_flag_marker',
        payload_json='{}',
        state='active',
        created_at='2026-06-01T00:00:00+00:00',
        task_id='task-expired',
        flag_type='drift_flag',
        expires_at='2026-06-15T00:00:00+00:00',
    )
    await store.upsert(expired_marker)
    await store.upsert(terminal_marker)
    await store.upsert(live_marker)
    await store.upsert(never_expire)
    await store.upsert(other_project_row)

    deleted_count = await store.gc(
        'proj-p', now='2026-07-01T00:00:00+00:00', terminal_task_ids=['task-terminal']
    )

    assert deleted_count == 2
    assert (
        await store.get_by_identity(
            'proj-p', 'stage1_flag_marker', task_id='task-expired', flag_type='drift_flag'
        )
        is None
    )
    assert (
        await store.get_by_identity(
            'proj-p', 'stage2_persistence_marker', task_id='task-terminal', flag_type='persistence_flag'
        )
        is None
    )
    assert (
        await store.get_by_identity(
            'proj-p', 'stage1_flag_marker', task_id='task-live', flag_type='drift_flag'
        )
        is not None
    )
    assert await store.get_by_identity('proj-p', 'cycle_summary', task_id='task-never') is not None
    assert (
        await store.get_by_identity(
            'proj-q', 'stage1_flag_marker', task_id='task-expired', flag_type='drift_flag'
        )
        is not None
    )


@pytest.mark.asyncio
async def test_gc_with_empty_terminal_task_ids_deletes_only_expired(store):
    """gc() with an empty terminal_task_ids list must not raise (SQLite
    rejects an empty `IN ()` list) and must delete only expired rows,
    leaving live and never-expire rows untouched."""
    expired = ReconLedgerRecord(
        project_id='proj-p',
        record_kind='stage1_flag_marker',
        payload_json='{}',
        state='active',
        created_at='2026-06-01T00:00:00+00:00',
        task_id='task-expired',
        flag_type='drift_flag',
        expires_at='2026-06-15T00:00:00+00:00',
    )
    live = ReconLedgerRecord(
        project_id='proj-p',
        record_kind='stage1_flag_marker',
        payload_json='{}',
        state='active',
        created_at='2026-06-01T00:00:00+00:00',
        task_id='task-live',
        flag_type='drift_flag',
        expires_at='2026-12-01T00:00:00+00:00',
    )
    never_expire = ReconLedgerRecord(
        project_id='proj-p',
        record_kind='cycle_summary',
        payload_json='{}',
        state='active',
        created_at='2026-06-01T00:00:00+00:00',
        task_id='task-never',
    )
    await store.upsert(expired)
    await store.upsert(live)
    await store.upsert(never_expire)

    deleted_count = await store.gc('proj-p', now='2026-07-01T00:00:00+00:00', terminal_task_ids=[])

    assert deleted_count == 1
    assert (
        await store.get_by_identity(
            'proj-p', 'stage1_flag_marker', task_id='task-expired', flag_type='drift_flag'
        )
        is None
    )
    assert (
        await store.get_by_identity(
            'proj-p', 'stage1_flag_marker', task_id='task-live', flag_type='drift_flag'
        )
        is not None
    )
    assert await store.get_by_identity('proj-p', 'cycle_summary', task_id='task-never') is not None


@pytest.mark.asyncio
async def test_mark_addressed_sets_state_and_stamps_payload(store):
    """mark_addressed() flips state to 'addressed' and stamps addressed_by /
    addressed_run_id into payload_json (read-modify-write)."""
    identity = dict(
        project_id='proj-a',
        record_kind='stage1_flag_marker',
        task_id='task-1',
        flag_type='drift_flag',
        run_id='run-1',
    )
    record = ReconLedgerRecord(
        **identity,
        payload_json='{"flag": "drift"}',
        state='active',
        created_at='2026-07-01T00:00:00+00:00',
        expires_at='2026-07-15T00:00:00+00:00',
    )
    await store.upsert(record)

    await store.mark_addressed(
        **identity,
        addressed_by='recon-stage-task_knowledge_sync',
        addressed_run_id='run-9',
    )

    fetched = await store.get_by_identity(**identity)
    assert fetched is not None
    assert fetched.state == 'addressed'
    payload = json.loads(fetched.payload_json)
    assert payload['flag'] == 'drift'
    assert payload['addressed_by'] == 'recon-stage-task_knowledge_sync'
    assert payload['addressed_run_id'] == 'run-9'


@pytest.mark.asyncio
async def test_mark_addressed_on_missing_identity_is_noop(store):
    """mark_addressed() against an identity with no row must not raise and
    must not create a row."""
    await store.mark_addressed(
        'proj-missing',
        'stage1_flag_marker',
        task_id='task-none',
        flag_type='drift_flag',
        run_id='run-none',
        addressed_by='recon-stage-task_knowledge_sync',
        addressed_run_id='run-9',
    )

    fetched = await store.get_by_identity(
        'proj-missing', 'stage1_flag_marker', task_id='task-none', flag_type='drift_flag', run_id='run-none'
    )
    assert fetched is None


@pytest.mark.asyncio
async def test_build_recon_ledger_store_initializes_schema(tmp_path):
    """server.main._build_recon_ledger_store() builds and initializes a
    ReconLedgerStore against data_dir/'reconciliation.db' (step-19)."""
    built_store = await server_main._build_recon_ledger_store(tmp_path)
    try:
        assert isinstance(built_store, ReconLedgerStore)
        db_file = tmp_path / 'reconciliation.db'
        assert db_file.exists()

        db = built_store._db
        assert db is not None
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='recon_ledger'"
        )
        row = await cursor.fetchone()
        assert row is not None
    finally:
        await built_store.close()
