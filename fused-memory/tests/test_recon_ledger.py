"""Tests for ReconLedgerStore — the SQLite control-plane ledger for recon
markers/suppressions/cycle-summaries (task 2219, PRD
plans/recon-reliability-prd.md §8.1, stream W5 foundations phase).

FOUNDATIONS-FIRST: this test module covers only the store itself (schema,
upsert/get_by_identity, list_suppressions, is_suppressed, gc, mark_addressed)
plus the server/main.py build-helper wiring test (step-19). Consumers ι/κ/λ
switch reads to the ledger under a later write-both/read-new cutover.

Task 2227 (consumer ι) step-01 adds the MemoryService.recon_ledger
construction-contract test: a fresh service has no ledger until
set_recon_ledger() wires one on, mirroring set_write_journal(). The actual
server/main.py call site (wiring the built store onto memory_service inside
the recon_ledger_enabled branch, before the same local is passed to
_collect_checkpoint_targets) is a one-line addition verified by code review,
not by a source-inspection meta-test — see commit da8e5a4c96 (TDD-architect
rule 5: grep-source meta-tests break on benign refactors with no real
regression) for why this repo removed that pattern previously.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import aiosqlite
import pytest
import pytest_asyncio

from fused_memory.reconciliation.mem0_tombstone import (
    RECORD_KIND_MEM0_TOMBSTONE,
    record_mem0_deletion_tombstone,
)
from fused_memory.reconciliation.recon_ledger import (
    MARKER_KINDS,
    ReconLedgerRecord,
    ReconLedgerStore,
)
from fused_memory.reconciliation.standing_decision_constants import (
    EXPIRY_REASON_TTL,
    GROUNDS_STRUCTURAL_SIZE_CONFLATION,
    RECORD_KIND_ENTITY_STANDING_DECISION,
    STATE_ACTIVE,
    STATE_EXPIRED,
)
from fused_memory.server import main as server_main
from fused_memory.services.memory_service import MemoryService

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
    'entity_uuid',
}

EXPECTED_INDEXES = {
    'ix_recon_ledger_project_kind_state',
    'ix_recon_ledger_project_expires',
    'ix_recon_ledger_project_kind_entity',
}

# The pre-migration recon_ledger schema (before task 2894 α added the nullable
# entity_uuid column) — used to construct an "old" DB and prove
# ReconLedgerStore.initialize() adds entity_uuid via its idempotent ADD COLUMN
# migration for pre-existing databases.
PRE_MIGRATION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS recon_ledger (
    project_id   TEXT NOT NULL,
    record_kind  TEXT NOT NULL,
    task_id      TEXT NOT NULL DEFAULT '',
    flag_type    TEXT NOT NULL DEFAULT '',
    run_id       TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL,
    state        TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    expires_at   TEXT,
    PRIMARY KEY (project_id, record_kind, task_id, flag_type, run_id)
);
"""


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
        assert index_names >= EXPECTED_INDEXES, (
            f'Missing indexes: {EXPECTED_INDEXES - index_names}; found: {index_names}'
        )
    finally:
        await s.close()


@pytest.mark.asyncio
async def test_initialize_migrates_pre_existing_db_adding_entity_uuid_column(tmp_path):
    """A recon_ledger DB created WITHOUT entity_uuid (pre-migration schema)
    gains the column after ReconLedgerStore.initialize() runs its idempotent
    ADD COLUMN migration; a second initialize() is a safe no-op (journal.py
    ALTER-TABLE house pattern)."""
    db_path = tmp_path / 'reconciliation.db'
    # Build a pre-migration DB: raw-connect and CREATE TABLE with the OLD column set.
    conn = await aiosqlite.connect(str(db_path))
    try:
        await conn.executescript(PRE_MIGRATION_TABLE_SQL)
        await conn.commit()
        cursor = await conn.execute('PRAGMA table_info(recon_ledger)')
        pre_cols = {row[1] for row in await cursor.fetchall()}
        # Sanity: the DB is genuinely pre-migration.
        assert 'entity_uuid' not in pre_cols
    finally:
        await conn.close()

    store = ReconLedgerStore(db_path)
    await store.initialize()
    try:
        db = store._db
        assert db is not None
        cursor = await db.execute('PRAGMA table_info(recon_ledger)')
        cols_after = {row[1] for row in await cursor.fetchall()}
        assert 'entity_uuid' in cols_after

        # Idempotent: a second initialize() re-runs the ADD COLUMN as a no-op
        # (does not raise, does not duplicate the column).
        await store.initialize()
        db = store._db
        assert db is not None
        cursor = await db.execute('PRAGMA table_info(recon_ledger)')
        cols_reinit = {row[1] for row in await cursor.fetchall()}
        assert cols_reinit == cols_after
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_upsert_round_trips_entity_uuid_set_and_none(store):
    """A generic ReconLedgerRecord round-trips entity_uuid through
    upsert→get_by_identity both when set to a uuid string and when left None
    (the trailing-field default)."""
    with_uuid = ReconLedgerRecord(
        project_id='proj-a',
        record_kind='stage1_flag_marker',
        payload_json='{}',
        state='active',
        created_at='2026-07-01T00:00:00+00:00',
        task_id='task-1',
        flag_type='drift_flag',
        run_id='run-1',
        entity_uuid='uuid-abc',
    )
    await store.upsert(with_uuid)
    fetched = await store.get_by_identity(
        'proj-a', 'stage1_flag_marker', task_id='task-1', flag_type='drift_flag', run_id='run-1'
    )
    assert fetched is not None
    assert fetched.entity_uuid == 'uuid-abc'
    assert fetched == with_uuid

    without_uuid = ReconLedgerRecord(
        project_id='proj-b',
        record_kind='stage1_flag_marker',
        payload_json='{}',
        state='active',
        created_at='2026-07-01T00:00:00+00:00',
        task_id='task-2',
        flag_type='drift_flag',
        run_id='run-2',
    )
    await store.upsert(without_uuid)
    fetched_none = await store.get_by_identity(
        'proj-b', 'stage1_flag_marker', task_id='task-2', flag_type='drift_flag', run_id='run-2'
    )
    assert fetched_none is not None
    assert fetched_none.entity_uuid is None
    assert fetched_none == without_uuid


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
async def test_gc_keeps_marker_kind_row_with_null_expires_when_task_not_terminal(store):
    """A marker-kind row (e.g. stage1_flag_marker) with expires_at=None must
    survive gc() when its task_id is NOT in terminal_task_ids — NULL
    expires_at means never-expire regardless of record_kind, and the
    terminal-referenced clause only deletes rows whose task_id actually
    appears in terminal_task_ids."""
    never_expire_marker = ReconLedgerRecord(
        project_id='proj-p',
        record_kind='stage1_flag_marker',
        payload_json='{}',
        state='active',
        created_at='2026-06-01T00:00:00+00:00',
        task_id='task-live-forever',
        flag_type='drift_flag',
        expires_at=None,
    )
    await store.upsert(never_expire_marker)

    deleted_count = await store.gc(
        'proj-p', now='2026-07-01T00:00:00+00:00', terminal_task_ids=['some-other-task']
    )

    assert deleted_count == 0
    assert (
        await store.get_by_identity(
            'proj-p', 'stage1_flag_marker', task_id='task-live-forever', flag_type='drift_flag'
        )
        is not None
    )


@pytest.mark.asyncio
async def test_marker_task_ids_returns_distinct_nonempty_marker_ids_scoped_to_kind_and_project(store):
    """marker_task_ids(project_id) returns the DISTINCT non-empty task_ids
    across MARKER_KINDS rows for that project (task 2228 W5-κ amendment).

    Exercises three predicates independently: (1) record_kind scoping — a
    non-marker-kind row (stage1_flag_suppression) is excluded even though its
    task_id is non-empty; (2) the task_id != '' exclusion — a MARKER_KINDS
    row with an empty task_id is excluded, proving the filter isn't merely
    incidental to the non-marker-kind rows also having empty ids; and (3)
    project scoping — a marker row in another project is excluded. It also
    confirms DISTINCT collapses a task_id shared by two different marker
    kinds (stage1_flag_marker and flag_for_stage2) to a single entry.
    """
    marker_a = ReconLedgerRecord(
        project_id='proj-a',
        record_kind='stage1_flag_marker',
        payload_json='{}',
        state='active',
        created_at='2026-06-01T00:00:00+00:00',
        task_id='task-1',
        flag_type='f1',
    )
    marker_b = ReconLedgerRecord(
        project_id='proj-a',
        record_kind='stage2_persistence_marker',
        payload_json='{}',
        state='active',
        created_at='2026-06-01T00:00:00+00:00',
        task_id='task-2',
        flag_type='f2',
    )
    marker_shared_task_id = ReconLedgerRecord(
        project_id='proj-a',
        record_kind='flag_for_stage2',
        payload_json='{}',
        state='active',
        created_at='2026-06-01T00:00:00+00:00',
        task_id='task-1',
        flag_type='f3',
    )
    non_marker_kind = ReconLedgerRecord(
        project_id='proj-a',
        record_kind='stage1_flag_suppression',
        payload_json='{}',
        state='active',
        created_at='2026-06-01T00:00:00+00:00',
        task_id='task-4',
        flag_type='f4',
    )
    non_marker_kind_empty_task_id = ReconLedgerRecord(
        project_id='proj-a',
        record_kind='cycle_summary',
        payload_json='{}',
        state='active',
        created_at='2026-06-01T00:00:00+00:00',
    )
    marker_kind_empty_task_id = ReconLedgerRecord(
        project_id='proj-a',
        record_kind='stage1_flag_marker',
        payload_json='{}',
        state='active',
        created_at='2026-06-01T00:00:00+00:00',
        flag_type='f6',
    )
    other_project_marker = ReconLedgerRecord(
        project_id='proj-other',
        record_kind='stage1_flag_marker',
        payload_json='{}',
        state='active',
        created_at='2026-06-01T00:00:00+00:00',
        task_id='task-x',
        flag_type='f7',
    )
    for record in (
        marker_a,
        marker_b,
        marker_shared_task_id,
        non_marker_kind,
        non_marker_kind_empty_task_id,
        marker_kind_empty_task_id,
        other_project_marker,
    ):
        await store.upsert(record)

    result = await store.marker_task_ids('proj-a')

    assert result == {'task-1', 'task-2'}


@pytest.mark.asyncio
async def test_checkpoint_returns_three_int_tuple(store):
    """checkpoint() runs PRAGMA wal_checkpoint(TRUNCATE) directly and returns
    a (busy, log, checkpointed) tuple of three ints — locks the return
    contract independently of its indirect coverage via
    _collect_checkpoint_targets membership."""
    result = await store.checkpoint()

    assert isinstance(result, tuple)
    assert len(result) == 3
    assert all(isinstance(value, int) for value in result)


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


# ---------------------------------------------------------------------------
# task 2227 step-01: MemoryService exposes the ledger
# ---------------------------------------------------------------------------


def test_memory_service_recon_ledger_defaults_to_none(mock_config):
    """A fresh MemoryService has no ledger wired until set_recon_ledger()
    attaches one — mirrors the _write_journal/set_write_journal default."""
    svc = MemoryService(mock_config)

    assert svc.recon_ledger is None


def test_memory_service_set_recon_ledger_wires_store(mock_config, tmp_path):
    """set_recon_ledger(store) attaches the given store as .recon_ledger,
    mirroring set_write_journal(journal) -> ._write_journal."""
    svc = MemoryService(mock_config)
    store = ReconLedgerStore(tmp_path / 'reconciliation.db')

    svc.set_recon_ledger(store)

    assert svc.recon_ledger is store


# ---------------------------------------------------------------------------
# entity_standing_decision write/list/lookup API (task 2894 α, step-5/6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_entity_standing_decision_writes_and_reads_back(store):
    """upsert_entity_standing_decision writes a row discoverable via both
    list_entity_standing_decisions and get_active_entity_standing_decision:
    the entity_uuid column is set, created_at mirrors decided_at, expires_at
    is set, state is active, and payload_json canonically carries grounds +
    edge_count_at_decision + evidence."""
    evidence = [{'type': 'edge', 'id': 'edge-1', 'locally_resolved': True}]
    await store.upsert_entity_standing_decision(
        project_id='proj-a',
        entity_uuid='uuid-abc',
        grounds=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
        decided_at='2026-07-01T00:00:00+00:00',
        expires_at='2026-09-29T00:00:00+00:00',
        edge_count_at_decision=7,
        evidence=evidence,
        state=STATE_ACTIVE,
    )

    listed = await store.list_entity_standing_decisions('proj-a')
    assert len(listed) == 1
    row = listed[0]
    assert row.record_kind == RECORD_KIND_ENTITY_STANDING_DECISION
    assert row.entity_uuid == 'uuid-abc'
    assert row.created_at == '2026-07-01T00:00:00+00:00'
    assert row.expires_at == '2026-09-29T00:00:00+00:00'
    assert row.state == STATE_ACTIVE

    payload = json.loads(row.payload_json)
    assert payload['grounds'] == GROUNDS_STRUCTURAL_SIZE_CONFLATION
    assert payload['edge_count_at_decision'] == 7
    assert payload['evidence'] == evidence

    active = await store.get_active_entity_standing_decision('proj-a', 'uuid-abc')
    assert active is not None
    assert active == row


@pytest.mark.asyncio
async def test_get_active_entity_standing_decision_none_for_unknown_and_non_active(store):
    """get_active_entity_standing_decision returns None for an unknown/other
    entity_uuid and for a decision that is not in the active state."""
    await store.upsert_entity_standing_decision(
        project_id='proj-a',
        entity_uuid='uuid-active',
        grounds=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
        decided_at='2026-07-01T00:00:00+00:00',
        expires_at='2026-09-29T00:00:00+00:00',
        edge_count_at_decision=3,
        evidence=[],
        state=STATE_ACTIVE,
    )
    await store.upsert_entity_standing_decision(
        project_id='proj-a',
        entity_uuid='uuid-expired',
        grounds=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
        decided_at='2026-07-01T00:00:00+00:00',
        expires_at='2026-09-29T00:00:00+00:00',
        edge_count_at_decision=3,
        evidence=[],
        state=STATE_EXPIRED,
    )

    # Unknown entity_uuid → None.
    assert await store.get_active_entity_standing_decision('proj-a', 'uuid-nope') is None
    # Known project but wrong project scope → None.
    assert await store.get_active_entity_standing_decision('proj-other', 'uuid-active') is None
    # A non-active (expired) row is NOT returned by the active lookup.
    assert await store.get_active_entity_standing_decision('proj-a', 'uuid-expired') is None


@pytest.mark.asyncio
async def test_get_active_entity_standing_decision_fast_fails_on_multiple_active(store):
    """get_active_entity_standing_decision raises (loud-over-silent, INV-1)
    instead of silently returning an arbitrary row when an entity_uuid carries
    more than one ACTIVE standing decision. This cannot arise through
    upsert_entity_standing_decision today (single-member GROUNDS_ENUM ⇒ at most
    one active grounds per entity), so two rows sharing an entity_uuid but
    differing in the grounds/flag_type PK slot are written via a raw upsert() to
    simulate a future multi-grounds enum. Guards against the γ/δ hooks getting
    non-deterministic results before a second grounds value is admitted."""
    for grounds in ('structural_size_conflation', 'some_future_grounds'):
        await store.upsert(
            ReconLedgerRecord(
                project_id='proj-a',
                record_kind=RECORD_KIND_ENTITY_STANDING_DECISION,
                payload_json='{}',
                state=STATE_ACTIVE,
                created_at='2026-07-01T00:00:00+00:00',
                task_id='',
                flag_type=grounds,  # distinct grounds → distinct PK, same entity_uuid
                run_id='uuid-dup',
                entity_uuid='uuid-dup',
            )
        )

    with pytest.raises(ValueError, match='more than one ACTIVE'):
        await store.get_active_entity_standing_decision('proj-a', 'uuid-dup')


@pytest.mark.asyncio
async def test_list_entity_standing_decisions_scoped_to_kind_project_and_state(store):
    """list_entity_standing_decisions returns only entity_standing_decision
    rows, scoped to the project, optionally filtered by state — never a
    marker-kind row and never another project's decision."""
    # A non-standing marker row in the same project must be excluded.
    await store.upsert(
        ReconLedgerRecord(
            project_id='proj-a',
            record_kind='stage1_flag_marker',
            payload_json='{}',
            state='active',
            created_at='2026-07-01T00:00:00+00:00',
            task_id='task-1',
            flag_type='drift_flag',
        )
    )
    await store.upsert_entity_standing_decision(
        project_id='proj-a',
        entity_uuid='uuid-1',
        grounds=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
        decided_at='2026-07-01T00:00:00+00:00',
        expires_at='2026-09-29T00:00:00+00:00',
        edge_count_at_decision=1,
        evidence=[],
        state=STATE_ACTIVE,
    )
    await store.upsert_entity_standing_decision(
        project_id='proj-a',
        entity_uuid='uuid-2',
        grounds=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
        decided_at='2026-07-01T00:00:00+00:00',
        expires_at='2026-09-29T00:00:00+00:00',
        edge_count_at_decision=2,
        evidence=[],
        state=STATE_EXPIRED,
    )
    # A decision in another project must be excluded.
    await store.upsert_entity_standing_decision(
        project_id='proj-b',
        entity_uuid='uuid-3',
        grounds=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
        decided_at='2026-07-01T00:00:00+00:00',
        expires_at='2026-09-29T00:00:00+00:00',
        edge_count_at_decision=3,
        evidence=[],
        state=STATE_ACTIVE,
    )

    all_rows = await store.list_entity_standing_decisions('proj-a')
    assert {r.entity_uuid for r in all_rows} == {'uuid-1', 'uuid-2'}
    assert all(r.record_kind == RECORD_KIND_ENTITY_STANDING_DECISION for r in all_rows)

    active_only = await store.list_entity_standing_decisions('proj-a', state=STATE_ACTIVE)
    assert {r.entity_uuid for r in active_only} == {'uuid-1'}

    expired_only = await store.list_entity_standing_decisions('proj-a', state=STATE_EXPIRED)
    assert {r.entity_uuid for r in expired_only} == {'uuid-2'}


@pytest.mark.asyncio
async def test_upsert_entity_standing_decision_uniqueness_last_write_wins(store):
    """Re-upsert for the same (entity_uuid, grounds) is last-write-wins (one
    row); two distinct entity_uuids yield two rows."""
    common = dict(
        project_id='proj-a',
        grounds=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
        decided_at='2026-07-01T00:00:00+00:00',
        expires_at='2026-09-29T00:00:00+00:00',
        evidence=[],
        state=STATE_ACTIVE,
    )
    await store.upsert_entity_standing_decision(
        entity_uuid='uuid-1', edge_count_at_decision=1, **common
    )
    await store.upsert_entity_standing_decision(
        entity_uuid='uuid-1', edge_count_at_decision=42, **common
    )
    rows = await store.list_entity_standing_decisions('proj-a')
    assert len(rows) == 1
    assert json.loads(rows[0].payload_json)['edge_count_at_decision'] == 42

    await store.upsert_entity_standing_decision(
        entity_uuid='uuid-2', edge_count_at_decision=5, **common
    )
    rows2 = await store.list_entity_standing_decisions('proj-a')
    assert {r.entity_uuid for r in rows2} == {'uuid-1', 'uuid-2'}


@pytest.mark.asyncio
async def test_upsert_entity_standing_decision_validation_raises(store):
    """Structured loud failure (INV-1): empty entity_uuid, expires_at=None, and
    a grounds value outside GROUNDS_ENUM each raise ValueError with a hint
    naming the failing field — never a silent no-op or a malformed row."""
    valid = dict(
        project_id='proj-a',
        entity_uuid='uuid-ok',
        grounds=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
        decided_at='2026-07-01T00:00:00+00:00',
        expires_at='2026-09-29T00:00:00+00:00',
        edge_count_at_decision=1,
        evidence=[],
    )

    with pytest.raises(ValueError, match='entity_uuid'):
        await store.upsert_entity_standing_decision(**{**valid, 'entity_uuid': ''})

    with pytest.raises(ValueError, match='expires_at'):
        await store.upsert_entity_standing_decision(**{**valid, 'expires_at': None})

    with pytest.raises(ValueError, match='grounds'):
        await store.upsert_entity_standing_decision(**{**valid, 'grounds': 'not_a_real_grounds'})

    # No malformed row was written by any rejected call.
    assert await store.list_entity_standing_decisions('proj-a') == []


# ---------------------------------------------------------------------------
# gc() TTL-flip for entity_standing_decision (task 2894 α, step-7/8)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gc_flips_expired_standing_decision_to_expired_with_ttl_reason(store):
    """An ACTIVE entity_standing_decision past its expires_at is NOT deleted by
    gc() (unlike marker kinds) — it is flipped to state='expired' with
    payload_json expiry_reason='ttl' (INV-2), and get_active then returns None.
    Exercises the empty-terminal_task_ids gc branch."""
    await store.upsert_entity_standing_decision(
        project_id='proj-p',
        entity_uuid='uuid-old',
        grounds=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
        decided_at='2026-04-01T00:00:00+00:00',
        expires_at='2026-06-30T00:00:00+00:00',  # < now
        edge_count_at_decision=4,
        evidence=[],
        state=STATE_ACTIVE,
    )

    count = await store.gc('proj-p', now='2026-07-01T00:00:00+00:00', terminal_task_ids=[])

    # Not deleted — still discoverable, but now expired with a ttl reason.
    listed = await store.list_entity_standing_decisions('proj-p')
    assert len(listed) == 1
    row = listed[0]
    assert row.state == STATE_EXPIRED
    assert json.loads(row.payload_json)['expiry_reason'] == EXPIRY_REASON_TTL
    # No longer active.
    assert await store.get_active_entity_standing_decision('proj-p', 'uuid-old') is None
    # gc acted on exactly one row (the flip).
    assert count == 1


@pytest.mark.asyncio
async def test_gc_leaves_unexpired_active_standing_decision_untouched(store):
    """An ACTIVE entity_standing_decision whose expires_at is in the future is
    left fully untouched by gc() — still active, no expiry_reason stamped."""
    await store.upsert_entity_standing_decision(
        project_id='proj-p',
        entity_uuid='uuid-live',
        grounds=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
        decided_at='2026-07-01T00:00:00+00:00',
        expires_at='2026-12-01T00:00:00+00:00',  # > now
        edge_count_at_decision=2,
        evidence=[],
        state=STATE_ACTIVE,
    )

    count = await store.gc('proj-p', now='2026-08-01T00:00:00+00:00', terminal_task_ids=[])

    active = await store.get_active_entity_standing_decision('proj-p', 'uuid-live')
    assert active is not None
    assert active.state == STATE_ACTIVE
    assert 'expiry_reason' not in json.loads(active.payload_json)
    assert count == 0


@pytest.mark.asyncio
async def test_gc_leaves_already_expired_standing_decision_idempotent(store):
    """An already state='expired' standing row with a past expires_at is left
    untouched — the flip is gated on state='active', so gc() neither deletes it
    nor re-stamps expiry_reason (idempotent)."""
    await store.upsert_entity_standing_decision(
        project_id='proj-p',
        entity_uuid='uuid-already',
        grounds=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
        decided_at='2026-04-01T00:00:00+00:00',
        expires_at='2026-06-30T00:00:00+00:00',  # < now
        edge_count_at_decision=1,
        evidence=[],
        state=STATE_EXPIRED,
    )

    count = await store.gc('proj-p', now='2026-07-01T00:00:00+00:00', terminal_task_ids=[])

    listed = await store.list_entity_standing_decisions('proj-p')
    assert len(listed) == 1  # not deleted
    assert listed[0].state == STATE_EXPIRED
    # Not re-stamped: the row carried no expiry_reason and still carries none.
    assert 'expiry_reason' not in json.loads(listed[0].payload_json)
    assert count == 0  # nothing acted on


@pytest.mark.asyncio
async def test_gc_mixed_flips_standing_and_deletes_markers_with_terminal_ids(store):
    """Mixed pass over the non-empty terminal_task_ids branch: an expired
    standing decision FLIPS (kept), an expired marker DELETES, a
    terminal-referenced marker DELETES; the returned count == deleted +
    flipped."""
    # Standing decision (active, past expires_at) → FLIP, not delete.
    await store.upsert_entity_standing_decision(
        project_id='proj-p',
        entity_uuid='uuid-flip',
        grounds=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
        decided_at='2026-04-01T00:00:00+00:00',
        expires_at='2026-06-30T00:00:00+00:00',
        edge_count_at_decision=9,
        evidence=[],
        state=STATE_ACTIVE,
    )
    # Expired marker-kind row → DELETE via the expiry arm.
    await store.upsert(
        ReconLedgerRecord(
            project_id='proj-p',
            record_kind='stage1_flag_marker',
            payload_json='{}',
            state='active',
            created_at='2026-06-01T00:00:00+00:00',
            task_id='task-expired',
            flag_type='drift_flag',
            expires_at='2026-06-15T00:00:00+00:00',
        )
    )
    # Terminal-referenced marker (not time-expired) → DELETE via the terminal arm.
    await store.upsert(
        ReconLedgerRecord(
            project_id='proj-p',
            record_kind='stage2_persistence_marker',
            payload_json='{}',
            state='active',
            created_at='2026-06-01T00:00:00+00:00',
            task_id='task-terminal',
            flag_type='persistence_flag',
            expires_at='2026-12-01T00:00:00+00:00',
        )
    )

    count = await store.gc(
        'proj-p', now='2026-07-01T00:00:00+00:00', terminal_task_ids=['task-terminal']
    )

    # Standing row flipped, not deleted.
    listed = await store.list_entity_standing_decisions('proj-p')
    assert len(listed) == 1
    assert listed[0].state == STATE_EXPIRED
    assert json.loads(listed[0].payload_json)['expiry_reason'] == EXPIRY_REASON_TTL
    assert await store.get_active_entity_standing_decision('proj-p', 'uuid-flip') is None
    # Both markers deleted.
    assert (
        await store.get_by_identity(
            'proj-p', 'stage1_flag_marker', task_id='task-expired', flag_type='drift_flag'
        )
        is None
    )
    assert (
        await store.get_by_identity(
            'proj-p',
            'stage2_persistence_marker',
            task_id='task-terminal',
            flag_type='persistence_flag',
        )
        is None
    )
    # Count == deleted (2 markers) + flipped (1 standing decision).
    assert count == 3


@pytest.mark.asyncio
async def test_gc_flip_defensively_wraps_non_dict_standing_payload(store):
    """The gc() TTL-flip tolerates a non-object payload_json on an
    entity_standing_decision row the same way mark_addressed does: a JSON
    scalar/list is wrapped as {'_payload': <value>} before expiry_reason='ttl'
    is stamped, and the row still flips to state='expired' (INV-2). Written via
    a raw upsert() because upsert_entity_standing_decision always builds a dict
    payload, so this defensive branch is otherwise unexercised."""
    # Raw upsert of a standing-decision-kind row whose payload_json is a JSON
    # list (valid JSON, but not an object). The entity_standing_decision PK-slot
    # layout (task_id='', flag_type=grounds, run_id=entity_uuid) is mirrored so
    # gc()'s flip SELECT — keyed on record_kind + state + expires_at — matches.
    await store.upsert(
        ReconLedgerRecord(
            project_id='proj-p',
            record_kind=RECORD_KIND_ENTITY_STANDING_DECISION,
            payload_json='[1, 2, 3]',
            state=STATE_ACTIVE,
            created_at='2026-04-01T00:00:00+00:00',
            task_id='',
            flag_type=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
            run_id='uuid-scalar',
            expires_at='2026-06-30T00:00:00+00:00',  # < now
            entity_uuid='uuid-scalar',
        )
    )

    count = await store.gc('proj-p', now='2026-07-01T00:00:00+00:00', terminal_task_ids=[])

    listed = await store.list_entity_standing_decisions('proj-p')
    assert len(listed) == 1  # flipped, not deleted
    row = listed[0]
    assert row.state == STATE_EXPIRED
    # Non-dict payload defensively wrapped under '_payload', then ttl-stamped.
    assert json.loads(row.payload_json) == {
        '_payload': [1, 2, 3],
        'expiry_reason': EXPIRY_REASON_TTL,
    }
    assert await store.get_active_entity_standing_decision('proj-p', 'uuid-scalar') is None
    assert count == 1


# --------------------------------------------------------------------------
# mem0_tombstone rows (task 3041)
#
# A tombstone is a plain recon_ledger row with record_kind='mem0_tombstone'
# and task_id=<the deleted Mem0 record's uuid>, written after every confirmed
# recon-initiated Mem0 delete. It deliberately reuses three existing store
# mechanisms rather than adding a fourth store: the five-part-PK ON CONFLICT
# for idempotence, get_by_identity for the lookup, and gc()'s expires_at pass
# for TTL-bounded growth.
#
# The load-bearing NON-membership invariant these tests pin: 'mem0_tombstone'
# is NOT in MARKER_KINDS, because its task_id column holds a Mem0 memory uuid
# rather than a Taskmaster task id. Both task-id-keyed paths (gc()'s
# terminal-task DELETE arm and marker_task_ids(), which _gc_recon_markers
# feeds straight back into gc()) must therefore never see it.
# --------------------------------------------------------------------------


def _tombstone_ms(store):
    """Minimal memory_service stand-in exposing the real store as recon_ledger.

    record_mem0_deletion_tombstone resolves its store via
    getattr(memory_service, 'recon_ledger', None), so this is the whole
    surface it needs — driving the REAL writer here keeps these store-level
    tests honest about the row shape the production path actually produces.
    """
    return SimpleNamespace(recon_ledger=store)


@pytest.mark.asyncio
async def test_get_mem0_tombstone_returns_row_for_memory_uuid(store):
    """get_mem0_tombstone(project, memory_id) answers on the auditor's ONLY key.

    An auditor arriving from `get_memory_by_id -> {found: false}` knows exactly
    one thing: the memory uuid. So the accessor must be satisfiable from
    (project_id, memory_id) alone — flag_type/run_id stay at their '' defaults.
    """
    written = await record_mem0_deletion_tombstone(
        _tombstone_ms(store),
        'proj-p',
        'mem-victim',
        victim_metadata={'kind': 'cycle_summary', 'record_type': 'ledger_stamp'},
        victim_created_at='2026-07-01T00:00:00+00:00',
        deleter='stage1_cycle_summary_trim',
        deleting_run_id='run-deleter',
        now=datetime(2026, 7, 20, tzinfo=UTC),
    )
    assert written is True

    fetched = await store.get_mem0_tombstone('proj-p', 'mem-victim')
    assert fetched is not None
    assert fetched.record_kind == RECORD_KIND_MEM0_TOMBSTONE
    assert fetched.task_id == 'mem-victim'
    assert fetched.flag_type == ''
    assert fetched.run_id == ''
    payload = json.loads(fetched.payload_json)
    assert payload['deleter'] == 'stage1_cycle_summary_trim'
    assert payload['deleting_run_id'] == 'run-deleter'


@pytest.mark.asyncio
async def test_get_mem0_tombstone_absent_returns_none(store):
    """No tombstone for this uuid (or for this project) returns None, not a raise.

    This is the ordinary never-existed case: the record was simply never
    written, which must stay distinguishable from a deliberate reap.
    """
    assert await store.get_mem0_tombstone('proj-p', 'mem-never-deleted') is None

    await record_mem0_deletion_tombstone(
        _tombstone_ms(store),
        'proj-p',
        'mem-victim',
        victim_metadata={},
        victim_created_at=None,
        deleter='stage1_flag_marker_gc_sweep',
        deleting_run_id='run-deleter',
    )
    # Scoped by project: the same uuid under another project is still absent.
    assert await store.get_mem0_tombstone('other-proj', 'mem-victim') is None


@pytest.mark.asyncio
async def test_repeat_tombstone_for_same_victim_keeps_one_row(store):
    """A second tombstone for the same victim upserts in place (row count 1).

    Idempotence comes free from the five-part PRIMARY KEY's ON CONFLICT — a
    sweep that somehow re-deletes (or is replayed against) the same uuid must
    not accumulate duplicate rows.
    """
    for run in ('run-1', 'run-2'):
        await record_mem0_deletion_tombstone(
            _tombstone_ms(store),
            'proj-p',
            'mem-victim',
            victim_metadata={'kind': 'cycle_summary'},
            victim_created_at='2026-07-01T00:00:00+00:00',
            deleter='stage1_cycle_summary_trim',
            deleting_run_id=run,
            now=datetime(2026, 7, 20, tzinfo=UTC),
        )

    cursor = await store._db.execute(
        'SELECT COUNT(*) FROM recon_ledger WHERE record_kind = ?',
        (RECORD_KIND_MEM0_TOMBSTONE,),
    )
    row = await cursor.fetchone()
    assert row[0] == 1

    fetched = await store.get_mem0_tombstone('proj-p', 'mem-victim')
    assert fetched is not None
    assert json.loads(fetched.payload_json)['deleting_run_id'] == 'run-2' # last write wins


@pytest.mark.asyncio
async def test_gc_expires_tombstones_but_keeps_unexpired(store):
    """gc()'s expires_at pass reaps an expired tombstone and spares a fresh one.

    This is what makes tombstone growth bounded with no new cleanup code: the
    pass already runs every cycle from _gc_recon_markers, and
    MEM0_TOMBSTONE_TTL_DAYS puts expires_at within its reach.
    """
    # Written at 2026-06-01 -> expires 30 days later, before `now` below.
    await record_mem0_deletion_tombstone(
        _tombstone_ms(store),
        'proj-p',
        'mem-old',
        victim_metadata={'kind': 'cycle_summary'},
        victim_created_at='2026-06-01T00:00:00+00:00',
        deleter='stage1_cycle_summary_trim',
        deleting_run_id='run-old',
        now=datetime(2026, 6, 1, tzinfo=UTC),
    )
    await record_mem0_deletion_tombstone(
        _tombstone_ms(store),
        'proj-p',
        'mem-fresh',
        victim_metadata={'kind': 'cycle_summary'},
        victim_created_at='2026-07-20T00:00:00+00:00',
        deleter='stage1_cycle_summary_trim',
        deleting_run_id='run-fresh',
        now=datetime(2026, 7, 20, tzinfo=UTC),
    )

    await store.gc('proj-p', now='2026-07-25T00:00:00+00:00', terminal_task_ids=[])

    assert await store.get_mem0_tombstone('proj-p', 'mem-old') is None
    assert await store.get_mem0_tombstone('proj-p', 'mem-fresh') is not None


@pytest.mark.asyncio
async def test_terminal_task_gc_never_reaches_an_unexpired_tombstone(store):
    """A memory uuid colliding with a terminal task id must not reap a tombstone.

    gc()'s terminal-referenced DELETE arm is gated on
    `record_kind IN MARKER_KINDS`; 'mem0_tombstone' is deliberately excluded
    from that tuple because its task_id column holds a Mem0 memory uuid, not a
    Taskmaster task id. Without that exclusion an unlucky id collision would
    delete the audit trail of the very record someone is investigating.
    """
    assert RECORD_KIND_MEM0_TOMBSTONE not in MARKER_KINDS

    await record_mem0_deletion_tombstone(
        _tombstone_ms(store),
        'proj-p',
        'mem-victim',
        victim_metadata={'kind': 'cycle_summary'},
        victim_created_at='2026-07-20T00:00:00+00:00',
        deleter='stage1_cycle_summary_trim',
        deleting_run_id='run-deleter',
        now=datetime(2026, 7, 20, tzinfo=UTC),
    )

    await store.gc(
        'proj-p',
        now='2026-07-25T00:00:00+00:00', # before the 30-day expiry
        terminal_task_ids=['mem-victim'], # the collision
    )

    assert await store.get_mem0_tombstone('proj-p', 'mem-victim') is not None


@pytest.mark.asyncio
async def test_marker_task_ids_excludes_tombstone_memory_uuids(store):
    """A memory uuid must never surface from marker_task_ids().

    _gc_recon_markers feeds marker_task_ids() straight back into gc()'s
    terminal_task_ids, so a uuid leaking into that set would widen the
    terminal DELETE arm against ids that are not tasks at all.
    """
    await record_mem0_deletion_tombstone(
        _tombstone_ms(store),
        'proj-p',
        'mem-victim',
        victim_metadata={'kind': 'cycle_summary'},
        victim_created_at='2026-07-20T00:00:00+00:00',
        deleter='stage1_cycle_summary_trim',
        deleting_run_id='run-deleter',
        now=datetime(2026, 7, 20, tzinfo=UTC),
    )
    await store.upsert(
        ReconLedgerRecord(
            project_id='proj-p',
            record_kind='stage1_flag_marker',
            payload_json='{}',
            state='active',
            created_at='2026-07-20T00:00:00+00:00',
            task_id='task-42',
        )
    )

    assert await store.marker_task_ids('proj-p') == {'task-42'}
