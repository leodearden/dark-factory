"""Tests for the write journal (SQLite persistence)."""

import asyncio
import json
import uuid

import pytest
import pytest_asyncio

from fused_memory.services.write_journal import WriteJournal


@pytest_asyncio.fixture
async def journal(tmp_path):
    j = WriteJournal(tmp_path / 'test_wj')
    await j.initialize()
    yield j
    await j.close()


@pytest.mark.asyncio
async def test_initialize_creates_db(tmp_path):
    j = WriteJournal(tmp_path / 'init_test')
    await j.initialize()
    assert (tmp_path / 'init_test' / 'write_journal.db').exists()
    await j.close()


@pytest.mark.asyncio
async def test_log_write_op_roundtrip(journal):
    op_id = str(uuid.uuid4())
    causation = str(uuid.uuid4())
    await journal.log_write_op(
        write_op_id=op_id,
        causation_id=causation,
        source='mcp_tool',
        operation='add_memory',
        project_id='test-project',
        agent_id='claude-interactive',
        params={'content': 'hello', 'category': 'observations_and_summaries'},
        result_summary={'memory_ids': ['m1']},
        success=True,
    )
    ops = await journal.get_ops_by_causation(causation)
    assert len(ops) == 1
    assert ops[0]['layer'] == 'write_op'
    assert ops[0]['id'] == op_id
    assert ops[0]['operation'] == 'add_memory'
    assert ops[0]['success'] == 1


@pytest.mark.asyncio
async def test_log_backend_op_roundtrip(journal):
    write_op_id = str(uuid.uuid4())
    causation = str(uuid.uuid4())
    await journal.log_backend_op(
        write_op_id=write_op_id,
        causation_id=causation,
        backend='mem0',
        operation='add',
        payload={'content': 'test fact'},
        result_summary={'results': [{'id': 'm1'}]},
        success=True,
    )
    ops = await journal.get_backend_ops_for_write_op(write_op_id)
    assert len(ops) == 1
    assert ops[0]['backend'] == 'mem0'
    assert ops[0]['write_op_id'] == write_op_id


@pytest.mark.asyncio
async def test_causation_id_queries_both_layers(journal):
    causation = str(uuid.uuid4())
    write_id = str(uuid.uuid4())

    await journal.log_write_op(
        write_op_id=write_id,
        causation_id=causation,
        operation='add_memory',
        project_id='test',
    )
    await journal.log_backend_op(
        write_op_id=write_id,
        causation_id=causation,
        backend='mem0',
        operation='add',
    )
    await journal.log_backend_op(
        write_op_id=write_id,
        causation_id=causation,
        backend='graphiti',
        operation='add_episode',
    )

    ops = await journal.get_ops_by_causation(causation)
    assert len(ops) == 3
    layers = {op['layer'] for op in ops}
    assert layers == {'write_op', 'backend_op'}


@pytest.mark.asyncio
async def test_error_recording(journal):
    op_id = str(uuid.uuid4())
    await journal.log_write_op(
        write_op_id=op_id,
        operation='add_memory',
        success=False,
        error='Connection refused',
    )
    ops = await journal.get_ops_since('2000-01-01T00:00:00')
    assert len(ops) == 1
    assert ops[0]['success'] == 0
    assert ops[0]['error'] == 'Connection refused'


@pytest.mark.asyncio
async def test_get_ops_since(journal):
    for _i in range(5):
        await journal.log_write_op(
            write_op_id=str(uuid.uuid4()),
            operation='add_memory',
            project_id='test',
        )
    ops = await journal.get_ops_since('2000-01-01T00:00:00', limit=3)
    assert len(ops) == 3


@pytest.mark.asyncio
async def test_concurrent_writes(journal):
    """Multiple concurrent writes should not corrupt the database."""
    async def write_one(i: int):
        await journal.log_write_op(
            write_op_id=str(uuid.uuid4()),
            causation_id='shared-causation',
            operation='add_memory',
            project_id=f'project-{i}',
        )

    await asyncio.gather(*(write_one(i) for i in range(20)))
    ops = await journal.get_ops_by_causation('shared-causation')
    assert len(ops) == 20


@pytest.mark.asyncio
async def test_log_write_op_never_raises(journal):
    """Journaling failures should be swallowed, not propagated."""
    # Close the DB to force an error
    await journal.close()
    journal._db = None
    # Should not raise
    await journal.log_write_op(
        write_op_id=str(uuid.uuid4()),
        operation='add_memory',
    )


@pytest.mark.asyncio
async def test_log_backend_op_never_raises(journal):
    await journal.close()
    journal._db = None
    await journal.log_backend_op(
        backend='mem0',
        operation='add',
    )


@pytest.mark.asyncio
async def test_provenance_field(journal):
    op_id = str(uuid.uuid4())
    causation = str(uuid.uuid4())
    await journal.log_write_op(
        write_op_id=op_id,
        causation_id=causation,
        source='dual_write',
        provenance='derived',
        operation='add_memory',
    )
    ops = await journal.get_ops_by_causation(causation)
    assert ops[0]['provenance'] == 'derived'
    assert ops[0]['source'] == 'dual_write'


# ------------------------------------------------------------------
# New columns: session_id, kind
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kind_column_persists(journal):
    """kind='read' is stored and retrievable."""
    op_id = str(uuid.uuid4())
    await journal.log_write_op(
        write_op_id=op_id,
        operation='search',
        kind='read',
        project_id='test',
    )
    ops = await journal.get_ops_since('2000-01-01T00:00:00')
    assert len(ops) == 1
    assert ops[0]['kind'] == 'read'


@pytest.mark.asyncio
async def test_kind_defaults_to_write(journal):
    """Omitting kind gives 'write'."""
    op_id = str(uuid.uuid4())
    await journal.log_write_op(
        write_op_id=op_id,
        operation='add_memory',
        project_id='test',
    )
    ops = await journal.get_ops_since('2000-01-01T00:00:00')
    assert ops[0]['kind'] == 'write'


@pytest.mark.asyncio
async def test_session_id_persists(journal):
    op_id = str(uuid.uuid4())
    await journal.log_write_op(
        write_op_id=op_id,
        operation='search',
        kind='read',
        session_id='sess-abc',
        agent_id='claude-task-7',
    )
    ops = await journal.get_ops_since('2000-01-01T00:00:00')
    assert ops[0]['session_id'] == 'sess-abc'
    assert ops[0]['agent_id'] == 'claude-task-7'


@pytest.mark.asyncio
async def test_get_ops_since_kind_filter(journal):
    """get_ops_since with kind filter returns only matching rows."""
    for _i in range(3):
        await journal.log_write_op(
            write_op_id=str(uuid.uuid4()),
            operation='add_memory',
            kind='write',
            project_id='test',
        )
    for _i in range(2):
        await journal.log_write_op(
            write_op_id=str(uuid.uuid4()),
            operation='search',
            kind='read',
            project_id='test',
        )
    reads = await journal.get_ops_since('2000-01-01T00:00:00', kind='read')
    writes = await journal.get_ops_since('2000-01-01T00:00:00', kind='write')
    assert len(reads) == 2
    assert len(writes) == 3


# ------------------------------------------------------------------
# Migration: existing DB without new columns
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_migration_adds_columns(tmp_path):
    """Initializing against a DB created with old schema gets new columns."""
    import aiosqlite

    db_dir = tmp_path / 'migrate_test'
    db_dir.mkdir()
    db_path = db_dir / 'write_journal.db'

    # Create old schema (no session_id, no kind)
    old_schema = """
    CREATE TABLE IF NOT EXISTS write_ops (
        id TEXT PRIMARY KEY,
        causation_id TEXT,
        source TEXT,
        provenance TEXT DEFAULT 'original',
        operation TEXT,
        project_id TEXT,
        agent_id TEXT,
        params TEXT DEFAULT '{}',
        result_summary TEXT,
        success INTEGER DEFAULT 1,
        error TEXT,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS backend_ops (
        id TEXT PRIMARY KEY,
        write_op_id TEXT,
        causation_id TEXT,
        backend TEXT,
        operation TEXT,
        payload TEXT DEFAULT '{}',
        result_summary TEXT,
        success INTEGER DEFAULT 1,
        error TEXT,
        created_at TEXT NOT NULL
    );
    """
    async with aiosqlite.connect(str(db_path)) as db:
        await db.executescript(old_schema)
        # Insert a pre-existing search row
        await db.execute(
            "INSERT INTO write_ops (id, operation, created_at) VALUES (?, ?, ?)",
            ('old-search', 'search', '2025-01-01T00:00:00'),
        )
        await db.commit()

    # Now initialize WriteJournal — should migrate
    j = WriteJournal(db_dir)
    await j.initialize()

    # Check columns exist
    db = j._require_db()
    async with db.execute('PRAGMA table_info(write_ops)') as cursor:
        cols = {row[1] for row in await cursor.fetchall()}
    assert 'session_id' in cols
    assert 'kind' in cols

    # Check backfill: old search row should have kind='read'
    async with db.execute(
        "SELECT kind FROM write_ops WHERE id = 'old-search'"
    ) as cursor:
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 'read'

    await j.close()


# ------------------------------------------------------------------
# Stats methods
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_usage_stats(journal):
    # 3 writes, 2 reads
    for _ in range(3):
        await journal.log_write_op(
            write_op_id=str(uuid.uuid4()),
            operation='add_memory',
            kind='write',
            project_id='proj',
            agent_id='agent-a',
        )
    for _ in range(2):
        await journal.log_write_op(
            write_op_id=str(uuid.uuid4()),
            operation='search',
            kind='read',
            project_id='proj',
            agent_id='agent-b',
        )

    stats = await journal.get_usage_stats('2000-01-01T00:00:00')
    assert stats['reads'] == 2
    assert stats['writes'] == 3
    assert stats['by_operation']['add_memory'] == 3
    assert stats['by_operation']['search'] == 2
    assert stats['by_agent']['agent-a'] == {'read': 0, 'write': 3}
    assert stats['by_agent']['agent-b'] == {'read': 2, 'write': 0}


@pytest.mark.asyncio
async def test_get_usage_stats_project_filter(journal):
    await journal.log_write_op(
        write_op_id=str(uuid.uuid4()),
        operation='add_memory',
        project_id='alpha',
    )
    await journal.log_write_op(
        write_op_id=str(uuid.uuid4()),
        operation='add_memory',
        project_id='beta',
    )
    stats = await journal.get_usage_stats('2000-01-01T00:00:00', project_id='alpha')
    assert stats['writes'] == 1


@pytest.mark.asyncio
async def test_get_session_ops(journal):
    for i in range(5):
        await journal.log_write_op(
            write_op_id=str(uuid.uuid4()),
            operation='search' if i % 2 == 0 else 'add_memory',
            kind='read' if i % 2 == 0 else 'write',
            agent_id='target-agent',
        )
    await journal.log_write_op(
        write_op_id=str(uuid.uuid4()),
        operation='add_memory',
        agent_id='other-agent',
    )

    ops = await journal.get_session_ops('target-agent')
    assert len(ops) == 5
    assert all(op['agent_id'] == 'target-agent' for op in ops)
    # Most recent first
    assert ops[0]['created_at'] >= ops[-1]['created_at']


@pytest.mark.asyncio
async def test_get_session_ops_with_limit(journal):
    for _ in range(10):
        await journal.log_write_op(
            write_op_id=str(uuid.uuid4()),
            operation='search',
            kind='read',
            agent_id='busy-agent',
        )
    ops = await journal.get_session_ops('busy-agent', limit=3)
    assert len(ops) == 3


# ------------------------------------------------------------------
# mem0_intents: write-ahead intent journal (task 2710)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_log_mem0_intent_roundtrip_pending(journal):
    """log_mem0_intent persists a pending row with FULL content + metadata intact."""
    intent_id = str(uuid.uuid4())
    write_op_id = str(uuid.uuid4())
    # > 200 chars to prove content is stored untruncated (write_ops truncates
    # to 200; a re-issued twin needs the full content).
    long_content = 'the quick brown fox ' * 30
    metadata = {'category': 'preferences_and_norms', 'kind': 'test', 'nested': {'a': 1}}
    await journal.log_mem0_intent(
        intent_id=intent_id,
        write_op_id=write_op_id,
        causation_id='cause-1',
        project_id='proj',
        agent_id='agent-a',
        session_id='sess-1',
        category='preferences_and_norms',
        content=long_content,
        metadata=metadata,
        payload_digest='digest-abc',
    )
    incomplete = await journal.get_incomplete_mem0_intents()
    assert len(incomplete) == 1
    row = incomplete[0]
    assert row['id'] == intent_id
    assert row['write_op_id'] == write_op_id
    assert row['causation_id'] == 'cause-1'
    assert row['project_id'] == 'proj'
    assert row['agent_id'] == 'agent-a'
    assert row['session_id'] == 'sess-1'
    assert row['category'] == 'preferences_and_norms'
    assert row['status'] == 'pending'
    assert row['payload_digest'] == 'digest-abc'
    # Content is stored FULL (untruncated) so a re-issue is faithful.
    assert row['content'] == long_content
    assert len(row['content']) > 200
    assert json.loads(row['metadata']) == metadata
    assert row['resolved_at'] is None


@pytest.mark.asyncio
async def test_resolve_mem0_intent_completed_removes_from_incomplete(journal):
    """resolve_mem0_intent(id, 'completed') stamps status+resolved_at and clears pending."""
    intent_id = str(uuid.uuid4())
    await journal.log_mem0_intent(
        intent_id=intent_id,
        write_op_id=str(uuid.uuid4()),
        content='hi',
        metadata={},
        payload_digest='d',
    )
    await journal.resolve_mem0_intent(intent_id, 'completed')

    assert await journal.get_incomplete_mem0_intents() == []
    rows = await journal.get_mem0_intents(status='completed')
    assert len(rows) == 1
    assert rows[0]['id'] == intent_id
    assert rows[0]['status'] == 'completed'
    assert rows[0]['resolved_at'] is not None


@pytest.mark.asyncio
async def test_resolve_mem0_intent_dead_persists_reason(journal):
    """resolve_mem0_intent(id, 'dead', reason=...) persists the reason and status='dead'."""
    intent_id = str(uuid.uuid4())
    await journal.log_mem0_intent(
        intent_id=intent_id,
        write_op_id=str(uuid.uuid4()),
        content='hi',
        metadata={},
        payload_digest='d',
    )
    reason = 'no backend_op: unknown outcome (not auto-re-issued)'
    await journal.resolve_mem0_intent(intent_id, 'dead', reason=reason)

    dead = await journal.get_mem0_intents(status='dead')
    assert len(dead) == 1
    assert dead[0]['id'] == intent_id
    assert dead[0]['status'] == 'dead'
    assert dead[0]['reason'] == reason
    assert dead[0]['resolved_at'] is not None


@pytest.mark.asyncio
async def test_get_mem0_intents_status_filter(journal):
    """get_mem0_intents(status='dead') returns only dead rows; None returns all."""
    ids = {}
    for status in ['completed', 'failed', 'dead', 'pending']:
        iid = str(uuid.uuid4())
        ids[status] = iid
        await journal.log_mem0_intent(
            intent_id=iid,
            write_op_id=str(uuid.uuid4()),
            content=f'content-{status}',
            metadata={},
            payload_digest='d',
        )
        if status != 'pending':
            await journal.resolve_mem0_intent(iid, status)

    dead = await journal.get_mem0_intents(status='dead')
    assert [r['id'] for r in dead] == [ids['dead']]

    completed = await journal.get_mem0_intents(status='completed')
    assert [r['id'] for r in completed] == [ids['completed']]

    all_rows = await journal.get_mem0_intents()
    assert len(all_rows) == 4


@pytest.mark.asyncio
async def test_get_incomplete_returns_only_pending(journal):
    """get_incomplete_mem0_intents() excludes completed/failed/dead rows."""
    pending_id = str(uuid.uuid4())
    await journal.log_mem0_intent(
        intent_id=pending_id,
        write_op_id=str(uuid.uuid4()),
        content='pending-one',
        metadata={},
        payload_digest='d',
    )
    for status in ['completed', 'failed', 'dead']:
        iid = str(uuid.uuid4())
        await journal.log_mem0_intent(
            intent_id=iid,
            write_op_id=str(uuid.uuid4()),
            content=f'c-{status}',
            metadata={},
            payload_digest='d',
        )
        await journal.resolve_mem0_intent(iid, status)

    incomplete = await journal.get_incomplete_mem0_intents()
    assert len(incomplete) == 1
    assert incomplete[0]['id'] == pending_id
    assert incomplete[0]['status'] == 'pending'


@pytest.mark.asyncio
async def test_prune_mem0_intents_ages_out_terminal_rows(journal):
    """prune deletes old completed/failed; preserves dead + recent + pending."""

    async def _seed(status, resolved_at=None):
        iid = str(uuid.uuid4())
        await journal.log_mem0_intent(
            intent_id=iid,
            write_op_id=str(uuid.uuid4()),
            content=f'c-{status}',
            metadata={},
            payload_digest='d',
        )
        if status != 'pending':
            await journal.resolve_mem0_intent(iid, status)
            if resolved_at is not None:
                # Backdate resolved_at to simulate an aged terminal row.
                await journal._db.execute(
                    'UPDATE mem0_intents SET resolved_at = ? WHERE id = ?',
                    (resolved_at, iid),
                )
                await journal._db.commit()
        return iid

    old = '2000-01-01T00:00:00+00:00'
    old_completed = await _seed('completed', resolved_at=old)
    old_failed = await _seed('failed', resolved_at=old)
    old_dead = await _seed('dead', resolved_at=old)
    recent_completed = await _seed('completed')  # resolved_at = now
    pending = await _seed('pending')

    deleted = await journal.prune_mem0_intents(older_than_days=7)
    assert deleted == 2  # old completed + old failed only

    remaining = {r['id'] for r in await journal.get_mem0_intents()}
    assert old_completed not in remaining
    assert old_failed not in remaining
    # dead-letter preserved regardless of age (manual-replay signal)
    assert old_dead in remaining
    # recent terminal + pending untouched
    assert recent_completed in remaining
    assert pending in remaining


@pytest.mark.asyncio
async def test_prune_mem0_intents_can_include_dead_explicitly(journal):
    """Overriding statuses to include 'dead' ages out old dead-letters too."""
    iid = str(uuid.uuid4())
    await journal.log_mem0_intent(
        intent_id=iid,
        write_op_id=str(uuid.uuid4()),
        content='c',
        metadata={},
        payload_digest='d',
    )
    await journal.resolve_mem0_intent(iid, 'dead', reason='unknown outcome')
    await journal._db.execute(
        'UPDATE mem0_intents SET resolved_at = ? WHERE id = ?',
        ('2000-01-01T00:00:00+00:00', iid),
    )
    await journal._db.commit()

    # Default statuses preserve dead...
    assert await journal.prune_mem0_intents(older_than_days=7) == 0
    # ...explicit override ages it out.
    deleted = await journal.prune_mem0_intents(
        older_than_days=7, statuses=('completed', 'failed', 'dead')
    )
    assert deleted == 1
    assert await journal.get_mem0_intents(status='dead') == []


@pytest.mark.asyncio
async def test_prune_mem0_intents_never_raises(journal):
    """A prune hiccup must not crash startup — returns 0, no raise."""
    await journal.close()
    journal._db = None
    assert await journal.prune_mem0_intents() == 0


# ------------------------------------------------------------------
# idempotent_ops: client-supplied idempotency keys (task 2712)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_and_get_idempotent_result_roundtrip(journal):
    """record_idempotent_result then get_idempotent_result returns the exact dict."""
    result = {'success': True, 'x': 1}
    await journal.record_idempotent_result('op-1', 'update_task', result)
    got = await journal.get_idempotent_result('op-1')
    assert got == {'success': True, 'x': 1}


@pytest.mark.asyncio
async def test_get_idempotent_result_miss_returns_none(journal):
    """A never-recorded client_op_id resolves to None (no hit)."""
    assert await journal.get_idempotent_result('never-seen') is None


@pytest.mark.asyncio
async def test_record_idempotent_result_first_write_wins(journal):
    """Re-recording the same client_op_id keeps the FIRST result (INSERT OR IGNORE)."""
    await journal.record_idempotent_result('op-dup', 'update_task', {'v': 'first'})
    # A second record with the SAME key but a DIFFERENT result must not raise
    # and must not overwrite the first-recorded outcome.
    await journal.record_idempotent_result('op-dup', 'update_task', {'v': 'second'})
    got = await journal.get_idempotent_result('op-dup')
    assert got == {'v': 'first'}


@pytest.mark.asyncio
async def test_get_idempotent_result_never_raises(journal):
    """A read hiccup fails open (None), never raises — journal never-block style."""
    await journal.close()
    journal._db = None
    assert await journal.get_idempotent_result('op-x') is None


@pytest.mark.asyncio
async def test_record_idempotent_result_never_raises(journal):
    """A record hiccup is swallowed, not propagated."""
    await journal.close()
    journal._db = None
    # Should not raise
    await journal.record_idempotent_result('op-y', 'update_task', {'ok': True})


@pytest.mark.asyncio
async def test_prune_idempotent_ops_ages_out_old_rows(journal):
    """prune deletes rows older than the window; preserves recent rows."""
    # Recent row (created_at = now) stays; backdated row is aged out.
    await journal.record_idempotent_result('recent', 'update_task', {'ok': True})
    await journal.record_idempotent_result('old', 'update_task', {'ok': True})
    await journal._db.execute(
        'UPDATE idempotent_ops SET created_at = ? WHERE client_op_id = ?',
        ('2000-01-01T00:00:00+00:00', 'old'),
    )
    await journal._db.commit()

    deleted = await journal.prune_idempotent_ops(older_than_days=7)
    assert deleted == 1  # only the backdated row

    assert await journal.get_idempotent_result('old') is None
    assert await journal.get_idempotent_result('recent') == {'ok': True}


@pytest.mark.asyncio
async def test_prune_idempotent_ops_never_raises(journal):
    """A prune hiccup must not crash startup — returns 0, no raise."""
    await journal.close()
    journal._db = None
    assert await journal.prune_idempotent_ops() == 0


# --- write_ops(created_at) seekability (task 3304) -------------------------
#
# PREDICATE_SQL is the property under test in its minimal form: idx_wo_created
# serves the bare `created_at >= ?` range constraint, and nothing in the SELECT
# list or GROUP BY affects whether that constraint is seekable. It is the
# shape-independent floor and it cannot drift.
#
# The two constants after it are copied VERBATIM from the dashboard's consumers
# — `get_memory_timeseries` and `get_agent_breakdown` in
# `dashboard/src/dashboard/data/write_journal.py` (cited by FUNCTION NAME, never
# by line number: line numbers in another package go stale on the next edit
# there, which is exactly how this comment's first version broke). The two
# packages have separate venvs and cannot import each other, and the schema
# under test is produced by fused-memory, so the copy is deliberate.
#
# KNOWN GAP: nothing here detects the dashboard changing its SQL (e.g. adding a
# `project_id = ?` filter or a BETWEEN range) — these copies would keep testing
# the old shape and stay green while the real endpoint regressed. The test that
# would close it belongs on the dashboard side, where the real SQL is
# importable; that file is outside this task's lock, so it is filed as
# follow-up rather than faked here. PREDICATE_SQL bounds the damage: the index's
# own contract stays pinned regardless of what the consumers do.

PREDICATE_SQL = 'SELECT COUNT(*) FROM write_ops WHERE created_at >= ?'

TIMESERIES_SQL = (
    "SELECT strftime('%Y-%m-%dT%H:00', created_at) AS hour,"
    ' kind, COUNT(*) AS cnt'
    ' FROM write_ops WHERE created_at >= ?'
    ' GROUP BY hour, kind'
)

AGENT_BREAKDOWN_SQL = (
    "SELECT COALESCE(agent_id, 'unknown') AS agent, COUNT(*) AS cnt"
    ' FROM write_ops WHERE created_at >= ?'
    ' GROUP BY agent ORDER BY cnt DESC'
)

SEEKABLE_QUERIES = (
    ('predicate only', PREDICATE_SQL),
    ('get_memory_timeseries', TIMESERIES_SQL),
    ('get_agent_breakdown', AGENT_BREAKDOWN_SQL),
)


async def _assert_created_at_seekable(db, *, context: str) -> None:
    """Assert every SEEKABLE_QUERIES shape range-SEEKs on created_at under *db*.

    Asserts the PROPERTY the acceptance names (a seekable range constraint),
    never the index NAME — see
    test_created_at_range_is_seekable_for_dashboard_queries for why. Shared by
    the fresh-schema test and the legacy-upgrade test so both prove the same
    thing: an index that exists but is unusable is not the deliverable.
    """
    since = '2026-07-30T00:00:00+00:00'
    for label, sql in SEEKABLE_QUERIES:
        # EXPLAIN QUERY PLAN rows are (id, parent, notused, detail).
        async with db.execute(f'EXPLAIN QUERY PLAN {sql}', (since,)) as cursor:
            plan = ' '.join(row[3] for row in await cursor.fetchall())

        assert 'SEARCH' in plan, f'{context}/{label}: expected a range seek, got: {plan}'
        assert 'created_at>?' in plan, (
            f'{context}/{label}: created_at must be the seek constraint, got: {plan}'
        )
        assert 'SCAN' not in plan, (
            f'{context}/{label}: still full-scanning write_ops: {plan}'
        )


@pytest.mark.asyncio
async def test_schema_creates_idx_wo_created(tmp_path):
    """SCHEMA_SQL creates idx_wo_created with created_at as the LEADING column.

    The leading-column assertion is the substance of this test. Five existing
    write_ops indexes already MENTION created_at, and every one of them is
    useless to a bare ``WHERE created_at >= ?`` precisely because created_at is
    not first. Both the name and the position are mandated — the sidecar
    delivered_check greps for ``idx_wo_created``.
    """
    j = WriteJournal(tmp_path / 'idx_test')
    await j.initialize()
    try:
        db = j._require_db()

        # PRAGMA index_list -> (seq, name, unique, origin, partial)
        async with db.execute("PRAGMA index_list('write_ops')") as cursor:
            names = {row[1] for row in await cursor.fetchall()}
        assert 'idx_wo_created' in names, f'idx_wo_created missing; have {sorted(names)}'

        # PRAGMA index_info -> (seqno, cid, name)
        async with db.execute("PRAGMA index_info('idx_wo_created')") as cursor:
            info = list(await cursor.fetchall())
        columns = [row[2] for row in sorted(info, key=lambda r: r[0])]
        assert columns[0] == 'created_at', f'created_at must lead; got {columns}'
    finally:
        await j.close()


@pytest.mark.asyncio
async def test_created_at_range_is_seekable_for_dashboard_queries(tmp_path):
    """The dashboard's bare ``created_at >= ?`` filter must range-SEEK, not SCAN.

    initialize() runs SCHEMA_SQL *and* _migrate(), so the DB under test carries
    the real six-index set (including idx_wo_kind_time / idx_wo_agent_time) that
    the live journal has — not a reduced test schema.

    This asserts the PROPERTY the acceptance names (a seekable range constraint),
    never the index NAME. Measured by the architect on sqlite 3.45.1 and 3.50.4:
    once ``sqlite_stat1`` exists (i.e. after any ANALYZE) the planner reaches the
    same range seek through a skip-scan on a pre-existing composite —
    ``SEARCH ... COVERING INDEX idx_wo_kind_time (ANY(kind) AND created_at>?)`` —
    rather than through idx_wo_created. Seekability still holds there, so a test
    pinning the index name in the query plan would go red on a change that broke
    nothing. The mandated name is pinned separately, and unambiguously, by
    test_schema_creates_idx_wo_created's PRAGMA assertions.

    Deliberately NOT asserted: ``get_operations_breakdown`` still plans as
    ``SCAN write_ops USING INDEX idx_wo_operation`` with idx_wo_created present,
    and that is expected, not a failure. Pinning "stays SCAN" would freeze
    planner behaviour we do not want and would fail the day SQLite improves. The
    measured plans and timings behind that live at the query itself
    (``get_operations_breakdown`` in dashboard/src/dashboard/data/write_journal.py)
    and, authoritatively, in the "Note on α's corrected signal" amendment in
    plans/dashboard-availability-prd.md — not restated here.
    """
    j = WriteJournal(tmp_path / 'plan_test')
    await j.initialize()
    try:
        await _assert_created_at_seekable(j._require_db(), context='fresh schema')
    finally:
        await j.close()


@pytest.mark.asyncio
async def test_existing_db_gains_idx_wo_created_on_initialize(tmp_path):
    """An existing pre-change journal gains a WORKING idx_wo_created on startup.

    This is the deployment claim: initialize() runs ``executescript(SCHEMA_SQL)``
    unconditionally on every start and the DDL is IF NOT EXISTS, so the SCHEMA_SQL
    placement ALONE upgrades the live 7 GB journal at the next fused-memory start,
    with no ``_migrate()`` entry needed.

    Seeds the full pre-change write_ops schema — current column set plus all five
    legacy indexes and no idx_wo_created — then asserts the index appears, the
    seeded rows survive, AND the upgraded DB actually range-seeks (via the shared
    _assert_created_at_seekable helper). The seek assertion is the point: name
    presence alone would pass while the real journal still scanned — e.g. a
    _migrate() reordering, or a legacy column affinity the index cannot serve —
    so the DB proven seekable must be the legacy-upgraded one, not only a fresh
    one.
    """
    import aiosqlite

    data_dir = tmp_path / 'legacy'
    data_dir.mkdir()
    db_path = data_dir / 'write_journal.db'

    pre_change_schema = """
    CREATE TABLE write_ops (
        id TEXT PRIMARY KEY,
        causation_id TEXT,
        source TEXT,
        provenance TEXT DEFAULT 'original',
        operation TEXT,
        project_id TEXT,
        agent_id TEXT,
        session_id TEXT,
        kind TEXT NOT NULL DEFAULT 'write',
        params TEXT DEFAULT '{}',
        result_summary TEXT,
        success INTEGER DEFAULT 1,
        error TEXT,
        created_at TEXT NOT NULL
    );
    CREATE INDEX idx_wo_causation ON write_ops(causation_id);
    CREATE INDEX idx_wo_project_time ON write_ops(project_id, created_at);
    CREATE INDEX idx_wo_operation ON write_ops(operation);
    CREATE INDEX idx_wo_kind_time ON write_ops(kind, created_at);
    CREATE INDEX idx_wo_agent_time ON write_ops(agent_id, created_at);
    """
    async with aiosqlite.connect(str(db_path)) as db:
        await db.executescript(pre_change_schema)
        await db.executemany(
            'INSERT INTO write_ops (id, operation, created_at) VALUES (?, ?, ?)',
            [
                ('legacy-1', 'add_memory', '2026-07-29T12:00:00+00:00'),
                ('legacy-2', 'search', '2026-07-30T12:00:00+00:00'),
            ],
        )
        await db.commit()

    j = WriteJournal(data_dir)
    await j.initialize()
    try:
        db_inner = j._require_db()

        async with db_inner.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_wo_created'"
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None, 'idx_wo_created should be present after initialize()'

        async with db_inner.execute('SELECT id FROM write_ops ORDER BY id') as cursor:
            ids = [r[0] for r in await cursor.fetchall()]
        assert ids == ['legacy-1', 'legacy-2'], f'seeded rows must survive; got {ids}'

        # The index must be USABLE on the upgraded DB, not merely present.
        await _assert_created_at_seekable(db_inner, context='legacy upgrade')
    finally:
        await j.close()


# ------------------------------------------------------------------
# Terminal queue outcome (task 3582)
#
# `write_ops.success` records that the ENQUEUE was accepted; the durable
# queue's eventual terminal state (completed / dead) was never written back,
# so a row for a write with a 0% landing rate was byte-for-byte
# indistinguishable from a row for a write that landed.
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_ops_has_terminal_columns(journal):
    """A fresh DB carries the terminal_* columns, NULL at Layer-1 log time."""
    db = journal._require_db()
    async with db.execute('PRAGMA table_info(write_ops)') as cursor:
        cols = {row[1] for row in await cursor.fetchall()}
    assert 'terminal_status' in cols
    assert 'terminal_at' in cols
    assert 'terminal_error' in cols

    op_id = str(uuid.uuid4())
    await journal.log_write_op(
        write_op_id=op_id,
        operation='add_episode',
        project_id='test-project',
    )
    ops = await journal.get_ops_since('2000-01-01T00:00:00')
    row = next(o for o in ops if o['id'] == op_id)
    # Terminal state is UNKNOWN at log time — NULL, not False.
    assert row['terminal_status'] is None
    assert row['terminal_at'] is None
    assert row['terminal_error'] is None
    # ...while `success` keeps its existing meaning: the enqueue was accepted.
    assert row['success'] == 1


@pytest.mark.asyncio
async def test_existing_db_gains_terminal_columns_on_initialize(tmp_path):
    """An existing journal gains terminal_* via ALTER, with no backfill."""
    import aiosqlite

    data_dir = tmp_path / 'terminal_migrate'
    data_dir.mkdir()
    db_path = data_dir / 'write_journal.db'

    pre_change_schema = """
    CREATE TABLE write_ops (
        id TEXT PRIMARY KEY,
        causation_id TEXT,
        source TEXT,
        provenance TEXT DEFAULT 'original',
        operation TEXT,
        project_id TEXT,
        agent_id TEXT,
        session_id TEXT,
        kind TEXT NOT NULL DEFAULT 'write',
        params TEXT DEFAULT '{}',
        result_summary TEXT,
        success INTEGER DEFAULT 1,
        error TEXT,
        created_at TEXT NOT NULL
    );
    """
    async with aiosqlite.connect(str(db_path)) as db:
        await db.executescript(pre_change_schema)
        await db.execute(
            'INSERT INTO write_ops (id, operation, success, created_at) '
            'VALUES (?, ?, ?, ?)',
            ('legacy-terminal-1', 'add_episode', 1, '2026-07-29T12:00:00+00:00'),
        )
        await db.commit()

    j = WriteJournal(data_dir)
    await j.initialize()
    # Migration must be idempotent — a second initialize() must not blow up on
    # a duplicate ALTER TABLE ADD COLUMN.
    await j.initialize()
    try:
        db_inner = j._require_db()
        async with db_inner.execute('PRAGMA table_info(write_ops)') as cursor:
            cols = {row[1] for row in await cursor.fetchall()}
        assert 'terminal_status' in cols
        assert 'terminal_at' in cols
        assert 'terminal_error' in cols

        async with db_inner.execute(
            'SELECT operation, success, terminal_status, terminal_at, terminal_error '
            "FROM write_ops WHERE id = 'legacy-terminal-1'"
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None, 'pre-existing row must survive the migration'
        assert row[0] == 'add_episode'
        assert row[1] == 1
        # No backfill: a historical row's terminal outcome is genuinely unknown.
        assert row[2] is None
        assert row[3] is None
        assert row[4] is None
    finally:
        await j.close()


@pytest.mark.asyncio
async def test_get_write_op_roundtrip(journal):
    """The no-join audit read: one write_ops row by id, or None."""
    op_id = str(uuid.uuid4())
    await journal.log_write_op(
        write_op_id=op_id,
        operation='add_episode',
        project_id='test-project',
        agent_id='claude-interactive',
    )

    row = await journal.get_write_op(op_id)
    assert row is not None
    assert row['id'] == op_id
    assert row['operation'] == 'add_episode'
    assert row['success'] == 1
    assert 'terminal_status' in row
    assert row['terminal_status'] is None

    assert await journal.get_write_op('no-such-id') is None
