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
