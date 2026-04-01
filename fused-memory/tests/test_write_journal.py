"""Tests for the write journal (SQLite persistence)."""

import asyncio
import uuid

import pytest
import pytest_asyncio

from fused_memory.services.write_journal import WriteJournal


@pytest_asyncio.fixture
async def journal(tmp_path):
    j = WriteJournal(tmp_path / 'test_wj')
    await j.open()
    yield j
    await j.close()


@pytest.mark.asyncio
async def test_initialize_creates_db(tmp_path):
    j = WriteJournal(tmp_path / 'init_test')
    await j.open()
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
    journal._conn = None
    # Should not raise
    await journal.log_write_op(
        write_op_id=str(uuid.uuid4()),
        operation='add_memory',
    )


@pytest.mark.asyncio
async def test_log_backend_op_never_raises(journal):
    await journal.close()
    journal._conn = None
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
    await j.open()

    # Check columns exist
    db = j._require_conn()
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
