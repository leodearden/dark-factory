"""Tests for the write journal (SQLite persistence)."""

import asyncio
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
