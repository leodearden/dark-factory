"""Integration tests verifying causation_id flows through all paths."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from fused_memory.services.memory_service import MemoryService
from fused_memory.services.write_journal import WriteJournal


@pytest_asyncio.fixture
async def write_journal(tmp_path):
    j = WriteJournal(tmp_path / 'wj_integration')
    await j.initialize()
    yield j
    await j.close()


@pytest.fixture
def service(mock_config, write_journal):
    """MemoryService with mocked backends and real WriteJournal."""
    from _fm_helpers import install_identity_mocks

    svc = MemoryService(mock_config)
    svc.graphiti = MagicMock()
    svc.graphiti.search = AsyncMock(return_value=[])
    svc.graphiti.search_nodes = AsyncMock(return_value=[])
    svc.graphiti.add_episode = AsyncMock(return_value=None)
    svc.graphiti.remove_episode = AsyncMock()
    svc.graphiti.remove_edge = AsyncMock()
    install_identity_mocks(svc.graphiti)

    svc.mem0 = MagicMock()
    svc.mem0.search = AsyncMock(return_value={'results': []})
    svc.mem0.add = AsyncMock(return_value={'results': [{'id': 'mem0-1'}]})
    svc.mem0.delete = AsyncMock(return_value={'message': 'deleted'})
    # delete_memory's child gate (task 3197) awaits count_by_metadata on
    # every mem0 delete; the childless default models the live corpus leaf α
    # measured (`metadata.parent_id` has zero footprint).
    svc.mem0.count_by_metadata = AsyncMock(return_value=0)
    svc.mem0.scroll_by_metadata = AsyncMock(return_value=[])

    svc.durable_queue = MagicMock()
    svc.durable_queue.enqueue = AsyncMock(return_value=1)
    svc.durable_queue.enqueue_batch = AsyncMock(return_value=[1])
    svc.durable_queue.close = AsyncMock()

    svc.set_write_journal(write_journal)
    return svc


@pytest.mark.asyncio
async def test_add_memory_logs_write_op_with_causation(service, write_journal):
    """MCP path: add_memory logs Layer 1 with causation_id.

    Mem0 writes are now synchronous direct calls (not deferred via queue),
    so both Layer 1 (write_op) and Layer 2 (backend_op) are logged before
    add_memory returns.
    """
    cid = str(uuid.uuid4())
    await service.add_memory(
        content='Test fact',
        category='preferences_and_norms',
        project_id='test',
        causation_id=cid,
    )
    ops = await write_journal.get_ops_by_causation(cid)
    write_ops = [o for o in ops if o['layer'] == 'write_op']
    assert len(write_ops) == 1
    assert write_ops[0]['operation'] == 'add_memory'
    assert write_ops[0]['success'] == 1

    # Mem0 now calls directly — no queue item enqueued
    service.durable_queue.enqueue.assert_not_called()
    # mem0.add was called synchronously
    service.mem0.add.assert_called_once()


@pytest.mark.asyncio
async def test_add_episode_logs_write_op_with_causation(service, write_journal):
    """MCP path: add_episode logs Layer 1."""
    cid = str(uuid.uuid4())
    await service.add_episode(
        content='User discussed something',
        project_id='test',
        causation_id=cid,
    )
    ops = await write_journal.get_ops_by_causation(cid)
    write_ops = [o for o in ops if o['layer'] == 'write_op']
    assert len(write_ops) == 1
    assert write_ops[0]['operation'] == 'add_episode'


@pytest.mark.asyncio
async def test_add_episode_propagates_causation_to_queue(service, write_journal):
    """Causation_id is injected into the queue payload for later Layer 2 logging."""
    cid = str(uuid.uuid4())
    await service.add_episode(
        content='Test content',
        project_id='test',
        causation_id=cid,
    )
    payload = service.durable_queue.enqueue.call_args[1]['payload']
    assert payload['_causation_id'] == cid
    assert '_write_op_id' in payload


@pytest.mark.asyncio
async def test_graphiti_write_logs_backend_op(service, write_journal):
    """_execute_graphiti_write logs Layer 2 with causation from payload."""
    cid = str(uuid.uuid4())
    wid = str(uuid.uuid4())
    payload = {
        'name': 'test',
        'content': 'test content',
        'source': 'text',
        'group_id': 'test',
        'source_description': '',
        '_causation_id': cid,
        '_write_op_id': wid,
    }
    await service._execute_graphiti_write('add_episode', payload)
    backend_ops = await write_journal.get_backend_ops_for_write_op(wid)
    assert len(backend_ops) == 1
    assert backend_ops[0]['backend'] == 'graphiti'
    assert backend_ops[0]['causation_id'] == cid


@pytest.mark.asyncio
async def test_dual_write_callback_enqueues_with_causation(service, write_journal):
    """Dual-write callback batch-enqueues facts with causation_id in payload."""
    from _fm_helpers import MockAddEpisodeResult, MockEdge

    cid = str(uuid.uuid4())
    result = MockAddEpisodeResult(entity_edges=[
        MockEdge(fact='Always use type hints in Python'),
    ])

    payload = {
        'project_id': 'test',
        'agent_id': 'test-agent',
        '_causation_id': cid,
    }
    await service._dual_write_callback('dual_write_episode', result, payload)

    service.durable_queue.enqueue_batch.assert_called_once()
    batch = service.durable_queue.enqueue_batch.call_args[0][0]
    assert len(batch) == 1
    assert batch[0]['operation'] == 'mem0_classify_and_add'
    assert batch[0]['payload']['_causation_id'] == cid
    assert batch[0]['payload']['fact_text'] == 'Always use type hints in Python'


@pytest.mark.asyncio
async def test_search_logged_when_causation_present(service, write_journal):
    """Search is only logged when causation_id is non-None."""
    cid = str(uuid.uuid4())

    # Without causation — no log
    await service.search(query='test', project_id='test')
    ops = await write_journal.get_ops_since('2000-01-01T00:00:00')
    assert len(ops) == 0

    # With causation — logged
    await service.search(query='test', project_id='test', causation_id=cid)
    ops = await write_journal.get_ops_by_causation(cid)
    assert len(ops) == 1
    assert ops[0]['operation'] == 'search'


@pytest.mark.asyncio
async def test_delete_memory_logs_both_layers(service, write_journal):
    cid = str(uuid.uuid4())
    await service.delete_memory(
        memory_id='00000000-0000-4000-8000-00000000000b',
        store='mem0',
        project_id='test',
        causation_id=cid,
    )
    ops = await write_journal.get_ops_by_causation(cid)
    write_ops = [o for o in ops if o['layer'] == 'write_op']
    backend_ops = [o for o in ops if o['layer'] == 'backend_op']
    assert len(write_ops) == 1
    assert write_ops[0]['operation'] == 'delete_memory'
    assert len(backend_ops) == 1
    assert backend_ops[0]['backend'] == 'mem0'


@pytest.mark.asyncio
async def test_targeted_recon_source_tag(service, write_journal):
    """Targeted recon path tags source as 'targeted_recon'."""
    cid = str(uuid.uuid4())
    await service.add_memory(
        content='Task completed',
        category='observations_and_summaries',
        project_id='test',
        causation_id=cid,
        _source='targeted_recon',
    )
    ops = await write_journal.get_ops_by_causation(cid)
    write_ops = [o for o in ops if o['layer'] == 'write_op']
    assert len(write_ops) == 1
    assert write_ops[0]['source'] == 'targeted_recon'


# ------------------------------------------------------------------
# Terminal queue outcome write-back (task 3582)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_queue_terminal_outcome_stamps_journal(service, write_journal):
    """The queue's on_terminal hook lands on the write_ops row."""
    op_id = str(uuid.uuid4())
    await write_journal.log_write_op(write_op_id=op_id, operation='add_episode')

    await service._record_queue_terminal_outcome(
        op_id, 'dead', 'RuntimeError: boom'
    )

    row = await write_journal.get_write_op(op_id)
    assert row is not None
    assert row['terminal_status'] == 'dead'
    assert 'boom' in row['terminal_error']
    # "Enqueue accepted" is still true and still recorded.
    assert row['success'] == 1


@pytest.mark.asyncio
async def test_record_queue_terminal_outcome_without_journal_is_noop(mock_config):
    """The hook must resolve self._write_journal at CALL time, not capture it.

    server/main.py calls memory_service.initialize() — where the
    DurableWriteQueue is constructed — BEFORE set_write_journal(), so the
    journal is None when the queue (and hence the hook) is wired.
    """
    svc = MemoryService(mock_config)
    assert svc._write_journal is None
    # Must not raise.
    await svc._record_queue_terminal_outcome('W-none', 'completed', None)


@pytest.mark.asyncio
async def test_execute_mem0_write_journals_on_failure(service, write_journal):
    """A dead-lettering mem0_add must still leave a Layer-1 row behind.

    This used to journal only after a successful await, so a queued mem0_add
    that failed its way to 'dead' produced no write_ops row at all — the
    terminal write-back then had nothing well-formed to land on and the
    failure was invisible to the journal.
    """
    op_id = str(uuid.uuid4())
    service.mem0.add = AsyncMock(side_effect=RuntimeError('mem0 exploded'))

    with pytest.raises(RuntimeError):
        await service._execute_mem0_write(
            {
                'content': 'a fact that never lands',
                'project_id': 'test',
                '_write_op_id': op_id,
                'metadata': {'category': 'preferences_and_norms'},
            }
        )

    row = await write_journal.get_write_op(op_id)
    assert row is not None, 'a failed queued mem0 write must still be journaled'
    assert row['operation'] == 'add_memory'
    assert row['success'] == 0
    assert 'mem0 exploded' in row['error']

    # ...and the queue's terminal write-back now has a real row to complete.
    await service._record_queue_terminal_outcome(
        op_id, 'dead', 'RuntimeError: mem0 exploded'
    )
    row = await write_journal.get_write_op(op_id)
    assert row['terminal_status'] == 'dead'
    assert row['operation'] == 'add_memory'


@pytest.mark.asyncio
async def test_execute_mem0_write_still_journals_on_success(service, write_journal):
    """The success path keeps its Layer-1 row, result summary and all."""
    op_id = str(uuid.uuid4())

    await service._execute_mem0_write(
        {
            'content': 'a fact that lands',
            'project_id': 'test',
            '_write_op_id': op_id,
            'metadata': {'category': 'preferences_and_norms'},
        }
    )

    row = await write_journal.get_write_op(op_id)
    assert row is not None
    assert row['operation'] == 'add_memory'
    assert row['success'] == 1
    assert row['error'] is None
    assert row['result_summary'] is not None
