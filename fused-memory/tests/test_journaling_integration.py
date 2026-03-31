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
    await j.open()
    yield j
    await j.close()


@pytest.fixture
def service(mock_config, write_journal):
    """MemoryService with mocked backends and real WriteJournal."""
    svc = MemoryService(mock_config)
    svc.graphiti = MagicMock()
    svc.graphiti.search = AsyncMock(return_value=[])
    svc.graphiti.search_nodes = AsyncMock(return_value=[])
    svc.graphiti.add_episode = AsyncMock(return_value=None)
    svc.graphiti.remove_episode = AsyncMock()
    svc.graphiti.remove_edge = AsyncMock()

    svc.mem0 = MagicMock()
    svc.mem0.search = AsyncMock(return_value={'results': []})
    svc.mem0.add = AsyncMock(return_value={'results': [{'id': 'mem0-1'}]})
    svc.mem0.delete = AsyncMock(return_value={'message': 'deleted'})

    svc.durable_queue = MagicMock()
    svc.durable_queue.enqueue = AsyncMock(return_value=1)
    svc.durable_queue.enqueue_batch = AsyncMock(return_value=[1])
    svc.durable_queue.close = AsyncMock()

    svc.set_write_journal(write_journal)
    return svc


@pytest.mark.asyncio
async def test_add_memory_logs_write_op_with_causation(service, write_journal):
    """MCP path: add_memory logs Layer 1 with causation_id.

    Layer 2 backend_op is logged later when the queue worker processes the
    item, so we only check Layer 1 here (Mem0 writes are now durable/deferred).
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

    # Verify causation_id was passed through to queue payload for deferred logging
    payload = service.durable_queue.enqueue.call_args[1]['payload']
    assert payload['_causation_id'] == cid


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
    from tests.conftest import MockAddEpisodeResult, MockEdge

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
        memory_id='test-id', store='mem0', project_id='test', causation_id=cid,
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
