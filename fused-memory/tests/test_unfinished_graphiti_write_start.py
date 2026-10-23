"""MemoryService.earliest_unfinished_graphiti_write_start, read through a REAL
DurableWriteQueue whose backend writes are held open until released."""

from __future__ import annotations

import pytest
import pytest_asyncio
from _fm_held_writes import HeldWrites
from _fm_helpers import poll_until

from fused_memory.services.durable_queue import DurableWriteQueue
from fused_memory.services.memory_service import MemoryService

HELD_WRITE = 'ep'


@pytest.fixture
def held() -> HeldWrites:
    return HeldWrites()


@pytest_asyncio.fixture
async def service(mock_config, tmp_path, held):
    svc = MemoryService(mock_config)
    queue = DurableWriteQueue(
        data_dir=tmp_path / 'queue',
        execute_write=held.execute,
        write_timeout_seconds=5.0,
    )
    await queue.initialize()
    svc.durable_queue = queue
    try:
        yield svc
    finally:
        held.release_all()
        await queue.close()


async def _start_held_write(svc: MemoryService, held: HeldWrites) -> None:
    assert svc.durable_queue is not None
    await svc.durable_queue.enqueue(
        group_id='proj', operation='add_episode',
        payload={'content': 'x', 'group_id': 'proj', 'name': HELD_WRITE},
    )
    await poll_until(
        lambda: HELD_WRITE in held.entered_at, message='the write never began executing',
    )


@pytest.mark.asyncio
async def test_no_durable_queue_reports_none(service):
    service.durable_queue = None
    assert service.earliest_unfinished_graphiti_write_start('proj') is None


@pytest.mark.asyncio
async def test_executing_write_is_reported_for_its_project(service, held):
    await _start_held_write(service, held)

    start = service.earliest_unfinished_graphiti_write_start('proj')
    assert start is not None
    assert start == service.durable_queue.earliest_executing_start('proj')


@pytest.mark.asyncio
async def test_executing_write_is_not_reported_for_another_project(service, held):
    await _start_held_write(service, held)

    assert service.earliest_unfinished_graphiti_write_start('other_project') is None


@pytest.mark.asyncio
async def test_completed_write_is_no_longer_reported(service, held):
    await _start_held_write(service, held)
    held.release(HELD_WRITE)

    async def _completed():
        stats = await service.durable_queue.get_stats()
        return stats['counts'].get('completed', 0) >= 1

    await poll_until(_completed, message='the released write never completed')
    assert service.earliest_unfinished_graphiti_write_start('proj') is None
