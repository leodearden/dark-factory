"""MemoryService.initialize_for_reads opens only what the read paths need.

The pins are the service's public attributes and the real filesystem under
``queue.data_dir``; nothing the method constructs is patched.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from _fm_helpers import MockEdge

from fused_memory.models.scope import Scope
from fused_memory.services.memory_service import MemoryService
from fused_memory.services.planned_episode_registry import PlannedEpisodeRegistry


@pytest_asyncio.fixture
async def svc(mock_config):
    service = MemoryService(mock_config)
    service.graphiti = MagicMock()
    service.graphiti.initialize = AsyncMock()
    service.graphiti.close = AsyncMock()
    service.mem0 = MagicMock()
    service.mem0.close = AsyncMock()
    yield service
    await service.close()


async def _seed_planned_registry(data_dir: Path) -> None:
    writer = PlannedEpisodeRegistry(data_dir=data_dir)
    await writer.initialize()
    await writer.register('ep-1', 'proj')
    await writer.close()


@pytest.mark.asyncio
async def test_graph_maintenance_is_skipped(svc):
    await svc.initialize_for_reads()

    svc.graphiti.initialize.assert_awaited_once_with(skip_maintenance=True)


@pytest.mark.asyncio
async def test_no_queue_dir_state_is_created_when_none_exists(svc, mock_config, caplog):
    data_dir = Path(mock_config.queue.data_dir)
    assert not data_dir.exists()

    with caplog.at_level(logging.WARNING):
        await svc.initialize_for_reads()

    assert svc.durable_queue is None
    assert svc.overreach_shadow is None
    assert svc.planned_episode_registry is None
    assert not data_dir.exists()
    assert any(
        record.levelno == logging.WARNING and str(data_dir) in record.getMessage()
        for record in caplog.records
    ), [record.getMessage() for record in caplog.records]


@pytest.mark.asyncio
async def test_a_populated_queue_dir_gains_no_queue_or_sidecar(svc, mock_config):
    data_dir = Path(mock_config.queue.data_dir)
    await _seed_planned_registry(data_dir)

    await svc.initialize_for_reads()

    assert sorted(p.name for p in data_dir.glob('*.db')) == ['planned_episodes.db']
    assert svc.durable_queue is None
    registry = svc.planned_episode_registry
    assert registry is not None
    assert await registry.get_planned_uuids('proj') == {'ep-1'}
    with pytest.raises(sqlite3.OperationalError):
        await registry.register('ep-2', 'proj')


@pytest.mark.asyncio
async def test_search_and_count_run_on_the_read_open(svc, mock_config):
    await _seed_planned_registry(Path(mock_config.queue.data_dir))
    svc.graphiti.search = AsyncMock(return_value=[
        MockEdge(fact='seen only in a plan', uuid='edge-planned', episodes=['ep-1']),
        MockEdge(fact='seen in an observation', uuid='edge-observed', episodes=['ep-2']),
    ])
    svc.mem0.search = AsyncMock(return_value={'results': [
        {'id': 'mem-1', 'memory': 'a mem0 fact', 'score': 0.9, 'metadata': {}},
    ]})
    svc.mem0.count_by_metadata = AsyncMock(return_value=3)

    await svc.initialize_for_reads()
    results = await svc.search('fact', project_id='proj', stores=['graphiti', 'mem0'])
    count = await svc.count_memories_by_metadata('proj', {'category': 'procedural_knowledge'})

    assert not results.degraded, results.failure_diagnostics
    assert sorted(r.id for r in results) == ['edge-observed', 'mem-1']
    assert count == 3
    svc.mem0.count_by_metadata.assert_awaited_once_with(
        Scope(project_id='proj'), {'category': 'procedural_knowledge'},
    )


@pytest.mark.asyncio
async def test_a_wired_registry_is_kept(svc, mock_config):
    data_dir = Path(mock_config.queue.data_dir)
    wired = PlannedEpisodeRegistry(data_dir=data_dir)
    await wired.initialize()
    svc.set_planned_registry(wired)

    await svc.initialize_for_reads()

    assert svc.planned_episode_registry is wired
    await wired.register('ep-1', 'proj')
    assert await wired.get_planned_uuids('proj') == {'ep-1'}
