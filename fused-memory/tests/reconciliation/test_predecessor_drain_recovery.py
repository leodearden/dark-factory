"""Events a dead predecessor drained but never wrote a run row for are re-buffered
at startup (task 9655).

The stores use the production layout: the journal and every event buffer share
one ``reconciliation.db``, and the journal creates its older ``event_buffer``
shape first. A second buffer named ``dead-predecessor`` stands in for the
hard-killed instance: it drains and then simply closes.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio

from fused_memory.models.reconciliation import EventSource, EventType, ReconciliationEvent
from fused_memory.reconciliation.event_buffer import DrainedRun, EventBuffer
from fused_memory.reconciliation.journal import ReconciliationJournal
from reconciliation.harness_fixtures import open_event_buffer, open_journal

PROJECT_P = 'project-p'
PROJECT_Q = 'project-q'
DEAD_INSTANCE = 'dead-predecessor'


@dataclass(frozen=True)
class _Stores:
    journal: ReconciliationJournal
    live: EventBuffer
    dead: EventBuffer
    db_path: Path


@asynccontextmanager
async def _open_production_stores(tmp_path: Path) -> AsyncIterator[_Stores]:
    db_path = tmp_path / 'reconciliation.db'
    async with (
        open_journal(tmp_path) as journal,
        open_event_buffer(db_path, instance_id='live') as live,
        open_event_buffer(db_path, instance_id=DEAD_INSTANCE) as dead,
    ):
        yield _Stores(journal=journal, live=live, dead=dead, db_path=db_path)


@pytest_asyncio.fixture
async def stores(tmp_path) -> AsyncIterator[_Stores]:
    async with _open_production_stores(tmp_path) as s:
        yield s


async def _push_events(buffer: EventBuffer, project_id: str, count: int) -> list[str]:
    ids = [str(uuid4()) for _ in range(count)]
    for event_id in ids:
        await buffer.push(ReconciliationEvent(
            id=event_id,
            type=EventType.episode_added,
            source=EventSource.agent,
            project_id=project_id,
            timestamp=datetime.now(UTC),
            payload={},
        ))
    return ids


@pytest.mark.asyncio
async def test_drained_runs_of_other_instances_names_each_run_with_the_instance_that_drained_it(
    stores,
):
    e1, e2, _e3 = await _push_events(stores.live, PROJECT_P, 3)
    await stores.dead.drain_by_ids(PROJECT_P, [e1, e2], run_id='run-dead-chunk')
    await stores.live.drain(PROJECT_P, run_id='run-live')
    await _push_events(stores.dead, PROJECT_Q, 1)
    await stores.live.drain(PROJECT_Q)

    assert await stores.live.drained_runs_of_other_instances() == [
        DrainedRun(project_id=PROJECT_P, run_id='run-dead-chunk', instance_id=DEAD_INSTANCE),
    ]
    assert await stores.dead.drained_runs_of_other_instances() == [
        DrainedRun(PROJECT_P, 'run-live', 'live'),
    ]

    await stores.live.restore_drained(PROJECT_P, run_id='run-dead-chunk')

    assert await stores.live.drained_runs_of_other_instances() == []
