"""restore_unless_completed: one run's drained events go back to the buffer
unless the journal records that run as completed (task 6453)."""
from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

import pytest
import pytest_asyncio

from fused_memory.models.reconciliation import (
    EventSource,
    EventType,
    ReconciliationEvent,
    ReconciliationRun,
    RunStatus,
    RunType,
)
from fused_memory.reconciliation.drained_event_restore import (
    restore_unless_completed,
    restore_unless_completed_best_effort,
)
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.journal import ReconciliationJournal

PROJECT_ID = 'p'


@pytest_asyncio.fixture
async def journal(tmp_path):
    j = ReconciliationJournal(tmp_path / 'j')
    await j.initialize()
    yield j
    await j.close()


@pytest_asyncio.fixture
async def buffer(tmp_path):
    b = EventBuffer(db_path=tmp_path / 'eb.db')
    await b.initialize()
    yield b
    await b.close()


async def _journaled_run(journal: ReconciliationJournal, run_id: str, status: str) -> None:
    await journal.start_run(ReconciliationRun(
        id=run_id,
        project_id=PROJECT_ID,
        run_type=RunType.full,
        trigger_reason='t',
        started_at=datetime.now(UTC),
        status=RunStatus.running,
    ))
    if status != RunStatus.running:
        await journal.complete_run(run_id, status)


async def _drain_new_events(buffer: EventBuffer, count: int, run_id: str) -> None:
    ids = []
    for _ in range(count):
        event = ReconciliationEvent(
            id=str(uuid.uuid4()),
            type=EventType.episode_added,
            source=EventSource.agent,
            project_id=PROJECT_ID,
            timestamp=datetime.now(UTC),
            payload={},
        )
        await buffer.push(event)
        ids.append(event.id)
    assert await buffer.drain_by_ids(PROJECT_ID, ids, run_id=run_id) == count


@pytest.mark.asyncio
async def test_completed_run_events_stay_drained(journal, buffer):
    await _drain_new_events(buffer, 2, run_id='A')
    await _journaled_run(journal, 'A', RunStatus.completed)

    assert await restore_unless_completed(journal, buffer, PROJECT_ID, 'A') == 0

    assert len(await buffer.get_drained_events(PROJECT_ID, 'A')) == 2
    assert await buffer.count_buffered(PROJECT_ID) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'b_status',
    [RunStatus.failed, RunStatus.interrupted, RunStatus.running, None],
    ids=['failed', 'interrupted', 'running', 'no_journal_row'],
)
async def test_unfinished_run_restores_only_its_own_events(journal, buffer, b_status):
    await _drain_new_events(buffer, 2, run_id='A')
    await _journaled_run(journal, 'A', RunStatus.completed)
    await _drain_new_events(buffer, 1, run_id='B')
    if b_status is not None:
        await _journaled_run(journal, 'B', b_status)

    assert await restore_unless_completed(journal, buffer, PROJECT_ID, 'B') == 1

    assert await buffer.count_buffered(PROJECT_ID) == 1
    assert len(await buffer.get_drained_events(PROJECT_ID, 'A')) == 2


@pytest.mark.asyncio
async def test_best_effort_restore_applies_the_same_policy(journal, buffer):
    await _drain_new_events(buffer, 2, run_id='A')
    await _journaled_run(journal, 'A', RunStatus.completed)
    await _drain_new_events(buffer, 1, run_id='B')

    await restore_unless_completed_best_effort(journal, buffer, PROJECT_ID, 'A')
    await restore_unless_completed_best_effort(journal, buffer, PROJECT_ID, 'B')

    assert await buffer.count_buffered(PROJECT_ID) == 1
    assert len(await buffer.get_drained_events(PROJECT_ID, 'A')) == 2


@pytest.mark.asyncio
async def test_best_effort_restore_logs_a_failing_restore_instead_of_raising(
    journal, buffer, caplog, monkeypatch,
):
    async def locked_restore(project_id, run_id, *, include_unattributed=False):
        raise RuntimeError('database is locked')

    monkeypatch.setattr(buffer, 'restore_drained', locked_restore)

    with caplog.at_level(logging.ERROR):
        await restore_unless_completed_best_effort(journal, buffer, PROJECT_ID, 'B')

    [line] = [
        r.getMessage() for r in caplog.records
        if r.levelno == logging.ERROR and r.getMessage().startswith('restore_drained failed:')
    ]
    assert 'run_id=B' in line
