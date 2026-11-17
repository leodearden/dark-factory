"""Events a dead predecessor drained but never wrote a run row for are re-buffered
at startup (task 9655).

The stores use the production layout: the journal and every event buffer share
one ``reconciliation.db``, and the journal creates its older ``event_buffer``
shape first. A second buffer named ``dead-predecessor`` stands in for the
hard-killed instance: it drains and then simply closes.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

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
from fused_memory.reconciliation.event_buffer import DrainedRun, EventBuffer
from fused_memory.reconciliation.journal import ReconciliationJournal
from fused_memory.reconciliation.predecessor_drain_recovery import (
    RecoveredDrain,
    recover_predecessor_drains,
)
from reconciliation.blocked_calls import block_first_call, wait_until_entered
from reconciliation.harness_fixtures import (
    make_harness,
    make_harness_config,
    make_memory_service,
    open_event_buffer,
    open_journal,
)
from reconciliation.legacy_drains import drain_unattributed

PROJECT_P = 'project-p'
PROJECT_Q = 'project-q'
DEAD_INSTANCE = 'dead-predecessor'
KILLED_RUN_ID = 'run-killed-before-start-run'
UNREADABLE_RUN_ID = 'run-a-row-that-fails-to-read'


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


async def _dead_predecessor_drains_two_events(stores: _Stores, *, holding_its_lock: bool) -> None:
    """The dead instance drains a chunk and is killed before it writes the run row."""
    if holding_its_lock:
        assert await stores.dead.mark_run_active(PROJECT_P)
    e1, e2 = await _push_events(stores.live, PROJECT_P, 2)
    assert await stores.dead.drain_by_ids(PROJECT_P, [e1, e2], run_id=KILLED_RUN_ID) == 2


@pytest.mark.asyncio
async def test_drained_runs_of_other_instances_names_each_run_with_the_instance_that_drained_it(
    stores,
):
    e1, e2, _e3 = await _push_events(stores.live, PROJECT_P, 3)
    await stores.dead.drain_by_ids(PROJECT_P, [e1, e2], run_id='run-dead-chunk')
    await stores.live.drain(PROJECT_P, run_id='run-live')
    await _push_events(stores.dead, PROJECT_Q, 1)
    assert drain_unattributed(stores.db_path, PROJECT_Q) == 1

    assert await stores.live.drained_runs_of_other_instances() == [
        DrainedRun(project_id=PROJECT_P, run_id='run-dead-chunk', instance_id=DEAD_INSTANCE),
    ]
    assert await stores.dead.drained_runs_of_other_instances() == [
        DrainedRun(PROJECT_P, 'run-live', 'live'),
    ]

    await stores.live.restore_drained(PROJECT_P, run_id='run-dead-chunk')

    assert await stores.live.drained_runs_of_other_instances() == []


@pytest.mark.asyncio
async def test_drain_oldest_chunk_names_its_own_drain_on_a_row_an_earlier_drain_restored(stores):
    e1, e2 = await _push_events(stores.live, PROJECT_P, 2)
    await stores.dead.drain_by_ids(PROJECT_P, [e1, e2], run_id='run-restored')
    await stores.dead.restore_drained(PROJECT_P, run_id='run-restored')

    await stores.live.drain_oldest_chunk(PROJECT_P, limit=10, run_id='run-live-chunk')

    assert await stores.live.drained_runs_of_other_instances() == []
    assert await stores.dead.drained_runs_of_other_instances() == [
        DrainedRun(PROJECT_P, 'run-live-chunk', 'live'),
    ]


@pytest.mark.asyncio
async def test_recover_predecessor_drains_rebuffers_a_dead_predecessors_runless_drain_and_releases_its_lock(
    stores, caplog,
):
    caplog.set_level(
        logging.INFO, logger='fused_memory.reconciliation.predecessor_drain_recovery',
    )
    await _dead_predecessor_drains_two_events(stores, holding_its_lock=True)

    recovered = await recover_predecessor_drains(stores.journal, stores.live)

    assert recovered == [RecoveredDrain(
        drained=DrainedRun(PROJECT_P, KILLED_RUN_ID, DEAD_INSTANCE),
        restored=2,
        lock_released=True,
    )]
    assert await stores.live.count_buffered(PROJECT_P) == 2
    assert await stores.live.get_lock_holder_instance_id(PROJECT_P) is None
    assert [
        r for r in caplog.records
        if r.name == 'fused_memory.reconciliation.predecessor_drain_recovery'
        and r.levelno == logging.INFO
        and r.getMessage() == 'reconciliation.predecessor_drain_recovered'
    ], caplog.records


@pytest.mark.asyncio
@pytest.mark.parametrize('run_status', [RunStatus.running, RunStatus.completed])
async def test_recover_predecessor_drains_leaves_a_drain_whose_run_has_a_journal_row(
    stores, run_status,
):
    await _dead_predecessor_drains_two_events(stores, holding_its_lock=True)
    await stores.journal.start_run(ReconciliationRun(
        id=KILLED_RUN_ID,
        project_id=PROJECT_P,
        run_type=RunType.full,
        trigger_reason='unit-test',
        started_at=datetime.now(UTC),
        status=RunStatus.running,
        instance_id=DEAD_INSTANCE,
    ))
    if run_status is RunStatus.completed:
        await stores.journal.complete_run(KILLED_RUN_ID, 'completed')

    assert await recover_predecessor_drains(stores.journal, stores.live) == []
    assert await stores.live.count_buffered(PROJECT_P) == 0
    assert await stores.live.get_lock_holder_instance_id(PROJECT_P) == DEAD_INSTANCE


@pytest.mark.asyncio
async def test_recover_predecessor_drains_rebuffers_but_keeps_a_lock_another_instance_holds(
    stores,
):
    await _dead_predecessor_drains_two_events(stores, holding_its_lock=False)
    async with open_event_buffer(stores.db_path, instance_id='third-instance') as third:
        assert await third.mark_run_active(PROJECT_P)

        [recovered] = await recover_predecessor_drains(stores.journal, stores.live)

    assert (recovered.restored, recovered.lock_released) == (2, False)
    assert await stores.live.get_lock_holder_instance_id(PROJECT_P) == 'third-instance'


@pytest.mark.asyncio
async def test_recover_predecessor_drains_rebuffers_when_no_lock_is_held(stores):
    await _dead_predecessor_drains_two_events(stores, holding_its_lock=False)

    [recovered] = await recover_predecessor_drains(stores.journal, stores.live)

    assert (recovered.restored, recovered.lock_released) == (2, False)
    assert await stores.live.count_buffered(PROJECT_P) == 2


@pytest.mark.asyncio
async def test_recover_predecessor_drains_never_touches_the_callers_own_drain(stores):
    assert await stores.live.mark_run_active(PROJECT_P)
    await _push_events(stores.live, PROJECT_P, 2)
    await stores.live.drain(PROJECT_P, run_id='run-live-in-flight')

    assert await recover_predecessor_drains(stores.journal, stores.live) == []
    assert await stores.live.count_buffered(PROJECT_P) == 0
    assert await stores.live.get_lock_holder_instance_id(PROJECT_P) == stores.live.instance_id


@pytest.mark.asyncio
async def test_run_loop_recovers_a_dead_predecessors_runless_drain_before_its_first_reaper_tick(
    stores, monkeypatch,
):
    await _dead_predecessor_drains_two_events(stores, holding_its_lock=True)
    await stores.live.defer_write(
        PROJECT_P, 'deferred-during-dead-tenure', 'observations_and_summaries', {},
    )
    memory_service = make_memory_service()
    harness = make_harness(
        journal=stores.journal,
        event_buffer=stores.live,
        taskmaster=AsyncMock(),
        memory_service=memory_service,
        config=make_harness_config(judge_enabled=False),
        known_projects={PROJECT_P: '/tmp/test-project'},
    )
    harness.drain()
    monkeypatch.setattr(harness, '_start_escalation_server', AsyncMock())
    monkeypatch.setattr(harness, '_stop_escalation_server', AsyncMock())
    first_reaper_tick = block_first_call(monkeypatch, harness.journal, 'get_stale_runs')

    loop_task = asyncio.create_task(harness.run_loop())
    try:
        await wait_until_entered(first_reaper_tick, loop_task)

        assert await stores.live.count_buffered(PROJECT_P) == 2
        assert await stores.live.get_lock_holder_instance_id(PROJECT_P) is None
        memory_service.add_memory.assert_awaited()
    finally:
        loop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await loop_task


@pytest.mark.asyncio
async def test_recover_predecessor_drains_recovers_the_later_drains_when_one_fails(
    stores, monkeypatch, caplog,
):
    caplog.set_level(
        logging.WARNING, logger='fused_memory.reconciliation.predecessor_drain_recovery',
    )
    await _dead_predecessor_drains_two_events(stores, holding_its_lock=False)
    [e3] = await _push_events(stores.live, PROJECT_P, 1)
    await stores.dead.drain_by_ids(PROJECT_P, [e3], run_id=UNREADABLE_RUN_ID)
    assert [d.run_id for d in await stores.live.drained_runs_of_other_instances()] == [
        UNREADABLE_RUN_ID, KILLED_RUN_ID,
    ]
    read_run = stores.journal.get_run

    async def get_run_failing_for_one_row(run_id: str) -> ReconciliationRun | None:
        if run_id == UNREADABLE_RUN_ID:
            raise ValueError('run row does not parse')
        return await read_run(run_id)

    monkeypatch.setattr(stores.journal, 'get_run', get_run_failing_for_one_row)

    recovered = await recover_predecessor_drains(stores.journal, stores.live)

    assert [r.drained.run_id for r in recovered] == [KILLED_RUN_ID]
    assert await stores.live.count_buffered(PROJECT_P) == 2
    assert [
        r for r in caplog.records
        if r.name == 'fused_memory.reconciliation.predecessor_drain_recovery'
        and r.levelno == logging.WARNING
        and r.getMessage() == 'reconciliation.predecessor_drain_recovery_failed'
        and getattr(r, 'run_id', None) == UNREADABLE_RUN_ID
    ], caplog.records
