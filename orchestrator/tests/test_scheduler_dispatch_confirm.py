"""The scheduler re-reads and re-checks its dispatch winner after claiming it (task 11430).

A tick's task snapshot can be ~90 s old by the time a winner is claimed. Each
test models the store as two views: ``get_tasks`` answers the tick's snapshot,
while the public ``read_task`` / ``get_statuses`` answer the rows as they
stand now. A row that differs between the two is a write that landed mid-tick.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _park_test_helpers import make_task
from shared.task_read import TaskAbsent, TaskFound, TaskRead, TaskReadFailed

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.scheduler import Scheduler

pytestmark = [
    pytest.mark.real_dispatch_row_refresh,
    pytest.mark.usefixtures('code_default_config'),
    pytest.mark.asyncio,
]

STAMP = '2026-11-24T00:00:00+00:00'


class _Board:
    """The task store: the tick's snapshot, and the rows and statuses as they stand now."""

    def __init__(
        self,
        scheduler: Scheduler,
        *,
        snapshot: list[dict],
        rows: list[dict],
        statuses: dict[str, str] | None = None,
    ) -> None:
        self.snapshot = copy.deepcopy(snapshot)
        self.rows = {str(row['id']): copy.deepcopy(row) for row in rows}
        self.statuses = dict(statuses or {})
        self.unreadable = False
        self.get_tasks = AsyncMock(side_effect=lambda **_: copy.deepcopy(self.snapshot))
        self.read_task = AsyncMock(side_effect=self._read_task)
        self.get_statuses = AsyncMock(side_effect=self._get_statuses)
        scheduler.get_tasks = self.get_tasks
        scheduler.read_task = self.read_task
        scheduler.get_statuses = self.get_statuses

    def _read_task(self, task_id: str) -> TaskRead:
        if self.unreadable:
            return TaskReadFailed('fused-memory unreachable')
        row = self.rows.get(task_id)
        return TaskAbsent() if row is None else TaskFound(copy.deepcopy(row))

    def _get_statuses(self, ids: list[str] | None = None) -> tuple[dict[str, str], Exception | None]:
        wanted = self.statuses if ids is None else {i: self.statuses[i] for i in ids if i in self.statuses}
        return dict(wanted), None


@dataclass(frozen=True)
class _Rig:
    scheduler: Scheduler
    store: EventStore


def _rig(tmp_path: Path) -> _Rig:
    store = EventStore(tmp_path / 'runs.db', 'run-1')
    scheduler = Scheduler(
        OrchestratorConfig(max_per_module=1),
        store,
        state_snapshot_path=tmp_path / 'scheduler_state.json',
        branch_work_reader=dict,
    )
    scheduler.finish_startup()
    return _Rig(scheduler, store)


def _with(task: dict, **fields: object) -> dict:
    return {**copy.deepcopy(task), **fields}


def _stamped(task: dict) -> dict:
    return _with(task, metadata={**task['metadata'], 'gate_escalated_at': STAMP})


def _holds_nothing(rig: _Rig, *task_ids: str) -> bool:
    return all(rig.scheduler.lock_table.held_modules(tid) == frozenset() for tid in task_ids)


def _skipped(rig: _Rig, task_id: str) -> bool:
    return task_id in rig.scheduler.get_state_snapshot().skip_counts or any(
        event['task_id'] == task_id for event in rig.store.fetch_events_by_type(EventType.task_skipped)
    )


A = make_task('A', 'high', ['a.py'])
B = make_task('B', 'medium', ['a.py'])
C = make_task('C', 'medium', ['c.py'])


async def test_a_dependency_added_mid_tick_holds_the_winner_until_it_is_done(tmp_path):
    rig = _rig(tmp_path)
    dep = make_task('9', 'high', ['n.py'])
    board = _Board(
        rig.scheduler,
        snapshot=[A],
        rows=[_with(A, dependencies=['9']), dep],
        statuses={'9': 'pending'},
    )

    assert await rig.scheduler.acquire_next() is None
    assert _holds_nothing(rig, 'A')

    board.rows['9']['status'] = 'done'
    board.statuses['9'] = 'done'
    board.snapshot = [copy.deepcopy(board.rows['A'])]
    result = await rig.scheduler.acquire_next()

    assert result is not None and result.task_id == 'A'


async def test_the_dispatched_row_is_the_store_s_row(tmp_path):
    rig = _rig(tmp_path)
    _Board(rig.scheduler, snapshot=[A], rows=[_stamped(A)])

    result = await rig.scheduler.acquire_next()

    assert result is not None and result.task_id == 'A'
    assert result.task['metadata']['gate_escalated_at'] == STAMP


async def test_fresh_dependency_statuses_outrank_the_snapshot_s(tmp_path):
    rig = _rig(tmp_path)
    dep = make_task('9', 'high', ['n.py'], status='done')
    waiting = _with(A, dependencies=['9'])
    _Board(rig.scheduler, snapshot=[waiting, dep], rows=[waiting, dep], statuses={'9': 'pending'})

    assert await rig.scheduler.acquire_next() is None
    assert _holds_nothing(rig, 'A')


async def test_a_top_dropped_at_confirmation_frees_its_claim_and_is_not_passed_over(tmp_path):
    rig = _rig(tmp_path)
    _Board(rig.scheduler, snapshot=[A, B], rows=[_with(A, status='blocked'), B])

    result = await rig.scheduler.acquire_next()

    assert result is not None and result.task_id == 'B'
    assert _holds_nothing(rig, 'A')
    assert not _skipped(rig, 'A')


async def test_a_winner_the_store_no_longer_holds_yields_to_the_next_candidate(tmp_path):
    rig = _rig(tmp_path)
    _Board(rig.scheduler, snapshot=[A, B], rows=[B])

    result = await rig.scheduler.acquire_next()

    assert result is not None and result.task_id == 'B'
    assert _holds_nothing(rig, 'A')


async def test_an_unreadable_store_stops_dispatch_for_the_tick(tmp_path):
    rig = _rig(tmp_path)
    board = _Board(rig.scheduler, snapshot=[A, C], rows=[A, C])
    board.unreadable = True

    assert await rig.scheduler.acquire_next() is None
    assert board.read_task.await_count == 1
    assert _holds_nothing(rig, 'A', 'C')

    board.unreadable = False
    first = await rig.scheduler.acquire_next()
    second = await rig.scheduler.acquire_next()

    assert {first and first.task_id, second and second.task_id} == {'A', 'C'}
