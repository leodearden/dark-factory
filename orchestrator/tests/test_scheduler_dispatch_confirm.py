"""The scheduler re-reads and re-checks its dispatch winner after claiming it (task 11430).

A tick's task snapshot can be ~90 s old by the time a winner is claimed. Each
test models the store as two views: ``get_tasks`` answers the tick's snapshot,
while the public ``read_task`` / ``get_statuses`` answer the rows as they
stand now. A row that differs between the two is a write that landed mid-tick.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _orch_helpers import wait_responsive
from _park_test_helpers import make_task
from shared.task_read import TaskAbsent, TaskFound, TaskRead, TaskReadFailed

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.overrides import OverrideStore
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
        self._held_read: tuple[asyncio.Event, asyncio.Event] | None = None
        self.get_tasks = AsyncMock(side_effect=lambda **_: copy.deepcopy(self.snapshot))
        self.read_task = AsyncMock(side_effect=self._read_task)
        self.get_statuses = AsyncMock(side_effect=self._get_statuses)
        scheduler.get_tasks = self.get_tasks
        scheduler.read_task = self.read_task
        scheduler.get_statuses = self.get_statuses

    def hold_next_read(self) -> tuple[asyncio.Event, asyncio.Event]:
        """The next read_task sets the first event, then waits for the second before answering."""
        self._held_read = (asyncio.Event(), asyncio.Event())
        return self._held_read

    async def _read_task(self, task_id: str) -> TaskRead:
        if self._held_read is not None:
            entered, release = self._held_read
            self._held_read = None
            entered.set()
            await release.wait()
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


def _rig(tmp_path: Path, *, pinned: tuple[str, ...] = ()) -> _Rig:
    config = OrchestratorConfig(max_per_module=1)
    overrides = OverrideStore(tmp_path / 'scheduler_overrides.db')
    for task_id in pinned:
        overrides.set_override(str(config.project_root), task_id, pinned=True)
    store = EventStore(tmp_path / 'runs.db', 'run-1')
    scheduler = Scheduler(
        config,
        store,
        override_store=overrides,
        state_snapshot_path=tmp_path / 'scheduler_state.json',
        branch_work_reader=dict,
    )
    scheduler.finish_startup()
    return _Rig(scheduler, store)


def _with(task: dict, **fields: object) -> dict:
    return {**copy.deepcopy(task), **fields}


def _stamped(task: dict) -> dict:
    return _with(task, metadata={**task['metadata'], 'gate_escalated_at': STAMP})


def _unclaimed(rig: _Rig, *task_ids: str) -> bool:
    """None of *task_ids* is started or holds a lock-table entry, keyless entries included."""
    return not any(rig.scheduler.is_actively_held(tid) for tid in task_ids)


def _skipped(rig: _Rig, task_id: str) -> bool:
    return task_id in rig.scheduler.get_state_snapshot().skip_counts or any(
        event['task_id'] == task_id for event in rig.store.fetch_events_by_type(EventType.task_skipped)
    )


def _pure_gate(tid: str) -> dict:
    """A pending pure gate with no milestone: it dispatches through the scored path, holding no module lock."""
    return {
        'id': tid,
        'title': f'Gate {tid}',
        'status': 'pending',
        'priority': 'low',
        'dependencies': [],
        'metadata': {'task_kind': 'deterministic', 'always_escalates': True},
    }


def _open_pure_gate(tid: str) -> dict:
    """A timed gate (orchestrator/timed_gates.py) whose dated milestone opened an hour ago."""
    gate = _pure_gate(tid)
    gate['metadata']['milestone'] = {'mode': 'dated', 'at': (datetime.now(UTC) - timedelta(hours=1)).isoformat()}
    return gate


A = make_task('A', 'high', ['a.py'])
B = make_task('B', 'medium', ['a.py'])
C = make_task('C', 'medium', ['c.py'])
P = make_task('P', 'high', ['p.py'])
T = _open_pure_gate('500')


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
    assert _unclaimed(rig, 'A')

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
    assert _unclaimed(rig, 'A')


async def test_a_top_dropped_at_confirmation_frees_its_claim_and_is_not_passed_over(tmp_path):
    rig = _rig(tmp_path)
    _Board(rig.scheduler, snapshot=[A, B], rows=[_with(A, status='blocked'), B])

    result = await rig.scheduler.acquire_next()

    assert result is not None and result.task_id == 'B'
    assert _unclaimed(rig, 'A')
    assert not _skipped(rig, 'A')


async def test_a_winner_the_store_no_longer_holds_yields_to_the_next_candidate(tmp_path):
    rig = _rig(tmp_path)
    _Board(rig.scheduler, snapshot=[A, B], rows=[B])

    result = await rig.scheduler.acquire_next()

    assert result is not None and result.task_id == 'B'
    assert _unclaimed(rig, 'A')


async def test_an_unreadable_store_stops_dispatch_for_the_tick(tmp_path):
    rig = _rig(tmp_path)
    board = _Board(rig.scheduler, snapshot=[A, C], rows=[A, C])
    board.unreadable = True

    assert await rig.scheduler.acquire_next() is None
    assert board.read_task.await_count == 1
    assert _unclaimed(rig, 'A', 'C')

    board.unreadable = False
    first = await rig.scheduler.acquire_next()
    second = await rig.scheduler.acquire_next()

    assert {first and first.task_id, second and second.task_id} == {'A', 'C'}


async def test_a_pin_the_store_has_blocked_yields_to_an_unpinned_candidate(tmp_path):
    rig = _rig(tmp_path, pinned=('P',))
    _Board(rig.scheduler, snapshot=[P, C], rows=[_with(P, status='blocked'), C])

    result = await rig.scheduler.acquire_next()

    assert result is not None and result.task_id == 'C'
    assert _unclaimed(rig, 'P')
    assert not _skipped(rig, 'P')


async def test_a_pin_on_an_unreadable_store_stops_dispatch_for_the_tick(tmp_path):
    rig = _rig(tmp_path, pinned=('P',))
    board = _Board(rig.scheduler, snapshot=[P, C], rows=[P, C])
    board.unreadable = True

    assert await rig.scheduler.acquire_next() is None
    assert board.read_task.await_count == 1
    assert _unclaimed(rig, 'P', 'C')


async def test_a_timed_gate_that_gained_a_pending_dependency_is_not_dispatched(tmp_path):
    rig = _rig(tmp_path)
    _Board(
        rig.scheduler,
        snapshot=[T],
        rows=[_with(T, dependencies=['9']), make_task('9', 'high', ['n.py'])],
        statuses={'9': 'pending'},
    )

    assert await rig.scheduler.acquire_next() is None
    assert _unclaimed(rig, T['id'])


async def test_a_pure_gate_that_gained_a_pending_dependency_leaves_no_claim(tmp_path):
    rig = _rig(tmp_path)
    gate = _pure_gate('600')
    _Board(
        rig.scheduler,
        snapshot=[gate],
        rows=[_with(gate, dependencies=['9']), make_task('9', 'high', ['n.py'])],
        statuses={'9': 'pending'},
    )

    assert await rig.scheduler.acquire_next() is None
    assert _unclaimed(rig, '600')


async def test_a_pin_dispatch_hands_out_the_store_s_row(tmp_path):
    rig = _rig(tmp_path, pinned=('P',))
    _Board(rig.scheduler, snapshot=[P, C], rows=[_stamped(P), C])

    result = await rig.scheduler.acquire_next()

    assert result is not None and result.task_id == 'P'
    assert result.task['metadata']['gate_escalated_at'] == STAMP


async def test_a_timed_gate_dispatch_hands_out_the_store_s_row(tmp_path):
    rig = _rig(tmp_path)
    _Board(rig.scheduler, snapshot=[T], rows=[_stamped(T)])

    result = await rig.scheduler.acquire_next()

    assert result is not None and result.task_id == T['id']
    assert result.task['metadata']['gate_escalated_at'] == STAMP


def _acquired(rig: _Rig, task_id: str) -> int:
    return sum(1 for event in rig.store.fetch_events_by_type(EventType.lock_acquired) if event['task_id'] == task_id)


def _assert_not_passed_over(rig: _Rig, task_id: str) -> None:
    snapshot = rig.scheduler.get_state_snapshot()
    assert task_id not in snapshot.skip_counts
    assert task_id not in snapshot.parks
    assert all(event['task_id'] != task_id for event in rig.store.fetch_events_by_type(EventType.park_refused))


async def _race(
    rig: _Rig, board: _Board, *, meanwhile: Callable[[], None] = lambda: None,
) -> tuple[str | None, str | None]:
    """Pass A suspends in its winner's confirmation read; *meanwhile* runs, then pass B runs whole; then A resumes."""
    entered, release = board.hold_next_read()
    pass_a = asyncio.create_task(rig.scheduler.acquire_next())
    await wait_responsive(entered.wait(), label="pass A reaches its winner's confirmation read")
    meanwhile()
    b = await rig.scheduler.acquire_next()
    release.set()
    a = await wait_responsive(pass_a, label='pass A finishes after the release')
    return (a.task_id if a else None), (b.task_id if b else None)


async def test_a_top_started_while_the_scan_was_confirming_is_not_passed_over(tmp_path):
    rig = _rig(tmp_path)
    holder = make_task('H', 'critical', ['m1.py'])
    top, confirming, last = (
        make_task('X', 'high', ['m1.py']),
        make_task('T', 'medium', ['m2.py']),
        make_task('W', 'low', ['m3.py']),
    )
    board = _Board(
        rig.scheduler,
        snapshot=[holder],
        rows=[holder, top, _with(confirming, status='blocked'), last],
    )
    held = await rig.scheduler.acquire_next()
    assert held is not None and held.task_id == 'H', 'premise: H holds m1.py'
    board.snapshot = [_with(holder, status='in-progress'), top, confirming, last]

    a, b = await _race(rig, board, meanwhile=lambda: rig.scheduler.release('H'))

    assert (b, a) == ('X', 'W')
    assert _acquired(rig, 'X') == 1
    _assert_not_passed_over(rig, 'X')


async def test_a_pin_started_while_an_earlier_pin_was_confirming_is_not_lock_blocked(tmp_path):
    rig = _rig(tmp_path, pinned=('P1', 'P2'))
    first, second = make_task('P1', 'high', ['p1.py']), make_task('P2', 'high', ['p2.py'])
    board = _Board(rig.scheduler, snapshot=[first, second], rows=[_with(first, status='blocked'), second])

    a, b = await _race(rig, board)

    assert b == 'P2'
    assert a != 'P2'
    assert _acquired(rig, 'P2') == 1
    snapshot = rig.scheduler.get_state_snapshot()
    assert 'P2' not in snapshot.parks
    assert 'P2' not in snapshot.pin_reservations
    assert all(event['task_id'] != 'P2' for event in rig.store.fetch_events_by_type(EventType.pin_blocked))
