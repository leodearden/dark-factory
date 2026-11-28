"""A main-health fix granted precedence dispatches first and parks behind a held lock (task 10399).

The grant pins the fix at the head of the override store's queue with
reserve_now armed.  The scheduler's existing phases do the rest: the pin phase
orders dispatch and reserves a lock-blocked head's modules at pin rank, the
reserve_now phase parks the fix on its first tick, and the override GC clears
the row once the fix lands or is cancelled.  A held lock is never preempted
(plans/scheduler-dispatch-scoring-and-lock-layer-prd.md §7).

Every module is a single-segment ``.py`` file, so its lock key is its own name.
Holders take their locks through ``lock_table.try_acquire`` and appear in the
task list as ``in-progress``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _park_test_helpers import ParkWorld, event_data_for, make_task, park_world

from orchestrator.config import OrchestratorConfig, PsiAdmissionConfig
from orchestrator.main_health_fix_owner import SIGNATURE_KEY
from orchestrator.overrides import OverrideSource
from orchestrator.pin_reservation import PinOrder
from orchestrator.scheduler import Scheduler


def _fix_task(tid: str, files: list[str], *, status: str = 'pending') -> dict:
    task = make_task(tid, 'high', files, status=status)
    task['metadata'].update({SIGNATURE_KEY: f'sig-{tid}', 'merge_lane': 'high'})
    return task


def _world(tmp_path: Path, *, pinned: tuple[str, ...] = ()) -> ParkWorld:
    return park_world(tmp_path, pinned=pinned, psi_admission=PsiAdmissionConfig(enabled=False))


def _board(w: ParkWorld, tasks: list[dict]) -> None:
    w.scheduler.get_tasks = AsyncMock(return_value=tasks)


def _drop(w: ParkWorld, *task_ids: str) -> None:
    """Release each holder's lock and remove it from the task list."""
    for tid in task_ids:
        w.scheduler.lock_table.release(tid)
    board = w.scheduler.get_tasks
    assert isinstance(board, AsyncMock)
    tasks = board.return_value
    _board(w, [t for t in tasks if t['id'] not in task_ids])


def _pin_queue(w: ParkWorld) -> list[tuple[str, int | None]]:
    return [(tid, row.pin_order) for tid, row in w.overrides.get_pin_queue(w.root)]


async def _grant(w: ParkWorld, task_id: str) -> None:
    assert await w.scheduler.grant_main_health_fix_precedence(task_id) == 1


@pytest.mark.asyncio
async def test_granted_fix_dispatches_before_an_operator_pin_and_a_critical(tmp_path):
    w = _world(tmp_path, pinned=('P',))
    _board(w, [
        make_task('P', 'medium', ['a.py']),
        make_task('C', 'critical', ['c.py']),
        _fix_task('F', ['f.py']),
    ])
    await _grant(w, 'F')

    result = await w.scheduler.acquire_next()

    assert result is not None and result.task_id == 'F'
    assert _pin_queue(w) == [('F', 1), ('P', 2)]


@pytest.mark.asyncio
async def test_reserve_now_parks_the_fix_on_its_first_tick(tmp_path):
    w = _world(tmp_path)
    assert w.scheduler.lock_table.try_acquire('W', ['f.py'])
    _board(w, [
        make_task('W', 'low', ['f.py'], status='in-progress'),
        _fix_task('F', ['f.py']),
    ])
    await _grant(w, 'F')

    assert await w.scheduler.acquire_next() is None

    assert [d['modules'] for d in event_data_for(w.store, 'reserve_now_consumed', 'F')] == [
        ['f.py'],
    ]
    assert w.overrides.get_overrides(w.root)['F'].reserve_now is False


@pytest.mark.asyncio
async def test_fix_parks_behind_a_held_lock_and_takes_it_first_on_release(tmp_path):
    w = _world(tmp_path, pinned=('P',))
    lock_table = w.scheduler.lock_table
    assert lock_table.try_acquire('W', ['m.py'])
    _board(w, [
        make_task('W', 'low', ['m.py'], status='in-progress'),
        make_task('P', 'medium', ['m.py']),
        make_task('C', 'critical', ['m.py']),
        _fix_task('F', ['m.py']),
    ])
    await _grant(w, 'F')

    assert await w.scheduler.acquire_next() is None, 'tick 1: the holder is not preempted'

    assert lock_table.held_modules('W') == {'m.py'}
    assert list(lock_table.snapshot_pin_reservations()) == ['F'], (
        'the fix, not the old head P, holds the pin reservation'
    )
    top = lock_table.snapshot_park_stacks()['m.py'][-1]
    assert (top['owner'], top['rank']) == ('F', PinOrder(1).rank)
    assert lock_table.held_modules('C') == frozenset()
    assert lock_table.held_modules('P') == frozenset()

    _drop(w, 'W')
    result = await w.scheduler.acquire_next()

    assert result is not None and result.task_id == 'F', 'tick 2: the fix takes the freed module'


@pytest.mark.asyncio
@pytest.mark.parametrize('final_status', ['done', 'cancelled'])
async def test_pin_survives_dispatch_and_clears_when_the_fix_lands_or_is_cancelled(
    tmp_path, final_status,
):
    w = _world(tmp_path)
    _board(w, [_fix_task('F', ['f.py'])])
    await _grant(w, 'F')
    result = await w.scheduler.acquire_next()
    assert result is not None and result.task_id == 'F', 'premise: the fix dispatches'

    _board(w, [_fix_task('F', ['f.py'], status='in-progress')])
    await w.scheduler.acquire_next()

    row = w.overrides.get_overrides(w.root)['F']
    assert row.pinned is True
    assert row.source is OverrideSource.MAIN_HEALTH_FIX

    _board(w, [_fix_task('F', ['f.py'], status=final_status)])
    await w.scheduler.acquire_next()

    assert 'F' not in w.overrides.get_overrides(w.root)


@pytest.mark.asyncio
async def test_grant_without_an_override_store_returns_none(tmp_path):
    config = OrchestratorConfig(project_root=tmp_path)
    scheduler = Scheduler(config)

    assert await scheduler.grant_main_health_fix_precedence('F') is None


@pytest.mark.asyncio
async def test_grant_never_raises_when_the_store_write_fails(tmp_path, monkeypatch, caplog):
    w = _world(tmp_path)

    def _boom(project_root: str, task_id: str) -> int | None:
        raise RuntimeError('store unavailable')

    monkeypatch.setattr(w.overrides, 'pin_main_health_fix', _boom)

    with caplog.at_level(logging.WARNING, logger='orchestrator.scheduler'):
        assert await w.scheduler.grant_main_health_fix_precedence('fix-77') is None

    assert any(
        record.levelno == logging.WARNING and 'fix-77' in record.getMessage()
        for record in caplog.records
    )
