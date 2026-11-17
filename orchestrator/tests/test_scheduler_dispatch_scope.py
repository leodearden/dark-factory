"""A dispatch tick restricted to a scope hands out only the tasks that scope admits (task 8480).

The harness runs a ``DispatchScope.MAIN_HEALTH_FIX`` tick when only the
main-health-fix run slot is free.  Such a tick must return the fix even when
plainer work outranks it, must not take a lock for, skip-bump or pin-dispatch a
task the reserve cannot run, must leave the pin reservations of the general
pool's pins in place, and must say in its ``scheduler_tick`` event which scope
it ran under.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _park_test_helpers import ParkWorld, event_payloads, make_task, park_world

from orchestrator.main_health_fix_owner import SIGNATURE_KEY
from orchestrator.run_slots import DispatchScope


def _fix_task(tid: str, files: list[str]) -> dict:
    task = make_task(tid, 'high', files)
    task['metadata'].update({SIGNATURE_KEY: f'sig-{tid}', 'merge_lane': 'high'})
    return task


PLAIN = make_task('P', 'critical', ['p.py'])
FIX = _fix_task('F', ['f.py'])


def _world(tmp_path: Path, tasks: list[dict], *, pinned: tuple[str, ...] = ()) -> ParkWorld:
    w = park_world(tmp_path, pinned=pinned)
    w.scheduler.get_tasks = AsyncMock(return_value=tasks)
    return w


def _ticks(w: ParkWorld) -> list[dict]:
    return [payload['data'] for payload in event_payloads(w.store, 'scheduler_tick')]


@pytest.mark.asyncio
async def test_a_restricted_tick_returns_the_fix_over_an_outranking_plain_task(tmp_path):
    w = _world(tmp_path, [PLAIN, FIX])

    result = await w.scheduler.acquire_next(DispatchScope.MAIN_HEALTH_FIX)

    assert result is not None and result.task_id == 'F'
    assert not w.scheduler.lock_table.is_held('P')
    assert 'P' not in w.scheduler.lock_table.snapshot_holders().values()


@pytest.mark.asyncio
async def test_a_restricted_tick_with_no_fix_takes_nothing_and_skips_nobody(tmp_path):
    w = _world(tmp_path, [PLAIN])

    result = await w.scheduler.acquire_next(DispatchScope.MAIN_HEALTH_FIX)

    assert result is None
    assert w.scheduler.lock_table.snapshot_holders() == {}
    assert w.scheduler.get_state_snapshot().skip_counts.get('P', 0) == 0


@pytest.mark.asyncio
async def test_a_pinned_plain_task_does_not_dispatch_on_a_restricted_tick(tmp_path):
    w = _world(tmp_path, [PLAIN], pinned=('P',))

    assert await w.scheduler.acquire_next(DispatchScope.MAIN_HEALTH_FIX) is None
    assert not w.scheduler.lock_table.is_held('P')


@pytest.mark.asyncio
async def test_a_restricted_tick_that_dispatches_a_fix_keeps_a_blocked_pin_s_reservation(tmp_path):
    holder = make_task('H', 'low', ['p.py'], status='in-progress')
    w = _world(tmp_path, [PLAIN, holder], pinned=('P',))
    assert w.scheduler.lock_table.try_acquire('H', ['p.py'])
    assert await w.scheduler.acquire_next() is None
    assert list(w.scheduler.get_state_snapshot().pin_reservations) == ['P'], 'premise: P reserves'
    w.scheduler.get_tasks = AsyncMock(return_value=[PLAIN, holder, FIX])

    result = await w.scheduler.acquire_next(DispatchScope.MAIN_HEALTH_FIX)

    assert result is not None and result.task_id == 'F'
    assert list(w.scheduler.get_state_snapshot().pin_reservations) == ['P']


@pytest.mark.asyncio
async def test_a_pinned_plain_task_yields_to_the_fix_on_a_restricted_tick(tmp_path):
    w = _world(tmp_path, [PLAIN, FIX], pinned=('P',))

    result = await w.scheduler.acquire_next(DispatchScope.MAIN_HEALTH_FIX)

    assert result is not None and result.task_id == 'F'
    assert not w.scheduler.lock_table.is_held('P')


@pytest.mark.asyncio
async def test_a_pinned_plain_task_still_dispatches_on_a_default_tick(tmp_path):
    w = _world(tmp_path, [PLAIN, FIX], pinned=('P',))

    result = await w.scheduler.acquire_next()

    assert result is not None and result.task_id == 'P'
    assert _ticks(w)[-1]['terminal_phase'] == 'select_pins'


@pytest.mark.asyncio
async def test_a_pinned_fix_dispatches_through_the_scored_path_on_a_restricted_tick(tmp_path):
    w = _world(tmp_path, [PLAIN, FIX], pinned=('F',))

    result = await w.scheduler.acquire_next(DispatchScope.MAIN_HEALTH_FIX)

    assert result is not None and result.task_id == 'F'
    assert _ticks(w)[-1]['terminal_phase'] == 'select_scored'


@pytest.mark.asyncio
async def test_the_default_scope_is_any(tmp_path):
    w = _world(tmp_path, [PLAIN, FIX])

    result = await w.scheduler.acquire_next()

    assert result is not None and result.task_id == 'P'
    assert _ticks(w)[-1]['scope'] == DispatchScope.ANY.value


@pytest.mark.asyncio
async def test_tick_telemetry_names_the_scope(tmp_path):
    w = _world(tmp_path, [PLAIN, FIX])

    await w.scheduler.acquire_next(DispatchScope.MAIN_HEALTH_FIX)
    await w.scheduler.acquire_next(DispatchScope.ANY)

    assert [tick['scope'] for tick in _ticks(w)] == ['main_health_fix', 'any']


@pytest.mark.asyncio
async def test_an_empty_project_tick_names_the_scope(tmp_path):
    w = _world(tmp_path, [])

    assert await w.scheduler.acquire_next(DispatchScope.MAIN_HEALTH_FIX) is None

    [tick] = _ticks(w)
    assert tick['terminal_phase'] == 'no_active_tasks'
    assert tick['scope'] == 'main_health_fix'
