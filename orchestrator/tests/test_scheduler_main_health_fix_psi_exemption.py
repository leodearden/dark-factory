"""A saturated host does not defer a main-health fix, while plain work is still deferred.

DA-D10 of docs/prds/dispatch-admission-load-cap.md (task 10600): a task that
carries a main-health signature is exempt from the dispatch-admission hold on
every tick scope, as a deterministic task is under DA-D4.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _park_test_helpers import ParkWorld, event_payloads, make_task, park_world
from shared.psi import PsiSample

from orchestrator.config import PsiAdmissionConfig
from orchestrator.main_health_fix_owner import SIGNATURE_KEY
from orchestrator.run_slots import DispatchScope

SATURATED = PsiSample(
    cpu_some10=0.0,
    mem_some10=0.0,
    mem_full10=5.0,
    io_some10=0.0,
    read_ok=True,
    runqueue_ratio=0.0,
    runqueue_read_ok=False,
    own_cpu_some10=0.0,
    own_cgroup='',
    own_read_ok=False,
    cpu_some60=0.0,
)

FILLER = make_task('W', 'low', ['w.py'])
PLAIN = make_task('P', 'critical', ['p.py'])


def _fix_task(tid: str, files: list[str]) -> dict:
    task = make_task(tid, 'high', files)
    task['metadata'].update({SIGNATURE_KEY: f'sig-{tid}', 'merge_lane': 'high'})
    return task


FIX = _fix_task('F', ['f.py'])


@pytest.fixture
def saturated_host(monkeypatch):
    monkeypatch.setattr(
        'orchestrator.scheduler.read_psi_sample', lambda *, project_id=None: SATURATED
    )


def _world(tmp_path: Path) -> ParkWorld:
    return park_world(tmp_path, psi_admission=PsiAdmissionConfig())


def _board(w: ParkWorld, tasks: list[dict]) -> None:
    w.scheduler.get_tasks = AsyncMock(return_value=tasks)


async def _one_in_flight(w: ParkWorld) -> None:
    _board(w, [FILLER])
    result = await w.scheduler.acquire_next()
    assert result is not None and result.task_id == 'W', 'premise: the floor admits the first dispatch'


def _deferred_ids(w: ParkWorld) -> list[str]:
    return [payload['task_id'] for payload in event_payloads(w.store, 'dispatch_deferred')]


def _ticks(w: ParkWorld) -> list[dict]:
    return [payload['data'] for payload in event_payloads(w.store, 'scheduler_tick')]


@pytest.mark.asyncio
async def test_a_held_restricted_tick_still_returns_the_fix(tmp_path, saturated_host):
    w = _world(tmp_path)
    await _one_in_flight(w)
    _board(w, [FILLER, PLAIN])
    assert await w.scheduler.acquire_next() is None, 'premise: the hold is engaged'
    assert _deferred_ids(w)[-1] == 'P'
    _board(w, [FILLER, PLAIN, FIX])

    result = await w.scheduler.acquire_next(DispatchScope.MAIN_HEALTH_FIX)

    assert result is not None and result.task_id == 'F'
    assert w.scheduler.lock_table.is_held('F')


@pytest.mark.asyncio
async def test_a_held_default_tick_defers_plain_work_and_dispatches_the_fix(
    tmp_path, saturated_host
):
    w = _world(tmp_path)
    await _one_in_flight(w)
    _board(w, [FILLER, PLAIN, FIX])

    result = await w.scheduler.acquire_next()

    assert result is not None and result.task_id == 'F'
    assert not w.scheduler.lock_table.is_held('P')
    assert _deferred_ids(w) == ['P']


@pytest.mark.asyncio
async def test_a_pinned_fix_dispatches_from_the_pin_loop_on_a_held_tick(tmp_path, saturated_host):
    w = _world(tmp_path)
    await _one_in_flight(w)
    w.overrides.set_override(w.root, 'F', pinned=True)
    _board(w, [FILLER, PLAIN, FIX])

    result = await w.scheduler.acquire_next()

    assert result is not None and result.task_id == 'F'
    assert _ticks(w)[-1]['terminal_phase'] == 'select_pins'
    assert 'F' not in _deferred_ids(w)
