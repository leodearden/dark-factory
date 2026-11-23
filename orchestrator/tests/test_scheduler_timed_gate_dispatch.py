"""An open timed gate dispatches ahead of the tier-scored queue, whatever its priority (task 11330).

A recurrence-minted predicate link is born low-tier, youngest and with the
highest id, so in score order it loses every tick to the older low backlog.
A timed gate needs no module lock, so the scheduler dispatches it first once
its milestone opens; a deploy, or a gate whose milestone has not opened, keeps
its scored place, and an operator pin still goes ahead of it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _park_test_helpers import ParkWorld, event_payloads, make_task, park_world


def _at(delta: timedelta) -> str:
    return (datetime.now(UTC) + delta).isoformat()


def _backlog(n: int) -> list[dict]:
    return [make_task(str(i), 'low', [f'mod{i}.py']) for i in range(1, n + 1)]


def _link(tid: str, *, at: str, priority: str = 'low', kind: str = 'predicate') -> dict:
    return {
        'id': tid,
        'title': f'Link {tid}',
        'status': 'pending',
        'priority': priority,
        'dependencies': [],
        'metadata': {
            'task_kind': 'deterministic',
            'before_done': {
                'script': 'scripts/check_write_triage_readiness_gate.py',
                'args': [],
                'env': {},
                'timeout_secs': 300,
                'kind': kind,
                'script_root': 'factory' if kind == 'predicate' else 'project',
            },
            'milestone': {'mode': 'dated', 'at': at},
            'recurrence': {
                'key': 'claude-code-mode-headless-support',
                'interval_secs': 86400,
                'minted_from': '400',
            },
            'source': 'recurrence-mint',
        },
    }


def _pure_gate(tid: str, *, at: str) -> dict:
    return {
        'id': tid,
        'title': f'Gate {tid}',
        'status': 'pending',
        'priority': 'low',
        'dependencies': [],
        'metadata': {
            'task_kind': 'deterministic',
            'always_escalates': True,
            'milestone': {'mode': 'dated', 'at': at},
        },
    }


def _world(tmp_path: Path, tasks: list[dict], *, pinned: tuple[str, ...] = ()) -> ParkWorld:
    w = park_world(tmp_path, pinned=pinned)
    w.scheduler.get_tasks = AsyncMock(return_value=tasks)
    return w


def _last_terminal_phase(w: ParkWorld) -> str:
    return event_payloads(w.store, 'scheduler_tick')[-1]['data']['terminal_phase']


async def _dispatched_id(w: ParkWorld) -> str | None:
    assignment = await w.scheduler.acquire_next()
    return None if assignment is None else assignment.task_id


@pytest.mark.asyncio
async def test_an_open_low_link_dispatches_ahead_of_a_500_task_low_backlog(tmp_path):
    w = _world(tmp_path, [*_backlog(500), _link('501', at=_at(-timedelta(hours=1)))])

    assert await _dispatched_id(w) == '501'
    assert _last_terminal_phase(w) == 'select_timed_gates'
    assert '501' not in w.scheduler.lock_table.snapshot_holders().values()


@pytest.mark.asyncio
async def test_an_open_low_pure_gate_dispatches_ahead_of_a_critical_task(tmp_path):
    critical = make_task('C', 'critical', ['c.py'])
    w = _world(tmp_path, [critical, _pure_gate('900', at=_at(-timedelta(hours=1)))])

    assert await _dispatched_id(w) == '900'


@pytest.mark.asyncio
async def test_a_link_whose_milestone_has_not_opened_waits_behind_the_scored_top(tmp_path):
    w = _world(tmp_path, [*_backlog(3), _link('501', at=_at(timedelta(days=1)))])

    assert await _dispatched_id(w) == '1'


@pytest.mark.asyncio
async def test_a_deploy_keeps_its_place_in_the_scored_queue(tmp_path):
    deploy = _link('501', at=_at(-timedelta(hours=1)), kind='deploy')
    w = _world(tmp_path, [*_backlog(3), deploy])

    assert await _dispatched_id(w) == '1'


@pytest.mark.asyncio
async def test_a_pin_still_dispatches_before_an_open_link(tmp_path):
    pin = make_task('P', 'low', ['p.py'])
    w = _world(
        tmp_path,
        [pin, *_backlog(3), _link('501', at=_at(-timedelta(hours=1)))],
        pinned=('P',),
    )

    assert await _dispatched_id(w) == 'P'
    assert _last_terminal_phase(w) == 'select_pins'
    assert await _dispatched_id(w) == '501'
