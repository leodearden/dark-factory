"""The fairness park completion rule and its single emission decision (task 5308).

A passed-over top that crosses its skip threshold asks for parks on every
module it has not parked yet — the REMAINDER — on every qualifying skip, so a
module that frees up is parked on the next one.  ``reservation_installed``
fires only when that attempt newly parked something and carries only that
increment; ``reservation_install_blocked`` fires, geometrically rate-limited on
the owner's consecutive blocked attempts, whenever it parked fewer modules
than it asked for.

Every module is a single-segment ``.py`` file, so its lock key is its own name
at any ``lock_depth`` (an extension-less name would be stripped as a directory
by ``derive_modules`` and fall back to the synthetic ``task-<id>`` lock).

A deterministic top is not exercised here: it locks nothing, so it always
acquires and is never passed over.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from _recording_event_store import _RecordingEventStore

from orchestrator.config import OrchestratorConfig
from orchestrator.scheduler import Scheduler

M1, M2, M3, M4 = 'm1.py', 'm2.py', 'm3.py', 'm4.py'


def _task(tid: str, priority: str, files: list[str], *, status: str = 'pending') -> dict:
    return {
        'id': tid,
        'title': f'Task {tid}',
        'status': status,
        'priority': priority,
        'dependencies': [],
        'metadata': {'files': list(files)},
    }


def _foreign_owner(files: list[str]) -> dict:
    """F: a running task that keeps its park — not a candidate, not GC'd."""
    return _task('F', 'high', files, status='in-progress')


def _scheduler() -> tuple[Scheduler, _RecordingEventStore]:
    config = OrchestratorConfig(max_per_module=1, lock_depth=2)
    config.fairness.skip_threshold = 1
    store = _RecordingEventStore()
    scheduler = Scheduler(config, event_store=store)  # type: ignore[arg-type]
    scheduler.finish_startup()
    return scheduler, store


def _events(store, name: str) -> list[dict]:
    """Payloads of every recorded event whose type string ends in *name*."""
    return [
        payload
        for event_type, payload in store.events
        if event_type.split('.')[-1] == name or event_type == name
    ]


def _for(store, name: str, task_id: str) -> list[dict]:
    return [e['data'] for e in _events(store, name) if e['task_id'] == task_id]


def _owners(scheduler: Scheduler, module: str) -> list[str]:
    stacks = scheduler.lock_table.snapshot_park_stacks()
    return [entry['owner'] for entry in stacks.get(module, [])]


@pytest.mark.asyncio
async def test_an_empty_install_emits_no_reservation_installed_and_one_blocked_event():
    scheduler, store = _scheduler()
    assert scheduler.lock_table.try_acquire('seed', [M1])
    scheduler.lock_table.install_parks('F', [M1, M2], 'high')
    t = _task('T', 'high', [M1, M2])
    scheduler.get_tasks = AsyncMock(return_value=[t, _foreign_owner([M1, M2])])

    assert await scheduler.acquire_next() is None, 'empty install: T must not dispatch'

    assert _events(store, 'reservation_installed') == [], (
        'empty install: no reservation_installed at all, least of all modules == []'
    )
    blocked = _events(store, 'reservation_install_blocked')
    assert [e['task_id'] for e in blocked] == ['T'], 'empty install: one blocked event, for T'
    assert blocked[0]['data'] == {
        'requested': [M1, M2],
        'installed': [],
        'blocked': [M1, M2],
        'attempts': 1,
        'skip_count': 1,
        'priority': 'high',
    }, 'empty install: blocked payload'
    assert 'T' not in scheduler.lock_table.snapshot_parks(), 'empty install: T parks nothing'


@pytest.mark.asyncio
async def test_a_blocked_install_is_retried_every_tick_but_reported_geometrically():
    scheduler, store = _scheduler()
    assert scheduler.lock_table.try_acquire('seed', [M1])
    scheduler.lock_table.install_parks('F', [M1, M2], 'high')
    t = _task('T', 'high', [M1, M2])
    scheduler.get_tasks = AsyncMock(return_value=[t, _foreign_owner([M1, M2])])

    for _ in range(12):
        assert await scheduler.acquire_next() is None, 'non-storming: T must not dispatch'

    attempts = [d['attempts'] for d in _for(store, 'reservation_install_blocked', 'T')]
    assert attempts == [1, 10], 'non-storming: blocked events only at geometric attempts'
    assert _events(store, 'reservation_installed') == [], (
        'non-storming: nothing was ever parked, so no reservation_installed'
    )
    assert len(_for(store, 'task_skipped', 'T')) == 12, (
        'non-storming: task_skipped still fires every tick under a finite threshold'
    )


@pytest.mark.asyncio
async def test_a_partial_install_parks_what_it_can_and_reports_the_rest():
    scheduler, store = _scheduler()
    scheduler.lock_table.install_parks('F', [M2], 'high')
    t = _task('T', 'high', [M1, M2])
    f = _foreign_owner([M2])
    scheduler.get_tasks = AsyncMock(return_value=[t, f])

    assert await scheduler.acquire_next() is None, 'partial install: T must not dispatch'

    assert _for(store, 'reservation_installed', 'T') == [
        {'modules': [M1], 'skip_count': 1, 'priority': 'high'},
    ], 'partial install: reservation_installed carries only the parked module'
    blocked = _for(store, 'reservation_install_blocked', 'T')
    assert len(blocked) == 1, 'partial install: one blocked event'
    assert blocked[0]['requested'] == [M1, M2], 'partial install: requested'
    assert blocked[0]['installed'] == [M1], 'partial install: installed'
    assert blocked[0]['blocked'] == [M2], 'partial install: blocked'
    assert blocked[0]['attempts'] == 1, 'partial install: attempts'

    low = _task('L', 'low', [M1])
    scheduler.get_tasks = AsyncMock(return_value=[t, f, low])
    result = await scheduler.acquire_next()
    assert result is None or result.task_id != 'L', (
        "partial install: T's park on m1 must refuse the lower-tier L"
    )


@pytest.mark.asyncio
async def test_a_partial_park_is_completed_once_the_foreign_park_clears():
    scheduler, store = _scheduler()
    scheduler.lock_table.install_parks('F', [M3], 'high')
    assert scheduler.lock_table.try_acquire('seed', [M1])
    t = _task('T', 'high', [M1, M2, M3])
    f = _foreign_owner([M3])
    scheduler.get_tasks = AsyncMock(return_value=[t, f])

    assert await scheduler.acquire_next() is None
    assert sorted(scheduler.lock_table.snapshot_parks()['T']['modules']) == [M1, M2], (
        'completion: tick 1 parks m1 and m2 only'
    )

    scheduler.lock_table.clear_parks_for('F')
    assert await scheduler.acquire_next() is None, 'completion: T is still held off m1'

    assert _for(store, 'reservation_installed', 'T')[-1]['modules'] == [M3], (
        'completion: tick 2 parks only the remainder'
    )
    for module in (M1, M2, M3):
        assert _owners(scheduler, module) == ['T'], (
            f'completion: exactly one T entry on {module}'
        )
    assert len(_for(store, 'reservation_install_blocked', 'T')) == 1, (
        'completion: a fully parked remainder reports nothing blocked'
    )

    low = _task('L', 'low', [M3])
    scheduler.get_tasks = AsyncMock(return_value=[t, f, low])
    result = await scheduler.acquire_next()
    assert result is None or result.task_id != 'L', (
        "completion: T's completed park on m3 must refuse L"
    )

    scheduler.get_tasks = AsyncMock(return_value=[t, f])
    scheduler.lock_table.install_parks('F', [M4], 'high')
    scheduler.seed_modules('T', [M1, M2, M3, M4])
    assert await scheduler.acquire_next() is None
    assert _for(store, 'reservation_install_blocked', 'T')[-1]['attempts'] == 1, (
        'completion: the blocked-attempt streak restarted once nothing was blocked'
    )


@pytest.mark.asyncio
async def test_a_fully_parked_top_makes_no_further_install_attempt():
    scheduler, store = _scheduler()
    assert scheduler.lock_table.try_acquire('seed', [M1])
    t = _task('T', 'high', [M1, M2])
    scheduler.get_tasks = AsyncMock(return_value=[t])

    assert await scheduler.acquire_next() is None
    assert sorted(scheduler.lock_table.snapshot_parks()['T']['modules']) == [M1, M2], (
        'full coverage: premise — T parks all its modules on tick 1'
    )
    installed_before = len(_events(store, 'reservation_installed'))
    stacks_before = scheduler.lock_table.snapshot_park_stacks()

    for tick in range(3):
        assert await scheduler.acquire_next() is None
        assert scheduler.lock_table.snapshot_park_stacks() == stacks_before, (
            f'full coverage: park stacks unchanged on extra tick {tick}'
        )

    assert len(_events(store, 'reservation_installed')) == installed_before, (
        'full coverage: no reservation_installed once T is fully parked'
    )
    assert _events(store, 'reservation_install_blocked') == [], (
        'full coverage: no reservation_install_blocked once T is fully parked'
    )
    assert len(_for(store, 'task_skipped', 'T')) == 4, (
        'full coverage: task_skipped still fires on every passed-over tick'
    )


@pytest.mark.asyncio
async def test_a_grown_lock_set_is_parked_on_the_next_qualifying_skip():
    scheduler, store = _scheduler()
    assert scheduler.lock_table.try_acquire('seed', [M1])
    t = _task('T', 'high', [M1, M2])
    scheduler.get_tasks = AsyncMock(return_value=[t])

    assert await scheduler.acquire_next() is None
    assert sorted(scheduler.lock_table.snapshot_parks()['T']['modules']) == [M1, M2], (
        'expansion: premise — T is fully parked on m1, m2'
    )

    scheduler.seed_modules('T', [M1, M2, M4])
    assert await scheduler.acquire_next() is None

    assert _for(store, 'reservation_installed', 'T')[-1]['modules'] == [M4], (
        'expansion: the grown lock set parks only the new module'
    )
    assert _owners(scheduler, M4) == ['T'], 'expansion: T now parks m4'
