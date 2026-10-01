"""The head lock-blocked pin earns a pin reservation (task 6040).

Before task 6040 a pinned task blocked by a module lock earned nothing while it
waited, so every module that freed up could be taken by whoever scored next.
Now:

- D1  the head is the lowest-pin_order pin that is pinned, present, eligible,
      not landed-outbox gated and not deterministic, and fails try_acquire;
- D2  at most ``pin_reservation_max_active`` heads hold one at a time;
- D3  a pin reservation shadows any fairness park, critical included, and
      restores it when used or released, but never preempts a held lock;
- D4  the head covers its CURRENT module set every tick, installed inline so
      a later pin or scored candidate cannot take a module it just freed;
- D5  dispatch uses it, owner state or the pin phase releases it, and the
      first tick after a restart rebuilds it from the durable override store;
- D6  EASY-backfill borrows through it exactly as through a fairness park
      (proved in test_scheduler_park_backfill.py).

Every module is a single-segment ``.py`` file, so its lock key is its own name.
Holders take their locks through ``lock_table.try_acquire`` and appear in the
task list as ``in-progress``: park GC keeps their parks and they are never
candidates.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from _park_test_helpers import (
    ParkWorld,
    event_data_for,
    event_index,
    event_matches,
    event_payloads,
    make_task,
    park_world,
)
from _recording_event_store import _RecordingEventStore

from orchestrator.scheduler import Scheduler


def _reservation_events(store: _RecordingEventStore, task_id: str) -> list[str]:
    """Names of every ``reservation_*`` event recorded about *task_id*."""
    return [
        event_type.split('.')[-1]
        for event_type, payload in store.events
        if event_type.split('.')[-1].startswith('reservation_')
        and payload['task_id'] == task_id
    ]


def _owners(scheduler: Scheduler, module: str) -> list[str]:
    stacks = scheduler.lock_table.snapshot_park_stacks()
    return [entry['owner'] for entry in stacks.get(module, [])]


def _3659_tasks() -> list[dict]:
    return [
        make_task('P', 'high', ['w.py', 'x.py']),
        make_task('H', 'low', ['w.py'], status='in-progress'),
        make_task('C', 'critical', ['w.py']),
        make_task('M', 'medium', ['x.py']),
    ]


def _3659_world(tmp_path, **config_overrides) -> ParkWorld:
    """Task 3659's shape: a pinned P blocked by a holder, under a critical park.

    P (pinned, high) needs w and x.  H holds w.  C (critical, unpinned) holds
    a fairness park on w.  M (medium) needs only x — the module that frees
    first and that a pin with no reservation loses.
    """
    w = park_world(tmp_path, pinned=('P',), **config_overrides)
    assert w.scheduler.lock_table.try_acquire('H', ['w.py'])
    w.scheduler.lock_table.install_parks('C', ['w.py'], 'critical')
    w.scheduler.get_tasks = AsyncMock(return_value=_3659_tasks())
    return w


def _drop(w: ParkWorld, *task_ids: str) -> None:
    """Release each holder's lock and remove it from the task list."""
    for tid in task_ids:
        w.scheduler.lock_table.release(tid)
    tasks = w.scheduler.get_tasks.return_value  # type: ignore[attr-defined]
    w.scheduler.get_tasks = AsyncMock(
        return_value=[t for t in tasks if t['id'] not in task_ids]
    )


# ---------------------------------------------------------------------------
# Acceptance 1 — the task 3659 replay
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_3659_replay(tmp_path):
    w = _3659_world(tmp_path)
    lock_table = w.scheduler.lock_table

    assert await w.scheduler.acquire_next() is None, 'tick 1: nothing dispatches'

    assert _owners(w.scheduler, 'w.py') == ['C', 'P'], (
        "tick 1: P's pin reservation shadows C's critical park on w"
    )
    assert _owners(w.scheduler, 'x.py') == ['P'], 'tick 1: P reserves x'
    assert 'M' not in lock_table.snapshot_holders().values(), 'tick 1: M is refused x'
    assert 'x.py' not in lock_table.snapshot_holders(), 'tick 1: x stays free for P'
    installed = event_data_for(w.store, 'reservation_installed', 'P')
    assert [(d['source'], d['modules'], d['pin_order']) for d in installed] == [
        ('pin', ['w.py', 'x.py'], 1),
    ]
    shadowed = event_data_for(w.store, 'reservation_shadowed', 'P')
    assert [d['victim'] for d in shadowed] == ['C']

    _drop(w, 'H')
    result = await w.scheduler.acquire_next()

    assert result is not None and result.task_id == 'P', 'tick 2: P dispatches'
    assert [d['source'] for d in event_data_for(w.store, 'reservation_used', 'P')] == ['pin']
    assert [e['data'] for e in event_payloads(w.store, 'reservation_restored')] == [
        {'restored_owner': 'C', 'modules': ['w.py'], 'source': 'fairness'},
    ]
    assert _owners(w.scheduler, 'w.py') == ['C'], "tick 2: C's park is the top again"
    assert event_index(w.store, 'reservation_used', 'P') < event_index(
        w.store, 'lock_acquired', 'P'
    )


# ---------------------------------------------------------------------------
# Acceptance 3 — a held lock is never preempted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_held_lock_is_never_preempted(tmp_path):
    w = _3659_world(tmp_path)

    for tick in range(3):
        result = await w.scheduler.acquire_next()
        assert result is None or result.task_id != 'P', f'tick {tick}: P must wait for H'
        assert w.scheduler.lock_table.snapshot_holders()['w.py'] == 'H', (
            f'tick {tick}: H keeps w'
        )

    _drop(w, 'H')
    result = await w.scheduler.acquire_next()

    assert result is not None and result.task_id == 'P'


# ---------------------------------------------------------------------------
# Acceptance 2 — only the head reserves, and the head blocks later pins
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_only_the_head_pin_reserves(tmp_path):
    w = park_world(tmp_path, pinned=('P1', 'P2'))
    assert w.scheduler.lock_table.try_acquire('H1', ['a.py'])
    assert w.scheduler.lock_table.try_acquire('H2', ['b.py'])
    w.scheduler.get_tasks = AsyncMock(return_value=[
        make_task('P1', 'medium', ['a.py']),
        make_task('P2', 'medium', ['b.py']),
        make_task('H1', 'low', ['a.py'], status='in-progress'),
        make_task('H2', 'low', ['b.py'], status='in-progress'),
    ])

    assert await w.scheduler.acquire_next() is None
    assert list(w.scheduler.lock_table.snapshot_pin_reservations()) == ['P1']

    _drop(w, 'H2')
    result = await w.scheduler.acquire_next()

    assert result is not None and result.task_id == 'P2', (
        'P2 is free of holders and of the head reservation, so it dispatches'
    )
    assert 'P1' in w.scheduler.lock_table.snapshot_pin_reservations(), (
        'the still-blocked head keeps its reservation'
    )


@pytest.mark.asyncio
async def test_the_head_reservation_refuses_a_later_pin(tmp_path):
    """P1 reserves a (unheld) and c (held by H1); P2 then cannot take a.

    a has no holder at all, so P2's refusal is the reservation's doing.
    """
    w = park_world(tmp_path, pinned=('P1', 'P2'))
    assert w.scheduler.lock_table.try_acquire('H1', ['c.py'])
    assert w.scheduler.lock_table.try_acquire('H2', ['b.py'])
    w.scheduler.get_tasks = AsyncMock(return_value=[
        make_task('P1', 'medium', ['a.py', 'c.py']),
        make_task('P2', 'medium', ['a.py', 'b.py']),
        make_task('H1', 'low', ['c.py'], status='in-progress'),
        make_task('H2', 'low', ['b.py'], status='in-progress'),
    ])

    assert await w.scheduler.acquire_next() is None
    assert w.scheduler.lock_table.snapshot_pin_reservations()['P1']['modules'] == [
        'a.py', 'c.py',
    ]

    _drop(w, 'H2')
    result = await w.scheduler.acquire_next()

    assert result is None or result.task_id != 'P2', "P1's reservation must refuse P2 on a"
    assert 'P2' not in w.scheduler.lock_table.snapshot_holders().values()


# ---------------------------------------------------------------------------
# D1 — a deterministic pin never reserves
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deterministic_pin_never_reserves(tmp_path):
    """A deterministic task locks nothing, so it is never lock-blocked.

    It dispatches through a held module and earns no reservation.
    """
    w = park_world(tmp_path, pinned=('D',))
    assert w.scheduler.lock_table.try_acquire('H', ['w.py'])
    gate = make_task('D', 'medium', ['w.py'])
    gate['metadata']['task_kind'] = 'deterministic'
    w.scheduler.get_tasks = AsyncMock(return_value=[
        gate,
        make_task('H', 'low', ['w.py'], status='in-progress'),
    ])

    result = await w.scheduler.acquire_next()

    assert result is not None and result.task_id == 'D'
    assert w.scheduler.lock_table.snapshot_pin_reservations() == {}
    assert _reservation_events(w.store, 'D') == []


# ---------------------------------------------------------------------------
# Acceptance 5 — a restart rebuilds the reservation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restart_rebuilds_the_reservation(tmp_path):
    w = _3659_world(tmp_path)
    assert await w.scheduler.acquire_next() is None
    before = w.scheduler.lock_table.snapshot_pin_reservations()['P']['modules']

    store = _RecordingEventStore()
    restarted = Scheduler(
        w.scheduler.config,
        event_store=store,  # type: ignore[arg-type]
        override_store=w.overrides,
    )
    restarted.finish_startup()
    assert restarted.lock_table.try_acquire('H', ['w.py'])
    restarted.get_tasks = AsyncMock(return_value=_3659_tasks())

    assert await restarted.acquire_next() is None

    assert restarted.lock_table.snapshot_pin_reservations()['P']['modules'] == before
    assert [d['source'] for d in event_data_for(store, 'reservation_installed', 'P')] == [
        'pin',
    ]


# ---------------------------------------------------------------------------
# Acceptance 4 (second bullet) — owner state releases through park GC
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_terminal_pin_is_released_by_park_gc(tmp_path):
    w = _3659_world(tmp_path)
    assert await w.scheduler.acquire_next() is None
    assert 'P' in w.scheduler.lock_table.snapshot_pin_reservations(), 'premise: P reserved'

    w.scheduler.get_tasks = AsyncMock(return_value=[
        make_task('P', 'high', ['w.py', 'x.py'], status='done'),
        *[t for t in _3659_tasks() if t['id'] != 'P'],
    ])
    await w.scheduler.acquire_next()

    assert event_data_for(w.store, 'reservation_expired', 'P') == [
        {'reason': 'terminal:done', 'source': 'pin'},
    ]
    assert {'restored_owner': 'C', 'modules': ['w.py'], 'source': 'fairness'} in [
        e['data'] for e in event_payloads(w.store, 'reservation_restored')
    ]
    assert w.scheduler.lock_table.snapshot_pin_reservations() == {}


@pytest.mark.asyncio
async def test_a_deps_unsatisfied_pin_is_released_by_park_gc(tmp_path):
    w = _3659_world(tmp_path)
    assert await w.scheduler.acquire_next() is None
    assert 'P' in w.scheduler.lock_table.snapshot_pin_reservations(), 'premise: P reserved'

    gated = make_task('P', 'high', ['w.py', 'x.py'])
    gated['dependencies'] = ['D']
    w.scheduler.get_tasks = AsyncMock(return_value=[
        gated,
        make_task('D', 'low', ['d.py']),
        *[t for t in _3659_tasks() if t['id'] != 'P'],
    ])
    await w.scheduler.acquire_next()

    assert event_data_for(w.store, 'reservation_expired', 'P') == [
        {'reason': 'deps_unsatisfied', 'source': 'pin'},
    ]


# ---------------------------------------------------------------------------
# Acceptance 6 (first half) — the kill switch is the old pin loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_switch_is_the_old_pin_loop(tmp_path):
    w = park_world(tmp_path, pinned=('P1', 'P2'), pin_reservations_enabled=False)
    assert w.scheduler.lock_table.try_acquire('H', ['w.py'])
    w.scheduler.get_tasks = AsyncMock(return_value=[
        make_task('P1', 'high', ['w.py']),
        make_task('P2', 'high', ['z.py']),
        make_task('H', 'low', ['w.py'], status='in-progress'),
    ])

    result = await w.scheduler.acquire_next()

    assert result is not None and result.task_id == 'P2', 'the next pin still dispatches'
    assert w.scheduler.lock_table.snapshot_pin_reservations() == {}
    assert _reservation_events(w.store, 'P1') == []
    assert not any(event_matches(t, 'pin_blocked') for t, _ in w.store.events)
