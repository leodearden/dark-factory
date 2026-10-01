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
from shared.psi import PsiSample

from orchestrator.config import PsiAdmissionConfig, apply_reload
from orchestrator.pin_reservation import pin_release_reason
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


def _psi_sample(*, cpu_some10: float) -> PsiSample:
    """A readable PSI sample: saturated when *cpu_some10* is past 85, else idle."""
    return PsiSample(
        cpu_some10=cpu_some10, mem_some10=0.0, mem_full10=0.0, io_some10=0.0, read_ok=True
    )


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


# ---------------------------------------------------------------------------
# Acceptance 4 (first and third bullets) — the pin phase releases and bounds
# ---------------------------------------------------------------------------


def _reload(scheduler: Scheduler, **changes: object) -> None:
    """Hot-apply *changes* to the LIVE config through the real ``apply_reload``.

    The fresh config is the live one with *changes* on top, so exactly those
    leaves differ and nothing else is re-applied.
    """
    result = apply_reload(scheduler.config, scheduler.config.model_copy(update=changes))
    assert result['reloaded'] is True, result
    assert set(changes) <= set(result['applied']), result


async def _reserved_3659_world(tmp_path, **config_overrides) -> ParkWorld:
    """The 3659 world after tick 1: P holds a pin reservation over C on w."""
    w = _3659_world(tmp_path, **config_overrides)
    assert await w.scheduler.acquire_next() is None
    assert _owners(w.scheduler, 'w.py') == ['C', 'P'], 'premise: P reserved over C'
    w.store.events.clear()
    return w


@pytest.mark.asyncio
async def test_unpinning_releases_within_one_tick(tmp_path):
    w = await _reserved_3659_world(tmp_path)

    assert w.overrides.clear_override(w.root, 'P', field='pinned')
    await w.scheduler.acquire_next()

    assert event_data_for(w.store, 'reservation_expired', 'P') == [
        {'reason': 'unpinned', 'source': 'pin'},
    ]
    assert {'restored_owner': 'C', 'modules': ['w.py'], 'source': 'fairness'} in [
        e['data'] for e in event_payloads(w.store, 'reservation_restored')
    ]
    assert w.scheduler.lock_table.snapshot_pin_reservations() == {}


@pytest.mark.asyncio
async def test_a_gated_pin_is_released(tmp_path):
    w = await _reserved_3659_world(tmp_path)

    w.scheduler._landed_outbox_gate = AsyncMock(side_effect=lambda tid: tid == 'P')
    await w.scheduler.acquire_next()

    assert event_data_for(w.store, 'reservation_expired', 'P') == [
        {'reason': 'gated', 'source': 'pin'},
    ]
    assert w.scheduler.lock_table.snapshot_pin_reservations() == {}


@pytest.mark.asyncio
async def test_a_lower_order_pin_displaces_the_head(tmp_path):
    """D5c: P1 outranks the reserved P2 the tick its dependency lands."""
    w = park_world(tmp_path, pinned=('P1', 'P2'))
    assert w.scheduler.lock_table.try_acquire('H', ['a.py'])
    p1 = make_task('P1', 'medium', ['a.py'])
    p1['dependencies'] = ['D']
    others = [
        make_task('P2', 'medium', ['a.py']),
        make_task('H', 'low', ['a.py'], status='in-progress'),
    ]
    w.scheduler.get_tasks = AsyncMock(
        return_value=[p1, make_task('D', 'low', ['d.py'], status='in-progress'), *others]
    )
    assert await w.scheduler.acquire_next() is None
    assert list(w.scheduler.lock_table.snapshot_pin_reservations()) == ['P2'], (
        'premise: P1 waits on D, so P2 is the head'
    )

    w.scheduler.get_tasks = AsyncMock(
        return_value=[p1, make_task('D', 'low', ['d.py'], status='done'), *others]
    )
    await w.scheduler.acquire_next()

    assert list(w.scheduler.lock_table.snapshot_pin_reservations()) == ['P1']
    assert event_data_for(w.store, 'reservation_expired', 'P2') == [
        {'reason': 'pin_displaced', 'source': 'pin'},
    ]
    assert _owners(w.scheduler, 'a.py')[-1] == 'P1', 'P1 tops the shared module'


@pytest.mark.asyncio
async def test_reloading_max_active_down_releases_the_excess(tmp_path):
    w = park_world(tmp_path, pinned=('P1', 'P2'), pin_reservation_max_active=2)
    assert w.scheduler.lock_table.try_acquire('H1', ['a.py'])
    assert w.scheduler.lock_table.try_acquire('H2', ['b.py'])
    w.scheduler.get_tasks = AsyncMock(return_value=[
        make_task('P1', 'medium', ['a.py']),
        make_task('P2', 'medium', ['b.py']),
        make_task('H1', 'low', ['a.py'], status='in-progress'),
        make_task('H2', 'low', ['b.py'], status='in-progress'),
    ])
    assert await w.scheduler.acquire_next() is None
    assert sorted(w.scheduler.lock_table.snapshot_pin_reservations()) == ['P1', 'P2']

    _reload(w.scheduler, pin_reservation_max_active=1)
    await w.scheduler.acquire_next()

    assert list(w.scheduler.lock_table.snapshot_pin_reservations()) == ['P1']
    assert event_data_for(w.store, 'reservation_expired', 'P2') == [
        {'reason': 'pin_displaced', 'source': 'pin'},
    ]
    assert event_data_for(w.store, 'reservation_expired', 'P1') == []


@pytest.mark.asyncio
async def test_hot_reload_toggles_the_kill_switch(tmp_path):
    w = await _reserved_3659_world(tmp_path)

    _reload(w.scheduler, pin_reservations_enabled=False)
    await w.scheduler.acquire_next()

    assert event_data_for(w.store, 'reservation_expired', 'P') == [
        {'reason': 'pin_reservations_disabled', 'source': 'pin'},
    ]
    assert {'restored_owner': 'C', 'modules': ['w.py'], 'source': 'fairness'} in [
        e['data'] for e in event_payloads(w.store, 'reservation_restored')
    ]
    assert event_payloads(w.store, 'reservation_installed') == []
    assert w.scheduler.lock_table.snapshot_pin_reservations() == {}

    _reload(w.scheduler, pin_reservations_enabled=True)
    await w.scheduler.acquire_next()

    assert 'P' in w.scheduler.lock_table.snapshot_pin_reservations()


@pytest.mark.asyncio
async def test_psi_held_tick_neither_grants_nor_revokes(tmp_path):
    """A held tick tries no acquire, so it has no evidence to revoke a head on.

    F is a free task that dispatches on tick 1, putting one task in flight so
    the hold's anti-deadlock floor does not suppress it.
    """
    w = _3659_world(tmp_path, psi_admission=PsiAdmissionConfig())
    w.scheduler.get_tasks = AsyncMock(
        return_value=[*_3659_tasks(), make_task('F', 'low', ['f.py'])]
    )
    result = await w.scheduler.acquire_next()
    assert result is not None and result.task_id == 'F', 'premise: F is in flight'
    before = w.scheduler.lock_table.snapshot_pin_reservations()
    assert 'P' in before, 'premise: P reserved'
    w.store.events.clear()

    w.scheduler._read_psi_sample = lambda: _psi_sample(cpu_some10=99.0)
    assert await w.scheduler.acquire_next() is None

    assert event_payloads(w.store, 'dispatch_deferred'), 'premise: the tick was held'
    assert w.scheduler.lock_table.snapshot_pin_reservations() == before
    assert event_payloads(w.store, 'reservation_expired') == []


@pytest.mark.parametrize(
    ('facts', 'reason'),
    [
        ({'enabled': False, 'reservable': True}, 'pin_reservations_disabled'),
        ({'reservable': True}, 'pin_displaced'),
        ({'pinned': False, 'gated': True}, 'unpinned'),
        ({'gated': True, 'deterministic': True}, 'gated'),
        ({'deterministic': True}, 'deterministic'),
        ({}, 'ineligible'),
    ],
)
def test_pin_release_reason_precedence(facts, reason):
    """disabled → displaced → unpinned → gated → deterministic → ineligible."""
    defaults = {
        'enabled': True, 'reservable': False, 'pinned': True,
        'gated': False, 'deterministic': False,
    }

    assert pin_release_reason(**(defaults | facts)) == reason


# ---------------------------------------------------------------------------
# Acceptance 7 / D7ii — pin_blocked names a blocked pin's blockers
# ---------------------------------------------------------------------------


class _Clock:
    """A settable monotonic clock: the Scheduler's ``time_source``."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _two_blocked_pins_world(tmp_path, clock: _Clock) -> ParkWorld:
    """P1 waits on H's a; P2 waits on a too and on H2's b."""
    w = park_world(tmp_path, pinned=('P1', 'P2'), time_source=clock)
    assert w.scheduler.lock_table.try_acquire('H', ['a.py'])
    assert w.scheduler.lock_table.try_acquire('H2', ['b.py'])
    w.scheduler.get_tasks = AsyncMock(return_value=[
        make_task('P1', 'medium', ['a.py']),
        make_task('P2', 'medium', ['a.py', 'b.py']),
        make_task('H', 'low', ['a.py'], status='in-progress'),
        make_task('H2', 'low', ['b.py'], status='in-progress'),
    ])
    return w


def _one_blocked_pin_world(
    tmp_path, clock: _Clock, *extra: dict, **config_overrides
) -> ParkWorld:
    """P1 waits on H's a; *extra* tasks join the task list."""
    w = park_world(tmp_path, pinned=('P1',), time_source=clock, **config_overrides)
    assert w.scheduler.lock_table.try_acquire('H', ['a.py'])
    w.scheduler.get_tasks = AsyncMock(return_value=[
        make_task('P1', 'medium', ['a.py']),
        make_task('H', 'low', ['a.py'], status='in-progress'),
        *extra,
    ])
    return w


@pytest.mark.asyncio
async def test_pin_blocked_names_each_pins_blockers_on_transition(tmp_path):
    """The head's blockers are read BEFORE it reserves, so it never names itself."""
    w = _two_blocked_pins_world(tmp_path, _Clock())

    assert await w.scheduler.acquire_next() is None

    assert event_data_for(w.store, 'pin_blocked', 'P1') == [{
        'task_id': 'P1',
        'pin_order': 1,
        'head': True,
        'blockers': [{'module': 'a.py', 'owner': 'H', 'kind': 'held'}],
    }]
    assert event_data_for(w.store, 'pin_blocked', 'P2') == [{
        'task_id': 'P2',
        'pin_order': 2,
        'head': False,
        'blockers': [
            {'module': 'a.py', 'owner': 'H', 'kind': 'held'},
            {'module': 'a.py', 'owner': 'P1', 'kind': 'parked'},
            {'module': 'b.py', 'owner': 'H2', 'kind': 'held'},
        ],
    }]


@pytest.mark.asyncio
async def test_pin_blocked_is_rate_limited_per_pin(tmp_path):
    clock = _Clock()
    w = _two_blocked_pins_world(tmp_path, clock)
    interval = w.scheduler.config.pin_blocked_emit_interval_secs
    assert await w.scheduler.acquire_next() is None

    clock.now += interval - 1
    await w.scheduler.acquire_next()

    assert len(event_payloads(w.store, 'pin_blocked')) == 2, 'inside the interval: silent'

    clock.now += 2
    await w.scheduler.acquire_next()

    assert len(event_data_for(w.store, 'pin_blocked', 'P1')) == 2
    assert len(event_data_for(w.store, 'pin_blocked', 'P2')) == 2


@pytest.mark.asyncio
async def test_pin_blocked_fires_again_after_a_dispatch(tmp_path):
    """A dispatch ends the blocked episode; the next block is a new transition."""
    w = _one_blocked_pin_world(tmp_path, _Clock())
    assert await w.scheduler.acquire_next() is None
    _drop(w, 'H')
    result = await w.scheduler.acquire_next()
    assert result is not None and result.task_id == 'P1', 'premise: P1 dispatched'

    w.scheduler.release('P1')
    assert w.scheduler.lock_table.try_acquire('H3', ['a.py'])
    w.scheduler.get_tasks = AsyncMock(return_value=[
        make_task('P1', 'medium', ['a.py']),
        make_task('H3', 'low', ['a.py'], status='in-progress'),
    ])
    assert await w.scheduler.acquire_next() is None

    assert [d['blockers'] for d in event_data_for(w.store, 'pin_blocked', 'P1')] == [
        [{'module': 'a.py', 'owner': 'H', 'kind': 'held'}],
        [{'module': 'a.py', 'owner': 'H3', 'kind': 'held'}],
    ]


@pytest.mark.asyncio
async def test_a_psi_held_tick_neither_emits_nor_resets_pin_blocked(tmp_path):
    """F dispatches on tick 1 so one task is in flight and the hold can engage."""
    w = _one_blocked_pin_world(
        tmp_path, _Clock(), make_task('F', 'low', ['f.py']),
        psi_admission=PsiAdmissionConfig(),
    )
    result = await w.scheduler.acquire_next()
    assert result is not None and result.task_id == 'F', 'premise: F is in flight'
    assert len(event_data_for(w.store, 'pin_blocked', 'P1')) == 1

    w.scheduler._read_psi_sample = lambda: _psi_sample(cpu_some10=99.0)
    await w.scheduler.acquire_next()

    assert event_payloads(w.store, 'dispatch_deferred'), 'premise: the tick was held'
    assert len(event_data_for(w.store, 'pin_blocked', 'P1')) == 1

    w.scheduler._read_psi_sample = lambda: _psi_sample(cpu_some10=0.0)
    await w.scheduler.acquire_next()

    assert len(event_data_for(w.store, 'pin_blocked', 'P1')) == 1, (
        'still blocked, still inside the interval: the hold did not reset it'
    )


@pytest.mark.asyncio
async def test_a_shorter_reloaded_interval_applies_next_tick(tmp_path):
    clock = _Clock()
    w = _one_blocked_pin_world(tmp_path, clock)
    assert await w.scheduler.acquire_next() is None

    clock.now += 100
    _reload(w.scheduler, pin_blocked_emit_interval_secs=50.0)
    await w.scheduler.acquire_next()

    assert len(event_data_for(w.store, 'pin_blocked', 'P1')) == 2
