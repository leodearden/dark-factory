"""A candidate refused its locks by a peer's park is named in ``park_refused`` (task 6328).

Only the scored scan's strict top earns skips and parks, so a non-top candidate
refused by a same-tier (or weaker) park earns nothing while it waits.  The
scheduler names that wait once it has lasted ``park_refused_emit_interval_secs``
and then once per interval; pinning the waiting task is the operator lever.

Every world keeps the waiting candidate NON-top: ``park_world`` parks the top's
full footprint on its first skip, so an older refused critical task always
outranks it.  Parks the tests rely on are installed directly.  Every module is
a single-segment ``.py`` file, so its lock key is its own name.  Holders take
their locks through ``lock_table.try_acquire`` and appear in the task list as
``in-progress``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from _park_test_helpers import ParkWorld, event_data_for, event_payloads, make_task, park_world

from orchestrator.lock_table import ModuleLockTable
from orchestrator.main_health_fix_owner import SIGNATURE_KEY
from orchestrator.run_slots import DispatchScope

A = '5221'  # the refused critical top, whose park blocks B
B = '6040'  # the waiting critical candidate
H = '3096'  # A's holder
H2 = '3097'
L = '7000'
P = '7100'
F = '8000'


class _Clock:
    """A settable monotonic clock: the Scheduler's ``time_source``."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _TaskList:
    """The task list the test owns; the scheduler's ``get_tasks`` returns a copy of it."""

    def __init__(self, world: ParkWorld) -> None:
        self.rows: list[dict] = []
        self._lock_table = world.scheduler.lock_table
        world.scheduler.get_tasks = AsyncMock(side_effect=lambda *args, **kwargs: list(self.rows))

    def set(self, rows: list[dict]) -> None:
        self.rows[:] = rows

    def hold(self, holder: str, module: str) -> None:
        """*holder* takes *module* and joins the task list as in-progress."""
        assert self._lock_table.try_acquire(holder, [module])
        self.rows.append(make_task(holder, 'low', [module], status='in-progress'))

    def drop(self, *task_ids: str) -> None:
        """Release each task's locks and remove it from the task list."""
        for tid in task_ids:
            self._lock_table.release(tid)
        self.rows[:] = [t for t in self.rows if t['id'] not in task_ids]

    def add(self, *rows: dict) -> None:
        self.rows.extend(rows)


def _peer_tasks(b_tier: str = 'critical') -> list[dict]:
    return [
        make_task(A, 'critical', ['x.py', 'z.py']),
        make_task(B, b_tier, ['x.py', 'y.py']),
        make_task(H, 'low', ['z.py'], status='in-progress'),
    ]


def _peer_park_world(
    tmp_path, clock: _Clock, *, b_tier: str = 'critical',
) -> tuple[ParkWorld, _TaskList]:
    """The incident: B waits on A's same-tier park on x, and A waits on H's hold of z."""
    w = park_world(tmp_path, time_source=clock)
    assert w.scheduler.lock_table.try_acquire(H, ['z.py'])
    w.scheduler.lock_table.install_parks(A, ['x.py', 'z.py'], 'critical')
    tasks = _TaskList(w)
    tasks.set(_peer_tasks(b_tier))
    return w, tasks


def _interval(w: ParkWorld) -> float:
    return w.scheduler.config.park_refused_emit_interval_secs


def _refusals(w: ParkWorld, task_id: str = B) -> list[dict]:
    return event_data_for(w.store, 'park_refused', task_id)


def _top(lock_table: ModuleLockTable, module: str) -> str | None:
    stack = lock_table.snapshot_park_stacks().get(module, [])
    return stack[-1]['owner'] if stack else None


def _without_park_ages(data: dict) -> dict:
    """*data* without its owners' ``park_age_secs``, each checked to be a non-negative number.

    Park ages are wall time, which is not under the test's clock.
    """
    owners = []
    for owner in data['park_owners']:
        age = owner['park_age_secs']
        assert isinstance(age, int | float) and age >= 0, owner
        owners.append({key: value for key, value in owner.items() if key != 'park_age_secs'})
    return {**data, 'park_owners': owners}


async def _idle_ticks(w: ParkWorld, clock: _Clock, *, count: int, step: float) -> None:
    for _ in range(count):
        assert await w.scheduler.acquire_next() is None
        clock.now += step


# ---------------------------------------------------------------------------
# Acceptance 1 + 2 — the 6040 replay
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_6040_replay_peer_park_is_named_and_the_pin_lever_dispatches_b(tmp_path):
    clock = _Clock()
    w, _ = _peer_park_world(tmp_path, clock)
    lock_table = w.scheduler.lock_table

    def assert_h_and_a_keep_z() -> None:
        assert lock_table.snapshot_holders()['z.py'] == H, 'H keeps its held lock on z'
        assert _top(lock_table, 'z.py') == A, "A's park on z is never disturbed"

    assert await w.scheduler.acquire_next() is None
    assert _refusals(w) == [], 'the first refusal only starts the episode'
    assert_h_and_a_keep_z()

    clock.now += _interval(w)
    assert await w.scheduler.acquire_next() is None
    assert_h_and_a_keep_z()

    refusals = _refusals(w)
    assert len(refusals) == 1
    assert _without_park_ages(refusals[0]) == {
        'task_id': B,
        'priority': 'critical',
        'episode_secs': _interval(w),
        'blockers': [{'module': 'x.py', 'owner': A, 'kind': 'parked'}],
        'park_owners': [{
            'owner': A,
            'source': 'fairness',
            'tier': 'critical',
            'waiting_on': [{'module': 'z.py', 'owner': H, 'kind': 'held'}],
        }],
    }

    w.overrides.set_override(w.root, B, pinned=True)
    result = None
    for _ in range(2):
        result = await w.scheduler.acquire_next()
        assert_h_and_a_keep_z()
        if result is not None:
            break

    assert result is not None and result.task_id == B, 'the pin lever dispatches B'
    assert set(result.modules) == {'x.py', 'y.py'}
    assert _top(lock_table, 'x.py') == A, "A's park on x is restored at B's dispatch"
    assert len(_refusals(w)) == 1


@pytest.mark.asyncio
async def test_park_refused_repeats_once_per_interval(tmp_path):
    clock = _Clock()
    w, _ = _peer_park_world(tmp_path, clock)
    interval = _interval(w)
    assert await w.scheduler.acquire_next() is None
    clock.now += interval
    assert await w.scheduler.acquire_next() is None
    assert len(_refusals(w)) == 1, 'premise: the first event came after one interval'

    clock.now += interval - 1
    assert await w.scheduler.acquire_next() is None
    assert len(_refusals(w)) == 1, 'inside the interval since the last emission: silent'

    clock.now += 1
    assert await w.scheduler.acquire_next() is None
    assert [d['episode_secs'] for d in _refusals(w)] == [interval, 2 * interval], (
        'the age is measured from the episode start'
    )


# ---------------------------------------------------------------------------
# The class: parks alone, none outranking the candidate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_lower_tier_park_blocking_a_critical_candidate_is_named(tmp_path):
    clock = _Clock()
    w = park_world(tmp_path, time_source=clock)
    lock_table = w.scheduler.lock_table
    assert lock_table.try_acquire(H, ['z.py'])
    assert lock_table.try_acquire(H2, ['q.py'])
    lock_table.install_parks(A, ['w.py', 'z.py'], 'critical')
    lock_table.install_parks(L, ['x.py'], 'medium')
    _TaskList(w).set([
        make_task(A, 'critical', ['w.py', 'z.py']),
        make_task(B, 'critical', ['x.py', 'y.py']),
        make_task(L, 'medium', ['x.py', 'q.py']),
        make_task(H, 'low', ['z.py'], status='in-progress'),
        make_task(H2, 'low', ['q.py'], status='in-progress'),
    ])

    await _idle_ticks(w, clock, count=2, step=_interval(w))

    assert [_without_park_ages(d) for d in _refusals(w)] == [{
        'task_id': B,
        'priority': 'critical',
        'episode_secs': _interval(w),
        'blockers': [{'module': 'x.py', 'owner': L, 'kind': 'parked'}],
        'park_owners': [{
            'owner': L,
            'source': 'fairness',
            'tier': 'medium',
            'waiting_on': [{'module': 'q.py', 'owner': H2, 'kind': 'held'}],
        }],
    }]


@pytest.mark.asyncio
async def test_a_higher_tier_park_never_emits(tmp_path):
    clock = _Clock()
    w, _ = _peer_park_world(tmp_path, clock, b_tier='high')

    await _idle_ticks(w, clock, count=4, step=_interval(w))

    assert _refusals(w) == [], "A's critical park outranks a high B: fairness as designed"


@pytest.mark.asyncio
async def test_a_held_blocker_suppresses_park_refused(tmp_path):
    clock = _Clock()
    w, tasks = _peer_park_world(tmp_path, clock)
    tasks.hold(H2, 'y.py')

    await _idle_ticks(w, clock, count=4, step=_interval(w))

    assert _refusals(w) == [], 'pinning cannot help past a held lock'


@pytest.mark.asyncio
async def test_a_pin_reservation_blocker_suppresses_park_refused(tmp_path):
    clock = _Clock()
    w = park_world(tmp_path, pinned=(P,), time_source=clock)
    assert w.scheduler.lock_table.try_acquire(H, ['z.py'])
    assert w.scheduler.lock_table.try_acquire(H2, ['q.py'])
    _TaskList(w).set([
        make_task(A, 'critical', ['x.py', 'z.py']),
        make_task(B, 'critical', ['x.py', 'y.py']),
        make_task(P, 'low', ['x.py', 'q.py']),
        make_task(H, 'low', ['z.py'], status='in-progress'),
        make_task(H2, 'low', ['q.py'], status='in-progress'),
    ])

    await _idle_ticks(w, clock, count=4, step=_interval(w))

    assert _top(w.scheduler.lock_table, 'x.py') == P, "premise: P's pin reservation tops x"
    assert _refusals(w) == [], 'a pin reservation already reports through pin_blocked'


# ---------------------------------------------------------------------------
# The episode: what ends it and what keeps it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_held_refusal_ends_the_episode(tmp_path):
    clock = _Clock()
    w, tasks = _peer_park_world(tmp_path, clock)
    interval = _interval(w)
    t0 = clock.now
    assert await w.scheduler.acquire_next() is None

    clock.now = t0 + interval / 2
    tasks.hold(H2, 'y.py')
    assert await w.scheduler.acquire_next() is None
    tasks.drop(H2)

    clock.now = t0 + interval
    assert await w.scheduler.acquire_next() is None
    assert _refusals(w) == [], 'the held refusal ended the episode that was due now'

    clock.now += interval - 1
    assert await w.scheduler.acquire_next() is None
    assert _refusals(w) == []

    clock.now += 1
    assert await w.scheduler.acquire_next() is None
    assert [d['episode_secs'] for d in _refusals(w)] == [interval]


@pytest.mark.asyncio
async def test_dispatch_ends_the_episode(tmp_path):
    clock = _Clock()
    w, tasks = _peer_park_world(tmp_path, clock)
    interval = _interval(w)
    t0 = clock.now
    assert await w.scheduler.acquire_next() is None

    clock.now = t0 + interval / 2
    tasks.set([t for t in _peer_tasks() if t['id'] != A])
    result = await w.scheduler.acquire_next()
    assert result is not None and result.task_id == B, "premise: A's park expired and B dispatched"

    w.scheduler.release(B)
    tasks.set(_peer_tasks())
    w.scheduler.lock_table.install_parks(A, ['x.py', 'z.py'], 'critical')

    clock.now = t0 + interval
    assert await w.scheduler.acquire_next() is None
    assert _refusals(w) == [], 'the dispatch ended the episode that was due now'

    clock.now += interval - 1
    assert await w.scheduler.acquire_next() is None
    assert _refusals(w) == []

    clock.now += 1
    assert await w.scheduler.acquire_next() is None
    assert [d['episode_secs'] for d in _refusals(w)] == [interval]


def _fix_task(tid: str, files: list[str]) -> dict:
    task = make_task(tid, 'high', files)
    task['metadata'].update({SIGNATURE_KEY: f'sig-{tid}', 'merge_lane': 'high'})
    return task


@pytest.mark.parametrize('admitted_fix', [False, True])
@pytest.mark.asyncio
async def test_a_scoped_tick_keeps_the_episode(tmp_path, admitted_fix):
    """A MAIN_HEALTH_FIX tick (task 8480) filters the candidates; it must not end other episodes."""
    clock = _Clock()
    w, tasks = _peer_park_world(tmp_path, clock)
    if admitted_fix:
        tasks.hold(H2, 'f.py')
        tasks.add(_fix_task(F, ['f.py']))
    interval = _interval(w)
    t0 = clock.now
    assert await w.scheduler.acquire_next() is None

    clock.now = t0 + interval / 2
    assert await w.scheduler.acquire_next(DispatchScope.MAIN_HEALTH_FIX) is None
    last_tick = event_payloads(w.store, 'scheduler_tick')[-1]['data']
    expected_phase = 'select_scored' if admitted_fix else 'empty_candidate_gate'
    assert last_tick['terminal_phase'] == expected_phase, (
        f'premise: the scoped tick ended at {expected_phase}'
    )

    clock.now = t0 + interval
    assert await w.scheduler.acquire_next() is None

    assert [d['episode_secs'] for d in _refusals(w)] == [interval]
