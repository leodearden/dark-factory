"""A dispatch ends its task's fairness episode on every dispatch path (task 5308).

Whether the pin loop or the scored loop dispatches a task, and whether it was
the scored top or not, its own parks are cleared at that moment
(``reservation_used``, plus ``reservation_restored`` for every shadow the clear
exposes).  A running task's park would otherwise keep blocking same-tier
installs (INV-3) until release, which is how a starving top ended up with a
partial park and lost its module to a lower tier (task 4541).

Every module is a single-segment ``.py`` file, so its lock key is its own name
at any ``lock_depth``.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple
from unittest.mock import AsyncMock

import pytest
from _park_test_helpers import event_data_for, event_index, event_payloads, make_task
from _recording_event_store import _RecordingEventStore

from orchestrator.config import PRIORITY_RANK, OrchestratorConfig
from orchestrator.overrides import OverrideStore
from orchestrator.scheduler import Scheduler


class _World(NamedTuple):
    scheduler: Scheduler
    store: _RecordingEventStore
    overrides: OverrideStore
    root: str


def _world(tmp_path: Path, *, pinned: tuple[str, ...] = ()) -> _World:
    config = OrchestratorConfig(max_per_module=1, lock_depth=2, project_root=tmp_path)
    config.fairness.skip_threshold = 1
    root = str(config.project_root)
    overrides = OverrideStore(tmp_path / 'o.db')
    for tid in pinned:
        overrides.set_override(root, tid, pinned=True)
    store = _RecordingEventStore()
    scheduler = Scheduler(config, event_store=store, override_store=overrides)  # type: ignore[arg-type]
    scheduler.finish_startup()
    return _World(scheduler, store, overrides, root)


def _stack_owners(scheduler: Scheduler) -> set[str]:
    return {
        entry['owner']
        for stack in scheduler.lock_table.snapshot_park_stacks().values()
        for entry in stack
    }


# ---------------------------------------------------------------------------
# Dispatch settles the dispatched task's own parks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_pin_dispatch_clears_its_own_park_and_restores_the_shadow(tmp_path):
    w = _world(tmp_path, pinned=('P',))
    w.scheduler.lock_table.install_parks('L', ['p1.py'], 'low')
    w.scheduler.lock_table.install_parks('P', ['p1.py', 'p2.py'], 'high')
    tasks = [
        make_task('P', 'high', ['p1.py', 'p2.py']),
        make_task('L', 'low', ['p1.py']),
        make_task('B', 'critical', ['b1.py']),
    ]
    w.scheduler.get_tasks = AsyncMock(return_value=tasks)

    result = await w.scheduler.acquire_next()

    assert result is not None and result.task_id == 'P', (
        'pin dispatch: the pin loop must dispatch P ahead of the critical B'
    )
    assert 'P' not in _stack_owners(w.scheduler), 'pin dispatch: P keeps no park entry'
    used = event_data_for(w.store, 'reservation_used', 'P')
    assert len(used) == 1, 'pin dispatch: exactly one reservation_used for P'
    assert used[0]['priority'] == 'high', 'pin dispatch: reservation_used priority'
    restored = [e['data'] for e in event_payloads(w.store, 'reservation_restored')]
    assert restored == [{'restored_owner': 'L', 'modules': ['p1.py']}], (
        "pin dispatch: clearing P's park restores L on p1"
    )
    assert event_index(w.store, 'reservation_used', 'P') < event_index(
        w.store, 'lock_acquired', 'P'
    ), 'pin dispatch: reservation_used precedes lock_acquired'


@pytest.mark.asyncio
async def test_a_non_top_scored_dispatch_clears_its_own_park(tmp_path):
    w = _world(tmp_path)
    assert w.scheduler.lock_table.try_acquire('seed', ['t1.py'])
    w.scheduler.lock_table.install_parks('N', ['n1.py'], 'medium')
    tasks = [make_task('T', 'critical', ['t1.py']), make_task('N', 'medium', ['n1.py'])]
    w.scheduler.get_tasks = AsyncMock(return_value=tasks)

    result = await w.scheduler.acquire_next()

    assert result is not None and result.task_id == 'N', 'non-top dispatch: N dispatches'
    assert 'N' not in _stack_owners(w.scheduler), 'non-top dispatch: N keeps no park entry'
    assert event_data_for(w.store, 'reservation_used', 'N') == [
        {'modules': ['n1.py'], 'priority': 'medium'},
    ], 'non-top dispatch: reservation_used for N'
    assert len(event_data_for(w.store, 'task_skipped', 'T')) == 1, (
        'non-top dispatch: the passed-over top T is bumped'
    )


@pytest.mark.asyncio
async def test_the_dispatched_tasks_park_no_longer_starves_the_top_in_the_same_tick(tmp_path):
    w = _world(tmp_path)
    assert w.scheduler.lock_table.try_acquire('seed', ['a.py'])
    w.scheduler.lock_table.install_parks('N', ['b.py'], 'critical')
    tasks = [make_task('T', 'critical', ['a.py', 'b.py']), make_task('N', 'high', ['b.py'])]
    w.scheduler.get_tasks = AsyncMock(return_value=tasks)

    result = await w.scheduler.acquire_next()

    assert result is not None and result.task_id == 'N', 'same tick: N dispatches (non-top)'
    parks = w.scheduler.lock_table.snapshot_parks()
    assert sorted(parks['T']['modules']) == ['a.py', 'b.py'], (
        "same tick: with N's park settled first, T parks both a and b"
    )
    assert event_data_for(w.store, 'reservation_install_blocked', 'T') == [], (
        'same tick: nothing of T is blocked'
    )


# ---------------------------------------------------------------------------
# Task 4541 replay
# ---------------------------------------------------------------------------
#
# The starved top and the pinned owner share the critical tier.  The top has
# the lower numeric id, so it is both older (larger age bonus) and first on
# the id tie-break: it is the scored top whenever both are candidates.

TOP = '4541'
PIN_OWNER = '4600'
MEDIUM = '4700'


@pytest.mark.asyncio
async def test_4541_replay_the_starved_top_keeps_cfg_from_the_medium_task(tmp_path):
    w = _world(tmp_path, pinned=(PIN_OWNER,))
    lock_table = w.scheduler.lock_table
    assert lock_table.try_acquire('H', ['x.py'])
    assert lock_table.try_acquire('Y', ['y.py'])
    lock_table.install_parks(PIN_OWNER, ['cfg.py'], 'critical')
    top = make_task(TOP, 'critical', ['cfg.py', 'x.py'])
    medium = make_task(MEDIUM, 'medium', ['cfg.py'])
    pin_owner = make_task(PIN_OWNER, 'critical', ['cfg.py', 'y.py'])
    w.scheduler.get_tasks = AsyncMock(return_value=[top, medium, pin_owner])

    assert await w.scheduler.acquire_next() is None, '4541 tick 1: nothing dispatches'
    assert lock_table.snapshot_parks()[TOP]['modules'] == ['x.py'], (
        "4541 tick 1: the pinned owner's park keeps cfg out of the top's install"
    )
    assert event_data_for(w.store, 'reservation_install_blocked', TOP)[0]['blocked'] == ['cfg.py'], (
        '4541 tick 1: the blocked install names cfg'
    )

    lock_table.release('Y')
    result = await w.scheduler.acquire_next()
    assert result is not None and result.task_id == PIN_OWNER, (
        '4541 tick 2: the pin loop dispatches the pinned owner'
    )
    assert len(event_data_for(w.store, 'reservation_used', PIN_OWNER)) == 1, (
        "4541 tick 2: the pin dispatch consumes the owner's park"
    )

    running_owner = make_task(PIN_OWNER, 'critical', ['cfg.py', 'y.py'], status='in-progress')
    w.scheduler.get_tasks = AsyncMock(return_value=[top, medium, running_owner])
    assert await w.scheduler.acquire_next() is None, '4541 tick 3: cfg is still held'
    assert event_data_for(w.store, 'reservation_installed', TOP)[-1]['modules'] == ['cfg.py'], (
        '4541 tick 3: the top completes its park onto cfg'
    )

    w.scheduler.release(PIN_OWNER)
    result = await w.scheduler.acquire_next()

    assert result is None or result.task_id != MEDIUM, (
        "4541 tick 4: the medium task must not take cfg from the starved top"
    )
    assert 'cfg.py' in lock_table.snapshot_parks()[TOP]['modules'], (
        '4541 end state: cfg is parked for the top'
    )
    assert lock_table.snapshot_holders().get('cfg.py') != MEDIUM, (
        '4541 end state: the medium task does not hold cfg'
    )


# ---------------------------------------------------------------------------
# reserve_now installs at the effective tier
# ---------------------------------------------------------------------------


def _a_entries(scheduler: Scheduler, module: str) -> list[dict]:
    stack = scheduler.lock_table.snapshot_park_stacks().get(module, [])
    return [entry for entry in stack if entry['owner'] == 'A']


@pytest.mark.asyncio
async def test_reserve_now_parks_at_the_boosted_tier(tmp_path):
    w = _world(tmp_path)
    assert w.scheduler.lock_table.try_acquire('seed', ['r1.py', 'r2.py'])
    w.overrides.set_override(w.root, 'A', boost_tier='critical', reserve_now=True)
    w.scheduler.get_tasks = AsyncMock(return_value=[make_task('A', 'medium', ['r1.py', 'r2.py'])])

    assert await w.scheduler.acquire_next() is None, 'reserve_now tier: A is held off'

    for module in ('r1.py', 'r2.py'):
        assert [e['rank'] for e in _a_entries(w.scheduler, module)] == [
            PRIORITY_RANK['critical']
        ], f'reserve_now tier: A parks {module} once, at the boosted rank'
    consumed = event_data_for(w.store, 'reserve_now_consumed', 'A')
    assert [d['priority'] for d in consumed] == ['critical'], (
        'reserve_now tier: reserve_now_consumed reports the effective tier'
    )
    installed, _ = w.scheduler.lock_table.install_parks('C', ['r1.py'], 'critical')
    assert installed == [], 'reserve_now tier: a same-tier competitor cannot shadow A'


@pytest.mark.asyncio
async def test_reserve_now_never_duplicates_a_park_the_owner_already_holds(tmp_path):
    w = _world(tmp_path)
    assert w.scheduler.lock_table.try_acquire('seed', ['r1.py', 'r2.py'])
    w.scheduler.lock_table.install_parks('A', ['r1.py'], 'critical')
    w.overrides.set_override(w.root, 'A', boost_tier='critical', reserve_now=True)
    w.scheduler.get_tasks = AsyncMock(return_value=[make_task('A', 'medium', ['r1.py', 'r2.py'])])

    assert await w.scheduler.acquire_next() is None, 'reserve_now dedupe: A is held off'

    assert [e['rank'] for e in _a_entries(w.scheduler, 'r1.py')] == [
        PRIORITY_RANK['critical']
    ], 'reserve_now dedupe: r1 keeps exactly one critical A entry'
    assert _a_entries(w.scheduler, 'r2.py'), 'reserve_now dedupe: A now parks r2'
    assert [d['modules'] for d in event_data_for(w.store, 'reserve_now_consumed', 'A')] == [
        ['r2.py']
    ], 'reserve_now dedupe: only the newly parked r2 is reported'


@pytest.mark.asyncio
async def test_reserve_now_raises_a_lower_tier_park_to_the_boosted_rank(tmp_path):
    w = _world(tmp_path)
    assert w.scheduler.lock_table.try_acquire('seed', ['r1.py', 'r2.py'])
    w.scheduler.lock_table.install_parks('A', ['r1.py'], 'medium')
    w.overrides.set_override(w.root, 'A', boost_tier='critical', reserve_now=True)
    w.scheduler.get_tasks = AsyncMock(return_value=[make_task('A', 'medium', ['r1.py', 'r2.py'])])

    assert await w.scheduler.acquire_next() is None, 'reserve_now upgrade: A is held off'

    for module in ('r1.py', 'r2.py'):
        assert [e['rank'] for e in _a_entries(w.scheduler, module)] == [
            PRIORITY_RANK['critical']
        ], f'reserve_now upgrade: A parks {module} once, at the boosted rank'
    assert [d['modules'] for d in event_data_for(w.store, 'reserve_now_consumed', 'A')] == [
        ['r1.py', 'r2.py']
    ], 'reserve_now upgrade: the re-ranked r1 is reported with the new r2'
    installed, _ = w.scheduler.lock_table.install_parks('C', ['r1.py'], 'high')
    assert installed == [], 'reserve_now upgrade: a high competitor cannot shadow A on r1'
