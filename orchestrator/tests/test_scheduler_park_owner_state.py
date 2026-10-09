"""One owner-state predicate decides whether an owner may hold parks (task 6518).

esc-6040-7 ruled that a park, pin reservation or fairness, lives only while
its owner is PENDING and no dependency of it (local, cross-project, delivered
check) is known to be unsatisfied.  An unknown dependency keeps the park:
expiry needs affirmative evidence.

Every module is a single-segment ``.py`` file, so its lock key is its own name.
Holders take their locks through ``lock_table.try_acquire`` and appear in the
task list as ``in-progress``: they own no parks and are never candidates.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from _park_test_helpers import (
    event_data_for,
    event_payloads,
    make_task,
    make_withheld_task,
    park_world,
)
from _recording_event_store import _RecordingEventStore

from orchestrator.config import OrchestratorConfig
from orchestrator.park_eviction_requests import ParkEvictionRequestStore
from orchestrator.scheduler import Scheduler, TickContext

EXTERNAL_DEP = 'other_project:99'


def _stack_owners(scheduler, module: str) -> list[str]:
    stacks = scheduler.lock_table.snapshot_park_stacks()
    return [entry['owner'] for entry in stacks.get(module, [])]


# ---------------------------------------------------------------------------
# D1 — only a pending owner may hold parks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['blocked', 'deferred', 'review', 'merge-deferred', 'in-progress'])
async def test_an_owner_that_leaves_pending_loses_its_parks_and_its_skip_count(tmp_path, status):
    w = park_world(tmp_path)
    assert w.scheduler.lock_table.try_acquire('H', ['m.py'])
    w.scheduler.lock_table.install_parks('L', ['m.py'], 'low')
    holder = make_task('H', 'low', ['m.py'], status='in-progress')
    shadowed = make_withheld_task('L', 'low', ['m.py'])
    w.scheduler.get_tasks = AsyncMock(
        return_value=[make_task('X', 'high', ['m.py']), shadowed, holder]
    )
    assert await w.scheduler.acquire_next() is None
    assert _stack_owners(w.scheduler, 'm.py') == ['L', 'X'], 'premise: X parks m above L'
    assert 'X' in w.scheduler.get_state_snapshot()['skip_counts'], 'premise: X was skipped'
    w.store.events.clear()

    w.scheduler.get_tasks = AsyncMock(
        return_value=[make_task('X', 'high', ['m.py'], status=status), shadowed, holder]
    )
    assert await w.scheduler.acquire_next() is None

    assert event_data_for(w.store, 'reservation_expired', 'X') == [
        {'reason': f'not_pending:{status}', 'source': 'fairness'},
    ]
    assert [e['data'] for e in event_payloads(w.store, 'reservation_restored')] == [
        {'restored_owner': 'L', 'modules': ['m.py'], 'source': 'fairness'},
    ]
    assert 'X' not in w.scheduler.lock_table.snapshot_parks()
    assert 'X' not in w.scheduler.get_state_snapshot()['skip_counts']


# ---------------------------------------------------------------------------
# D2 — a dependency known to be unsatisfied expires; an unknown one keeps
# ---------------------------------------------------------------------------
#
# Phase-level: owner A (pending) holds a park on m.py and park GC runs once
# against a hand-built TickContext, so each dependency arm's cache is set
# directly.


def _gc_world() -> tuple[Scheduler, _RecordingEventStore]:
    store = _RecordingEventStore()
    scheduler = Scheduler(
        OrchestratorConfig(max_per_module=1, lock_depth=2),
        event_store=store,  # type: ignore[arg-type]
    )
    scheduler.finish_startup()
    scheduler.lock_table.install_parks('A', ['m.py'], 'medium')
    return scheduler, store


def _ctx(*tasks: dict, **fields) -> TickContext:
    return TickContext(
        tasks=list(tasks),
        status_map={t['id']: t['status'] for t in tasks},
        tasks_by_id={t['id']: t for t in tasks},
        **fields,
    )


def _external_owner() -> dict:
    owner = make_task('A', 'medium', ['m.py'])
    owner['metadata']['external_deps'] = [EXTERNAL_DEP]
    return owner


def _delivered_check_ctx(delivered_check_cache: dict[str, bool] | None) -> TickContext:
    """A depends on the done D, whose record (outside the active fetch) carries a delivered check."""
    owner = make_task('A', 'medium', ['m.py'])
    owner['dependencies'] = ['D']
    producer = make_task('D', 'low', ['d.py'], status='done')
    producer['metadata']['delivered_checks'] = [
        {'name': 'c', 'kind': 'grep', 'pattern': 'x', 'paths': ['y']},
    ]
    ctx = _ctx(owner, delivered_check_cache=delivered_check_cache, terminal_dep_records={'D': producer})
    ctx.status_map['D'] = 'done'
    return ctx


async def _assert_gc_expires(ctx: TickContext, reason: str) -> None:
    scheduler, store = _gc_world()
    await scheduler._phase_park_gc(ctx)
    assert event_data_for(store, 'reservation_expired', 'A') == [
        {'reason': reason, 'source': 'fairness'},
    ]
    assert not scheduler.lock_table.has_parks('A')


async def _assert_gc_keeps(ctx: TickContext) -> None:
    scheduler, store = _gc_world()
    await scheduler._phase_park_gc(ctx)
    assert event_data_for(store, 'reservation_expired', 'A') == []
    assert scheduler.lock_table.has_parks('A')


@pytest.mark.asyncio
@pytest.mark.parametrize('reported', ['in-progress', 'cancelled', 'unknown_task'])
async def test_an_external_dependency_the_resolver_reports_not_done_expires_the_park(reported):
    ctx = _ctx(_external_owner(), external_cache={EXTERNAL_DEP: reported})
    await _assert_gc_expires(ctx, 'deps_unsatisfied:external')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('external_cache', 'resolver_failed'),
    [
        ({EXTERNAL_DEP: 'done'}, False),
        ({EXTERNAL_DEP: 'in-progress'}, True),
        ({}, True),
        ({}, False),
    ],
    ids=['done', 'resolver-failed-with-cache', 'resolver-failed', 'absent-from-cache'],
)
async def test_an_external_dependency_that_is_done_or_unknown_keeps_the_park(
    external_cache, resolver_failed
):
    ctx = _ctx(
        _external_owner(),
        external_cache=external_cache,
        external_resolver_failed=resolver_failed,
    )
    await _assert_gc_keeps(ctx)


@pytest.mark.asyncio
async def test_a_failed_delivered_check_expires_the_park():
    await _assert_gc_expires(_delivered_check_ctx({'D': False}), 'deps_unsatisfied:delivered_check')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'delivered_check_cache',
    [{'D': True}, {}, None],
    ids=['delivered', 'not-evaluated', 'gate-disabled'],
)
async def test_a_delivered_or_unevaluated_check_keeps_the_park(delivered_check_cache):
    await _assert_gc_keeps(_delivered_check_ctx(delivered_check_cache))


@pytest.mark.asyncio
async def test_an_unsatisfied_local_dependency_keeps_its_reason():
    owner = make_task('A', 'medium', ['m.py'])
    owner['dependencies'] = ['7']
    await _assert_gc_expires(
        _ctx(owner, make_task('7', 'low', ['s.py'], status='in-progress')), 'deps_unsatisfied'
    )


# ---------------------------------------------------------------------------
# D3 — park GC sees this tick's dependency caches
# ---------------------------------------------------------------------------


def _external_dep_world(tmp_path, resolved: tuple[dict[str, str], Exception | None]):
    """A pending A, held off m.py for good, parks it high; *resolved* is what the resolver answers."""
    w = park_world(tmp_path)
    assert w.scheduler.lock_table.try_acquire('H', ['m.py'])
    w.scheduler.lock_table.install_parks('A', ['m.py'], 'high')
    owner = make_task('A', 'high', ['m.py'])
    owner['metadata']['external_deps'] = [EXTERNAL_DEP]
    w.scheduler.get_tasks = AsyncMock(return_value=[
        owner,
        make_task('H', 'low', ['m.py'], status='in-progress'),
    ])
    w.scheduler.get_external_statuses = AsyncMock(return_value=resolved)
    return w


@pytest.mark.asyncio
async def test_a_tick_expires_the_park_of_an_owner_whose_external_dependency_is_not_done(tmp_path):
    w = _external_dep_world(tmp_path, ({EXTERNAL_DEP: 'in-progress'}, None))

    assert await w.scheduler.acquire_next() is None

    assert event_data_for(w.store, 'reservation_expired', 'A') == [
        {'reason': 'deps_unsatisfied:external', 'source': 'fairness'},
    ]
    assert 'A' not in w.scheduler.get_state_snapshot()['skip_counts']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'resolved',
    [({}, RuntimeError('resolver down')), ({EXTERNAL_DEP: 'done'}, None)],
    ids=['resolver-failed', 'done'],
)
async def test_a_tick_keeps_the_park_of_an_owner_whose_external_dependency_is_unknown_or_done(
    tmp_path, resolved
):
    w = _external_dep_world(tmp_path, resolved)

    assert await w.scheduler.acquire_next() is None

    assert event_data_for(w.store, 'reservation_expired', 'A') == []
    assert _stack_owners(w.scheduler, 'm.py') == ['A']


# ---------------------------------------------------------------------------
# D4 — the operator eviction guard and park GC never disagree
# ---------------------------------------------------------------------------
#
# Wherever park GC would expire an owner, the eviction drain (which runs
# first) evicts it; wherever park GC keeps it, the drain refuses.


def _eviction_world(tmp_path, owner: dict, resolved=({}, None)):
    """X holds a high fairness park on e.py and an eviction for X is queued."""
    evictions = ParkEvictionRequestStore(tmp_path / 'park_eviction_requests.db')
    w = park_world(tmp_path, park_eviction_store=evictions)
    w.scheduler.lock_table.install_parks('X', ['e.py'], 'high')
    evictions.enqueue('X', w.root)
    w.scheduler.get_tasks = AsyncMock(return_value=[owner])
    w.scheduler.get_external_statuses = AsyncMock(return_value=resolved)
    return w


def _owner_waiting_on_external_dep() -> dict:
    owner = make_task('X', 'high', ['e.py'])
    owner['metadata']['external_deps'] = [EXTERNAL_DEP]
    return owner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('owner', 'resolved'),
    [
        (make_task('X', 'high', ['e.py'], status='deferred'), ({}, None)),
        (_owner_waiting_on_external_dep(), ({EXTERNAL_DEP: 'in-progress'}, None)),
    ],
    ids=['deferred', 'external-dep-not-done'],
)
async def test_an_owner_park_gc_would_expire_is_force_evicted(tmp_path, owner, resolved):
    w = _eviction_world(tmp_path, owner, resolved)

    await w.scheduler.acquire_next()

    assert event_data_for(w.store, 'reservation_force_evicted', 'X') == [
        {'owner': 'X', 'modules': ['e.py'], 'source': 'fairness'},
    ]
    assert event_data_for(w.store, 'reservation_force_evict_refused', 'X') == []
    assert event_data_for(w.store, 'reservation_expired', 'X') == []


@pytest.mark.asyncio
async def test_an_owner_park_gc_keeps_on_an_unknown_dependency_is_refused_eviction(tmp_path):
    w = _eviction_world(
        tmp_path, _owner_waiting_on_external_dep(), ({}, RuntimeError('resolver down'))
    )

    await w.scheduler.acquire_next()

    assert event_data_for(w.store, 'reservation_force_evict_refused', 'X') == [
        {'reason': 'live_owner', 'source': 'fairness'},
    ]
    assert _stack_owners(w.scheduler, 'e.py') == ['X']
    assert event_data_for(w.store, 'reservation_expired', 'X') == []


# ---------------------------------------------------------------------------
# D6 — reserve_now installs only for an owner that may hold parks
# ---------------------------------------------------------------------------


def _owner_with_unsatisfied_local_dep() -> dict:
    owner = make_task('A', 'high', ['a.py'])
    owner['dependencies'] = ['7']
    return owner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('owner', 'reason'),
    [
        (make_task('A', 'high', ['a.py'], status='blocked'), 'not_pending:blocked'),
        (make_task('A', 'high', ['a.py'], status='deferred'), 'not_pending:deferred'),
        (make_task('A', 'high', ['a.py'], status='in-progress'), 'not_pending:in-progress'),
        (_owner_with_unsatisfied_local_dep(), 'deps_unsatisfied'),
    ],
    ids=['blocked', 'deferred', 'in-progress', 'local-dep-unsatisfied'],
)
async def test_reserve_now_on_an_owner_that_may_not_hold_parks_is_refused_once(
    tmp_path, owner, reason
):
    w = park_world(tmp_path)
    w.overrides.set_override(w.root, 'A', reserve_now=True)
    w.scheduler.get_tasks = AsyncMock(return_value=[
        owner,
        make_task('7', 'low', ['s.py'], status='in-progress'),
    ])

    await w.scheduler.acquire_next()

    assert event_data_for(w.store, 'reserve_now_refused', 'A') == [{'reason': reason}]
    assert event_data_for(w.store, 'reserve_now_consumed', 'A') == []
    assert 'A' not in w.scheduler.lock_table.snapshot_parks()
    stored = w.overrides.get_overrides(w.root).get('A')
    assert stored is None or not stored.reserve_now, 'the refused flag is consumed'

    w.store.events.clear()
    await w.scheduler.acquire_next()

    assert event_data_for(w.store, 'reserve_now_refused', 'A') == []
    assert event_data_for(w.store, 'reservation_expired', 'A') == []
