"""Promotion of held follow-ups through followup_hold's public seam (plans/follow-up-filing-shape-prd.md §4.3).

A held task leaves the hold by one backend write that drops its holder edge and stamps
``followup_hold.promoted``; promoting a task promotes every held task it depends on. Every scenario
runs against the real SQLite store, holds no write lock (the caller-holds-lock precondition is the
interceptor's), and records what the seam emits.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pytest
import pytest_asyncio
from _submit_stack import Stack
from _submit_stack import passthrough_main_checkout as passthrough_main_checkout
from _submit_stack import stack as stack
from shared.followup_hold_records import FOLLOWUP_HOLD_METADATA_KEY

from fused_memory.middleware.followup_hold import (
    FollowupHolders,
    PromotionDeferral,
    hold_at_create,
    promote,
    promote_dependencies_of,
    register_promotion_guard,
    release_held,
    unregister_promotion_guard,
)
from fused_memory.models.reconciliation import EventType

_FOLLOWUP = {'source': 'agent-followup', 'spawn_context': 'agent_followup', 'spawned_from': '101'}
_SET_MEMBER = {
    'source': 'review-all', 'x_finding_key': ['fk-1'], 'finding_set_hold': {'ruling': 'Leo 2026-11-12 on fk-1'},
}
_HELD_AT = '2026-11-19T00:00:00+00:00'


async def _unannounced(project_root: str, holder_id: int) -> None:
    return None


@dataclass
class _Store:
    stack: Stack
    holder: int
    emitted: list[tuple[EventType, dict[str, Any]]] = field(default_factory=list)

    async def emit(self, event_type: EventType, payload: Mapping[str, object]) -> None:
        self.emitted.append((event_type, dict(payload)))

    def events(self, event_type: EventType) -> list[dict[str, Any]]:
        return [payload for emitted_type, payload in self.emitted if emitted_type == event_type]

    async def held(
        self, title: str, *, dependencies: Sequence[int] = (), status: str = 'pending',
        metadata: Mapping[str, Any] = _FOLLOWUP,
    ) -> int:
        kwargs, held_metadata = hold_at_create(
            {'title': title, 'description': f'{title}: one observation.', 'priority': 'low',
             'dependencies': ','.join(str(dependency) for dependency in dependencies)},
            metadata, self.holder, held_at=_HELD_AT,
        )
        return await self._add(metadata=held_metadata, status=status, **kwargs)

    async def plain(self, title: str, *, dependencies: Sequence[int] = (), status: str = 'pending') -> int:
        return await self._add(
            title=title, description=f'{title}: one task.', priority='medium',
            dependencies=','.join(str(dependency) for dependency in dependencies),
            metadata={'files': ['src/app/a.py']}, status=status,
        )

    async def _add(self, *, metadata: Mapping[str, Any], **fields: Any) -> int:
        created = await self.stack.backend.add_task(
            project_root=self.stack.root, metadata=json.dumps(metadata), **fields,
        )
        return int(created['id'])

    async def row(self, task_id: int) -> dict[str, Any]:
        return await self.stack.backend.get_task(str(task_id), self.stack.root)

    async def dependencies(self, task_id: int) -> list[int]:
        return [int(dependency) for dependency in (await self.row(task_id)).get('dependencies') or []]

    async def record(self, task_id: int) -> dict[str, Any]:
        return (await self.row(task_id))['metadata'][FOLLOWUP_HOLD_METADATA_KEY]

    async def promote(self, task_id: int, *, by: str = 'citation', ref: str | None = 'fk-1') -> tuple[int, ...]:
        return await promote(self.stack.backend, self.stack.root, task_id, by=by, ref=ref, emit=self.emit)


@pytest_asyncio.fixture
async def store(stack: Stack) -> _Store:
    holder = await FollowupHolders(on_minted=_unannounced).ensure_holder(stack.backend, stack.root)
    return _Store(stack=stack, holder=holder)


async def _assert_held(store: _Store, task_id: int) -> None:
    assert store.holder in await store.dependencies(task_id), task_id
    assert 'promoted' not in await store.record(task_id), task_id


@pytest.mark.asyncio
async def test_a_citation_drops_the_holder_edge_and_stamps_the_promotion(store: _Store) -> None:
    other = await store.plain('Upstream fix')
    held = await store.held('The retry loop swallows the timeout', dependencies=[other])
    pending_since_before = (await store.row(held))['metadata'].get('pending_since')

    promoted = await store.promote(held)

    assert promoted == (held,)
    assert await store.dependencies(held) == [other]
    record = await store.record(held)
    assert record['holder'] == store.holder
    assert record['held_at'] == _HELD_AT
    assert {key: value for key, value in record['promoted'].items() if key != 'at'} == {
        'by': 'citation', 'ref': 'fk-1',
    }
    datetime.fromisoformat(record['promoted']['at'])
    assert (await store.row(held))['metadata'].get('pending_since') == pending_since_before
    assert store.emitted == [
        (EventType.followup_promoted, {'task_id': str(held), 'by': 'citation', 'ref': 'fk-1'}),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize('shape', ['unheld', 'cancelled-held', 'missing'])
async def test_promoting_what_is_not_held_writes_and_emits_nothing(store: _Store, shape: str) -> None:
    if shape == 'unheld':
        task_id = await store.plain('Not a follow-up')
    elif shape == 'cancelled-held':
        task_id = await store.held('A cancelled follow-up', status='cancelled')
    else:
        task_id = 9999
    before = None if shape == 'missing' else await store.row(task_id)

    assert await store.promote(task_id) == ()

    if before is not None:
        assert await store.row(task_id) == before
    assert store.emitted == []


@pytest.mark.asyncio
async def test_promoting_a_task_promotes_the_held_task_it_depends_on(store: _Store) -> None:
    upstream = await store.held('The cache key ignores the tenant')
    unheld = await store.plain('Upstream fix')
    unheld_before = await store.row(unheld)
    dependent = await store.held('The tenant leaks into the log', dependencies=[upstream, unheld])

    assert await store.promote(dependent) == (dependent, upstream)

    upstream_record = await store.record(upstream)
    assert upstream_record['promoted']['by'] == 'dependent'
    assert upstream_record['promoted']['ref'] == str(dependent)
    assert await store.dependencies(upstream) == []
    assert store.events(EventType.followup_promoted) == [
        {'task_id': str(dependent), 'by': 'citation', 'ref': 'fk-1'},
        {'task_id': str(upstream), 'by': 'dependent', 'ref': str(dependent)},
    ]
    assert await store.row(unheld) == unheld_before


@pytest.mark.asyncio
async def test_a_diamond_promotes_its_shared_dependency_once(store: _Store) -> None:
    shared_dependency = await store.held('Shared root cause')
    left = await store.held('Left symptom', dependencies=[shared_dependency])
    right = await store.held('Right symptom', dependencies=[shared_dependency])
    top = await store.held('Combined symptom', dependencies=[left, right])

    promoted = await store.promote(top)

    assert sorted(promoted) == sorted([top, left, right, shared_dependency])
    assert promoted.count(shared_dependency) == 1
    assert [event['task_id'] for event in store.events(EventType.followup_promoted)].count(
        str(shared_dependency),
    ) == 1
    for task_id in promoted:
        assert store.holder not in await store.dependencies(task_id), task_id


@pytest.mark.asyncio
async def test_an_unknown_cause_is_refused(store: _Store) -> None:
    held = await store.held('The retry loop swallows the timeout')

    with pytest.raises(ValueError):
        await store.promote(held, by='age_out')

    await _assert_held(store, held)


@pytest.mark.asyncio
async def test_a_pending_unheld_dependent_promotes_its_held_dependency(store: _Store) -> None:
    held = await store.held('The retry loop swallows the timeout')
    dependent = await store.plain('Ship the retry budget', dependencies=[held])

    promoted = await promote_dependencies_of(store.stack.backend, store.stack.root, dependent, emit=store.emit)

    assert promoted == (held,)
    record = await store.record(held)
    assert record['promoted']['by'] == 'dependent'
    assert record['promoted']['ref'] == str(dependent)
    assert store.holder not in await store.dependencies(held)


@pytest.mark.asyncio
@pytest.mark.parametrize('shape', ['held', 'deferred', 'in-progress'])
async def test_a_dependent_that_is_not_waiting_unheld_promotes_nothing(store: _Store, shape: str) -> None:
    held = await store.held('The retry loop swallows the timeout')
    if shape == 'held':
        dependent = await store.held('Another follow-up', dependencies=[held])
    else:
        dependent = await store.plain('Ship the retry budget', dependencies=[held], status=shape)

    promoted = await promote_dependencies_of(store.stack.backend, store.stack.root, dependent, emit=store.emit)

    assert promoted == ()
    await _assert_held(store, held)
    assert store.emitted == []


@pytest.mark.asyncio
async def test_releasing_the_holder_promotes_every_active_held_task_by_human(store: _Store) -> None:
    first = await store.held('First follow-up')
    second = await store.held('Second follow-up')
    third = await store.held('Third follow-up')
    await store.stack.backend.add_dependency(str(first), str(third), store.stack.root)
    cancelled = await store.held('Cancelled follow-up', status='cancelled')

    promoted = await release_held(store.stack.backend, store.stack.root, store.holder, emit=store.emit)

    assert sorted(promoted) == sorted([first, second, third])
    for task_id in (first, second, third):
        record = await store.record(task_id)
        assert record['promoted']['by'] == 'human', task_id
        assert record['promoted']['ref'] == 'release_holder', task_id
    events = store.events(EventType.followup_promoted)
    assert len(events) == 3
    assert {event['by'] for event in events} == {'human'}
    assert 'promoted' not in await store.record(cancelled)
    edges = await store.stack.backend.get_dependency_edges(store.stack.root)
    assert [task_id for task_id, dependencies in edges.items() if store.holder in dependencies] == [cancelled]


@dataclass
class _StubGuard:
    deferred: set[int] = field(default_factory=set)
    calls: list[tuple[dict[str, Any], str, str | None]] = field(default_factory=list)

    def __call__(self, task: Mapping[str, object], by: str, ref: str | None) -> PromotionDeferral | None:
        self.calls.append((dict(task), by, ref))
        if int(str(task['id'])) in self.deferred:
            return PromotionDeferral(guard='stub-guard', reason='riders still open')
        return None


@pytest.fixture
def guard() -> Iterator[_StubGuard]:
    stub = _StubGuard()
    register_promotion_guard(stub)
    try:
        yield stub
    finally:
        unregister_promotion_guard(stub)


@pytest.mark.asyncio
async def test_a_guard_deferral_leaves_the_task_held_and_says_why(store: _Store, guard: _StubGuard) -> None:
    held = await store.held('The retry loop swallows the timeout')
    guard.deferred.add(held)
    live_row = await store.row(held)

    assert await store.promote(held) == ()

    await _assert_held(store, held)
    assert guard.calls == [(live_row, 'citation', 'fk-1')]
    assert store.emitted == [(EventType.followup_promotion_deferred, {
        'task_id': str(held), 'by': 'citation', 'ref': 'fk-1',
        'guard': 'stub-guard', 'reason': 'riders still open',
    })]


@pytest.mark.asyncio
async def test_a_deferred_task_does_not_promote_its_held_dependency(store: _Store, guard: _StubGuard) -> None:
    upstream = await store.held('The cache key ignores the tenant')
    dependent = await store.held('The tenant leaks into the log', dependencies=[upstream])
    guard.deferred.add(dependent)

    assert await store.promote(dependent) == ()

    await _assert_held(store, upstream)
    assert store.events(EventType.followup_promoted) == []


@pytest.mark.asyncio
async def test_a_guard_that_answers_none_changes_nothing(store: _Store, guard: _StubGuard) -> None:
    held = await store.held('The retry loop swallows the timeout')

    assert await store.promote(held) == (held,)

    assert store.holder not in await store.dependencies(held)
    assert (await store.record(held))['promoted']['by'] == 'citation'
    assert store.emitted == [
        (EventType.followup_promoted, {'task_id': str(held), 'by': 'citation', 'ref': 'fk-1'}),
    ]


@pytest.mark.asyncio
async def test_a_guard_registered_twice_is_consulted_once(store: _Store, guard: _StubGuard) -> None:
    register_promotion_guard(guard)
    held = await store.held('The retry loop swallows the timeout')

    await store.promote(held)

    assert len(guard.calls) == 1


@dataclass
class _FaultyGuard:
    faulty: set[int] = field(default_factory=set)

    def __call__(self, task: Mapping[str, object], by: str, ref: str | None) -> PromotionDeferral | None:
        if int(str(task['id'])) in self.faulty:
            raise RuntimeError('rider index unavailable')
        return None


@pytest.fixture
def faulty_guard() -> Iterator[_FaultyGuard]:
    stub = _FaultyGuard()
    register_promotion_guard(stub)
    try:
        yield stub
    finally:
        unregister_promotion_guard(stub)


@pytest.mark.asyncio
async def test_a_guard_that_raises_defers_its_task_and_the_release_goes_on(
    store: _Store, faulty_guard: _FaultyGuard,
) -> None:
    faulted = await store.held('The retry loop swallows the timeout')
    released = await store.held('The cache key ignores the tenant')
    faulty_guard.faulty.add(faulted)

    promoted = await release_held(store.stack.backend, store.stack.root, store.holder, emit=store.emit)

    assert promoted == (released,)
    await _assert_held(store, faulted)
    [deferred] = store.events(EventType.followup_promotion_deferred)
    assert {key: deferred[key] for key in ('task_id', 'by', 'ref', 'guard')} == {
        'task_id': str(faulted), 'by': 'human', 'ref': 'release_holder', 'guard': '_FaultyGuard',
    }
    assert 'rider index unavailable' in deferred['reason']


@pytest.mark.asyncio
async def test_releasing_again_promotes_only_what_the_first_release_left_held(
    store: _Store, guard: _StubGuard,
) -> None:
    left_held = await store.held('The retry loop swallows the timeout')
    released = await store.held('The cache key ignores the tenant')
    guard.deferred.add(left_held)
    assert await release_held(store.stack.backend, store.stack.root, store.holder, emit=store.emit) == (released,)
    guard.deferred.clear()
    store.emitted.clear()

    promoted = await release_held(store.stack.backend, store.stack.root, store.holder, emit=store.emit)

    assert promoted == (left_held,)
    assert (await store.record(left_held))['promoted']['ref'] == 'release_holder'
    assert store.events(EventType.followup_promoted) == [
        {'task_id': str(left_held), 'by': 'human', 'ref': 'release_holder'},
    ]


@pytest.mark.asyncio
async def test_a_citation_leaves_a_held_finding_set_member_held(store: _Store) -> None:
    held = await store.held('Switch judge.py to the public name', metadata=_SET_MEMBER)

    promoted = await store.promote(held, by='citation', ref='fk-1')

    assert promoted == ()
    await _assert_held(store, held)
    [deferred] = store.events(EventType.followup_promotion_deferred)
    assert {key: deferred[key] for key in ('task_id', 'by', 'ref', 'guard')} == {
        'task_id': str(held), 'by': 'citation', 'ref': 'fk-1', 'guard': 'finding_set_hold',
    }
    assert 'Leo 2026-11-12 on fk-1' in deferred['reason']
    assert store.events(EventType.followup_promoted) == []


@pytest.mark.asyncio
@pytest.mark.parametrize('by', ['dependent', 'combine', 'pack', 'rider', 'human'])
async def test_every_other_cause_still_promotes_a_held_finding_set_member(store: _Store, by: str) -> None:
    held = await store.held('Switch judge.py to the public name', metadata=_SET_MEMBER)

    promoted = await store.promote(held, by=by, ref='r')

    assert promoted == (held,)
    assert store.holder not in await store.dependencies(held)
    assert (await store.record(held))['promoted']['by'] == by
    assert store.events(EventType.followup_promotion_deferred) == []
