"""What promotes a held follow-up, through the public tools (plans/follow-up-filing-shape-prd.md §4.3).

Boundary rows 8-13 and 15: an edge from a pending unheld task, every landing on pending, a
citation, a combine, a human removal of the holder edge and the explicit release. Every scenario
files through ``submit_task`` / ``resolve_ticket`` on the real submit stack and reads back through
``get_task`` and the journaled events.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping
from typing import Any

import pytest
from _submit_stack import Stack
from _submit_stack import curator_creates as curator_creates
from _submit_stack import passthrough_main_checkout as passthrough_main_checkout
from _submit_stack import stack as stack
from shared.followup_hold_records import FOLLOWUP_HOLD_METADATA_KEY

from fused_memory.middleware.curator_types import PreparedCandidate
from fused_memory.middleware.followup_hold import (
    PromotionDeferral,
    register_promotion_guard,
    unregister_promotion_guard,
)
from fused_memory.models.reconciliation import EventType

_FILE = 'src/app/a.py'
_DEFERRAL = {'kind': 'until_condition', 'condition': 'x'}


def _followup(**extra: Any) -> dict[str, Any]:
    return {'source': 'agent-followup', 'spawn_context': 'agent_followup', 'spawned_from': '101', 'files': [_FILE], **extra}


async def _file_followup(stack: Stack, title: str, *, priority: str = 'low', **args: Any) -> str:
    return await stack.file_normal(
        title=title, description=f'{title}: one observation.', priority=priority, metadata=_followup(), **args,
    )


async def _file_held(stack: Stack, title: str) -> str:
    held = await _file_followup(stack, title)
    assert await _holder_edge(stack, held), held
    return held


async def _holder(stack: Stack) -> str:
    [holder] = await stack.holder_ids()
    return holder


async def _holder_edge(stack: Stack, task_id: str) -> bool:
    dependencies = await stack.dependencies(task_id)
    return any(holder in dependencies for holder in await stack.holder_ids())


async def _promotion(stack: Stack, task_id: str) -> dict[str, Any] | None:
    return (await stack.metadata(task_id))[FOLLOWUP_HOLD_METADATA_KEY].get('promoted')


async def _assert_promoted(stack: Stack, task_id: str, *, by: str, ref: str) -> None:
    assert not await _holder_edge(stack, task_id), task_id
    promotion = await _promotion(stack, task_id)
    assert promotion is not None, task_id
    assert (promotion['by'], promotion['ref']) == (by, ref), promotion


async def _assert_held(stack: Stack, task_id: str) -> None:
    assert await _holder_edge(stack, task_id), task_id
    assert await _promotion(stack, task_id) is None, task_id


async def _promoted_events(stack: Stack) -> list[dict[str, Any]]:
    return await stack.events(EventType.followup_promoted)


def _ok(answer: Any) -> None:
    assert isinstance(answer, dict) and 'error' not in answer, answer


@pytest.mark.asyncio
async def test_an_edge_from_a_pending_task_promotes_the_held_task(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    held = await _file_held(stack, 'The retry loop swallows the timeout')
    pending_since = (await stack.metadata(held)).get('pending_since')
    dependent = await _file_followup(stack, 'Ship the retry budget', priority='medium')

    _ok(await stack.call('add_dependency', id=dependent, depends_on=held))

    await _assert_promoted(stack, held, by='dependent', ref=dependent)
    assert (await stack.metadata(held)).get('pending_since') == pending_since
    assert await _promoted_events(stack) == [{'task_id': held, 'by': 'dependent', 'ref': dependent}]


@pytest.mark.asyncio
async def test_a_task_born_depending_on_a_held_task_promotes_it(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    held = await _file_held(stack, 'The retry loop swallows the timeout')

    dependent = await _file_followup(stack, 'Ship the retry budget', priority='medium', dependencies=held)

    await _assert_promoted(stack, held, by='dependent', ref=dependent)


@pytest.mark.asyncio
async def test_an_edge_between_two_held_tasks_promotes_neither(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    upstream = await _file_held(stack, 'The cache key ignores the tenant')
    dependent = await _file_held(stack, 'The tenant leaks into the log')

    _ok(await stack.call('add_dependency', id=dependent, depends_on=upstream))

    await _assert_held(stack, upstream)
    await _assert_held(stack, dependent)
    assert await _promoted_events(stack) == []


@pytest.mark.asyncio
@pytest.mark.parametrize('flip', ['commit_planning', 'set_task_status', 'deferral_sweep'])
async def test_a_dependent_landing_on_pending_promotes_the_held_task(
    stack: Stack, curator_creates: list[PreparedCandidate], flip: str,
) -> None:
    held = await _file_held(stack, 'The retry loop swallows the timeout')
    planned = await stack.file_planned(
        title='PRD leaf: name the retry budget', description='A planning-mode task.',
        priority='medium', metadata={'files': [_FILE]},
    )
    _ok(await stack.call('add_dependency', id=planned, depends_on=held))
    await _assert_held(stack, held)
    assert await _promoted_events(stack) == []

    if flip == 'commit_planning':
        _ok(await stack.call('commit_planning', task_ids=planned))
    elif flip == 'set_task_status':
        _ok(await stack.call('set_task_status', id=planned, status='pending'))
    else:
        _ok(await stack.call(
            'set_task_status', id=planned, status='pending', client_op_id=f'deferral-sweep-{uuid.uuid4()}',
        ))

    assert (await stack.row(planned))['status'] == 'pending'
    await _assert_promoted(stack, held, by='dependent', ref=planned)


@pytest.mark.asyncio
async def test_a_held_task_landing_back_on_pending_stays_held(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    held = await _file_held(stack, 'The retry loop swallows the timeout')

    _ok(await stack.call('set_task_status', id=held, status='deferred', deferral=_DEFERRAL))
    _ok(await stack.call('set_task_status', id=held, status='pending'))

    await _assert_held(stack, held)
    assert await _promoted_events(stack) == []


@pytest.mark.asyncio
async def test_a_human_removing_the_holder_edge_promotes_the_task(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    held = await _file_held(stack, 'The retry loop swallows the timeout')
    holder = await _holder(stack)

    _ok(await stack.call('remove_dependency', id=held, depends_on=holder))

    await _assert_promoted(stack, held, by='human', ref='remove_dependency')
    assert await _promoted_events(stack) == [{'task_id': held, 'by': 'human', 'ref': 'remove_dependency'}]


@pytest.mark.asyncio
async def test_removing_an_ordinary_edge_of_a_held_task_leaves_it_held(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    held = await _file_held(stack, 'The retry loop swallows the timeout')
    upstream = await _file_followup(stack, 'Ship the retry budget', priority='medium')
    _ok(await stack.call('add_dependency', id=held, depends_on=upstream))

    _ok(await stack.call('remove_dependency', id=held, depends_on=upstream))

    assert upstream not in await stack.dependencies(held)
    await _assert_held(stack, held)
    assert await _promoted_events(stack) == []


def _defer_everything(task: Mapping[str, object], by: str, ref: str | None) -> PromotionDeferral:
    return PromotionDeferral(guard='stub-guard', reason='riders still open')


@pytest.fixture
def deferring_guard() -> Iterator[None]:
    register_promotion_guard(_defer_everything)
    try:
        yield
    finally:
        unregister_promotion_guard(_defer_everything)


@pytest.mark.asyncio
async def test_a_guard_deferral_refuses_the_human_removal(
    stack: Stack, curator_creates: list[PreparedCandidate], deferring_guard: None,
) -> None:
    held = await _file_held(stack, 'The retry loop swallows the timeout')
    holder = await _holder(stack)

    answer = await stack.call('remove_dependency', id=held, depends_on=holder)

    assert answer.get('error') == 'followup_promotion_deferred', answer
    assert answer['task_id'] == held
    assert answer['holder_id'] == holder
    assert answer['guard'] == 'stub-guard'
    assert answer['reason'] == 'riders still open'
    await _assert_held(stack, held)
    assert [event['guard'] for event in await stack.events(EventType.followup_promotion_deferred)] == ['stub-guard']


@pytest.mark.asyncio
@pytest.mark.parametrize(('status', 'extra'), [
    ('cancelled', {}),
    ('deferred', {'deferral': _DEFERRAL}),
], ids=['cancelled', 'deferred'])
async def test_releasing_the_holder_promotes_every_active_held_task_by_human(
    stack: Stack, curator_creates: list[PreparedCandidate], status: str, extra: dict[str, Any],
) -> None:
    held = [
        await _file_held(stack, 'The retry loop swallows the timeout'),
        await _file_held(stack, 'The cache key ignores the tenant'),
        await _file_held(stack, 'The tenant leaks into the log'),
    ]
    _ok(await stack.call('add_dependency', id=held[2], depends_on=held[1]))
    dropped = await _file_held(stack, 'The retry budget is unnamed')
    _ok(await stack.call('set_task_status', id=dropped, status='cancelled'))
    holder = await _holder(stack)

    _ok(await stack.call('set_task_status', id=holder, status=status, release_holder=True, **extra))

    assert (await stack.row(holder))['status'] == status
    for task_id in held:
        await _assert_promoted(stack, task_id, by='human', ref='release_holder')
    events = await _promoted_events(stack)
    assert sorted(event['task_id'] for event in events) == sorted(held)
    assert {event['by'] for event in events} == {'human'}
    assert await _promotion(stack, dropped) is None
