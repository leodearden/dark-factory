"""What promotes a held follow-up, through the public tools (plans/follow-up-filing-shape-prd.md §4.3).

Boundary rows 8-13 and 15: an edge from a waiting unheld task, every landing on pending, a
citation, a combine, a human removal or replacement of the holder edge and the explicit release. Every scenario
files through ``submit_task`` / ``resolve_ticket`` on the real submit stack and reads back through
``get_task`` and the journaled events.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping
from typing import Any

import pytest
from _submit_stack import CuratorScript, Stack
from _submit_stack import curator_creates as curator_creates
from _submit_stack import curator_script as curator_script
from _submit_stack import passthrough_main_checkout as passthrough_main_checkout
from _submit_stack import stack as stack
from shared.followup_hold_records import FOLLOWUP_HOLD_METADATA_KEY

from fused_memory.middleware.curator_types import CuratorDecision, PreparedCandidate, RewrittenTask
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
@pytest.mark.parametrize('status', ['in-progress', 'blocked'])
async def test_an_edge_from_an_unheld_task_that_is_not_pending_promotes_the_held_task(
    stack: Stack, curator_creates: list[PreparedCandidate], status: str,
) -> None:
    held = await _file_held(stack, 'The retry loop swallows the timeout')
    dependent = await stack.seed('Ship the retry budget', {'files': [_FILE]}, status=status)

    _ok(await stack.call('add_dependency', id=dependent, depends_on=held))
    _ok(await stack.call('set_task_status', id=dependent, status='pending'))

    await _assert_promoted(stack, held, by='dependent', ref=dependent)
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
@pytest.mark.parametrize('status', ['done', 'cancelled'])
async def test_a_settled_dependent_reopened_to_pending_promotes_the_held_task(
    stack: Stack, curator_creates: list[PreparedCandidate], status: str,
) -> None:
    held = await _file_held(stack, 'The retry loop swallows the timeout')
    dependent = await stack.seed('Ship the retry budget', {'files': [_FILE]}, status=status)
    _ok(await stack.call('add_dependency', id=dependent, depends_on=held))
    await _assert_held(stack, held)
    assert await _promoted_events(stack) == []

    _ok(await stack.call(
        'set_task_status', id=dependent, status='pending', reopen_reason='reopened: the retry budget regressed',
    ))

    assert (await stack.row(dependent))['status'] == 'pending'
    await _assert_promoted(stack, held, by='dependent', ref=dependent)


@pytest.mark.asyncio
async def test_a_deferred_dependent_landing_on_blocked_promotes_the_held_task(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    held = await _file_held(stack, 'The retry loop swallows the timeout')
    planned = await stack.file_planned(
        title='PRD leaf: name the retry budget', description='A planning-mode task.',
        priority='medium', metadata={'files': [_FILE]},
    )
    _ok(await stack.call('add_dependency', id=planned, depends_on=held))
    await _assert_held(stack, held)

    _ok(await stack.call('set_task_status', id=planned, status='blocked'))

    assert (await stack.row(planned))['status'] == 'blocked'
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


@pytest.mark.asyncio
@pytest.mark.parametrize(('citation', 'ref'), [
    ({'x_finding_key': ['fk-1']}, 'fk-1'),
    ({'prd_path': 'plans/x-prd.md'}, 'plans/x-prd.md'),
], ids=['finding-key', 'prd-path'])
async def test_citing_a_held_task_promotes_it(
    stack: Stack, curator_creates: list[PreparedCandidate], citation: dict[str, Any], ref: str,
) -> None:
    held = await _file_held(stack, 'The retry loop swallows the timeout')

    _ok(await stack.call('update_task', id=held, metadata=citation, metadata_mode='additive'))

    await _assert_promoted(stack, held, by='citation', ref=ref)
    assert await _promoted_events(stack) == [{'task_id': held, 'by': 'citation', 'ref': ref}]


@pytest.mark.asyncio
async def test_citing_a_held_finding_set_member_leaves_it_held_until_a_human_releases_it(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    held = await stack.file_normal(
        title='Switch judge.py to the public name',
        description='Switch judge.py to the public name: one member of a finding set.',
        priority='low',
        metadata={
            'source': 'review-all', 'x_finding_key': ['fk-1'],
            'finding_set_hold': {'ruling': 'Leo 2026-11-12 on fk-1'}, 'files': [_FILE],
        },
    )
    assert await _holder_edge(stack, held), held

    _ok(await stack.call('update_task', id=held, metadata={'x_finding_key': ['fk-1', 'fk-2']}, metadata_mode='additive'))

    await _assert_held(stack, held)
    assert await _promoted_events(stack) == []
    [deferred] = await stack.events(EventType.followup_promotion_deferred)
    assert (deferred['guard'], deferred['by']) == ('finding_set_hold', 'citation')

    _ok(await stack.call('remove_dependency', id=held, depends_on=await _holder(stack)))

    await _assert_promoted(stack, held, by='human', ref='remove_dependency')


@pytest.mark.asyncio
async def test_citing_a_held_task_promotes_the_held_task_it_depends_on(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    upstream = await _file_held(stack, 'The cache key ignores the tenant')
    cited = await _file_held(stack, 'The tenant leaks into the log')
    _ok(await stack.call('add_dependency', id=cited, depends_on=upstream))

    _ok(await stack.call('update_task', id=cited, metadata={'x_finding_key': ['fk-1']}, metadata_mode='additive'))

    await _assert_promoted(stack, cited, by='citation', ref='fk-1')
    await _assert_promoted(stack, upstream, by='dependent', ref=cited)


@pytest.mark.asyncio
async def test_citing_a_task_that_is_not_held_promotes_nothing(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    await _file_held(stack, 'The retry loop swallows the timeout')
    unheld = await _file_followup(stack, 'Ship the retry budget', priority='medium')

    _ok(await stack.call('update_task', id=unheld, metadata={'x_finding_key': ['fk-1']}, metadata_mode='additive'))

    assert await _promoted_events(stack) == []


@pytest.mark.asyncio
async def test_a_dependency_replacement_onto_a_held_task_promotes_it(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    held = await _file_held(stack, 'The retry loop swallows the timeout')
    dependent = await _file_followup(stack, 'Ship the retry budget', priority='medium')

    _ok(await stack.call('update_task', id=dependent, dependencies=[held]))

    await _assert_promoted(stack, held, by='dependent', ref=dependent)


@pytest.mark.asyncio
async def test_a_human_replacement_dropping_the_holder_edge_promotes_the_task(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    held = await _file_held(stack, 'The retry loop swallows the timeout')

    _ok(await stack.call('update_task', id=held, dependencies=[], title='Renamed'))

    assert (await stack.row(held))['title'] == 'Renamed'
    await _assert_promoted(stack, held, by='human', ref='update_task')
    assert await _promoted_events(stack) == [{'task_id': held, 'by': 'human', 'ref': 'update_task'}]


@pytest.mark.asyncio
async def test_a_human_replacement_the_backend_refuses_leaves_the_task_held(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    held = await _file_held(stack, 'The retry loop swallows the timeout')
    holder = await _holder(stack)

    answer = await stack.call('update_task', id=held, dependencies=['99999'], title='Renamed')

    assert answer.get('error'), answer
    assert (await stack.row(held))['title'] == 'The retry loop swallows the timeout'
    assert await stack.dependencies(held) == [holder]
    await _assert_held(stack, held)
    assert await _promoted_events(stack) == []


@pytest.mark.asyncio
async def test_a_guard_deferral_refuses_the_human_replacement_with_nothing_written(
    stack: Stack, curator_creates: list[PreparedCandidate], deferring_guard: None,
) -> None:
    held = await _file_held(stack, 'The retry loop swallows the timeout')
    holder = await _holder(stack)

    answer = await stack.call('update_task', id=held, dependencies=[], title='Renamed')

    assert answer.get('error') == 'followup_promotion_deferred', answer
    assert answer['guard'] == 'stub-guard'
    assert (await stack.row(held))['title'] == 'The retry loop swallows the timeout'
    assert await stack.dependencies(held) == [holder]
    await _assert_held(stack, held)
    assert [event['guard'] for event in await stack.events(EventType.followup_promotion_deferred)] == ['stub-guard']


@pytest.mark.asyncio
@pytest.mark.parametrize('agent_id', ['orchestrator', 'steward-1'])
async def test_a_machine_replacement_dropping_the_holder_edge_is_refused(
    stack: Stack, curator_creates: list[PreparedCandidate], agent_id: str,
) -> None:
    held = await _file_held(stack, 'The retry loop swallows the timeout')
    holder = await _holder(stack)
    before = await stack.row(held)

    answer = await stack.call('update_task', id=held, dependencies=[], title='Renamed', agent_id=agent_id)

    assert answer.get('error') == 'followup_holder_edge_strip_refused', answer
    assert answer['task_id'] == held
    assert answer['holder_id'] == holder
    after = await stack.row(held)
    assert after['title'] == before['title']
    assert await stack.dependencies(held) == [holder]
    assert await _promoted_events(stack) == []


@pytest.mark.asyncio
async def test_a_machine_replacement_keeping_the_holder_edge_leaves_the_task_held(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    held = await _file_held(stack, 'The retry loop swallows the timeout')
    holder = await _holder(stack)
    upstream = await _file_followup(stack, 'Ship the retry budget', priority='medium')

    _ok(await stack.call('update_task', id=held, dependencies=[holder, upstream], agent_id='orchestrator'))

    assert sorted(await stack.dependencies(held)) == sorted([holder, upstream])
    await _assert_held(stack, held)
    assert await _promoted_events(stack) == []


_REWRITE = RewrittenTask(
    title='Harden the retry path',
    description='Fold the retry follow-ups into one task.',
    details='- retry timeout\n- retry budget',
    files_to_modify=[_FILE],
    priority='low',
)


async def _combine_candidate_into(stack: Stack, script: CuratorScript, *, priority: str) -> tuple[str, str, dict[str, Any]]:
    """A held target and the ticket of a candidate the curator combines into it."""
    target_title = 'The retry loop swallows the timeout'
    target = await _file_held(stack, target_title)
    script.decisions.append(CuratorDecision(
        action='combine', target_id=target, target_fingerprint=target_title,
        rewritten_task=_REWRITE, justification='the same retry path',
    ))
    ticket, resolved = await stack.submit(
        title='The retry budget is unnamed', description='One more observation.', priority=priority,
        metadata=_followup(spawned_from='202'),
    )
    return target, ticket, resolved


@pytest.mark.asyncio
async def test_combining_an_unheld_candidate_into_a_held_task_promotes_it(
    stack: Stack, curator_script: CuratorScript,
) -> None:
    target, ticket, resolved = await _combine_candidate_into(stack, curator_script, priority='medium')

    assert {key: resolved.get(key) for key in ('status', 'task_id')} == {'status': 'combined', 'task_id': target}
    await _assert_promoted(stack, target, by='combine', ref=ticket)


@pytest.mark.asyncio
async def test_combining_a_held_candidate_into_a_held_task_leaves_it_held(
    stack: Stack, curator_script: CuratorScript,
) -> None:
    target, _ticket, resolved = await _combine_candidate_into(stack, curator_script, priority='low')

    assert {key: resolved.get(key) for key in ('status', 'task_id')} == {'status': 'combined', 'task_id': target}
    await _assert_held(stack, target)
    assert await _promoted_events(stack) == []


@pytest.mark.asyncio
async def test_a_replace_write_keeps_the_hold_and_a_replace_citation_promotes(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    held = await _file_held(stack, 'The retry loop swallows the timeout')
    stored = await stack.metadata(held)
    record = stored[FOLLOWUP_HOLD_METADATA_KEY]
    without_record = {key: value for key, value in stored.items() if key != FOLLOWUP_HOLD_METADATA_KEY}

    _ok(await stack.call('update_task', id=held, metadata=without_record, metadata_mode='replace'))

    assert (await stack.metadata(held))[FOLLOWUP_HOLD_METADATA_KEY] == record
    await _assert_held(stack, held)

    _ok(await stack.call(
        'update_task', id=held, metadata={**without_record, 'x_finding_key': ['fk-1']}, metadata_mode='replace',
    ))

    await _assert_promoted(stack, held, by='citation', ref='fk-1')


def _deferring(task_id: str) -> Any:
    def guard(task: Mapping[str, object], by: str, ref: str | None) -> PromotionDeferral | None:
        return PromotionDeferral(guard='stub-guard', reason='riders still open') if str(task['id']) == task_id else None

    return guard


@pytest.mark.asyncio
@pytest.mark.parametrize(('status', 'first', 'again'), [
    ('cancelled', {}, {}),
    ('deferred', {'deferral': _DEFERRAL}, {}),
    ('deferred', {'deferral': _DEFERRAL}, {'deferral': _DEFERRAL}),
], ids=['cancelled', 'deferred', 'deferred-restamped'])
async def test_releasing_a_released_holder_again_promotes_what_is_still_held(
    stack: Stack, curator_creates: list[PreparedCandidate], status: str, first: dict[str, Any], again: dict[str, Any],
) -> None:
    left_held = await _file_held(stack, 'The retry loop swallows the timeout')
    released = await _file_held(stack, 'The cache key ignores the tenant')
    holder = await _holder(stack)
    guard = _deferring(left_held)
    register_promotion_guard(guard)
    try:
        _ok(await stack.call('set_task_status', id=holder, status=status, release_holder=True, **first))
    finally:
        unregister_promotion_guard(guard)
    await _assert_promoted(stack, released, by='human', ref='release_holder')
    await _assert_held(stack, left_held)

    _ok(await stack.call('set_task_status', id=holder, status=status, **again))
    await _assert_held(stack, left_held)

    _ok(await stack.call('set_task_status', id=holder, status=status, release_holder=True, **again))

    assert (await stack.row(holder))['status'] == status
    await _assert_promoted(stack, left_held, by='human', ref='release_holder')
    assert sorted(event['task_id'] for event in await _promoted_events(stack)) == sorted([left_held, released])


@pytest.mark.asyncio
async def test_a_release_that_fails_after_the_status_write_warns_and_can_be_finished(
    stack: Stack, curator_creates: list[PreparedCandidate], monkeypatch: pytest.MonkeyPatch,
) -> None:
    held = await _file_held(stack, 'The retry loop swallows the timeout')
    holder = await _holder(stack)

    async def unreadable(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError('dependency table locked')

    with monkeypatch.context() as patch:
        patch.setattr(stack.backend, 'get_dependency_edges', unreadable)
        answer = await stack.call('set_task_status', id=holder, status='cancelled', release_holder=True)

    _ok(answer)
    assert answer['post_write_warnings'] == [{'stage': 'promote_on_transition', 'error': 'dependency table locked'}]
    assert (await stack.row(holder))['status'] == 'cancelled'
    assert [event['task_id'] for event in await stack.events(EventType.task_status_changed)][-1] == holder
    await _assert_held(stack, held)

    _ok(await stack.call('set_task_status', id=holder, status='cancelled', release_holder=True))

    await _assert_promoted(stack, held, by='human', ref='release_holder')
