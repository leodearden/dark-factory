"""The follow-up holder through the public tools: hold at filing and the one-holder mint.

plans/follow-up-filing-shape-prd.md §4.3 and boundary rows 1 (hold half) and 7. Every scenario
files through ``submit_task`` / ``resolve_ticket`` on the real submit stack and reads back through
``get_task`` and ``find_tasks_by_metadata``.
"""

from __future__ import annotations

from typing import Any

import pytest
from _submit_stack import Stack
from _submit_stack import curator_creates as curator_creates
from _submit_stack import passthrough_main_checkout as passthrough_main_checkout
from _submit_stack import stack as stack
from mcp.server.fastmcp.exceptions import ToolError
from shared.followup_hold_records import FOLLOWUP_HOLD_METADATA_KEY, FOLLOWUP_HOLDER_METADATA_KEY

from fused_memory.middleware.curator_types import PreparedCandidate
from fused_memory.models.reconciliation import EventType
from fused_memory.models.scope import resolve_project_id

_FILE = 'src/app/a.py'
_MARKER = {'project_id': 'proj', 'minted_at': '2026-11-08T00:00:00+00:00'}


def _followup(**extra: Any) -> dict[str, Any]:
    return {
        'source': 'agent-followup',
        'spawn_context': 'agent_followup',
        'spawned_from': '101',
        'files': [_FILE],
        **extra,
    }


async def _file_followup(stack: Stack, title: str, *, priority: str = 'low', **metadata: Any) -> str:
    return await stack.file_normal(
        title=title, description=f'{title}: one observation.', priority=priority,
        metadata=_followup(**metadata),
    )


async def _holder_ids(stack: Stack) -> list[str]:
    found = await stack.call('find_tasks_by_metadata', key=FOLLOWUP_HOLDER_METADATA_KEY)
    assert 'error' not in found, found
    return [str(row['task_id']) for row in found['results']]


async def _row(stack: Stack, task_id: str) -> dict[str, Any]:
    result = await stack.call('get_task', id=task_id)
    assert 'error' not in result, result
    return result


def _dependencies(row: dict[str, Any]) -> list[str]:
    return [str(dependency) for dependency in row.get('dependencies') or []]


async def _assert_unheld(stack: Stack, task_id: str) -> None:
    row = await _row(stack, task_id)
    assert _dependencies(row) == [], row
    assert FOLLOWUP_HOLD_METADATA_KEY not in row['metadata'], row['metadata']


@pytest.mark.asyncio
async def test_the_first_low_followup_mints_the_holder_and_is_born_held(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    held = await _file_followup(stack, 'The retry loop swallows the timeout')

    holders = await _holder_ids(stack)
    assert len(holders) == 1, holders
    [holder] = holders
    holder_row = await _row(stack, holder)
    holder_metadata = holder_row['metadata']
    assert holder_metadata['task_kind'] == 'deterministic'
    assert holder_metadata['always_escalates'] is True
    assert 'before_done' not in holder_metadata
    assert holder_metadata['execution_class'] == 'decision'
    assert holder_metadata['deferred_watch']
    assert not holder_metadata.get('trigger_met')
    assert holder_metadata[FOLLOWUP_HOLDER_METADATA_KEY]['project_id'] == resolve_project_id(stack.root)
    assert holder_row['priority'] == 'low'
    assert _dependencies(holder_row) == []

    held_row = await _row(stack, held)
    assert _dependencies(held_row) == [holder]
    assert str(held_row['metadata'][FOLLOWUP_HOLD_METADATA_KEY]['holder']) == holder
    assert await stack.events(EventType.followup_held) == [{'task_id': held, 'holder_id': holder}]


@pytest.mark.asyncio
async def test_later_low_followups_and_review_suggestions_reuse_the_one_holder(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    first = await _file_followup(stack, 'The retry loop swallows the timeout')
    second = await _file_followup(stack, 'The cache key ignores the tenant')
    suggestion = await stack.file_normal(
        title='Reviewer: name the retry budget', description='A reviewer suggestion.',
        priority='low', metadata={'spawn_context': 'review_suggestions', 'files': [_FILE]},
    )

    [holder] = await _holder_ids(stack)
    for task_id in (first, second, suggestion):
        assert _dependencies(await _row(stack, task_id)) == [holder], task_id


@pytest.mark.asyncio
async def test_a_medium_a_planned_and_a_cited_followup_are_not_held(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    await _file_followup(stack, 'The retry loop swallows the timeout')
    medium = await _file_followup(stack, 'The cache key ignores the tenant', priority='medium')
    planned = await stack.file_planned(
        title='PRD leaf: name the retry budget', description='A planning-mode follow-up.',
        priority='low', metadata=_followup(),
    )
    committed = await stack.call('set_task_status', id=planned, status='pending')
    assert 'error' not in committed, committed
    cited = await _file_followup(stack, 'The tenant leaks into the log', x_finding_key=['fk-1'])

    assert len(await _holder_ids(stack)) == 1
    for task_id in (medium, planned, cited):
        await _assert_unheld(stack, task_id)


@pytest.mark.asyncio
async def test_a_medium_followup_mints_no_holder(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    medium = await _file_followup(stack, 'The cache key ignores the tenant', priority='medium')

    assert await _holder_ids(stack) == []
    await _assert_unheld(stack, medium)


@pytest.mark.asyncio
async def test_a_cancelled_holder_is_not_live_so_the_next_followup_mints_anew(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    released = await stack.seed('Follow-up holder (proj)', {FOLLOWUP_HOLDER_METADATA_KEY: _MARKER}, status='cancelled')

    held = await _file_followup(stack, 'The retry loop swallows the timeout')

    holders = await _holder_ids(stack)
    assert released in holders
    [minted] = [holder for holder in holders if holder != released]
    assert _dependencies(await _row(stack, held)) == [minted]


@pytest.mark.asyncio
async def test_two_live_holders_fail_the_filing_naming_both(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    first = await stack.seed('Follow-up holder (proj)', {FOLLOWUP_HOLDER_METADATA_KEY: _MARKER})
    second = await stack.seed('Follow-up holder (proj) again', {FOLLOWUP_HOLDER_METADATA_KEY: _MARKER})
    count_before = await stack.task_count()

    _ticket, resolved = await stack.submit(
        title='The retry loop swallows the timeout', description='One observation.',
        priority='low', metadata=_followup(),
    )

    assert resolved.get('status') == 'failed', resolved
    assert first in resolved['reason'] and second in resolved['reason'], resolved
    assert await stack.task_count() == count_before


async def _arrange_held(stack: Stack) -> tuple[str, list[str]]:
    """The holder and its two active held follow-ups; a third held follow-up is cancelled."""
    held = [
        await _file_followup(stack, 'The retry loop swallows the timeout'),
        await _file_followup(stack, 'The cache key ignores the tenant'),
    ]
    dropped = await _file_followup(stack, 'The tenant leaks into the log')
    dropped_answer = await stack.call('set_task_status', id=dropped, status='cancelled')
    assert 'error' not in dropped_answer, dropped_answer
    [holder] = await _holder_ids(stack)
    return holder, held


def _assert_refused(answer: dict[str, Any], holder: str, *, protection: str, attempted_status: str | None) -> None:
    assert answer.get('error') == 'followup_holder_protected', answer
    assert answer['error_type'] == 'FollowupHolderProtectedError'
    assert answer['holder_id'] == holder
    assert answer['protection'] == protection
    assert answer['attempted_status'] == attempted_status
    assert answer['held_count'] == 2


@pytest.mark.asyncio
async def test_cancelling_the_holder_is_refused_with_its_held_count(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    holder, _held = await _arrange_held(stack)

    answer = await stack.call('set_task_status', id=holder, status='cancelled')

    _assert_refused(answer, holder, protection='protected_status', attempted_status='cancelled')
    assert (await _row(stack, holder))['status'] == 'pending'
    assert await stack.events(EventType.followup_holder_protected) == [{
        'holder_id': holder, 'attempted_status': 'cancelled', 'protection': 'protected_status',
        'held_count': 2,
    }]


@pytest.mark.asyncio
@pytest.mark.parametrize(('status', 'extra'), [
    ('done', {}),
    ('deferred', {'deferral': {'kind': 'until_condition', 'condition': 'x'}}),
], ids=['done', 'deferred'])
async def test_settling_the_holder_any_other_way_is_refused(
    stack: Stack, curator_creates: list[PreparedCandidate], status: str, extra: dict[str, Any],
) -> None:
    holder, _held = await _arrange_held(stack)

    answer = await stack.call('set_task_status', id=holder, status=status, **extra)

    _assert_refused(answer, holder, protection='protected_status', attempted_status=status)
    assert (await _row(stack, holder))['status'] == 'pending'


@pytest.mark.asyncio
async def test_a_csv_cancel_refuses_the_holder_and_applies_the_rest(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    holder, _held = await _arrange_held(stack)
    other = await _file_followup(stack, 'Rename the retry budget', priority='medium')

    answer = await stack.call('set_task_status', id=f'{holder},{other}', status='cancelled')

    by_id = {entry['task_id']: entry['result'] for entry in answer['results']}
    _assert_refused(by_id[holder], holder, protection='protected_status', attempted_status='cancelled')
    assert 'error' not in by_id[other], by_id[other]
    assert (await _row(stack, holder))['status'] == 'pending'
    assert (await _row(stack, other))['status'] == 'cancelled'


@pytest.mark.asyncio
async def test_setting_trigger_met_on_the_holder_is_refused(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    holder, _held = await _arrange_held(stack)

    answer = await stack.call('update_task', id=holder, metadata={'trigger_met': True})

    _assert_refused(answer, holder, protection='trigger_met_set', attempted_status=None)
    assert 'trigger_met' not in (await _row(stack, holder))['metadata']


@pytest.mark.asyncio
async def test_stripping_the_holder_gate_is_refused(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    holder, _held = await _arrange_held(stack)
    stored = (await _row(stack, holder))['metadata']
    unmarked = {key: value for key, value in stored.items() if key != FOLLOWUP_HOLDER_METADATA_KEY}

    unwatched = await stack.call('update_task', id=holder, metadata={'deferred_watch': None})
    replaced = await stack.call('update_task', id=holder, metadata=unmarked, metadata_mode='replace')

    _assert_refused(unwatched, holder, protection='deferred_watch_stripped', attempted_status=None)
    _assert_refused(replaced, holder, protection='marker_stripped', attempted_status=None)
    assert (await _row(stack, holder))['metadata'] == stored


@pytest.mark.asyncio
async def test_updates_that_keep_the_gate_are_allowed(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    holder, held = await _arrange_held(stack)

    noted = await stack.call('update_task', id=holder, metadata={'x_note': 'kept'})
    triggered = await stack.call('update_task', id=held[0], metadata={'trigger_met': True})

    assert 'error' not in noted, noted
    assert 'error' not in triggered, triggered
    assert (await _row(stack, holder))['metadata']['x_note'] == 'kept'
    assert (await _row(stack, held[0]))['metadata']['trigger_met'] is True


@pytest.mark.asyncio
async def test_an_explicit_release_cancels_the_holder_and_settles_every_held_dependency(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    holder, held = await _arrange_held(stack)

    answer = await stack.call('set_task_status', id=holder, status='cancelled', release_holder=True)

    assert 'error' not in answer, answer
    assert (await _row(stack, holder))['status'] == 'cancelled'
    for task_id in held:
        for dependency in _dependencies(await _row(stack, task_id)):
            assert (await _row(stack, dependency))['status'] in {'done', 'cancelled'}, (task_id, dependency)

    successor = await _file_followup(stack, 'The retry budget is unnamed')

    [minted] = [found for found in await _holder_ids(stack) if found != holder]
    assert _dependencies(await _row(stack, successor)) == [minted]


@pytest.mark.asyncio
async def test_a_truthy_string_never_releases_the_holder(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    holder, _held = await _arrange_held(stack)

    try:
        answer: dict[str, Any] | None = await stack.call(
            'set_task_status', id=holder, status='cancelled', release_holder='true',
        )
    except ToolError:
        answer = None

    assert answer is None or answer.get('error'), answer
    assert (await _row(stack, holder))['status'] == 'pending'
