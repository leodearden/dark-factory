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
