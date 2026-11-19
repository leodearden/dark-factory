"""The startup backfill that holds the pending follow-up stock behind the holder.

plans/follow-up-filing-shape-prd.md §3 decision 12 and boundary row 17. Rows are seeded straight into
the store, as the stock filed before the hold existed is, and read back through ``get_task``.
"""

from __future__ import annotations

from typing import Any

import pytest
from _submit_stack import Stack
from _submit_stack import passthrough_main_checkout as passthrough_main_checkout
from _submit_stack import stack as stack
from shared.followup_hold_records import FOLLOWUP_HOLD_METADATA_KEY, FOLLOWUP_HOLDER_METADATA_KEY

from fused_memory.models.reconciliation import EventType
from fused_memory.models.scope import resolve_project_id

_FOLLOWUP = {'source': 'agent-followup', 'spawn_context': 'agent_followup', 'spawned_from': '101'}
_REVIEW = {'spawn_context': 'review_suggestions'}
_EARLIER_HOLD = {'holder': 1, 'held_at': '2026-11-01T00:00:00+00:00'}


async def _row(stack: Stack, task_id: str) -> dict[str, Any]:
    result = await stack.call('get_task', id=task_id)
    assert 'error' not in result, result
    return result


def _dependencies(row: dict[str, Any]) -> list[str]:
    return [str(dependency) for dependency in row.get('dependencies') or []]


async def _holder_ids(stack: Stack) -> list[str]:
    found = await stack.call('find_tasks_by_metadata', key=FOLLOWUP_HOLDER_METADATA_KEY)
    assert 'error' not in found, found
    return [str(row['task_id']) for row in found['results']]


async def _seed_stock(stack: Stack) -> tuple[list[str], list[str]]:
    """Three pending follow-ups the hold rule matches, and the rows it must leave alone."""
    blocker = await stack.seed('An ordinary pending task', {})
    matching = [
        await stack.seed('Low agent follow-up', dict(_FOLLOWUP), priority='low'),
        await stack.seed('Low reviewer suggestion', dict(_REVIEW), priority='low'),
        await stack.seed('Polish agent follow-up', dict(_FOLLOWUP), priority='polish'),
    ]
    await stack.backend.add_dependency(matching[2], blocker, stack.root)
    untouched = [
        blocker,
        await stack.seed('Medium agent follow-up', dict(_FOLLOWUP), priority='medium'),
        await stack.seed('Done low follow-up', dict(_FOLLOWUP), priority='low', status='done'),
        await stack.seed('Planned low follow-up', {**_FOLLOWUP, 'human_decomposed': True}, priority='low'),
        await stack.seed('PRD low follow-up', {**_FOLLOWUP, 'prd_path': 'plans/x-prd.md'}, priority='low'),
        await stack.seed(
            'Released low follow-up', {**_FOLLOWUP, FOLLOWUP_HOLD_METADATA_KEY: _EARLIER_HOLD}, priority='low',
        ),
    ]
    return matching, untouched


@pytest.mark.asyncio
async def test_the_backfill_holds_every_matching_pending_followup_in_one_event(stack: Stack) -> None:
    matching, untouched = await _seed_stock(stack)
    prior = {task_id: _dependencies(await _row(stack, task_id)) for task_id in matching}
    untouched_before = {task_id: await _row(stack, task_id) for task_id in untouched}

    count = await stack.interceptor.backfill_followup_holds(stack.root)

    assert count == 3
    [holder] = await _holder_ids(stack)
    for task_id in matching:
        row = await _row(stack, task_id)
        assert _dependencies(row) == [*prior[task_id], holder], task_id
        assert str(row['metadata'][FOLLOWUP_HOLD_METADATA_KEY]['holder']) == holder
    for task_id, before in untouched_before.items():
        assert await _row(stack, task_id) == before, task_id
    assert await stack.events(EventType.followup_hold_backfilled) == [
        {'project_id': resolve_project_id(stack.root), 'count': 3},
    ]
    assert await stack.events(EventType.task_modified) == []


@pytest.mark.asyncio
async def test_a_second_backfill_adds_nothing_and_still_reports(stack: Stack) -> None:
    matching, _untouched = await _seed_stock(stack)
    await stack.interceptor.backfill_followup_holds(stack.root)
    held = {task_id: _dependencies(await _row(stack, task_id)) for task_id in matching}

    count = await stack.interceptor.backfill_followup_holds(stack.root)

    assert count == 0
    for task_id in matching:
        assert _dependencies(await _row(stack, task_id)) == held[task_id], task_id
    assert [event['count'] for event in await stack.events(EventType.followup_hold_backfilled)] == [3, 0]


@pytest.mark.asyncio
async def test_with_nothing_to_hold_no_holder_is_minted(stack: Stack) -> None:
    await stack.seed('Medium agent follow-up', dict(_FOLLOWUP), priority='medium')

    count = await stack.interceptor.backfill_followup_holds(stack.root)

    assert count == 0
    assert await _holder_ids(stack) == []
    assert await stack.events(EventType.followup_hold_backfilled) == [
        {'project_id': resolve_project_id(stack.root), 'count': 0},
    ]
