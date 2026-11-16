"""Boundary row 16 and the curator's follow-up signals, through the real submit stack (task 8415).

plans/follow-up-filing-shape-prd.md §4.3 decision 13: a follow-up holder is never in a curator pool,
from any stream. §4.2: the curator reads each candidate's and pool entry's declared-scope stamp and
item count from the store.
"""

from __future__ import annotations

from typing import Any

import pytest
from _submit_stack import Stack
from _submit_stack import curator_creates as curator_creates
from _submit_stack import passthrough_main_checkout as passthrough_main_checkout
from _submit_stack import stack as stack
from shared.declared_scope import DECLARED_SCOPE_METADATA_KEY, DeclaredScope, derive_declared_scope
from shared.followup_items import FOLLOWUP_ITEMS_METADATA_KEY, observed_item

from fused_memory.middleware.curator_types import PoolEntry, PreparedCandidate
from fused_memory.middleware.followup_hold import FOLLOWUP_HOLDER_METADATA_KEY

_FILE = 'src/app/a.py'
_HOLDER_MARKER = {'project_id': 'p', 'minted_at': '2026-11-08T00:00:00+00:00'}


def _followup(spawned_from: str, **extra: Any) -> dict[str, Any]:
    return {
        'source': 'agent-followup',
        'spawn_context': 'agent_followup',
        'spawned_from': spawned_from,
        'files': [_FILE],
        **extra,
    }


def _pool(handed: list[PreparedCandidate]) -> list[PoolEntry]:
    assert len(handed) == 1, handed
    return handed[0].pool


async def _row(stack: Stack, task_id: str) -> dict[str, Any]:
    result = await stack.call('get_task', id=task_id)
    assert 'error' not in result, result
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize('parent', ['anchor', 'holder'])
async def test_a_holder_is_in_no_pool_stream(
    stack: Stack, curator_creates: list[PreparedCandidate], parent: str,
) -> None:
    anchor = await stack.seed('The held anchor', {'files': [_FILE]})
    holder = await stack.seed(
        'Follow-up holder', {FOLLOWUP_HOLDER_METADATA_KEY: _HOLDER_MARKER, 'files': [_FILE], 'spawned_from': anchor},
    )
    control = await stack.seed('Control twin', {'files': [_FILE], 'spawned_from': anchor})
    for neighbour in (holder, control):
        await stack.backend.add_dependency(anchor, neighbour, stack.root)
    holder_before = await _row(stack, holder)

    await stack.file_normal(
        title='A follow-up in src/app', description='One observation.',
        metadata=_followup(anchor if parent == 'anchor' else holder),
    )

    pooled = {entry.task_id: entry.source for entry in _pool(curator_creates)}
    assert control in pooled, pooled
    assert holder not in pooled, pooled
    assert await _row(stack, holder) == holder_before


@pytest.mark.asyncio
async def test_a_candidate_carries_the_stamp_the_store_shows(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    task_id = await stack.file_normal(
        title='A follow-up in src/app', description='One observation.',
        metadata=_followup('8000'),
    )

    stored = DeclaredScope.model_validate((await stack.metadata(task_id))[DECLARED_SCOPE_METADATA_KEY])
    assert curator_creates[0].candidate.declared_scope == stored


@pytest.mark.asyncio
async def test_a_pool_entry_carries_its_item_count_and_stamp(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    stamp = derive_declared_scope([_FILE], 'Two observations.', '', basis='submit')
    items = [
        observed_item(summary=summary, files=[_FILE], origin_task='8000').model_dump(mode='json')
        for summary in ('The retry loop swallows the timeout', 'The cache key ignores the tenant')
    ]
    pooled_id = await stack.seed('Follow-ups in src/app', {
        'files': [_FILE],
        FOLLOWUP_ITEMS_METADATA_KEY: items,
        DECLARED_SCOPE_METADATA_KEY: stamp.model_dump(),
    })

    await stack.file_normal(
        title='Another follow-up in src/app', description='A third observation.',
        metadata=_followup('9000'),
    )

    entry = next(entry for entry in _pool(curator_creates) if entry.task_id == pooled_id)
    assert entry.item_count == 2
    assert entry.declared_scope == stamp
