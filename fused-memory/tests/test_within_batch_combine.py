"""Within-batch combine, through a real curator and the worker's batch dispatch (task 8416).

plans/follow-up-filing-shape-prd.md §3 decision 2, §4.4 boundary rows 4 and 5 and §9 open
question 5: a candidate may combine into a sibling the same worker batch creates; a combine into
a sibling that was not created leaves the candidate created on its own, and audited; and R4
resolves a candidate before any batch edge can point at it.

Driven through public seams: the curator's injected invoker answers the batch call, submissions
go through ``submit_task`` as one worker batch, and outcomes are read from ``resolve_ticket``,
the task store and the combine audit file.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _curator_helpers import (
    agent_result,
    candidate_titles,
    invoke_by_role,
    submit_as_one_batch,
    tripwire,
)
from _real_batch_stack import RealBatchStack, real_batch_stack
from shared.declared_scope import DECLARED_SCOPE_METADATA_KEY
from shared.followup_items import FOLLOWUP_ITEMS_METADATA_KEY, item_key, observed_item
from shared.task_metadata_wire import coerce_task_metadata

_PARENT = '8400'
_ITEMS = {
    'A': {'summary': 'The retry loop swallows the timeout', 'files': ['src/app/retry.py']},
    'B': {'summary': 'The retry loop never backs off', 'files': ['src/app/retry.py']},
    'C': {'summary': 'The lock is never released on error', 'files': ['src/app/lock.py']},
}
_TITLES = {
    'A': 'Retry loop swallows the timeout',
    'B': 'Retry loop never backs off',
    'C': 'Lock leaks on error',
}
_REWRITE = {
    'title': 'Harden the retry loop',
    'description': 'Both retry-loop follow-ups as one task.',
    'details': '- surface the timeout\n- back off between attempts',
    'files_to_modify': ['src/app/retry.py'],
    'priority': 'low',
}
_BATCH_ROLE = 'task_curator_batch'


def _filing(name: str) -> dict[str, Any]:
    """One low-priority follow-up of _PARENT carrying item *name*: in hold scope, so born held."""
    item = _ITEMS[name]
    return {
        'title': _TITLES[name],
        'description': f'Follow-up {name} of {_PARENT}.',
        'priority': 'low',
        'metadata': {
            'source': 'agent-followup',
            'spawn_context': 'agent_followup',
            'spawned_from': _PARENT,
            'files': list(item['files']),
            FOLLOWUP_ITEMS_METADATA_KEY: [dict(item)],
        },
    }


def _key(name: str) -> str:
    item = _ITEMS[name]
    return item_key(origin_task=_PARENT, files=item['files'], summary=item['summary'])


def _create() -> dict[str, Any]:
    return {'action': 'create', 'justification': 'new work'}


def _drop_onto(index: int) -> dict[str, Any]:
    return {
        'action': 'drop', 'target_id': None, 'batch_target_index': index,
        'justification': f'a duplicate of batch item {index}',
    }


def _combine_onto(index: int, sibling: str) -> dict[str, Any]:
    return {
        'action': 'combine', 'target_id': None, 'batch_target_index': index,
        'target_fingerprint': _TITLES[sibling], 'rewritten_task': dict(_REWRITE),
        'justification': f'the same retry loop as batch item {index}',
    }


def _answering(*decisions: dict[str, Any]) -> AsyncMock:
    """An invoker answering the one batch call with *decisions*, in candidate order."""
    return invoke_by_role(
        single=tripwire('the batch must be judged in one batch call'),
        batch=agent_result({
            'decisions': [
                {'candidate_index': index, **decision} for index, decision in enumerate(decisions)
            ],
        }),
    )


def _batch_prompts(invoke: AsyncMock) -> list[str]:
    return [call.kwargs['prompt'] for call in invoke.await_args_list if call.kwargs['role'] == _BATCH_ROLE]


def _audit_rows() -> list[dict[str, Any]]:
    path = Path(os.environ['DARK_FACTORY_DATA_DIR']) / 'combine_audit.jsonl'
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


async def _drain_as_one_batch(
    stack: RealBatchStack, names: Sequence[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Submit the filings *names* as one worker batch; their tickets and answers."""
    tickets = await submit_as_one_batch(
        stack.interceptor, stack.curator, [_filing(name) for name in names], project_root=stack.root,
    )
    answers = [await stack.interceptor.resolve_ticket(ticket, stack.root, 30.0) for ticket in tickets]
    return tickets, answers


async def _metadata(stack: RealBatchStack, task_id: str) -> dict[str, Any]:
    metadata = coerce_task_metadata((await stack.backend.get_task(task_id, stack.root))['metadata'])
    assert metadata is not None
    return metadata


def _item_keys(metadata: dict[str, Any]) -> list[str]:
    return [item['key'] for item in metadata[FOLLOWUP_ITEMS_METADATA_KEY]]


@pytest.mark.asyncio
async def test_two_combinable_siblings_in_one_batch_resolve_as_one_task(tmp_path):
    invoke = _answering(_create(), _combine_onto(0, 'A'), _create())

    async with real_batch_stack(tmp_path, invoke) as stack:
        tickets, (a, b, c) = await _drain_as_one_batch(stack, ['A', 'B', 'C'])
        after = await _metadata(stack, a['task_id'])

    assert [answer['status'] for answer in (a, b, c)] == ['created', 'combined', 'created']
    task_x = a['task_id']
    assert b['task_id'] == task_x
    assert c['task_id'] != task_x
    assert _item_keys(after) == [_key('A'), _key('B')]
    assert after[DECLARED_SCOPE_METADATA_KEY]['basis'] == 'combine'
    [audit] = _audit_rows()
    assert {
        key: audit.get(key)
        for key in (
            'action', 'target_id', 'batch_internal', 'candidate_ticket_id', 'items_before', 'items_after',
        )
    } == {
        'action': 'combine',
        'target_id': task_x,
        'batch_internal': True,
        'candidate_ticket_id': tickets[1],
        'items_before': [_key('A')],
        'items_after': [_key('A'), _key('B')],
    }
    assert 'refused_reason' not in audit


@pytest.mark.asyncio
async def test_a_combine_onto_a_dropped_sibling_degrades_to_create_and_is_audited(tmp_path):
    invoke = _answering(_create(), _drop_onto(0), _combine_onto(1, 'B'))

    async with real_batch_stack(tmp_path, invoke) as stack:
        tickets, (c, b, a) = await _drain_as_one_batch(stack, ['C', 'B', 'A'])
        a_metadata = await _metadata(stack, a['task_id'])

    assert (c['status'], b['status'], a['status']) == ('created', 'combined', 'created')
    task_z = c['task_id']
    assert b['task_id'] == task_z
    assert a['task_id'] != task_z
    assert 'batch-target-not-create' in a['reason']
    assert _item_keys(a_metadata) == [_key('A')]
    drop_row, refusal = _audit_rows()
    assert {key: drop_row.get(key) for key in ('action', 'target_id', 'batch_internal')} == {
        'action': 'drop', 'target_id': task_z, 'batch_internal': True,
    }
    assert {
        key: refusal.get(key)
        for key in (
            'action', 'refused_reason', 'batch_internal', 'candidate_ticket_id', 'target_id', 'items_after',
            'sibling_ticket_id', 'sibling_ticket_status',
        )
    } == {
        'action': 'combine',
        'refused_reason': 'batch-target-not-create',
        'batch_internal': True,
        'candidate_ticket_id': tickets[2],
        'target_id': task_z,
        'items_after': [_key('A')],
        'sibling_ticket_id': tickets[1],
        'sibling_ticket_status': 'combined',
    }
    assert (refusal['old']['title'], refusal['old']['status']) == (_TITLES['B'], '')


@pytest.mark.asyncio
async def test_a_mistargeted_combine_leaves_the_sibling_untouched_and_is_created_on_its_own(tmp_path):
    """The kept LLM fingerprint is the one guard against a batch_target_index that names the wrong sibling."""
    mistargeted = {**_combine_onto(0, 'A'), 'target_fingerprint': 'Some other title'}
    invoke = _answering(_create(), mistargeted)

    async with real_batch_stack(tmp_path, invoke) as stack:
        _tickets, (a, b) = await _drain_as_one_batch(stack, ['A', 'B'])
        task_a = await stack.backend.get_task(a['task_id'], stack.root)
        a_metadata = await _metadata(stack, a['task_id'])
        b_metadata = await _metadata(stack, b['task_id'])

    assert (a['status'], b['status']) == ('created', 'created')
    assert b['task_id'] != a['task_id']
    assert task_a['title'] == _TITLES['A']
    assert _item_keys(a_metadata) == [_key('A')]
    assert _item_keys(b_metadata) == [_key('B')]
    assert [row for row in _audit_rows() if 'refused_reason' not in row] == []


@pytest.mark.asyncio
async def test_a_drop_onto_a_sibling_writes_an_audit_row(tmp_path):
    invoke = _answering(_create(), _drop_onto(0))

    async with real_batch_stack(tmp_path, invoke) as stack:
        tickets, (a, b) = await _drain_as_one_batch(stack, ['A', 'B'])
        a_metadata = await _metadata(stack, a['task_id'])

    assert (a['status'], b['status']) == ('created', 'combined')
    task_x = a['task_id']
    assert b['task_id'] == task_x
    assert _item_keys(a_metadata) == [_key('A')]
    [audit] = _audit_rows()
    assert {
        key: audit.get(key)
        for key in (
            'action', 'batch_internal', 'target_id', 'candidate_ticket_id',
            'candidate_spawned_from', 'items_before', 'items_after',
        )
    } == {
        'action': 'drop',
        'batch_internal': True,
        'target_id': task_x,
        'candidate_ticket_id': tickets[1],
        'candidate_spawned_from': _PARENT,
        'items_before': [_key('A')],
        'items_after': [_key('A')],
    }
    assert audit['new']['title'] == audit['old']['title'] == _TITLES['A']
    assert 'refused_reason' not in audit


@pytest.mark.asyncio
async def test_an_r4_hit_resolves_before_any_batch_edge_exists(tmp_path):
    invoke = _answering(_create(), _combine_onto(0, 'A'))

    async with real_batch_stack(tmp_path, invoke) as stack:
        seeded = await stack.backend.add_task(
            stack.root,
            title='Earlier follow-ups of the retry loop',
            description='Filed before this batch.',
            priority='medium',
            metadata=json.dumps({
                'source': 'agent-followup',
                'spawn_context': 'agent_followup',
                'spawned_from': _PARENT,
                'files': list(_ITEMS['B']['files']),
                FOLLOWUP_ITEMS_METADATA_KEY: [
                    observed_item(origin_task=_PARENT, **_ITEMS['B']).model_dump(mode='json'),
                ],
            }),
        )
        _tickets, (a, b, c) = await _drain_as_one_batch(stack, ['A', 'B', 'C'])

    seeded_id = str(seeded['id'])
    assert b == {'status': 'combined', 'task_id': seeded_id, 'reason': 'idempotency_hit'}
    [batch_prompt] = _batch_prompts(invoke)
    assert candidate_titles(batch_prompt) == [_TITLES['A'], _TITLES['C']]
    assert a['status'] == 'created'
    assert (c['status'], c['task_id']) == ('combined', a['task_id'])
