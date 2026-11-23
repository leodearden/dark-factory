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
import pytest_asyncio
from _curator_helpers import (
    agent_result,
    candidate_titles,
    escalator_double,
    invoke_by_role,
    make_config,
    prepared,
    submit_as_one_batch,
    tripwire,
)
from shared.declared_scope import DECLARED_SCOPE_METADATA_KEY
from shared.followup_items import FOLLOWUP_ITEMS_METADATA_KEY, item_key, observed_item
from shared.task_metadata_wire import coerce_task_metadata

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.curator_types import CandidateTask, PreparedCandidate
from fused_memory.middleware.task_curator import TaskCurator
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer

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


@pytest_asyncio.fixture
async def store(tmp_path):
    ticket_store = TicketStore(tmp_path / 'tickets.db')
    await ticket_store.initialize()
    yield ticket_store
    await ticket_store.close()


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    buffer = EventBuffer(db_path=tmp_path / 'events.db', buffer_size_threshold=100)
    await buffer.initialize()
    yield buffer
    await buffer.close()


@pytest_asyncio.fixture
async def backend(tmp_path):
    task_backend = SqliteTaskBackend(TaskmasterConfig(project_root=str(tmp_path)))
    await task_backend.start()
    yield task_backend
    await task_backend.close()


@pytest.fixture
def root(tmp_path) -> str:
    return str(tmp_path)


async def _drain_as_one_batch(
    backend: SqliteTaskBackend,
    store: TicketStore,
    event_buffer: EventBuffer,
    root: str,
    names: Sequence[str],
    invoke: AsyncMock,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Submit the filings *names* as one worker batch judged by a real curator; their tickets and answers."""
    curator = TaskCurator(
        config=make_config(), taskmaster=backend, escalator=escalator_double(), invoke=invoke,
    )

    async def bundle_for(candidate: CandidateTask, *_args: Any) -> PreparedCandidate:
        return prepared(candidate.title, candidate=candidate)

    curator.prepare_candidate = AsyncMock(side_effect=bundle_for)
    curator.record_task = AsyncMock()
    curator.reembed_task = AsyncMock()
    interceptor = TaskInterceptor(backend, None, event_buffer, ticket_store=store, curator=curator)
    try:
        tickets = await submit_as_one_batch(
            interceptor, curator, [_filing(name) for name in names], project_root=root,
        )
        answers = [await interceptor.resolve_ticket(ticket, root, 30.0) for ticket in tickets]
    finally:
        await interceptor.close()
    return tickets, answers


async def _metadata(backend: SqliteTaskBackend, task_id: str, root: str) -> dict[str, Any]:
    metadata = coerce_task_metadata((await backend.get_task(task_id, root))['metadata'])
    assert metadata is not None
    return metadata


def _item_keys(metadata: dict[str, Any]) -> list[str]:
    return [item['key'] for item in metadata[FOLLOWUP_ITEMS_METADATA_KEY]]


@pytest.mark.asyncio
async def test_two_combinable_siblings_in_one_batch_resolve_as_one_task(
    backend, store, event_buffer, root,
):
    invoke = _answering(_create(), _combine_onto(0, 'A'), _create())

    tickets, (a, b, c) = await _drain_as_one_batch(
        backend, store, event_buffer, root, ['A', 'B', 'C'], invoke,
    )

    assert [answer['status'] for answer in (a, b, c)] == ['created', 'combined', 'created']
    task_x = a['task_id']
    assert b['task_id'] == task_x
    assert c['task_id'] != task_x
    after = await _metadata(backend, task_x, root)
    assert _item_keys(after) == [_key('A'), _key('B')]
    assert after[DECLARED_SCOPE_METADATA_KEY]['basis'] == 'combine'
    [audit] = _audit_rows()
    assert {
        key: audit.get(key)
        for key in ('target_id', 'batch_internal', 'candidate_ticket_id', 'items_before', 'items_after')
    } == {
        'target_id': task_x,
        'batch_internal': True,
        'candidate_ticket_id': tickets[1],
        'items_before': [_key('A')],
        'items_after': [_key('A'), _key('B')],
    }
    assert 'refused_reason' not in audit


@pytest.mark.asyncio
async def test_a_combine_onto_a_dropped_sibling_degrades_to_create_and_is_audited(
    backend, store, event_buffer, root,
):
    invoke = _answering(_create(), _drop_onto(0), _combine_onto(1, 'B'))

    tickets, (c, b, a) = await _drain_as_one_batch(
        backend, store, event_buffer, root, ['C', 'B', 'A'], invoke,
    )

    assert (c['status'], b['status'], a['status']) == ('created', 'combined', 'created')
    task_z = c['task_id']
    assert b['task_id'] == task_z
    assert a['task_id'] != task_z
    assert 'batch-target-not-create' in a['reason']
    assert _item_keys(await _metadata(backend, a['task_id'], root)) == [_key('A')]
    [refusal] = [row for row in _audit_rows() if row.get('refused_reason')]
    assert {
        key: refusal.get(key)
        for key in ('refused_reason', 'batch_internal', 'candidate_ticket_id', 'target_id', 'items_after')
    } == {
        'refused_reason': 'batch-target-not-create',
        'batch_internal': True,
        'candidate_ticket_id': tickets[2],
        'target_id': task_z,
        'items_after': [_key('A')],
    }


@pytest.mark.asyncio
async def test_an_r4_hit_resolves_before_any_batch_edge_exists(
    backend, store, event_buffer, root,
):
    seeded = await backend.add_task(
        root,
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
    seeded_id = str(seeded['id'])
    invoke = _answering(_create(), _combine_onto(0, 'A'))

    _tickets, (a, b, c) = await _drain_as_one_batch(
        backend, store, event_buffer, root, ['A', 'B', 'C'], invoke,
    )

    assert b == {'status': 'combined', 'task_id': seeded_id, 'reason': 'idempotency_hit'}
    [batch_prompt] = _batch_prompts(invoke)
    assert candidate_titles(batch_prompt) == [_TITLES['A'], _TITLES['C']]
    assert a['status'] == 'created'
    assert (c['status'], c['task_id']) == ('combined', a['task_id'])
