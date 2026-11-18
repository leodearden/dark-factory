"""Tests for the curator doubles in _curator_helpers.py.

Each double is driven through TaskInterceptor's public constructor seam,
``TaskInterceptor(..., curator=<double>)``, with tickets submitted through
``submit_task`` and answered through ``resolve_ticket``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from _curator_helpers import (
    curator_always,
    curator_double,
    curator_judging,
    curator_returning,
)

from fused_memory.middleware.curator_types import CandidateTask, CuratorDecision
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer

PROJECT = '/project'


@pytest.fixture
def taskmaster():
    tm = AsyncMock()
    tm.get_tasks = AsyncMock(return_value={'tasks': []})
    tm.add_task = AsyncMock(return_value={'id': '42', 'title': 'x'})
    tm.ensure_connected = AsyncMock()
    return tm


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    buf = EventBuffer(db_path=tmp_path / 'helpers_eb.db', buffer_size_threshold=100)
    await buf.initialize()
    yield buf
    await buf.close()


@pytest_asyncio.fixture
async def ticket_store(tmp_path):
    store = TicketStore(tmp_path / 'helpers_tickets.db')
    await store.initialize()
    yield store
    await store.close()


async def _submit_and_resolve(interceptor: TaskInterceptor, title: str) -> dict:
    response = await interceptor.submit_task(PROJECT, title=title)
    assert 'ticket' in response, f'submit_task returned no ticket: {response!r}'
    return await interceptor.resolve_ticket(
        response['ticket'], PROJECT, timeout_seconds=10.0,
    )


@pytest.mark.asyncio
async def test_curator_always_answers_every_candidate_with_its_decision(
    taskmaster, event_buffer, ticket_store,
):
    curator = curator_always(CuratorDecision(action='create', justification='novel'))
    interceptor = TaskInterceptor(
        taskmaster, None, event_buffer, ticket_store=ticket_store, curator=curator,
    )

    first = await _submit_and_resolve(interceptor, 'First candidate')
    second = await _submit_and_resolve(interceptor, 'Second candidate')
    await interceptor.close()

    assert (first['status'], second['status']) == ('created', 'created')
    assert curator.curate.await_count == 2
    curator.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_curator_returning_answers_its_decisions_in_order(
    taskmaster, event_buffer, ticket_store,
):
    curator = curator_returning(
        CuratorDecision(action='create'),
        CuratorDecision(action='drop', target_id='5', justification='dup'),
    )
    interceptor = TaskInterceptor(
        taskmaster, None, event_buffer, ticket_store=ticket_store, curator=curator,
    )

    first = await _submit_and_resolve(interceptor, 'First candidate')
    second = await _submit_and_resolve(interceptor, 'Second candidate')
    await interceptor.close()

    assert (first['status'], first['task_id']) == ('created', '42')
    assert (second['status'], second['task_id']) == ('combined', '5')
    curator.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_curator_judging_hands_each_candidate_to_its_verdict(
    taskmaster, event_buffer, ticket_store,
):
    judged: list[str] = []

    async def verdict(candidate: CandidateTask) -> CuratorDecision:
        judged.append(candidate.title)
        return CuratorDecision(action='create', justification='novel')

    curator = curator_judging(verdict)
    interceptor = TaskInterceptor(
        taskmaster, None, event_buffer, ticket_store=ticket_store, curator=curator,
    )

    first = await _submit_and_resolve(interceptor, 'First candidate')
    second = await _submit_and_resolve(interceptor, 'Second candidate')
    await interceptor.close()

    assert (first['status'], second['status']) == ('created', 'created')
    assert judged == ['First candidate', 'Second candidate']
    assert curator.curate.await_count == 2


@pytest.mark.asyncio
async def test_curator_double_serves_the_interceptor_without_a_verdict(
    taskmaster, event_buffer,
):
    taskmaster.remove_tasks = AsyncMock(return_value={
        'successful': 1, 'failed': 0, 'removed_ids': ['42'], 'message': 'ok',
    })
    curator = curator_double()
    interceptor = TaskInterceptor(taskmaster, None, event_buffer, curator=curator)

    await interceptor.remove_tasks(['42'], PROJECT)
    await interceptor.close()

    curator.evict_task.assert_awaited_once_with('project', '42')
    curator.close.assert_awaited_once()
