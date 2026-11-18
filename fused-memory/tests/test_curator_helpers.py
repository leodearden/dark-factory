"""Tests for the curator doubles in _curator_helpers.py.

Each double is driven through TaskInterceptor's public constructor seam,
``TaskInterceptor(..., curator=<double>)``, with tickets submitted through
``submit_task`` and answered through ``resolve_ticket``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from _curator_helpers import curator_always, curator_double, curator_returning

from fused_memory.middleware.curator_types import CuratorDecision
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer

PROJECT_ROOT = '/project'


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
    response = await interceptor.submit_task(PROJECT_ROOT, title=title)
    assert 'ticket' in response, f'submit_task returned no ticket: {response!r}'
    return await interceptor.resolve_ticket(
        response['ticket'], PROJECT_ROOT, timeout_seconds=10.0,
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
async def test_curator_double_is_inert_until_wired():
    curator = curator_double()

    assert await curator.record_task('42', None, 'project') is None
    assert await curator.reembed_task('42', None, 'project') is None
    assert await curator.evict_task('42', 'project') is None
    assert await curator.close() is None
    curator.note_created('42')
