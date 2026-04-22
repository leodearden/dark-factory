"""Tests for the TaskInterceptor curator worker (ticket queue drain loop)."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from fused_memory.middleware.task_curator import CuratorDecision, RewrittenTask
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def taskmaster():
    tm = AsyncMock()
    tm.get_tasks = AsyncMock(return_value={'tasks': []})
    tm.add_task = AsyncMock(return_value={'id': '42', 'title': 'New Task'})
    tm.update_task = AsyncMock(return_value={'success': True})
    tm.set_task_status = AsyncMock(return_value={'success': True})
    tm.ensure_connected = AsyncMock()
    return tm


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    buf = EventBuffer(db_path=tmp_path / 'worker_eb.db', buffer_size_threshold=100)
    await buf.initialize()
    yield buf
    await buf.close()


@pytest_asyncio.fixture
async def ticket_store(tmp_path):
    store = TicketStore(tmp_path / 'worker_tickets.db')
    await store.initialize()
    yield store
    await store.close()


@pytest_asyncio.fixture
async def interceptor_with_store(taskmaster, event_buffer, ticket_store):
    ti = TaskInterceptor(taskmaster, None, event_buffer, ticket_store=ticket_store)
    yield ti
    # Cancel worker if running
    if ti._worker_task and not ti._worker_task.done():
        ti._worker_task.cancel()
        try:
            await ti._worker_task
        except (asyncio.CancelledError, Exception):
            pass


# ---------------------------------------------------------------------------
# step-21: worker processes a 'create' decision
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_processes_create_decision(
    interceptor_with_store, ticket_store, taskmaster,
):
    """When the curator returns action='create', the worker calls tm.add_task,
    marks the ticket as status='created' with task_id from tm.add_task, and
    populates result_json with the full add_task return dict.
    """
    # Arrange: curator mock that returns a 'create' decision
    mock_curator = MagicMock()
    mock_curator.curate = AsyncMock(
        return_value=CuratorDecision(action='create')
    )
    mock_curator.note_created = MagicMock()
    mock_curator.record_task = AsyncMock()

    with patch.object(
        type(interceptor_with_store), '_get_curator',
        new=AsyncMock(return_value=mock_curator),
    ):
        # Submit a ticket — this enqueues the ticket_id
        result = await interceptor_with_store.submit_task(
            project_root='/project',
            title='Test Task',
            description='A task for the worker',
        )
        assert result.get('ticket', '').startswith('tkt_'), (
            f'Expected ticket in result, got: {result}'
        )
        ticket_id = result['ticket']

        # Let the worker drain the queue
        await asyncio.sleep(0.1)

    # Assert: tm.add_task was called once
    taskmaster.add_task.assert_called_once()

    # Assert: ticket row is now terminal
    row = await ticket_store.get(ticket_id)
    assert row is not None, 'Ticket row should still exist'
    assert row['status'] == 'created', (
        f'Expected status=created, got: {row["status"]}'
    )
    assert row['task_id'] == '42', (
        f'Expected task_id=42, got: {row["task_id"]}'
    )
    assert row['resolved_at'] is not None, 'resolved_at should be set'

    # Assert: result_json contains the tm.add_task return value
    result_json = json.loads(row['result_json'])
    assert result_json.get('id') == '42', (
        f'result_json should contain add_task return: {result_json}'
    )


# ---------------------------------------------------------------------------
# step-23: worker processes a 'drop' decision
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_processes_drop_decision(
    interceptor_with_store, ticket_store, taskmaster,
):
    """When curator returns action='drop' with target_id, the worker:
    - marks ticket as status='combined' with task_id=target_id
    - includes 'drop' and the justification in the reason field
    - does NOT call tm.add_task
    - result_json matches the legacy drop shape: {id, deduplicated, action, justification, ...}
    """
    mock_curator = MagicMock()
    mock_curator.curate = AsyncMock(
        return_value=CuratorDecision(action='drop', target_id='5', justification='dup')
    )
    mock_curator.note_created = MagicMock()
    mock_curator.record_task = AsyncMock()

    with patch.object(
        type(interceptor_with_store), '_get_curator',
        new=AsyncMock(return_value=mock_curator),
    ):
        result = await interceptor_with_store.submit_task(
            project_root='/project',
            title='Duplicate Task',
            description='Should be dropped',
        )
        assert result.get('ticket', '').startswith('tkt_'), f'Got: {result}'
        ticket_id = result['ticket']

        # Let the worker drain
        await asyncio.sleep(0.1)

    # tm.add_task must NOT have been called
    taskmaster.add_task.assert_not_called()

    # Ticket must be terminal with status='combined'
    row = await ticket_store.get(ticket_id)
    assert row is not None
    assert row['status'] == 'combined', f'Expected combined, got {row["status"]}'
    assert row['task_id'] == '5', f'Expected task_id=5, got {row["task_id"]}'
    assert row['resolved_at'] is not None

    # reason must mention 'drop' and 'dup'
    reason = row['reason'] or ''
    assert 'drop' in reason.lower(), f'reason should mention drop: {reason!r}'
    assert 'dup' in reason.lower(), f'reason should include justification: {reason!r}'

    # result_json matches legacy drop shape
    result_data = json.loads(row['result_json'])
    assert result_data.get('id') == '5', f'id mismatch: {result_data}'
    assert result_data.get('deduplicated') is True
    assert result_data.get('action') == 'drop'
    assert 'justification' in result_data
