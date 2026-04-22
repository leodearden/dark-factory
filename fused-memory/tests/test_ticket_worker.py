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


# ---------------------------------------------------------------------------
# step-25: worker processes a 'combine' decision
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_processes_combine_decision(
    interceptor_with_store, ticket_store, taskmaster,
):
    """When curator returns action='combine' with target_id and rewritten_task:
    - _execute_combine is called (implying write_lock was held)
    - A reembed background task is created via curator.reembed_task
    - Ticket is status='combined' with task_id='5'
    - tm.add_task is NOT called
    """
    rewritten = RewrittenTask(
        title='Combined Task',
        description='Merged description',
        details='Merged details',
        files_to_modify=[],
        priority='medium',
    )
    mock_curator = MagicMock()
    mock_curator.curate = AsyncMock(
        return_value=CuratorDecision(
            action='combine',
            target_id='5',
            rewritten_task=rewritten,
            justification='similar scope',
        )
    )
    mock_curator.note_created = MagicMock()
    mock_curator.record_task = AsyncMock()
    mock_curator.reembed_task = AsyncMock(return_value=None)

    # Sentinel: track whether _execute_combine ran (it runs under write_lock)
    execute_combine_calls = []

    async def fake_execute_combine(project_root, decision):
        execute_combine_calls.append({'project_root': project_root, 'decision': decision})
        return {'updated': True, 'target_id': decision.target_id}

    with (
        patch.object(
            type(interceptor_with_store), '_get_curator',
            new=AsyncMock(return_value=mock_curator),
        ),
        patch.object(
            interceptor_with_store, '_execute_combine',
            side_effect=fake_execute_combine,
        ),
    ):
        result = await interceptor_with_store.submit_task(
            project_root='/project',
            title='Combine Candidate',
            description='Should be combined',
        )
        assert result.get('ticket', '').startswith('tkt_'), f'Got: {result}'
        ticket_id = result['ticket']

        # Let the worker drain
        await asyncio.sleep(0.2)

    # _execute_combine was called (confirms write_lock path was taken)
    assert len(execute_combine_calls) == 1, (
        f'Expected _execute_combine to be called once, got {execute_combine_calls}'
    )

    # tm.add_task must NOT have been called
    taskmaster.add_task.assert_not_called()

    # Ticket must be terminal with status='combined'
    row = await ticket_store.get(ticket_id)
    assert row is not None
    assert row['status'] == 'combined', f'Expected combined, got {row["status"]}'
    assert row['task_id'] == '5', f'Expected task_id=5, got {row["task_id"]}'
    assert row['resolved_at'] is not None

    # result_json matches legacy combine shape
    result_data = json.loads(row['result_json'])
    assert result_data.get('id') == '5'
    assert result_data.get('deduplicated') is True
    assert result_data.get('action') == 'combine'
    assert result_data.get('title') == 'Combined Task'

    # reembed background task was spawned (curator.reembed_task was called)
    mock_curator.reembed_task.assert_called_once()


# ---------------------------------------------------------------------------
# step-27: worker R4 escalation idempotency short-circuit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_r4_escalation_idempotency_returns_existing_task(
    interceptor_with_store, ticket_store, taskmaster,
):
    """When metadata contains escalation_id + suggestion_hash that match an
    existing non-cancelled task, the worker short-circuits:
    - ticket resolves as status='combined' with task_id=<existing>
    - curator.curate is NOT called
    - tm.add_task is NOT called
    """
    existing_task = {
        'id': '99',
        'title': 'Existing Escalation Task',
        'status': 'pending',
        'metadata': {
            'escalation_id': 'e1',
            'suggestion_hash': 'h1',
        },
    }
    taskmaster.get_tasks = AsyncMock(return_value={'tasks': [existing_task]})

    mock_curator = MagicMock()
    mock_curator.curate = AsyncMock()  # should NOT be called
    mock_curator.note_created = MagicMock()
    mock_curator.record_task = AsyncMock()

    with patch.object(
        type(interceptor_with_store), '_get_curator',
        new=AsyncMock(return_value=mock_curator),
    ):
        result = await interceptor_with_store.submit_task(
            project_root='/project',
            title='Escalation Task',
            description='Should be idempotency-gated',
            metadata={'escalation_id': 'e1', 'suggestion_hash': 'h1'},
        )
        assert result.get('ticket', '').startswith('tkt_'), f'Got: {result}'
        ticket_id = result['ticket']

        # Let the worker drain
        await asyncio.sleep(0.2)

    # curator.curate must NOT have been called (short-circuit)
    mock_curator.curate.assert_not_called()

    # tm.add_task must NOT have been called
    taskmaster.add_task.assert_not_called()

    # Ticket must be terminal with status='combined'
    row = await ticket_store.get(ticket_id)
    assert row is not None
    assert row['status'] == 'combined', f'Expected combined, got {row["status"]}'
    assert row['task_id'] == '99', f'Expected task_id=99, got {row["task_id"]}'
    assert row['resolved_at'] is not None
    assert row['reason'] == 'idempotency_hit'

    # result_json captures the idempotency hit details
    result_data = json.loads(row['result_json'])
    assert result_data.get('id') == '99'
    assert result_data.get('action') == 'idempotency_hit'


# ---------------------------------------------------------------------------
# step-29: worker degrades to create on CuratorFailureError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_curator_failure_degrades_to_create(
    interceptor_with_store, ticket_store, taskmaster,
):
    """When curator.curate raises CuratorFailureError, the worker falls through
    to the create path:
    - tm.add_task IS called
    - ticket resolved status='created'
    - result_json contains a justification indicating degrade-to-create
    """
    from fused_memory.middleware.task_curator import CuratorFailureError

    mock_curator = MagicMock()
    mock_curator.curate = AsyncMock(
        side_effect=CuratorFailureError('boom', timed_out=False, duration_ms=100)
    )
    mock_curator.note_created = MagicMock()
    mock_curator.record_task = AsyncMock()

    with patch.object(
        type(interceptor_with_store), '_get_curator',
        new=AsyncMock(return_value=mock_curator),
    ):
        result = await interceptor_with_store.submit_task(
            project_root='/project',
            title='Degrade Test',
            description='Should fall through to create',
        )
        assert result.get('ticket', '').startswith('tkt_'), f'Got: {result}'
        ticket_id = result['ticket']

        # Let the worker drain
        await asyncio.sleep(0.2)

    # tm.add_task MUST have been called (fallback to create)
    taskmaster.add_task.assert_called_once()

    # Ticket must be terminal with status='created'
    row = await ticket_store.get(ticket_id)
    assert row is not None
    assert row['status'] == 'created', f'Expected created, got {row["status"]}'
    assert row['task_id'] == '42', f'Expected task_id=42, got {row["task_id"]}'
    assert row['resolved_at'] is not None

    # result_json contains a justification indicating degrade-to-create
    result_data = json.loads(row['result_json'])
    assert 'id' in result_data, f'result_json should have id: {result_data}'
    # Must indicate that this was a curator-failure fallback
    degrade_hint = result_data.get('curator_degrade_reason', '')
    assert degrade_hint, (
        f'result_json should indicate curator degrade-to-create: {result_data}'
    )
