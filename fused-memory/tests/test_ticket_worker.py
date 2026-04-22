"""Tests for the TaskInterceptor curator worker (ticket queue drain loop)."""

import asyncio
import contextlib
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
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await ti._worker_task


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


# ---------------------------------------------------------------------------
# step-31: tm.add_task failure marks ticket failed (no journal event)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_tm_add_task_failure_marks_ticket_failed(
    interceptor_with_store, ticket_store, taskmaster,
):
    """When tm.add_task raises, the ticket resolves as status='failed',
    reason contains the error message, result_json is NULL, and no
    task_created journal event is emitted.
    """
    taskmaster.add_task = AsyncMock(side_effect=RuntimeError('db locked'))

    # Capture journal calls via the event buffer
    journal_calls = []
    original_journal = interceptor_with_store._journal

    async def capturing_journal(event):
        journal_calls.append(event)
        return await original_journal(event)

    interceptor_with_store._journal = capturing_journal

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
        result = await interceptor_with_store.submit_task(
            project_root='/project',
            title='Failing Task',
            description='Should fail on add_task',
        )
        assert result.get('ticket', '').startswith('tkt_'), f'Got: {result}'
        ticket_id = result['ticket']

        # Let the worker drain
        await asyncio.sleep(0.2)

    # Ticket must be terminal with status='failed'
    row = await ticket_store.get(ticket_id)
    assert row is not None
    assert row['status'] == 'failed', f'Expected failed, got {row["status"]}'
    assert 'db locked' in (row['reason'] or ''), (
        f'reason should mention the error: {row["reason"]!r}'
    )
    assert row['result_json'] is None, (
        f'result_json should be NULL on failure: {row["result_json"]!r}'
    )
    assert row['resolved_at'] is not None

    # No task_created journal event should have been emitted
    from fused_memory.models.reconciliation import EventType
    task_created_events = [
        e for e in journal_calls
        if e.type == EventType.task_created
    ]
    assert len(task_created_events) == 0, (
        f'No task_created event should be emitted on failure: {task_created_events}'
    )


# ---------------------------------------------------------------------------
# step-33: created path emits journal event and schedules commit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_created_path_emits_journal_event_and_schedules_commit(
    interceptor_with_store, ticket_store, taskmaster, event_buffer,
):
    """Regression-pin: on create success, exactly one EventType.task_created
    event is journalled with the task_id in its payload, and _schedule_commit
    is called with operation='add_task' (i.e. task_committer.commit was called).
    """
    from fused_memory.models.reconciliation import EventType

    # Wire a mock task_committer so _schedule_commit actually fires
    mock_committer = MagicMock()
    mock_committer.commit = AsyncMock(return_value=None)
    interceptor_with_store.task_committer = mock_committer

    # Capture journal calls
    journal_calls = []
    original_journal = interceptor_with_store._journal

    async def capturing_journal(event):
        journal_calls.append(event)
        return await original_journal(event)

    interceptor_with_store._journal = capturing_journal

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
        result = await interceptor_with_store.submit_task(
            project_root='/project',
            title='Journal Test Task',
            description='Checking event emission',
        )
        assert result.get('ticket', '').startswith('tkt_'), f'Got: {result}'

        # Let the worker drain and commit task fire
        await asyncio.sleep(0.2)

    # Exactly one task_created event must have been journalled
    task_created_events = [
        e for e in journal_calls
        if hasattr(e, 'type') and e.type == EventType.task_created
    ]
    assert len(task_created_events) == 1, (
        f'Expected exactly 1 task_created event, got: {task_created_events}'
    )

    # Payload must contain task_id
    payload = task_created_events[0].payload
    assert payload.get('task_id') == '42', (
        f'task_created event payload must have task_id=42: {payload}'
    )
    assert payload.get('operation') == 'add_task', (
        f'task_created event payload must have operation=add_task: {payload}'
    )

    # _schedule_commit must have been called with operation='add_task'
    mock_committer.commit.assert_called_once()
    call_args = mock_committer.commit.call_args
    assert 'add_task' in str(call_args), (
        f'task_committer.commit should be called with add_task: {call_args}'
    )


# ---------------------------------------------------------------------------
# step-35: note_created + record_task run under write_lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_note_created_and_record_task_run_under_write_lock(
    interceptor_with_store, ticket_store, taskmaster,
):
    """Verifies that curator.note_created and curator.record_task are both
    called with the correct task_id, in order, and while the write_lock is
    held (sentinel injection).
    """
    from fused_memory.models.scope import resolve_project_id

    project_id = resolve_project_id('/project')
    call_order = []
    lock_held_at_note = []
    lock_held_at_record = []

    def spy_note_created(pid, candidate, task_id):
        call_order.append(('note_created', task_id))
        lock = interceptor_with_store._write_locks.get(project_id)
        lock_held_at_note.append(lock.locked() if lock else False)

    async def spy_record_task(task_id, candidate, pid):
        call_order.append(('record_task', task_id))
        lock = interceptor_with_store._write_locks.get(project_id)
        lock_held_at_record.append(lock.locked() if lock else False)

    mock_curator = MagicMock()
    mock_curator.curate = AsyncMock(
        return_value=CuratorDecision(action='create')
    )
    mock_curator.note_created = spy_note_created
    mock_curator.record_task = spy_record_task

    with patch.object(
        type(interceptor_with_store), '_get_curator',
        new=AsyncMock(return_value=mock_curator),
    ):
        result = await interceptor_with_store.submit_task(
            project_root='/project',
            title='Lock Test',
            description='Checking write_lock',
        )
        assert result.get('ticket', '').startswith('tkt_'), f'Got: {result}'

        await asyncio.sleep(0.2)

    # Both were called in order: note_created before record_task
    assert len(call_order) == 2, f'Expected 2 calls, got {call_order}'
    assert call_order[0][0] == 'note_created', f'First call should be note_created: {call_order}'
    assert call_order[1][0] == 'record_task', f'Second call should be record_task: {call_order}'

    # Both called with task_id='42'
    assert call_order[0][1] == '42', f'note_created task_id mismatch: {call_order[0]}'
    assert call_order[1][1] == '42', f'record_task task_id mismatch: {call_order[1]}'

    # write_lock was held during both calls
    assert lock_held_at_note == [True], (
        f'write_lock should be held during note_created: {lock_held_at_note}'
    )
    assert lock_held_at_record == [True], (
        f'write_lock should be held during record_task: {lock_held_at_record}'
    )


# ---------------------------------------------------------------------------
# step-37: resolve_ticket returns immediately for a terminal ticket
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_ticket_returns_immediately_for_terminal_ticket(
    interceptor_with_store, ticket_store,
):
    """When the ticket is already in a terminal status, resolve_ticket
    returns immediately with {status, task_id} without waiting on any event.
    """
    # Pre-populate a terminal ticket directly in the store
    ticket_id = await ticket_store.submit(
        project_id='project',
        candidate_json='{}',
        ttl_seconds=600,
    )
    await ticket_store.mark_resolved(
        ticket_id, status='created', task_id='42',
    )

    # resolve_ticket should return promptly without blocking
    result = await interceptor_with_store.resolve_ticket(ticket_id, '/project')

    assert result['status'] == 'created', f'Expected status=created: {result}'
    assert result['task_id'] == '42', f'Expected task_id=42: {result}'
    # result_json should NOT be exposed
    assert 'result_json' not in result, f'result_json should not be exposed: {result}'


# ---------------------------------------------------------------------------
# step-39: resolve_ticket wakes when worker completes the ticket
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_ticket_wakes_on_worker_completion(
    interceptor_with_store, ticket_store, taskmaster,
):
    """When resolve_ticket is awaiting a pending ticket, the worker completing
    that ticket wakes the caller promptly (well under the 5s timeout).
    """
    mock_curator = MagicMock()
    mock_curator.curate = AsyncMock(return_value=CuratorDecision(action='create'))
    mock_curator.note_created = MagicMock()
    mock_curator.record_task = AsyncMock()

    with patch.object(
        type(interceptor_with_store), '_get_curator',
        new=AsyncMock(return_value=mock_curator),
    ):
        # Submit the ticket (enqueues ticket_id, starts worker)
        submit_result = await interceptor_with_store.submit_task(
            project_root='/project',
            title='Wake Test',
            description='Checks that resolve wakes on completion',
        )
        assert submit_result.get('ticket', '').startswith('tkt_'), f'Got: {submit_result}'
        ticket_id = submit_result['ticket']

        # Await submit+resolve in parallel; worker should complete the ticket
        # before the 5s timeout expires.
        import time
        start = time.monotonic()
        resolve_result = await asyncio.wait_for(
            interceptor_with_store.resolve_ticket(ticket_id, '/project', timeout_seconds=5.0),
            timeout=6.0,  # outer safety bound
        )
        elapsed = time.monotonic() - start

    assert resolve_result['status'] == 'created', f'Expected created: {resolve_result}'
    assert resolve_result.get('task_id') == '42', f'Expected task_id=42: {resolve_result}'
    assert 'result_json' not in resolve_result, f'result_json should not be exposed: {resolve_result}'
    # Should complete quickly (worker processes immediately)
    assert elapsed < 3.0, f'resolve_ticket took too long ({elapsed:.2f}s)'


# ---------------------------------------------------------------------------
# step-41: resolve_ticket timeout returns failed without mutating the row
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_ticket_timeout_returns_failed_without_mutating_row(
    interceptor_with_store, ticket_store, taskmaster,
):
    """When resolve_ticket times out, it returns a synthetic failed dict
    but the ticket row remains status='pending' (unchanged in the store).
    """
    # Block the worker by using a slow curator mock (won't be called because
    # we block the worker via a never-resolving event, but we also need the
    # ticket to be submitted and in pending state).
    # Strategy: pause the curator so it never responds during our test window.
    paused = asyncio.Event()  # never set during the test

    async def blocking_curate(*args, **kwargs):
        await paused.wait()  # blocks indefinitely during this test
        return CuratorDecision(action='create')

    mock_curator = MagicMock()
    mock_curator.curate = blocking_curate
    mock_curator.note_created = MagicMock()
    mock_curator.record_task = AsyncMock()

    with patch.object(
        type(interceptor_with_store), '_get_curator',
        new=AsyncMock(return_value=mock_curator),
    ):
        submit_result = await interceptor_with_store.submit_task(
            project_root='/project',
            title='Timeout Test',
            description='Should time out',
        )
        assert submit_result.get('ticket', '').startswith('tkt_'), f'Got: {submit_result}'
        ticket_id = submit_result['ticket']

        # Call resolve_ticket with a very short timeout
        result = await interceptor_with_store.resolve_ticket(
            ticket_id, '/project', timeout_seconds=0.1,
        )

    # Result must indicate timeout failure (not raise)
    assert result['status'] == 'failed', f'Expected failed: {result}'
    assert result.get('reason') == 'timeout', f'Expected reason=timeout: {result}'
    assert result.get('task_id') is None, f'Expected task_id=None: {result}'

    # The ticket row must still be pending (no mutation)
    row = await ticket_store.get(ticket_id)
    assert row is not None
    assert row['status'] == 'pending', (
        f'Ticket row should still be pending after timeout: {row["status"]!r}'
    )


# ---------------------------------------------------------------------------
# step-43: resolve_ticket unknown ticket returns failed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_ticket_unknown_ticket_returns_failed(
    interceptor_with_store,
):
    """Calling resolve_ticket with an id that doesn't exist in the store
    returns {status:failed, reason:unknown_ticket} without raising.
    """
    result = await interceptor_with_store.resolve_ticket(
        'tkt_nonexistent', '/project',
    )

    assert result == {'status': 'failed', 'reason': 'unknown_ticket', 'task_id': None}, (
        f'Expected unknown_ticket response: {result}'
    )


# ---------------------------------------------------------------------------
# step-55: close() drains worker and closes ticket store
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_drains_worker_and_closes_store(
    taskmaster, event_buffer, tmp_path,
):
    """interceptor.close() cancels the curator worker and closes the ticket store.

    After close():
    - The worker task is done (cancelled or finished).
    - ticket_store._db is None (closed).
    - A subsequent submit_task raises RuntimeError (interceptor is closed).
    """
    from fused_memory.middleware.ticket_store import TicketStore

    store = TicketStore(tmp_path / 'close_test_tickets.db')
    await store.initialize()

    ti = TaskInterceptor(taskmaster, None, event_buffer, ticket_store=store)

    # Submit a ticket so the worker is lazily started.
    # Use a paused asyncio.Event to block the worker indefinitely without
    # the 'coroutine as iterator' bug that AsyncMock(side_effect=coroutine) has.
    paused_event = asyncio.Event()  # never set during this test

    async def blocking_curate(*args, **kwargs):
        await paused_event.wait()  # blocks until cancelled
        return CuratorDecision(action='create')

    mock_curator = MagicMock()
    mock_curator.curate = blocking_curate
    mock_curator.note_created = MagicMock()
    mock_curator.record_task = AsyncMock()
    mock_curator.reembed_task = AsyncMock()

    with patch.object(ti, '_get_curator', AsyncMock(return_value=mock_curator)):
        await ti.submit_task('/project', title='Test')
        # Worker is now running (blocked on curator).
        assert ti._worker_task is not None
        assert not ti._worker_task.done()

        # Close should cancel the worker and close the store.
        await ti.close()

    # Worker task should be done after close.
    assert ti._worker_task is None or ti._worker_task.done(), (
        'Worker task should be done after close()'
    )
    # Ticket store should be closed.
    assert store._db is None, 'TicketStore._db should be None after close()'

    # Subsequent submit_task should raise or return an error (closed guard).
    result = await ti.submit_task('/project', title='AfterClose')
    assert 'error' in result, f'submit_task after close should return an error: {result}'
