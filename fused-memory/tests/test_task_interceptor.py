"""Tests for task interceptor middleware."""

import asyncio
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.reconciliation.event_buffer import EventBuffer


@pytest.fixture
def taskmaster():
    tm = AsyncMock()
    tm.get_task = AsyncMock(return_value={'id': '1', 'status': 'pending', 'title': 'Test Task'})
    tm.set_task_status = AsyncMock(return_value={'success': True})
    tm.get_tasks = AsyncMock(return_value={'tasks': []})
    tm.add_task = AsyncMock(return_value={'id': '2', 'title': 'New Task'})
    tm.update_task = AsyncMock(return_value={'success': True})
    tm.add_subtask = AsyncMock(return_value={'id': '1.1'})
    tm.remove_task = AsyncMock(return_value={'success': True})
    tm.add_dependency = AsyncMock(return_value={'success': True})
    tm.remove_dependency = AsyncMock(return_value={'success': True})
    tm.expand_task = AsyncMock(return_value={'subtasks': [{'id': '1.1'}, {'id': '1.2'}]})
    tm.parse_prd = AsyncMock(return_value={'tasks': [{'id': '1'}, {'id': '2'}]})
    return tm


@pytest.fixture
def reconciler():
    r = AsyncMock()
    r.reconcile_task = AsyncMock(return_value={'actions': [{'type': 'knowledge_captured'}]})
    r.reconcile_bulk_tasks = AsyncMock(return_value={'actions': []})
    return r


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    buf = EventBuffer(db_path=tmp_path / 'interceptor_eb.db', buffer_size_threshold=100)
    await buf.initialize()
    yield buf
    await buf.close()


@pytest.fixture
def interceptor(taskmaster, reconciler, event_buffer):
    return TaskInterceptor(taskmaster, reconciler, event_buffer)


@pytest.mark.asyncio
async def test_set_task_status_non_trigger(interceptor, taskmaster, reconciler, event_buffer):
    """Non-triggering status change: emits event, no reconciliation."""
    result = await interceptor.set_task_status('1', 'in-progress', '/project')
    assert result == {'success': True}
    taskmaster.set_task_status.assert_called_once()
    reconciler.reconcile_task.assert_not_called()
    # Event should be buffered
    stats = await event_buffer.get_buffer_stats('project')
    assert stats['size'] == 1


@pytest.mark.asyncio
async def test_set_task_status_done_triggers_async_reconciliation(interceptor, reconciler, event_buffer):
    """Done status triggers async targeted reconciliation."""
    result = await interceptor.set_task_status('1', 'done', '/project')
    assert 'reconciliation' in result
    assert result['reconciliation']['status'] == 'async'
    assert result['reconciliation']['task_id'] == '1'
    # Let the event loop tick so the background task runs
    await asyncio.sleep(0)
    reconciler.reconcile_task.assert_called_once_with(
        task_id='1',
        transition='done',
        project_id='project',
        project_root='/project',
        task_before={'id': '1', 'status': 'pending', 'title': 'Test Task'},
    )


@pytest.mark.asyncio
async def test_set_task_status_blocked_triggers(interceptor, reconciler):
    result = await interceptor.set_task_status('1', 'blocked', '/project')
    assert 'reconciliation' in result
    assert result['reconciliation']['status'] == 'async'
    await asyncio.sleep(0)
    reconciler.reconcile_task.assert_called_once()


@pytest.mark.asyncio
async def test_set_task_status_cancelled_triggers(interceptor, reconciler):
    result = await interceptor.set_task_status('1', 'cancelled', '/project')
    assert 'reconciliation' in result
    assert result['reconciliation']['status'] == 'async'


@pytest.mark.asyncio
async def test_read_operations_no_events(interceptor, taskmaster, event_buffer):
    """Pure reads don't emit events."""
    await interceptor.get_tasks('/project')
    await interceptor.get_task('1', '/project')
    stats = await event_buffer.get_buffer_stats('project')
    assert stats['size'] == 0


@pytest.mark.asyncio
async def test_add_task_emits_event(interceptor, event_buffer):
    await interceptor.add_task('/project', prompt='Test')
    stats = await event_buffer.get_buffer_stats('project')
    assert stats['size'] == 1


@pytest.mark.asyncio
async def test_expand_task_triggers_async_bulk_reconciliation(interceptor, reconciler, event_buffer):
    result = await interceptor.expand_task('1', '/project')
    assert 'reconciliation' in result
    assert result['reconciliation']['status'] == 'async'
    await asyncio.sleep(0)
    reconciler.reconcile_bulk_tasks.assert_called_once()
    stats = await event_buffer.get_buffer_stats('project')
    assert stats['size'] == 1


@pytest.mark.asyncio
async def test_parse_prd_triggers_async_bulk_reconciliation(interceptor, reconciler, event_buffer):
    result = await interceptor.parse_prd('prd.md', '/project')
    assert 'reconciliation' in result
    assert result['reconciliation']['status'] == 'async'
    await asyncio.sleep(0)
    reconciler.reconcile_bulk_tasks.assert_called_once()


@pytest.mark.asyncio
async def test_no_reconciler_still_proxies(taskmaster, event_buffer):
    """Without a reconciler, interceptor still proxies to taskmaster."""
    interceptor = TaskInterceptor(taskmaster, None, event_buffer)
    result = await interceptor.set_task_status('1', 'done', '/project')
    assert result == {'success': True}
    # No reconciliation key
    assert 'reconciliation' not in result


@pytest.mark.asyncio
async def test_remove_task_emits_event(interceptor, event_buffer):
    await interceptor.remove_task('1', '/project')
    stats = await event_buffer.get_buffer_stats('project')
    assert stats['size'] == 1


@pytest.mark.asyncio
async def test_dependency_operations_emit_events(interceptor, event_buffer):
    await interceptor.add_dependency('2', '1', '/project')
    await interceptor.remove_dependency('2', '1', '/project')
    stats = await event_buffer.get_buffer_stats('project')
    assert stats['size'] == 2


@pytest.mark.asyncio
async def test_async_reconciliation_error_logged(interceptor, reconciler, event_buffer):
    """Background reconciliation failure should not propagate to caller."""
    reconciler.reconcile_task = AsyncMock(side_effect=RuntimeError('boom'))
    result = await interceptor.set_task_status('1', 'done', '/project')
    assert result['reconciliation']['status'] == 'async'
    # Let the background task run and fail
    await asyncio.sleep(0)
    # The caller still got a result — error is logged, not raised
    assert 'success' in result


# ── Tests for resolved project_id (step-3) ────────────────────────────


@pytest.mark.asyncio
async def test_event_project_id_is_resolved(interceptor, event_buffer):
    """Event in buffer should have logical project_id, not filesystem path."""
    await interceptor.set_task_status('1', 'in-progress', '/home/leo/src/dark-factory')
    # Buffer should be queryable by the resolved project_id
    stats = await event_buffer.get_buffer_stats('dark_factory')
    assert stats['size'] == 1
    # And NOT by the raw path
    stats_raw = await event_buffer.get_buffer_stats('/home/leo/src/dark-factory')
    assert stats_raw['size'] == 0


@pytest.mark.asyncio
async def test_event_payload_contains_project_root(interceptor, event_buffer):
    """Event payload should include _project_root with original filesystem path."""
    await interceptor.set_task_status('1', 'in-progress', '/home/leo/src/dark-factory')
    events = await event_buffer.drain('dark_factory')
    assert len(events) == 1
    assert events[0].payload['_project_root'] == '/home/leo/src/dark-factory'


@pytest.mark.asyncio
async def test_reconciler_receives_both_ids(interceptor, reconciler):
    """reconcile_task should be called with project_id (logical) and project_root (path)."""
    await interceptor.set_task_status('1', 'done', '/home/leo/src/dark-factory')
    await asyncio.sleep(0)
    reconciler.reconcile_task.assert_called_once_with(
        task_id='1',
        transition='done',
        project_id='dark_factory',
        project_root='/home/leo/src/dark-factory',
        task_before={'id': '1', 'status': 'pending', 'title': 'Test Task'},
    )


@pytest.mark.asyncio
async def test_event_roundtrip_preserves_both_ids(taskmaster, event_buffer):
    """End-to-end: interceptor -> buffer -> drain preserves both project_id and _project_root."""
    interceptor = TaskInterceptor(taskmaster, None, event_buffer)
    project_path = '/home/leo/src/dark-factory'

    # Multiple operations
    await interceptor.set_task_status('1', 'in-progress', project_path)
    await interceptor.add_task(project_path, prompt='New task')
    await interceptor.update_task('1', project_path, prompt='Updated')

    # Buffer queryable by resolved id
    stats = await event_buffer.get_buffer_stats('dark_factory')
    assert stats['size'] == 3

    # Drain by resolved id
    events = await event_buffer.drain('dark_factory')
    assert len(events) == 3

    for ev in events:
        # Event project_id is the logical identifier
        assert ev.project_id == 'dark_factory'
        # Payload carries the original path
        assert ev.payload['_project_root'] == project_path

    # Buffer is now empty
    stats = await event_buffer.get_buffer_stats('dark_factory')
    assert stats['size'] == 0


# ── Tests for None / disconnected taskmaster ───────────────────────


@pytest.mark.asyncio
async def test_none_taskmaster_raises_structured_error(event_buffer):
    """TaskInterceptor(None, None, buf) → get_tasks() raises RuntimeError."""
    interceptor = TaskInterceptor(None, None, event_buffer)
    with pytest.raises(RuntimeError, match='not configured'):
        await interceptor.get_tasks('/project')


@pytest.mark.asyncio
async def test_disconnected_taskmaster_calls_ensure_connected(event_buffer):
    """ensure_connected is called before proxying to taskmaster."""
    tm = AsyncMock()
    tm.ensure_connected = AsyncMock()
    tm.get_tasks = AsyncMock(return_value={'tasks': []})
    interceptor = TaskInterceptor(tm, None, event_buffer)

    await interceptor.get_tasks('/project')
    tm.ensure_connected.assert_called_once()


@pytest.mark.asyncio
async def test_ensure_taskmaster_error_propagates(event_buffer):
    """When ensure_connected raises, the method propagates the error."""
    tm = AsyncMock()
    tm.ensure_connected = AsyncMock(
        side_effect=RuntimeError('Taskmaster reconnection failed: spawn error')
    )
    interceptor = TaskInterceptor(tm, None, event_buffer)

    with pytest.raises(RuntimeError, match='reconnection failed'):
        await interceptor.get_tasks('/project')


# ── Tests for terminal status guard (defense in depth) ──────────────


@pytest.mark.asyncio
async def test_set_task_status_allows_done_to_blocked(taskmaster, reconciler, event_buffer):
    """Transitions from terminal states (done->blocked) are allowed."""
    taskmaster.get_task = AsyncMock(return_value={'id': '1', 'status': 'done', 'title': 'T'})
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status('1', 'blocked', '/project')

    taskmaster.set_task_status.assert_called_once()
    assert 'error' not in result


@pytest.mark.asyncio
async def test_set_task_status_allows_done_to_done(taskmaster, reconciler, event_buffer):
    """Idempotent done->done transitions are a no-op: not forwarded to taskmaster."""
    taskmaster.get_task = AsyncMock(return_value={'id': '1', 'status': 'done', 'title': 'T'})
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status('1', 'done', '/project')

    # Taskmaster.set_task_status must NOT be called for a no-op
    taskmaster.set_task_status.assert_not_called()
    # Result carries no_op flag
    assert result.get('no_op') is True
    assert result.get('success') is True


@pytest.mark.asyncio
async def test_set_task_status_allows_inprogress_to_blocked(taskmaster, reconciler, event_buffer):
    """Normal in-progress->blocked transitions pass through."""
    taskmaster.get_task = AsyncMock(
        return_value={'id': '1', 'status': 'in-progress', 'title': 'T'}
    )
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status('1', 'blocked', '/project')

    taskmaster.set_task_status.assert_called_once()
    assert 'error' not in result


# ── Tests for same-status no-op guard (step-1) ─────────────────────────────


@pytest.mark.asyncio
async def test_set_task_status_done_to_done_noop(taskmaster, reconciler, event_buffer):
    """done->done is a no-op: early return, no taskmaster call, no event, no reconciliation."""
    taskmaster.get_task = AsyncMock(return_value={'id': '1', 'status': 'done', 'title': 'T'})
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status('1', 'done', '/project')

    # Must return a no-op result
    assert result.get('success') is True
    assert result.get('no_op') is True
    assert result.get('task_id') == '1'
    # Taskmaster.set_task_status must NOT have been called
    taskmaster.set_task_status.assert_not_called()
    # No event should be buffered
    stats = await event_buffer.get_buffer_stats('project')
    assert stats['size'] == 0
    # No reconciliation should be triggered
    reconciler.reconcile_task.assert_not_called()


@pytest.mark.asyncio
async def test_set_task_status_cancelled_to_cancelled_noop(taskmaster, reconciler, event_buffer):
    """cancelled->cancelled is a no-op: early return, no taskmaster call, no event, no reconciliation."""
    taskmaster.get_task = AsyncMock(
        return_value={'id': '2', 'status': 'cancelled', 'title': 'T'}
    )
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status('2', 'cancelled', '/project')

    # Must return a no-op result
    assert result.get('success') is True
    assert result.get('no_op') is True
    assert result.get('task_id') == '2'
    # Taskmaster.set_task_status must NOT have been called
    taskmaster.set_task_status.assert_not_called()
    # No event should be buffered
    stats = await event_buffer.get_buffer_stats('project')
    assert stats['size'] == 0
    # No reconciliation should be triggered
    reconciler.reconcile_task.assert_not_called()


# ── Tests for background task retention (step-3) ───────────────────────────


def test_background_tasks_set_exists(taskmaster, reconciler, event_buffer):
    """TaskInterceptor should have a _background_tasks set after init."""
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)
    assert hasattr(interceptor, '_background_tasks')
    assert isinstance(interceptor._background_tasks, set)


@pytest.mark.asyncio
async def test_background_tasks_retained_during_reconciliation(taskmaster, reconciler, event_buffer):
    """Background task should be in _background_tasks while running, removed after completion."""
    # Use a future to control when reconcile_task finishes
    started = asyncio.Event()
    done_future: asyncio.Future = asyncio.get_event_loop().create_future()

    async def slow_reconcile(**kwargs):
        started.set()
        await done_future
        return {'actions': []}

    reconciler.reconcile_task = AsyncMock(side_effect=slow_reconcile)
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    await interceptor.set_task_status('1', 'done', '/project')

    # Wait for the background task to start
    await started.wait()
    # The task should be retained in the set while running
    assert len(interceptor._background_tasks) == 1

    # Let the background task complete
    done_future.set_result(None)
    await asyncio.sleep(0)
    await asyncio.sleep(0)  # Two ticks to ensure done callback fires

    # Task should be removed from the set after completion
    assert len(interceptor._background_tasks) == 0


@pytest.mark.asyncio
async def test_background_tasks_retained_for_bulk_operations(taskmaster, reconciler, event_buffer):
    """Background task from expand_task should be in _background_tasks during execution."""
    started = asyncio.Event()
    done_future: asyncio.Future = asyncio.get_event_loop().create_future()

    async def slow_bulk_reconcile(**kwargs):
        started.set()
        await done_future
        return {'actions': []}

    reconciler.reconcile_bulk_tasks = AsyncMock(side_effect=slow_bulk_reconcile)
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    await interceptor.expand_task('1', '/project')

    await started.wait()
    assert len(interceptor._background_tasks) == 1

    done_future.set_result(None)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert len(interceptor._background_tasks) == 0


# ── Tests for auto-commit scheduling ────────────────────────────────


@pytest.fixture
def committer():
    c = AsyncMock()
    c.commit = AsyncMock()
    return c


@pytest.fixture
def interceptor_with_committer(taskmaster, reconciler, event_buffer, committer):
    return TaskInterceptor(taskmaster, reconciler, event_buffer, committer)


@pytest.mark.asyncio
async def test_write_methods_commit(interceptor_with_committer, committer):
    """All 9 write methods should commit (7 fire-and-forget, 2 awaited for bulk ops)."""
    i = interceptor_with_committer
    pr = '/project'

    await i.set_task_status('1', 'in-progress', pr)
    await i.add_task(pr, prompt='T')
    await i.update_task('1', pr, prompt='U')
    await i.add_subtask('1', pr, title='S')
    await i.remove_task('1', pr)
    await i.add_dependency('2', '1', pr)
    await i.remove_dependency('2', '1', pr)
    await i.expand_task('1', pr)
    await i.parse_prd('prd.md', pr)

    # Let background tasks run
    await asyncio.sleep(0)

    assert committer.commit.call_count == 9
    # Verify project_root is always passed
    for call in committer.commit.call_args_list:
        assert call[0][0] == pr


@pytest.mark.asyncio
async def test_reads_do_not_schedule_commit(taskmaster, event_buffer, committer):
    """get_tasks and get_task should not trigger commits."""
    interceptor = TaskInterceptor(taskmaster, None, event_buffer, committer)
    await interceptor.get_tasks('/project')
    await interceptor.get_task('1', '/project')
    committer.commit.assert_not_called()


@pytest.mark.asyncio
async def test_noop_status_change_no_commit(taskmaster, reconciler, event_buffer, committer):
    """Same-status no-op should not schedule a commit."""
    taskmaster.get_task = AsyncMock(return_value={'id': '1', 'status': 'done', 'title': 'T'})
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer, committer)
    result = await interceptor.set_task_status('1', 'done', '/project')
    assert result.get('no_op') is True
    committer.commit.assert_not_called()


@pytest.mark.asyncio
async def test_no_committer_still_works(taskmaster, event_buffer):
    """task_committer=None should not break any write methods."""
    interceptor = TaskInterceptor(taskmaster, None, event_buffer, None)
    result = await interceptor.add_task('/project', prompt='T')
    assert result == {'id': '2', 'title': 'New Task'}


@pytest.mark.asyncio
async def test_terminal_state_transition_commits(taskmaster, reconciler, event_buffer, committer):
    """Transitions from terminal states (done->blocked) schedule a commit."""
    taskmaster.get_task = AsyncMock(return_value={'id': '1', 'status': 'done', 'title': 'T'})
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer, committer)
    result = await interceptor.set_task_status('1', 'blocked', '/project')
    assert 'error' not in result
    committer.commit.assert_called_once()


# ── Tests for bulk ops awaiting commit (not fire-and-forget) ─────────


@pytest.mark.asyncio
async def test_parse_prd_awaits_commit(taskmaster, reconciler, event_buffer):
    """parse_prd should await the commit, not fire-and-forget."""
    commit_order: list[str] = []

    async def tracking_commit(project_root: str, operation: str) -> None:
        commit_order.append(operation)

    committer = AsyncMock()
    committer.commit = AsyncMock(side_effect=tracking_commit)
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer, committer)

    await interceptor.parse_prd('prd.md', '/project')

    # Commit must have been called already (awaited, not background)
    assert 'parse_prd' in commit_order
    # No background commit tasks for this op
    commit_tasks = [
        t for t in interceptor._background_tasks
        if 'auto-commit' in (t.get_name() or '')
    ]
    assert len(commit_tasks) == 0


@pytest.mark.asyncio
async def test_expand_task_awaits_commit(taskmaster, reconciler, event_buffer):
    """expand_task should await the commit, not fire-and-forget."""
    committer = AsyncMock()
    committer.commit = AsyncMock()
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer, committer)

    await interceptor.expand_task('1', '/project')

    # Commit called synchronously (awaited) — verify it was called
    committer.commit.assert_called_once()
    assert 'expand_task(1)' in committer.commit.call_args[0][1]
    # No background commit tasks
    commit_tasks = [
        t for t in interceptor._background_tasks
        if 'auto-commit' in (t.get_name() or '')
    ]
    assert len(commit_tasks) == 0


# ── Tests for drain ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_drain_empty(taskmaster, event_buffer):
    """drain() with no pending tasks completes immediately."""
    interceptor = TaskInterceptor(taskmaster, None, event_buffer)
    await interceptor.drain()  # should not raise


@pytest.mark.asyncio
async def test_drain_awaits_pending_commits(taskmaster, event_buffer):
    """drain() awaits all pending fire-and-forget commits."""
    commit_done = asyncio.Event()

    async def slow_commit(project_root: str, operation: str) -> None:
        await commit_done.wait()

    committer = AsyncMock()
    committer.commit = AsyncMock(side_effect=slow_commit)
    interceptor = TaskInterceptor(taskmaster, None, event_buffer, committer)

    # Fire-and-forget commits
    await interceptor.add_task('/project', prompt='A')
    await interceptor.add_task('/project', prompt='B')
    await asyncio.sleep(0)  # let tasks start

    # Background tasks should be pending
    assert len(interceptor._background_tasks) >= 1

    # Unblock the commits
    commit_done.set()

    # drain should await them all
    await interceptor.drain()
    assert len(interceptor._background_tasks) == 0
