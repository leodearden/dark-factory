"""Tests for task interceptor middleware."""

import asyncio
import contextlib
import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from _fm_helpers import make_8df8_scenario
from _fm_helpers import submit_and_resolve as _submit_and_resolve

from fused_memory.config.schema import CuratorConfig, FusedMemoryConfig
from fused_memory.middleware.task_curator import CuratorDecision, RewrittenTask
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.models.scope import resolve_project_id
from fused_memory.reconciliation.event_buffer import EventBuffer


@pytest.fixture
def taskmaster():
    tm = AsyncMock()
    tm.get_task = AsyncMock(return_value={'id': '1', 'status': 'pending', 'title': 'Test Task'})
    tm.set_task_status = AsyncMock(return_value={'success': True})
    tm.get_tasks = AsyncMock(return_value={'tasks': []})
    tm.add_task = AsyncMock(return_value={'id': '2', 'title': 'New Task'})
    tm.update_task = AsyncMock(return_value={'success': True})
    tm.remove_tasks = AsyncMock(return_value={'success': True})
    tm.add_dependency = AsyncMock(return_value={'success': True})
    tm.remove_dependency = AsyncMock(return_value={'success': True})
    return tm


@pytest.fixture
def reconciler():
    r = AsyncMock()
    r.reconcile_task = AsyncMock(return_value={'actions': [{'type': 'knowledge_captured'}]})
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


@pytest_asyncio.fixture
async def interceptor_facade(taskmaster, reconciler, event_buffer, tmp_path):
    """Interceptor variant wired with a real TicketStore for facade tests."""
    from fused_memory.middleware.ticket_store import TicketStore

    store = TicketStore(tmp_path / 'facade_tickets.db')
    await store.initialize()
    ti = TaskInterceptor(taskmaster, reconciler, event_buffer, ticket_store=store)
    yield ti
    await store.close()
    for t in list(ti._worker_tasks.values()):
        if not t.done():
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t


def test_task_interceptor_has_no_add_task_method():
    """Facade removal contract: TaskInterceptor must no longer expose add_task.

    This test is RED until step-4 deletes the method from task_interceptor.py.
    """
    assert not hasattr(TaskInterceptor, 'add_task'), (
        'TaskInterceptor.add_task must be removed; migrate callers to submit_task + resolve_ticket'
    )


@pytest.mark.asyncio
async def test_submit_and_resolve_helper_returns_legacy_shape(
    interceptor_facade,
    taskmaster,
):
    """_submit_and_resolve returns the same dict shape the old add_task facade returned.

    Verifies that the helper correctly reconstructs result_json into a dict
    with the 'id' and 'title' keys that downstream assertions rely on.
    """
    from fused_memory.middleware.task_curator import CuratorDecision

    # _mock_curator is defined later in the module; Python resolves at call-time.
    interceptor_facade._curator = _mock_curator(CuratorDecision(action='create'))
    result = await _submit_and_resolve(interceptor_facade, '/project', title='Test')
    # taskmaster.add_task fixture returns {'id': '2', 'title': 'New Task'}
    assert 'id' in result, f'result missing id key: {result}'
    assert 'title' in result, f'result missing title key: {result}'
    assert result['id'] == '2'
    assert result['title'] == 'New Task'


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
async def test_set_task_status_done_triggers_async_reconciliation(
    interceptor, reconciler, event_buffer
):
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
        reopen_reason=None,
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


# ------------------------------------------------------------------
# merge-deferred invariant regression guards (task 1519,
# PRD orchestrator-atomic-train-merge §9.2)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_task_status_merge_deferred_does_not_trigger_reconciliation(
    interceptor, taskmaster, reconciler, event_buffer
):
    """Transitioning to merge-deferred must NOT fire targeted reconciliation.

    merge-deferred is a non-terminal holding state; targeted reconciliation
    only fires for STATUS_TRIGGERS = {done, blocked, cancelled, deferred}.
    merge-deferred is deliberately excluded from that set so the reconciler
    is not invoked spuriously on every atomic-train hold.
    Regression guard: will FAIL if merge-deferred is ever added to STATUS_TRIGGERS.
    """
    result = await interceptor.set_task_status('1', 'merge-deferred', '/project')
    # Let the event loop tick so any accidentally-scheduled background tasks run.
    await asyncio.sleep(0)
    assert 'reconciliation' not in result, (
        f"Expected no reconciliation key for merge-deferred, got result={result}"
    )
    reconciler.reconcile_task.assert_not_called()


@pytest.mark.asyncio
async def test_merge_deferred_is_non_terminal(interceptor, taskmaster):
    """Transitions OUT of merge-deferred must succeed without reopen_reason.

    merge-deferred is NOT in TERMINAL_STATUSES = {done, cancelled}; the
    terminal-exit gate must not fire on transitions: merge-deferred → in-progress
    and merge-deferred → review.
    The second leg uses 'review' (not 'blocked') because 'blocked' is in
    STATUS_TRIGGERS and would spawn a fire-and-forget reconciliation task that
    the test teardown would destroy with a warning. The terminal-exit invariant
    is orthogonal to whether the target status fires reconciliation.
    Regression guard: will FAIL if merge-deferred is ever added to TERMINAL_STATUSES.
    """
    # Simulate a task that is currently in merge-deferred holding state.
    taskmaster.get_task = AsyncMock(
        return_value={'id': '1', 'status': 'merge-deferred', 'title': 'Test Task'}
    )

    # merge-deferred → in-progress (sibling-driven re-dispatch), no reopen_reason.
    result_in_progress = await interceptor.set_task_status('1', 'in-progress', '/project')
    assert 'error' not in result_in_progress, (
        f"Expected success for merge-deferred→in-progress, got: {result_in_progress}"
    )
    # The underlying taskmaster.set_task_status should have been invoked (gate did NOT fire).
    taskmaster.set_task_status.assert_called()

    taskmaster.set_task_status.reset_mock()

    # merge-deferred → review (e.g. operator inspects the hold), no reopen_reason.
    # Using 'review' rather than 'blocked' here because 'blocked' IS in STATUS_TRIGGERS;
    # that would spawn a fire-and-forget reconciliation task, causing 'Task was destroyed
    # but it is pending!' warnings on some runners. The invariant being locked is purely
    # 'old_status=merge-deferred bypasses the terminal-exit gate', which is orthogonal
    # to whether the NEW status triggers reconciliation.
    result_review = await interceptor.set_task_status('1', 'review', '/project')
    assert 'error' not in result_review, (
        f"Expected success for merge-deferred→review, got: {result_review}"
    )
    taskmaster.set_task_status.assert_called()


@pytest.mark.asyncio
async def test_get_statuses_returns_merge_deferred_verbatim(interceptor, taskmaster):
    """get_statuses must return 'merge-deferred' verbatim in the status mapping.

    The holding state is a first-class status; callers (orchestrator, dashboard)
    must receive the exact string so they can distinguish it from other statuses.
    """
    taskmaster.get_statuses_raw = AsyncMock(return_value={'42': 'merge-deferred'})
    mapping = await interceptor.get_statuses('/project')
    assert '42' in mapping, f"Expected task '42' in mapping, got keys: {list(mapping.keys())}"
    assert mapping['42'] == 'merge-deferred', (
        f"Expected mapping['42'] == 'merge-deferred', got: {mapping['42']!r}"
    )


@pytest.mark.asyncio
async def test_read_operations_no_events(interceptor, taskmaster, event_buffer):
    """Pure reads don't emit events."""
    await interceptor.get_tasks('/project')
    await interceptor.get_task('1', '/project')
    stats = await event_buffer.get_buffer_stats('project')
    assert stats['size'] == 0


@pytest.mark.asyncio
async def test_add_task_emits_event(interceptor_facade, event_buffer):
    """add_task (facade path) emits a task_created event after the worker resolves."""
    await _submit_and_resolve(interceptor_facade, '/project', prompt='Test')
    stats = await event_buffer.get_buffer_stats('project')
    assert stats['size'] == 1


@pytest.mark.asyncio
async def test_add_task_persists_metadata_atomically(interceptor_facade, taskmaster):
    """R5: add_task with metadata forwards it to tm.add_task in one call.

    The racy two-step pattern (add_task then update_task(metadata=...)) is
    gone; metadata must be written atomically to prevent a concurrent
    reader from observing a task without its files_to_modify — the bug
    that left #1922/#1923/#1924 running in parallel.

    After step-46 (facade rewrite) this still goes through submit+resolve;
    the worker writes metadata to tm.add_task inside _process_add_ticket.
    """
    import json

    metadata = {'source': 'review-cycle', 'files': ['my-project/src']}
    result = await _submit_and_resolve(
        interceptor_facade, '/project', prompt='Test', metadata=metadata
    )
    assert result == {'id': '2', 'title': 'New Task'}
    taskmaster.add_task.assert_called_once()
    kwargs = taskmaster.add_task.call_args.kwargs
    # Metadata forwarded as a JSON string (the MCP wire format).
    assert kwargs.get('metadata') == json.dumps(metadata)
    # No follow-up update_task for metadata — the atomic path wrote it.
    taskmaster.update_task.assert_not_called()


@pytest.mark.asyncio
async def test_add_task_metadata_string_passed_through(interceptor_facade, taskmaster):
    """Pre-serialised metadata JSON is forwarded unchanged."""
    metadata_json = '{"escalation_id":"esc-1","suggestion_hash":"x"}'
    await _submit_and_resolve(
        interceptor_facade,
        '/project',
        prompt='Test',
        metadata=metadata_json,
    )
    kwargs = taskmaster.add_task.call_args.kwargs
    assert kwargs.get('metadata') == metadata_json


@pytest.mark.asyncio
async def test_add_task_without_metadata_skips_update(interceptor_facade, taskmaster):
    """add_task without metadata does not call update_task."""
    await _submit_and_resolve(interceptor_facade, '/project', prompt='Test')
    taskmaster.update_task.assert_not_called()
    # Backend still receives metadata=None kwarg but the value is falsy.
    kwargs = taskmaster.add_task.call_args.kwargs
    assert kwargs.get('metadata') in (None, '')


@pytest.mark.asyncio
async def test_add_task_falls_back_to_two_step_on_typeerror(event_buffer, tmp_path):
    """Legacy fallback: a backend that rejects ``metadata=`` still works.

    ``TaskmasterBackend.add_task`` on older installs may not accept the
    new ``metadata`` kwarg (the taskmaster-ai MCP tool was extended in
    R5). Keep the fallback during rollout so mixed versions don't break.
    """
    import json

    from fused_memory.middleware.ticket_store import TicketStore

    tm = AsyncMock()
    tm.get_tasks = AsyncMock(return_value={'tasks': []})
    tm.update_task = AsyncMock(return_value={'success': True})

    call_log: list[dict] = []

    async def add_task(**kwargs):
        call_log.append(kwargs)
        if 'metadata' in kwargs:
            # First attempt: simulate old-signature backend rejecting
            # the unknown kwarg.
            raise TypeError("add_task() got an unexpected keyword argument 'metadata'")
        return {'id': '7', 'title': 'Legacy'}

    tm.add_task = add_task

    store = TicketStore(tmp_path / 'fallback_tickets.db')
    await store.initialize()
    interceptor: TaskInterceptor | None = None
    try:
        interceptor = TaskInterceptor(tm, None, event_buffer, ticket_store=store)
        metadata = {'escalation_id': 'esc-x', 'suggestion_hash': 'h'}
        await _submit_and_resolve(interceptor, '/project', prompt='Test', metadata=metadata)

        # Two add_task attempts: atomic first (with metadata), retry without.
        assert len(call_log) == 2
        assert 'metadata' in call_log[0]
        assert 'metadata' not in call_log[1]
        # Legacy update_task follow-up ran because atomic write failed.
        tm.update_task.assert_called_once()
        kwargs = tm.update_task.call_args.kwargs
        assert kwargs['task_id'] == '7'
        assert kwargs['metadata'] == json.dumps(metadata)
    finally:
        await store.close()
        if interceptor is not None:
            for t in list(interceptor._worker_tasks.values()):
                if not t.done():
                    t.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await t


# ─────────────────────────────────────────────────────────────────────
# WP-B: fire-and-forget event queue
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_task_with_queue_persists_to_real_sqlite(taskmaster, tmp_path):
    """WP-B smoke: end-to-end through real EventQueue + real EventBuffer.

    No mocks on the journal path — this catches wiring mistakes that
    unit tests with AsyncMock(EventBuffer) would miss.
    """
    from fused_memory.middleware.ticket_store import TicketStore
    from fused_memory.reconciliation.event_queue import EventQueue

    buf = EventBuffer(db_path=tmp_path / 'wpb_smoke.db', buffer_size_threshold=100)
    await buf.initialize()
    queue = EventQueue(
        buf,
        dead_letter_path=tmp_path / 'dl.jsonl',
        maxsize=100,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.1,
        shutdown_flush_seconds=2.0,
    )
    await queue.start()

    store = TicketStore(tmp_path / 'wpb_tickets.db')
    await store.initialize()
    interceptor = TaskInterceptor(taskmaster, None, buf, event_queue=queue, ticket_store=store)
    try:
        await _submit_and_resolve(interceptor, '/project', prompt='Test 1')
        await interceptor.set_task_status('1', 'in-progress', '/project')
        await interceptor.remove_tasks(['1'], '/project')
        # Let the drainer catch up (5s budget tolerates xdist load on slow CI).
        await queue._drain_for_test(timeout=5.0)

        stats = await buf.get_buffer_stats('project')
        # 3 events: task_created + task_status_changed + task_deleted
        assert stats['size'] == 3
        qs = queue.stats()
        assert qs['events_committed'] == 3
        assert qs['dead_letters'] == 0
        assert qs['overflow_drops'] == 0
    finally:
        await store.close()
        for t in list(interceptor._worker_tasks.values()):
            if not t.done():
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t
        await queue.close()
        await buf.close()


@pytest.mark.asyncio
async def test_add_task_hot_path_immunity_with_queue(taskmaster, tmp_path):
    """WP-B: add_task must return fast even when the event buffer is locked.

    Before WP-B, a locked ``reconciliation.db`` surfaced as an MCP error and
    agents retried → duplicate tasks. With the EventQueue wired in, the
    hot path enqueues non-blocking and returns immediately; journal
    persistence becomes eventually consistent.
    """
    import time

    import aiosqlite

    from fused_memory.middleware.ticket_store import TicketStore
    from fused_memory.reconciliation.event_queue import EventQueue

    # Buffer whose push always raises — simulates the 2026-04-17 lock state.
    buf = AsyncMock()
    buf.push = AsyncMock(side_effect=aiosqlite.OperationalError('database is locked'))

    queue = EventQueue(
        buf,
        dead_letter_path=tmp_path / 'dl.jsonl',
        maxsize=1000,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.05,
        shutdown_flush_seconds=0.1,
    )
    await queue.start()

    store = TicketStore(tmp_path / 'hotpath_tickets.db')
    await store.initialize()
    interceptor = TaskInterceptor(taskmaster, None, buf, event_queue=queue, ticket_store=store)
    try:
        t0 = time.perf_counter()
        result = await _submit_and_resolve(interceptor, '/project', prompt='Test')
        elapsed = time.perf_counter() - t0
        # Canonical write returned successfully — no exception from lock.
        assert result == {'id': '2', 'title': 'New Task'}
        # Under 500ms budget even with SQLite pinned.
        assert elapsed < 0.5, f'hot path took {elapsed:.3f}s under lock'
        # The event is either queued, in-flight (being retried), dead-lettered,
        # or committed — but NOT raised to the caller.  With the facade path,
        # the worker emits the event during resolve, so multiple asyncio ticks
        # pass before the assertion runs; the drain task typically dequeues the
        # event before we get here, putting it in retry_in_flight.
        stats = queue.stats()
        in_system = (
            stats['queue_depth']
            + stats['dead_letters']
            + stats['events_committed']
            + stats.get('retry_in_flight', 0)
        )
        assert in_system >= 1, f'event vanished from queue tracking: {stats}'
    finally:
        await store.close()
        for t in list(interceptor._worker_tasks.values()):
            if not t.done():
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t
        await queue.close()


# ─────────────────────────────────────────────────────────────────────
# Curator gate integration
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def curator_enabled_config():
    cfg = FusedMemoryConfig()
    cfg.curator = CuratorConfig(enabled=True)
    return cfg


@pytest_asyncio.fixture
async def curator_interceptor(
    taskmaster, reconciler, event_buffer, curator_enabled_config, tmp_path
):
    from fused_memory.middleware.ticket_store import TicketStore

    store = TicketStore(tmp_path / 'curator_tickets.db')
    await store.initialize()
    ti = TaskInterceptor(
        taskmaster,
        reconciler,
        event_buffer,
        config=curator_enabled_config,
        ticket_store=store,
    )
    yield ti
    await store.close()
    for _wt in list(ti._worker_tasks.values()):
        if not _wt.done():
            _wt.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await _wt


def _mock_curator(decision: CuratorDecision) -> MagicMock:
    """Mock TaskCurator returning a fixed decision."""
    curator = MagicMock()
    curator.curate = AsyncMock(return_value=decision)

    # curate_batch delegates to curate() per candidate so that assertions on
    # curator.curate (e.g. assert_called_once) continue to pass for tests that
    # use the batch worker path.  The inner curate call uses the same AsyncMock,
    # so call counts accumulate correctly.
    async def _curate_batch(candidates, *a, **kw):
        return [await curator.curate(c, *a, **kw) for c in candidates]

    curator.curate_batch = AsyncMock(side_effect=_curate_batch)
    curator.record_task = AsyncMock()
    curator.reembed_task = AsyncMock()
    # note_created is a plain sync method on the real TaskCurator.
    curator.note_created = MagicMock()
    return curator


def _seed_existing_r4_task(
    taskmaster,
    *,
    task_id: str,
    escalation_id: str,
    suggestion_hash: str,
    title: str = 'Existing R4 task',
    status: str = 'pending',
) -> None:
    """Seed taskmaster.get_tasks with a single pending task carrying R4 idempotency keys."""
    taskmaster.get_tasks = AsyncMock(
        return_value={
            'tasks': [
                {
                    'id': task_id,
                    'title': title,
                    'status': status,
                    'metadata': {
                        'escalation_id': escalation_id,
                        'suggestion_hash': suggestion_hash,
                    },
                },
            ],
        }
    )


def _assert_r4_common(curator_mock, taskmaster) -> None:
    """Shared tail assertions for R4 idempotency-hit tests.

    Both the add_task and submit_task entry-path tests must verify that
    the curator was bypassed and no new task was created.
    """
    curator_mock.curate.assert_not_called()
    taskmaster.add_task.assert_not_called()


@pytest.mark.asyncio
async def test_curator_drop_short_circuits_add_task(
    curator_interceptor,
    taskmaster,
):
    """A drop decision returns the target_id without calling tm.add_task."""
    decision = CuratorDecision(
        action='drop',
        target_id='99',
        justification='already covered by task 99',
    )
    curator_interceptor._curator = _mock_curator(decision)

    result = await _submit_and_resolve(
        curator_interceptor,
        '/project',
        title='Fix parser bug',
        description='The parser explodes on empty input',
    )

    assert result['id'] == '99'
    assert result['deduplicated'] is True
    assert result['action'] == 'drop'
    taskmaster.add_task.assert_not_called()


@pytest.mark.asyncio
async def test_curator_combine_updates_target_and_returns_id(
    curator_interceptor,
    taskmaster,
):
    """A combine decision updates the target via update_task and returns its id."""
    rewritten = RewrittenTask(
        title='Harden parser',
        description='Combined parser hardening',
        details='Fix line 42; add test for empty input at tests/test_parser.py:88',
        files_to_modify=['src/parser.py', 'tests/test_parser.py'],
        priority='high',
    )
    decision = CuratorDecision(
        action='combine',
        target_id='50',
        target_fingerprint='Test Task',  # matches taskmaster fixture's mocked title
        rewritten_task=rewritten,
        justification='same root cause as task 50',
    )
    curator_interceptor._curator = _mock_curator(decision)

    result = await _submit_and_resolve(
        curator_interceptor,
        '/project',
        title='Fix parser on empty input',
        description='Parser panics on empty string',
    )

    assert result['id'] == '50'
    assert result['action'] == 'combine'
    # Combine writes structured fields directly — no prompt, no LLM rewrite.
    taskmaster.update_task.assert_called_once()
    call = taskmaster.update_task.call_args
    assert call.kwargs['task_id'] == '50'
    assert call.kwargs.get('prompt') is None
    assert call.kwargs['title'] == 'Harden parser'
    assert call.kwargs['priority'] == 'high'
    assert 'line 42' in call.kwargs['details']  # specifics preserved verbatim
    taskmaster.add_task.assert_not_called()


@pytest.mark.asyncio
async def test_curator_create_proceeds_with_add_task(
    curator_interceptor,
    taskmaster,
):
    """A create decision forwards to tm.add_task normally."""
    decision = CuratorDecision(action='create', justification='genuinely new')
    curator_interceptor._curator = _mock_curator(decision)

    result = await _submit_and_resolve(
        curator_interceptor,
        '/project',
        title='Novel unrelated work',
    )

    assert result == {'id': '2', 'title': 'New Task'}
    taskmaster.add_task.assert_called_once()


@pytest.mark.asyncio
async def test_curator_combine_failure_falls_through_to_create(
    curator_interceptor,
    taskmaster,
):
    """If tm.update_task raises during combine, fall back to creating the task."""
    rewritten = RewrittenTask(
        title='x',
        description='',
        details='d',
        files_to_modify=[],
        priority='medium',
    )
    decision = CuratorDecision(
        action='combine',
        target_id='50',
        target_fingerprint='Test Task',  # matches fixture — guard passes
        rewritten_task=rewritten,
        justification='...',
    )
    curator_interceptor._curator = _mock_curator(decision)
    taskmaster.update_task.side_effect = RuntimeError('taskmaster failed')

    result = await _submit_and_resolve(
        curator_interceptor,
        '/project',
        title='Fix x',
    )

    # Fell through to create path
    assert result == {'id': '2', 'title': 'New Task'}
    taskmaster.add_task.assert_called_once()


def test_task_interceptor_has_no_add_subtask_method():
    """TaskInterceptor must NOT have an add_subtask method after DF-D (task 1543).

    This is a RED assertion: it fails while add_subtask is still present and
    passes once step-4 deletes it.
    """
    from fused_memory.middleware.task_interceptor import TaskInterceptor
    assert not hasattr(TaskInterceptor, 'add_subtask'), (
        'TaskInterceptor.add_subtask still exists; '
        'DF-D (task 1543) step-4 must delete it.'
    )


# ─────────────────────────────────────────────────────────────────────
# WP-F: combine-safety guard (fingerprint + status + audit log)
# ─────────────────────────────────────────────────────────────────────


def _combine_audit_lines(audit_dir):
    """Read JSONL records written by _append_combine_audit."""
    path = audit_dir / 'combine_audit.jsonl'
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text(encoding='utf-8').splitlines() if ln.strip()]
    import json as _json

    return [_json.loads(ln) for ln in lines]


@pytest.fixture
def audit_dir(tmp_path, monkeypatch):
    """Redirect combine_audit.jsonl writes to a per-test tmp dir."""
    monkeypatch.setenv('DARK_FACTORY_DATA_DIR', str(tmp_path))
    return tmp_path


@pytest.mark.asyncio
async def test_curator_combine_fingerprint_match_proceeds(
    curator_interceptor,
    taskmaster,
    audit_dir,
):
    """Fingerprint matches the live target → combine proceeds + audit written."""
    # Taskmaster fixture returns title='Test Task' for any get_task.
    rewritten = RewrittenTask(
        title='Unified parser work',
        description='Combined',
        details='Do the thing at src/parser.py:42',
        files_to_modify=['src/parser.py'],
        priority='high',
    )
    decision = CuratorDecision(
        action='combine',
        target_id='50',
        target_fingerprint='Test Task',
        rewritten_task=rewritten,
        justification='same concern',
    )
    curator_interceptor._curator = _mock_curator(decision)

    result = await _submit_and_resolve(curator_interceptor, '/project', title='candidate')

    assert result['action'] == 'combine'
    assert result['id'] == '50'
    taskmaster.update_task.assert_called_once()
    records = _combine_audit_lines(audit_dir)
    assert len(records) == 1
    rec = records[0]
    assert rec['target_id'] == '50'
    assert rec['old']['title'] == 'Test Task'
    assert rec['old']['status'] == 'pending'
    assert rec['new']['title'] == 'Unified parser work'
    assert rec['justification_truncated'].startswith('same concern')
    assert 'curator_decision_id' in rec and rec['curator_decision_id']


@pytest.mark.asyncio
async def test_curator_combine_fingerprint_mismatch_aborts(
    curator_interceptor,
    taskmaster,
    audit_dir,
):
    """Fingerprint doesn't match live target → abort, fall through to create."""
    rewritten = RewrittenTask(
        title='x',
        description='',
        details='d',
        files_to_modify=[],
        priority='medium',
    )
    decision = CuratorDecision(
        action='combine',
        target_id='50',
        target_fingerprint='Wrong Title',  # fixture returns 'Test Task'
        rewritten_task=rewritten,
        justification='...',
    )
    curator_interceptor._curator = _mock_curator(decision)

    result = await _submit_and_resolve(curator_interceptor, '/project', title='Fix x')

    # Fell through to create path — combine rejected.
    assert result == {'id': '2', 'title': 'New Task'}
    taskmaster.update_task.assert_not_called()
    taskmaster.add_task.assert_called_once()
    assert _combine_audit_lines(audit_dir) == []


@pytest.mark.asyncio
async def test_curator_combine_missing_fingerprint_aborts(
    curator_interceptor,
    taskmaster,
    audit_dir,
):
    """Decision with no fingerprint (LLM skipped the field) → abort."""
    rewritten = RewrittenTask(
        title='x',
        description='',
        details='d',
        files_to_modify=[],
        priority='medium',
    )
    decision = CuratorDecision(
        action='combine',
        target_id='50',
        target_fingerprint=None,
        rewritten_task=rewritten,
        justification='...',
    )
    curator_interceptor._curator = _mock_curator(decision)

    result = await _submit_and_resolve(curator_interceptor, '/project', title='Fix x')

    assert result == {'id': '2', 'title': 'New Task'}
    taskmaster.update_task.assert_not_called()
    assert _combine_audit_lines(audit_dir) == []


@pytest.mark.asyncio
async def test_curator_combine_target_done_aborts(
    curator_interceptor,
    taskmaster,
    audit_dir,
):
    """Target with status=done → abort (would silently drop candidate work)."""
    taskmaster.get_task = AsyncMock(
        return_value={
            'id': '50',
            'status': 'done',
            'title': 'Done Task',
        }
    )
    rewritten = RewrittenTask(
        title='x',
        description='',
        details='d',
        files_to_modify=[],
        priority='medium',
    )
    decision = CuratorDecision(
        action='combine',
        target_id='50',
        target_fingerprint='Done Task',  # fingerprint matches
        rewritten_task=rewritten,
        justification='...',
    )
    curator_interceptor._curator = _mock_curator(decision)

    result = await _submit_and_resolve(curator_interceptor, '/project', title='Fix x')

    assert result == {'id': '2', 'title': 'New Task'}
    taskmaster.update_task.assert_not_called()
    assert _combine_audit_lines(audit_dir) == []


@pytest.mark.asyncio
async def test_curator_combine_target_cancelled_aborts(
    curator_interceptor,
    taskmaster,
    audit_dir,
):
    """Target with status=cancelled → abort, same reasoning as done."""
    taskmaster.get_task = AsyncMock(
        return_value={
            'id': '50',
            'status': 'cancelled',
            'title': 'Cancelled Task',
        }
    )
    rewritten = RewrittenTask(
        title='x',
        description='',
        details='d',
        files_to_modify=[],
        priority='medium',
    )
    decision = CuratorDecision(
        action='combine',
        target_id='50',
        target_fingerprint='Cancelled Task',
        rewritten_task=rewritten,
        justification='...',
    )
    curator_interceptor._curator = _mock_curator(decision)

    result = await _submit_and_resolve(curator_interceptor, '/project', title='Fix x')

    assert result == {'id': '2', 'title': 'New Task'}
    taskmaster.update_task.assert_not_called()
    assert _combine_audit_lines(audit_dir) == []


@pytest.mark.asyncio
async def test_curator_combine_fingerprint_normalization(
    curator_interceptor,
    taskmaster,
    audit_dir,
):
    """Case / whitespace drift on the title is tolerated by the guard."""
    taskmaster.get_task = AsyncMock(
        return_value={
            'id': '50',
            'status': 'pending',
            'title': 'Harden Parser for Empty Input',
        }
    )
    rewritten = RewrittenTask(
        title='Harden parser',
        description='',
        details='d',
        files_to_modify=[],
        priority='medium',
    )
    decision = CuratorDecision(
        action='combine',
        target_id='50',
        # extra whitespace + different case — should still match
        target_fingerprint='   harden  parser   for empty INPUT   ',
        rewritten_task=rewritten,
        justification='normalize',
    )
    curator_interceptor._curator = _mock_curator(decision)

    result = await _submit_and_resolve(curator_interceptor, '/project', title='c')

    assert result['action'] == 'combine'
    taskmaster.update_task.assert_called_once()


@pytest.mark.asyncio
async def test_concurrent_add_task_produces_single_task(
    taskmaster,
    reconciler,
    event_buffer,
    curator_enabled_config,
    tmp_path,
):
    """Two concurrent add_task calls for identical candidates produce
    exactly one new task. The second is caught by the pre-LLM
    exact-match short-circuit after the first's ``note_created`` fires
    inside the project lock.

    Regression: plans/floating-snuggling-pebble.md §R3. Before R3,
    reviewers could create #1922/#1923 as twin tasks because Qdrant's
    record_task was fire-and-forget and the second triage's embedding
    lookup missed the first task's vector.
    """
    from fused_memory.middleware.task_curator import (
        CuratorDecision,
        TaskCurator,
    )

    # Use a real curator so the exact-match cache is exercised; stub
    # corpus + LLM so we don't spin up Qdrant.
    async def empty_corpus(*a, **k):
        return [], {'anchor': 0, 'module': 0, 'embedding': 0, 'dependency': 0}

    real_curator = TaskCurator(config=curator_enabled_config, taskmaster=taskmaster)
    real_curator.record_task = AsyncMock()

    llm_calls = 0

    async def fake_call_llm(*a, **k):
        nonlocal llm_calls
        llm_calls += 1
        return CuratorDecision(action='create', justification='novel')

    real_curator._build_corpus = empty_corpus  # type: ignore[method-assign]
    real_curator._call_llm = fake_call_llm  # type: ignore[method-assign]

    from fused_memory.middleware.ticket_store import TicketStore

    store = TicketStore(tmp_path / 'concurrent_tickets.db')
    await store.initialize()
    interceptor = TaskInterceptor(
        taskmaster,
        reconciler,
        event_buffer,
        config=curator_enabled_config,
        ticket_store=store,
    )
    interceptor._curator = real_curator

    # Give each add_task its own unique task_id. First call creates '100',
    # second (if it ever reaches tm.add_task) would create '101'.
    add_task_counter = {'n': 99}

    async def fake_add_task(**kwargs):
        add_task_counter['n'] += 1
        return {'id': str(add_task_counter['n']), 'title': 'x'}

    taskmaster.add_task = fake_add_task

    candidate_kwargs: dict[str, Any] = dict(
        title='Log release-mode warning on duplicate template names',
        description='...',
    )

    try:
        results = await asyncio.gather(
            _submit_and_resolve(interceptor, '/project', **candidate_kwargs),
            _submit_and_resolve(interceptor, '/project', **candidate_kwargs),
        )
    finally:
        await store.close()
        for _wt in list(interceptor._worker_tasks.values()):
            if not _wt.done():
                _wt.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await _wt

    # Exactly one created; the second is a pre-LLM drop pointing at the first.
    ids = {r['id'] for r in results}
    assert ids == {'100'}
    # Exactly one LLM call — the second never reached _call_llm because
    # the pre-LLM exact-match cache caught it.
    assert llm_calls == 1
    # Only one survivor in taskmaster.
    assert add_task_counter['n'] == 100


@pytest.mark.asyncio
async def test_note_created_is_called_inside_lock(curator_interceptor, taskmaster):
    """Sanity check: note_created fires on a real create so the next
    waiter's pre-LLM check can see it."""
    decision = CuratorDecision(action='create', justification='novel')
    curator_mock = _mock_curator(decision)
    curator_interceptor._curator = curator_mock

    await _submit_and_resolve(curator_interceptor, '/project', title='Fresh work')

    curator_mock.note_created.assert_called_once()
    args, _ = curator_mock.note_created.call_args
    assert args[0] == 'project'  # project_id
    assert args[2] == '2'  # task_id from taskmaster fixture


@pytest.mark.asyncio
async def test_pre_llm_exact_match_via_note_created(curator_enabled_config, taskmaster):
    """Directly exercise TaskCurator.note_created + _pre_llm_exact_match
    without going through the interceptor.
    """
    from fused_memory.middleware.task_curator import (
        CandidateTask,
        TaskCurator,
    )

    curator = TaskCurator(config=curator_enabled_config, taskmaster=taskmaster)

    candidate = CandidateTask(
        title='Add Type::Error arm',
        files_to_modify=['crates/reify-compiler/src/parser.rs'],
    )
    curator.note_created('proj', candidate, '1922')

    # get_task returns a pending task — match valid → drop.
    taskmaster.get_task = AsyncMock(
        return_value={'id': '1922', 'status': 'pending', 'title': 'x'},
    )
    decision = await curator._pre_llm_exact_match(
        candidate,
        project_id='proj',
        project_root='/x',
    )
    assert decision is not None
    assert decision.action == 'drop'
    assert decision.target_id == '1922'
    assert decision.justification == 'pre-llm-exact-match'

    # If the cached task is cancelled, pre_llm should fall through.
    taskmaster.get_task = AsyncMock(
        return_value={'id': '1922', 'status': 'cancelled', 'title': 'x'},
    )
    curator.note_created('proj', candidate, '1922')  # re-seed
    decision2 = await curator._pre_llm_exact_match(
        candidate,
        project_id='proj',
        project_root='/x',
    )
    assert decision2 is None


# ─────────────────────────────────────────────────────────────────────
# R4: escalation-level idempotency on (escalation_id, suggestion_hash)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_r4_idempotency_hit_add_task(
    taskmaster,
    reconciler,
    event_buffer,
    curator_enabled_config,
    tmp_path,
):
    """R4: idempotency hit short-circuits the add_task entry path.

    A stamped (escalation_id, suggestion_hash) pair must return the existing
    task without consulting the curator or creating a new task.
    """
    from fused_memory.middleware.ticket_store import TicketStore

    _seed_existing_r4_task(
        taskmaster,
        task_id='555',
        escalation_id='esc-r4-986',
        suggestion_hash='h986h986h986h986',
    )
    curator_mock = _mock_curator(CuratorDecision(action='create', justification='novel'))

    store = TicketStore(tmp_path / 'idemp_hit_add_task.db')
    await store.initialize()
    interceptor = TaskInterceptor(
        taskmaster,
        reconciler,
        event_buffer,
        config=curator_enabled_config,
        ticket_store=store,
    )
    interceptor._curator = curator_mock

    metadata = {
        'escalation_id': 'esc-r4-986',
        'suggestion_hash': 'h986h986h986h986',
        'files': ['fused-memory/src'],
    }
    try:
        result = await _submit_and_resolve(
            interceptor,
            # Use /dark-factory so the path-scope guard (which rejects fused-memory/
            # paths filed under non-dark-factory projects) does not block the ticket.
            '/dark-factory',
            title='T',
            description='D',
            metadata=metadata,
        )
    finally:
        await store.close()
        for _wt in list(interceptor._worker_tasks.values()):
            if not _wt.done():
                _wt.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await _wt

    assert result['id'] == '555'
    assert result['deduplicated'] is True
    assert result['action'] == 'idempotency_hit'
    _assert_r4_common(curator_mock, taskmaster)


@pytest.mark.asyncio
async def test_r4_idempotency_hit_submit_task(
    taskmaster,
    reconciler,
    event_buffer,
    curator_enabled_config,
    tmp_path,
):
    """R4: idempotency hit short-circuits the submit_task/resolve_ticket entry path.

    A stamped (escalation_id, suggestion_hash) pair must surface an
    idempotency_hit reason through the async ticket queue without consulting
    the curator or creating a new task.
    """
    from fused_memory.middleware.ticket_store import TicketStore

    _seed_existing_r4_task(
        taskmaster,
        task_id='555',
        escalation_id='esc-r4-986',
        suggestion_hash='h986h986h986h986',
    )
    curator_mock = _mock_curator(CuratorDecision(action='create', justification='novel'))

    store = TicketStore(tmp_path / 'idemp_hit_submit_task.db')
    await store.initialize()
    interceptor = TaskInterceptor(
        taskmaster,
        reconciler,
        event_buffer,
        config=curator_enabled_config,
        ticket_store=store,
    )
    interceptor._curator = curator_mock

    metadata = {
        'escalation_id': 'esc-r4-986',
        'suggestion_hash': 'h986h986h986h986',
        'files': ['fused-memory/src'],
    }
    try:
        submit_result = await interceptor.submit_task(
            # Use /dark-factory so the path-scope guard (which rejects fused-memory/
            # paths filed under non-dark-factory projects) does not block the ticket.
            '/dark-factory',
            title='T',
            description='D',
            metadata=metadata,
        )
        # Phase 2: resolve_ticket waits for the worker and returns the R4 decision.
        ticket = submit_result['ticket']
        result = await interceptor.resolve_ticket(ticket, '/dark-factory', timeout_seconds=5.0)
    finally:
        await store.close()
        for _wt in list(interceptor._worker_tasks.values()):
            if not _wt.done():
                _wt.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await _wt

    assert result.get('reason') != 'timeout', (
        f'Worker timed out before returning R4 decision — possible regression '
        f'where R4 gate is bypassed or worker stalled: {result!r}'
    )
    assert result.get('status') == 'combined', (
        f"resolve_ticket should return status='combined' on R4 hit, got {result!r}"
    )
    assert result.get('task_id') == '555', (
        f'resolve_ticket should return task_id of existing task, got {result!r}'
    )
    assert result.get('reason') == 'idempotency_hit', (
        f"resolve_ticket should return reason='idempotency_hit', got {result!r}"
    )
    _assert_r4_common(curator_mock, taskmaster)


@pytest.mark.asyncio
async def test_idempotency_accepts_metadata_as_json_string(
    taskmaster,
    reconciler,
    event_buffer,
    curator_enabled_config,
    tmp_path,
):
    """Metadata that arrives as a pre-serialised JSON string also dedupes."""
    import json

    from fused_memory.middleware.ticket_store import TicketStore

    taskmaster.get_tasks = AsyncMock(
        return_value={
            'tasks': [
                {
                    'id': '555',
                    'status': 'pending',
                    'title': 'T',
                    'metadata': {
                        'escalation_id': 'esc-x',
                        'suggestion_hash': 'hash1',
                    },
                },
            ],
        }
    )

    store = TicketStore(tmp_path / 'idemp_str_tickets.db')
    await store.initialize()
    interceptor = TaskInterceptor(
        taskmaster,
        reconciler,
        event_buffer,
        config=curator_enabled_config,
        ticket_store=store,
    )
    interceptor._curator = _mock_curator(CuratorDecision(action='create'))

    try:
        meta_str = json.dumps({'escalation_id': 'esc-x', 'suggestion_hash': 'hash1'})
        result = await _submit_and_resolve(interceptor, '/project', title='T', metadata=meta_str)
    finally:
        await store.close()
        for _wt in list(interceptor._worker_tasks.values()):
            if not _wt.done():
                _wt.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await _wt
    assert result['id'] == '555'
    assert result['action'] == 'idempotency_hit'


@pytest.mark.asyncio
async def test_idempotency_miss_falls_through_to_curator(
    taskmaster,
    reconciler,
    event_buffer,
    curator_enabled_config,
    tmp_path,
):
    """No matching (escalation_id, suggestion_hash) → curator runs normally."""
    from fused_memory.middleware.ticket_store import TicketStore

    taskmaster.get_tasks = AsyncMock(
        return_value={
            'tasks': [
                {
                    'id': '500',
                    'status': 'pending',
                    'title': 'Unrelated',
                    'metadata': {
                        'escalation_id': 'esc-zzz',
                        'suggestion_hash': 'different',
                    },
                },
            ],
        }
    )

    curator_mock = _mock_curator(CuratorDecision(action='create', justification='novel'))
    store = TicketStore(tmp_path / 'idemp_miss_tickets.db')
    await store.initialize()
    interceptor = TaskInterceptor(
        taskmaster,
        reconciler,
        event_buffer,
        config=curator_enabled_config,
        ticket_store=store,
    )
    interceptor._curator = curator_mock

    try:
        await _submit_and_resolve(
            interceptor,
            '/project',
            title='New',
            metadata={'escalation_id': 'esc-new', 'suggestion_hash': 'fresh'},
        )
    finally:
        await store.close()
        for _wt in list(interceptor._worker_tasks.values()):
            if not _wt.done():
                _wt.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await _wt
    curator_mock.curate.assert_called_once()
    taskmaster.add_task.assert_called_once()


@pytest.mark.asyncio
async def test_idempotency_skips_cancelled_match(
    taskmaster,
    reconciler,
    event_buffer,
    curator_enabled_config,
    tmp_path,
):
    """A cancelled task with matching metadata must not win the dedupe."""
    from fused_memory.middleware.ticket_store import TicketStore

    taskmaster.get_tasks = AsyncMock(
        return_value={
            'tasks': [
                {
                    'id': '500',
                    'status': 'cancelled',
                    'title': 'Was the dupe',
                    'metadata': {
                        'escalation_id': 'esc-y',
                        'suggestion_hash': 'hash-y',
                    },
                },
            ],
        }
    )

    curator_mock = _mock_curator(CuratorDecision(action='create', justification='novel'))
    store = TicketStore(tmp_path / 'idemp_cancel_tickets.db')
    await store.initialize()
    interceptor = TaskInterceptor(
        taskmaster,
        reconciler,
        event_buffer,
        config=curator_enabled_config,
        ticket_store=store,
    )
    interceptor._curator = curator_mock

    try:
        await _submit_and_resolve(
            interceptor,
            '/project',
            title='Retry',
            metadata={'escalation_id': 'esc-y', 'suggestion_hash': 'hash-y'},
        )
    finally:
        await store.close()
        for _wt in list(interceptor._worker_tasks.values()):
            if not _wt.done():
                _wt.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await _wt
    curator_mock.curate.assert_called_once()


@pytest.mark.asyncio
async def test_idempotency_requires_both_keys(
    taskmaster,
    reconciler,
    event_buffer,
    curator_enabled_config,
    tmp_path,
):
    """Metadata without escalation_id+suggestion_hash skips the R4 check."""
    from fused_memory.middleware.ticket_store import TicketStore

    curator_mock = _mock_curator(CuratorDecision(action='create', justification='novel'))
    store = TicketStore(tmp_path / 'idemp_both_tickets.db')
    await store.initialize()
    interceptor = TaskInterceptor(
        taskmaster,
        reconciler,
        event_buffer,
        config=curator_enabled_config,
        ticket_store=store,
    )
    interceptor._curator = curator_mock

    try:
        # Only escalation_id, no suggestion_hash → not eligible.
        await _submit_and_resolve(
            interceptor,
            '/project',
            title='T',
            metadata={'escalation_id': 'esc-x'},
        )
    finally:
        await store.close()
        for _wt in list(interceptor._worker_tasks.values()):
            if not _wt.done():
                _wt.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await _wt
    curator_mock.curate.assert_called_once()
    # get_tasks for the idempotency check should not have been invoked
    # because we bail before the walk when a key is missing.
    # (get_tasks may still be called by curator _build_corpus under some
    # paths — our curator_mock stubs that; so the AsyncMock
    # ``taskmaster.get_tasks`` call count must be zero.)
    assert taskmaster.get_tasks.call_count == 0


@pytest.mark.asyncio
async def test_curator_disabled_still_proxies(taskmaster, reconciler, event_buffer, tmp_path):
    """With curator.enabled=False, add_task proxies straight to Taskmaster."""
    from fused_memory.middleware.ticket_store import TicketStore

    cfg = FusedMemoryConfig()
    cfg.curator = CuratorConfig(enabled=False)
    store = TicketStore(tmp_path / 'disabled_curator_tickets.db')
    await store.initialize()
    interceptor = TaskInterceptor(
        taskmaster, reconciler, event_buffer, config=cfg, ticket_store=store
    )

    try:
        result = await _submit_and_resolve(interceptor, '/project', title='T')
    finally:
        await store.close()
        for _wt in list(interceptor._worker_tasks.values()):
            if not _wt.done():
                _wt.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await _wt

    assert result == {'id': '2', 'title': 'New Task'}
    taskmaster.add_task.assert_called_once()


@pytest.mark.asyncio
async def test_update_task_reembeds_on_title_change(
    curator_interceptor,
    taskmaster,
):
    """update_task triggers fire-and-forget reembed when title/details change."""
    curator_mock = _mock_curator(CuratorDecision(action='create'))
    curator_interceptor._curator = curator_mock
    taskmaster.get_task.return_value = {
        'id': '7',
        'status': 'pending',
        'title': 'Updated title',
        'description': 'desc',
        'details': 'details',
    }

    await curator_interceptor.update_task(
        '7',
        '/project',
        prompt='rename title to updated',
    )
    await asyncio.sleep(0)  # let fire-and-forget run
    await curator_interceptor.drain()

    curator_mock.reembed_task.assert_called_once()


@pytest.mark.asyncio
async def test_no_reconciler_still_proxies(taskmaster, event_buffer):
    """Without a reconciler, interceptor still proxies to taskmaster."""
    interceptor = TaskInterceptor(taskmaster, None, event_buffer)
    result = await interceptor.set_task_status('1', 'done', '/project')
    assert result == {'success': True}
    # No reconciliation key
    assert 'reconciliation' not in result


@pytest.mark.asyncio
async def test_remove_tasks_emits_event_per_id(interceptor, event_buffer):
    await interceptor.remove_tasks(['1'], '/project')
    stats = await event_buffer.get_buffer_stats('project')
    assert stats['size'] == 1


@pytest.mark.asyncio
async def test_remove_tasks_multi_id_emits_one_event_per_id(
    interceptor,
    event_buffer,
):
    await interceptor.remove_tasks(['1', '2', '3'], '/project')
    stats = await event_buffer.get_buffer_stats('project')
    assert stats['size'] == 3


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
        reopen_reason=None,
    )


@pytest.mark.asyncio
async def test_event_roundtrip_preserves_both_ids(taskmaster, event_buffer, tmp_path):
    """End-to-end: interceptor -> buffer -> drain preserves both project_id and _project_root."""
    from fused_memory.middleware.ticket_store import TicketStore

    store = TicketStore(tmp_path / 'roundtrip_tickets.db')
    await store.initialize()
    interceptor = TaskInterceptor(taskmaster, None, event_buffer, ticket_store=store)
    project_path = '/home/leo/src/dark-factory'

    try:
        # Multiple operations
        await interceptor.set_task_status('1', 'in-progress', project_path)
        await _submit_and_resolve(interceptor, project_path, prompt='New task')
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
    finally:
        await store.close()
        for _wt in list(interceptor._worker_tasks.values()):
            if not _wt.done():
                _wt.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await _wt


# ── Tests for get_statuses ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_statuses_returns_all_id_to_status_mapping(taskmaster, event_buffer):
    """get_statuses returns {id_str: status_str} for every task; no events emitted."""
    taskmaster.get_statuses_raw = AsyncMock(
        return_value={'1': 'pending', '2': 'done', '3': 'in-progress'}
    )
    # Spy on the canonical add path (EventBuffer.push at event_buffer.py:201).
    # AsyncMock(wraps=...) preserves real behaviour while recording calls, so a
    # rogue push attempt would still flow through to the buffer (caught by the
    # belt-and-suspenders stats['size']==0 check below) AND show up here.
    event_buffer.push = AsyncMock(wraps=event_buffer.push)
    interceptor = TaskInterceptor(taskmaster, None, event_buffer)

    result = await interceptor.get_statuses('/project')

    # Primary contract: pure read — no event-emit path is invoked.
    event_buffer.push.assert_not_called()

    assert result == {'1': 'pending', '2': 'done', '3': 'in-progress'}

    # Belt-and-suspenders: even if a future refactor bypasses push() and
    # writes directly to the underlying store, the buffer remains empty.
    stats = await event_buffer.get_buffer_stats('project')
    assert stats['size'] == 0


@pytest.mark.asyncio
async def test_get_statuses_filters_by_ids_list(taskmaster, event_buffer):
    """When ids=['1', '3'], only those two keys appear in the result.

    The backend now owns the filtering; the interceptor is a thin delegator.
    """
    taskmaster.get_statuses_raw = AsyncMock(
        return_value={'1': 'pending', '3': 'in-progress'}
    )
    interceptor = TaskInterceptor(taskmaster, None, event_buffer)

    result = await interceptor.get_statuses('/project', ids=['1', '3'])

    assert result == {'1': 'pending', '3': 'in-progress'}
    taskmaster.get_statuses_raw.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_statuses_omits_unknown_ids(taskmaster, event_buffer):
    """Unknown ids in the filter list are silently omitted (no error, no key).

    The backend now owns the omission; the interceptor delegates verbatim.
    """
    taskmaster.get_statuses_raw = AsyncMock(return_value={'1': 'pending'})
    interceptor = TaskInterceptor(taskmaster, None, event_buffer)

    result = await interceptor.get_statuses('/project', ids=['1', '9999'])

    assert result == {'1': 'pending'}
    assert '9999' not in result


@pytest.mark.asyncio
async def test_get_statuses_raises_when_taskmaster_not_configured(event_buffer):
    """TaskInterceptor(None, None, buf) → get_statuses() raises RuntimeError."""
    interceptor = TaskInterceptor(None, None, event_buffer)
    with pytest.raises(RuntimeError, match='not configured'):
        await interceptor.get_statuses('/project')


@pytest.mark.asyncio
async def test_get_statuses_calls_ensure_connected(event_buffer):
    """ensure_connected is called before proxying to taskmaster in get_statuses."""
    tm = AsyncMock()
    tm.ensure_connected = AsyncMock()
    tm.get_statuses_raw = AsyncMock(return_value={})
    interceptor = TaskInterceptor(tm, None, event_buffer)

    await interceptor.get_statuses('/project')
    tm.ensure_connected.assert_called_once()


@pytest.mark.asyncio
async def test_get_statuses_missing_status_key_defaults_to_unknown(taskmaster, event_buffer):
    """A task dict without a 'status' key is included with status='unknown'.

    Contract: the sentinel 'unknown' is the documented default when the raw
    task dict omits 'status'.  Callers that need to distinguish a genuine
    'unknown' status from a missing field should treat any 'unknown' as
    indeterminate.
    """
    # The backend now owns the None->'unknown' coercion; the interceptor
    # delegates verbatim.  Mock get_statuses_raw to return the already-coerced
    # mapping so the interceptor's passthrough contract is verified.
    taskmaster.get_statuses_raw = AsyncMock(
        return_value={'1': 'unknown', '2': 'done'}
    )
    interceptor = TaskInterceptor(taskmaster, None, event_buffer)

    result = await interceptor.get_statuses('/project')

    assert result == {'1': 'unknown', '2': 'done'}


@pytest.mark.asyncio
async def test_get_statuses_routes_through_get_statuses_raw(event_buffer):
    """interceptor.get_statuses delegates to tm.get_statuses_raw, not tm.get_tasks.

    Proves the O(K) routing fix: get_statuses_raw is called once with the
    correct arguments; get_tasks is never called; result is verbatim passthrough;
    no events are emitted.
    """
    tm = AsyncMock()
    tm.ensure_connected = AsyncMock()
    tm.get_statuses_raw = AsyncMock(return_value={'1': 'pending'})
    tm.get_tasks = AsyncMock(return_value={'tasks': []})

    event_buffer.push = AsyncMock(wraps=event_buffer.push)
    interceptor = TaskInterceptor(tm, None, event_buffer)

    result = await interceptor.get_statuses('/project', ids=['1'], tag='master')

    # Verbatim passthrough from get_statuses_raw.
    assert result == {'1': 'pending'}

    # Routing: get_statuses_raw called; get_tasks NOT called.
    tm.get_statuses_raw.assert_awaited_once()
    call_kwargs = tm.get_statuses_raw.call_args
    # project_root, ids and tag must be forwarded.
    assert call_kwargs.args[0] == '/project' or call_kwargs.kwargs.get('project_root') == '/project'
    tm.get_tasks.assert_not_called()

    # Pure read: no events.
    event_buffer.push.assert_not_called()
    stats = await event_buffer.get_buffer_stats('project')
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
async def test_set_task_status_allows_done_to_blocked_with_reopen_reason(
    taskmaster,
    reconciler,
    event_buffer,
):
    """done->blocked is allowed when an explicit reopen_reason is passed."""
    taskmaster.get_task = AsyncMock(return_value={'id': '1', 'status': 'done', 'title': 'T'})
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status(
        '1',
        'blocked',
        '/project',
        reopen_reason='manual re-scope',
    )

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
    taskmaster.get_task = AsyncMock(return_value={'id': '1', 'status': 'in-progress', 'title': 'T'})
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
    taskmaster.get_task = AsyncMock(return_value={'id': '2', 'status': 'cancelled', 'title': 'T'})
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


# ── Tests for phantom-done gate (metadata.files existence check) ───────────


@pytest.mark.asyncio
async def test_done_gate_rejects_when_declared_files_missing(
    taskmaster, reconciler, event_buffer, tmp_path
):
    """status=done is refused if metadata.files lists a file that doesn't exist."""
    taskmaster.get_task = AsyncMock(
        return_value={
            'id': '1746',
            'status': 'in-progress',
            'title': 'Named views',
            'metadata': {'files': ['gui/src/panels/ViewSelector.tsx']},
        }
    )
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status('1746', 'done', str(tmp_path))

    assert result['success'] is False
    assert result['error'] == 'done_gate_missing_files'
    assert result['missing_files'] == ['gui/src/panels/ViewSelector.tsx']
    assert result['task_id'] == '1746'
    # Taskmaster write must not have fired
    taskmaster.set_task_status.assert_not_called()
    # No event, no reconciliation
    stats = await event_buffer.get_buffer_stats('project')
    assert stats['size'] == 0
    reconciler.reconcile_task.assert_not_called()


@pytest.mark.asyncio
async def test_done_gate_passes_when_files_exist(taskmaster, reconciler, event_buffer, tmp_path):
    """status=done succeeds when every declared file exists under project_root."""
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'mod.rs').write_text('// shipped')
    taskmaster.get_task = AsyncMock(
        return_value={
            'id': '42',
            'status': 'in-progress',
            'title': 'Legit task',
            'metadata': {'files': ['src/mod.rs']},
        }
    )
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status('42', 'done', str(tmp_path))

    assert 'error' not in result
    taskmaster.set_task_status.assert_called_once()


@pytest.mark.asyncio
async def test_done_gate_noop_without_metadata_files(taskmaster, reconciler, event_buffer):
    """Gate does not fire when metadata.files is absent — back-compat for legacy tasks."""
    # default taskmaster fixture returns {'id':'1','status':'pending','title':'Test Task'} — no metadata
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status('1', 'done', '/project')

    assert 'error' not in result
    taskmaster.set_task_status.assert_called_once()


@pytest.mark.asyncio
async def test_done_gate_reports_partial_missing(taskmaster, reconciler, event_buffer, tmp_path):
    """When some declared files exist and others don't, only the missing ones are reported."""
    (tmp_path / 'exists.rs').write_text('')
    taskmaster.get_task = AsyncMock(
        return_value={
            'id': '99',
            'status': 'in-progress',
            'title': 'Partial',
            'metadata': {'files': ['exists.rs', 'missing.rs', 'also_missing.ts']},
        }
    )
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status('99', 'done', str(tmp_path))

    assert result['success'] is False
    assert sorted(result['missing_files']) == ['also_missing.ts', 'missing.rs']
    assert sorted(result['files_checked']) == sorted(['exists.rs', 'missing.rs', 'also_missing.ts'])
    taskmaster.set_task_status.assert_not_called()


@pytest.mark.asyncio
async def test_done_gate_skipped_when_verified_provenance_supplied(
    taskmaster, reconciler, event_buffer, tmp_path
):
    """Phantom-done gate is bypassed when done_provenance.kind ∈
    {'merged', 'found_on_main'} AND its commit is ancestor-checked.

    The architect's plan can include files that get squashed/refactored
    away before merge; the ancestor-checked commit vouches the work is on
    main, so the gate's defense-in-depth role no longer applies.
    """
    sha = _init_git_repo(tmp_path)
    taskmaster.get_task = AsyncMock(
        return_value={
            'id': '1746',
            'status': 'in-progress',
            'title': 'Named views',
            # metadata.files lists a path that's been refactored away
            'metadata': {'files': ['gui/src/panels/ViewSelector.tsx']},
        }
    )
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status(
        '1746',
        'done',
        str(tmp_path),
        done_provenance={'kind': 'merged', 'commit': sha},
    )

    # Gate skipped — no done_gate_missing_files.
    assert result.get('error') != 'done_gate_missing_files'
    # Transition succeeded.
    taskmaster.set_task_status.assert_called_once()


@pytest.mark.asyncio
async def test_done_gate_still_fires_without_provenance(
    taskmaster, reconciler, event_buffer, tmp_path
):
    """Defense for non-workflow callers: when done_provenance is absent,
    the phantom-done gate still rejects missing-files transitions.
    """
    taskmaster.get_task = AsyncMock(
        return_value={
            'id': '1747',
            'status': 'in-progress',
            'metadata': {'files': ['does_not_exist.py']},
        }
    )
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status(
        '1747',
        'done',
        str(tmp_path),
    )

    assert result['success'] is False
    assert result['error'] == 'done_gate_missing_files'
    taskmaster.set_task_status.assert_not_called()


@pytest.mark.asyncio
async def test_done_gate_does_not_fire_for_non_done_transitions(
    taskmaster, reconciler, event_buffer
):
    """blocked/cancelled/deferred transitions bypass the file-existence gate."""
    taskmaster.get_task = AsyncMock(
        return_value={
            'id': '5',
            'status': 'in-progress',
            'metadata': {'files': ['does_not_exist.rs']},
        }
    )
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    for status in ('blocked', 'cancelled', 'deferred'):
        taskmaster.set_task_status.reset_mock()
        result = await interceptor.set_task_status('5', status, '/project')
        assert 'error' not in result, f'gate should not fire on {status}'
        taskmaster.set_task_status.assert_called_once()


# ── Tests for done_provenance gate ─────────────────────────────────────────


def _init_git_repo(path) -> str:
    """Create a minimal git repo at path with one commit; return full SHA."""
    import subprocess

    subprocess.run(['git', 'init', '-q', '-b', 'main', str(path)], check=True)
    subprocess.run(
        ['git', '-C', str(path), 'config', 'user.email', 't@e.example'],
        check=True,
    )
    subprocess.run(
        ['git', '-C', str(path), 'config', 'user.name', 'T'],
        check=True,
    )
    (path / 'seed.txt').write_text('seed\n')
    subprocess.run(['git', '-C', str(path), 'add', '-A'], check=True)
    subprocess.run(
        ['git', '-C', str(path), 'commit', '-q', '-m', 'seed'],
        check=True,
    )
    return subprocess.run(
        ['git', '-C', str(path), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def config_with_strict_provenance():
    """FusedMemoryConfig with require_done_provenance=True."""
    cfg = FusedMemoryConfig()
    cfg.reconciliation.require_done_provenance = True
    return cfg


@pytest.mark.asyncio
async def test_done_provenance_warn_only_when_missing_by_default(
    taskmaster, reconciler, event_buffer
):
    """Without require_done_provenance, a missing payload logs a warning but proceeds."""
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status('1', 'done', '/project')

    assert 'error' not in result
    taskmaster.set_task_status.assert_called_once()
    # metadata was not touched because no provenance was provided
    taskmaster.update_task.assert_not_called()


@pytest.mark.asyncio
async def test_done_provenance_rejects_missing_when_required(
    taskmaster, reconciler, event_buffer, config_with_strict_provenance
):
    """With the gate enabled, a missing payload is rejected with a structured error."""
    interceptor = TaskInterceptor(
        taskmaster,
        reconciler,
        event_buffer,
        config=config_with_strict_provenance,
    )

    result = await interceptor.set_task_status('1', 'done', '/project')

    assert result['success'] is False
    assert result['error'] == 'done_provenance_required'
    taskmaster.set_task_status.assert_not_called()


@pytest.mark.asyncio
async def test_done_provenance_rejects_empty_payload(
    taskmaster, reconciler, event_buffer, config_with_strict_provenance
):
    """An object with empty commit AND empty note is invalid even with gate on."""
    interceptor = TaskInterceptor(
        taskmaster,
        reconciler,
        event_buffer,
        config=config_with_strict_provenance,
    )

    result = await interceptor.set_task_status(
        '1',
        'done',
        '/project',
        done_provenance={'commit': '', 'note': ''},
    )

    assert result['success'] is False
    assert result['error'] == 'done_provenance_invalid'
    taskmaster.set_task_status.assert_not_called()


@pytest.mark.asyncio
async def test_done_provenance_rejects_invalid_commit_ref(
    taskmaster, reconciler, event_buffer, tmp_path
):
    """A commit that can't be resolved by git rev-parse errors regardless of gate."""
    _init_git_repo(tmp_path)
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status(
        '1',
        'done',
        str(tmp_path),
        done_provenance={
            'kind': 'merged',
            'commit': 'deadbeefdeadbeefdeadbeefdeadbeefdeadbeef',
        },
    )

    assert result['success'] is False
    assert result['error'] == 'done_provenance_invalid'
    taskmaster.set_task_status.assert_not_called()


@pytest.mark.asyncio
async def test_done_provenance_resolves_short_sha_and_persists(
    taskmaster, reconciler, event_buffer, tmp_path
):
    """A short SHA is resolved to full SHA and persisted via update_task metadata."""
    sha = _init_git_repo(tmp_path)
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status(
        '1',
        'done',
        str(tmp_path),
        done_provenance={'kind': 'merged', 'commit': sha[:7]},
    )

    assert 'error' not in result
    taskmaster.update_task.assert_called_once()
    kwargs = taskmaster.update_task.call_args.kwargs
    persisted = json.loads(kwargs['metadata'])
    assert persisted['done_provenance']['kind'] == 'merged'
    assert persisted['done_provenance']['commit'] == sha
    assert persisted['done_provenance']['commit_input'] == sha[:7]


@pytest.mark.asyncio
async def test_done_provenance_commit_plus_note_both_persisted(
    taskmaster, reconciler, event_buffer, tmp_path
):
    """Both commit and note may be provided; both are recorded."""
    sha = _init_git_repo(tmp_path)
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status(
        '1',
        'done',
        str(tmp_path),
        done_provenance={
            'kind': 'merged',
            'commit': sha,
            'note': 'ff-merged after review',
        },
    )

    assert 'error' not in result
    persisted = json.loads(taskmaster.update_task.call_args.kwargs['metadata'])
    assert persisted['done_provenance']['kind'] == 'merged'
    assert persisted['done_provenance']['commit'] == sha
    assert persisted['done_provenance']['note'] == 'ff-merged after review'
    # No commit_input when the full SHA was supplied
    assert 'commit_input' not in persisted['done_provenance']


@pytest.mark.asyncio
async def test_done_provenance_reopen_does_not_require_provenance(
    taskmaster, reconciler, event_buffer, config_with_strict_provenance
):
    """Transitioning out of done (e.g. done → in-progress) bypasses the
    done_provenance gate but still requires reopen_reason to pass the
    terminal-exit gate.
    """
    taskmaster.get_task = AsyncMock(
        return_value={'id': '1', 'status': 'done', 'title': 'T'},
    )
    interceptor = TaskInterceptor(
        taskmaster,
        reconciler,
        event_buffer,
        config=config_with_strict_provenance,
    )

    result = await interceptor.set_task_status(
        '1',
        'in-progress',
        '/project',
        reopen_reason='resuming after investigation',
    )

    assert 'error' not in result, result
    taskmaster.set_task_status.assert_called_once()


@pytest.mark.asyncio
async def test_done_provenance_malformed_shape_errors_even_warn_only(
    taskmaster, reconciler, event_buffer
):
    """Wrong type (list instead of dict) always errors — never persists corrupt data."""
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status(
        '1',
        'done',
        '/project',
        done_provenance=['not', 'a', 'dict'],  # type: ignore[arg-type]
    )

    assert result['success'] is False
    assert result['error'] == 'done_provenance_invalid'


@pytest.mark.asyncio
async def test_done_provenance_included_in_event_payload(
    taskmaster, reconciler, event_buffer, tmp_path
):
    """The task_status_changed event carries resolved provenance for downstream recon."""
    sha = _init_git_repo(tmp_path)
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    await interceptor.set_task_status(
        '1',
        'done',
        str(tmp_path),
        done_provenance={'kind': 'merged', 'commit': sha},
    )

    project_id = resolve_project_id(str(tmp_path))
    events = await event_buffer.peek_buffered(project_id, limit=10)
    assert events, 'event should be buffered'
    payload = events[-1].payload
    assert payload['done_provenance']['kind'] == 'merged'
    assert payload['done_provenance']['commit'] == sha


# ── Tests for done_provenance.kind discriminator (2026-04-27 hardening) ────


@pytest.mark.asyncio
async def test_done_provenance_rejects_missing_kind(taskmaster, reconciler, event_buffer, tmp_path):
    """A payload without `kind` is rejected with a helpful pointer."""
    sha = _init_git_repo(tmp_path)
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status(
        '1',
        'done',
        str(tmp_path),
        done_provenance={'commit': sha},
    )

    assert result['success'] is False
    assert result['error'] == 'done_provenance_invalid'
    assert 'kind' in result['reason'].lower()
    taskmaster.set_task_status.assert_not_called()


@pytest.mark.asyncio
async def test_done_provenance_rejects_unknown_kind(taskmaster, reconciler, event_buffer, tmp_path):
    """Only `merged` and `found_on_main` are accepted as kinds."""
    sha = _init_git_repo(tmp_path)
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status(
        '1',
        'done',
        str(tmp_path),
        done_provenance={'kind': 'cherry_picked', 'commit': sha},
    )

    assert result['success'] is False
    assert result['error'] == 'done_provenance_invalid'
    assert 'cherry_picked' in result['reason']
    taskmaster.set_task_status.assert_not_called()


@pytest.mark.asyncio
async def test_done_provenance_merged_requires_commit(taskmaster, reconciler, event_buffer):
    """kind="merged" without a commit is rejected even with a note."""
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status(
        '1',
        'done',
        '/project',
        done_provenance={'kind': 'merged', 'note': 'merged via the queue'},
    )

    assert result['success'] is False
    assert result['error'] == 'done_provenance_invalid'
    assert 'commit' in result['reason']
    taskmaster.set_task_status.assert_not_called()


@pytest.mark.asyncio
async def test_done_provenance_found_on_main_requires_note(
    taskmaster, reconciler, event_buffer, tmp_path
):
    """kind="found_on_main" without a note is rejected even with a commit."""
    sha = _init_git_repo(tmp_path)
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status(
        '1',
        'done',
        str(tmp_path),
        done_provenance={'kind': 'found_on_main', 'commit': sha},
    )

    assert result['success'] is False
    assert result['error'] == 'done_provenance_invalid'
    assert 'note' in result['reason']
    taskmaster.set_task_status.assert_not_called()


@pytest.mark.asyncio
async def test_done_provenance_found_on_main_requires_commit(taskmaster, reconciler, event_buffer):
    """kind='found_on_main' without a commit is rejected (post-3092 hardening)."""
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status(
        '1',
        'done',
        '/project',
        done_provenance={
            'kind': 'found_on_main',
            'note': 'covered by parent task 1745',
        },
    )

    assert result['success'] is False
    assert result['error'] == 'done_provenance_invalid'
    assert 'commit' in result['reason'].lower()
    taskmaster.set_task_status.assert_not_called()


@pytest.mark.asyncio
async def test_done_provenance_merged_rejects_branch_only_sha(
    taskmaster, reconciler, event_buffer, tmp_path
):
    """kind="merged" with a SHA that exists but is not on main is rejected.

    This is the key backstop: the steward MUST not be able to record a
    branch-only SHA as a merge commit. ``git merge-base --is-ancestor`` is
    the source of truth.
    """
    import subprocess

    _init_git_repo(tmp_path)
    # Create a branch commit that is NOT on main
    subprocess.run(
        ['git', '-C', str(tmp_path), 'checkout', '-q', '-b', 'feature'],
        check=True,
    )
    (tmp_path / 'feature.txt').write_text('feature\n')
    subprocess.run(['git', '-C', str(tmp_path), 'add', '-A'], check=True)
    subprocess.run(
        ['git', '-C', str(tmp_path), 'commit', '-q', '-m', 'feature commit'],
        check=True,
    )
    branch_sha = subprocess.run(
        ['git', '-C', str(tmp_path), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    # Switch back to main so HEAD on the worktree is on main
    subprocess.run(
        ['git', '-C', str(tmp_path), 'checkout', '-q', 'main'],
        check=True,
    )

    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)
    result = await interceptor.set_task_status(
        '1',
        'done',
        str(tmp_path),
        done_provenance={'kind': 'merged', 'commit': branch_sha},
    )

    assert result['success'] is False
    assert result['error'] == 'done_provenance_invalid'
    assert 'not on main' in result['reason']
    taskmaster.set_task_status.assert_not_called()


@pytest.mark.asyncio
async def test_done_provenance_found_on_main_rejects_branch_only_sha(
    taskmaster, reconciler, event_buffer, tmp_path
):
    """kind='found_on_main' with a commit not on main is rejected (post-3092 hardening).

    A branch-only SHA is rejected the same way as kind='merged'.
    """
    import subprocess

    _init_git_repo(tmp_path)
    # Create a branch commit that is NOT on main
    subprocess.run(
        ['git', '-C', str(tmp_path), 'checkout', '-q', '-b', 'feature'],
        check=True,
    )
    (tmp_path / 'feature.txt').write_text('feature\n')
    subprocess.run(['git', '-C', str(tmp_path), 'add', '-A'], check=True)
    subprocess.run(
        ['git', '-C', str(tmp_path), 'commit', '-q', '-m', 'feature commit'],
        check=True,
    )
    branch_sha = subprocess.run(
        ['git', '-C', str(tmp_path), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    # Switch back to main so HEAD on the worktree is on main
    subprocess.run(
        ['git', '-C', str(tmp_path), 'checkout', '-q', 'main'],
        check=True,
    )

    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)
    result = await interceptor.set_task_status(
        '1',
        'done',
        str(tmp_path),
        done_provenance={
            'kind': 'found_on_main',
            'commit': branch_sha,
            'note': 'sibling task 99 landed this',
        },
    )

    assert result['success'] is False
    assert result['error'] == 'done_provenance_invalid'
    assert 'not on main' in result['reason']
    taskmaster.set_task_status.assert_not_called()


@pytest.mark.asyncio
async def test_done_provenance_found_on_main_with_on_main_commit_passes(
    taskmaster, reconciler, event_buffer, tmp_path
):
    """kind='found_on_main' with commit+note where commit is on main is accepted."""
    sha = _init_git_repo(tmp_path)
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status(
        '1',
        'done',
        str(tmp_path),
        done_provenance={
            'kind': 'found_on_main',
            'commit': sha,
            'note': 'sibling task 99 landed this on main',
        },
    )

    assert 'error' not in result
    taskmaster.update_task.assert_called_once()
    persisted = json.loads(taskmaster.update_task.call_args.kwargs['metadata'])
    dp = persisted['done_provenance']
    assert dp['kind'] == 'found_on_main'
    assert dp['commit'] == sha
    assert dp['note'] == 'sibling task 99 landed this on main'
    # No commit_input when the full SHA was supplied
    assert 'commit_input' not in dp


@pytest.mark.asyncio
async def test_done_provenance_found_on_main_short_sha_resolved(
    taskmaster, reconciler, event_buffer, tmp_path
):
    """kind='found_on_main' with a short-SHA prefix resolves to the full 40-char SHA."""
    sha = _init_git_repo(tmp_path)
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status(
        '1',
        'done',
        str(tmp_path),
        done_provenance={
            'kind': 'found_on_main',
            'commit': sha[:7],
            'note': 'sibling task 99 landed this on main',
        },
    )

    assert 'error' not in result
    persisted = json.loads(taskmaster.update_task.call_args.kwargs['metadata'])
    dp = persisted['done_provenance']
    assert dp['commit'] == sha  # resolved to full SHA
    assert dp['commit_input'] == sha[:7]  # original short ref preserved


@pytest.mark.asyncio
async def test_update_task_rejects_metadata_done_provenance(taskmaster, reconciler, event_buffer):
    """update_task must NOT be a side door for writing done_provenance.

    The 2026-04-27 incident had a workflow agent stamp self-contradicting
    provenance via update_task; the role allowlist is layer 2, this schema
    block is layer 1.
    """
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.update_task(
        '1',
        '/project',
        metadata=json.dumps(
            {'done_provenance': {'kind': 'merged', 'commit': 'abc123'}},
        ),
    )

    assert result['success'] is False
    assert result['error'] == 'done_provenance_via_update_task'
    taskmaster.update_task.assert_not_called()


@pytest.mark.asyncio
async def test_update_task_allows_other_metadata(taskmaster, reconciler, event_buffer):
    """The done_provenance block does not affect other metadata writes."""
    taskmaster.update_task.return_value = {'success': True}
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.update_task(
        '1',
        '/project',
        metadata=json.dumps({'files': ['orchestrator/']}),
    )

    assert 'error' not in result
    taskmaster.update_task.assert_called_once()


# ── Tests for background task retention (step-3) ───────────────────────────


def test_background_tasks_set_exists(taskmaster, reconciler, event_buffer):
    """TaskInterceptor should have a _background_tasks set after init."""
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)
    assert hasattr(interceptor, '_background_tasks')
    assert isinstance(interceptor._background_tasks, set)


@pytest.mark.asyncio
async def test_background_tasks_retained_during_reconciliation(
    taskmaster, reconciler, event_buffer
):
    """Background task should be in _background_tasks while running, removed after completion."""
    # Use a future to control when reconcile_task finishes
    started = asyncio.Event()
    done_future: asyncio.Future = asyncio.Future()

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


# Auto-commit of tasks.json was retired post-SQLite-cutover (2026-05-06).
# The TaskInterceptor no longer fans out to a TaskFileCommitter; SQLite is
# the durable store and there is no sidecar JSON snapshot to maintain.


# ─────────────────────────────────────────────────────────────────────
# WP-E: per-project serialisation of mutating taskmaster calls
# ─────────────────────────────────────────────────────────────────────


class _OverlapTracker:
    """Records peak in-flight concurrent entries to instrumented calls.

    Wrap a mock's side_effect with ``tracker.wrap(project, return_value)``
    to probe whether the interceptor's per-project lock serialises
    mutations. If the lock is effective, per-project peak is 1. Across
    distinct projects, peak can exceed 1 (the lock is per-project).
    """

    def __init__(self) -> None:
        self.in_flight: dict[str, int] = {}
        self.peak: dict[str, int] = {}
        self.total_peak = 0
        self._global_in_flight = 0

    def wrap(self, project_key: str, return_value):
        async def _side_effect(*args, **kwargs):
            self.in_flight[project_key] = self.in_flight.get(project_key, 0) + 1
            self._global_in_flight += 1
            self.peak[project_key] = max(
                self.peak.get(project_key, 0),
                self.in_flight[project_key],
            )
            self.total_peak = max(self.total_peak, self._global_in_flight)
            try:
                # Yield to the loop so concurrent tasks really do interleave
                # — a zero-sleep await is enough to surface lock violations.
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                return return_value(*args, **kwargs) if callable(return_value) else return_value
            finally:
                self.in_flight[project_key] -= 1
                self._global_in_flight -= 1

        return _side_effect


@pytest.fixture
def overlap_tm():
    """Taskmaster mock whose mutating methods all yield to the event loop.

    Shared across the WP-E concurrency tests. Each test points the
    ``side_effect`` at its own _OverlapTracker.
    """
    tm = AsyncMock()
    tm.get_task = AsyncMock(
        return_value={'id': '1', 'status': 'pending', 'title': 'T'},
    )
    tm.get_tasks = AsyncMock(return_value={'tasks': []})
    return tm


@pytest.mark.asyncio
async def test_concurrent_add_task_burst_all_distinct(
    overlap_tm,
    reconciler,
    event_buffer,
    tmp_path,
):
    """WP-E: 20 concurrent add_task calls to the same project serialise
    through the per-project lock — every task gets a distinct id and the
    taskmaster backend never sees overlapping invocations."""
    from fused_memory.middleware.ticket_store import TicketStore

    tracker = _OverlapTracker()
    counter = {'n': 0}
    id_lock = asyncio.Lock()

    async def fake_add_task(**kwargs):
        async with id_lock:
            counter['n'] += 1
            my_id = counter['n']
        # Simulate I/O
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return {'id': str(my_id), 'title': kwargs.get('title', '')}

    overlap_tm.add_task = AsyncMock(side_effect=fake_add_task)
    # Instrument by also counting overlap via a wrapper.
    original = overlap_tm.add_task

    async def instrumented(**kwargs):
        tracker.in_flight['p'] = tracker.in_flight.get('p', 0) + 1
        tracker.peak['p'] = max(
            tracker.peak.get('p', 0),
            tracker.in_flight['p'],
        )
        try:
            return await original(**kwargs)
        finally:
            tracker.in_flight['p'] -= 1

    overlap_tm.add_task = instrumented

    store = TicketStore(tmp_path / 'burst_tickets.db')
    await store.initialize()
    interceptor = TaskInterceptor(overlap_tm, reconciler, event_buffer, ticket_store=store)

    try:
        N = 20
        results = await asyncio.gather(
            *[_submit_and_resolve(interceptor, '/project', title=f'Task {i}') for i in range(N)]
        )
    finally:
        await store.close()
        for _wt in list(interceptor._worker_tasks.values()):
            if not _wt.done():
                _wt.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await _wt

    assert len(results) == N
    ids = {r['id'] for r in results}
    assert len(ids) == N, f'duplicate ids produced: {results}'
    assert tracker.peak.get('p', 0) == 1, (
        f'per-project mutation overlap detected: peak={tracker.peak}'
    )


@pytest.mark.asyncio
async def test_mixed_op_concurrency_serialises_on_one_project(
    overlap_tm,
    reconciler,
    event_buffer,
):
    """WP-E: add + set_task_status + update_task concurrent on the same
    project all serialise through the per-project lock. The backend never
    observes two mutating calls in flight simultaneously."""
    tracker = _OverlapTracker()

    async def _delay(_return):
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return _return

    def instrument(method_name: str, return_value):
        async def side_effect(*args, **kwargs):
            tracker.in_flight['p'] = tracker.in_flight.get('p', 0) + 1
            tracker.peak['p'] = max(
                tracker.peak.get('p', 0),
                tracker.in_flight['p'],
            )
            try:
                return await _delay(return_value)
            finally:
                tracker.in_flight['p'] -= 1

        return side_effect

    overlap_tm.add_task = AsyncMock(
        side_effect=instrument('add_task', {'id': '99', 'title': 'x'}),
    )
    overlap_tm.update_task = AsyncMock(
        side_effect=instrument('update_task', {'success': True}),
    )
    overlap_tm.set_task_status = AsyncMock(
        side_effect=instrument('set_task_status', {'success': True}),
    )
    # get_task also counts — set_task_status holds the lock across it.
    overlap_tm.get_task = AsyncMock(
        side_effect=instrument(
            'get_task',
            {'id': '1', 'status': 'pending', 'title': 'T'},
        ),
    )
    overlap_tm.remove_tasks = AsyncMock(
        side_effect=instrument('remove_tasks', {'success': True}),
    )
    overlap_tm.add_dependency = AsyncMock(
        side_effect=instrument('add_dependency', {'success': True}),
    )

    interceptor = TaskInterceptor(overlap_tm, reconciler, event_buffer)

    coros = []
    # N adds
    for i in range(5):
        coros.append(_submit_and_resolve(interceptor, '/project', title=f'A{i}'))
    # M set_task_status (force an 'in-progress' transition — non-noop)
    for i in range(5):
        coros.append(interceptor.set_task_status(str(i), 'in-progress', '/project'))
    # K update_task
    for i in range(5):
        coros.append(
            interceptor.update_task(str(i), '/project', prompt=f'u{i}'),
        )
    # plus a couple of deps + removes
    coros.append(interceptor.remove_tasks(['7'], '/project'))
    coros.append(interceptor.add_dependency('3', '2', '/project'))

    await asyncio.gather(*coros)

    assert tracker.peak.get('p', 0) == 1, (
        f'concurrent mutation observed on same project: peak={tracker.peak}'
    )


@pytest.mark.asyncio
async def test_two_projects_do_not_serialise(
    overlap_tm,
    reconciler,
    event_buffer,
    tmp_path,
):
    """WP-E: per-project ticket queues serialise within each project but allow
    cross-project concurrency.

    After step-68 sharding, each project_id gets its own asyncio.Queue and
    asyncio.Task worker.  Concurrent add_task calls on distinct projects
    therefore run concurrently (total_peak may reach the number of active
    projects == 2 here).  Same-project ops are still serialised within each
    project worker (peak_a <= 1, peak_b <= 1).

    This is the correct behaviour: a slow LLM on projA must NOT block projB.
    """
    from fused_memory.middleware.ticket_store import TicketStore
    from fused_memory.models.scope import resolve_project_id

    tracker = _OverlapTracker()
    assert resolve_project_id('/projA') != resolve_project_id('/projB')

    # Events for guaranteed rendezvous: each project signals when it has
    # entered tm.add_task and waits for the other project to also be in-flight.
    # This replaces the previous timing-based approach (50 sleep(0) iterations)
    # which was flaky under heavy CI load (16 xdist workers).
    projA_entered = asyncio.Event()
    projB_entered = asyncio.Event()

    async def side_effect(**kwargs):
        pr = kwargs.get('project_root', '')
        key = resolve_project_id(pr)
        tracker.in_flight[key] = tracker.in_flight.get(key, 0) + 1
        tracker._global_in_flight += 1
        tracker.total_peak = max(tracker.total_peak, tracker._global_in_flight)
        try:
            # Signal this project's entry and wait for the other project to
            # enter too — guaranteeing true simultaneous overlap (total_peak==2)
            # without relying on event-loop scheduling timing.
            if key == resolve_project_id('/projA'):
                projA_entered.set()
                try:
                    await asyncio.wait_for(projB_entered.wait(), timeout=10.0)
                except TimeoutError:
                    pytest.fail(
                        f'project A entered but B never did — '
                        f'projA_entered={projA_entered.is_set()} '
                        f'projB_entered={projB_entered.is_set()}'
                    )
            else:
                projB_entered.set()
                try:
                    await asyncio.wait_for(projA_entered.wait(), timeout=10.0)
                except TimeoutError:
                    pytest.fail(
                        f'project B entered but A never did — '
                        f'projA_entered={projA_entered.is_set()} '
                        f'projB_entered={projB_entered.is_set()}'
                    )
            return {'id': '1', 'title': kwargs.get('title', '')}
        finally:
            tracker.in_flight[key] -= 1
            tracker._global_in_flight -= 1

    overlap_tm.add_task = AsyncMock(side_effect=side_effect)
    store = TicketStore(tmp_path / 'two_proj_tickets.db')
    await store.initialize()
    interceptor = TaskInterceptor(overlap_tm, reconciler, event_buffer, ticket_store=store)

    coros = []
    for i in range(5):
        coros.append(_submit_and_resolve(interceptor, '/projA', title=f'a{i}'))
        coros.append(_submit_and_resolve(interceptor, '/projB', title=f'b{i}'))
    try:
        await asyncio.gather(*coros)
    finally:
        await store.close()
        for _wt in list(interceptor._worker_tasks.values()):
            if not _wt.done():
                _wt.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await _wt

    # Per-project peak is 1 (each project worker serialises same-project ops).
    peak_a = tracker.peak.get(resolve_project_id('/projA'), 0)
    peak_b = tracker.peak.get(resolve_project_id('/projB'), 0)
    assert peak_a <= 1 and peak_b <= 1, f'same-project overlap: {tracker.peak}'
    # With per-project workers, projA and projB can run concurrently;
    # total_peak must equal the number of active projects (2), confirming
    # cross-project parallelism is achieved.
    assert tracker.total_peak == 2, (
        f'expected per-project parallelism (total_peak==2): total_peak={tracker.total_peak}'
    )


@pytest.mark.asyncio
async def test_set_task_status_holds_lock_across_read_and_write(
    overlap_tm,
    reconciler,
    event_buffer,
):
    """WP-E: two concurrent set_task_status calls on the same project see
    a consistent before-state. Without the lock, both could read
    'pending' and both call tm.set_task_status; with the lock, the second
    reader observes the first's write and short-circuits.
    """
    # Simulate a stateful backend: get_task returns current status,
    # set_task_status updates it.
    state = {'status': 'pending'}
    call_log: list[str] = []

    async def get_task(task_id, project_root, tag=None):
        await asyncio.sleep(0)
        return {'id': task_id, 'status': state['status'], 'title': 'T'}

    async def set_task_status(task_id, status, project_root, tag=None):
        call_log.append(f'{task_id}:{state["status"]}->{status}')
        # Yield between the read above and committing the new state so
        # the race window is widened.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        state['status'] = status
        return {'success': True}

    overlap_tm.get_task = AsyncMock(side_effect=get_task)
    overlap_tm.set_task_status = AsyncMock(side_effect=set_task_status)

    interceptor = TaskInterceptor(overlap_tm, reconciler, event_buffer)

    # Two concurrent transitions to different target statuses.
    r1, r2 = await asyncio.gather(
        interceptor.set_task_status('1', 'in-progress', '/project'),
        interceptor.set_task_status('1', 'done', '/project'),
    )

    # Second caller must see the first's mutation — one of the two must
    # be a no-op (idempotent against the already-applied status) OR the
    # transitions must chain pending->in-progress->done (no stale read
    # of 'pending' for the second call).
    assert len(call_log) <= 2
    # Both statuses recorded should be among the ones we asked for;
    # crucially the `from` side of the second must NOT still say 'pending'
    # if the first already mutated it.
    if len(call_log) == 2:
        first_from, first_to = call_log[0].split(':')[1].split('->')
        second_from, second_to = call_log[1].split(':')[1].split('->')
        assert first_from == 'pending'
        # With the lock, second read sees the first write.
        assert second_from == first_to, f'stale before-state observed: {call_log}'
    assert r1.get('success') or r1.get('no_op')
    assert r2.get('success') or r2.get('no_op')


@pytest.mark.asyncio
@pytest.mark.skipif(
    bool(os.environ.get('PYTEST_XDIST_WORKER')),
    reason='latency threshold unreliable under xdist I/O contention; run without -n to exercise this guard',
)
async def test_single_call_latency_smoke(
    overlap_tm,
    reconciler,
    event_buffer,
):
    """WP-E: smoke check — a sequence of sequential mutating calls under
    no contention finishes within a 5 s budget, confirming the lock itself
    adds no significant per-call overhead.

    Observed single-worker runtime is ~0.6 s (pure in-process AsyncMock
    I/O), so the 5 s ceiling gives roughly 8x headroom — wide enough to
    absorb CI noise while still catching 2-3x regressions.

    Skipped automatically when PYTEST_XDIST_WORKER is set: SQLite disk I/O
    contention from 32 concurrent workers inflated the worst-case to 11.3 s,
    making a meaningful threshold flaky.  Running in single-worker mode keeps
    the bound deterministic.  To exercise this guard in CI, invoke pytest
    without the -n flag in a dedicated single-worker step, e.g.:
    ``pytest fused-memory/tests/test_task_interceptor.py::test_single_call_latency_smoke``
    """
    import time

    overlap_tm.set_task_status = AsyncMock(return_value={'success': True})
    overlap_tm.get_task = AsyncMock(
        return_value={'id': '1', 'status': 'pending', 'title': 'T'},
    )
    interceptor = TaskInterceptor(overlap_tm, None, event_buffer)

    N = 200
    start = time.perf_counter()
    for i in range(N):
        status = 'in-progress' if i % 2 == 0 else 'pending'
        await interceptor.set_task_status('1', status, '/project')
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f'{N} sequential calls took {elapsed:.3f}s (latency regression guard)'


@pytest.mark.asyncio
async def test_set_task_status_does_not_block_during_add_task_curator(
    taskmaster,
    reconciler,
    event_buffer,
    curator_enabled_config,
    tmp_path,
):
    """Split-lock regression (2026-04-20): a long-running curator.curate()
    inside add_task MUST NOT block a concurrent set_task_status on the
    same project.

    Before the split, both ops took the single ``_project_lock``; a 25-35 s
    curator LLM call under add_task stalled every set_task_status on the
    same project for the full duration, blowing past the orchestrator's
    15 s client timeout and logging 50+ empty-str "Failed to set task X
    status to Y:" errors per run (reify log
    /tmp/orch-reify-20260420-082733.log). After the split, add_task holds
    ``_curator_lock`` for the LLM call and only briefly acquires
    ``_write_lock`` for the tm.add_task write; set_task_status takes just
    ``_write_lock`` and so completes promptly.
    """
    from fused_memory.middleware.ticket_store import TicketStore

    store = TicketStore(tmp_path / 'split_lock_tickets.db')
    await store.initialize()
    interceptor = TaskInterceptor(
        taskmaster,
        reconciler,
        event_buffer,
        config=curator_enabled_config,
        ticket_store=store,
    )

    CURATOR_LATENCY_S = 2.0

    async def slow_curate(candidate, project_id, project_root):
        await asyncio.sleep(CURATOR_LATENCY_S)
        # action='create' so the flow falls through to tm.add_task
        return CuratorDecision(
            action='create',
            justification='ok',
        )

    curator = MagicMock()
    curator.curate = AsyncMock(side_effect=slow_curate)

    # curate_batch delegates to curate() so slow_curate is still invoked
    # and the latency-based assertion holds.
    async def _slow_curate_batch(candidates, pid, project_root):
        return [await curator.curate(c, pid, project_root) for c in candidates]

    curator.curate_batch = AsyncMock(side_effect=_slow_curate_batch)
    curator.record_task = AsyncMock()
    curator.reembed_task = AsyncMock()
    curator.note_created = MagicMock()
    interceptor._curator = curator

    start = asyncio.get_event_loop().time()

    async def timed_set_status():
        # Fire slightly after add_task so add_task is the one holding
        # _curator_lock first.
        await asyncio.sleep(0.05)
        t0 = asyncio.get_event_loop().time()
        result = await interceptor.set_task_status(
            '1',
            'in-progress',
            '/project',
        )
        return result, asyncio.get_event_loop().time() - t0

    try:
        add_result, (status_result, status_elapsed) = await asyncio.gather(
            _submit_and_resolve(interceptor, '/project', title='concurrent add'),
            timed_set_status(),
        )
    finally:
        await store.close()
        for _wt in list(interceptor._worker_tasks.values()):
            if not _wt.done():
                _wt.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await _wt
    total_elapsed = asyncio.get_event_loop().time() - start

    # add_task ran the full curator → ~CURATOR_LATENCY_S
    assert total_elapsed >= CURATOR_LATENCY_S * 0.9, (
        f'add_task did not wait for curator: total={total_elapsed:.3f}s'
    )
    # set_task_status must NOT wait for curator.
    # Budget: well under curator latency. Mocked tm writes yield only once,
    # so half a second is generous for CI jitter.
    assert status_elapsed < 0.5, (
        f'set_task_status blocked behind add_task curator: '
        f'status_elapsed={status_elapsed:.3f}s (budget 0.5s), '
        f'curator_latency={CURATOR_LATENCY_S}s'
    )
    # Both writes landed.
    assert add_result.get('id') == '2'  # taskmaster fixture default
    assert status_result.get('success') or status_result.get('no_op')
    taskmaster.add_task.assert_called_once()
    taskmaster.set_task_status.assert_called_once()


# ---------------------------------------------------------------------------
# step-15: _is_ticket_id helper
# ---------------------------------------------------------------------------
def test_is_ticket_id_recognises_tkt_prefix():
    """_is_ticket_id() returns True for tkt_-prefixed strings, False otherwise."""
    from fused_memory.middleware.task_interceptor import _is_ticket_id

    assert _is_ticket_id('tkt_0000000000000000000000000000') is True
    assert _is_ticket_id('tkt_abc') is True
    assert _is_ticket_id('') is False
    assert _is_ticket_id('123') is False
    assert _is_ticket_id('1.2') is False
    assert _is_ticket_id(None) is False


# ---------------------------------------------------------------------------
# step-19: submit_task persists a pending ticket and returns its id
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def ticket_store(tmp_path):
    """A real TicketStore backed by a temporary SQLite file."""
    from fused_memory.middleware.ticket_store import TicketStore

    store = TicketStore(tmp_path / 'tickets.db')
    await store.initialize()
    yield store
    await store.close()


@pytest_asyncio.fixture
async def interceptor_with_store(taskmaster, reconciler, event_buffer, ticket_store):
    """TaskInterceptor with a real TicketStore wired in."""
    ti = TaskInterceptor(taskmaster, reconciler, event_buffer, ticket_store=ticket_store)
    yield ti


@pytest.mark.asyncio
async def test_submit_task_persists_ticket_and_returns_id(
    interceptor_with_store,
    ticket_store,
    taskmaster,
):
    """submit_task enqueues a ticket immediately and returns {'ticket': 'tkt_...'}.

    The taskmaster backend must NOT be called — curator processing is deferred
    to the worker.
    """
    result = await interceptor_with_store.submit_task(
        project_root='/project', title='T', description='D'
    )

    assert isinstance(result, dict), f'Expected dict, got {result!r}'
    assert 'ticket' in result, f'Expected ticket key in result: {result}'
    ticket_id = result['ticket']
    assert ticket_id.startswith('tkt_'), f'Ticket id should start with tkt_: {ticket_id!r}'

    # Row should be persisted as pending
    row = await ticket_store.get(ticket_id)
    assert row is not None, 'Ticket row should exist in store'
    assert row['status'] == 'pending'
    assert row['project_id'] is not None

    # The taskmaster backend must NOT have been called
    taskmaster.add_task.assert_not_called()


# ---------------------------------------------------------------------------
# step-57: start() flushes prior pending tickets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_flushes_prior_pending_tickets(
    taskmaster,
    reconciler,
    event_buffer,
    tmp_path,
):
    """interceptor.start() flushes tickets left pending by a previous run.

    Scenario: a previous server run submitted a ticket but crashed before
    resolving it.  On restart, a fresh TaskInterceptor with the same
    tickets.db calls start(), which marks the orphaned pending ticket as
    failed with reason='server_restart'.
    """
    from fused_memory.middleware.ticket_store import TicketStore

    # --- "previous run" ---
    # Manually insert a pending ticket directly via the store API.
    store = TicketStore(tmp_path / 'restart_tickets.db')
    await store.initialize()
    orphan_id = await store.submit(project_id='project', candidate_json='{}')
    row_before = await store.get(orphan_id)
    assert row_before is not None and row_before['status'] == 'pending', (
        f'Setup: expected pending ticket, got {row_before}'
    )
    # Simulate the previous run finishing (store closed, process exited).
    await store.close()

    # --- "new run": fresh interceptor with the same on-disk db ---
    fresh_store = TicketStore(tmp_path / 'restart_tickets.db')
    await fresh_store.initialize()
    ti = TaskInterceptor(taskmaster, reconciler, event_buffer, ticket_store=fresh_store)

    try:
        await ti.start()

        # The orphaned ticket must now be failed with reason='server_restart'.
        row_after = await fresh_store.get(orphan_id)
        assert row_after is not None, 'Ticket row should still exist after start()'
        assert row_after['status'] == 'failed', (
            f'Expected status=failed after start(), got {row_after["status"]!r}'
        )
        assert row_after.get('reason') == 'server_restart', (
            f'Expected reason=server_restart, got {row_after.get("reason")!r}'
        )
    finally:
        await ti.close()


# ---------------------------------------------------------------------------
# step-59: server/main.py wires TicketStore into TaskInterceptor via helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_main_wires_ticket_store_into_interceptor(
    taskmaster,
    reconciler,
    event_buffer,
    tmp_path,
):
    """_build_ticket_store (server.main helper) constructs and initialises a
    TicketStore that main.py wires into TaskInterceptor.

    Asserts:
    1. _build_ticket_store returns a TicketStore backed by data_dir/tickets.db.
    2. The returned store's _db is connected (not None) — initialize() was called.
    3. A TaskInterceptor built with ticket_store=store exposes it as _ticket_store.
    """
    from fused_memory.server.main import _build_ticket_store  # noqa: PLC0415

    store = await _build_ticket_store(tmp_path)

    from fused_memory.middleware.ticket_store import TicketStore

    assert isinstance(store, TicketStore)
    assert store._db_path == tmp_path / 'tickets.db', (
        f'Expected db path {tmp_path / "tickets.db"}, got {store._db_path}'
    )
    assert store._db is not None, 'TicketStore._db should be connected after _build_ticket_store'

    # Verify TaskInterceptor accepts and stores the ticket_store kwarg correctly.
    ti = TaskInterceptor(taskmaster, reconciler, event_buffer, ticket_store=store)
    assert ti._ticket_store is store, (
        'TaskInterceptor._ticket_store should be the store passed at construction'
    )

    await store.close()


# step-61: regression-guard — add_task no longer takes _curator_lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_task_worker_takes_curator_lock_for_r3(
    interceptor_facade,
    taskmaster,
):
    """R3 invariant: the add_task worker path must acquire _curator_lock so
    it is mutually exclusive with remove_task curator calls.

    Earlier in the ticket-queue refactor (step-46) the worker did not acquire
    this lock — the single-worker queue was treated as sufficient serialisation.
    The worker now takes _curator_lock(project_id) across curate() → note_created
    → record_task, preserving the R3 invariant while retaining per-project
    queue+worker fairness.
    """
    acquisition_count = 0

    class _CountingLock:
        """asyncio.Lock wrapper that increments acquisition_count on __aenter__."""

        def __init__(self):
            self._lock = asyncio.Lock()

        async def __aenter__(self):
            nonlocal acquisition_count
            acquisition_count += 1
            await self._lock.acquire()
            return self

        async def __aexit__(self, *args):
            self._lock.release()

    counting_lock = _CountingLock()
    # Replace the per-project lock factory with one that always returns our counter.
    interceptor_facade._curator_lock = lambda project_id: counting_lock

    # --- add_task (facade via submit_task → worker): MUST acquire curator_lock once ---
    await _submit_and_resolve(interceptor_facade, project_root='/project', title='CL guard test')
    assert acquisition_count == 1, (
        f'add_task worker should acquire _curator_lock exactly once; got {acquisition_count}'
    )


# ---------------------------------------------------------------------------
# step-63: regression — no lost-wakeup between terminal-check and event-register
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_ticket_no_lost_wakeup_between_read_and_register(
    interceptor_with_store,
    ticket_store,
):
    """Regression: worker completing between the initial row-read and event
    registration must NOT cause resolve_ticket to hang.

    The old flow (read → register) had a window where _signal_ticket_event
    fires with no registered event, so the signal is lost and event.wait()
    blocks indefinitely.  The fixed flow (register → read → re-check) closes
    this: either the read already sees terminal, or the signal arrives after
    registration and event.wait() returns immediately.

    This test simulates the race by monkeypatching ticket_store.get so that
    the first pending-returning call:
      (a) marks the ticket resolved in the store, and
      (b) calls interceptor._signal_ticket_event(ticket_id)
    *before* returning the (now stale) pending row.

    Under the OLD implementation _signal_ticket_event finds an empty
    _ticket_events dict and the signal is lost, so event.wait() hangs and
    asyncio.wait_for(timeout=2) raises TimeoutError.

    Under the FIXED implementation the event is registered BEFORE the first
    get() call, so _signal_ticket_event finds and sets the event; event.wait()
    returns immediately and resolve_ticket returns the terminal result.
    """
    # Submit a ticket — creates a pending row in the store.
    # Don't start the worker (don't call submit_task which would start it);
    # we insert the ticket directly to avoid real worker interference.
    ticket_id = await ticket_store.submit(
        project_id='p',
        candidate_json='{}',
    )

    original_get = ticket_store.get
    call_count = 0

    async def racing_get(tid: str):
        nonlocal call_count
        row = await original_get(tid)
        # On the first pending-returning call only: simulate the worker
        # completing between the caller's terminal-check and event-registration.
        if call_count == 0 and row is not None and row['status'] == 'pending':
            call_count += 1
            # Mark resolved in the store (worker's write).
            await ticket_store.mark_resolved(tid, status='created', task_id='42')
            # Signal the event — under the FIXED flow the event is already
            # registered so the signal is not lost; under OLD flow it is lost.
            interceptor_with_store._signal_ticket_event(tid)
        # Return the stale row (as it was before mark_resolved) so the caller
        # falls through to the wait path even under the fixed implementation.
        return row

    ticket_store.get = racing_get
    try:
        result = await asyncio.wait_for(
            interceptor_with_store.resolve_ticket(ticket_id, '/p', timeout_seconds=None),
            timeout=2.0,
        )
    finally:
        ticket_store.get = original_get

    assert result.get('status') == 'created', f'Expected status=created but got: {result!r}'
    assert result.get('task_id') == '42', f'Expected task_id=42 but got: {result!r}'


# ---------------------------------------------------------------------------
# cancel_ticket: interceptor-level contract tests (steps 1, 3, 5)
# + amendment: ConfigError and TOCTOU race
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_ticket_no_store_returns_config_error(interceptor):
    """cancel_ticket returns ConfigError when ticket_store is not configured.

    amend: distinguish misconfigured-server from genuine missing-ticket so
    callers can tell the difference between a stale ticket_id and a server
    wiring failure.
    """
    result = await interceptor.cancel_ticket('tkt_ANY')
    assert result == {
        'error': 'ticket_store not configured',
        'error_type': 'ConfigError',
        'ticket_id': 'tkt_ANY',
    }, f'Expected ConfigError dict, got: {result!r}'


@pytest.mark.asyncio
async def test_cancel_ticket_missing_returns_not_found(interceptor_with_store):
    """cancel_ticket returns not_found for a ticket_id that does not exist.

    RED in step-1: TaskInterceptor.cancel_ticket does not yet exist.
    """
    result = await interceptor_with_store.cancel_ticket('tkt_DOES_NOT_EXIST')
    assert result == {'error': 'not_found', 'ticket_id': 'tkt_DOES_NOT_EXIST'}, (
        f'Expected not_found dict, got: {result!r}'
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('terminal_status', ['failed', 'cancelled', 'created'])
async def test_cancel_ticket_terminal_returns_noop(
    interceptor_with_store,
    ticket_store,
    terminal_status,
):
    """cancel_ticket returns no_op for a ticket already in a terminal status.

    RED in step-3: the placeholder in step-2 raises NotImplementedError for
    non-pending rows.
    """
    # Insert a pending ticket directly via the store.
    ticket_id = await ticket_store.submit(project_id='p', candidate_json='{}')
    # Push it to the given terminal status.
    await ticket_store.mark_resolved(ticket_id, status=terminal_status, reason='test')

    result = await interceptor_with_store.cancel_ticket(ticket_id)
    assert result == {
        'status': terminal_status,
        'ticket_id': ticket_id,
        'no_op': True,
    }, f'Expected no_op dict with status={terminal_status!r}, got: {result!r}'


@pytest.mark.asyncio
async def test_cancel_ticket_pending_marks_cancelled(
    interceptor_with_store,
    ticket_store,
):
    """cancel_ticket marks a pending ticket as cancelled and returns the right shape.

    RED in step-5: NotImplementedError from the pending-cancel placeholder.
    """
    ticket_id = await ticket_store.submit(project_id='p', candidate_json='{}')

    result = await interceptor_with_store.cancel_ticket(ticket_id)

    assert result == {'status': 'cancelled', 'ticket_id': ticket_id}, (
        f'Expected cancelled dict, got: {result!r}'
    )
    # Confirm the row was actually updated in the store.
    row = await ticket_store.get(ticket_id)
    assert row is not None
    assert row['status'] == 'cancelled', f'Row status should be cancelled: {row!r}'
    assert row['reason'] == 'user_cancelled', f'Row reason should be user_cancelled: {row!r}'


@pytest.mark.asyncio
async def test_cancel_ticket_signals_resolve_ticket_waiter(
    interceptor_with_store,
    ticket_store,
):
    """cancel_ticket wakes a concurrent resolve_ticket waiter so it exits promptly.

    RED in step-5: the pending-cancel placeholder raises NotImplementedError,
    which causes cancel_ticket to crash before calling _signal_ticket_event.
    """
    ticket_id = await ticket_store.submit(project_id='p', candidate_json='{}')

    # Start a resolve_ticket waiter in the background.  It blocks on the
    # asyncio.Event registered for this ticket.
    waiter = asyncio.create_task(
        interceptor_with_store.resolve_ticket(ticket_id, '/p', timeout_seconds=5.0),
    )
    # Yield control so resolve_ticket can register its Event before we cancel.
    await asyncio.sleep(0)

    # Cancel the ticket — this should signal the waiter's Event.
    cancel_result = await interceptor_with_store.cancel_ticket(ticket_id)
    assert cancel_result == {'status': 'cancelled', 'ticket_id': ticket_id}

    # The waiter should wake and return the cancelled row within the timeout.
    waiter_result = await asyncio.wait_for(waiter, timeout=3.0)
    assert waiter_result.get('status') == 'cancelled', (
        f'resolve_ticket waiter expected status=cancelled, got: {waiter_result!r}'
    )


@pytest.mark.asyncio
async def test_cancel_ticket_race_returns_noop_with_actual_status(
    interceptor_with_store,
    ticket_store,
):
    """cancel_ticket returns no_op with the real status on a TOCTOU race.

    amend: mark_resolved returns False when a concurrent writer terminates the
    ticket between cancel_ticket's get() and the UPDATE.  In that case the
    method must re-fetch the actual status and return the no_op shape instead
    of falsely reporting status='cancelled'.
    """
    ticket_id = await ticket_store.submit(project_id='p', candidate_json='{}')

    # Intercept mark_resolved to simulate a concurrent worker that terminates
    # the ticket (to 'created') *before* our cancel UPDATE lands.
    original_mark_resolved = ticket_store.mark_resolved

    async def racing_mark_resolved(tid: str, *, status: str, **kwargs):
        if tid == ticket_id and status == 'cancelled':
            # The racing writer wins first: force the row to terminal 'created'.
            await original_mark_resolved(tid, status='created', reason='raced_first')
        # Now our cancel UPDATE runs — it returns False because status != 'pending'.
        return await original_mark_resolved(tid, status=status, **kwargs)

    ticket_store.mark_resolved = racing_mark_resolved
    try:
        result = await interceptor_with_store.cancel_ticket(ticket_id)
    finally:
        ticket_store.mark_resolved = original_mark_resolved

    assert result == {
        'status': 'created',
        'ticket_id': ticket_id,
        'no_op': True,
    }, f'Expected no_op with actual status=created, got: {result!r}'


# ---------------------------------------------------------------------------
# Terminal-exit gate: server-side FSM that refuses done/cancelled -> non-same
# without an explicit reopen_reason.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_terminal_exit_rejects_done_to_pending_without_reason(
    interceptor,
    taskmaster,
):
    """done -> pending with no reopen_reason returns terminal_exit_rejected."""
    taskmaster.get_task = AsyncMock(
        return_value={'id': '1', 'status': 'done', 'title': 'T'},
    )
    result = await interceptor.set_task_status('1', 'pending', '/project')
    assert result.get('error') == 'terminal_exit_rejected', result
    assert result.get('from_status') == 'done'
    assert result.get('to_status') == 'pending'
    # The backing Taskmaster must NOT be mutated when the gate trips.
    taskmaster.set_task_status.assert_not_called()


@pytest.mark.asyncio
async def test_terminal_exit_rejects_cancelled_to_pending_without_reason(
    interceptor,
    taskmaster,
):
    taskmaster.get_task = AsyncMock(
        return_value={'id': '1', 'status': 'cancelled', 'title': 'T'},
    )
    result = await interceptor.set_task_status('1', 'pending', '/project')
    assert result.get('error') == 'terminal_exit_rejected'
    taskmaster.set_task_status.assert_not_called()


@pytest.mark.asyncio
async def test_terminal_exit_accepts_with_reopen_reason(
    interceptor,
    taskmaster,
):
    """done -> pending with a non-empty reopen_reason succeeds and persists reason."""
    taskmaster.get_task = AsyncMock(
        return_value={'id': '1', 'status': 'done', 'title': 'T'},
    )
    result = await interceptor.set_task_status(
        '1',
        'pending',
        '/project',
        reopen_reason='un-defer script',
    )
    assert result.get('success') or 'error' not in result, result
    taskmaster.set_task_status.assert_called_once()
    # update_task called with metadata containing reopen_reason.
    assert taskmaster.update_task.called, 'reopen_reason must be persisted'
    persisted_metadata = None
    for call in taskmaster.update_task.call_args_list:
        md = call.kwargs.get('metadata')
        if md and 'reopen_reason' in md:
            persisted_metadata = md
            break
    assert persisted_metadata is not None
    parsed = json.loads(persisted_metadata)
    assert parsed['reopen_reason'] == 'un-defer script'
    assert parsed['reopen_from'] == 'done'
    assert 'reopen_at' in parsed


@pytest.mark.asyncio
async def test_terminal_exit_rejects_empty_string_reason(interceptor, taskmaster):
    """A whitespace-only reopen_reason is treated as missing."""
    taskmaster.get_task = AsyncMock(
        return_value={'id': '1', 'status': 'done', 'title': 'T'},
    )
    result = await interceptor.set_task_status(
        '1',
        'pending',
        '/project',
        reopen_reason='   ',
    )
    assert result.get('error') == 'terminal_exit_rejected'


@pytest.mark.asyncio
async def test_terminal_same_status_is_noop(interceptor, taskmaster):
    """done -> done returns a no-op even without reopen_reason (same-status guard first)."""
    taskmaster.get_task = AsyncMock(
        return_value={'id': '1', 'status': 'done', 'title': 'T'},
    )
    result = await interceptor.set_task_status('1', 'done', '/project')
    assert result == {'success': True, 'no_op': True, 'task_id': '1'}
    taskmaster.set_task_status.assert_not_called()


@pytest.mark.asyncio
async def test_terminal_exit_event_payload_includes_reopen_reason(
    interceptor,
    taskmaster,
    event_buffer,
):
    """Emitted event carries reopen_reason and reopen_from for audit."""
    taskmaster.get_task = AsyncMock(
        return_value={'id': '1', 'status': 'cancelled', 'title': 'T'},
    )
    await interceptor.set_task_status(
        '1',
        'pending',
        '/project',
        reopen_reason='manual re-scope',
    )
    events = await event_buffer.peek_buffered('project', limit=10)
    assert events, 'expected a task_status_changed event'
    payload = events[-1].payload
    assert payload.get('reopen_reason') == 'manual re-scope'
    assert payload.get('reopen_from') == 'cancelled'


# ---------------------------------------------------------------------------
# Batch-aware set_task_status: CSV input runs gates per-id.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_csv_set_task_status_runs_per_id_gates(interceptor, taskmaster):
    """CSV task_id input applies the terminal-exit gate to each id independently."""
    statuses = {'1': 'done', '2': 'pending', '3': 'pending'}

    async def get_task(task_id, project_root, tag=None):
        return {'id': task_id, 'status': statuses[task_id], 'title': 'T'}

    taskmaster.get_task.side_effect = get_task

    result = await interceptor.set_task_status(
        '1,2,3',
        'pending',
        '/project',
    )
    assert 'results' in result
    per_id = {r['task_id']: r['result'] for r in result['results']}
    # Task 1 was 'done' — rejected by the gate.
    assert per_id['1'].get('error') == 'terminal_exit_rejected'
    # Task 2 is already 'pending' — no-op.
    assert per_id['2'].get('no_op') is True
    # Task 3 is also 'pending' — no-op.
    assert per_id['3'].get('no_op') is True
    # all_ok is False because one id hit the gate.
    assert result['success'] is False


@pytest.mark.asyncio
async def test_csv_set_task_status_mixed_statuses_partial_success(
    interceptor,
    taskmaster,
):
    """CSV input where some ids succeed and others hit the gate."""
    statuses = {'1': 'done', '2': 'in-progress'}

    async def get_task(task_id, project_root, tag=None):
        return {'id': task_id, 'status': statuses[task_id], 'title': 'T'}

    taskmaster.get_task.side_effect = get_task

    result = await interceptor.set_task_status('1,2', 'pending', '/project')
    per_id = {r['task_id']: r['result'] for r in result['results']}
    assert per_id['1'].get('error') == 'terminal_exit_rejected'
    # Task 2: in-progress -> pending, standard allowed transition.
    assert per_id['2'].get('success') is True or 'error' not in per_id['2']
    assert result['success'] is False  # overall false because 1 failed


# ---------------------------------------------------------------------------
# step-21 / step-23: BulkResetGuard integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def bulk_reset_guard():
    """BulkResetGuard with test-friendly thresholds.

    Both done_threshold and in_progress_threshold are set to 3 so the two
    integration tests (done→pending and in-progress→pending) both trip the
    guard with a 4-task CSV, matching the original single-threshold=3 behaviour.
    """
    from fused_memory.reconciliation.bulk_reset_guard import BulkResetGuard

    return BulkResetGuard(
        done_threshold=3,
        in_progress_threshold=3,
        window_seconds=60.0,
        escalation_rate_limit_seconds=900.0,
    )


@pytest.fixture
def interceptor_with_guard(taskmaster, reconciler, event_buffer, bulk_reset_guard):
    """TaskInterceptor with BulkResetGuard wired in."""
    return TaskInterceptor(taskmaster, reconciler, event_buffer, bulk_reset_guard=bulk_reset_guard)


@pytest.mark.asyncio
async def test_set_task_status_csv_done_to_pending_tripping_guard_rejects_and_escalates(
    interceptor_with_guard,
    taskmaster,
    tmp_path,
):
    """CSV done→pending: first 3 apply, tasks 4 and 5 are rejected by the guard."""

    async def get_task_done(task_id, project_root, tag=None):
        return {'id': task_id, 'status': 'done', 'title': f'Task {task_id}'}

    taskmaster.get_task.side_effect = get_task_done

    result = await interceptor_with_guard.set_task_status(
        task_id='1,2,3,4,5',
        status='pending',
        project_root=str(tmp_path),
        reopen_reason='test bulk autopilot reset',
    )

    # (a) top-level: overall failure, five per-id results
    assert result['success'] is False
    assert len(result['results']) == 5

    per_id = {r['task_id']: r['result'] for r in result['results']}

    # (b) First three applied successfully
    for tid in ('1', '2', '3'):
        r = per_id[tid]
        assert r.get('success') is True, f'task {tid}: expected success, got {r}'

    # (c) Tasks 4 and 5 rejected by guard
    for tid in ('4', '5'):
        r = per_id[tid]
        assert r.get('error_type') == 'BulkResetGuardTripped', (
            f'task {tid}: expected BulkResetGuardTripped, got {r}'
        )
        assert r.get('success') is False
        assert 'affected_task_ids' in r
        assert 'triggering_timestamps' in r
        # kind must be 'done_to_pending' (step-10 wires this via to_error_dict).
        assert r.get('kind') == 'done_to_pending', (
            f'task {tid}: expected kind=done_to_pending, got {r.get("kind")!r}'
        )

    # (d) Escalation JSON exists under <project_root>/data/escalations/
    esc_dir = tmp_path / 'data' / 'escalations'
    esc_files = list(esc_dir.glob('esc-bulk-reset-*.json'))
    assert len(esc_files) >= 1, f'Expected at least 1 escalation file, found {esc_files}'

    # (e) tm.set_task_status NOT called for tasks 4 and 5 (guard short-circuited)
    called_ids = {call.args[0] for call in taskmaster.set_task_status.call_args_list}
    assert '4' not in called_ids, 'set_task_status should not have been called for task 4'
    assert '5' not in called_ids, 'set_task_status should not have been called for task 5'
    # Tasks 1, 2, 3 were called
    for tid in ('1', '2', '3'):
        assert tid in called_ids, f'set_task_status should have been called for task {tid}'


@pytest.mark.asyncio
async def test_set_task_status_csv_in_progress_to_pending_trips_guard(
    interceptor_with_guard,
    taskmaster,
    tmp_path,
):
    """CSV in-progress→pending: first 3 apply, task 4 is rejected by the guard.

    in-progress→pending does not hit the terminal-exit gate, so no reopen_reason
    needed. This exercises the non-terminal reversal path.
    """

    async def get_task_in_progress(task_id, project_root, tag=None):
        return {'id': task_id, 'status': 'in-progress', 'title': f'Task {task_id}'}

    taskmaster.get_task.side_effect = get_task_in_progress

    result = await interceptor_with_guard.set_task_status(
        task_id='1,2,3,4',
        status='pending',
        project_root=str(tmp_path),
    )

    assert result['success'] is False
    assert len(result['results']) == 4

    per_id = {r['task_id']: r['result'] for r in result['results']}

    # First three ok
    for tid in ('1', '2', '3'):
        r = per_id[tid]
        assert r.get('success') is True, f'task {tid}: expected success, got {r}'

    # Task 4 rejected by guard
    r4 = per_id['4']
    assert r4.get('error_type') == 'BulkResetGuardTripped', (
        f'task 4: expected BulkResetGuardTripped, got {r4}'
    )
    # kind must be 'in_progress_to_pending' (step-10 wires this via to_error_dict).
    assert r4.get('kind') == 'in_progress_to_pending', (
        f'task 4: expected kind=in_progress_to_pending, got {r4.get("kind")!r}'
    )

    # tm.set_task_status NOT called for task 4
    called_ids = {call.args[0] for call in taskmaster.set_task_status.call_args_list}
    assert '4' not in called_ids, 'set_task_status should not have been called for task 4'


# ─────────────────────────────────────────────────────────────────────
# submit_task path-scope guardrails
# ─────────────────────────────────────────────────────────────────────


async def _cancel_interceptor_workers(ti) -> None:
    """Cancel any background ticket-worker tasks on *ti* and await them silently.

    Used in test teardown so fixture cleanup (DB close) never races a live
    worker. ``TestSubmitTaskGuardrail`` exercises only the submit path —
    no resolve waiters, no curator, no fire-and-forget tasks — so this
    partial cleanup is sufficient. If a future change wires the suite
    up to any of those, switch this teardown to ``await ti.close()``
    instead.
    """
    for t in list(ti._worker_tasks.values()):
        if not t.done():
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t


class TestSubmitTaskGuardrail:
    """Integration tests: path-scope guard wired into submit_task."""

    @pytest.mark.asyncio
    async def test_submit_task_rejects_dark_factory_paths_in_wrong_project(
        self,
        interceptor_with_store,
        ticket_store,
        taskmaster,
    ):
        """Filing a task referencing orchestrator/ under a non-dark-factory project
        returns a DarkFactoryPathScopeViolation error and does NOT persist a ticket.
        """
        try:
            result = await interceptor_with_store.submit_task(
                project_root='/some-other-project',
                title='Investigate orchestrator/harness.py deadlock',
                description='harness deadlock',
            )
        finally:
            # Ensure any background worker is cancelled before the ticket_store
            # fixture closes the DB, preventing "closed database" background errors.
            await _cancel_interceptor_workers(interceptor_with_store)

        # Guard must return a structured error
        assert isinstance(result, dict)
        assert result.get('error_type') == 'DarkFactoryPathScopeViolation', (
            f'Expected DarkFactoryPathScopeViolation error, got: {result}'
        )
        assert 'orchestrator/' in result.get('matched_paths', []), (
            f'Expected orchestrator/ in matched_paths: {result}'
        )

        # Ticket store must have zero rows (guard fires before persist)
        db = ticket_store._db
        assert db is not None
        cursor = await db.execute('SELECT COUNT(*) FROM tickets')
        row = await cursor.fetchone()
        assert row[0] == 0, f'Expected 0 tickets in store, found {row[0]}'

        # Taskmaster backend must never have been called
        taskmaster.add_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_submit_task_allows_dark_factory_paths_in_dark_factory_project(
        self,
        interceptor_with_store,
        taskmaster,
    ):
        """Filing the same task content under /dark-factory is allowed (project_id
        resolves to dark_factory and the guard no-ops).
        """
        try:
            result = await interceptor_with_store.submit_task(
                project_root='/dark-factory',
                title='Investigate orchestrator/harness.py deadlock',
                description='harness deadlock referencing orchestrator/harness.py',
            )
        finally:
            # Ensure any background worker is cancelled before the ticket_store
            # fixture closes the DB, preventing "closed database" background errors.
            await _cancel_interceptor_workers(interceptor_with_store)

        assert isinstance(result, dict)
        ticket_id = result.get('ticket', '')
        assert ticket_id.startswith('tkt_'), f'Expected ticket id starting with tkt_, got: {result}'
        assert 'error_type' not in result, (
            f'Should not have error_type for correctly-filed task: {result}'
        )

    @pytest.mark.asyncio
    async def test_submit_task_skips_build_candidate_for_dark_factory_project(
        self,
        interceptor_with_store,
        taskmaster,
        monkeypatch,
    ):
        """Hoist optimisation: _build_candidate is not invoked for the dark_factory
        project_id, since the path guard short-circuits to 'ok' anyway.

        Persistence-shape coverage (project_id column, candidate_json blob fields) is
        owned by ``test_submit_task_persists_canonical_blob`` — one place to update
        when the blob schema intentionally changes.
        """
        calls: list[dict] = []
        original = TaskInterceptor._build_candidate

        def spy(kwargs):
            calls.append(kwargs)
            return original(kwargs)

        monkeypatch.setattr(TaskInterceptor, '_build_candidate', staticmethod(spy))

        try:
            result = await interceptor_with_store.submit_task(
                project_root='/dark-factory',
                title='Investigate orchestrator/harness.py deadlock',
                description='harness deadlock',
            )
            # Snapshot call count immediately after submit_task returns, before
            # any cancellation/await — this ensures the assertion is unaffected
            # by a background worker that may also call _build_candidate.
            calls_after_submit = len(calls)
        finally:
            # Ensure any background worker is cancelled before the ticket_store
            # fixture closes the DB, preventing "closed database" background errors.
            await _cancel_interceptor_workers(interceptor_with_store)

        assert result.get('ticket', '').startswith('tkt_')
        assert calls_after_submit == 0, (
            f'Expected _build_candidate to be skipped for dark_factory; got {calls_after_submit} calls'
        )

    @pytest.mark.asyncio
    async def test_submit_task_persists_canonical_blob(
        self,
        interceptor_with_store,
        ticket_store,
        taskmaster,
    ):
        """Persistence contract: submit_task for /dark-factory stores a row whose
        project_id is 'dark_factory' and whose candidate_json blob contains the
        un-mutated kwargs (title, description) and metadata=None.

        This is the single owning test for the candidate_json serialisation format;
        update here when the blob schema intentionally changes.
        """
        try:
            result = await interceptor_with_store.submit_task(
                project_root='/dark-factory',
                title='Investigate orchestrator/harness.py deadlock',
                description='harness deadlock',
            )
        finally:
            # Ensure any background worker is cancelled before the ticket_store
            # fixture closes the DB, preventing "closed database" background errors.
            await _cancel_interceptor_workers(interceptor_with_store)

        assert result.get('ticket', '').startswith('tkt_'), (
            f'Expected tkt_-prefixed ticket, got: {result}'
        )

        # Direct _db access is intentional: we're pinning the storage-layer
        # serialisation contract, which has no public query path.  This mirrors
        # the pattern used by sibling tests in this class (e.g.
        # test_submit_task_rejects_dark_factory_paths_in_wrong_project).
        db = ticket_store._db
        assert db is not None
        cursor = await db.execute(
            'SELECT project_id, candidate_json FROM tickets WHERE ticket_id = ?',
            (result['ticket'],),
        )
        row = await cursor.fetchone()
        assert row is not None, f'Expected persisted ticket row for ticket_id={result["ticket"]!r}'
        assert row['project_id'] == 'dark_factory', (
            f"Expected project_id 'dark_factory', got: {row['project_id']!r}"
        )
        blob = json.loads(row['candidate_json'])
        assert blob['project_root'] == '/dark-factory', (
            f"Expected project_root '/dark-factory' in blob, got: {blob['project_root']!r}"
        )
        assert blob['kwargs']['title'] == 'Investigate orchestrator/harness.py deadlock', (
            f'Expected title in blob kwargs un-mutated, got: {blob["kwargs"].get("title")!r}'
        )
        assert blob['kwargs']['description'] == 'harness deadlock', (
            f'Expected description in blob kwargs un-mutated, got: {blob["kwargs"].get("description")!r}'
        )
        assert blob['metadata'] is None, (
            f'Expected metadata=None in blob (no metadata was passed), got: {blob["metadata"]!r}'
        )

    @pytest.mark.asyncio
    async def test_submit_task_allows_clean_task_in_other_project(
        self,
        interceptor_with_store,
        taskmaster,
    ):
        """A task with no dark-factory paths in a non-dark-factory project proceeds
        normally (returns a ticket id).
        """
        try:
            result = await interceptor_with_store.submit_task(
                project_root='/some-other-project',
                title='Clean task',
                description='Generic refactor of foo/bar.py',
            )
        finally:
            # Ensure any background worker is cancelled before the ticket_store
            # fixture closes the DB, preventing "closed database" background errors.
            await _cancel_interceptor_workers(interceptor_with_store)

        assert isinstance(result, dict)
        ticket_id = result.get('ticket', '')
        assert ticket_id.startswith('tkt_'), f'Expected ticket id starting with tkt_, got: {result}'
        assert 'error_type' not in result

    @pytest.mark.asyncio
    async def test_submit_task_rejects_prompt_only_dark_factory_paths_in_wrong_project(
        self,
        interceptor_with_store,
        ticket_store,
        taskmaster,
    ):
        """A prompt-only submit_task (no title) referencing orchestrator/ under a
        non-dark-factory project returns a DarkFactoryPathScopeViolation error,
        persists no ticket, and never calls taskmaster.add_task.
        """
        try:
            result = await interceptor_with_store.submit_task(
                project_root='/some-other-project',
                prompt='Edit orchestrator/harness.py for the deadlock',
                # Deliberately NO title kwarg — this is the prompt-only path
            )
        finally:
            await _cancel_interceptor_workers(interceptor_with_store)

        # Guard must return a structured error
        assert isinstance(result, dict)
        assert result.get('error_type') == 'DarkFactoryPathScopeViolation', (
            f'Expected DarkFactoryPathScopeViolation error, got: {result}'
        )
        assert 'orchestrator/' in result.get('matched_paths', []), (
            f'Expected orchestrator/ in matched_paths: {result}'
        )

        # Ticket store must have zero rows (guard fires before persist)
        db = ticket_store._db
        assert db is not None
        cursor = await db.execute('SELECT COUNT(*) FROM tickets')
        row = await cursor.fetchone()
        assert row[0] == 0, f'Expected 0 tickets in store, found {row[0]}'

        # Taskmaster backend must never have been called
        taskmaster.add_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_submit_task_allows_prompt_only_dark_factory_paths_in_dark_factory_project(
        self,
        interceptor_with_store,
        taskmaster,
    ):
        """Prompt-only submit_task filed under /dark-factory is always allowed.

        The dark_factory short-circuit in check_text_for_dark_factory_paths must fire
        and the result must be a 'tkt_'-prefixed ticket with no error_type.
        """
        try:
            result = await interceptor_with_store.submit_task(
                project_root='/dark-factory',
                prompt='Edit orchestrator/harness.py for the deadlock',
                # No title — prompt-only path
            )
        finally:
            # Ensure any background worker is cancelled before the ticket_store
            # fixture closes the DB, preventing "closed database" background errors.
            await _cancel_interceptor_workers(interceptor_with_store)

        assert isinstance(result, dict)
        ticket_id = result.get('ticket', '')
        assert ticket_id.startswith('tkt_'), f'Expected ticket id starting with tkt_, got: {result}'
        assert 'error_type' not in result, (
            f'Should not have error_type for dark_factory project: {result}'
        )

    @pytest.mark.asyncio
    async def test_submit_task_allows_clean_prompt_only_in_other_project(
        self,
        interceptor_with_store,
        taskmaster,
    ):
        """Prompt-only submit_task with no dark-factory paths in a non-dark-factory
        project must not be rejected (returns a 'tkt_'-prefixed ticket).
        """
        try:
            result = await interceptor_with_store.submit_task(
                project_root='/some-other-project',
                prompt='Refactor foo/bar.py routing',
                # No title — prompt-only path, but no dark-factory paths
            )
        finally:
            # Ensure any background worker is cancelled before the ticket_store
            # fixture closes the DB, preventing "closed database" background errors.
            await _cancel_interceptor_workers(interceptor_with_store)

        assert isinstance(result, dict)
        ticket_id = result.get('ticket', '')
        assert ticket_id.startswith('tkt_'), f'Expected ticket id starting with tkt_, got: {result}'
        assert 'error_type' not in result, f'Should not have error_type for clean prompt: {result}'

    @pytest.mark.parametrize('field', ['prompt', 'description', 'details'])
    @pytest.mark.asyncio
    async def test_submit_task_rejects_dark_factory_path_in_any_fallback_field(
        self,
        field,
        interceptor_with_store,
        ticket_store,
        taskmaster,
    ):
        """The fallback text guard scans prompt, description, AND details — not just
        prompt.

        Each parametrised case passes a dark-factory path in ``field`` with no
        title kwarg, routing _build_candidate to return None and engaging the
        fallback branch.  All three channels must trigger
        DarkFactoryPathScopeViolation and persist no ticket.
        """
        try:
            result = await interceptor_with_store.submit_task(
                project_root='/some-other-project',
                **{field: 'Edit orchestrator/harness.py for the deadlock'},
                # Deliberately NO title — forces _build_candidate to return None
            )
        finally:
            await _cancel_interceptor_workers(interceptor_with_store)

        assert isinstance(result, dict)
        assert result.get('error_type') == 'DarkFactoryPathScopeViolation', (
            f'Field {field!r}: expected DarkFactoryPathScopeViolation, got: {result}'
        )
        assert 'orchestrator/' in result.get('matched_paths', []), (
            f'Field {field!r}: expected orchestrator/ in matched_paths: {result}'
        )

        # Ticket store must have zero rows (guard fires before persist)
        db = ticket_store._db
        assert db is not None
        cursor = await db.execute('SELECT COUNT(*) FROM tickets')
        row = await cursor.fetchone()
        assert row[0] == 0, f'Field {field!r}: expected 0 tickets, found {row[0]}'

        taskmaster.add_task.assert_not_called()


# ---------------------------------------------------------------------------
# Unit tests for TaskInterceptor._extract_meta_files
# ---------------------------------------------------------------------------


class TestExtractMetaFiles:
    """Unit tests for the TaskInterceptor._extract_meta_files static helper."""

    def test_dict_metadata_files_to_modify(self):
        """dict metadata with files_to_modify → returns the list verbatim."""
        kwargs = {'metadata': {'files_to_modify': ['orchestrator/harness.py', 'src/foo.py']}}
        result = TaskInterceptor._extract_meta_files(kwargs)
        assert result == ['orchestrator/harness.py', 'src/foo.py']

    def test_dict_metadata_files_only(self):
        """dict metadata with only files (canonical key) → returns files."""
        kwargs = {'metadata': {'files': ['fused-memory/src', 'orchestrator/']}}
        result = TaskInterceptor._extract_meta_files(kwargs)
        assert result == ['fused-memory/src', 'orchestrator/']

    def test_dict_metadata_both_keys_prefers_files(self):
        """dict metadata with BOTH keys → returns files (precedence over files_to_modify)."""
        kwargs = {
            'metadata': {
                'files': ['a.py'],
                'files_to_modify': ['legacy.py'],
            }
        }
        result = TaskInterceptor._extract_meta_files(kwargs)
        assert result == ['a.py']

    def test_dict_metadata_scalar_string_coerced_to_list(self):
        """dict metadata with files_to_modify as a string → coerced to single-element list."""
        kwargs = {'metadata': {'files_to_modify': 'orchestrator/harness.py'}}
        result = TaskInterceptor._extract_meta_files(kwargs)
        assert result == ['orchestrator/harness.py']

    def test_json_string_metadata_parsed(self):
        """JSON string metadata → parsed and files_to_modify extracted."""
        import json as _json

        meta_str = _json.dumps({'files_to_modify': ['orchestrator/harness.py']})
        kwargs = {'metadata': meta_str}
        result = TaskInterceptor._extract_meta_files(kwargs)
        assert result == ['orchestrator/harness.py']

    def test_malformed_json_string_returns_empty(self):
        """Malformed JSON string metadata → returns []."""
        kwargs = {'metadata': '{not valid json}'}
        result = TaskInterceptor._extract_meta_files(kwargs)
        assert result == []

    def test_none_metadata_returns_empty(self):
        """metadata=None → returns []."""
        kwargs = {'metadata': None}
        result = TaskInterceptor._extract_meta_files(kwargs)
        assert result == []

    def test_missing_metadata_key_returns_empty(self):
        """Missing metadata key → returns []."""
        kwargs = {}
        result = TaskInterceptor._extract_meta_files(kwargs)
        assert result == []

    def test_non_dict_metadata_list_returns_empty(self):
        """Non-dict metadata (e.g. list) → returns []."""
        kwargs = {'metadata': ['some', 'list']}
        result = TaskInterceptor._extract_meta_files(kwargs)
        assert result == []

    def test_falsy_entries_filtered_out(self):
        """Falsy entries ('', None) inside the list → filtered out."""
        kwargs = {'metadata': {'files_to_modify': ['', None, 'src/bar.py', '']}}
        result = TaskInterceptor._extract_meta_files(kwargs)
        assert result == ['src/bar.py']

    def test_non_string_entries_str_coerced(self):
        """Non-string truthy entries → str-coerced."""
        kwargs = {'metadata': {'files_to_modify': [42, 'src/foo.py']}}
        result = TaskInterceptor._extract_meta_files(kwargs)
        assert result == ['42', 'src/foo.py']

    def test_build_candidate_parses_metadata_once(self, monkeypatch):
        """_build_candidate must call _parse_metadata at most once per invocation.

        Regression guard for the hot-path dedupe: before the fix, _build_candidate
        called _parse_metadata directly (line 909) AND indirectly via
        _extract_meta_files (line 910) — two parses per title-bearing submission.
        After the fix (_build_candidate delegates to _extract_meta_files_from_meta
        using the meta already in scope), the count drops to ≤1.

        The guard uses <=1 rather than ==1 so a future short-circuit that skips
        _parse_metadata entirely (count==0) is not penalised; only the regression
        of calling it more than once (count>=2) fails.
        """
        original_parse = TaskInterceptor._parse_metadata
        call_count: list[int] = [0]

        def counting_parse(kwargs):
            call_count[0] += 1
            return original_parse(kwargs)

        monkeypatch.setattr(TaskInterceptor, '_parse_metadata', staticmethod(counting_parse))

        kwargs = {'title': 'Foo', 'metadata': {'files_to_modify': ['a.py']}}
        candidate = TaskInterceptor._build_candidate(kwargs)

        assert candidate is not None
        assert candidate.files_to_modify == ['a.py']
        assert call_count[0] <= 1, (
            f'_parse_metadata should be called at most once by _build_candidate, '
            f'but was called {call_count[0]} time(s)'
        )


# ---------------------------------------------------------------------------
# Regression tests — prompt-only fallback must also scan metadata files
# ---------------------------------------------------------------------------


class TestPathGuardFallbackMetadataFiles:
    """Regression tests: prompt-only path-guard also scans metadata files.

    4 parametrised cases (2 meta_key × 2 endpoint) verify that hiding a
    dark-factory path inside metadata['files'] or metadata['files_to_modify']
    cannot bypass the path-scope guard when the free-text fields are clean.
    """

    @pytest.mark.parametrize('meta_key', ['files', 'files_to_modify'])
    @pytest.mark.asyncio
    async def test_submit_task_fallback_rejects_dark_factory_path_in_metadata(
        self,
        meta_key,
        interceptor_with_store,
        ticket_store,
        taskmaster,
    ):
        """prompt-only submit_task with dark-factory path ONLY in metadata[meta_key]
        must be rejected even though all free-text fields are clean.
        """
        try:
            result = await interceptor_with_store.submit_task(
                project_root='/some-other-project',
                prompt='Generic refactor',  # no dark-factory path here
                # Deliberately NO title — forces _build_candidate → None → fallback
                metadata={meta_key: ['orchestrator/harness.py']},
            )
        finally:
            await _cancel_interceptor_workers(interceptor_with_store)

        assert isinstance(result, dict)
        assert result.get('error_type') == 'DarkFactoryPathScopeViolation', (
            f'meta_key={meta_key!r}: expected DarkFactoryPathScopeViolation, got: {result}'
        )
        assert 'orchestrator/' in result.get('matched_paths', []), (
            f'meta_key={meta_key!r}: expected orchestrator/ in matched_paths: {result}'
        )

        # Ticket store must have zero rows (guard fires before persist)
        db = ticket_store._db
        assert db is not None
        cursor = await db.execute('SELECT COUNT(*) FROM tickets')
        row = await cursor.fetchone()
        assert row[0] == 0, f'meta_key={meta_key!r}: expected 0 tickets in store, found {row[0]}'

        # Taskmaster backend must never have been called
        taskmaster.add_task.assert_not_called()


# ---------------------------------------------------------------------------
# Negative-control — clean metadata files must NOT be rejected
# ---------------------------------------------------------------------------


class TestPathGuardFallbackMetadataFilesNegativeControl:
    """Negative-control: prompt-only submissions with clean metadata are allowed.

    These tests verify the absence of false positives: if a future refactor
    accidentally short-circuits on metadata presence rather than content, they
    will fail loudly.
    """

    @pytest.mark.asyncio
    async def test_submit_task_allows_clean_metadata_files_in_other_project(
        self,
        interceptor_with_store,
        taskmaster,
    ):
        """prompt-only submit_task with non-dark-factory paths in metadata
        must NOT be rejected — only dark-factory paths should trigger the guard.
        """
        try:
            result = await interceptor_with_store.submit_task(
                project_root='/some-other-project',
                prompt='Refactor foo/bar.py routing',
                # No title — prompt-only path
                metadata={'files_to_modify': ['foo/bar.py', 'src/baz.py']},
            )
        finally:
            await _cancel_interceptor_workers(interceptor_with_store)

        assert isinstance(result, dict)
        ticket_id = result.get('ticket', '')
        assert ticket_id.startswith('tkt_'), f'Expected ticket id starting with tkt_, got: {result}'
        assert 'error_type' not in result, (
            f'Should not have error_type for clean metadata files: {result}'
        )


# ---------------------------------------------------------------------------
# Unit tests for _path_guard_or_skip helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPathGuardOrSkip:
    """Unit tests for TaskInterceptor._path_guard_or_skip (now async).

    Verifies the helper's contract: dark_factory short-circuit, lazy-build
    of candidate, pass-through of a pre-built candidate, and error propagation.
    No adjudicator wired — behaviour is identical to the previous sync version.
    """

    # -- Case 1 -----------------------------------------------------------
    async def test_path_guard_or_skip_returns_none_for_dark_factory_project(
        self,
        interceptor,
        monkeypatch,
    ):
        """dark_factory short-circuit (back-compat: no registry configured):
        returns None without calling _build_candidate or _path_guard_check.
        """
        from fused_memory.middleware.path_scope_guard import PathGuardVerdict

        build_calls: list = []
        guard_calls: list = []

        def fake_build(kwargs):
            build_calls.append(kwargs)
            return None

        def fake_check(self, candidate, kwargs, project_id):
            guard_calls.append((candidate, kwargs, project_id))
            return PathGuardVerdict(outcome='ok', project_id=project_id)

        monkeypatch.setattr(TaskInterceptor, '_build_candidate', staticmethod(fake_build))
        monkeypatch.setattr(TaskInterceptor, '_path_guard_check', fake_check)

        result = await interceptor._path_guard_or_skip(
            {'title': 'Edit orchestrator/harness.py'},
            '/home/leo/src/dark-factory',
            'dark_factory',
        )

        assert result is None
        assert build_calls == [], '_build_candidate must NOT be called for dark_factory'
        assert guard_calls == [], '_path_guard_check must NOT be called for dark_factory'

    # -- Case 2 -----------------------------------------------------------
    async def test_path_guard_or_skip_lazy_builds_candidate_when_unset(
        self,
        interceptor,
        monkeypatch,
    ):
        """When no candidate is supplied and project is non-dark_factory, the
        helper builds a candidate via _build_candidate and passes it to
        _path_guard_check.
        """
        from fused_memory.middleware.path_scope_guard import PathGuardVerdict
        from fused_memory.middleware.task_curator import CandidateTask

        built = CandidateTask(
            title='Generic refactor',
            description='',
            details='',
            files_to_modify=[],
            priority='medium',
        )
        build_calls: list = []
        guard_calls: list = []

        def fake_build(kwargs):
            build_calls.append(kwargs)
            return built

        def fake_check(self, candidate, kwargs, project_id):
            guard_calls.append((candidate, kwargs, project_id))
            return PathGuardVerdict(outcome='ok', project_id=project_id)

        monkeypatch.setattr(TaskInterceptor, '_build_candidate', staticmethod(fake_build))
        monkeypatch.setattr(TaskInterceptor, '_path_guard_check', fake_check)

        kwargs = {'title': 'Generic refactor'}
        result = await interceptor._path_guard_or_skip(
            kwargs,
            '/some/project_root',
            'some_other_project',
        )

        assert result is None
        assert len(build_calls) == 1, (
            f'Expected _build_candidate called once, got {len(build_calls)}'
        )
        assert build_calls[0] is kwargs
        assert len(guard_calls) == 1, (
            f'Expected _path_guard_check called once, got {len(guard_calls)}'
        )
        assert guard_calls[0][0] is built

    # -- Case 3 -----------------------------------------------------------
    async def test_path_guard_or_skip_uses_provided_candidate(
        self,
        interceptor,
        monkeypatch,
    ):
        """When a pre-built candidate is supplied, _build_candidate is NOT called;
        _path_guard_check is called with the supplied candidate.
        """
        from fused_memory.middleware.path_scope_guard import PathGuardVerdict
        from fused_memory.middleware.task_curator import CandidateTask

        sentinel = CandidateTask(
            title='Sentinel',
            description='',
            details='',
            files_to_modify=[],
            priority='medium',
        )
        build_calls: list = []
        guard_calls: list = []

        def fake_build(kwargs):
            build_calls.append(kwargs)
            return None

        def fake_check(self, candidate, kwargs, project_id):
            guard_calls.append((candidate, kwargs, project_id))
            return PathGuardVerdict(outcome='ok', project_id=project_id)

        monkeypatch.setattr(TaskInterceptor, '_build_candidate', staticmethod(fake_build))
        monkeypatch.setattr(TaskInterceptor, '_path_guard_check', fake_check)

        kwargs = {'title': 'Generic refactor'}
        result = await interceptor._path_guard_or_skip(
            kwargs,
            '/some/project_root',
            'some_other_project',
            candidate=sentinel,
        )

        assert result is None
        assert build_calls == [], '_build_candidate must NOT be called when candidate is supplied'
        assert len(guard_calls) == 1, (
            f'Expected _path_guard_check called once, got {len(guard_calls)}'
        )
        assert guard_calls[0][0] is sentinel

    # -- Case 4 -----------------------------------------------------------
    async def test_path_guard_or_skip_propagates_rejection(
        self,
        interceptor,
        monkeypatch,
    ):
        """When _path_guard_check returns a rejection verdict, the helper
        returns its to_error_dict() output.
        """
        from fused_memory.middleware.path_scope_guard import PathGuardVerdict

        verdict = PathGuardVerdict(
            outcome='rejection',
            project_id='some_other_project',
            matched_paths=('orchestrator/',),
            suggested_project='dark_factory',
        )

        def fake_build(kwargs):
            return None

        def fake_check(self, candidate, kwargs, project_id):
            return verdict

        monkeypatch.setattr(TaskInterceptor, '_build_candidate', staticmethod(fake_build))
        monkeypatch.setattr(TaskInterceptor, '_path_guard_check', fake_check)

        result = await interceptor._path_guard_or_skip(
            {'prompt': 'something'},
            '/some/project_root',
            'some_other_project',
        )
        assert result is not None
        assert result.get('error_type') == 'DarkFactoryPathScopeViolation'
        assert result.get('matched_paths') == ['orchestrator/']
        assert result.get('suggested_project') == 'dark_factory'


# ---------------------------------------------------------------------------
# Multi-project routing — registry + escalator wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMultiProjectRoutingWiring:
    """Verify that when prefix_registry + scope_violation_escalator are wired,
    the path guard runs for every project (including dark_factory) and a
    rejection fires the escalator with the expected payload.

    All tests are async (no adjudicator wired — behaviour identical to before).
    """

    async def test_registry_supersedes_dark_factory_short_circuit(
        self,
        interceptor,
        monkeypatch,
        tmp_path,
    ):
        """With a registry configured, dark_factory submissions are also
        guarded — a reify path landing in dark_factory triggers rejection."""
        from fused_memory.middleware.path_scope_guard import PathGuardVerdict
        from fused_memory.middleware.project_prefix_registry import (
            ProjectPrefixRegistry,
        )

        # Build a registry covering both projects.
        (tmp_path / 'reify').mkdir()
        (tmp_path / 'reify' / 'crates').mkdir()
        (tmp_path / 'dark-factory').mkdir()
        (tmp_path / 'dark-factory' / 'fused-memory').mkdir()
        registry = ProjectPrefixRegistry.from_roots(
            [
                str(tmp_path / 'reify'),
                str(tmp_path / 'dark-factory'),
            ]
        )
        interceptor._prefix_registry = registry

        check_calls: list = []

        def fake_check(self, candidate, kwargs, project_id):
            check_calls.append(project_id)
            return PathGuardVerdict(outcome='ok', project_id=project_id)

        monkeypatch.setattr(TaskInterceptor, '_path_guard_check', fake_check)

        result = await interceptor._path_guard_or_skip(
            {'title': 'Edit something'},
            str(tmp_path / 'dark-factory'),
            'dark_factory',
        )
        # Guard ran for dark_factory (no short-circuit) and returned ok.
        assert result is None
        assert check_calls == ['dark_factory']

    async def test_rejection_fires_scope_violation_escalator(
        self,
        interceptor,
        monkeypatch,
        tmp_path,
    ):
        """A path-guard rejection invokes scope_violation_escalator.report_rejection
        with project_root, matched_paths, and suggested_project from the verdict.
        """
        from fused_memory.middleware.path_scope_guard import PathGuardVerdict
        from fused_memory.middleware.project_prefix_registry import (
            ProjectPrefixRegistry,
        )

        # Set up a registry so the new path runs.
        (tmp_path / 'reify').mkdir()
        (tmp_path / 'reify' / 'crates').mkdir()
        (tmp_path / 'dark-factory').mkdir()
        (tmp_path / 'dark-factory' / 'fused-memory').mkdir()
        registry = ProjectPrefixRegistry.from_roots(
            [
                str(tmp_path / 'reify'),
                str(tmp_path / 'dark-factory'),
            ]
        )
        interceptor._prefix_registry = registry

        # Spy escalator.
        escalator_calls: list = []

        class FakeEscalator:
            def report_rejection(self, **kwargs):
                escalator_calls.append(kwargs)
                return 'esc-task-path-guard-1'

        interceptor._scope_violation_escalator = FakeEscalator()

        # Force a rejection verdict from the check function.
        verdict = PathGuardVerdict(
            outcome='rejection',
            project_id='reify',
            matched_paths=('fused-memory/',),
            suggested_project='dark_factory',
        )
        monkeypatch.setattr(
            TaskInterceptor,
            '_path_guard_check',
            lambda self, c, k, p: verdict,
        )

        result = await interceptor._path_guard_or_skip(
            {'title': 'Edit fused-memory/X'},
            str(tmp_path / 'reify'),
            'reify',
        )
        # Rejection error dict is still returned to the caller.
        assert result is not None
        assert result['error_type'] == 'DarkFactoryPathScopeViolation'
        # Escalator was called exactly once with the routing context.
        assert len(escalator_calls) == 1
        call = escalator_calls[0]
        assert call['project_root'] == str(tmp_path / 'reify')
        assert call['project_id'] == 'reify'
        assert call['matched_paths'] == ('fused-memory/',)
        assert call['suggested_project'] == 'dark_factory'
        # suggested_root resolved from registry.
        assert call['suggested_root'] == str((tmp_path / 'dark-factory').resolve())

    async def test_escalator_failure_swallowed(
        self,
        interceptor,
        monkeypatch,
        tmp_path,
    ):
        """An escalator that raises must NOT turn the rejection into an exception."""
        from fused_memory.middleware.path_scope_guard import PathGuardVerdict
        from fused_memory.middleware.project_prefix_registry import (
            ProjectPrefixRegistry,
        )

        (tmp_path / 'reify').mkdir()
        (tmp_path / 'reify' / 'crates').mkdir()
        registry = ProjectPrefixRegistry.from_roots([str(tmp_path / 'reify')])
        interceptor._prefix_registry = registry

        class BoomEscalator:
            def report_rejection(self, **kwargs):
                raise RuntimeError('boom')

        interceptor._scope_violation_escalator = BoomEscalator()
        verdict = PathGuardVerdict(
            outcome='rejection',
            project_id='other',
            matched_paths=('crates/',),
            suggested_project='reify',
        )
        monkeypatch.setattr(
            TaskInterceptor,
            '_path_guard_check',
            lambda self, c, k, p: verdict,
        )

        # Must not raise.
        result = await interceptor._path_guard_or_skip(
            {'title': 'crates/widget'},
            '/foo',
            'other',
        )
        assert result is not None
        assert result['error_type'] == 'DarkFactoryPathScopeViolation'

    @pytest.mark.asyncio
    async def test_stage2_no_hit_adjudicator_not_called(
        self,
        interceptor,
        monkeypatch,
        tmp_path,
    ):
        """NO-HIT hot path: when Stage-1 returns OK, the adjudicator is never called."""
        from unittest.mock import AsyncMock

        from fused_memory.middleware.path_scope_guard import PathGuardVerdict
        from fused_memory.middleware.project_prefix_registry import (
            ProjectPrefixRegistry,
        )

        (tmp_path / 'reify').mkdir()
        (tmp_path / 'reify' / 'crates').mkdir()
        registry = ProjectPrefixRegistry.from_roots([str(tmp_path / 'reify')])
        interceptor._prefix_registry = registry

        # Fake adjudicator with a spy adjudicate method.
        fake_adj = AsyncMock()
        fake_adj.adjudicate = AsyncMock()
        interceptor._path_scope_adjudicator = fake_adj

        # Force OK verdict — no heuristic hit.
        monkeypatch.setattr(
            TaskInterceptor,
            '_path_guard_check',
            lambda self, c, k, p: PathGuardVerdict(outcome='ok', project_id=p),
        )

        result = await interceptor._path_guard_or_skip(
            {'title': 'Normal task'},
            str(tmp_path / 'reify'),
            'reify',
        )
        # Allowed — no error dict.
        assert result is None
        # Adjudicator must NOT have been called on the no-hit path.
        fake_adj.adjudicate.assert_not_called()

    @pytest.mark.asyncio
    async def test_stage2_hit_reject_returns_error_with_llm_reason(
        self,
        interceptor,
        monkeypatch,
        tmp_path,
    ):
        """HIT + REJECT: adjudicator confirms misroute → error dict returned,
        escalator called once carrying the adjudicator's llm_reason."""
        from unittest.mock import AsyncMock

        from fused_memory.middleware.path_scope_adjudicator import AdjudicationVerdict
        from fused_memory.middleware.path_scope_guard import PathGuardVerdict
        from fused_memory.middleware.project_prefix_registry import (
            ProjectPrefixRegistry,
        )

        (tmp_path / 'reify').mkdir()
        (tmp_path / 'reify' / 'crates').mkdir()
        (tmp_path / 'dark-factory').mkdir()
        (tmp_path / 'dark-factory' / 'fused-memory').mkdir()
        registry = ProjectPrefixRegistry.from_roots(
            [str(tmp_path / 'reify'), str(tmp_path / 'dark-factory')]
        )
        interceptor._prefix_registry = registry

        reject_verdict = AdjudicationVerdict(
            verdict='reject',
            reason='genuine misroute — task edits orchestrator/harness.py',
            llm_used=True,
        )
        fake_adj = AsyncMock()
        fake_adj.adjudicate = AsyncMock(return_value=reject_verdict)
        interceptor._path_scope_adjudicator = fake_adj

        escalator_calls: list = []

        class SpyEscalator:
            def report_rejection(self, **kwargs):
                escalator_calls.append(kwargs)

        interceptor._scope_violation_escalator = SpyEscalator()

        verdict = PathGuardVerdict(
            outcome='rejection',
            project_id='reify',
            matched_paths=('fused-memory/',),
            suggested_project='dark_factory',
        )
        monkeypatch.setattr(
            TaskInterceptor,
            '_path_guard_check',
            lambda self, c, k, p: verdict,
        )

        result = await interceptor._path_guard_or_skip(
            {'title': 'Edit fused-memory/X'},
            str(tmp_path / 'reify'),
            'reify',
        )
        # Rejection error dict returned.
        assert result is not None
        assert result['error_type'] == 'DarkFactoryPathScopeViolation'
        # Escalator called exactly once with the LLM reason.
        assert len(escalator_calls) == 1
        call = escalator_calls[0]
        assert call['llm_reason'] == 'genuine misroute — task edits orchestrator/harness.py'

    @pytest.mark.asyncio
    async def test_stage2_hit_failsafe_rejects_and_escalates_with_reason(
        self,
        interceptor,
        monkeypatch,
        tmp_path,
    ):
        """HIT + FAIL-SAFE (uncertain/failed): LLM outage never lets misroute through;
        escalator is called with llm_reason from the fail-safe verdict."""
        from unittest.mock import AsyncMock

        from fused_memory.middleware.path_scope_adjudicator import AdjudicationVerdict
        from fused_memory.middleware.path_scope_guard import PathGuardVerdict
        from fused_memory.middleware.project_prefix_registry import (
            ProjectPrefixRegistry,
        )

        (tmp_path / 'reify').mkdir()
        (tmp_path / 'reify' / 'crates').mkdir()
        registry = ProjectPrefixRegistry.from_roots([str(tmp_path / 'reify')])
        interceptor._prefix_registry = registry

        # Fail-safe verdict (uncertain + failed — simulates breaker-open / hang).
        failsafe_verdict = AdjudicationVerdict(
            verdict='uncertain',
            reason='breaker-open',
            failed=True,
            llm_used=False,
        )
        fake_adj = AsyncMock()
        fake_adj.adjudicate = AsyncMock(return_value=failsafe_verdict)
        interceptor._path_scope_adjudicator = fake_adj

        escalator_calls: list = []

        class SpyEscalator:
            def report_rejection(self, **kwargs):
                escalator_calls.append(kwargs)

        interceptor._scope_violation_escalator = SpyEscalator()

        verdict = PathGuardVerdict(
            outcome='rejection',
            project_id='other',
            matched_paths=('crates/',),
            suggested_project='reify',
        )
        monkeypatch.setattr(
            TaskInterceptor,
            '_path_guard_check',
            lambda self, c, k, p: verdict,
        )

        result = await interceptor._path_guard_or_skip(
            {'title': 'crates/widget'},
            '/foo',
            'other',
        )
        # Guard preserved — misroute rejected even with LLM outage.
        assert result is not None
        assert result['error_type'] == 'DarkFactoryPathScopeViolation'
        # Escalated once, carrying the fail-safe reason.
        assert len(escalator_calls) == 1
        assert escalator_calls[0]['llm_reason'] == 'breaker-open'

    @pytest.mark.asyncio
    async def test_stage2_hit_allow_permits_creation_no_escalation(
        self,
        interceptor,
        monkeypatch,
        tmp_path,
    ):
        """HIT + ALLOW: adjudicator confident allow → creation permitted, no escalation."""
        from unittest.mock import AsyncMock

        from fused_memory.middleware.path_scope_adjudicator import AdjudicationVerdict
        from fused_memory.middleware.path_scope_guard import PathGuardVerdict
        from fused_memory.middleware.project_prefix_registry import (
            ProjectPrefixRegistry,
        )

        (tmp_path / 'reify').mkdir()
        (tmp_path / 'reify' / 'crates').mkdir()
        (tmp_path / 'dark-factory').mkdir()
        (tmp_path / 'dark-factory' / 'fused-memory').mkdir()
        registry = ProjectPrefixRegistry.from_roots(
            [str(tmp_path / 'reify'), str(tmp_path / 'dark-factory')]
        )
        interceptor._prefix_registry = registry

        # Adjudicator returns confident allow.
        allow_verdict = AdjudicationVerdict(
            verdict='allow',
            reason='incidental example mention in description',
            llm_used=True,
        )
        fake_adj = AsyncMock()
        fake_adj.adjudicate = AsyncMock(return_value=allow_verdict)
        interceptor._path_scope_adjudicator = fake_adj

        # Spy escalator — must NOT be called on allow.
        escalator_calls: list = []

        class SpyEscalator:
            def report_rejection(self, **kwargs):
                escalator_calls.append(kwargs)

        interceptor._scope_violation_escalator = SpyEscalator()

        # Force Stage-1 rejection.
        verdict = PathGuardVerdict(
            outcome='rejection',
            project_id='reify',
            matched_paths=('fused-memory/',),
            suggested_project='dark_factory',
        )
        monkeypatch.setattr(
            TaskInterceptor,
            '_path_guard_check',
            lambda self, c, k, p: verdict,
        )

        result = await interceptor._path_guard_or_skip(
            {'title': 'Task mentioning fused-memory/ as example'},
            str(tmp_path / 'reify'),
            'reify',
        )
        # Adjudicator downgraded heuristic hit → creation permitted.
        assert result is None
        # Adjudicator was called exactly once.
        fake_adj.adjudicate.assert_called_once()
        # Escalator must NOT be called when adjudicator allows.
        assert escalator_calls == []


# ─────────────────────────────────────────────────────────────────────
# planning_mode: synchronous, curator-bypassing submit_task path
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_planning_mode_returns_task_id_synchronously(
    interceptor_facade,
    taskmaster,
):
    """planning_mode=True returns task_id directly with status=deferred — no ticket."""
    result = await interceptor_facade.submit_task(
        '/project',
        title='Decomposed task',
        planning_mode=True,
    )
    assert result == {
        'task_id': '2',
        'status': 'deferred',
        'planning_mode': True,
    }
    taskmaster.add_task.assert_called_once()
    # The row is created directly in deferred via add_task(status=...);
    # there is no separate set_task_status flip to observe (and race against).
    assert taskmaster.add_task.call_args.kwargs.get('status') == 'deferred'
    taskmaster.set_task_status.assert_not_called()


@pytest.mark.asyncio
async def test_planning_mode_persists_human_decomposed_metadata(
    interceptor_facade,
    taskmaster,
):
    """planning_mode injects human_decomposed=True into the metadata sent to tm.add_task."""
    await interceptor_facade.submit_task(
        '/project',
        title='X',
        planning_mode=True,
    )
    metadata_arg = taskmaster.add_task.call_args.kwargs.get('metadata')
    assert metadata_arg is not None
    assert json.loads(metadata_arg) == {'human_decomposed': True}


@pytest.mark.asyncio
async def test_planning_mode_merges_caller_metadata(
    interceptor_facade,
    taskmaster,
):
    """Caller-supplied metadata is preserved alongside human_decomposed=True."""
    await interceptor_facade.submit_task(
        '/project',
        title='X',
        planning_mode=True,
        metadata={'source': 'planning-session', 'files': ['m1', 'm2']},
    )
    decoded = json.loads(taskmaster.add_task.call_args.kwargs['metadata'])
    assert decoded == {
        'source': 'planning-session',
        'files': ['m1', 'm2'],
        'human_decomposed': True,
    }


@pytest.mark.asyncio
async def test_planning_mode_accepts_metadata_json_string(
    interceptor_facade,
    taskmaster,
):
    """JSON-string metadata is decoded, merged, and re-encoded."""
    await interceptor_facade.submit_task(
        '/project',
        title='X',
        planning_mode=True,
        metadata='{"escalation_id": "esc-1"}',
    )
    decoded = json.loads(taskmaster.add_task.call_args.kwargs['metadata'])
    assert decoded == {'escalation_id': 'esc-1', 'human_decomposed': True}


@pytest.mark.asyncio
async def test_planning_mode_rejects_invalid_metadata_string(
    interceptor_facade,
):
    """Non-JSON metadata string returns a structured ValidationError."""
    result = await interceptor_facade.submit_task(
        '/project',
        title='X',
        planning_mode=True,
        metadata='{not json',
    )
    assert result.get('error_type') == 'ValidationError'
    assert 'JSON-decode' in result['error']


@pytest.mark.asyncio
async def test_planning_mode_rejects_non_object_metadata(
    interceptor_facade,
):
    """JSON metadata that decodes to a non-dict is rejected."""
    result = await interceptor_facade.submit_task(
        '/project',
        title='X',
        planning_mode=True,
        metadata='[1,2,3]',
    )
    assert result.get('error_type') == 'ValidationError'
    assert 'object' in result['error']


@pytest.mark.asyncio
async def test_planning_mode_emits_task_created_event(
    interceptor_facade,
    event_buffer,
):
    """planning_mode emits a task_created event into the buffer."""
    await interceptor_facade.submit_task(
        '/project',
        title='X',
        planning_mode=True,
    )
    stats = await event_buffer.get_buffer_stats('project')
    assert stats['size'] == 1


@pytest.mark.asyncio
async def test_planning_mode_skips_curator_worker(
    interceptor_facade,
    monkeypatch,
):
    """planning_mode never triggers the per-project curator worker."""
    started: list[str] = []
    original = TaskInterceptor._start_worker_if_needed

    def spy(self, project_id):
        started.append(project_id)
        return original(self, project_id)

    monkeypatch.setattr(TaskInterceptor, '_start_worker_if_needed', spy)

    await interceptor_facade.submit_task(
        '/project',
        title='X',
        planning_mode=True,
    )
    assert started == [], (
        f'planning_mode must not start the curator worker; got starts for {started}'
    )


@pytest.mark.asyncio
async def test_planning_mode_default_false_preserves_two_phase(
    interceptor_facade,
):
    """planning_mode defaults to False; submit_task still returns a ticket."""
    result = await interceptor_facade.submit_task('/project', title='X')
    assert 'ticket' in result, f'expected ticket-shape result, got {result!r}'
    assert 'task_id' not in result


@pytest.mark.asyncio
async def test_planning_mode_add_task_failure_returns_error(
    interceptor_facade,
    taskmaster,
):
    """If tm.add_task itself fails, planning_mode returns a structured error dict."""
    taskmaster.add_task.side_effect = RuntimeError('add_task wire failure')
    result = await interceptor_facade.submit_task(
        '/project',
        title='X',
        planning_mode=True,
    )
    assert result.get('error_type') == 'RuntimeError'
    assert 'add_task wire failure' in result['error']
    taskmaster.set_task_status.assert_not_called()


# ─────────────────────────────────────────────────────────────────────
# _looks_like_task_id helper
# ─────────────────────────────────────────────────────────────────────


def test_looks_like_task_id_accepts_numeric_strings():
    from fused_memory.middleware.task_interceptor import _looks_like_task_id

    assert _looks_like_task_id('42')
    assert _looks_like_task_id('  42  ')
    assert _looks_like_task_id('0')


def test_looks_like_task_id_accepts_int():
    from fused_memory.middleware.task_interceptor import _looks_like_task_id

    assert _looks_like_task_id(42)
    assert _looks_like_task_id(0)


def test_looks_like_task_id_rejects_bool():
    from fused_memory.middleware.task_interceptor import _looks_like_task_id

    assert not _looks_like_task_id(True)
    assert not _looks_like_task_id(False)


def test_looks_like_task_id_rejects_negative_and_non_numeric():
    from fused_memory.middleware.task_interceptor import _looks_like_task_id

    assert not _looks_like_task_id(-1)
    assert not _looks_like_task_id('-1')
    assert not _looks_like_task_id('abc')
    assert not _looks_like_task_id('')
    assert not _looks_like_task_id('   ')
    assert not _looks_like_task_id(None)
    assert not _looks_like_task_id('tkt_abc')
    assert not _looks_like_task_id('1.5')


# ─────────────────────────────────────────────────────────────────────
# planning_mode end-to-end: decompose batch → set deps → commit
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_planning_mode_end_to_end_batch_with_dependencies(tmp_path):
    """Full planner flow: submit 5 tasks in planning_mode, then commit-flip them.

    Exercises: synchronous returns, no curator interaction, deferred status
    while batch is in flight, set_task_status CSV-bulk path on commit.  Uses
    a real TicketStore + EventBuffer so the journaling and ticket-persistence
    code paths run; mock Taskmaster tracks state in-memory.
    """
    from fused_memory.middleware.ticket_store import TicketStore

    # Stateful mock: tracks status per task id.
    statuses: dict[str, str] = {}
    next_id = [100]

    async def fake_add_task(**kwargs):
        tid = str(next_id[0])
        next_id[0] += 1
        statuses[tid] = kwargs.get('status', 'pending')
        return {'id': tid, 'title': kwargs.get('title') or 'untitled'}

    async def fake_set_status(task_id, status, project_root, tag=None):
        statuses[task_id] = status
        return {'success': True, 'task_id': task_id, 'status': status}

    async def fake_get_task(task_id, project_root, tag=None):
        return {'id': task_id, 'status': statuses.get(task_id, 'pending'), 'title': 't'}

    tm = AsyncMock()
    tm.add_task = AsyncMock(side_effect=fake_add_task)
    tm.set_task_status = AsyncMock(side_effect=fake_set_status)
    tm.get_task = AsyncMock(side_effect=fake_get_task)
    tm.get_tasks = AsyncMock(return_value={'tasks': []})
    tm.update_task = AsyncMock(return_value={'success': True})

    buf = EventBuffer(db_path=tmp_path / 'e2e_eb.db', buffer_size_threshold=100)
    await buf.initialize()
    store = TicketStore(tmp_path / 'e2e_tickets.db')
    await store.initialize()

    # Spy on store.submit so we can assert it was never called.
    submit_calls: list[dict] = []
    original_submit = store.submit

    async def submit_spy(**kwargs):
        submit_calls.append(kwargs)
        return await original_submit(**kwargs)

    store.submit = submit_spy  # type: ignore[method-assign]

    interceptor = TaskInterceptor(tm, None, buf, ticket_store=store)
    try:
        # Submit 5 sibling tasks in planning_mode.
        task_ids: list[str] = []
        for i in range(5):
            result = await interceptor.submit_task(
                '/project',
                title=f'sibling-{i}',
                planning_mode=True,
            )
            assert result['planning_mode'] is True, result
            assert result['status'] == 'deferred', result
            task_ids.append(result['task_id'])

        # All 5 are deferred — none picked up by anyone yet.
        assert all(statuses[tid] == 'deferred' for tid in task_ids), statuses

        # Each created task carries human_decomposed=True in its metadata.
        for call in tm.add_task.call_args_list:
            metadata_arg = call.kwargs.get('metadata')
            assert metadata_arg is not None
            assert json.loads(metadata_arg).get('human_decomposed') is True

        # Commit the batch: deferred → pending via the CSV bulk path.
        commit_result = await interceptor.set_task_status(
            ','.join(task_ids),
            'pending',
            '/project',
        )
        assert commit_result['success'] is True
        # All 5 results report no error.
        per_results = commit_result['results']
        assert len(per_results) == 5
        for r in per_results:
            assert r['result'].get('error') is None, r

        # Final state: every task pending, ready for the scheduler.
        assert all(statuses[tid] == 'pending' for tid in task_ids), statuses

        # No tickets were persisted in planning mode — the store's submit()
        # was never called (verified by tracking the call count via wrapper).
        assert submit_calls == [], f'planning_mode must not persist tickets; got: {submit_calls}'
    finally:
        await store.close()
        for t in list(interceptor._worker_tasks.values()):
            if not t.done():
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t
        await buf.close()


# ─────────────────────────────────────────────────────────────────────
# Write-journal integration (Commit 7)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_journaled_write_emits_write_op_and_backend_op(
    taskmaster,
    reconciler,
    event_buffer,
    tmp_path,
):
    """A write through the interceptor with a journal wired must leave
    one ``write_op`` row and at least one ``backend_op`` row, both tagged
    with the same ``write_op_id``."""
    from fused_memory.services.write_journal import WriteJournal

    journal = WriteJournal(tmp_path / 'wj')
    await journal.initialize()
    try:
        interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)
        interceptor.set_write_journal(journal)

        # update_task is the simplest path.
        await interceptor.update_task('1', '/project', prompt='tweak')

        # Verify the rows.
        assert journal._db is not None
        async with journal._db.execute(
            "SELECT id, operation FROM write_ops WHERE operation = 'update_task'",
        ) as cur:
            wo_rows = list(await cur.fetchall())
        assert len(wo_rows) == 1, wo_rows
        write_op_id = wo_rows[0][0]

        async with journal._db.execute(
            'SELECT backend, success FROM backend_ops '
            "WHERE write_op_id = ? AND operation = 'update_task'",
            (write_op_id,),
        ) as cur:
            bo_rows = list(await cur.fetchall())
        assert len(bo_rows) >= 1, bo_rows
        # Backend label resolves from AsyncMock's class — 'unknown' is
        # acceptable, what matters is the row exists and is tied to the
        # same write_op.
        assert all(row[1] == 1 for row in bo_rows), bo_rows
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_journaled_write_logs_failure_row(
    reconciler,
    event_buffer,
    tmp_path,
):
    """A failing tm.* call still produces a write_op row plus a failed
    backend_op row — never silently disappears."""
    from fused_memory.backends.task_backend_errors import TaskmasterError
    from fused_memory.services.write_journal import WriteJournal

    failing_tm = AsyncMock()
    failing_tm.update_task = AsyncMock(
        side_effect=TaskmasterError(
            'TASKMASTER_TOOL_ERROR',
            'simulated',
        )
    )

    journal = WriteJournal(tmp_path / 'wj_fail')
    await journal.initialize()
    try:
        interceptor = TaskInterceptor(failing_tm, reconciler, event_buffer)
        interceptor.set_write_journal(journal)

        with pytest.raises(TaskmasterError):
            await interceptor.update_task('1', '/project', prompt='x')

        assert journal._db is not None
        async with journal._db.execute(
            "SELECT COUNT(*) FROM write_ops WHERE operation = 'update_task'",
        ) as cur:
            row = await cur.fetchone()
            assert row is not None
            wo_count = row[0]
        assert wo_count == 1
        async with journal._db.execute(
            "SELECT success, error FROM backend_ops WHERE operation = 'update_task'",
        ) as cur:
            bo_rows = list(await cur.fetchall())
        assert len(bo_rows) == 1
        assert bo_rows[0][0] == 0
        assert 'simulated' in (bo_rows[0][1] or '')
    finally:
        await journal.close()


# ── Tests for update_task status-kwarg rejection (defence-in-depth) ─────────


@pytest.mark.asyncio
@pytest.mark.parametrize('bad_status', ['done', 'pending', 'cancelled', 'in-progress', 'blocked'])
async def test_update_task_rejects_status_kwarg(
    interceptor,
    taskmaster,
    bad_status,
):
    """The interceptor's update_task path also rejects status=…

    Defence-in-depth alongside the same gate at the MCP tool surface.
    Closes the bypass route used to mark reify tasks done without going
    through the terminal-exit, phantom-done, and done-provenance gates.
    """
    result = await interceptor.update_task(
        task_id='1',
        project_root='/project',
        status=bad_status,
    )
    assert isinstance(result, dict)
    assert result.get('error') == 'status_via_update_task'
    assert result.get('status') == bad_status
    assert 'set_task_status' in result.get('hint', '')
    taskmaster.update_task.assert_not_called()


@pytest.mark.asyncio
async def test_update_task_status_none_passes_through(interceptor, taskmaster):
    """status=None (the metadata-only path) is unchanged — the gate only blocks non-None."""
    await interceptor.update_task(
        task_id='1',
        project_root='/project',
        status=None,
        details='x',
    )
    taskmaster.update_task.assert_called_once()


# ── Tests for set_task_status audit-metadata read-modify-write (2026-05-08) ─


@pytest.mark.asyncio
async def test_set_task_status_with_reopen_reason_preserves_metadata(
    taskmaster,
    reconciler,
    event_buffer,
):
    """Reopening a done task must NOT clobber existing metadata (files, memory_hints).

    Bug: the audit-metadata write at task_interceptor.py:681-694 used
    update_task(metadata=json.dumps({reopen_reason, …}), append=False) — that
    overwrites the entire metadata blob, dropping memory_hints and files.
    Fix: read-modify-write so audit fields merge with existing metadata.
    """
    taskmaster.get_task = AsyncMock(
        return_value={
            'id': '7',
            'status': 'done',
            'title': 'T',
            'metadata': {
                'files': ['a.py', 'b.py'],
                'memory_hints': {'queries': ['ctx']},
                'spawned_from': '5',
            },
        }
    )
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status(
        '7',
        'pending',
        '/project',
        reopen_reason='manual reopen',
    )

    assert 'error' not in result
    taskmaster.update_task.assert_called_once()
    persisted = json.loads(taskmaster.update_task.call_args.kwargs['metadata'])
    # Audit fields are written.
    assert persisted['reopen_reason'] == 'manual reopen'
    assert persisted['reopen_from'] == 'done'
    assert 'reopen_at' in persisted
    # Prior metadata is preserved.
    assert persisted['files'] == ['a.py', 'b.py']
    assert persisted['memory_hints'] == {'queries': ['ctx']}
    assert persisted['spawned_from'] == '5'


@pytest.mark.asyncio
async def test_set_task_status_done_with_provenance_preserves_metadata(
    taskmaster,
    reconciler,
    event_buffer,
    tmp_path,
):
    """Marking done with done_provenance must NOT clobber existing metadata."""
    sha = _init_git_repo(tmp_path)
    # Create the declared file so the phantom-done gate doesn't trip on it.
    (tmp_path / 'x.py').write_text('# shipped\n')
    taskmaster.get_task = AsyncMock(
        return_value={
            'id': '9',
            'status': 'in-progress',
            'title': 'T',
            'metadata': {
                'files': ['x.py'],
                'memory_hints': {'queries': ['hint']},
            },
        }
    )
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status(
        '9',
        'done',
        str(tmp_path),
        done_provenance={'kind': 'merged', 'commit': sha},
    )

    assert 'error' not in result
    taskmaster.update_task.assert_called_once()
    persisted = json.loads(taskmaster.update_task.call_args.kwargs['metadata'])
    assert persisted['done_provenance']['kind'] == 'merged'
    assert persisted['done_provenance']['commit'] == sha
    # Prior metadata is preserved.
    assert persisted['files'] == ['x.py']
    assert persisted['memory_hints'] == {'queries': ['hint']}


# ── task-1184: interceptor_write_succeeded helper contract ──


class TestInterceptorWriteSucceeded:
    """Unit tests for the module-level ``interceptor_write_succeeded(resp)`` helper.

    The helper centralises the success/failure contract for all three rejection-dict
    shapes produced by TaskInterceptor gates:
      • ``_reject_status_in_update_task`` → ``{'success': False, 'error': '<code>', …}``
      • ``_reject_done_provenance_in_update_metadata`` → same shape
      • ``BacklogVerdict.to_error_dict()`` → ``{'error': '<msg>', 'error_type': '<code>', …}``
        (no ``success`` key)
    """

    def _fn(self):
        from fused_memory.middleware.task_interceptor import interceptor_write_succeeded

        return interceptor_write_succeeded

    def test_explicit_success_true(self):
        """{'success': True} → True."""
        assert self._fn()({'success': True}) is True

    def test_explicit_success_true_with_extra_keys(self):
        """{'success': True, 'id': '1.1'} → True (extra keys, no error key → success)."""
        assert self._fn()({'success': True, 'id': '1.1'}) is True

    def test_empty_dict_is_success(self):
        """{} → True (defaults: success=True, error=None — some fixtures use bare {})."""
        assert self._fn()({}) is True

    def test_reject_status_via_update_task(self):
        """_reject_status_in_update_task shape → False."""
        resp = {
            'success': False,
            'error': 'status_via_update_task',
            'task_id': '1',
            'status': 'done',
            'hint': 'use set_task_status',
        }
        assert self._fn()(resp) is False

    def test_reject_done_provenance_via_update_task(self):
        """_reject_done_provenance_in_update_metadata shape → False."""
        resp = {
            'success': False,
            'error': 'done_provenance_via_update_task',
            'task_id': '1',
            'hint': 'use set_task_status',
        }
        assert self._fn()(resp) is False

    def test_backlog_verdict_error_dict(self):
        """BacklogVerdict.to_error_dict() shape → False (no 'success' key, has 'error')."""
        resp = {
            'error': 'ReconciliationBacklogExceeded: backlog 600 > threshold 500',
            'error_type': 'ReconciliationBacklogExceeded',
            'backlog': 600,
            'threshold': 500,
            'project_id': 'test-project',
        }
        assert self._fn()(resp) is False

    def test_none_response_is_failure(self):
        """None → False (defensive: non-dict must never be treated as success)."""
        assert self._fn()(None) is False

    def test_string_response_is_failure(self):
        """Unexpected string → False."""
        assert self._fn()('unexpected string') is False

    def test_list_response_is_failure(self):
        """[] → False."""
        assert self._fn()([]) is False


# ── Tests for pre-done hook gate ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_task_status_done_skips_predone_hook_when_env_unset(
    taskmaster, reconciler, event_buffer, monkeypatch
):
    """When FUSED_MEMORY_PREDONE_HOOK_PROJECT is unset, done transition succeeds normally."""
    monkeypatch.delenv('FUSED_MEMORY_PREDONE_HOOK_PROJECT', raising=False)
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status('1', 'done', '/project')

    assert 'error' not in result
    taskmaster.set_task_status.assert_called_once()


@pytest.mark.asyncio
async def test_set_task_status_done_rejected_by_predone_hook(
    taskmaster, reconciler, event_buffer, monkeypatch, tmp_path
):
    """When hook exits non-zero, the done transition is refused and tm.set_task_status is not called."""
    # Derive env var key from tmp_path basename (e.g. test_set_task_status_done_rejected0)
    project_id_upper = resolve_project_id(str(tmp_path)).upper()
    monkeypatch.setenv(f'FUSED_MEMORY_PREDONE_HOOK_{project_id_upper}', '/bin/false')
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status('1', 'done', str(tmp_path))

    assert result['success'] is False
    assert result['error'] == 'pre_done_hook_rejected'
    assert result['task_id'] == '1'
    taskmaster.set_task_status.assert_not_called()
    reconciler.reconcile_task.assert_not_called()


@pytest.mark.asyncio
async def test_set_task_status_done_passes_when_predone_hook_succeeds(
    taskmaster, reconciler, event_buffer, monkeypatch, tmp_path
):
    """When hook exits 0, the done transition proceeds and taskmaster.set_task_status is called."""
    project_id_upper = resolve_project_id(str(tmp_path)).upper()
    monkeypatch.setenv(f'FUSED_MEMORY_PREDONE_HOOK_{project_id_upper}', '/bin/true')
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status('1', 'done', str(tmp_path))

    assert 'error' not in result
    taskmaster.set_task_status.assert_called_once()


@pytest.mark.asyncio
async def test_predone_hook_only_fires_on_done_transition(
    taskmaster, reconciler, event_buffer, monkeypatch, tmp_path
):
    """The pre-done hook must NOT fire for non-done transitions (blocked, in-progress).

    Belt-and-braces: even when the hook is /bin/false, blocked and in-progress
    transitions must succeed because the gate is strictly 'done'-only.
    """
    project_id_upper = resolve_project_id(str(tmp_path)).upper()
    monkeypatch.setenv(f'FUSED_MEMORY_PREDONE_HOOK_{project_id_upper}', '/bin/false')
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    # blocked: default fixture has status='pending', so pending→blocked is fine
    taskmaster.get_task = AsyncMock(
        return_value={'id': '1', 'status': 'pending', 'title': 'Test Task'}
    )
    result_blocked = await interceptor.set_task_status('1', 'blocked', str(tmp_path))
    assert 'error' not in result_blocked, (
        f'blocked transition should not be rejected: {result_blocked}'
    )

    # in-progress: go pending→in-progress (no reset_mock — count accumulates)
    taskmaster.get_task = AsyncMock(
        return_value={'id': '1', 'status': 'pending', 'title': 'Test Task'}
    )
    result_inprog = await interceptor.set_task_status('1', 'in-progress', str(tmp_path))
    assert 'error' not in result_inprog, (
        f'in-progress transition should not be rejected: {result_inprog}'
    )

    # taskmaster.set_task_status must have been called for both transitions
    assert taskmaster.set_task_status.call_count == 2


@pytest.mark.asyncio
async def test_predone_hook_skipped_on_done_to_done_noop(
    taskmaster, reconciler, event_buffer, monkeypatch
):
    """Same-status guard short-circuits before the pre-done hook fires.

    Even when the hook env var is /bin/false (which would reject the done
    transition if it ran), the done→done same-status no-op guard fires first
    and returns without ever invoking run_hook.

    The spy patches the bound alias on the consuming module
    (``fused_memory.middleware.task_interceptor._run_hook``) — the correct
    target for ``monkeypatch`` when the call site holds a module-level alias
    bound at import time.  Compare with
    test_predone_hook_spy_intercepts_bound_alias_on_pending_to_done which
    shows the same patch target IS intercepted when the guard does not
    short-circuit.
    """
    # Env var set to /bin/false — would reject if the hook fired
    monkeypatch.setenv('FUSED_MEMORY_PREDONE_HOOK_PROJECT', '/bin/false')
    # Task is already done — same-status guard should short-circuit
    taskmaster.get_task = AsyncMock(return_value={'id': '1', 'status': 'done', 'title': 'T'})
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    # Spy on the bound alias to verify it is never called.
    # Patching the source module attribute (_hook_mod.run_hook) would NOT
    # intercept the call site because task_interceptor already holds the
    # reference as _run_hook at import time.
    spy_calls: list = []

    async def _spy_run_hook(task_id, project_root, **kwargs):
        spy_calls.append((task_id, project_root))
        return None  # should never be reached

    monkeypatch.setattr('fused_memory.middleware.task_interceptor._run_hook', _spy_run_hook)

    result = await interceptor.set_task_status('1', 'done', '/project')

    # Must be the no-op shape — NOT a hook rejection
    assert result.get('success') is True
    assert result.get('no_op') is True
    assert result.get('task_id') == '1'
    # run_hook must not have been called
    assert spy_calls == [], f'run_hook should not have fired; got calls: {spy_calls}'
    # taskmaster.set_task_status must NOT have been called
    taskmaster.set_task_status.assert_not_called()
    reconciler.reconcile_task.assert_not_called()


@pytest.mark.asyncio
async def test_predone_hook_spy_intercepts_bound_alias_on_pending_to_done(
    taskmaster, reconciler, event_buffer, monkeypatch
):
    """Bound-alias patch target intercepts the hook call on a pending→done transition.

    Verifies that patching ``fused_memory.middleware.task_interceptor._run_hook``
    (the module-level alias bound at import time) correctly intercepts the hook
    invocation — confirming the patch target is correct.

    When the task is pending (not done), the same-status guard does NOT
    short-circuit, so the hook gate IS reached.  The spy therefore MUST be
    called.  Compare with test_predone_hook_skipped_on_done_to_done_noop where
    the guard fires first and the spy is never reached.
    """
    # taskmaster fixture default already returns status='pending' — same-status
    # guard will NOT short-circuit the done transition.
    monkeypatch.setenv('FUSED_MEMORY_PREDONE_HOOK_PROJECT', '/bin/false')
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    spy_calls: list = []

    async def _spy_run_hook(task_id, project_root, **kwargs):
        spy_calls.append((task_id, project_root))
        return None  # return None = success so the transition proceeds

    # Patch the bound alias on the consuming module, not the source module.
    monkeypatch.setattr('fused_memory.middleware.task_interceptor._run_hook', _spy_run_hook)

    result = await interceptor.set_task_status('1', 'done', '/project')

    # Spy must have been reached: pending→done bypasses the same-status guard
    assert spy_calls == [('1', '/project')], (
        f'_run_hook spy was not called; spy_calls={spy_calls!r}'
    )
    # Hook returned None (success) → transition proceeds; no error in result
    assert 'error' not in result, f'Expected successful transition, got: {result}'
    # Taskmaster must have been invoked (transition completed)
    taskmaster.set_task_status.assert_called_once()


@pytest.mark.asyncio
async def test_predone_hook_per_project_isolation(
    taskmaster, reconciler, event_buffer, monkeypatch, tmp_path
):
    """Per-project env-var keying: hook for project-a must not affect project-b.

    project-a has /bin/false → done transition is rejected.
    project-b has no env var set → done transition succeeds as normal.
    Confirms env-var lookup is per-call and per-project, not global.
    """
    project_a = tmp_path / 'project-a'
    project_b = tmp_path / 'project-b'
    project_a.mkdir()
    project_b.mkdir()

    pid_a = resolve_project_id(str(project_a)).upper()  # e.g. PROJECT_A
    pid_b = resolve_project_id(str(project_b)).upper()  # e.g. PROJECT_B

    monkeypatch.setenv(f'FUSED_MEMORY_PREDONE_HOOK_{pid_a}', '/bin/false')
    monkeypatch.delenv(f'FUSED_MEMORY_PREDONE_HOOK_{pid_b}', raising=False)
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    # project-a: hook fires and rejects
    result_a = await interceptor.set_task_status('1', 'done', str(project_a))
    assert result_a['success'] is False
    assert result_a['error'] == 'pre_done_hook_rejected', (
        f'Expected pre_done_hook_rejected for project-a, got: {result_a}'
    )

    # project-b: no hook, transition succeeds
    taskmaster.set_task_status.reset_mock()
    result_b = await interceptor.set_task_status('2', 'done', str(project_b))
    assert 'error' not in result_b, f'project-b should succeed without hook, got: {result_b}'
    taskmaster.set_task_status.assert_called_once()


# ---------------------------------------------------------------------------
# Task 1272: orphan-race observability + cancel_ticket logs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_add_ticket_orphan_race_logs_warning_with_task_id(
    interceptor_with_store,
    ticket_store,
    taskmaster,
    caplog,
):
    """_process_add_ticket emits a WARNING when the terminal mark_resolved returns
    False because cancel_ticket won the TOCTOU race.

    Setup: a racing_mark_resolved wrapper simulates cancel_ticket winning the
    race by persisting status='cancelled' BEFORE forwarding the worker's
    mark_resolved(status='created') call.  The worker's call therefore returns
    False.  We assert that a WARNING record exists whose message contains both
    the ticket_id and the distinctive task_id 'task-orphan-99' returned by
    tm.add_task (overridden below), and that the message uses the neutral
    'orphan-race:' label rather than the misleading '_process_add_ticket:' prefix.

    RED: the WARNING message currently starts with '_process_add_ticket: orphan-race'
    instead of the neutral 'orphan-race:' label.
    """
    import logging

    # Override add_task to return a distinctive id that can't accidentally match
    # other tokens in the log message.
    taskmaster.add_task.return_value = {'id': 'task-orphan-99', 'title': 'New Task'}

    ticket_id = await ticket_store.submit(
        project_id='project',
        candidate_json=json.dumps(
            {
                'project_root': '/project',
                'kwargs': {'title': 'T', 'description': 'D'},
                'metadata': None,
            }
        ),
    )

    original_mark_resolved = ticket_store.mark_resolved

    async def racing_mark_resolved(tid: str, *, status: str, **kwargs):
        if tid == ticket_id and status == 'created':
            # Simulate cancel_ticket winning the race: flip the row to
            # 'cancelled' BEFORE the worker's mark_resolved lands.
            await original_mark_resolved(tid, status='cancelled', reason='user_cancelled')
        # Forward the worker's call — returns False because row is no longer pending.
        return await original_mark_resolved(tid, status=status, **kwargs)

    ticket_store.mark_resolved = racing_mark_resolved
    try:
        with caplog.at_level(logging.WARNING, logger='fused_memory.middleware.task_interceptor'):
            await interceptor_with_store._process_add_ticket(ticket_id)
    finally:
        ticket_store.mark_resolved = original_mark_resolved

    # (a) WARNING record must contain both ticket_id and orphan task_id 'task-orphan-99'
    warning_records = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and ticket_id in r.message and 'task-orphan-99' in r.message
    ]
    assert warning_records, (
        f'Expected a WARNING containing ticket_id={ticket_id!r} and task_id="task-orphan-99"; '
        f'got records: {[(r.levelno, r.message) for r in caplog.records]}'
    )

    # (a') WARNING must use the neutral 'orphan-race:' label
    for r in warning_records:
        assert 'orphan-race:' in r.message, (
            f'Expected WARNING to contain "orphan-race:" but got: {r.message!r}'
        )

    # (a'') WARNING must include the caller label so per-caller grep patterns work
    for r in warning_records:
        assert 'caller=add_ticket' in r.message, (
            f'Expected WARNING to contain "caller=add_ticket" but got: {r.message!r}'
        )

    # (b) Row status is 'cancelled' (cancel_ticket won the race)
    row = await ticket_store.get(ticket_id)
    assert row is not None
    assert row['status'] == 'cancelled', f'Expected cancelled, got {row["status"]!r}'

    # (c) tm.add_task was called once — the task is live in tasks.json
    taskmaster.add_task.assert_called_once()


@pytest.mark.asyncio
async def test_cancel_ticket_clean_win_logs_info(
    interceptor_with_store,
    ticket_store,
    caplog,
):
    """cancel_ticket emits an INFO log when it successfully cancels a pending ticket.

    RED: cancel_ticket currently emits no log on the clean-cancel path.
    """
    import logging

    ticket_id = await ticket_store.submit(project_id='p', candidate_json='{}')

    with caplog.at_level(logging.INFO, logger='fused_memory.middleware.task_interceptor'):
        result = await interceptor_with_store.cancel_ticket(ticket_id)

    # (a) Existing contract is preserved
    assert result == {'status': 'cancelled', 'ticket_id': ticket_id}, (
        f'Expected cancelled result, got: {result!r}'
    )

    # (b) Exactly one INFO record with ticket_id and 'cancelled'
    info_records = [
        r
        for r in caplog.records
        if r.levelno == logging.INFO and ticket_id in r.message and 'cancelled' in r.message
    ]
    assert info_records, (
        f'Expected an INFO record containing ticket_id={ticket_id!r} and "cancelled"; '
        f'got records: {[(r.levelno, r.message) for r in caplog.records]}'
    )


@pytest.mark.asyncio
async def test_cancel_ticket_race_loss_logs_info_with_status(
    interceptor_with_store,
    ticket_store,
    caplog,
):
    """cancel_ticket emits an INFO log when it loses the TOCTOU race to a concurrent worker.

    Reuses the racing_mark_resolved pattern from
    test_cancel_ticket_race_returns_noop_with_actual_status: a concurrent
    worker terminalizes the row to 'created' between cancel's get() and the
    UPDATE.  After the race, cancel_ticket re-fetches and returns the no_op
    shape.  We assert an INFO record exists containing the ticket_id, the
    actual recovered status ('created'), and a race indicator.

    Level is INFO (not WARNING) because not every race-loss implies an orphan:
    if the worker finished normally the race is benign.  The authoritative
    orphan WARNING lives in _persist_worker_terminal.
    """
    import logging

    ticket_id = await ticket_store.submit(project_id='p', candidate_json='{}')

    original_mark_resolved = ticket_store.mark_resolved

    async def racing_mark_resolved(tid: str, *, status: str, **kwargs):
        if tid == ticket_id and status == 'cancelled':
            # The racing writer wins first: force the row to terminal 'created'.
            await original_mark_resolved(tid, status='created', reason='raced_first')
        # Now our cancel UPDATE runs — returns False because status != 'pending'.
        return await original_mark_resolved(tid, status=status, **kwargs)

    ticket_store.mark_resolved = racing_mark_resolved
    try:
        with caplog.at_level(logging.INFO, logger='fused_memory.middleware.task_interceptor'):
            result = await interceptor_with_store.cancel_ticket(ticket_id)
    finally:
        ticket_store.mark_resolved = original_mark_resolved

    # (a) Existing contract is preserved
    assert result == {'status': 'created', 'ticket_id': ticket_id, 'no_op': True}, (
        f'Expected no_op with actual status=created, got: {result!r}'
    )

    # (b) INFO record with ticket_id, actual status, and a race indicator
    info_records = [
        r
        for r in caplog.records
        if r.levelno == logging.INFO
        and ticket_id in r.message
        and 'created' in r.message
        and any(kw in r.message for kw in ('race', 'raced', 'lost'))
    ]
    assert info_records, (
        f'Expected an INFO record containing ticket_id={ticket_id!r}, "created", and a '
        f'race indicator; got records: {[(r.levelno, r.message) for r in caplog.records]}'
    )


@pytest.mark.asyncio
async def test_persist_worker_terminal_orphan_race_emits_warning(
    interceptor_with_store,
    ticket_store,
    caplog,
):
    """_persist_worker_terminal emits a WARNING when mark_resolved returns False
    while status='created' and task_id is non-None.

    This is a focused unit test for the helper itself.  We pre-terminate the
    ticket row to 'cancelled' so that the subsequent mark_resolved(status='created')
    returns False, and verify the orphan-race WARNING is emitted with both the
    ticket_id and the orphan task_id.

    The authoritative orphan-WARNING lives here — not in cancel_ticket — because
    only the worker knows whether a live task was created before the race was lost.
    """
    import logging

    # Submit a ticket and immediately terminate it to 'cancelled', simulating
    # cancel_ticket winning the race before the worker reaches mark_resolved.
    ticket_id = await ticket_store.submit(project_id='p', candidate_json='{}')
    await ticket_store.mark_resolved(ticket_id, status='cancelled', reason='pre_cancelled')

    orphan_task_id = 'task-orphan-42'
    with caplog.at_level(logging.WARNING, logger='fused_memory.middleware.task_interceptor'):
        result = await interceptor_with_store._persist_worker_terminal(
            ticket_id,
            status='created',
            task_id=orphan_task_id,
            reason='worker_completed',
            result_dict=None,
            caller='unit_test',
        )

    # mark_resolved returned False because the row was no longer pending
    assert result is False, f'Expected False (row was pre-cancelled), got {result!r}'

    # A WARNING must be emitted containing both ticket_id and orphan task_id
    warning_records = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and ticket_id in r.message and orphan_task_id in r.message
    ]
    assert warning_records, (
        f'Expected a WARNING containing ticket_id={ticket_id!r} and '
        f'task_id={orphan_task_id!r}; '
        f'got records: {[(r.levelno, r.message) for r in caplog.records]}'
    )

    # WARNING must use the neutral 'orphan-race:' label
    for r in warning_records:
        assert 'orphan-race:' in r.message, (
            f'Expected WARNING to contain "orphan-race:" but got: {r.message!r}'
        )

    # WARNING must include the caller label so per-caller grep patterns work
    for r in warning_records:
        assert 'caller=unit_test' in r.message, (
            f'Expected WARNING to contain "caller=unit_test" but got: {r.message!r}'
        )


@pytest.mark.asyncio
async def test_process_add_ticket_cancelled_after_dispatch_emits_orphan_warning(
    interceptor_with_store,
    ticket_store,
    caplog,
):
    """_process_add_ticket's CancelledError handler routes through _persist_worker_terminal
    so orphan-race WARNINGs are emitted uniformly on the cancellation path.

    Strategy: hook _dispatch_ticket_decision to pre-cancel the row (simulating
    cancel_ticket winning the race), schedule task cancellation, and return the
    success tuple (status='created', task_id='task-cancelled-99').  Hook
    _curator_lock with a simple @asynccontextmanager that yields then does
    asyncio.sleep(0); the sleep is the first suspension after the dispatch tuple
    is unpacked, so the pending cancellation fires there — still inside
    _process_add_ticket's try block with status='created' already set.

    Robustness: the cancel is scheduled in fake_dispatch (where status/task_id
    are known to be set) rather than in a finally clause, making the trigger
    point explicit and independent of contextmanager cleanup ordering.

    RED: the inline asyncio.shield(mark_resolved(...)) in the CancelledError
    handler emits no WARNING, so this test fails before the fix.
    """
    import logging
    from contextlib import asynccontextmanager

    ticket_id = await ticket_store.submit(
        project_id='project',
        candidate_json=json.dumps(
            {
                'project_root': '/project',
                'kwargs': {'title': 'T', 'description': 'D'},
                'metadata': None,
            }
        ),
    )

    # Stub: pre-cancel the row, schedule task cancellation, then return the success
    # tuple so that _process_add_ticket's locals reach status='created',
    # task_id='task-cancelled-99' before the pending cancel is delivered.
    async def fake_dispatch(**kwargs):
        await ticket_store.mark_resolved(ticket_id, status='cancelled', reason='user_cancelled')
        # Schedule cancellation here, while status/task_id are about to be set.
        # The CancelledError fires at the first suspension inside fake_curator_lock
        # (the post-yield asyncio.sleep(0)), which is still inside _process_add_ticket's
        # try block — so the except asyncio.CancelledError handler sees status='created'.
        task = asyncio.current_task()
        if task is not None:
            task.cancel()
        return (
            'created',
            'task-cancelled-99',
            None,
            {'id': 'task-cancelled-99', 'title': 'T'},
            None,
        )

    # Stub: yield for the body (fake_dispatch runs here), then suspend once so the
    # pending cancellation scheduled by fake_dispatch is delivered while still inside
    # _process_add_ticket's try block.
    def fake_curator_lock(project_id):
        @asynccontextmanager
        async def _ctx():
            yield
            # Suspension-point invariant: this sleep is the FIRST yield after
            # fake_dispatch's `task.cancel()`, so the pending CancelledError is
            # delivered HERE — still inside `_process_add_ticket`'s try block with
            # status='created' and task_id already set.  See the surrounding
            # docstring for the full timing chain.  Do NOT reorder or remove
            # without also auditing the cancellation-timing contract documented
            # in the test docstring above.
            await asyncio.sleep(0)

        return _ctx()

    interceptor_with_store._dispatch_ticket_decision = fake_dispatch
    interceptor_with_store._curator_lock = fake_curator_lock
    try:
        with (
            caplog.at_level(logging.WARNING, logger='fused_memory.middleware.task_interceptor'),
            pytest.raises(asyncio.CancelledError),
        ):
            await interceptor_with_store._process_add_ticket(ticket_id)
    finally:
        del interceptor_with_store._dispatch_ticket_decision
        del interceptor_with_store._curator_lock

    # (a) WARNING must contain ticket_id and orphan task_id
    warning_records = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and ticket_id in r.message
        and 'task-cancelled-99' in r.message
    ]
    assert warning_records, (
        f'Expected a WARNING containing ticket_id={ticket_id!r} and '
        f'"task-cancelled-99"; '
        f'got records: {[(r.levelno, r.message) for r in caplog.records]}'
    )

    # (b) WARNING must use the neutral 'orphan-race:' label
    for r in warning_records:
        assert 'orphan-race:' in r.message, (
            f'Expected WARNING to contain "orphan-race:" but got: {r.message!r}'
        )

    # (c) WARNING must include the caller label so per-caller grep patterns work
    for r in warning_records:
        assert 'caller=add_ticket_cancel' in r.message, (
            f'Expected WARNING to contain "caller=add_ticket_cancel" but got: {r.message!r}'
        )


# ── Regression: cycle 8df8bdcd title↔task_id contract (task 1379) ──────────
# Scenario shared via _fm_helpers.make_8df8_scenario (str ids, status='pending').
# Tests lock the before-capture/spawn contract in _apply_status_transition.

# Fixture: 8df8bdcd scenario (str ids, pending) — canonical definition in _fm_helpers.py
_8DF8_TASKS_INTERCEPTOR, _8DF8_TITLE_BY_ID_INTERCEPTOR = make_8df8_scenario(id_type=str, status='pending')


@pytest.mark.asyncio
async def test_multicompletion_window_each_reconcile_gets_own_task_before(
    event_buffer, tmp_path,
):
    """CSV path: each spawned reconcile_task call receives its OWN task_before.

    Reproduces cycle 8df8bdcd at the task_interceptor spawn path: drive
    set_task_status("1355,1361,1369", "done") and verify that each spawned
    reconciler.reconcile_task call received a task_before dict whose 'title'
    matches that call's own task_id — no aliasing/late-binding title swap.

    Expected GREEN: `before` is a fresh local per _apply_status_transition
    invocation; CSV loops sequentially (sequential awaits); asyncio.create_task
    args bind eagerly at coroutine creation time — no closure/shared-mutable
    aliasing possible across the multi-completion window.
    """
    # Build a taskmaster that returns the right task dict for each id
    task_map = {t['id']: dict(t) for t in _8DF8_TASKS_INTERCEPTOR}

    async def mock_get_task(task_id, *args, **kwargs):
        # Return the per-id dict (or a generic fallback)
        return task_map.get(str(task_id), {'id': str(task_id), 'status': 'pending', 'title': f'Task {task_id}'})

    tm = AsyncMock()
    tm.get_task = mock_get_task
    tm.set_task_status = AsyncMock(return_value={'success': True})
    tm.get_tasks = AsyncMock(return_value={'tasks': []})
    tm.update_task = AsyncMock(return_value={'success': True})

    # Record reconcile_task calls by task_id → task_before
    captured: dict[str, dict] = {}

    async def mock_reconcile(*, task_id, transition, project_id, project_root, task_before, **kwargs):
        captured[task_id] = task_before
        return {'actions': [{'type': 'knowledge_captured'}]}

    reconciler = AsyncMock()
    reconciler.reconcile_task = mock_reconcile

    interceptor = TaskInterceptor(tm, reconciler, event_buffer)

    # Drive CSV completion (completion order: 1355,1361,1369 comma-joined)
    csv_ids = '1355,1361,1369'
    result = await interceptor.set_task_status(csv_ids, 'done', '/project')
    assert result.get('success') is True, f'Unexpected result: {result}'

    # Drain background tasks so reconcile calls run
    for _ in range(10):
        await asyncio.sleep(0)

    # All 3 tasks should have been reconciled
    assert set(captured.keys()) == {'1355', '1361', '1369'}, (
        f'Expected reconcile calls for all 3 tasks; got: {set(captured.keys())}'
    )

    # Each task_before must carry ITS OWN title — no neighbor bleed
    for task_id, task_before in captured.items():
        expected_title = _8DF8_TITLE_BY_ID_INTERCEPTOR[task_id]
        actual_title = task_before.get('title')
        assert actual_title == expected_title, (
            f'task_id={task_id}: task_before.title={actual_title!r} '
            f'but expected own title={expected_title!r}.\n'
            f'  task_before={task_before}'
        )
        # task_before.id must also match
        assert str(task_before.get('id')) == task_id, (
            f'task_id={task_id}: task_before["id"]={task_before.get("id")!r} mismatch'
        )


@pytest.mark.asyncio
async def test_single_completion_reconcile_gets_correct_task_before(
    event_buffer, tmp_path,
):
    """Single-id path: reconcile_task receives the correct task_before.

    Interleaved variant of the 8df8bdcd scenario: drive each task id
    individually (single set_task_status calls) and verify title↔id binding.
    """
    task_map = {t['id']: dict(t) for t in _8DF8_TASKS_INTERCEPTOR}

    async def mock_get_task(task_id, *args, **kwargs):
        return task_map.get(str(task_id), {'id': str(task_id), 'status': 'pending', 'title': f'Task {task_id}'})

    tm = AsyncMock()
    tm.get_task = mock_get_task
    tm.set_task_status = AsyncMock(return_value={'success': True})
    tm.get_tasks = AsyncMock(return_value={'tasks': []})
    tm.update_task = AsyncMock(return_value={'success': True})

    captured: dict[str, dict] = {}

    async def mock_reconcile(*, task_id, transition, project_id, project_root, task_before, **kwargs):
        captured[task_id] = task_before
        return {'actions': [{'type': 'knowledge_captured'}]}

    reconciler = AsyncMock()
    reconciler.reconcile_task = mock_reconcile

    interceptor = TaskInterceptor(tm, reconciler, event_buffer)

    # Drive individually in non-id order (1369 → 1355 → 1361)
    for task in _8DF8_TASKS_INTERCEPTOR:
        await interceptor.set_task_status(task['id'], 'done', '/project')
    for _ in range(10):
        await asyncio.sleep(0)

    assert set(captured.keys()) == {'1355', '1361', '1369'}
    for task_id, task_before in captured.items():
        expected_title = _8DF8_TITLE_BY_ID_INTERCEPTOR[task_id]
        actual_title = task_before.get('title')
        assert actual_title == expected_title, (
            f'Single-id path task_id={task_id}: got title={actual_title!r}, '
            f'expected={expected_title!r}'
        )



# ── Qualified dep seam tests ───────────────────────────────────────


@pytest.mark.asyncio
async def test_interceptor_add_dependency_forwards_qualified_verbatim(
    interceptor, taskmaster, event_buffer,
):
    """Qualified depends_on string is forwarded verbatim to tm.add_dependency
    and a task_modified event is emitted.
    """
    taskmaster.add_dependency = AsyncMock(
        return_value={'id': '1', 'dependency_id': 'dark_factory:13', 'message': 'ok'},
    )
    await interceptor.add_dependency('1', 'dark_factory:13', '/project')

    taskmaster.add_dependency.assert_awaited_once()
    _, kwargs = taskmaster.add_dependency.call_args
    assert kwargs.get('depends_on') == 'dark_factory:13' or (
        len(taskmaster.add_dependency.call_args.args) >= 2
        and taskmaster.add_dependency.call_args.args[1] == 'dark_factory:13'
    )

    stats = await event_buffer.get_buffer_stats('project')
    assert stats['size'] == 1


@pytest.mark.asyncio
async def test_interceptor_remove_dependency_forwards_qualified_verbatim(
    interceptor, taskmaster, event_buffer,
):
    """Qualified depends_on string is forwarded verbatim to tm.remove_dependency
    and a task_modified event is emitted.
    """
    taskmaster.remove_dependency = AsyncMock(
        return_value={'id': '1', 'dependency_id': 'dark_factory:13', 'message': 'ok'},
    )
    await interceptor.remove_dependency('1', 'dark_factory:13', '/project')

    taskmaster.remove_dependency.assert_awaited_once()
    _, kwargs = taskmaster.remove_dependency.call_args
    assert kwargs.get('depends_on') == 'dark_factory:13' or (
        len(taskmaster.remove_dependency.call_args.args) >= 2
        and taskmaster.remove_dependency.call_args.args[1] == 'dark_factory:13'
    )

    stats = await event_buffer.get_buffer_stats('project')
    assert stats['size'] == 1


@pytest.mark.asyncio
async def test_get_tasks_threads_statuses(interceptor, taskmaster):
    """interceptor.get_tasks forwards statuses kwarg to taskmaster.get_tasks.

    (a) statuses=['pending','in-progress'] → taskmaster called with that value.
    (b) statuses omitted (default) → taskmaster called with statuses=None.
    """
    # (a) Explicit statuses filter
    taskmaster.get_tasks = AsyncMock(return_value={'tasks': []})
    await interceptor.get_tasks('/project', statuses=['pending', 'in-progress'])

    taskmaster.get_tasks.assert_awaited_once()
    _, kwargs = taskmaster.get_tasks.call_args
    assert kwargs.get('statuses') == ['pending', 'in-progress'], (
        f'Expected statuses forwarded as kwarg, call_args: {taskmaster.get_tasks.call_args}'
    )

    # (b) Default (omitted) → statuses=None forwarded
    taskmaster.get_tasks = AsyncMock(return_value={'tasks': []})
    await interceptor.get_tasks('/project')

    taskmaster.get_tasks.assert_awaited_once()
    _, kwargs2 = taskmaster.get_tasks.call_args
    assert kwargs2.get('statuses') is None, (
        f'Expected statuses=None when omitted, call_args: {taskmaster.get_tasks.call_args}'
    )


# ── task-1810 step-11/12: _extract_metadata_files WARNING on present-but-malformed ──

_TI_LOGGER = 'fused_memory.middleware.task_interceptor'


class TestExtractMetadataFilesWarns:
    """Module-level _extract_metadata_files emits WARNING when metadata.files is
    present-but-malformed.

    Step-11 (RED): all malformed paths return silently.
    Step-12 (GREEN): WARNING when files present-but-not-a-list, and when non-str/empty entries
    are dropped from a non-empty list.
    """

    def _fn(self):
        from fused_memory.middleware.task_interceptor import _extract_metadata_files
        return _extract_metadata_files

    def test_files_not_a_list_returns_empty_and_warns(self, caplog):
        """{'metadata':{'files':'a.py'}} => [] AND a WARNING.

        Currently RED: isinstance-guard returns [] silently.
        """
        import logging
        fn = self._fn()
        with caplog.at_level(logging.WARNING, logger=_TI_LOGGER):
            result = fn({'metadata': {'files': 'a.py'}})
        assert result == [], f"expected [], got {result!r}"
        warns = [
            r for r in caplog.records
            if r.name == _TI_LOGGER and r.levelno >= logging.WARNING
        ]
        assert warns, (
            "expected a WARNING when metadata.files is present but not a list; "
            f"got warns={[r.message for r in warns]!r}"
        )

    def test_list_with_dropped_entries_warns(self, caplog):
        """{'metadata':{'files':['ok.py', 123, '']}} => ['ok.py'] AND a WARNING noting drops.

        Currently RED: filter runs silently, no WARNING emitted.
        """
        import logging
        fn = self._fn()
        with caplog.at_level(logging.WARNING, logger=_TI_LOGGER):
            result = fn({'metadata': {'files': ['ok.py', 123, '']}})
        assert result == ['ok.py'], f"expected ['ok.py'], got {result!r}"
        warns = [
            r for r in caplog.records
            if r.name == _TI_LOGGER and r.levelno >= logging.WARNING
        ]
        assert warns, (
            "expected a WARNING noting dropped non-str/empty entries; "
            f"got warns={[r.message for r in warns]!r}"
        )

    def test_absent_files_key_no_warning(self, caplog):
        """REGRESSION: metadata dict with no 'files' key => [] with ZERO warnings."""
        import logging
        fn = self._fn()
        with caplog.at_level(logging.WARNING, logger=_TI_LOGGER):
            result = fn({'metadata': {}})
        assert result == []
        warns = [
            r for r in caplog.records
            if r.name == _TI_LOGGER and r.levelno >= logging.WARNING
        ]
        assert not warns, f"absent files must not emit WARNINGs; got {warns!r}"

    def test_absent_metadata_no_warning(self, caplog):
        """REGRESSION: task dict with no 'metadata' key => [] with ZERO warnings."""
        import logging
        fn = self._fn()
        with caplog.at_level(logging.WARNING, logger=_TI_LOGGER):
            result = fn({'id': '1', 'title': 'task'})
        assert result == []
        warns = [
            r for r in caplog.records
            if r.name == _TI_LOGGER and r.levelno >= logging.WARNING
        ]
        assert not warns, f"absent metadata must not emit WARNINGs; got {warns!r}"

    def test_non_dict_task_data_no_warning(self, caplog):
        """REGRESSION: non-dict task_data => [] with ZERO warnings."""
        import logging
        fn = self._fn()
        with caplog.at_level(logging.WARNING, logger=_TI_LOGGER):
            result = fn('not-a-dict')
        assert result == []
        warns = [
            r for r in caplog.records
            if r.name == _TI_LOGGER and r.levelno >= logging.WARNING
        ]
        assert not warns, f"non-dict task_data must not emit WARNINGs; got {warns!r}"

    def test_empty_files_list_no_warning(self, caplog):
        """REGRESSION: metadata.files=[] => [] with ZERO warnings."""
        import logging
        fn = self._fn()
        with caplog.at_level(logging.WARNING, logger=_TI_LOGGER):
            result = fn({'metadata': {'files': []}})
        assert result == []
        warns = [
            r for r in caplog.records
            if r.name == _TI_LOGGER and r.levelno >= logging.WARNING
        ]
        assert not warns, f"empty files list must not emit WARNINGs; got {warns!r}"

    def test_valid_files_list_no_warning(self, caplog):
        """REGRESSION: metadata.files=['a.py','b.py'] => ['a.py','b.py'] with ZERO warnings."""
        import logging
        fn = self._fn()
        with caplog.at_level(logging.WARNING, logger=_TI_LOGGER):
            result = fn({'metadata': {'files': ['a.py', 'b.py']}})
        assert result == ['a.py', 'b.py']
        warns = [
            r for r in caplog.records
            if r.name == _TI_LOGGER and r.levelno >= logging.WARNING
        ]
        assert not warns, f"valid files list must not emit WARNINGs; got {warns!r}"
