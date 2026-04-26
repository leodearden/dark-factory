"""Tests for task interceptor middleware."""

import asyncio
import contextlib
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
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
        'TaskInterceptor.add_task must be removed; migrate callers to '
        'submit_task + resolve_ticket'
    )


@pytest.mark.asyncio
async def test_submit_and_resolve_helper_returns_legacy_shape(
    interceptor_facade, taskmaster,
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

    metadata = {'source': 'review-cycle', 'modules': ['my-project/src']}
    result = await _submit_and_resolve(interceptor_facade, '/project', prompt='Test', metadata=metadata)
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
    await _submit_and_resolve(interceptor_facade, 
        '/project', prompt='Test', metadata=metadata_json,
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
            raise TypeError(
                "add_task() got an unexpected keyword argument 'metadata'"
            )
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
        await interceptor.remove_task('1', '/project')
        # Let the drainer catch up.
        await asyncio.wait_for(queue._queue.join(), timeout=1.0)

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
async def curator_interceptor(taskmaster, reconciler, event_buffer, curator_enabled_config, tmp_path):
    from fused_memory.middleware.ticket_store import TicketStore
    store = TicketStore(tmp_path / 'curator_tickets.db')
    await store.initialize()
    ti = TaskInterceptor(
        taskmaster, reconciler, event_buffer, config=curator_enabled_config,
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
    taskmaster.get_tasks = AsyncMock(return_value={
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
    })


def _assert_r4_common(curator_mock, taskmaster) -> None:
    """Shared tail assertions for R4 idempotency-hit tests.

    Both the add_task and submit_task entry-path tests must verify that
    the curator was bypassed and no new task was created.
    """
    curator_mock.curate.assert_not_called()
    taskmaster.add_task.assert_not_called()


@pytest.mark.asyncio
async def test_curator_drop_short_circuits_add_task(
    curator_interceptor, taskmaster,
):
    """A drop decision returns the target_id without calling tm.add_task."""
    decision = CuratorDecision(
        action='drop', target_id='99',
        justification='already covered by task 99',
    )
    curator_interceptor._curator = _mock_curator(decision)

    result = await _submit_and_resolve(curator_interceptor, 
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
    curator_interceptor, taskmaster,
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

    result = await _submit_and_resolve(curator_interceptor, 
        '/project',
        title='Fix parser on empty input',
        description='Parser panics on empty string',
    )

    assert result['id'] == '50'
    assert result['action'] == 'combine'
    # Combine calls update_task with a prompt instructing the rewrite.
    taskmaster.update_task.assert_called_once()
    call = taskmaster.update_task.call_args
    assert call.kwargs['task_id'] == '50'
    assert 'Harden parser' in call.kwargs['prompt']
    assert 'line 42' in call.kwargs['prompt']  # specifics preserved verbatim
    taskmaster.add_task.assert_not_called()


@pytest.mark.asyncio
async def test_curator_create_proceeds_with_add_task(
    curator_interceptor, taskmaster,
):
    """A create decision forwards to tm.add_task normally."""
    decision = CuratorDecision(action='create', justification='genuinely new')
    curator_interceptor._curator = _mock_curator(decision)

    result = await _submit_and_resolve(curator_interceptor, 
        '/project', title='Novel unrelated work',
    )

    assert result == {'id': '2', 'title': 'New Task'}
    taskmaster.add_task.assert_called_once()


@pytest.mark.asyncio
async def test_curator_combine_failure_falls_through_to_create(
    curator_interceptor, taskmaster,
):
    """If tm.update_task raises during combine, fall back to creating the task."""
    rewritten = RewrittenTask(
        title='x', description='', details='d',
        files_to_modify=[], priority='medium',
    )
    decision = CuratorDecision(
        action='combine', target_id='50',
        target_fingerprint='Test Task',  # matches fixture — guard passes
        rewritten_task=rewritten,
        justification='...',
    )
    curator_interceptor._curator = _mock_curator(decision)
    taskmaster.update_task.side_effect = RuntimeError('taskmaster failed')

    result = await _submit_and_resolve(curator_interceptor, 
        '/project', title='Fix x',
    )

    # Fell through to create path
    assert result == {'id': '2', 'title': 'New Task'}
    taskmaster.add_task.assert_called_once()


@pytest.mark.asyncio
async def test_curator_drop_short_circuits_add_subtask(
    curator_interceptor, taskmaster,
):
    """add_subtask also runs the curator gate — previously bypassed."""
    decision = CuratorDecision(
        action='drop', target_id='88', justification='duplicate of sibling',
    )
    curator_interceptor._curator = _mock_curator(decision)

    result = await curator_interceptor.add_subtask(
        '1', '/project', title='Duplicate subtask work',
    )

    assert result['id'] == '88'
    assert result['action'] == 'drop'
    taskmaster.add_subtask.assert_not_called()


# ─────────────────────────────────────────────────────────────────────
# WP-F: combine-safety guard (fingerprint + status + audit log)
# ─────────────────────────────────────────────────────────────────────


def _combine_audit_lines(audit_dir):
    """Read JSONL records written by _append_combine_audit."""
    path = audit_dir / 'combine_audit.jsonl'
    if not path.exists():
        return []
    lines = [
        ln for ln in path.read_text(encoding='utf-8').splitlines() if ln.strip()
    ]
    import json as _json
    return [_json.loads(ln) for ln in lines]


@pytest.fixture
def audit_dir(tmp_path, monkeypatch):
    """Redirect combine_audit.jsonl writes to a per-test tmp dir."""
    monkeypatch.setenv('DARK_FACTORY_DATA_DIR', str(tmp_path))
    return tmp_path


@pytest.mark.asyncio
async def test_curator_combine_fingerprint_match_proceeds(
    curator_interceptor, taskmaster, audit_dir,
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
        action='combine', target_id='50',
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
    curator_interceptor, taskmaster, audit_dir,
):
    """Fingerprint doesn't match live target → abort, fall through to create."""
    rewritten = RewrittenTask(
        title='x', description='', details='d',
        files_to_modify=[], priority='medium',
    )
    decision = CuratorDecision(
        action='combine', target_id='50',
        target_fingerprint='Wrong Title',  # fixture returns 'Test Task'
        rewritten_task=rewritten, justification='...',
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
    curator_interceptor, taskmaster, audit_dir,
):
    """Decision with no fingerprint (LLM skipped the field) → abort."""
    rewritten = RewrittenTask(
        title='x', description='', details='d',
        files_to_modify=[], priority='medium',
    )
    decision = CuratorDecision(
        action='combine', target_id='50',
        target_fingerprint=None,
        rewritten_task=rewritten, justification='...',
    )
    curator_interceptor._curator = _mock_curator(decision)

    result = await _submit_and_resolve(curator_interceptor, '/project', title='Fix x')

    assert result == {'id': '2', 'title': 'New Task'}
    taskmaster.update_task.assert_not_called()
    assert _combine_audit_lines(audit_dir) == []


@pytest.mark.asyncio
async def test_curator_combine_target_done_aborts(
    curator_interceptor, taskmaster, audit_dir,
):
    """Target with status=done → abort (would silently drop candidate work)."""
    taskmaster.get_task = AsyncMock(return_value={
        'id': '50', 'status': 'done', 'title': 'Done Task',
    })
    rewritten = RewrittenTask(
        title='x', description='', details='d',
        files_to_modify=[], priority='medium',
    )
    decision = CuratorDecision(
        action='combine', target_id='50',
        target_fingerprint='Done Task',  # fingerprint matches
        rewritten_task=rewritten, justification='...',
    )
    curator_interceptor._curator = _mock_curator(decision)

    result = await _submit_and_resolve(curator_interceptor, '/project', title='Fix x')

    assert result == {'id': '2', 'title': 'New Task'}
    taskmaster.update_task.assert_not_called()
    assert _combine_audit_lines(audit_dir) == []


@pytest.mark.asyncio
async def test_curator_combine_target_cancelled_aborts(
    curator_interceptor, taskmaster, audit_dir,
):
    """Target with status=cancelled → abort, same reasoning as done."""
    taskmaster.get_task = AsyncMock(return_value={
        'id': '50', 'status': 'cancelled', 'title': 'Cancelled Task',
    })
    rewritten = RewrittenTask(
        title='x', description='', details='d',
        files_to_modify=[], priority='medium',
    )
    decision = CuratorDecision(
        action='combine', target_id='50',
        target_fingerprint='Cancelled Task',
        rewritten_task=rewritten, justification='...',
    )
    curator_interceptor._curator = _mock_curator(decision)

    result = await _submit_and_resolve(curator_interceptor, '/project', title='Fix x')

    assert result == {'id': '2', 'title': 'New Task'}
    taskmaster.update_task.assert_not_called()
    assert _combine_audit_lines(audit_dir) == []


@pytest.mark.asyncio
async def test_curator_combine_fingerprint_normalization(
    curator_interceptor, taskmaster, audit_dir,
):
    """Case / whitespace drift on the title is tolerated by the guard."""
    taskmaster.get_task = AsyncMock(return_value={
        'id': '50', 'status': 'pending',
        'title': 'Harden Parser for Empty Input',
    })
    rewritten = RewrittenTask(
        title='Harden parser', description='', details='d',
        files_to_modify=[], priority='medium',
    )
    decision = CuratorDecision(
        action='combine', target_id='50',
        # extra whitespace + different case — should still match
        target_fingerprint='   harden  parser   for empty INPUT   ',
        rewritten_task=rewritten, justification='normalize',
    )
    curator_interceptor._curator = _mock_curator(decision)

    result = await _submit_and_resolve(curator_interceptor, '/project', title='c')

    assert result['action'] == 'combine'
    taskmaster.update_task.assert_called_once()


@pytest.mark.asyncio
async def test_curator_combine_bulk_dedupe_respects_guard(
    taskmaster, reconciler, event_buffer, curator_enabled_config, audit_dir,
):
    """_dedupe_bulk_created inherits the guard — a curator that picks a done
    task as the combine target must not clobber it; the new task is kept.

    Regression-style: exercises the bulk path to confirm WP-F's guard in
    _execute_combine protects it without additional changes.
    """
    # Pre-existing: task '50' is done. New bulk-created: task '100'.
    pre_snapshot = {'tasks': [{'id': '50', 'title': 'Done', 'status': 'done'}]}
    post_snapshot = {'tasks': [
        {'id': '50', 'title': 'Done', 'status': 'done'},
        {'id': '100', 'title': 'Fresh work', 'status': 'pending'},
    ]}

    taskmaster.get_tasks = AsyncMock(return_value=post_snapshot)
    # get_task for target '50' must look done so the guard aborts.
    taskmaster.get_task = AsyncMock(return_value={
        'id': '50', 'status': 'done', 'title': 'Done',
    })

    rewritten = RewrittenTask(
        title='x', description='', details='d',
        files_to_modify=[], priority='medium',
    )
    curator = _mock_curator(CuratorDecision(
        action='combine', target_id='50',
        target_fingerprint='Done',
        rewritten_task=rewritten, justification='wrong pick',
    ))

    interceptor = TaskInterceptor(
        taskmaster, reconciler, event_buffer, config=curator_enabled_config,
    )
    interceptor._curator = curator

    result = await interceptor._dedupe_bulk_created(
        '/project', pre_snapshot=pre_snapshot,
    )

    # New task kept (guard refused the combine), NOT removed.
    assert any(k['task_id'] == '100' for k in result['kept'])
    assert all(r['task_id'] != '100' for r in result['removed'])
    # Neither the target nor the new task was written.
    taskmaster.update_task.assert_not_called()
    assert _combine_audit_lines(audit_dir) == []


# ─────────────────────────────────────────────────────────────────────
# R3: concurrency hardening — per-project lock + pre-LLM short-circuit
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_add_task_produces_single_task(
    taskmaster, reconciler, event_buffer, curator_enabled_config, tmp_path,
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
        taskmaster, reconciler, event_buffer, config=curator_enabled_config,
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
        candidate, project_id='proj', project_root='/x',
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
        candidate, project_id='proj', project_root='/x',
    )
    assert decision2 is None


# ─────────────────────────────────────────────────────────────────────
# R4: escalation-level idempotency on (escalation_id, suggestion_hash)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_r4_idempotency_hit_add_task(
    taskmaster, reconciler, event_buffer, curator_enabled_config, tmp_path,
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
        taskmaster, reconciler, event_buffer, config=curator_enabled_config,
        ticket_store=store,
    )
    interceptor._curator = curator_mock

    metadata = {
        'escalation_id': 'esc-r4-986',
        'suggestion_hash': 'h986h986h986h986',
        'modules': ['fused-memory/src'],
    }
    try:
        result = await _submit_and_resolve(interceptor,
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
    taskmaster, reconciler, event_buffer, curator_enabled_config, tmp_path,
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
        taskmaster, reconciler, event_buffer, config=curator_enabled_config,
        ticket_store=store,
    )
    interceptor._curator = curator_mock

    metadata = {
        'escalation_id': 'esc-r4-986',
        'suggestion_hash': 'h986h986h986h986',
        'modules': ['fused-memory/src'],
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
        f"resolve_ticket should return task_id of existing task, got {result!r}"
    )
    assert result.get('reason') == 'idempotency_hit', (
        f"resolve_ticket should return reason='idempotency_hit', got {result!r}"
    )
    _assert_r4_common(curator_mock, taskmaster)


@pytest.mark.asyncio
async def test_idempotency_accepts_metadata_as_json_string(
    taskmaster, reconciler, event_buffer, curator_enabled_config, tmp_path,
):
    """Metadata that arrives as a pre-serialised JSON string also dedupes."""
    import json

    from fused_memory.middleware.ticket_store import TicketStore

    taskmaster.get_tasks = AsyncMock(return_value={
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
    })

    store = TicketStore(tmp_path / 'idemp_str_tickets.db')
    await store.initialize()
    interceptor = TaskInterceptor(
        taskmaster, reconciler, event_buffer, config=curator_enabled_config,
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
    taskmaster, reconciler, event_buffer, curator_enabled_config, tmp_path,
):
    """No matching (escalation_id, suggestion_hash) → curator runs normally."""
    from fused_memory.middleware.ticket_store import TicketStore

    taskmaster.get_tasks = AsyncMock(return_value={
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
    })

    curator_mock = _mock_curator(CuratorDecision(action='create', justification='novel'))
    store = TicketStore(tmp_path / 'idemp_miss_tickets.db')
    await store.initialize()
    interceptor = TaskInterceptor(
        taskmaster, reconciler, event_buffer, config=curator_enabled_config,
        ticket_store=store,
    )
    interceptor._curator = curator_mock

    try:
        await _submit_and_resolve(interceptor, 
            '/project', title='New',
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
    taskmaster, reconciler, event_buffer, curator_enabled_config, tmp_path,
):
    """A cancelled task with matching metadata must not win the dedupe."""
    from fused_memory.middleware.ticket_store import TicketStore

    taskmaster.get_tasks = AsyncMock(return_value={
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
    })

    curator_mock = _mock_curator(CuratorDecision(action='create', justification='novel'))
    store = TicketStore(tmp_path / 'idemp_cancel_tickets.db')
    await store.initialize()
    interceptor = TaskInterceptor(
        taskmaster, reconciler, event_buffer, config=curator_enabled_config,
        ticket_store=store,
    )
    interceptor._curator = curator_mock

    try:
        await _submit_and_resolve(interceptor, 
            '/project', title='Retry',
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
    taskmaster, reconciler, event_buffer, curator_enabled_config, tmp_path,
):
    """Metadata without escalation_id+suggestion_hash skips the R4 check."""
    from fused_memory.middleware.ticket_store import TicketStore

    curator_mock = _mock_curator(CuratorDecision(action='create', justification='novel'))
    store = TicketStore(tmp_path / 'idemp_both_tickets.db')
    await store.initialize()
    interceptor = TaskInterceptor(
        taskmaster, reconciler, event_buffer, config=curator_enabled_config,
        ticket_store=store,
    )
    interceptor._curator = curator_mock

    try:
        # Only escalation_id, no suggestion_hash → not eligible.
        await _submit_and_resolve(interceptor, 
            '/project', title='T', metadata={'escalation_id': 'esc-x'},
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
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer, config=cfg, ticket_store=store)

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
    curator_interceptor, taskmaster,
):
    """update_task triggers fire-and-forget reembed when title/details change."""
    curator_mock = _mock_curator(CuratorDecision(action='create'))
    curator_interceptor._curator = curator_mock
    taskmaster.get_task.return_value = {
        'id': '7', 'status': 'pending', 'title': 'Updated title',
        'description': 'desc', 'details': 'details',
    }

    await curator_interceptor.update_task(
        '7', '/project', prompt='rename title to updated',
    )
    await asyncio.sleep(0)  # let fire-and-forget run
    await curator_interceptor.drain()

    curator_mock.reembed_task.assert_called_once()


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
    taskmaster.get_tasks = AsyncMock(return_value={
        'tasks': [
            {'id': 1, 'status': 'pending'},
            {'id': 2, 'status': 'done'},
            {'id': 3, 'status': 'in-progress'},
        ]
    })
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
    """When ids=['1', '3'], only those two keys appear in the result."""
    taskmaster.get_tasks = AsyncMock(return_value={
        'tasks': [
            {'id': 1, 'status': 'pending'},
            {'id': 2, 'status': 'done'},
            {'id': 3, 'status': 'in-progress'},
        ]
    })
    interceptor = TaskInterceptor(taskmaster, None, event_buffer)

    result = await interceptor.get_statuses('/project', ids=['1', '3'])

    assert result == {'1': 'pending', '3': 'in-progress'}


@pytest.mark.asyncio
async def test_get_statuses_omits_unknown_ids(taskmaster, event_buffer):
    """Unknown ids in the filter list are silently omitted (no error, no key)."""
    taskmaster.get_tasks = AsyncMock(return_value={
        'tasks': [
            {'id': 1, 'status': 'pending'},
        ]
    })
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
    tm.get_tasks = AsyncMock(return_value={'tasks': []})
    interceptor = TaskInterceptor(tm, None, event_buffer)

    await interceptor.get_statuses('/project')
    tm.ensure_connected.assert_called_once()


@pytest.mark.asyncio
async def test_get_statuses_missing_status_key_defaults_to_unknown(
    taskmaster, event_buffer
):
    """A task dict without a 'status' key is included with status='unknown'.

    Contract: the sentinel 'unknown' is the documented default when the raw
    task dict omits 'status'.  Callers that need to distinguish a genuine
    'unknown' status from a missing field should treat any 'unknown' as
    indeterminate.
    """
    taskmaster.get_tasks = AsyncMock(return_value={
        'tasks': [
            {'id': 1},              # no 'status' key
            {'id': 2, 'status': 'done'},
        ]
    })
    interceptor = TaskInterceptor(taskmaster, None, event_buffer)

    result = await interceptor.get_statuses('/project')

    assert result == {'1': 'unknown', '2': 'done'}


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
    taskmaster, reconciler, event_buffer,
):
    """done->blocked is allowed when an explicit reopen_reason is passed."""
    taskmaster.get_task = AsyncMock(return_value={'id': '1', 'status': 'done', 'title': 'T'})
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status(
        '1', 'blocked', '/project', reopen_reason='manual re-scope',
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
async def test_done_gate_passes_when_files_exist(
    taskmaster, reconciler, event_buffer, tmp_path
):
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
async def test_done_gate_noop_without_metadata_files(
    taskmaster, reconciler, event_buffer
):
    """Gate does not fire when metadata.files is absent — back-compat for legacy tasks."""
    # default taskmaster fixture returns {'id':'1','status':'pending','title':'Test Task'} — no metadata
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status('1', 'done', '/project')

    assert 'error' not in result
    taskmaster.set_task_status.assert_called_once()


@pytest.mark.asyncio
async def test_done_gate_reports_partial_missing(
    taskmaster, reconciler, event_buffer, tmp_path
):
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
    assert sorted(result['files_checked']) == sorted(
        ['exists.rs', 'missing.rs', 'also_missing.ts']
    )
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
        ['git', '-C', str(path), 'config', 'user.email', 't@e.example'], check=True,
    )
    subprocess.run(
        ['git', '-C', str(path), 'config', 'user.name', 'T'], check=True,
    )
    (path / 'seed.txt').write_text('seed\n')
    subprocess.run(['git', '-C', str(path), 'add', '-A'], check=True)
    subprocess.run(
        ['git', '-C', str(path), 'commit', '-q', '-m', 'seed'], check=True,
    )
    return subprocess.run(
        ['git', '-C', str(path), 'rev-parse', 'HEAD'],
        check=True, capture_output=True, text=True,
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
        taskmaster, reconciler, event_buffer, config=config_with_strict_provenance,
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
        taskmaster, reconciler, event_buffer, config=config_with_strict_provenance,
    )

    result = await interceptor.set_task_status(
        '1', 'done', '/project', done_provenance={'commit': '', 'note': ''},
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
        '1', 'done', str(tmp_path),
        done_provenance={'commit': 'deadbeefdeadbeefdeadbeefdeadbeefdeadbeef'},
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
        '1', 'done', str(tmp_path), done_provenance={'commit': sha[:7]},
    )

    assert 'error' not in result
    taskmaster.update_task.assert_called_once()
    kwargs = taskmaster.update_task.call_args.kwargs
    persisted = json.loads(kwargs['metadata'])
    assert persisted['done_provenance']['commit'] == sha
    assert persisted['done_provenance']['commit_input'] == sha[:7]


@pytest.mark.asyncio
async def test_done_provenance_note_only_accepted_and_persisted(
    taskmaster, reconciler, event_buffer
):
    """A note-only payload is accepted without git validation and persisted."""
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status(
        '1', 'done', '/project',
        done_provenance={'note': 'covered by parent task 1745'},
    )

    assert 'error' not in result
    taskmaster.update_task.assert_called_once()
    persisted = json.loads(taskmaster.update_task.call_args.kwargs['metadata'])
    assert persisted['done_provenance'] == {'note': 'covered by parent task 1745'}


@pytest.mark.asyncio
async def test_done_provenance_commit_plus_note_both_persisted(
    taskmaster, reconciler, event_buffer, tmp_path
):
    """Both commit and note may be provided; both are recorded."""
    sha = _init_git_repo(tmp_path)
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status(
        '1', 'done', str(tmp_path),
        done_provenance={'commit': sha, 'note': 'ff-merged after review'},
    )

    assert 'error' not in result
    persisted = json.loads(taskmaster.update_task.call_args.kwargs['metadata'])
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
        taskmaster, reconciler, event_buffer, config=config_with_strict_provenance,
    )

    result = await interceptor.set_task_status(
        '1', 'in-progress', '/project', reopen_reason='resuming after investigation',
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
        '1', 'done', '/project', done_provenance=['not', 'a', 'dict'],  # type: ignore[arg-type]
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
        '1', 'done', str(tmp_path), done_provenance={'commit': sha},
    )

    project_id = resolve_project_id(str(tmp_path))
    events = await event_buffer.peek_buffered(project_id, limit=10)
    assert events, 'event should be buffered'
    payload = events[-1].payload
    assert payload['done_provenance']['commit'] == sha


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


@pytest.mark.asyncio
async def test_background_tasks_retained_for_bulk_operations(taskmaster, reconciler, event_buffer):
    """Background task from expand_task should be in _background_tasks during execution."""
    started = asyncio.Event()
    done_future: asyncio.Future = asyncio.Future()

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


@pytest_asyncio.fixture
async def interceptor_with_committer(taskmaster, reconciler, event_buffer, committer, tmp_path):
    from fused_memory.middleware.ticket_store import TicketStore
    store = TicketStore(tmp_path / 'committer_tickets.db')
    await store.initialize()
    ti = TaskInterceptor(taskmaster, reconciler, event_buffer, committer, ticket_store=store)
    yield ti
    await store.close()
    for _wt in list(ti._worker_tasks.values()):
        if not _wt.done():
            _wt.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await _wt


@pytest.mark.asyncio
async def test_write_methods_commit(interceptor_with_committer, committer):
    """All 9 write methods should commit (7 fire-and-forget, 2 awaited for bulk ops)."""
    i = interceptor_with_committer
    pr = '/project'

    await i.set_task_status('1', 'in-progress', pr)
    await _submit_and_resolve(i, pr, prompt='T')
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
async def test_no_committer_still_works(taskmaster, event_buffer, tmp_path):
    """task_committer=None should not break any write methods."""
    from fused_memory.middleware.ticket_store import TicketStore

    store = TicketStore(tmp_path / 'no_committer_tickets.db')
    await store.initialize()
    interceptor = TaskInterceptor(taskmaster, None, event_buffer, None, ticket_store=store)
    try:
        result = await _submit_and_resolve(interceptor, '/project', prompt='T')
    finally:
        await store.close()
        for _wt in list(interceptor._worker_tasks.values()):
            if not _wt.done():
                _wt.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await _wt
    assert result == {'id': '2', 'title': 'New Task'}


@pytest.mark.asyncio
async def test_terminal_state_transition_commits(taskmaster, reconciler, event_buffer, committer):
    """Reopened terminal transitions (done->blocked with reason) schedule a commit."""
    taskmaster.get_task = AsyncMock(return_value={'id': '1', 'status': 'done', 'title': 'T'})
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer, committer)
    result = await interceptor.set_task_status(
        '1', 'blocked', '/project', reopen_reason='manual re-scope',
    )
    assert 'error' not in result, result
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
async def test_drain_awaits_pending_commits(taskmaster, event_buffer, tmp_path):
    """drain() awaits all pending fire-and-forget commits."""
    from fused_memory.middleware.ticket_store import TicketStore

    commit_done = asyncio.Event()

    async def slow_commit(project_root: str, operation: str) -> None:
        await commit_done.wait()

    committer = AsyncMock()
    committer.commit = AsyncMock(side_effect=slow_commit)
    store = TicketStore(tmp_path / 'drain_tickets.db')
    await store.initialize()
    interceptor = TaskInterceptor(taskmaster, None, event_buffer, committer, ticket_store=store)

    try:
        # Fire-and-forget commits
        await _submit_and_resolve(interceptor, '/project', prompt='A')
        await _submit_and_resolve(interceptor, '/project', prompt='B')
        await asyncio.sleep(0)  # let tasks start

        # Background tasks should be pending
        assert len(interceptor._background_tasks) >= 1

        # Unblock the commits
        commit_done.set()

        # drain should await them all
        await interceptor.drain()
        assert len(interceptor._background_tasks) == 0
    finally:
        await store.close()
        for _wt in list(interceptor._worker_tasks.values()):
            if not _wt.done():
                _wt.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await _wt


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
                self.peak.get(project_key, 0), self.in_flight[project_key],
            )
            self.total_peak = max(self.total_peak, self._global_in_flight)
            try:
                # Yield to the loop so concurrent tasks really do interleave
                # — a zero-sleep await is enough to surface lock violations.
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                return (
                    return_value(*args, **kwargs)
                    if callable(return_value) else return_value
                )
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
    overlap_tm, reconciler, event_buffer, tmp_path,
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
            tracker.peak.get('p', 0), tracker.in_flight['p'],
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
        results = await asyncio.gather(*[
            _submit_and_resolve(interceptor, '/project', title=f'Task {i}')
            for i in range(N)
        ])
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
    overlap_tm, reconciler, event_buffer,
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
                tracker.peak.get('p', 0), tracker.in_flight['p'],
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
            'get_task', {'id': '1', 'status': 'pending', 'title': 'T'},
        ),
    )
    overlap_tm.remove_task = AsyncMock(
        side_effect=instrument('remove_task', {'success': True}),
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
    coros.append(interceptor.remove_task('7', '/project'))
    coros.append(interceptor.add_dependency('3', '2', '/project'))

    await asyncio.gather(*coros)

    assert tracker.peak.get('p', 0) == 1, (
        f'concurrent mutation observed on same project: peak={tracker.peak}'
    )


@pytest.mark.asyncio
async def test_two_projects_do_not_serialise(
    overlap_tm, reconciler, event_buffer, tmp_path,
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
                        f"project A entered but B never did — "
                        f"projA_entered={projA_entered.is_set()} "
                        f"projB_entered={projB_entered.is_set()}"
                    )
            else:
                projB_entered.set()
                try:
                    await asyncio.wait_for(projA_entered.wait(), timeout=10.0)
                except TimeoutError:
                    pytest.fail(
                        f"project B entered but A never did — "
                        f"projA_entered={projA_entered.is_set()} "
                        f"projB_entered={projB_entered.is_set()}"
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
    assert peak_a <= 1 and peak_b <= 1, (
        f'same-project overlap: {tracker.peak}'
    )
    # With per-project workers, projA and projB can run concurrently;
    # total_peak must equal the number of active projects (2), confirming
    # cross-project parallelism is achieved.
    assert tracker.total_peak == 2, (
        f'expected per-project parallelism (total_peak==2): total_peak={tracker.total_peak}'
    )


@pytest.mark.asyncio
async def test_set_task_status_holds_lock_across_read_and_write(
    overlap_tm, reconciler, event_buffer,
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
        assert second_from == first_to, (
            f'stale before-state observed: {call_log}'
        )
    assert r1.get('success') or r1.get('no_op')
    assert r2.get('success') or r2.get('no_op')


@pytest.mark.asyncio
async def test_expand_task_dedup_mutations_are_locked(
    reconciler, event_buffer,
):
    """WP-E: mutations inside _dedupe_bulk_created (tm.remove_task etc.)
    also acquire the per-project lock. A concurrent add_task fired while
    dedup is deciding which new tasks to drop must not observe a dedup
    mid-mutation."""
    tracker = _OverlapTracker()

    # Minimal stateful backend. expand_task returns a subtasks payload;
    # get_tasks reports pre-existing tasks that dedup would consult.
    tm = AsyncMock()

    async def _mut(kind, retval):
        tracker.in_flight['p'] = tracker.in_flight.get('p', 0) + 1
        tracker.peak['p'] = max(
            tracker.peak.get('p', 0), tracker.in_flight['p'],
        )
        try:
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            return retval
        finally:
            tracker.in_flight['p'] -= 1

    async def expand_side(*a, **k):
        return await _mut('expand', {'subtasks': [{'id': '1.1'}]})

    async def add_side(**k):
        return await _mut('add', {'id': '99', 'title': 'x'})

    async def remove_side(*a, **k):
        return await _mut('remove', {'success': True})

    async def get_task_side(*a, **k):
        await asyncio.sleep(0)
        return {'id': '1', 'status': 'pending', 'title': 'T'}

    # get_tasks is a read — not tracked.
    tm.get_tasks = AsyncMock(return_value={'tasks': []})
    tm.get_task = AsyncMock(side_effect=get_task_side)
    tm.expand_task = AsyncMock(side_effect=expand_side)
    tm.add_task = AsyncMock(side_effect=add_side)
    tm.remove_task = AsyncMock(side_effect=remove_side)
    tm.ensure_connected = AsyncMock()

    interceptor = TaskInterceptor(tm, reconciler, event_buffer)

    # Fire expand + a concurrent add_task on the same project. Both are
    # mutating; the lock must prevent overlap.
    results = await asyncio.gather(
        interceptor.expand_task('1', '/project'),
        _submit_and_resolve(interceptor, '/project', title='concurrent add'),
    )

    assert results[0] is not None
    assert results[1] is not None
    assert tracker.peak.get('p', 0) == 1, (
        f'overlap during expand + add: peak={tracker.peak}'
    )


@pytest.mark.asyncio
async def test_single_call_latency_not_regressed(
    overlap_tm, reconciler, event_buffer,
):
    """WP-E: guardrail — a sequence of sequential mutating calls under
    no contention finishes under a generous budget, so the lock itself
    adds no meaningful per-call overhead.
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
    # Very generous bound — on a mock this should complete in well under
    # 4s even with real SQLite event-buffer writes and 32 xdist workers
    # competing for disk I/O; bumping from 2s to 4s for CI jitter.
    assert elapsed < 4.0, f'{N} sequential calls took {elapsed:.3f}s'


@pytest.mark.asyncio
async def test_set_task_status_does_not_block_during_add_task_curator(
    taskmaster, reconciler, event_buffer, curator_enabled_config, tmp_path,
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
        taskmaster, reconciler, event_buffer, config=curator_enabled_config,
        ticket_store=store,
    )

    CURATOR_LATENCY_S = 2.0

    async def slow_curate(candidate, project_id, project_root):
        await asyncio.sleep(CURATOR_LATENCY_S)
        # action='create' so the flow falls through to tm.add_task
        return CuratorDecision(
            action='create', justification='ok',
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
            '1', 'in-progress', '/project',
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
    interceptor_with_store, ticket_store, taskmaster,
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
    taskmaster, reconciler, event_buffer, tmp_path,
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
    orphan_id = await store.submit(project_id='project', candidate_json='{}', ttl_seconds=600)
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
    taskmaster, reconciler, event_buffer, tmp_path,
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
    interceptor_facade, taskmaster,
):
    """R3 invariant: the add_task worker path must acquire _curator_lock so
    it is mutually exclusive with add_subtask / remove_task curator calls.

    Earlier in the ticket-queue refactor (step-46) the worker did not acquire
    this lock — the single-worker queue was treated as sufficient serialisation.
    Reviewer feedback (esc-919-148) correctly pointed out that add_subtask
    still enters _curator_lock directly and could therefore race against the
    worker's curate() on a stale pre-note_created snapshot, with both paths
    deciding "create" for the same candidate.  The worker now takes
    _curator_lock(project_id) across curate() → note_created → record_task,
    preserving the old cross-family R3 invariant while retaining per-project
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

    # --- add_subtask: must also acquire curator_lock exactly once ---
    await interceptor_facade.add_subtask(parent_id='1', project_root='/project', title='Sub')
    assert acquisition_count == 2, (
        f'add_subtask should acquire _curator_lock exactly once; got {acquisition_count - 1} '
        'after the add_task acquisition'
    )


# ---------------------------------------------------------------------------
# step-63: regression — no lost-wakeup between terminal-check and event-register
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_ticket_no_lost_wakeup_between_read_and_register(
    interceptor_with_store, ticket_store,
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
        ttl_seconds=600,
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

    assert result.get('status') == 'created', (
        f'Expected status=created but got: {result!r}'
    )
    assert result.get('task_id') == '42', (
        f'Expected task_id=42 but got: {result!r}'
    )


# ---------------------------------------------------------------------------
# Terminal-exit gate: server-side FSM that refuses done/cancelled -> non-same
# without an explicit reopen_reason.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_terminal_exit_rejects_done_to_pending_without_reason(
    interceptor, taskmaster,
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
    interceptor, taskmaster,
):
    taskmaster.get_task = AsyncMock(
        return_value={'id': '1', 'status': 'cancelled', 'title': 'T'},
    )
    result = await interceptor.set_task_status('1', 'pending', '/project')
    assert result.get('error') == 'terminal_exit_rejected'
    taskmaster.set_task_status.assert_not_called()


@pytest.mark.asyncio
async def test_terminal_exit_accepts_with_reopen_reason(
    interceptor, taskmaster,
):
    """done -> pending with a non-empty reopen_reason succeeds and persists reason."""
    taskmaster.get_task = AsyncMock(
        return_value={'id': '1', 'status': 'done', 'title': 'T'},
    )
    result = await interceptor.set_task_status(
        '1', 'pending', '/project', reopen_reason='un-defer script',
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
        '1', 'pending', '/project', reopen_reason='   ',
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
    interceptor, taskmaster, event_buffer,
):
    """Emitted event carries reopen_reason and reopen_from for audit."""
    taskmaster.get_task = AsyncMock(
        return_value={'id': '1', 'status': 'cancelled', 'title': 'T'},
    )
    await interceptor.set_task_status(
        '1', 'pending', '/project', reopen_reason='manual re-scope',
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
        '1,2,3', 'pending', '/project',
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
    interceptor, taskmaster,
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
    interceptor_with_guard, taskmaster, tmp_path,
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
    interceptor_with_guard, taskmaster, tmp_path,
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
# Intra-batch deduplication (task-1004)
# Step-3: integration tests — written before step-4 adds the pre-pass.
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_expand_task_removes_intra_batch_duplicates(
    curator_interceptor, taskmaster,
):
    """expand_task should remove intra-batch duplicates before passing unique
    survivors to the curator.

    Three new subtasks are created: '1.2' is a case+whitespace variant of
    '1.1' → intra-batch duplicate and must be removed.  '1.3' is unrelated.
    Expectations:
      - removed has 1 entry: task_id='1.2', reason='intra_batch_duplicate',
        matched_task_id='1.1'
      - kept has 2 entries: '1.1' and '1.3'
      - taskmaster.remove_task called exactly once (for '1.2')
      - curator.curate called exactly twice (for '1.1' and '1.3' only)
    """
    # Parent task '1' exists before expand; subtasks are newly created.
    parent_task = {'id': '1', 'title': 'Parent', 'status': 'pending'}
    pre_snapshot = {'tasks': [parent_task]}
    post_snapshot = {'tasks': [
        {**parent_task, 'subtasks': [
            {'id': '1.1', 'title': 'Fix foo', 'description': 'bar'},
            {'id': '1.2', 'title': 'FIX FOO', 'description': ' bar '},
            {'id': '1.3', 'title': 'Unrelated', 'description': 'baz'},
        ]},
    ]}

    taskmaster.get_tasks = AsyncMock(side_effect=[pre_snapshot, post_snapshot])
    taskmaster.remove_task = AsyncMock(return_value={'success': True})

    curator_interceptor._curator = _mock_curator(
        CuratorDecision(action='create', justification='new')
    )

    result = await curator_interceptor.expand_task('1', '/project')

    dedup = result['dedup']

    # (a) removed has exactly 1 entry for '1.2'
    assert len(dedup['removed']) == 1, f"removed={dedup['removed']}"
    removed_entry = dedup['removed'][0]
    assert removed_entry['task_id'] == '1.2', removed_entry
    assert removed_entry['reason'] == 'intra_batch_duplicate', removed_entry
    assert removed_entry['matched_task_id'] == '1.1', removed_entry

    # (b) kept has exactly 2 entries: '1.1' and '1.3'
    kept_ids = {k['task_id'] for k in dedup['kept']}
    assert kept_ids == {'1.1', '1.3'}, f"kept_ids={kept_ids}"

    # (c) remove_task called exactly once for '1.2'
    assert taskmaster.remove_task.await_count == 1
    taskmaster.remove_task.assert_awaited_once_with('1.2', '/project')

    # (d) curator.curate called exactly twice — unique survivors only
    assert curator_interceptor._curator.curate.await_count == 2


@pytest.mark.asyncio
async def test_parse_prd_removes_intra_batch_duplicates(
    curator_interceptor, taskmaster,
):
    """parse_prd routes through _dedupe_bulk_created so the intra-batch
    pre-pass applies identically.

    Three top-level tasks: id=11 is a case+whitespace variant of id=10 →
    intra-batch duplicate removed.  id=12 is unrelated.
    Proves the pre-pass is shared between expand_task and parse_prd paths.
    """
    pre_snapshot = {'tasks': []}
    post_snapshot = {'tasks': [
        {'id': '10', 'title': 'Setup DB', 'description': 'postgres'},
        {'id': '11', 'title': 'SETUP db', 'description': 'Postgres'},
        {'id': '12', 'title': 'Write tests', 'description': 'unit tests'},
    ]}

    taskmaster.get_tasks = AsyncMock(side_effect=[pre_snapshot, post_snapshot])
    taskmaster.remove_task = AsyncMock(return_value={'success': True})
    # parse_prd must return a dict so result['dedup'] assignment works
    taskmaster.parse_prd = AsyncMock(return_value={'tasks': []})

    curator_interceptor._curator = _mock_curator(
        CuratorDecision(action='create', justification='new')
    )

    result = await curator_interceptor.parse_prd('prd.md', '/project')

    dedup = result['dedup']

    # exactly 1 removed entry for '11'
    assert len(dedup['removed']) == 1, f"removed={dedup['removed']}"
    removed_entry = dedup['removed'][0]
    assert removed_entry['task_id'] == '11', removed_entry
    assert removed_entry['reason'] == 'intra_batch_duplicate', removed_entry
    assert removed_entry['matched_task_id'] == '10', removed_entry

    # 2 kept entries: '10' and '12'
    kept_ids = {k['task_id'] for k in dedup['kept']}
    assert kept_ids == {'10', '12'}, f"kept_ids={kept_ids}"

    # curator.curate called exactly twice (unique survivors only, not 3)
    assert curator_interceptor._curator.curate.await_count == 2


@pytest.mark.asyncio
async def test_expand_task_intra_batch_remove_is_locked(
    reconciler, event_buffer,
):
    """WP-E: the intra-batch remove_task in _dedupe_bulk_created acquires
    the per-project write_lock, so it cannot overlap with concurrent
    mutations (e.g. add_task) on the same project.

    Modelled on test_expand_task_dedup_mutations_are_locked.
    """
    tracker = _OverlapTracker()

    tm = AsyncMock()

    async def _mut(kind, retval):
        tracker.in_flight['p'] = tracker.in_flight.get('p', 0) + 1
        tracker.peak['p'] = max(
            tracker.peak.get('p', 0), tracker.in_flight['p'],
        )
        try:
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            return retval
        finally:
            tracker.in_flight['p'] -= 1

    async def expand_side(*a, **k):
        return await _mut('expand', {'subtasks': []})

    async def add_side(**k):
        return await _mut('add', {'id': '99', 'title': 'x'})

    async def remove_side(*a, **k):
        return await _mut('remove', {'success': True})

    async def get_task_side(*a, **k):
        await asyncio.sleep(0)
        return {'id': '1', 'status': 'pending', 'title': 'T'}

    # Two-call get_tasks: pre-snapshot (empty) then post-snapshot with two
    # intra-batch duplicates so the pre-pass fires remove_task.
    parent_task = {'id': '1', 'title': 'T', 'status': 'pending'}
    pre_snapshot = {'tasks': [parent_task]}
    post_snapshot = {'tasks': [{
        **parent_task,
        'subtasks': [
            {'id': '1.1', 'title': 'Dup task', 'description': 'same'},
            {'id': '1.2', 'title': 'DUP TASK', 'description': ' same '},
        ],
    }]}

    tm.get_tasks = AsyncMock(side_effect=[pre_snapshot, post_snapshot])
    tm.get_task = AsyncMock(side_effect=get_task_side)
    tm.expand_task = AsyncMock(side_effect=expand_side)
    tm.add_task = AsyncMock(side_effect=add_side)
    tm.remove_task = AsyncMock(side_effect=remove_side)
    tm.ensure_connected = AsyncMock()

    interceptor = TaskInterceptor(tm, reconciler, event_buffer)

    # Fire expand (which will trigger intra-batch remove for the dup) +
    # a concurrent add_task. The lock must prevent remove_task and the
    # tracked mutations from overlapping.
    results = await asyncio.gather(
        interceptor.expand_task('1', '/project'),
        _submit_and_resolve(interceptor, '/project', title='concurrent add'),
    )

    assert results[0] is not None
    assert results[1] is not None

    # The intra-batch remove_task for '1.2' should have been called.
    tm.remove_task.assert_awaited_once_with('1.2', '/project')

    # Peak must be 1: no overlap between expand, intra-batch remove_task,
    # and any concurrent tracked mutation.
    assert tracker.peak.get('p', 0) == 1, (
        f'overlap detected during expand + intra-batch remove: peak={tracker.peak}'
    )


@pytest.mark.asyncio
async def test_no_intra_batch_duplicates_preserves_existing_behaviour(
    curator_interceptor, taskmaster,
):
    """Regression guard: when no intra-batch duplicates exist the pre-pass
    is a true no-op — removed is empty, kept has all tasks, remove_task
    is NOT called, and curator.curate is called exactly once per unique task.

    This ensures the pre-pass does not alter existing cross-task dedup
    behaviour when the batch contains genuinely distinct tasks.
    """
    parent_task = {'id': '1', 'title': 'Parent', 'status': 'pending'}
    pre_snapshot = {'tasks': [parent_task]}
    post_snapshot = {'tasks': [{
        **parent_task,
        'subtasks': [
            {'id': '1.1', 'title': 'Task Alpha', 'description': 'does alpha work'},
            {'id': '1.2', 'title': 'Task Beta', 'description': 'does beta work'},
        ],
    }]}

    taskmaster.get_tasks = AsyncMock(side_effect=[pre_snapshot, post_snapshot])
    taskmaster.remove_task = AsyncMock(return_value={'success': True})

    curator_interceptor._curator = _mock_curator(
        CuratorDecision(action='create', justification='new')
    )

    result = await curator_interceptor.expand_task('1', '/project')

    dedup = result['dedup']

    # No intra-batch removals
    assert dedup['removed'] == [], f"expected no removals, got: {dedup['removed']}"

    # Both tasks kept
    kept_ids = {k['task_id'] for k in dedup['kept']}
    assert kept_ids == {'1.1', '1.2'}, f"kept_ids={kept_ids}"

    # remove_task NOT called at all
    assert taskmaster.remove_task.await_count == 0, (
        f'remove_task should not have been called: {taskmaster.remove_task.call_args_list}'
    )

    # curator.curate called for both tasks (2 unique survivors)
    assert curator_interceptor._curator.curate.await_count == 2


@pytest.mark.asyncio
async def test_dedupe_bulk_remove_failure_keeps_task_in_both_errors_and_kept(
    curator_interceptor, taskmaster,
):
    """Remove-failure fall-through invariant: when intra-batch duplicate removal
    raises transiently the failing task must appear in BOTH errors AND kept.

    Pins the defensive contract of the remove-failure fall-through (the except
    block in the intra-batch pre-pass that appends the failing task back to
    unique_new_tasks rather than discarding it): "Removal failed transiently —
    fall through to curator so this task still appears in `kept` rather than
    silently disappearing from both `removed` and `kept`."

    The dual-membership invariant (task lives in BOTH errors and kept) is
    intentional: asserting only the errors entry would allow a future refactor
    that drops the task from unique_new_tasks (e.g. a bare `continue` in the
    except block) to regress silently.
    """
    PROJECT = '/project'
    post_snapshot = {'tasks': [
        {'id': '10', 'title': 'Fix foo', 'description': 'bar'},
        {'id': '11', 'title': 'FIX FOO', 'description': ' bar '},
    ]}

    taskmaster.get_tasks = AsyncMock(return_value=post_snapshot)
    taskmaster.remove_task = AsyncMock(
        side_effect=RuntimeError('transient backend failure')
    )
    curator_interceptor._curator = _mock_curator(
        CuratorDecision(action='create', justification='novel')
    )

    result = await curator_interceptor._dedupe_bulk_created(
        PROJECT, pre_snapshot={'tasks': []},
    )

    # (a) Exactly one error entry, for task '11', mentioning the backend failure.
    assert len(result['errors']) == 1
    assert result['errors'][0]['task_id'] == '11'
    assert 'transient backend failure' in result['errors'][0]['error']

    # (b) Task '11' appears in kept (the fall-through re-added it to
    # unique_new_tasks which then passed through pass-2 as a 'create').
    assert any(k['task_id'] == '11' for k in result['kept']), (
        f"expected '11' in kept, got: {result['kept']}"
    )

    # (c) Task '11' must NOT appear in removed (the removal attempt failed).
    assert all(r['task_id'] != '11' for r in result['removed']), (
        f"unexpected '11' in removed: {result['removed']}"
    )

    # (d) remove_task was called exactly once, for task '11'.
    taskmaster.remove_task.assert_awaited_once_with('11', PROJECT)

    # (e) Both '10' and '11' reach pass-2 (curate called twice: '11' was
    # re-appended to unique_new_tasks after the except block).
    assert curator_interceptor._curator.curate.await_count == 2


@pytest.mark.asyncio
async def test_dedupe_bulk_blank_title_tasks_bypass_intra_batch_pre_pass(
    curator_interceptor, taskmaster,
):
    """Blank-title bypass: tasks with whitespace-only titles must not be
    collapsed by the intra-batch pre-pass into the first occurrence.

    Without the blank-title bypass in the intra-batch pre-pass every
    blank-title task would hash to the same _intra_batch_key('', ...) bucket
    regardless of description (because title normalises to ''), so the second
    and subsequent malformed subtasks would be incorrectly removed.  The guard
    passes them straight through so both tasks reach pass-2.

    Pass-1 uses ``if not title.strip():`` and pass-2 uses
    ``if not candidate.title.strip():``, so both passes reject empty and
    whitespace-only titles symmetrically.  Whitespace tasks bypass pass-1's
    hash-collapse and short-circuit in pass-2 before reaching the curator,
    landing in ``kept`` without any curator call or removal.  This is the
    resolution of the formerly-tracked pass-1/pass-2 asymmetry.
    """
    PROJECT = '/project'
    post_snapshot = {'tasks': [
        {'id': '20', 'title': '   ', 'description': 'first malformed task'},
        {'id': '21', 'title': '\t\n', 'description': 'second malformed task'},
    ]}

    taskmaster.get_tasks = AsyncMock(return_value=post_snapshot)
    taskmaster.remove_task = AsyncMock(return_value={'success': True})
    curator_interceptor._curator = _mock_curator(
        CuratorDecision(action='create', justification='novel')
    )

    result = await curator_interceptor._dedupe_bulk_created(
        PROJECT, pre_snapshot={'tasks': []},
    )

    # (a) No removals.
    assert result['removed'] == [], f"expected no removals, got: {result['removed']}"

    # (b) No errors.
    assert result['errors'] == [], f"expected no errors, got: {result['errors']}"

    # (c) Both tasks present in kept.
    kept_ids = {k['task_id'] for k in result['kept']}
    assert kept_ids == {'20', '21'}, f"kept_ids={kept_ids}"

    # (d) remove_task was never called (the pre-pass guard fired so no removal
    # attempt was made for these malformed tasks).
    assert taskmaster.remove_task.await_count == 0, (
        f'remove_task should not have been called: {taskmaster.remove_task.call_args_list}'
    )

    # (e) pass-2's blank-title short-circuit (``if not candidate.title.strip():``)
    # catches whitespace-only titles, so the curator is never invoked for these
    # tasks; they still land in `kept` via the short-circuit's own kept.append.
    assert curator_interceptor._curator.curate.await_count == 0, (
        f'pass-2 blank-title short-circuit must prevent curator calls for '
        f'whitespace titles: {curator_interceptor._curator.curate.call_args_list}'
    )


@pytest.mark.asyncio
async def test_dedupe_bulk_pass1_intra_batch_and_pass2_curator_drops_compose(
    curator_interceptor, taskmaster,
):
    """Composition contract: a single batch can drop in pass-1 AND pass-2.

    Pins that when both an intra-batch duplicate (pass-1) and a
    curator-drop (pass-2) occur in the same batch, both appear correctly
    in result['removed'] — one with reason='intra_batch_duplicate', one
    with reason='curator_drop' — and the sole surviving new task ends up
    in result['kept'].

    Setup
    -----
    pre_snapshot: one pre-existing task '50' (becomes the curator drop
      target for task '62').
    post_snapshot: '50' plus three new top-level tasks '60', '61', '62'.
      '61' is a case+whitespace dup of '60' → pass-1 drops it.
      '62' is semantically subsumed by '50' → curator drops it in pass-2.
    Per-candidate routing curator: returns 'drop' for '62' (target '50'),
      'create' for everything else.
    """
    PROJECT = '/project'
    pre_snapshot = {'tasks': [
        {'id': '50', 'title': 'Existing alpha work', 'status': 'pending'},
    ]}
    post_snapshot = {'tasks': [
        {'id': '50', 'title': 'Existing alpha work', 'status': 'pending'},
        {'id': '60', 'title': 'Build feature', 'description': 'core impl'},
        {'id': '61', 'title': 'BUILD FEATURE', 'description': ' core IMPL '},
        {'id': '62', 'title': 'Refactor alpha module', 'description': 'cleanup'},
    ]}

    taskmaster.get_tasks = AsyncMock(return_value=post_snapshot)
    taskmaster.remove_task = AsyncMock(return_value={'success': True})

    # Per-candidate routing: use _mock_curator as base (consistent mock surface)
    # then override curate with a side_effect that routes by candidate title.
    # _mock_curator's curate_batch delegates to curator.curate at call time, so
    # it automatically picks up the new side_effect without re-wiring.
    async def _route(candidate, *a, **kw):
        if candidate.title == 'Refactor alpha module':
            return CuratorDecision(
                action='drop', target_id='50',
                justification='subsumed by existing alpha task',
            )
        return CuratorDecision(action='create', justification='novel')

    curator = _mock_curator(CuratorDecision(action='create', justification='novel'))
    curator.curate = AsyncMock(side_effect=_route)
    curator_interceptor._curator = curator

    result = await curator_interceptor._dedupe_bulk_created(
        PROJECT, pre_snapshot=pre_snapshot,
    )

    # (a) Exactly two removals total (one from each pass).
    assert len(result['removed']) == 2, f"expected 2 removals, got: {result['removed']}"

    # (b) '61' removed as intra-batch duplicate of '60'; no justification key
    # (pass-1 is mechanical — no LLM involvement, no justification emitted).
    removed_61 = next((r for r in result['removed'] if r['task_id'] == '61'), None)
    assert removed_61 is not None, "'61' missing from removed"
    assert removed_61['reason'] == 'intra_batch_duplicate'
    assert removed_61['matched_task_id'] == '60'
    assert 'justification' not in removed_61 or not removed_61.get('justification')

    # (c) '62' removed by the curator, matched to pre-existing task '50'.
    removed_62 = next((r for r in result['removed'] if r['task_id'] == '62'), None)
    assert removed_62 is not None, "'62' missing from removed"
    assert removed_62['reason'] == 'curator_drop'
    assert removed_62['matched_task_id'] == '50'
    assert removed_62.get('justification'), f"expected non-empty justification: {removed_62}"

    # (d) Only '60' ends up in kept (the unique survivor from both passes).
    kept_ids = {k['task_id'] for k in result['kept']}
    assert kept_ids == {'60'}, f"kept_ids={kept_ids}"

    # (e) No errors.
    assert result['errors'] == [], f"unexpected errors: {result['errors']}"

    # (f) remove_task called twice — once for '61' (pass-1) and once for
    # '62' (pass-2) — coupling to PROJECT makes the path explicit.
    assert taskmaster.remove_task.await_count == 2, (
        f'remove_task call count wrong: {taskmaster.remove_task.call_args_list}'
    )
    actual_calls = {tuple(c.args) for c in taskmaster.remove_task.call_args_list}
    assert actual_calls == {('61', PROJECT), ('62', PROJECT)}, (
        f'remove_task called with unexpected args: {actual_calls}'
    )

    # (g) curator.curate called exactly twice: pass-2 sees only the unique
    # survivors '60' and '62', not '61' (which was already removed in pass-1).
    assert curator.curate.await_count == 2, (
        f'curate call count wrong: {curator.curate.call_args_list}'
    )


@pytest.mark.asyncio
async def test_dedupe_bulk_intra_batch_acquires_write_lock_once_for_batch(
    curator_interceptor, taskmaster,
):
    """Intra-batch pre-pass must hold _write_lock continuously across all N removals.

    Pins the batched-lock discipline introduced by task-981 item 1: all N
    intra-batch removals are grouped first (outside the lock) and then
    issued inside a single ``async with self._write_lock(project_id):``
    block, rather than serially entering/releasing the lock N times.

    Verification strategy (contract pin for lock-hold semantics across the batch):
    Gate tm.remove_task on paired asyncio.Events and probe the write lock between
    consecutive calls.  Under the batched form the lock is held continuously so
    both probes raise TimeoutError; under the old per-item form the lock would
    be released and re-acquired between items, so the first probe would succeed.
    """
    PROJECT = '/project'
    PROJECT_ID = resolve_project_id(PROJECT)  # 'project' (resolve strips leading slash)
    post_snapshot = {'tasks': [
        {'id': '10', 'title': 'Fix foo', 'description': 'bar'},
        {'id': '11', 'title': 'FIX FOO', 'description': ' bar '},
        {'id': '12', 'title': 'fix Foo', 'description': 'BAR '},
        {'id': '13', 'title': 'Fix  foo', 'description': 'bar'},
    ]}

    taskmaster.get_tasks = AsyncMock(return_value=post_snapshot)

    curator_interceptor._curator = _mock_curator(
        CuratorDecision(action='create', justification='novel')
    )

    # Gate the first two remove_task calls on paired Events so the test can
    # observe the lock state between consecutive removals without polling.
    # The lock is acquired under PROJECT_ID (not PROJECT) — _dedupe_bulk_created
    # calls resolve_project_id(project_root) before locking.
    inside_first = asyncio.Event()
    release_first = asyncio.Event()
    inside_second = asyncio.Event()
    release_second = asyncio.Event()
    _call_count = 0

    async def _gated_remove(tid, _project_root):
        nonlocal _call_count
        _call_count += 1
        if _call_count == 1:
            inside_first.set()
            await release_first.wait()
        elif _call_count == 2:
            inside_second.set()
            await release_second.wait()
        return {'success': True}

    taskmaster.remove_task = AsyncMock(side_effect=_gated_remove)

    # Schedule the dedupe as a background task so this coroutine can interleave.
    dedupe_task = asyncio.create_task(
        curator_interceptor._dedupe_bulk_created(PROJECT, pre_snapshot={'tasks': []})
    )

    # Wait until we are inside the first removal — the batched lock is held.
    await inside_first.wait()

    # Probe 1: lock must be held (TimeoutError proves the batched scope is active).
    # In the happy path the lock IS held by the gated _dedupe_bulk_created task, so
    # asyncio.wait_for(lock.acquire(), timeout=0.5) blocks the FULL 0.5 s before
    # raising TimeoutError — TimeoutError is the pass condition, not the failure branch.
    # The value is sized for slow-CI robustness (e.g. coverage-instrumented runs);
    # two probes add ~1 s total wall-clock, kept acceptable by the bounded timeout.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            curator_interceptor._write_lock(PROJECT_ID).acquire(), timeout=0.5
        )
    assert curator_interceptor._write_lock(PROJECT_ID).locked(), (
        'lock must still be held after the probe — asyncio.Lock cancellation race '
        '(pre-3.11) could leave the lock free if a waiter was cancelled just as it '
        'became available; gating closes that window here but this assertion catches '
        'any future refactor that re-opens it'
    )

    # Advance to the second removal.
    release_first.set()
    await inside_second.wait()

    # Probe 2: lock must STILL be held between consecutive removals.
    # Under the old per-item form the lock would have been released here and
    # the probe would succeed, distinguishing the two implementations.
    # Same reasoning as Probe 1: the full 0.5 s is spent in the happy path because
    # the lock is still held; TimeoutError is the pass condition.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            curator_interceptor._write_lock(PROJECT_ID).acquire(), timeout=0.5
        )
    assert curator_interceptor._write_lock(PROJECT_ID).locked(), (
        'lock must still be held after the probe — same cancellation-race rationale '
        'as Probe 1; gating closes the window but the assertion fails loudly if a '
        'future edit re-opens it'
    )

    # Release the second removal and let the batch complete.
    release_second.set()
    result = await dedupe_task

    # Anchor that the gating counter was incremented for every pass-1 removal —
    # if pass-1 batches differently in the future, the Event barriers would
    # cover different calls and this assertion would catch the divergence first.
    assert _call_count == 3, (
        f'_gated_remove called {_call_count} times; expected 3 (one per intra-batch dup)'
    )

    # (a) 3 tasks removed (the 3 intra-batch duplicates of '10').
    assert len(result['removed']) == 3, f"expected 3 removals, got: {result['removed']}"

    # (b) remove_task called 3 times, once per duplicate.
    assert taskmaster.remove_task.await_count == 3, (
        f'remove_task call count wrong: {taskmaster.remove_task.call_args_list}'
    )
    actual_calls = {tuple(c.args) for c in taskmaster.remove_task.call_args_list}
    assert actual_calls == {('11', PROJECT), ('12', PROJECT), ('13', PROJECT)}, (
        f'remove_task called with unexpected args: {actual_calls}'
    )

    # (c) Sanity check: the lock is released once the batch exits its scope.
    assert not curator_interceptor._write_lock(PROJECT_ID).locked(), (
        'write lock must be released after the batch completes'
    )
    #
    # NOTE — contract pin for lock-hold semantics across the batch: the two
    # TimeoutError probes above pin the key invariant — the lock is held
    # continuously across all N removals, not released and re-acquired per item.
    # A regression to per-item locking would be caught by probe 2 succeeding
    # instead of raising TimeoutError.
    # See the two-phase discipline comment in _dedupe_bulk_created for the
    # rationale (no LLM calls between removals → safe to batch; pass-2 stays
    # per-item because LLM calls happen between its removals).


@pytest.mark.asyncio
async def test_dedupe_bulk_intra_batch_partial_failure_under_batched_lock_continues_remaining(
    curator_interceptor, taskmaster,
):
    """Partial-failure under the batched lock must not abort the rest of the batch.

    Generalises test_dedupe_bulk_remove_failure_keeps_task_in_both_errors_and_kept
    (N=1) to the N>1 batched-lock case.  With per-item try/except INSIDE the
    lock, a transient backend failure on the middle duplicate ('12') must not
    prevent the other two ('11', '13') from being removed.

    A single try/except wrapping the ENTIRE loop would abort after the first
    failure — remove_task would only be called twice (for '11' and '12'),
    and '13' would silently slip through to kept instead of removed.
    This test catches that tempting-but-wrong refactor by asserting
    remove_task.await_count == 3 and '13' in result['removed'].
    """
    PROJECT = '/project'
    post_snapshot = {'tasks': [
        {'id': '10', 'title': 'Fix foo', 'description': 'bar'},
        {'id': '11', 'title': 'FIX FOO', 'description': ' bar '},
        {'id': '12', 'title': 'fix Foo', 'description': 'BAR '},
        {'id': '13', 'title': 'Fix  foo', 'description': 'bar'},
    ]}

    taskmaster.get_tasks = AsyncMock(return_value=post_snapshot)

    # '12' raises transiently; '11' and '13' succeed.
    def _side_effect(tid, _project_root):
        if tid == '12':
            raise RuntimeError('transient backend failure')
        return {'success': True}

    taskmaster.remove_task = AsyncMock(side_effect=_side_effect)

    curator_interceptor._curator = _mock_curator(
        CuratorDecision(action='create', justification='novel')
    )

    result = await curator_interceptor._dedupe_bulk_created(
        PROJECT, pre_snapshot={'tasks': []},
    )

    # (a) remove_task called for all three duplicates — the failure on '12'
    # did NOT abort the rest of the batch.  A single try/except wrapping the
    # whole loop would call it only twice (stop after '12' raises).
    assert taskmaster.remove_task.await_count == 3, (
        f'remove_task call count wrong (expected 3): {taskmaster.remove_task.call_args_list}'
    )

    # (b) Exactly one error entry, for '12', mentioning the backend failure.
    assert len(result['errors']) == 1, f"expected 1 error, got: {result['errors']}"
    assert result['errors'][0]['task_id'] == '12'
    assert 'transient backend failure' in result['errors'][0]['error']

    # (c) '12' appears in kept (dual-append fall-through).
    assert any(k['task_id'] == '12' for k in result['kept']), (
        f"expected '12' in kept, got: {result['kept']}"
    )

    # (d) '12' does NOT appear in removed (removal failed).
    assert all(r['task_id'] != '12' for r in result['removed']), (
        f"unexpected '12' in removed: {result['removed']}"
    )

    # (e) '11' and '13' appear in removed with correct metadata.
    removed_ids = {r['task_id'] for r in result['removed']}
    assert removed_ids == {'11', '13'}, f"removed_ids={removed_ids}"
    for r in result['removed']:
        assert r['reason'] == 'intra_batch_duplicate'
        assert r['matched_task_id'] == '10'

    # (f) curator.curate called twice: pass-2 receives '10' and '12' (the
    # fall-through re-added '12' to unique_new_tasks); '11' and '13' were
    # removed in pass-1.
    assert curator_interceptor._curator.curate.await_count == 2, (
        f'curate call count wrong: {curator_interceptor._curator.curate.call_args_list}'
    )


@pytest.mark.asyncio
async def test_dedupe_bulk_intra_batch_partial_failure_curator_drop_routes_to_removed(
    curator_interceptor, taskmaster,
):
    """Partial-failure with curator='drop' must land the failing task in removed, not kept.

    Pins the corrected docstring contract: when tm.remove_task raises during
    pass-1 for a duplicate, the task falls through to pass-2 (unique_new_tasks).
    If the curator then returns action='drop', pass-2 re-issues tm.remove_task
    and the task lands in ``removed`` with reason='curator_drop' — NOT in
    ``kept``.  The task also appears in ``errors`` (dual-append contract).

    This is the curator='drop' sibling of
    test_dedupe_bulk_intra_batch_partial_failure_under_batched_lock_continues_remaining
    (which only exercises curator='create', so the failing task lands in kept).
    Together the two tests pin both branches of the ``errors ∪ (kept ∪ removed)``
    contract described in the _dedupe_bulk_created docstring.
    """
    PROJECT = '/project'
    post_snapshot = {'tasks': [
        {'id': '10', 'title': 'Fix foo', 'description': 'bar'},
        {'id': '11', 'title': 'FIX FOO', 'description': ' bar '},
        {'id': '12', 'title': 'fix Foo', 'description': 'BAR '},
        {'id': '13', 'title': 'Fix  foo', 'description': 'bar'},
    ]}

    taskmaster.get_tasks = AsyncMock(return_value=post_snapshot)

    # '12' raises on its first removal (pass-1); succeeds on the second call
    # (pass-2 curator-drop re-removal).  '11' and '13' succeed immediately.
    _remove_call_count: dict[str, int] = {}

    def _side_effect(tid, _project_root):
        _remove_call_count[tid] = _remove_call_count.get(tid, 0) + 1
        if tid == '12' and _remove_call_count[tid] == 1:
            raise RuntimeError('transient backend failure')
        return {'success': True}

    taskmaster.remove_task = AsyncMock(side_effect=_side_effect)

    # Per-candidate routing: '12' (title='fix Foo') is dropped onto '10';
    # '10' (title='Fix foo') is novel and kept.
    async def _route(candidate, *a, **kw):
        if candidate.title == 'fix Foo':
            return CuratorDecision(
                action='drop', target_id='10',
                justification='subsumed by intra-batch original',
            )
        return CuratorDecision(action='create', justification='novel')

    curator = _mock_curator(CuratorDecision(action='create', justification='novel'))
    curator.curate = AsyncMock(side_effect=_route)
    curator_interceptor._curator = curator

    result = await curator_interceptor._dedupe_bulk_created(
        PROJECT, pre_snapshot={'tasks': []},
    )

    # (a) remove_task called 4 times: 3 from pass-1 (one per dup, including the
    # failing '12') + 1 from pass-2's curator-drop re-removal of '12'.
    assert taskmaster.remove_task.await_count == 4, (
        f'remove_task call count wrong (expected 4): {taskmaster.remove_task.call_args_list}'
    )

    # (b) Exactly one error entry, for '12', mentioning the backend failure.
    assert len(result['errors']) == 1, f"expected 1 error, got: {result['errors']}"
    assert result['errors'][0]['task_id'] == '12'
    assert 'transient backend failure' in result['errors'][0]['error']

    # (c) '12' appears in removed with reason='curator_drop' and matched_task_id='10'.
    # PINS the corrected docstring contract: partial-failure + curator='drop' routes
    # the task to removed (not to kept, as the old docstring incorrectly claimed).
    removed_12 = next((r for r in result['removed'] if r['task_id'] == '12'), None)
    assert removed_12 is not None, f"expected '12' in removed, got: {result['removed']}"
    assert removed_12['reason'] == 'curator_drop'
    assert removed_12['matched_task_id'] == '10'

    # (d) '12' does NOT appear in kept (dual-append means errors + removed, not errors + kept).
    assert all(k['task_id'] != '12' for k in result['kept']), (
        f"unexpected '12' in kept: {result['kept']}"
    )

    # (e) '11' and '13' appear in removed with reason='intra_batch_duplicate'.
    for dup_id in ('11', '13'):
        dup = next((r for r in result['removed'] if r['task_id'] == dup_id), None)
        assert dup is not None, f"expected '{dup_id}' in removed, got: {result['removed']}"
        assert dup['reason'] == 'intra_batch_duplicate'
        assert dup['matched_task_id'] == '10'

    # curator.curate called twice: pass-2 receives '10' and the fall-through '12'.
    assert curator.curate.await_count == 2, (
        f'curate call count wrong: {curator.curate.call_args_list}'
    )
    # Pin the routing inputs so a title-key change fails loudly rather than
    # silently flipping the drop→create routing (CandidateTask has no id field;
    # call_args_list is the stable way to verify which candidates were routed).
    curate_call_titles = {c.args[0].title for c in curator.curate.call_args_list}
    assert curate_call_titles == {'Fix foo', 'fix Foo'}, (
        f'curator.curate received unexpected candidate titles {curate_call_titles!r}; '
        f'routing-key mismatch: task 12 must reach curator with title "fix Foo"'
    )


# ---------------------------------------------------------------------------
# step-7/9: path-scope guard integration tests
# ---------------------------------------------------------------------------
# These tests verify that the DarkFactoryPathScopeViolation guard is wired
# into submit_task (step-7) and add_subtask (step-9).
# They FAIL until steps 8 and 10 wire the guard into task_interceptor.py.
# ---------------------------------------------------------------------------


async def _cancel_interceptor_workers(ti) -> None:
    """Cancel any background ticket-worker tasks on *ti* and await them silently.

    Used in test teardown so fixture cleanup (DB close) never races a live worker.

    Why partial cleanup is sufficient here:
    The production ``TaskInterceptor.close()`` path does five additional things
    beyond cancelling worker tasks: (1) sets ``_closed = True``, (2) signals all
    pending ``_ticket_events`` so blocked ``resolve_ticket`` callers unblock,
    (3) drains fire-and-forget ``_background_tasks``, (4) closes the curator's
    Qdrant connection, and (5) closes the ``TicketStore`` SQLite connection.
    This helper skips all five because ``TestSubmitTaskGuardrail`` intentionally
    exercises none of those paths — no ``resolve_ticket`` waiters are registered,
    no curator is wired up, no fire-and-forget tasks are scheduled, and the
    ``ticket_store`` fixture (function-scoped) closes the DB in its own teardown.
    A class-level ``close()`` hook would race the fixture's per-test DB-close and
    risk spurious "closed database" errors.

    If a future change wires the suite up to any of these paths (resolve_ticket,
    curator, fire-and-forget tasks), or relies on close()-only side-effects to
    surface leaks, switch this teardown to ``await ti.close()`` instead.
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
        self, interceptor_with_store, ticket_store, taskmaster,
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
        self, interceptor_with_store, taskmaster,
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
        assert ticket_id.startswith('tkt_'), (
            f'Expected ticket id starting with tkt_, got: {result}'
        )
        assert 'error_type' not in result, (
            f'Should not have error_type for correctly-filed task: {result}'
        )

    @pytest.mark.asyncio
    async def test_submit_task_skips_build_candidate_for_dark_factory_project(
        self, interceptor_with_store, taskmaster, monkeypatch,
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
        self, interceptor_with_store, ticket_store, taskmaster,
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
        assert row is not None, (
            f'Expected persisted ticket row for ticket_id={result["ticket"]!r}'
        )
        assert row['project_id'] == 'dark_factory', (
            f"Expected project_id 'dark_factory', got: {row['project_id']!r}"
        )
        blob = json.loads(row['candidate_json'])
        assert blob['project_root'] == '/dark-factory', (
            f"Expected project_root '/dark-factory' in blob, got: {blob['project_root']!r}"
        )
        assert blob['kwargs']['title'] == 'Investigate orchestrator/harness.py deadlock', (
            f"Expected title in blob kwargs un-mutated, got: {blob['kwargs'].get('title')!r}"
        )
        assert blob['kwargs']['description'] == 'harness deadlock', (
            f"Expected description in blob kwargs un-mutated, got: {blob['kwargs'].get('description')!r}"
        )
        assert blob['metadata'] is None, (
            f"Expected metadata=None in blob (no metadata was passed), got: {blob['metadata']!r}"
        )

    @pytest.mark.asyncio
    async def test_submit_task_allows_clean_task_in_other_project(
        self, interceptor_with_store, taskmaster,
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
        assert ticket_id.startswith('tkt_'), (
            f'Expected ticket id starting with tkt_, got: {result}'
        )
        assert 'error_type' not in result

    @pytest.mark.asyncio
    async def test_submit_task_rejects_prompt_only_dark_factory_paths_in_wrong_project(
        self, interceptor_with_store, ticket_store, taskmaster,
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
        self, interceptor_with_store, taskmaster,
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
        assert ticket_id.startswith('tkt_'), (
            f'Expected ticket id starting with tkt_, got: {result}'
        )
        assert 'error_type' not in result, (
            f'Should not have error_type for dark_factory project: {result}'
        )

    @pytest.mark.asyncio
    async def test_submit_task_allows_clean_prompt_only_in_other_project(
        self, interceptor_with_store, taskmaster,
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
        assert ticket_id.startswith('tkt_'), (
            f'Expected ticket id starting with tkt_, got: {result}'
        )
        assert 'error_type' not in result, (
            f'Should not have error_type for clean prompt: {result}'
        )

    @pytest.mark.parametrize('field', ['prompt', 'description', 'details'])
    @pytest.mark.asyncio
    async def test_submit_task_rejects_dark_factory_path_in_any_fallback_field(
        self, field, interceptor_with_store, ticket_store, taskmaster,
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


class TestAddSubtaskGuardrail:
    """Integration tests: path-scope guard wired into add_subtask."""

    @pytest.mark.asyncio
    async def test_add_subtask_rejects_dark_factory_paths_in_wrong_project(
        self, interceptor, taskmaster,
    ):
        """add_subtask referencing fused-memory/ under a non-dark-factory project
        returns a DarkFactoryPathScopeViolation error and does NOT call taskmaster.
        """
        result = await interceptor.add_subtask(
            parent_id='1',
            project_root='/some-other-project',
            title='Edit fused-memory/src/fused_memory/middleware/task_curator.py',
            description='Fix drop logic',
        )

        assert isinstance(result, dict)
        assert result.get('error_type') == 'DarkFactoryPathScopeViolation', (
            f'Expected DarkFactoryPathScopeViolation error, got: {result}'
        )
        assert 'fused-memory/' in result.get('matched_paths', []) or \
               'fused_memory/' in result.get('matched_paths', []), (
            f'Expected fused-memory/ or fused_memory/ in matched_paths: {result}'
        )

        # Taskmaster backend must never have been called
        taskmaster.add_subtask.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_subtask_allows_dark_factory_paths_in_dark_factory_project(
        self, interceptor, taskmaster,
    ):
        """add_subtask with dark-factory paths filed under /dark-factory proceeds
        normally (project_id resolves to dark_factory and the guard no-ops).
        """
        result = await interceptor.add_subtask(
            parent_id='1',
            project_root='/dark-factory',
            title='Edit fused-memory/src/fused_memory/middleware/task_curator.py',
            description='Fix drop logic',
        )

        # Should return the taskmaster mock's add_subtask result
        assert isinstance(result, dict)
        assert 'error_type' not in result, (
            f'Should not have error_type for correctly-filed task: {result}'
        )
        # taskmaster.add_subtask should eventually be called (after curator)
        # We don't assert exact call count here since the curator may drop/combine.

    @pytest.mark.asyncio
    async def test_add_subtask_rejects_prompt_only_dark_factory_paths_in_wrong_project(
        self, interceptor, taskmaster,
    ):
        """A prompt-only add_subtask (no title) referencing fused-memory/ under a
        non-dark-factory project returns a DarkFactoryPathScopeViolation error
        and does NOT call taskmaster.add_subtask.
        """
        result = await interceptor.add_subtask(
            parent_id='1',
            project_root='/some-other-project',
            prompt='Edit fused-memory/src/fused_memory/middleware/task_curator.py',
            # Deliberately NO title kwarg — this is the prompt-only path
        )

        assert isinstance(result, dict)
        assert result.get('error_type') == 'DarkFactoryPathScopeViolation', (
            f'Expected DarkFactoryPathScopeViolation error, got: {result}'
        )
        matched = result.get('matched_paths', [])
        assert 'fused-memory/' in matched or 'fused_memory/' in matched, (
            f'Expected fused-memory/ or fused_memory/ in matched_paths: {result}'
        )

        # Taskmaster backend must never have been called
        taskmaster.add_subtask.assert_not_called()

    @pytest.mark.parametrize('field', ['prompt', 'description', 'details'])
    @pytest.mark.asyncio
    async def test_add_subtask_rejects_dark_factory_path_in_any_fallback_field(
        self, field, interceptor, taskmaster,
    ):
        """The fallback text guard scans prompt, description, AND details in add_subtask.

        Each parametrised case passes a dark-factory path in ``field`` with no
        title kwarg, routing _build_candidate to return None and engaging the
        fallback branch.  All three channels must trigger
        DarkFactoryPathScopeViolation.
        """
        result = await interceptor.add_subtask(
            parent_id='1',
            project_root='/some-other-project',
            **{field: 'Edit fused-memory/src/fused_memory/middleware/task_curator.py'},
            # Deliberately NO title — forces _build_candidate to return None
        )

        assert isinstance(result, dict)
        assert result.get('error_type') == 'DarkFactoryPathScopeViolation', (
            f'Field {field!r}: expected DarkFactoryPathScopeViolation, got: {result}'
        )
        matched = result.get('matched_paths', [])
        assert 'fused-memory/' in matched or 'fused_memory/' in matched, (
            f'Field {field!r}: expected fused-memory/ or fused_memory/ in matched_paths: {result}'
        )
        taskmaster.add_subtask.assert_not_called()


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

    def test_dict_metadata_modules_only(self):
        """dict metadata with only modules → returns modules (fallback)."""
        kwargs = {'metadata': {'modules': ['fused-memory/src', 'orchestrator/']}}
        result = TaskInterceptor._extract_meta_files(kwargs)
        assert result == ['fused-memory/src', 'orchestrator/']

    def test_dict_metadata_both_keys_prefers_files_to_modify(self):
        """dict metadata with BOTH keys → returns files_to_modify (precedence over modules)."""
        kwargs = {'metadata': {
            'files_to_modify': ['a.py'],
            'modules': ['module_a'],
        }}
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
        """_build_candidate must call _parse_metadata exactly once per invocation.

        Regression guard for the hot-path dedupe: before the fix, _build_candidate
        called _parse_metadata directly (line 909) AND indirectly via
        _extract_meta_files (line 910) — two parses per title-bearing submission.
        After the fix (_build_candidate delegates to _extract_meta_files_from_meta
        using the meta already in scope), the count drops to 1.
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
        assert call_count[0] == 1, (
            f'_parse_metadata should be called exactly once by _build_candidate, '
            f'but was called {call_count[0]} time(s)'
        )


# ---------------------------------------------------------------------------
# Regression tests — prompt-only fallback must also scan metadata files
# ---------------------------------------------------------------------------


class TestPathGuardFallbackMetadataFiles:
    """Regression tests: prompt-only path-guard also scans metadata files/modules.

    4 parametrised cases (2 meta_key × 2 endpoint) verify that hiding a
    dark-factory path inside metadata['files_to_modify'] or metadata['modules']
    cannot bypass the path-scope guard when the free-text fields are clean.
    """

    @pytest.mark.parametrize('meta_key', ['files_to_modify', 'modules'])
    @pytest.mark.asyncio
    async def test_submit_task_fallback_rejects_dark_factory_path_in_metadata(
        self, meta_key, interceptor_with_store, ticket_store, taskmaster,
    ):
        """prompt-only submit_task with dark-factory path ONLY in metadata[meta_key]
        must be rejected even though all free-text fields are clean.
        """
        try:
            result = await interceptor_with_store.submit_task(
                project_root='/some-other-project',
                prompt='Generic refactor',            # no dark-factory path here
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
        assert row[0] == 0, (
            f'meta_key={meta_key!r}: expected 0 tickets in store, found {row[0]}'
        )

        # Taskmaster backend must never have been called
        taskmaster.add_task.assert_not_called()

    @pytest.mark.parametrize('meta_key', ['files_to_modify', 'modules'])
    @pytest.mark.asyncio
    async def test_add_subtask_fallback_rejects_dark_factory_path_in_metadata(
        self, meta_key, interceptor, taskmaster,
    ):
        """prompt-only add_subtask with dark-factory path ONLY in metadata[meta_key]
        must be rejected even though all free-text fields are clean.
        """
        result = await interceptor.add_subtask(
            parent_id='1',
            project_root='/some-other-project',
            prompt='Generic refactor',            # no dark-factory path here
            # Deliberately NO title — forces _build_candidate → None → fallback
            metadata={meta_key: ['orchestrator/harness.py']},
        )

        assert isinstance(result, dict)
        assert result.get('error_type') == 'DarkFactoryPathScopeViolation', (
            f'meta_key={meta_key!r}: expected DarkFactoryPathScopeViolation, got: {result}'
        )
        assert 'orchestrator/' in result.get('matched_paths', []), (
            f'meta_key={meta_key!r}: expected orchestrator/ in matched_paths: {result}'
        )

        # Taskmaster backend must never have been called
        taskmaster.add_subtask.assert_not_called()

    @pytest.mark.asyncio
    async def test_path_guard_detects_dark_factory_path_in_multi_entry_metadata_list(
        self, interceptor, taskmaster,
    ):
        """Dark-factory path in the SECOND entry of a multi-entry metadata list
        must still trigger the guard.

        Regression against a future scanner that stops scanning after the first
        entry (or treats the joined text as a single path): if the join-with-newlines
        approach ever stops scanning each entry independently, this test catches it.
        """
        result = await interceptor.add_subtask(
            parent_id='1',
            project_root='/some-other-project',
            prompt='Generic refactor',
            # The dark-factory path is the SECOND entry; the first is clean.
            metadata={'files_to_modify': ['non_df_module/file.py', 'orchestrator/harness.py']},
        )

        assert isinstance(result, dict)
        assert result.get('error_type') == 'DarkFactoryPathScopeViolation', (
            'expected DarkFactoryPathScopeViolation when dark-factory path is '
            f'second entry in metadata list, got: {result}'
        )
        assert 'orchestrator/' in result.get('matched_paths', []), (
            f'expected orchestrator/ in matched_paths: {result}'
        )
        taskmaster.add_subtask.assert_not_called()


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
        self, interceptor_with_store, taskmaster,
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
        assert ticket_id.startswith('tkt_'), (
            f'Expected ticket id starting with tkt_, got: {result}'
        )
        assert 'error_type' not in result, (
            f'Should not have error_type for clean metadata files: {result}'
        )

    @pytest.mark.asyncio
    async def test_add_subtask_allows_clean_metadata_files_in_other_project(
        self, interceptor, taskmaster,
    ):
        """prompt-only add_subtask with non-dark-factory paths in metadata
        must NOT be rejected — only dark-factory paths should trigger the guard.
        """
        result = await interceptor.add_subtask(
            parent_id='1',
            project_root='/some-other-project',
            prompt='Refactor foo/bar.py routing',
            # No title — prompt-only path
            metadata={'files_to_modify': ['foo/bar.py', 'src/baz.py']},
        )

        assert isinstance(result, dict)
        assert 'error_type' not in result, (
            f'Should not have error_type for clean metadata files: {result}'
        )


# ---------------------------------------------------------------------------
# Unit tests for _path_guard_or_skip helper
# ---------------------------------------------------------------------------


class TestPathGuardOrSkip:
    """Unit tests for TaskInterceptor._path_guard_or_skip.

    Verifies the helper's contract: dark_factory short-circuit, lazy-build
    of candidate, pass-through of a pre-built candidate, and error propagation.
    """

    # -- Case 1 -----------------------------------------------------------
    def test_path_guard_or_skip_returns_none_for_dark_factory_project(
        self, interceptor, monkeypatch,
    ):
        """dark_factory short-circuit: returns None without calling
        _build_candidate or _path_guard_error.
        """
        build_calls: list = []
        guard_calls: list = []

        def fake_build(kwargs):
            build_calls.append(kwargs)
            return None

        def fake_guard(self, candidate, kwargs, project_id):
            guard_calls.append((candidate, kwargs, project_id))
            return None

        monkeypatch.setattr(TaskInterceptor, '_build_candidate', staticmethod(fake_build))
        monkeypatch.setattr(TaskInterceptor, '_path_guard_error', fake_guard)

        result = interceptor._path_guard_or_skip(
            {'title': 'Edit orchestrator/harness.py'}, 'dark_factory',
        )

        assert result is None
        assert build_calls == [], '_build_candidate must NOT be called for dark_factory'
        assert guard_calls == [], '_path_guard_error must NOT be called for dark_factory'

    # -- Case 2 -----------------------------------------------------------
    def test_path_guard_or_skip_lazy_builds_candidate_when_unset(
        self, interceptor, monkeypatch,
    ):
        """When no candidate is supplied and project is non-dark_factory, the
        helper builds a candidate via _build_candidate and passes it to
        _path_guard_error.
        """
        from fused_memory.middleware.task_curator import CandidateTask

        built = CandidateTask(
            title='Generic refactor', description='', details='',
            files_to_modify=[], priority='medium',
        )
        build_calls: list = []
        guard_calls: list = []

        def fake_build(kwargs):
            build_calls.append(kwargs)
            return built

        def fake_guard(self, candidate, kwargs, project_id):
            guard_calls.append((candidate, kwargs, project_id))
            return None

        monkeypatch.setattr(TaskInterceptor, '_build_candidate', staticmethod(fake_build))
        monkeypatch.setattr(TaskInterceptor, '_path_guard_error', fake_guard)

        kwargs = {'title': 'Generic refactor'}
        result = interceptor._path_guard_or_skip(kwargs, 'some_other_project')

        assert result is None
        assert len(build_calls) == 1, (
            f'Expected _build_candidate called once, got {len(build_calls)}'
        )
        assert build_calls[0] is kwargs
        assert len(guard_calls) == 1, (
            f'Expected _path_guard_error called once, got {len(guard_calls)}'
        )
        assert guard_calls[0][0] is built

    # -- Case 3 -----------------------------------------------------------
    def test_path_guard_or_skip_uses_provided_candidate(
        self, interceptor, monkeypatch,
    ):
        """When a pre-built candidate is supplied, _build_candidate is NOT called;
        _path_guard_error is called with the supplied candidate.
        """
        from fused_memory.middleware.task_curator import CandidateTask

        sentinel = CandidateTask(
            title='Sentinel', description='', details='',
            files_to_modify=[], priority='medium',
        )
        build_calls: list = []
        guard_calls: list = []

        def fake_build(kwargs):
            build_calls.append(kwargs)
            return None

        def fake_guard(self, candidate, kwargs, project_id):
            guard_calls.append((candidate, kwargs, project_id))
            return None

        monkeypatch.setattr(TaskInterceptor, '_build_candidate', staticmethod(fake_build))
        monkeypatch.setattr(TaskInterceptor, '_path_guard_error', fake_guard)

        kwargs = {'title': 'Generic refactor'}
        result = interceptor._path_guard_or_skip(kwargs, 'some_other_project', candidate=sentinel)

        assert result is None
        assert build_calls == [], '_build_candidate must NOT be called when candidate is supplied'
        assert len(guard_calls) == 1, (
            f'Expected _path_guard_error called once, got {len(guard_calls)}'
        )
        assert guard_calls[0][0] is sentinel

    # -- Case 4 -----------------------------------------------------------
    def test_path_guard_or_skip_propagates_rejection(
        self, interceptor, monkeypatch,
    ):
        """When _path_guard_error returns a rejection dict, the helper returns it as-is."""
        rejection = {
            'error_type': 'DarkFactoryPathScopeViolation',
            'matched_paths': ['orchestrator/'],
        }

        def fake_build(kwargs):
            return None

        def fake_guard(self, candidate, kwargs, project_id):
            return rejection

        monkeypatch.setattr(TaskInterceptor, '_build_candidate', staticmethod(fake_build))
        monkeypatch.setattr(TaskInterceptor, '_path_guard_error', fake_guard)

        result = interceptor._path_guard_or_skip({'prompt': 'something'}, 'some_other_project')
        assert result is rejection
