"""Tests for task interceptor middleware."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from fused_memory.config.schema import CuratorConfig, FusedMemoryConfig
from fused_memory.middleware.task_curator import CuratorDecision, RewrittenTask
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
async def test_add_task_persists_metadata_atomically(interceptor, taskmaster):
    """R5: add_task with metadata forwards it to tm.add_task in one call.

    The racy two-step pattern (add_task then update_task(metadata=...)) is
    gone; metadata must be written atomically to prevent a concurrent
    reader from observing a task without its files_to_modify — the bug
    that left #1922/#1923/#1924 running in parallel.
    """
    import json

    metadata = {'source': 'review-cycle', 'modules': ['fused-memory/src']}
    result = await interceptor.add_task('/project', prompt='Test', metadata=metadata)
    assert result == {'id': '2', 'title': 'New Task'}
    taskmaster.add_task.assert_called_once()
    kwargs = taskmaster.add_task.call_args.kwargs
    # Metadata forwarded as a JSON string (the MCP wire format).
    assert kwargs.get('metadata') == json.dumps(metadata)
    # No follow-up update_task for metadata — the atomic path wrote it.
    taskmaster.update_task.assert_not_called()


@pytest.mark.asyncio
async def test_add_task_metadata_string_passed_through(interceptor, taskmaster):
    """Pre-serialised metadata JSON is forwarded unchanged."""
    metadata_json = '{"escalation_id":"esc-1","suggestion_hash":"x"}'
    await interceptor.add_task(
        '/project', prompt='Test', metadata=metadata_json,
    )
    kwargs = taskmaster.add_task.call_args.kwargs
    assert kwargs.get('metadata') == metadata_json


@pytest.mark.asyncio
async def test_add_task_without_metadata_skips_update(interceptor, taskmaster):
    """add_task without metadata does not call update_task."""
    await interceptor.add_task('/project', prompt='Test')
    taskmaster.update_task.assert_not_called()
    # Backend still receives metadata=None kwarg but the value is falsy.
    kwargs = taskmaster.add_task.call_args.kwargs
    assert kwargs.get('metadata') in (None, '')


@pytest.mark.asyncio
async def test_add_task_falls_back_to_two_step_on_typeerror(event_buffer):
    """Legacy fallback: a backend that rejects ``metadata=`` still works.

    ``TaskmasterBackend.add_task`` on older installs may not accept the
    new ``metadata`` kwarg (the taskmaster-ai MCP tool was extended in
    R5). Keep the fallback during rollout so mixed versions don't break.
    """
    import json

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

    interceptor = TaskInterceptor(tm, None, event_buffer)
    metadata = {'escalation_id': 'esc-x', 'suggestion_hash': 'h'}
    await interceptor.add_task('/project', prompt='Test', metadata=metadata)

    # Two add_task attempts: atomic first (with metadata), retry without.
    assert len(call_log) == 2
    assert 'metadata' in call_log[0]
    assert 'metadata' not in call_log[1]
    # Legacy update_task follow-up ran because atomic write failed.
    tm.update_task.assert_called_once()
    kwargs = tm.update_task.call_args.kwargs
    assert kwargs['task_id'] == '7'
    assert kwargs['metadata'] == json.dumps(metadata)


# ─────────────────────────────────────────────────────────────────────
# WP-B: fire-and-forget event queue
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_task_with_queue_persists_to_real_sqlite(taskmaster, tmp_path):
    """WP-B smoke: end-to-end through real EventQueue + real EventBuffer.

    No mocks on the journal path — this catches wiring mistakes that
    unit tests with AsyncMock(EventBuffer) would miss.
    """
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

    interceptor = TaskInterceptor(taskmaster, None, buf, event_queue=queue)
    try:
        await interceptor.add_task('/project', prompt='Test 1')
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

    interceptor = TaskInterceptor(taskmaster, None, buf, event_queue=queue)
    try:
        t0 = time.perf_counter()
        result = await interceptor.add_task('/project', prompt='Test')
        elapsed = time.perf_counter() - t0
        # Canonical write returned successfully — no exception from lock.
        assert result == {'id': '2', 'title': 'New Task'}
        # Under 500ms budget even with SQLite pinned.
        assert elapsed < 0.5, f'hot path took {elapsed:.3f}s under lock'
        # The event is either queued for retry or already dead-lettered,
        # but NOT raised to the caller.
        stats = queue.stats()
        assert stats['queue_depth'] + stats['dead_letters'] + stats['events_committed'] >= 1
    finally:
        await queue.close()


# ─────────────────────────────────────────────────────────────────────
# Curator gate integration
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def curator_enabled_config():
    cfg = FusedMemoryConfig()
    cfg.curator = CuratorConfig(enabled=True)
    return cfg


@pytest.fixture
def curator_interceptor(taskmaster, reconciler, event_buffer, curator_enabled_config):
    return TaskInterceptor(
        taskmaster, reconciler, event_buffer, config=curator_enabled_config,
    )


def _mock_curator(decision: CuratorDecision) -> MagicMock:
    """Mock TaskCurator returning a fixed decision."""
    curator = MagicMock()
    curator.curate = AsyncMock(return_value=decision)
    curator.record_task = AsyncMock()
    curator.reembed_task = AsyncMock()
    # note_created is a plain sync method on the real TaskCurator.
    curator.note_created = MagicMock()
    return curator


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

    result = await curator_interceptor.add_task(
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

    result = await curator_interceptor.add_task(
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

    result = await curator_interceptor.add_task(
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

    result = await curator_interceptor.add_task(
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

    result = await curator_interceptor.add_task('/project', title='candidate')

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

    result = await curator_interceptor.add_task('/project', title='Fix x')

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

    result = await curator_interceptor.add_task('/project', title='Fix x')

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

    result = await curator_interceptor.add_task('/project', title='Fix x')

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

    result = await curator_interceptor.add_task('/project', title='Fix x')

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

    result = await curator_interceptor.add_task('/project', title='c')

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
    taskmaster, reconciler, event_buffer, curator_enabled_config,
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

    interceptor = TaskInterceptor(
        taskmaster, reconciler, event_buffer, config=curator_enabled_config,
    )
    interceptor._curator = real_curator

    # Give each add_task its own unique task_id. First call creates '100',
    # second (if it ever reaches tm.add_task) would create '101'.
    add_task_counter = {'n': 99}

    async def fake_add_task(**kwargs):
        add_task_counter['n'] += 1
        return {'id': str(add_task_counter['n']), 'title': 'x'}

    taskmaster.add_task = fake_add_task

    candidate_kwargs = dict(
        title='Log release-mode warning on duplicate template names',
        description='...',
    )

    results = await asyncio.gather(
        interceptor.add_task('/project', **candidate_kwargs),
        interceptor.add_task('/project', **candidate_kwargs),
    )

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

    await curator_interceptor.add_task('/project', title='Fresh work')

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
async def test_idempotency_hit_skips_curator_and_returns_existing(
    taskmaster, reconciler, event_buffer, curator_enabled_config,
):
    """R4: steward-requeue duplicate suggestion → existing task id.

    Simulates `esc-1912-179 → esc-1912-190` from plan §R4: a re-queued
    triage sends the same suggestion with a stamped
    ``(escalation_id, suggestion_hash)`` tuple. The interceptor must
    find the previously-created task and return its id without the
    curator firing.
    """
    # Existing task carries metadata with the idempotency keys.
    taskmaster.get_tasks = AsyncMock(return_value={
        'tasks': [
            {
                'id': '555',
                'title': 'Add Type::Error defensive arm(s)',
                'status': 'pending',
                'metadata': {
                    'escalation_id': 'esc-1912-179',
                    'suggestion_hash': 'abcd1234abcd1234',
                },
            },
        ],
    })

    decision = CuratorDecision(action='create', justification='novel')
    curator_mock = _mock_curator(decision)

    interceptor = TaskInterceptor(
        taskmaster, reconciler, event_buffer, config=curator_enabled_config,
    )
    interceptor._curator = curator_mock

    metadata = {
        'escalation_id': 'esc-1912-179',
        'suggestion_hash': 'abcd1234abcd1234',
        'modules': ['crates/reify-compiler'],
    }
    result = await interceptor.add_task(
        '/project',
        title='Add Type::Error defensive arm(s)',
        description='same suggestion from requeued triage',
        metadata=metadata,
    )

    assert result['id'] == '555'
    assert result['deduplicated'] is True
    assert result['action'] == 'idempotency_hit'
    # Curator must not have been consulted.
    curator_mock.curate.assert_not_called()
    # Taskmaster add_task never ran.
    taskmaster.add_task.assert_not_called()


@pytest.mark.asyncio
async def test_idempotency_accepts_metadata_as_json_string(
    taskmaster, reconciler, event_buffer, curator_enabled_config,
):
    """Metadata that arrives as a pre-serialised JSON string also dedupes."""
    import json

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

    interceptor = TaskInterceptor(
        taskmaster, reconciler, event_buffer, config=curator_enabled_config,
    )
    interceptor._curator = _mock_curator(CuratorDecision(action='create'))

    meta_str = json.dumps({'escalation_id': 'esc-x', 'suggestion_hash': 'hash1'})
    result = await interceptor.add_task('/project', title='T', metadata=meta_str)
    assert result['id'] == '555'
    assert result['action'] == 'idempotency_hit'


@pytest.mark.asyncio
async def test_idempotency_miss_falls_through_to_curator(
    taskmaster, reconciler, event_buffer, curator_enabled_config,
):
    """No matching (escalation_id, suggestion_hash) → curator runs normally."""
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
    interceptor = TaskInterceptor(
        taskmaster, reconciler, event_buffer, config=curator_enabled_config,
    )
    interceptor._curator = curator_mock

    await interceptor.add_task(
        '/project', title='New',
        metadata={'escalation_id': 'esc-new', 'suggestion_hash': 'fresh'},
    )
    curator_mock.curate.assert_called_once()
    taskmaster.add_task.assert_called_once()


@pytest.mark.asyncio
async def test_idempotency_skips_cancelled_match(
    taskmaster, reconciler, event_buffer, curator_enabled_config,
):
    """A cancelled task with matching metadata must not win the dedupe."""
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
    interceptor = TaskInterceptor(
        taskmaster, reconciler, event_buffer, config=curator_enabled_config,
    )
    interceptor._curator = curator_mock

    await interceptor.add_task(
        '/project', title='Retry',
        metadata={'escalation_id': 'esc-y', 'suggestion_hash': 'hash-y'},
    )
    curator_mock.curate.assert_called_once()


@pytest.mark.asyncio
async def test_idempotency_requires_both_keys(
    taskmaster, reconciler, event_buffer, curator_enabled_config,
):
    """Metadata without escalation_id+suggestion_hash skips the R4 check."""
    curator_mock = _mock_curator(CuratorDecision(action='create', justification='novel'))
    interceptor = TaskInterceptor(
        taskmaster, reconciler, event_buffer, config=curator_enabled_config,
    )
    interceptor._curator = curator_mock

    # Only escalation_id, no suggestion_hash → not eligible.
    await interceptor.add_task(
        '/project', title='T', metadata={'escalation_id': 'esc-x'},
    )
    curator_mock.curate.assert_called_once()
    # get_tasks for the idempotency check should not have been invoked
    # because we bail before the walk when a key is missing.
    # (get_tasks may still be called by curator _build_corpus under some
    # paths — our curator_mock stubs that; so the AsyncMock
    # ``taskmaster.get_tasks`` call count must be zero.)
    assert taskmaster.get_tasks.call_count == 0


@pytest.mark.asyncio
async def test_curator_disabled_still_proxies(taskmaster, reconciler, event_buffer):
    """With curator.enabled=False, add_task proxies straight to Taskmaster."""
    cfg = FusedMemoryConfig()
    cfg.curator = CuratorConfig(enabled=False)
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer, config=cfg)

    result = await interceptor.add_task('/project', title='T')

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
async def test_done_gate_unwraps_data_envelope(
    taskmaster, reconciler, event_buffer, tmp_path
):
    """Gate handles the {'data': {...}} wrapper shape used by some taskmaster responses."""
    taskmaster.get_task = AsyncMock(
        return_value={
            'data': {
                'id': '7',
                'status': 'in-progress',
                'metadata': {'files': ['ghost.py']},
            }
        }
    )
    interceptor = TaskInterceptor(taskmaster, reconciler, event_buffer)

    result = await interceptor.set_task_status('7', 'done', str(tmp_path))

    assert result['success'] is False
    assert result['missing_files'] == ['ghost.py']


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
    overlap_tm, reconciler, event_buffer,
):
    """WP-E: 20 concurrent add_task calls to the same project serialise
    through the per-project lock — every task gets a distinct id and the
    taskmaster backend never sees overlapping invocations."""
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

    interceptor = TaskInterceptor(overlap_tm, reconciler, event_buffer)

    N = 20
    results = await asyncio.gather(*[
        interceptor.add_task('/project', title=f'Task {i}')
        for i in range(N)
    ])

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
        coros.append(interceptor.add_task('/project', title=f'A{i}'))
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
    overlap_tm, reconciler, event_buffer,
):
    """WP-E: the lock is per-project, so ops on distinct projects can
    overlap. Fire concurrent add_task bursts on /projA and /projB and
    observe total peak >= 2."""
    from fused_memory.models.scope import resolve_project_id

    tracker = _OverlapTracker()
    assert resolve_project_id('/projA') != resolve_project_id('/projB')

    async def side_effect(**kwargs):
        pr = kwargs.get('project_root', '')
        key = resolve_project_id(pr)
        tracker.in_flight[key] = tracker.in_flight.get(key, 0) + 1
        tracker._global_in_flight += 1
        tracker.total_peak = max(tracker.total_peak, tracker._global_in_flight)
        try:
            # Enough ticks for the other project's task to enter
            for _ in range(10):
                await asyncio.sleep(0)
            return {'id': '1', 'title': kwargs.get('title', '')}
        finally:
            tracker.in_flight[key] -= 1
            tracker._global_in_flight -= 1

    overlap_tm.add_task = AsyncMock(side_effect=side_effect)
    interceptor = TaskInterceptor(overlap_tm, reconciler, event_buffer)

    coros = []
    for i in range(5):
        coros.append(interceptor.add_task('/projA', title=f'a{i}'))
        coros.append(interceptor.add_task('/projB', title=f'b{i}'))
    await asyncio.gather(*coros)

    # Per-project peak is 1 (lock held), cross-project overlap is allowed.
    peak_a = tracker.peak.get(resolve_project_id('/projA'), 0)
    peak_b = tracker.peak.get(resolve_project_id('/projB'), 0)
    assert peak_a <= 1 and peak_b <= 1, (
        f'same-project overlap: {tracker.peak}'
    )
    assert tracker.total_peak >= 2, (
        f'two projects never ran concurrently: total_peak={tracker.total_peak}'
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
        interceptor.add_task('/project', title='concurrent add'),
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
    # 2s; bumping for CI jitter.
    assert elapsed < 2.0, f'{N} sequential calls took {elapsed:.3f}s'
