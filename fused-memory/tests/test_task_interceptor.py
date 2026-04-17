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
        action='combine', target_id='50', rewritten_task=rewritten,
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
        CandidateTask,
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
