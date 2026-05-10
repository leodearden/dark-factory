"""Tests for targeted reconciliation."""

import re
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
from fused_memory.models.enums import SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.models.reconciliation import VerificationResult, VerificationVerdict
from fused_memory.reconciliation.journal import ReconciliationJournal
from fused_memory.reconciliation.targeted import TargetedReconciler


@pytest_asyncio.fixture
async def journal(tmp_path):
    j = ReconciliationJournal(tmp_path / 'targeted_test')
    await j.initialize()
    yield j
    await j.close()


@pytest.fixture
def mock_memory_service():
    svc = AsyncMock()
    svc.search = AsyncMock(return_value=[])
    svc.add_memory = AsyncMock(return_value=AsyncMock(model_dump=lambda: {}))
    return svc


@pytest.fixture
def mock_taskmaster():
    tm = AsyncMock()
    tm.get_tasks = AsyncMock(return_value={'tasks': []})
    tm.update_task = AsyncMock(return_value={'success': True})
    return tm


@pytest.fixture
def config():
    return FusedMemoryConfig(
        reconciliation=ReconciliationConfig(
            enabled=True,
            explore_codebase_root='/tmp/test',
            agent_llm_provider='anthropic',
            agent_llm_model='claude-sonnet-4-20250514',
        )
    )


@pytest.fixture
def mock_event_buffer():
    buf = AsyncMock()
    buf.is_full_recon_active = AsyncMock(return_value=False)
    buf.defer_write = AsyncMock(return_value='deferred-id')
    return buf


@pytest.fixture
def reconciler(mock_memory_service, mock_taskmaster, journal, config, mock_event_buffer):
    r = TargetedReconciler(
        mock_memory_service, mock_taskmaster, journal, config, mock_event_buffer,
    )
    # Mock the verifier to avoid actual LLM calls
    r.verifier = AsyncMock()
    r.verifier.verify = AsyncMock(return_value=VerificationResult(
        verdict=VerificationVerdict.confirmed,
        confidence=0.9,
        evidence=[{'file_path': 'test.py', 'snippet': 'def test()'}],
        summary='Confirmed via test.py',
    ))
    return r


@pytest.mark.asyncio
async def test_on_task_done_fast_path_write(reconciler, mock_memory_service):
    """Done transition writes completion fact immediately before search/verify."""
    task_before = {'id': '1', 'title': 'Add tests', 'status': 'in-progress', 'description': 'Test suite'}
    result = await reconciler.reconcile_task(
        task_id='1', transition='done', project_id='test-project', project_root='/tmp/test',
        task_before=task_before,
    )
    # First call should be the fast-path write (before any search)
    calls = mock_memory_service.add_memory.call_args_list
    assert len(calls) >= 1
    first_call = calls[0]
    assert 'observations_and_summaries' in str(first_call)
    assert any(a['type'] == 'knowledge_captured_fast' for a in result.get('actions', []))


@pytest.mark.asyncio
async def test_on_task_done_passes_causation_id(reconciler, mock_memory_service):
    """All memory calls during targeted recon pass causation_id=run_id."""
    task_before = {'id': '1', 'title': 'Test', 'status': 'in-progress'}
    await reconciler.reconcile_task(
        task_id='1', transition='done', project_id='test-project', project_root='/tmp/test',
        task_before=task_before,
    )
    for call in mock_memory_service.add_memory.call_args_list:
        assert call.kwargs.get('causation_id') is not None, (
            f'add_memory called without causation_id: {call}'
        )
        assert call.kwargs.get('_source') == 'targeted_recon'
    for call in mock_memory_service.search.call_args_list:
        assert call.kwargs.get('causation_id') is not None


@pytest.mark.asyncio
async def test_on_task_done_logs_run_actions(reconciler, journal):
    """Targeted recon logs run_actions to journal."""
    task_before = {'id': '1', 'title': 'Test', 'status': 'in-progress'}
    await reconciler.reconcile_task(
        task_id='1', transition='done', project_id='test-project', project_root='/tmp/test',
        task_before=task_before,
    )
    # Find the run
    runs = await journal.get_recent_runs('test-project', limit=1)
    assert len(runs) == 1
    actions = await journal.get_run_actions(runs[0].id)
    # At minimum: fast-path write + search
    assert len(actions) >= 2
    ops = {a['operation'] for a in actions}
    assert 'add_memory' in ops
    assert 'search' in ops


@pytest.mark.asyncio
async def test_on_task_done_sparse_knowledge(reconciler, mock_memory_service):
    """When task is done and knowledge is sparse, should verify and write."""
    task_before = {'id': '1', 'title': 'Add tests', 'status': 'in-progress', 'description': 'Test suite'}

    result = await reconciler.reconcile_task(
        task_id='1', transition='done', project_id='test-project', project_root='/tmp/test',
        task_before=task_before,
    )

    assert any(a['type'] == 'knowledge_captured' for a in result.get('actions', []))
    # Fast-path write + verification write = at least 2 calls
    assert mock_memory_service.add_memory.call_count >= 2


@pytest.mark.asyncio
async def test_on_task_done_rich_knowledge(reconciler, mock_memory_service):
    """When task is done and knowledge is rich, should not verify."""
    mock_memory_service.search = AsyncMock(return_value=[
        MemoryResult(id='1', content='test', source_store=SourceStore.mem0),
        MemoryResult(id='2', content='test2', source_store=SourceStore.mem0),
        MemoryResult(id='3', content='test3', source_store=SourceStore.graphiti),
    ])

    await reconciler.reconcile_task(
        task_id='1', transition='done', project_id='test-project', project_root='/tmp/test',
        task_before={'id': '1', 'title': 'Test', 'status': 'in-progress'},
    )

    reconciler.verifier.verify.assert_not_called()


@pytest.mark.asyncio
async def test_on_task_done_checks_dependents(reconciler, mock_taskmaster):
    """Should check for unblocked dependents when task completes."""
    mock_taskmaster.get_tasks = AsyncMock(return_value={
        'tasks': [
            {'id': '1', 'status': 'done', 'dependencies': []},
            {'id': '2', 'status': 'pending', 'title': 'Next task', 'dependencies': ['1']},
        ]
    })

    result = await reconciler.reconcile_task(
        task_id='1', transition='done', project_id='test-project', project_root='/tmp/test',
        task_before={'id': '1', 'title': 'Dep task', 'status': 'in-progress'},
    )

    unblocked = [a for a in result.get('actions', []) if a['type'] == 'dependent_unblocked']
    assert len(unblocked) == 1
    assert unblocked[0]['task_id'] == '2'


@pytest.mark.asyncio
async def test_on_task_blocked_attaches_hints(reconciler, mock_memory_service, mock_taskmaster):
    """Blocked task should get memory hints attached."""
    mock_memory_service.search = AsyncMock(return_value=[
        MemoryResult(id='1', content='relevant info', source_store=SourceStore.mem0, entities=['EntityA']),
    ])

    result = await reconciler.reconcile_task(
        task_id='1', transition='blocked', project_id='test-project', project_root='/tmp/test',
        task_before={'id': '1', 'title': 'Blocked task', 'status': 'in-progress'},
    )

    hints_actions = [a for a in result.get('actions', []) if a['type'] == 'hints_attached']
    assert len(hints_actions) == 1
    mock_taskmaster.update_task.assert_called_once()


@pytest.mark.asyncio
async def test_on_task_cancelled_checks_subtasks(reconciler, mock_taskmaster):
    """Cancelled task should flag active subtasks for review."""
    result = await reconciler.reconcile_task(
        task_id='1', transition='cancelled', project_id='test-project', project_root='/tmp/test',
        task_before={
            'id': '1', 'title': 'Cancelled', 'status': 'in-progress',
            'subtasks': [
                {'id': '1.1', 'status': 'pending', 'title': 'Sub1'},
                {'id': '1.2', 'status': 'done', 'title': 'Sub2'},
            ],
        },
    )

    review_actions = [a for a in result.get('actions', []) if a['type'] == 'subtasks_need_review']
    assert len(review_actions) == 1
    assert review_actions[0]['count'] == 1


@pytest.mark.asyncio
async def test_on_task_deferred_same_as_blocked(reconciler, mock_memory_service, mock_taskmaster):
    """Deferred should behave like blocked (attach hints)."""
    mock_memory_service.search = AsyncMock(return_value=[
        MemoryResult(id='1', content='info', source_store=SourceStore.mem0, entities=['X']),
    ])

    result = await reconciler.reconcile_task(
        task_id='1', transition='deferred', project_id='test-project', project_root='/tmp/test',
        task_before={'id': '1', 'title': 'Deferred task', 'status': 'in-progress'},
    )

    hints_actions = [a for a in result.get('actions', []) if a['type'] == 'hints_attached']
    assert len(hints_actions) == 1


@pytest.mark.asyncio
async def test_reconcile_task_failure_handling(reconciler, journal):
    """Failure during reconciliation should be caught and recorded."""
    reconciler.verifier.verify = AsyncMock(side_effect=Exception('LLM error'))

    result = await reconciler.reconcile_task(
        task_id='1', transition='done', project_id='test-project', project_root='/tmp/test',
        task_before={'id': '1', 'title': 'Failing', 'status': 'in-progress'},
    )

    # Should still return a result (no unhandled exception)
    assert 'task_id' in result


# ── Tests for project_id / project_root separation (step-5) ──────────


@pytest.mark.asyncio
async def test_done_memory_ops_use_project_id(reconciler, mock_memory_service):
    """Memory calls (add_memory, search) should use logical project_id, not filesystem path."""
    await reconciler.reconcile_task(
        task_id='1',
        transition='done',
        project_id='dark_factory',
        project_root='/home/leo/src/dark-factory',
        task_before={'id': '1', 'title': 'Test', 'status': 'in-progress'},
    )

    # All add_memory calls should use the logical project_id
    for call in mock_memory_service.add_memory.call_args_list:
        assert call.kwargs.get('project_id') == 'dark_factory', (
            f'add_memory called with wrong project_id: {call}'
        )
    # search should use logical project_id
    for call in mock_memory_service.search.call_args_list:
        assert call.kwargs.get('project_id') == 'dark_factory', (
            f'search called with wrong project_id: {call}'
        )


@pytest.mark.asyncio
async def test_done_task_ops_use_project_root(reconciler, mock_taskmaster):
    """Taskmaster calls (get_tasks) should use filesystem project_root, not logical id."""
    await reconciler.reconcile_task(
        task_id='1',
        transition='done',
        project_id='dark_factory',
        project_root='/home/leo/src/dark-factory',
        task_before={'id': '1', 'title': 'Test', 'status': 'in-progress'},
    )

    # get_tasks should use the filesystem path
    mock_taskmaster.get_tasks.assert_called_once_with(
        project_root='/home/leo/src/dark-factory'
    )


@pytest.mark.asyncio
async def test_blocked_task_update_uses_project_root(reconciler, mock_memory_service, mock_taskmaster):
    """Hints attachment via taskmaster.update_task should use project_root."""
    mock_memory_service.search = AsyncMock(return_value=[
        MemoryResult(id='1', content='info', source_store=SourceStore.mem0, entities=['EntityA']),
    ])

    await reconciler.reconcile_task(
        task_id='1',
        transition='blocked',
        project_id='dark_factory',
        project_root='/home/leo/src/dark-factory',
        task_before={'id': '1', 'title': 'Blocked', 'status': 'in-progress'},
    )

    # update_task for hints should use filesystem path
    mock_taskmaster.update_task.assert_called_once()
    call_kwargs = mock_taskmaster.update_task.call_args.kwargs
    assert call_kwargs['project_root'] == '/home/leo/src/dark-factory'


@pytest.mark.asyncio
async def test_done_defers_write_during_active_cycle(
    mock_memory_service, mock_taskmaster, journal, config, mock_event_buffer,
):
    """When a full cycle is active, memory writes are deferred to the buffer."""
    mock_event_buffer.is_full_recon_active = AsyncMock(return_value=True)
    r = TargetedReconciler(
        mock_memory_service, mock_taskmaster, journal, config, mock_event_buffer,
    )
    r.verifier = AsyncMock()

    task_before = {'id': '1', 'title': 'Test', 'status': 'in-progress'}
    result = await r.reconcile_task(
        task_id='1', transition='done', project_id='test-project', project_root='/tmp/test',
        task_before=task_before,
    )

    # Memory service should NOT have been called for writes
    mock_memory_service.add_memory.assert_not_called()
    # Buffer should have received the deferred write
    mock_event_buffer.defer_write.assert_called()
    # Action should indicate deferral
    assert any(a['type'] == 'knowledge_deferred_fast' for a in result.get('actions', []))


@pytest.mark.asyncio
async def test_done_writes_normally_when_no_cycle(reconciler, mock_memory_service, mock_event_buffer):
    """When no full cycle is active, writes proceed normally."""
    task_before = {'id': '1', 'title': 'Test', 'status': 'in-progress'}
    result = await reconciler.reconcile_task(
        task_id='1', transition='done', project_id='test-project', project_root='/tmp/test',
        task_before=task_before,
    )

    # Memory service should have been called
    assert mock_memory_service.add_memory.call_count >= 1
    # Buffer defer should NOT have been called
    mock_event_buffer.defer_write.assert_not_called()
    # Action should indicate normal write
    assert any(a['type'] == 'knowledge_captured_fast' for a in result.get('actions', []))


@pytest.mark.asyncio
async def test_reads_proceed_during_active_cycle(
    mock_memory_service, mock_taskmaster, journal, config, mock_event_buffer,
):
    """Reads (search) still execute even when a full cycle is active."""
    mock_event_buffer.is_full_recon_active = AsyncMock(return_value=True)
    r = TargetedReconciler(
        mock_memory_service, mock_taskmaster, journal, config, mock_event_buffer,
    )
    r.verifier = AsyncMock()

    task_before = {'id': '1', 'title': 'Test', 'status': 'in-progress'}
    await r.reconcile_task(
        task_id='1', transition='done', project_id='test-project', project_root='/tmp/test',
        task_before=task_before,
    )

    # Search should still proceed
    mock_memory_service.search.assert_called()


@pytest.mark.asyncio
async def test_blocked_proceeds_during_active_cycle(
    mock_memory_service, mock_taskmaster, journal, config, mock_event_buffer,
):
    """Blocked handler only does reads + taskmaster writes, so it's unaffected by the fence."""
    mock_event_buffer.is_full_recon_active = AsyncMock(return_value=True)
    mock_memory_service.search = AsyncMock(return_value=[
        MemoryResult(id='1', content='info', source_store=SourceStore.mem0, entities=['X']),
    ])
    r = TargetedReconciler(
        mock_memory_service, mock_taskmaster, journal, config, mock_event_buffer,
    )
    r.verifier = AsyncMock()

    task_before = {'id': '1', 'title': 'Blocked', 'status': 'in-progress'}
    result = await r.reconcile_task(
        task_id='1', transition='blocked', project_id='test-project', project_root='/tmp/test',
        task_before=task_before,
    )

    # Search and taskmaster update should still work
    mock_memory_service.search.assert_called()
    hints_actions = [a for a in result.get('actions', []) if a['type'] == 'hints_attached']
    assert len(hints_actions) == 1


@pytest.mark.asyncio
async def test_no_buffer_writes_normally(mock_memory_service, mock_taskmaster, journal, config):
    """When event_buffer is None, writes proceed without fence check."""
    r = TargetedReconciler(mock_memory_service, mock_taskmaster, journal, config)
    r.verifier = AsyncMock()

    task_before = {'id': '1', 'title': 'Test', 'status': 'in-progress'}
    result = await r.reconcile_task(
        task_id='1', transition='done', project_id='test-project', project_root='/tmp/test',
        task_before=task_before,
    )

    assert mock_memory_service.add_memory.call_count >= 1
    assert any(a['type'] == 'knowledge_captured_fast' for a in result.get('actions', []))


# ── Tests for project_root validation (task-156) ──────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize('project_root', ['', 'dark_factory', '.'])
async def test_reconcile_task_rejects_bad_project_root(reconciler, project_root):
    """reconcile_task() raises ValueError for non-absolute project_root values."""
    with pytest.raises(ValueError, match=re.escape(repr(project_root))):
        await reconciler.reconcile_task(
            task_id='1', transition='done', project_id='test-project',
            project_root=project_root,
            task_before={'id': '1', 'title': 'Test', 'status': 'in-progress'},
        )


@pytest.mark.asyncio
async def test_reconcile_task_validation_error_leaves_journal_trace(reconciler, journal):
    """Validation failures in reconcile_task() must leave a 'failed' run in the journal."""
    with pytest.raises(ValueError):
        await reconciler.reconcile_task(
            task_id='1', transition='done', project_id='test-project',
            project_root='',
            task_before={'id': '1', 'title': 'Test', 'status': 'in-progress'},
        )

    runs = await journal.get_recent_runs('test-project', limit=1)
    assert len(runs) == 1, 'Expected exactly one journal run for the failed call'
    assert runs[0].status == 'failed', f'Expected status=failed, got {runs[0].status!r}'


# ── Exception masking safety tests (task-290) ────────────────────────


@pytest.mark.asyncio
async def test_validation_error_emits_warning_log(reconciler, caplog):
    """reconcile_task() must emit a WARNING log when a validation error is caught.

    This mirrors the sibling except Exception block which calls logger.error().
    The warning must contain 'rejected invalid input' so operators can grep for it.
    """
    import logging
    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.targeted'), pytest.raises(ValueError):
        await reconciler.reconcile_task(
            task_id='1', transition='done', project_id='test-project',
            project_root='',  # invalid — triggers InputValidationError
            task_before={'id': '1', 'title': 'Test', 'status': 'in-progress'},
        )

    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any('rejected invalid input' in msg for msg in warning_messages), (
        f'Expected WARNING containing "rejected invalid input", got: {warning_messages}'
    )


@pytest.mark.asyncio
async def test_handler_valueerror_caught_as_exception_not_reraised(
    mock_memory_service, mock_taskmaster, journal, config, mock_event_buffer,
):
    """A plain ValueError from a handler (_on_task_done) must NOT be re-raised.

    With ``except ValueError`` catching all ValueErrors, handler errors are silently
    re-raised with no log — indistinguishable from input validation errors.  After
    narrowing to ``except InputValidationError``, plain ValueErrors from handlers fall
    through to ``except Exception`` which returns an error dict instead of re-raising.
    """
    r = TargetedReconciler(
        mock_memory_service, mock_taskmaster, journal, config, mock_event_buffer,
    )
    r.verifier = AsyncMock()

    # Make _on_task_done raise a plain ValueError (simulating internal handler error)
    r._on_task_done = AsyncMock(side_effect=ValueError('internal handler error'))

    # Should NOT raise — handler ValueError must fall through to except Exception
    result = await r.reconcile_task(
        task_id='1', transition='done', project_id='test-project',
        project_root='/tmp/test',  # valid project_root
        task_before={'id': '1', 'title': 'Test', 'status': 'in-progress'},
    )

    # except Exception returns an error dict, not raises
    assert 'error' in result, (
        f'Expected error dict from handler ValueError, got: {result}'
    )


@pytest.mark.asyncio
async def test_journal_failure_does_not_mask_validation_error(
    mock_memory_service, mock_taskmaster, journal, config, mock_event_buffer,
):
    """If journal.complete_run raises RuntimeError, the original InputValidationError propagates.

    Python does NOT retry except-clauses for exceptions raised inside a handler, so a
    RuntimeError raised inside ``except ValueError`` propagates instead of the original error.
    Wrapping journal.complete_run in contextlib.suppress must prevent this masking.
    """
    r = TargetedReconciler(
        mock_memory_service, mock_taskmaster, journal, config, mock_event_buffer,
    )
    r.verifier = AsyncMock()

    # Patch the journal on the reconciler instance to blow up on complete_run
    r.journal = AsyncMock()
    r.journal.start_run = AsyncMock()
    r.journal.complete_run = AsyncMock(side_effect=RuntimeError('DB down'))
    r.journal.add_run_action = AsyncMock()

    # Should raise ValueError (or InputValidationError), NOT RuntimeError
    with pytest.raises(ValueError):
        await r.reconcile_task(
            task_id='1', transition='done', project_id='test-project',
            project_root='',  # invalid — triggers InputValidationError
            task_before={'id': '1', 'title': 'Test', 'status': 'in-progress'},
        )


class TestPlannedEpisodePromotion:
    """step-21: _on_task_done promotes planned episodes related to the task."""

    @pytest.fixture
    def reconciler_with_registry(self, reconciler):
        """Reconciler with a mocked planned_episode_registry."""
        from unittest.mock import AsyncMock, MagicMock
        mock_registry = MagicMock()
        mock_registry.promote = AsyncMock()
        reconciler.planned_episode_registry = mock_registry
        return reconciler

    @pytest.mark.asyncio
    async def test_promotion_called_for_planned_edges(
        self, reconciler_with_registry, mock_memory_service
    ):
        """When _on_task_done fires, planned edges from search results have their
        episode UUIDs promoted via planned_episode_registry.promote()."""
        from fused_memory.models.enums import SourceStore
        from fused_memory.models.memory import MemoryResult

        # First search call (normal, exclude planned) returns empty.
        # Second search call (include_planned=True) returns a planned edge.
        ep_uuid = 'plan-ep-abc'
        planned_result = MemoryResult(
            id='edge-planned-1',
            content='PRD: CostStore extends AgentResult',
            category=None,
            source_store=SourceStore.graphiti,
            relevance_score=0.9,
            provenance=[ep_uuid],
            metadata={'planned': True},
        )

        # Configure mock: first call (include_planned not specified / False) returns []
        # second call (include_planned=True) returns [planned_result]
        call_results = [[], [planned_result]]
        call_index = {'n': 0}

        async def mock_search(**kwargs):
            idx = call_index['n']
            call_index['n'] += 1
            if idx < len(call_results):
                return call_results[idx]
            return []

        mock_memory_service.search = mock_search

        await reconciler_with_registry.reconcile_task(
            task_id='1', transition='done', project_id='test-project',
            project_root='/tmp/test',
            task_before={'id': '1', 'title': 'CostStore', 'status': 'in-progress'},
        )

        reconciler_with_registry.planned_episode_registry.promote.assert_called_with(ep_uuid)

    @pytest.mark.asyncio
    async def test_promotion_not_called_when_no_planned_edges(
        self, reconciler_with_registry, mock_memory_service
    ):
        """When no planned edges are found (include_planned search returns non-planned
        results), promote() is not called."""
        from fused_memory.models.enums import SourceStore
        from fused_memory.models.memory import MemoryResult

        non_planned_result = MemoryResult(
            id='edge-real-1',
            content='Implemented CostStore',
            category=None,
            source_store=SourceStore.graphiti,
            relevance_score=0.9,
            provenance=['real-ep-1'],
            metadata={},
        )
        mock_memory_service.search = AsyncMock(return_value=[non_planned_result])

        await reconciler_with_registry.reconcile_task(
            task_id='1', transition='done', project_id='test-project',
            project_root='/tmp/test',
            task_before={'id': '1', 'title': 'CostStore', 'status': 'in-progress'},
        )

        reconciler_with_registry.planned_episode_registry.promote.assert_not_called()

    @pytest.mark.asyncio
    async def test_dedup_promotion_across_shared_provenance(
        self, reconciler_with_registry, mock_memory_service
    ):
        """When multiple planned search results share provenance episode UUIDs,
        promote() is called exactly once per unique UUID, not once per occurrence."""
        from fused_memory.models.enums import SourceStore
        from fused_memory.models.memory import MemoryResult

        # Two results with overlapping provenance: ep-B appears in both.
        # Without deduplication, promote() would be called 4 times (A, B, B, C).
        # With deduplication, it should be called exactly 3 times (A, B, C).
        result1 = MemoryResult(
            id='edge-planned-1',
            content='PRD: FeatureA',
            category=None,
            source_store=SourceStore.graphiti,
            relevance_score=0.9,
            provenance=['ep-A', 'ep-B'],
            metadata={'planned': True},
        )
        result2 = MemoryResult(
            id='edge-planned-2',
            content='PRD: FeatureB',
            category=None,
            source_store=SourceStore.graphiti,
            relevance_score=0.85,
            provenance=['ep-B', 'ep-C'],
            metadata={'planned': True},
        )

        call_results = [[], [result1, result2]]
        call_index = {'n': 0}

        async def mock_search(**kwargs):
            idx = call_index['n']
            call_index['n'] += 1
            if idx < len(call_results):
                return call_results[idx]
            return []

        mock_memory_service.search = mock_search

        await reconciler_with_registry.reconcile_task(
            task_id='1', transition='done', project_id='test-project',
            project_root='/tmp/test',
            task_before={'id': '1', 'title': 'FeatureAB', 'status': 'in-progress'},
        )

        promote = reconciler_with_registry.planned_episode_registry.promote
        # Exactly 3 unique UUIDs promoted, not 4
        assert promote.call_count == 3, (
            f'Expected promote() called 3 times (one per unique UUID), '
            f'got {promote.call_count}'
        )
        promoted_uuids = {call.args[0] for call in promote.call_args_list}
        assert promoted_uuids == {'ep-A', 'ep-B', 'ep-C'}

    @pytest.mark.asyncio
    async def test_no_registry_no_error_on_task_done(self, reconciler, mock_memory_service):
        """When planned_episode_registry is None (not wired), _on_task_done runs normally."""
        reconciler.planned_episode_registry = None
        mock_memory_service.search = AsyncMock(return_value=[])

        # Should not raise
        result = await reconciler.reconcile_task(
            task_id='1', transition='done', project_id='test-project',
            project_root='/tmp/test',
            task_before={'id': '1', 'title': 'Test', 'status': 'in-progress'},
        )
        assert 'error' not in result


# ── task-1136: route _on_task_blocked metadata write through TaskInterceptor ──


@pytest.mark.asyncio
async def test_blocked_routes_update_through_task_interceptor_when_wired(
    reconciler, mock_memory_service, mock_taskmaster
):
    """When task_interceptor is wired, _on_task_blocked must route update_task
    through the interceptor (which holds the per-project write_lock) instead of
    calling self.taskmaster.update_task directly.

    Asserts:
    (a) mock_interceptor.update_task called exactly once with task_id, project_root,
        and metadata= kwarg containing 'memory_hints'.
    (b) mock_taskmaster.update_task NOT called — routing must be exclusive.
    (c) Result still records a hints_attached action.
    """
    from unittest.mock import AsyncMock

    mock_interceptor = AsyncMock()
    mock_interceptor.update_task = AsyncMock(return_value={'success': True})
    reconciler.task_interceptor = mock_interceptor

    mock_memory_service.search = AsyncMock(return_value=[
        MemoryResult(
            id='1', content='blocker info', source_store=SourceStore.mem0,
            entities=['EntityA'],
        ),
    ])

    result = await reconciler.reconcile_task(
        task_id='42',
        transition='blocked',
        project_id='test-project',
        project_root='/tmp/test',
        task_before={'id': '42', 'title': 'Blocked task', 'status': 'in-progress'},
    )

    # (a) interceptor.update_task called exactly once with correct kwargs
    mock_interceptor.update_task.assert_called_once()
    call_kwargs = mock_interceptor.update_task.call_args.kwargs
    assert call_kwargs.get('task_id') == '42', (
        f'Expected task_id="42", got {call_kwargs.get("task_id")!r}'
    )
    assert call_kwargs.get('project_root') == '/tmp/test', (
        f'Expected project_root="/tmp/test", got {call_kwargs.get("project_root")!r}'
    )
    metadata_raw = call_kwargs.get('metadata')
    assert metadata_raw is not None, 'update_task must be called with metadata= kwarg'
    import json as _json
    metadata = _json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
    assert 'memory_hints' in metadata, (
        f'metadata must contain "memory_hints", got keys: {list(metadata.keys())}'
    )

    # (b) direct taskmaster bypass must NOT happen
    mock_taskmaster.update_task.assert_not_called()

    # (c) result records hints_attached
    hints_actions = [a for a in result.get('actions', []) if a['type'] == 'hints_attached']
    assert len(hints_actions) == 1, (
        f'Expected exactly one hints_attached action, got: {result.get("actions", [])}'
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'rejection_dict, expected_error, expected_reason',
    [
        # (1) backlog-gate rejection — the original covered shape (synthetic contract dict)
        (
            {'success': False, 'error': 'backlog_gate_rejected', 'reason': 'queue_lag'},
            'backlog_gate_rejected',
            'queue_lag',
        ),
        # (2) _reject_status_in_update_task shape
        (
            {
                'success': False,
                'error': 'status_via_update_task',
                'task_id': '42',
                'status': 'done',
                'hint': 'use set_task_status',
            },
            'status_via_update_task',
            None,
        ),
        # (3) _reject_done_provenance_in_update_metadata shape
        (
            {
                'success': False,
                'error': 'done_provenance_via_update_task',
                'task_id': '42',
                'hint': 'use set_task_status',
            },
            'done_provenance_via_update_task',
            None,
        ),
    ],
    ids=['backlog_gate_rejected', 'status_via_update_task', 'done_provenance_via_update_task'],
)
async def test_blocked_skips_hints_attached_on_interceptor_rejection(
    reconciler, mock_memory_service, mock_taskmaster,
    rejection_dict, expected_error, expected_reason,
):
    """When task_interceptor.update_task() returns a rejection dict, the audit signals
    must honestly reflect the failed write across all three interceptor gate shapes.

    Assertions:
    (1) mock_interceptor.update_task called once  — the write was attempted.
    (2) mock_taskmaster.update_task NOT called     — no fallback bypass when interceptor set.
    (3) No 'hints_attached' action in result       — must not lie about a write that failed.
    (4) No 'hints_attached' journal entry          — ditto in the journal.
    (5) Exactly one 'hints_skipped' action with error/reason matching the rejection.
    (6) Journal contains exactly one 'skip' row for 'update_task' with the correct
        detail dict (type='hints_skipped', task_id='42', error=expected_error,
        reason=expected_reason) — durable audit trail for rejection-rate queries.
    """
    from unittest.mock import AsyncMock as _AsyncMock

    # Wire interceptor returning the parametrized rejection shape
    mock_interceptor = _AsyncMock()
    mock_interceptor.update_task = _AsyncMock(return_value=rejection_dict)
    reconciler.task_interceptor = mock_interceptor

    # Spy on journal.add_run_action so we can inspect every emitted action row
    journal_spy = _AsyncMock()
    reconciler.journal.add_run_action = journal_spy

    # Seed search so the hints-attach branch fires
    mock_memory_service.search = _AsyncMock(return_value=[
        MemoryResult(id='1', content='blocker info', source_store=SourceStore.mem0, entities=['EntityA']),
    ])

    result = await reconciler.reconcile_task(
        task_id='42',
        transition='blocked',
        project_id='test-project',
        project_root='/tmp/test',
        task_before={'id': '42', 'title': 'Blocked task', 'status': 'in-progress'},
    )

    # (1) the write was attempted through the interceptor
    mock_interceptor.update_task.assert_called_once()

    # (2) no fallback to direct taskmaster
    mock_taskmaster.update_task.assert_not_called()

    # (3) 'hints_attached' must NOT appear in result (write didn't succeed)
    hints_attached = [a for a in result.get('actions', []) if a.get('type') == 'hints_attached']
    assert hints_attached == [], (
        f'hints_attached must not appear in actions on rejection; '
        f'got full actions list: {result.get("actions", [])}'
    )

    # (4) journal must NOT contain a hints_attached row
    journal_hints_rows = [
        c for c in journal_spy.call_args_list
        if len(c.args) >= 5
        and isinstance(c.args[4], dict)
        and c.args[4].get('type') == 'hints_attached'
    ]
    assert journal_hints_rows == [], (
        f'journal must not record hints_attached on rejection; got: {journal_hints_rows}'
    )

    # (5) exactly one 'hints_skipped' action carrying the rejection error/reason
    hints_skipped = [a for a in result.get('actions', []) if a.get('type') == 'hints_skipped']
    assert len(hints_skipped) == 1, (
        f'Expected exactly one hints_skipped action, got: {result.get("actions", [])}'
    )
    skip = hints_skipped[0]
    assert skip.get('error') == expected_error, (
        f'hints_skipped must carry the rejection error code, got: {skip}'
    )
    assert skip.get('reason') == expected_reason, (
        f'hints_skipped must carry the rejection reason, got: {skip}'
    )

    # (6) journal must contain exactly one 'skip' row for update_task with correct detail
    journal_skip_rows = [
        c for c in journal_spy.call_args_list
        if len(c.args) >= 4
        and c.args[1] == 'skip'
        and c.args[2] == 'taskmaster'
        and c.args[3] == 'update_task'
    ]
    assert len(journal_skip_rows) == 1, (
        f'Expected exactly one journal skip row for update_task, '
        f'got: {journal_skip_rows!r} from calls: {journal_spy.call_args_list!r}'
    )
    detail = journal_skip_rows[0].args[4] if len(journal_skip_rows[0].args) >= 5 else {}
    assert detail.get('type') == 'hints_skipped', (
        f"journal skip row detail must have type='hints_skipped', got: {detail}"
    )
    assert detail.get('task_id') == '42', (
        f"journal skip row detail must carry task_id='42', got: {detail}"
    )
    assert detail.get('error') == expected_error, (
        f'journal skip row detail must carry error={expected_error!r}, got: {detail}'
    )
    assert detail.get('reason') == expected_reason, (
        f'journal skip row detail must carry reason={expected_reason!r}, got: {detail}'
    )


@pytest.mark.asyncio
async def test_blocked_treats_non_dict_response_as_failure(
    reconciler, mock_memory_service, mock_taskmaster
):
    """When task_interceptor.update_task() returns None (non-dict), it must be treated
    as a failure — not silently classified as success.

    Under the old formula ``not (isinstance(resp, dict) and resp.get('error'))``,
    ``None`` evaluates to success (the inner expression is False, outer not→True).
    After the fix (``interceptor_write_succeeded``), non-dicts always → False.

    Assertions:
    (1) interceptor.update_task called once.
    (2) No 'hints_attached' in result['actions'].
    (3) Exactly one 'hints_skipped' action with error='unknown' and reason=None.
    (4) Journal contains exactly one 'skip' row with detail error='unknown'.
    (5) Journal contains NO 'hints_attached' row.
    """
    from unittest.mock import AsyncMock as _AsyncMock

    mock_interceptor = _AsyncMock()
    mock_interceptor.update_task = _AsyncMock(return_value=None)  # non-dict response
    reconciler.task_interceptor = mock_interceptor

    journal_spy = _AsyncMock()
    reconciler.journal.add_run_action = journal_spy

    mock_memory_service.search = _AsyncMock(return_value=[
        MemoryResult(id='1', content='blocker info', source_store=SourceStore.mem0, entities=['EntityA']),
    ])

    result = await reconciler.reconcile_task(
        task_id='42',
        transition='blocked',
        project_id='test-project',
        project_root='/tmp/test',
        task_before={'id': '42', 'title': 'Blocked task', 'status': 'in-progress'},
    )

    # (1) interceptor was called
    mock_interceptor.update_task.assert_called_once()

    # (2) no hints_attached action
    hints_attached = [a for a in result.get('actions', []) if a.get('type') == 'hints_attached']
    assert hints_attached == [], (
        f'None response must NOT produce hints_attached; got: {result.get("actions", [])}'
    )

    # (3) exactly one hints_skipped with error='unknown'
    hints_skipped = [a for a in result.get('actions', []) if a.get('type') == 'hints_skipped']
    assert len(hints_skipped) == 1, (
        f'Expected exactly one hints_skipped, got: {result.get("actions", [])}'
    )
    skip = hints_skipped[0]
    assert skip.get('error') == 'unknown', (
        f"Non-dict response must produce error='unknown', got: {skip}"
    )
    assert skip.get('reason') is None, (
        f"Non-dict response must produce reason=None, got: {skip}"
    )

    # (4) journal must contain exactly one 'skip' row with error='unknown'
    journal_skip_rows = [
        c for c in journal_spy.call_args_list
        if len(c.args) >= 4
        and c.args[1] == 'skip'
        and c.args[2] == 'taskmaster'
        and c.args[3] == 'update_task'
    ]
    assert len(journal_skip_rows) == 1, (
        f'Expected one journal skip row, got: {journal_skip_rows!r}'
    )
    detail = journal_skip_rows[0].args[4] if len(journal_skip_rows[0].args) >= 5 else {}
    assert detail.get('error') == 'unknown', (
        f"journal skip row must carry error='unknown', got: {detail}"
    )

    # (5) journal must NOT contain a hints_attached row
    journal_hints_rows = [
        c for c in journal_spy.call_args_list
        if len(c.args) >= 5
        and isinstance(c.args[4], dict)
        and c.args[4].get('type') == 'hints_attached'
    ]
    assert journal_hints_rows == [], (
        f'journal must not record hints_attached on None response; got: {journal_hints_rows}'
    )


class TestServerWiringContract:
    """step-34/35: TargetedReconciler.planned_episode_registry must be wired from
    MemoryService after MemoryService.initialize() creates it."""

    def test_planned_episode_registry_wired_from_memory_service(self):
        """After MemoryService.initialize() and TargetedReconciler construction,
        the wiring step must assign targeted.planned_episode_registry from
        memory_service.planned_episode_registry.

        This is a unit test of the wiring sequence expected in server/main.py.
        """
        from unittest.mock import MagicMock

        from fused_memory.reconciliation.targeted import TargetedReconciler

        # Simulate MemoryService after initialize() — has a non-None registry
        mock_registry = MagicMock()
        mock_memory_service = MagicMock()
        mock_memory_service.planned_episode_registry = mock_registry

        # Construct a minimal TargetedReconciler (registry starts as None by default)
        mock_taskmaster = MagicMock()
        mock_journal = MagicMock()
        mock_config = MagicMock()
        targeted = TargetedReconciler(
            mock_memory_service, mock_taskmaster, mock_journal, mock_config
        )

        # Before wiring: registry must be None (the reconciler doesn't auto-wire)
        assert targeted.planned_episode_registry is None, (
            'TargetedReconciler must start with planned_episode_registry=None'
        )

        # Apply the wiring (what main.py must do)
        targeted.planned_episode_registry = mock_memory_service.planned_episode_registry

        # After wiring: identity check
        assert targeted.planned_episode_registry is mock_registry, (
            'After wiring, targeted.planned_episode_registry must be the same object '
            'as memory_service.planned_episode_registry'
        )

    def test_task_interceptor_wired_to_targeted_reconciler(self):
        """server/main.py must assign targeted.task_interceptor = task_interceptor
        after both TargetedReconciler and TaskInterceptor are constructed, so that
        _on_task_blocked routes metadata writes through the per-project write_lock
        (task 1136 locking invariant).

        Uses AST analysis of server/main.py to verify the wiring assignment is present.
        This catches the regression where the wiring line is accidentally removed from
        server/main.py — something the unit test above cannot detect because it only
        exercises the wired code path, not the wiring step itself.
        """
        import ast
        import pathlib

        main_py = (
            pathlib.Path(__file__).parents[1]
            / 'src' / 'fused_memory' / 'server' / 'main.py'
        )
        assert main_py.exists(), f'server/main.py not found at {main_py}'

        tree = ast.parse(main_py.read_text())

        # Walk every node looking for:  targeted.task_interceptor = <anything>
        # The assignment may appear inside an `if targeted is not None:` guard —
        # ast.walk descends into all children so it finds it regardless of nesting.
        found = any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Attribute)
                and isinstance(t.value, ast.Name)
                and t.value.id == 'targeted'
                and t.attr == 'task_interceptor'
                for t in node.targets
            )
            for node in ast.walk(tree)
        )

        assert found, (
            'server/main.py must assign targeted.task_interceptor after TaskInterceptor '
            'is constructed. This wiring ensures _on_task_blocked routes metadata writes '
            'through the per-project write_lock (task 1136). '
            'Restore the assignment in server/main.py if this assertion fails.'
        )
