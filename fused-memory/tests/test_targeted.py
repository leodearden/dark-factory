"""Tests for targeted reconciliation."""

import re
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from _fm_helpers import make_8df8_scenario, pydantic_spec

from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
from fused_memory.models.enums import SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.models.reconciliation import VerificationResult, VerificationVerdict
from fused_memory.reconciliation.backlog_policy import BacklogVerdict
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
    # Plain string attribute (not an AsyncMock child): matches the real
    # EventBuffer.instance_id wire shape so ReconciliationRun's
    # ``instance_id: str | None`` Pydantic field validates.
    buf.instance_id = 'mock-buffer-instance'
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
        # (1) backlog-gate rejection — real BacklogVerdict.to_error_dict() shape
        # (no 'success' key, no 'reason' key; 'error_type' = 'ReconciliationBacklogExceeded'
        #  is the stable machine-friendly code; 'error' is the rendered human message —
        #  task 1215: assert stable code so audit queries work with = '<code>')
        (
            BacklogVerdict(outcome='rejection', backlog=600, threshold=500, project_id='test-project').to_error_dict(),
            'ReconciliationBacklogExceeded',
            None,  # no 'reason' key in BacklogVerdict dict
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
    ids=['backlog_verdict', 'status_via_update_task', 'done_provenance_via_update_task'],
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
        mock_config = MagicMock(spec_set=pydantic_spec(FusedMemoryConfig))
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


# ── Auto-sweep dep-tree on task cancellation ─────────────────────────────────
#
# When a parent task is cancelled, descendants that reference the cancelled
# branch's artifacts (review-followups spawned by the architect during the
# parent's review) become orphan. Surface them immediately via three routes:
#
#   1. Cancel — deterministic orphan (spawned_from == A AND escalation_id
#      present AND files don't exist on main).
#   2. L1 escalate — ambiguous + orchestrator live for the project.
#   3. Block — ambiguous + orchestrator dead. set_task_status('blocked') +
#      metadata carrying parent_cancelled + (optional) review_escalation_id.
#
# Dependents (deps include A) never auto-cancel; they take the block-or-L1
# branch.  Subtasks remain flag-only (out of scope per existing behaviour).
#
# Recursion guard: when the sweep's own cancel/block writes fire
# `_on_task_cancelled` again on the descendant, the inner call short-circuits
# via the ``reopen_reason.startswith('parent_cancelled:')`` check.


class TestSweepCancelledDescendants:
    """Auto-sweep dep-tree on task cancellation."""

    @pytest.fixture
    def mock_interceptor(self):
        from unittest.mock import AsyncMock
        m = AsyncMock()
        m.set_task_status = AsyncMock(return_value={'success': True})
        m.update_task = AsyncMock(return_value={'success': True})
        return m

    @pytest.fixture
    def wired_reconciler(self, reconciler, mock_interceptor):
        reconciler.task_interceptor = mock_interceptor
        return reconciler

    @pytest.fixture(autouse=True)
    def _patch_orchestrator_live(self, monkeypatch):
        """Default to orchestrator dead; tests override per case."""
        from fused_memory.reconciliation import targeted
        monkeypatch.setattr(
            targeted, 'is_orchestrator_live_for', lambda _project_root: False,
        )
        return None

    @pytest.mark.asyncio
    async def test_deterministic_cancel_for_review_followup(
        self, wired_reconciler, mock_taskmaster, mock_interceptor,
    ):
        """B is a review-followup of A (spawned_from + escalation_id, no
        surviving files); on A's cancellation, B is auto-cancelled with a
        reopen_reason rooted at A."""
        mock_taskmaster.get_tasks = AsyncMock(return_value={
            'tasks': [
                {
                    'id': 'B', 'status': 'pending', 'title': 'review-followup',
                    'metadata': {
                        'spawned_from': 'A',
                        'escalation_id': 'esc-A-1',
                    },
                    'dependencies': [],
                    'subtasks': [],
                },
            ],
        })

        result = await wired_reconciler.reconcile_task(
            task_id='A', transition='cancelled',
            project_id='test-project', project_root='/tmp/test',
            task_before={'id': 'A', 'title': 'Parent', 'status': 'in-progress'},
        )

        # set_task_status was called for B with status='cancelled' and
        # reopen_reason starting 'parent_cancelled:A'
        mock_interceptor.set_task_status.assert_called_once()
        call = mock_interceptor.set_task_status.call_args
        assert call.kwargs.get('task_id') == 'B'
        assert call.kwargs.get('status') == 'cancelled'
        reopen = call.kwargs.get('reopen_reason') or ''
        assert reopen.startswith('parent_cancelled:A'), (
            f'expected reopen_reason to start with parent_cancelled:A, got {reopen!r}'
        )

        # The recorded action surfaces the descendant cancel
        cancels = [a for a in result.get('actions', []) if a.get('type') == 'descendant_cancelled']
        assert len(cancels) == 1
        assert cancels[0].get('task_id') == 'B'

    @pytest.mark.asyncio
    async def test_unrelated_top_level_is_left_untouched(
        self, wired_reconciler, mock_taskmaster, mock_interceptor,
    ):
        """C has no spawned_from chain to A and is not a dependent; sweep
        leaves it alone (no set_task_status, no update_task)."""
        mock_taskmaster.get_tasks = AsyncMock(return_value={
            'tasks': [
                {
                    'id': 'C', 'status': 'pending', 'title': 'unrelated',
                    'metadata': {},
                    'dependencies': [],
                    'subtasks': [],
                },
            ],
        })

        result = await wired_reconciler.reconcile_task(
            task_id='A', transition='cancelled',
            project_id='test-project', project_root='/tmp/test',
            task_before={'id': 'A', 'title': 'Parent', 'status': 'in-progress'},
        )

        mock_interceptor.set_task_status.assert_not_called()
        mock_interceptor.update_task.assert_not_called()
        # Sweep produced no descendant actions
        kinds = {a.get('type') for a in result.get('actions', [])}
        assert 'descendant_cancelled' not in kinds
        assert 'descendant_blocked' not in kinds
        assert 'descendant_escalated' not in kinds

    @pytest.mark.asyncio
    async def test_ambiguous_with_orchestrator_live_files_l1_escalation(
        self, wired_reconciler, mock_taskmaster, mock_interceptor, monkeypatch, tmp_path,
    ):
        """B is ambiguous (spawned_from=A but no escalation_id); orchestrator
        is live → L1 escalation file is written and B's status is unchanged."""
        from fused_memory.reconciliation import targeted
        monkeypatch.setattr(targeted, 'is_orchestrator_live_for', lambda _pr: True)

        project_root = str(tmp_path)
        mock_taskmaster.get_tasks = AsyncMock(return_value={
            'tasks': [
                {
                    'id': 'B', 'status': 'pending', 'title': 'ambiguous',
                    'metadata': {'spawned_from': 'A'},  # no escalation_id
                    'dependencies': [],
                    'subtasks': [],
                },
            ],
        })

        result = await wired_reconciler.reconcile_task(
            task_id='A', transition='cancelled',
            project_id='test-project', project_root=project_root,
            task_before={'id': 'A', 'title': 'Parent', 'status': 'in-progress'},
        )

        # No status change for B
        mock_interceptor.set_task_status.assert_not_called()
        mock_interceptor.update_task.assert_not_called()

        # Escalation JSON written under <project_root>/data/escalations/
        esc_dir = tmp_path / 'data' / 'escalations'
        files = list(esc_dir.glob('esc-*.json'))
        assert len(files) == 1, f'expected one escalation file, got {files}'

        import json as _json
        payload = _json.loads(files[0].read_text())
        assert payload['task_id'] == 'B'
        assert payload['level'] == 1
        assert payload['severity'] == 'blocking'
        assert payload['category'] == 'scope_violation'

        # And the action surfaces in the result
        escs = [a for a in result.get('actions', []) if a.get('type') == 'descendant_escalated']
        assert len(escs) == 1
        assert escs[0].get('task_id') == 'B'

    @pytest.mark.asyncio
    async def test_ambiguous_with_orchestrator_dead_blocks_with_metadata(
        self, wired_reconciler, mock_taskmaster, mock_interceptor, tmp_path,
    ):
        """Same shape as the prior test but with orchestrator dead → B is
        blocked via set_task_status('blocked', reopen_reason=...) and
        update_task(metadata=..., append=True) carrying parent_cancelled."""
        project_root = str(tmp_path)
        mock_taskmaster.get_tasks = AsyncMock(return_value={
            'tasks': [
                {
                    'id': 'B', 'status': 'pending', 'title': 'ambiguous',
                    'metadata': {'spawned_from': 'A'},  # no escalation_id
                    'dependencies': [],
                    'subtasks': [],
                },
            ],
        })

        result = await wired_reconciler.reconcile_task(
            task_id='A', transition='cancelled',
            project_id='test-project', project_root=project_root,
            task_before={'id': 'A', 'title': 'Parent', 'status': 'in-progress'},
        )

        # set_task_status('blocked') with the parent_cancelled reopen_reason
        mock_interceptor.set_task_status.assert_called_once()
        sts_call = mock_interceptor.set_task_status.call_args
        assert sts_call.kwargs.get('task_id') == 'B'
        assert sts_call.kwargs.get('status') == 'blocked'
        reopen = sts_call.kwargs.get('reopen_reason') or ''
        assert reopen.startswith('parent_cancelled:A'), (
            f'expected reopen_reason to start with parent_cancelled:A, got {reopen!r}'
        )

        # update_task carries the parent_cancelled metadata with append=True
        mock_interceptor.update_task.assert_called_once()
        ut_call = mock_interceptor.update_task.call_args
        assert ut_call.kwargs.get('task_id') == 'B'
        assert ut_call.kwargs.get('append') is True
        import json as _json
        meta_raw = ut_call.kwargs.get('metadata')
        meta = _json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
        assert meta.get('parent_cancelled') == 'A'

        # No escalation file written when orchestrator is dead
        esc_dir = tmp_path / 'data' / 'escalations'
        files = list(esc_dir.glob('esc-*.json')) if esc_dir.exists() else []
        assert files == []

        # And the action surfaces in the result
        blocks = [a for a in result.get('actions', []) if a.get('type') == 'descendant_blocked']
        assert len(blocks) == 1
        assert blocks[0].get('task_id') == 'B'

    @pytest.mark.asyncio
    async def test_dependent_blocked_never_cancelled(
        self, wired_reconciler, mock_taskmaster, mock_interceptor, tmp_path,
    ):
        """D depends on A but has no spawned_from link; orchestrator dead →
        D is blocked, never cancelled, irrespective of escalation_id presence."""
        project_root = str(tmp_path)
        mock_taskmaster.get_tasks = AsyncMock(return_value={
            'tasks': [
                {
                    'id': 'D', 'status': 'pending', 'title': 'dependent',
                    'metadata': {},  # no spawned_from
                    'dependencies': ['A'],
                    'subtasks': [],
                },
            ],
        })

        result = await wired_reconciler.reconcile_task(
            task_id='A', transition='cancelled',
            project_id='test-project', project_root=project_root,
            task_before={'id': 'A', 'title': 'Parent', 'status': 'in-progress'},
        )

        # Must be blocked, never cancelled
        mock_interceptor.set_task_status.assert_called_once()
        call = mock_interceptor.set_task_status.call_args
        assert call.kwargs.get('task_id') == 'D'
        assert call.kwargs.get('status') == 'blocked'

        cancels = [a for a in result.get('actions', []) if a.get('type') == 'descendant_cancelled']
        blocks = [a for a in result.get('actions', []) if a.get('type') == 'descendant_blocked']
        assert cancels == []
        assert len(blocks) == 1

    @pytest.mark.asyncio
    async def test_subtask_flag_only_regression(
        self, wired_reconciler, mock_taskmaster, mock_interceptor,
    ):
        """Subtasks of A remain flag-only — sweep does not touch them
        (preserves the existing subtasks_need_review behaviour)."""
        # No top-level descendants to sweep
        mock_taskmaster.get_tasks = AsyncMock(return_value={'tasks': []})

        result = await wired_reconciler.reconcile_task(
            task_id='A', transition='cancelled',
            project_id='test-project', project_root='/tmp/test',
            task_before={
                'id': 'A', 'title': 'Parent', 'status': 'in-progress',
                'subtasks': [
                    {'id': 'A.1', 'status': 'pending', 'title': 'Sub1'},
                    {'id': 'A.2', 'status': 'done', 'title': 'Sub2'},
                ],
            },
        )

        review_actions = [a for a in result.get('actions', []) if a.get('type') == 'subtasks_need_review']
        assert len(review_actions) == 1
        assert review_actions[0]['count'] == 1
        # Subtasks are not auto-cancelled or auto-blocked
        mock_interceptor.set_task_status.assert_not_called()
        mock_interceptor.update_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_recursion_guard_short_circuits_on_parent_cancelled_reason(
        self, wired_reconciler, mock_taskmaster, mock_interceptor,
    ):
        """A second-level cancel triggered by the sweep itself (carrying a
        reopen_reason starting 'parent_cancelled:') must not re-enter the
        sweep — guard short-circuits before get_tasks is even called."""
        get_tasks_spy = AsyncMock(return_value={'tasks': []})
        mock_taskmaster.get_tasks = get_tasks_spy

        result = await wired_reconciler.reconcile_task(
            task_id='B', transition='cancelled',
            project_id='test-project', project_root='/tmp/test',
            task_before={'id': 'B', 'title': 'orphan child', 'status': 'in-progress'},
            reopen_reason='parent_cancelled:A',
        )

        # Sweep is skipped — no descendant enumeration occurred
        get_tasks_spy.assert_not_called()
        mock_interceptor.set_task_status.assert_not_called()
        mock_interceptor.update_task.assert_not_called()
        sweep_kinds = {'descendant_cancelled', 'descendant_blocked', 'descendant_escalated'}
        assert not sweep_kinds.intersection(
            a.get('type') for a in result.get('actions', [])
        )

    @pytest.mark.asyncio
    async def test_escalation_file_is_atomic_and_parseable(
        self, wired_reconciler, mock_taskmaster, monkeypatch, tmp_path,
    ):
        """L1 escalation submit leaves no .tmp files and produces a JSON file
        that round-trips through Escalation.from_json."""
        from fused_memory.reconciliation import targeted
        monkeypatch.setattr(targeted, 'is_orchestrator_live_for', lambda _pr: True)

        project_root = str(tmp_path)
        mock_taskmaster.get_tasks = AsyncMock(return_value={
            'tasks': [
                {
                    'id': 'B', 'status': 'pending', 'title': 'ambiguous',
                    'metadata': {'spawned_from': 'A'},
                    'dependencies': [],
                    'subtasks': [],
                },
            ],
        })

        await wired_reconciler.reconcile_task(
            task_id='A', transition='cancelled',
            project_id='test-project', project_root=project_root,
            task_before={'id': 'A', 'title': 'Parent', 'status': 'in-progress'},
        )

        esc_dir = tmp_path / 'data' / 'escalations'
        # No leaked .tmp files
        leftover_tmp = list(esc_dir.glob('*.tmp'))
        assert leftover_tmp == [], f'leaked tmp files: {leftover_tmp}'

        # Final file round-trips through Escalation.from_json
        from escalation.models import Escalation
        files = list(esc_dir.glob('esc-*.json'))
        assert len(files) == 1
        esc = Escalation.from_json(files[0].read_text())
        assert esc.task_id == 'B'
        assert esc.level == 1
        assert esc.severity == 'blocking'
        assert esc.category == 'scope_violation'

    # ── Co-cancellation race suppression (task 1490) ────────────────────────

    @pytest.mark.asyncio
    @pytest.mark.parametrize('case', [
        'orch_live_co_cancelled',
        'orch_dead_co_cancelled',
        'orch_live_genuinely_active',
    ])
    async def test_co_cancelled_dependent_suppressed(
        self,
        wired_reconciler,
        mock_taskmaster,
        mock_interceptor,
        monkeypatch,
        tmp_path,
        case,
    ):
        """Suppress spurious scope_violation L1s for co-cancelled sibling dependents.

        Parametrized cases:
        (a) orch_live_co_cancelled  — live re-check shows D now 'cancelled':
            no escalation file written, no descendant_escalated action, one
            descendant_skipped_co_cancelled action (task_id=='D').
        (b) orch_dead_co_cancelled  — orch dead, re-check shows D 'cancelled':
            set_task_status NOT called, no descendant_blocked action, one
            descendant_skipped_co_cancelled action.
        (c) orch_live_genuinely_active — re-check STILL shows D 'pending'
            (genuine orphan): escalation file present (scope_violation/L1/blocking),
            NO descendant_skipped_co_cancelled action.  Guards against
            over-suppression.
        """
        from escalation.models import Escalation

        from fused_memory.reconciliation import targeted

        project_root = str(tmp_path)

        # D is a pure dependent: no spawned_from / escalation_id → takes the
        # ambiguous escalate-or-block route, never the deterministic-cancel route.
        d_pending = {
            'id': 'D', 'status': 'pending', 'title': 'dependent',
            'metadata': {}, 'dependencies': ['A'], 'subtasks': [],
        }
        snapshot1 = {'tasks': [d_pending]}  # stale initial sweep snapshot (L548)

        orch_live = case in ('orch_live_co_cancelled', 'orch_live_genuinely_active')
        monkeypatch.setattr(targeted, 'is_orchestrator_live_for', lambda _pr: orch_live)

        if case == 'orch_live_genuinely_active':
            # Re-check: D is still pending (genuinely-active orphan)
            snapshot2 = {'tasks': [dict(d_pending)]}
        else:
            # Re-check: D is now cancelled (co-cancelled sibling)
            snapshot2 = {'tasks': [dict(d_pending, status='cancelled')]}

        # side_effect: snapshot1 → initial get_tasks (L548 in sweep),
        #              snapshot2 → live re-check in _live_status_map
        mock_taskmaster.get_tasks = AsyncMock(side_effect=[snapshot1, snapshot2])

        result = await wired_reconciler.reconcile_task(
            task_id='A', transition='cancelled',
            project_id='test-project', project_root=project_root,
            task_before={'id': 'A', 'title': 'Parent', 'status': 'in-progress'},
        )

        esc_dir = tmp_path / 'data' / 'escalations'
        esc_files = list(esc_dir.glob('esc-*.json')) if esc_dir.exists() else []
        actions = result.get('actions', [])
        skip_actions = [
            a for a in actions if a.get('type') == 'descendant_skipped_co_cancelled'
        ]

        if case == 'orch_live_co_cancelled':
            # (a) Suppressed: no escalation file, no escalated action, one skip
            assert esc_files == [], (
                f'expected no escalation files, got {esc_files}'
            )
            assert not any(a.get('type') == 'descendant_escalated' for a in actions), (
                'unexpected descendant_escalated action present'
            )
            assert len(skip_actions) == 1, (
                f'expected exactly one skip action, got {skip_actions}'
            )
            assert skip_actions[0].get('task_id') == 'D'

        elif case == 'orch_dead_co_cancelled':
            # (b) Suppressed: no block call, no blocked action, one skip
            mock_interceptor.set_task_status.assert_not_called()
            assert not any(a.get('type') == 'descendant_blocked' for a in actions), (
                'unexpected descendant_blocked action present'
            )
            assert len(skip_actions) == 1, (
                f'expected exactly one skip action, got {skip_actions}'
            )
            assert skip_actions[0].get('task_id') == 'D'

        elif case == 'orch_live_genuinely_active':
            # (c) NOT suppressed: escalation file present, no skip action
            assert len(esc_files) == 1, (
                f'expected one escalation file for genuine orphan, got {esc_files}'
            )
            esc = Escalation.from_json(esc_files[0].read_text())
            assert esc.category == 'scope_violation'
            assert esc.level == 1
            assert esc.severity == 'blocking'
            assert not skip_actions, (
                f'expected no skip actions for genuine orphan, got {skip_actions}'
            )

    @pytest.mark.asyncio
    async def test_recheck_failure_falls_back_to_escalation(
        self, wired_reconciler, mock_taskmaster, mock_interceptor, monkeypatch, tmp_path,
    ):
        """Live re-check I/O failure must NOT drop a genuine orphan (fail-open).

        The initial sweep snapshot succeeds; the live re-check raises RuntimeError.
        The sweep must fall through to the existing escalate path and write one
        blocking scope_violation L1 file.  No descendant_skipped_co_cancelled
        action must appear — the guard never suppresses on a re-check failure.
        """
        from escalation.models import Escalation

        from fused_memory.reconciliation import targeted

        monkeypatch.setattr(targeted, 'is_orchestrator_live_for', lambda _pr: True)

        project_root = str(tmp_path)

        # D is a pure dependent (no spawned_from / escalation_id) → ambiguous route.
        d_pending = {
            'id': 'D', 'status': 'pending', 'title': 'dependent',
            'metadata': {}, 'dependencies': ['A'], 'subtasks': [],
        }
        snapshot1 = {'tasks': [d_pending]}  # initial sweep snapshot succeeds

        # side_effect: snapshot1 → initial get_tasks (L548),
        #              RuntimeError → live re-check in _live_status_map
        mock_taskmaster.get_tasks = AsyncMock(
            side_effect=[snapshot1, RuntimeError('db down')],
        )

        result = await wired_reconciler.reconcile_task(
            task_id='A', transition='cancelled',
            project_id='test-project', project_root=project_root,
            task_before={'id': 'A', 'title': 'Parent', 'status': 'in-progress'},
        )

        # Re-check failure must fall-open: escalation file is still written
        esc_dir = tmp_path / 'data' / 'escalations'
        esc_files = list(esc_dir.glob('esc-*.json')) if esc_dir.exists() else []
        assert len(esc_files) == 1, (
            f'expected one escalation file on re-check failure, got {esc_files}'
        )
        esc = Escalation.from_json(esc_files[0].read_text())
        assert esc.task_id == 'D'
        assert esc.category == 'scope_violation'
        assert esc.level == 1

        # No skip action — the guard must not suppress on a transient re-check error
        actions = result.get('actions', [])
        skip_actions = [
            a for a in actions if a.get('type') == 'descendant_skipped_co_cancelled'
        ]
        assert not skip_actions, (
            f'expected no skip actions on re-check failure, got {skip_actions}'
        )


# ── Regression: cycle 8df8bdcd title↔task_id contract (task 1379) ──────────
# Scenario shared via _fm_helpers.make_8df8_scenario (str ids, status='in-progress').

# Fixture: 8df8bdcd scenario (str ids, in-progress) — canonical definition in _fm_helpers.py
_8DF8_TASKS, _8DF8_TITLE_BY_ID = make_8df8_scenario(id_type=str, status='in-progress')


@pytest.mark.asyncio
async def test_on_task_done_writes_own_title_in_multicompletion_window(
    reconciler, mock_memory_service
):
    """Targeted-recon write path: each completion memory carries its OWN title.

    Reproduces cycle 8df8bdcd at the write-path layer: three tasks complete in
    completion order 1369→1355→1361 (non-consecutive ids, distinct titles).
    Each resulting add_memory call must embed 'Task '<own title>' completed.'
    and have metadata.task_id == that task's id — no neighbor-title bleed.

    This pins the write-content contract.  Expected GREEN per the pipeline
    audit: _on_task_done reads title from the passed task_before dict (same
    dict as metadata.task_id), so no positional/commit-order aliasing exists.
    """
    for task in _8DF8_TASKS:
        mock_memory_service.add_memory.reset_mock()
        await reconciler.reconcile_task(
            task_id=task['id'],
            transition='done',
            project_id='test-project',
            project_root='/tmp/test',
            task_before=task,
        )
        calls = mock_memory_service.add_memory.call_args_list
        assert len(calls) >= 1, f'No add_memory calls for task {task["id"]}'

        # The first (fast-path) call embeds "Task '<title>' completed."
        first_call = calls[0]
        content = first_call.kwargs.get('content', '') or (
            first_call.args[0] if first_call.args else ''
        )
        own_title = _8DF8_TITLE_BY_ID[task['id']]
        assert own_title in content, (
            f"Task {task['id']}: fast-path write contains wrong title.\n"
            f"  Expected own title: {own_title!r}\n"
            f"  Actual content:     {content!r}"
        )

        # metadata.task_id must match
        metadata = first_call.kwargs.get('metadata', {}) or {}
        assert metadata.get('task_id') == task['id'], (
            f"Task {task['id']}: metadata.task_id mismatch: {metadata}"
        )

        # No other task's title should appear in this memory's content
        for other_id, other_title in _8DF8_TITLE_BY_ID.items():
            if other_id != task['id']:
                assert other_title not in content, (
                    f"Task {task['id']}: content contains neighbor title {other_title!r}.\n"
                    f"  content: {content!r}"
                )
