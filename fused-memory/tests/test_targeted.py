"""Tests for targeted reconciliation."""

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
from fused_memory.models.enums import SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.models.reconciliation import VerificationResult
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
def reconciler(mock_memory_service, mock_taskmaster, journal, config):
    r = TargetedReconciler(mock_memory_service, mock_taskmaster, journal, config)
    # Mock the verifier to avoid actual LLM calls
    r.verifier = AsyncMock()
    r.verifier.verify = AsyncMock(return_value=VerificationResult(
        verdict='confirmed',
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
        task_id='1', transition='done', project_id='test-project', task_before=task_before
    )
    # First call should be the fast-path write (before any search)
    calls = mock_memory_service.add_memory.call_args_list
    assert len(calls) >= 1
    first_call = calls[0]
    assert 'observations_and_summaries' in str(first_call)
    assert any(a['type'] == 'knowledge_captured_fast' for a in result.get('actions', []))


@pytest.mark.asyncio
async def test_on_task_done_sparse_knowledge(reconciler, mock_memory_service):
    """When task is done and knowledge is sparse, should verify and write."""
    task_before = {'id': '1', 'title': 'Add tests', 'status': 'in-progress', 'description': 'Test suite'}

    result = await reconciler.reconcile_task(
        task_id='1', transition='done', project_id='test-project', task_before=task_before
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
        task_id='1', transition='done', project_id='test-project',
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
        task_id='1', transition='done', project_id='test-project',
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
        task_id='1', transition='blocked', project_id='test-project',
        task_before={'id': '1', 'title': 'Blocked task', 'status': 'in-progress'},
    )

    hints_actions = [a for a in result.get('actions', []) if a['type'] == 'hints_attached']
    assert len(hints_actions) == 1
    mock_taskmaster.update_task.assert_called_once()


@pytest.mark.asyncio
async def test_on_task_cancelled_checks_subtasks(reconciler, mock_taskmaster):
    """Cancelled task should flag active subtasks for review."""
    result = await reconciler.reconcile_task(
        task_id='1', transition='cancelled', project_id='test-project',
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
        task_id='1', transition='deferred', project_id='test-project',
        task_before={'id': '1', 'title': 'Deferred task', 'status': 'in-progress'},
    )

    hints_actions = [a for a in result.get('actions', []) if a['type'] == 'hints_attached']
    assert len(hints_actions) == 1


@pytest.mark.asyncio
async def test_reconcile_task_failure_handling(reconciler, journal):
    """Failure during reconciliation should be caught and recorded."""
    reconciler.verifier.verify = AsyncMock(side_effect=Exception('LLM error'))

    result = await reconciler.reconcile_task(
        task_id='1', transition='done', project_id='test-project',
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
async def test_bulk_reconcile_separates_ids(reconciler, mock_memory_service, mock_taskmaster):
    """reconcile_bulk_tasks uses project_id for memory.search and project_root for taskmaster calls."""
    mock_taskmaster.get_tasks = AsyncMock(return_value={
        'tasks': [
            {'id': '1', 'title': 'Task 1', 'status': 'pending', 'dependencies': []},
        ]
    })
    mock_memory_service.search = AsyncMock(return_value=[
        MemoryResult(id='m1', content='info', source_store=SourceStore.mem0, entities=['E1']),
    ])

    await reconciler.reconcile_bulk_tasks(
        parent_task_id=None,
        project_id='dark_factory',
        project_root='/home/leo/src/dark-factory',
    )

    # taskmaster.get_tasks should use filesystem path
    mock_taskmaster.get_tasks.assert_called_once_with(
        project_root='/home/leo/src/dark-factory'
    )
    # memory.search should use logical project_id
    for call in mock_memory_service.search.call_args_list:
        assert call.kwargs.get('project_id') == 'dark_factory'
    # update_task for hints should use filesystem path
    if mock_taskmaster.update_task.called:
        call_kwargs = mock_taskmaster.update_task.call_args.kwargs
        assert call_kwargs['project_root'] == '/home/leo/src/dark-factory'
