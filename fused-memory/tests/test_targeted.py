"""Tests for targeted reconciliation."""

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
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
        MemoryResult(id='1', content='test', source_store='mem0'),
        MemoryResult(id='2', content='test2', source_store='mem0'),
        MemoryResult(id='3', content='test3', source_store='graphiti'),
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
        MemoryResult(id='1', content='relevant info', source_store='mem0', entities=['EntityA']),
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
        MemoryResult(id='1', content='info', source_store='mem0', entities=['X']),
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
