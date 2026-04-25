"""Tests for crash recovery — surviving worktree detection and plan injection."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig
from orchestrator.harness import Harness


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
    )


@pytest.fixture
def harness(tmp_path: Path, git_config: GitConfig):
    """Create a Harness with mocked internals for unit testing recovery."""
    config = MagicMock()
    config.git = git_config
    config.project_root = tmp_path
    config.usage_cap.enabled = False
    config.sandbox.backend = 'auto'

    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(config)

    # Replace scheduler with async mocks
    h.scheduler = MagicMock()
    h.scheduler.get_tasks = AsyncMock(return_value=[])
    h.scheduler.set_task_status = AsyncMock()

    # Replace git_ops cleanup with async mock but keep worktree_base real
    h.git_ops.worktree_base = (tmp_path / '.worktrees').resolve()
    h.git_ops.cleanup_worktree = AsyncMock()

    return h


def _make_plan(steps_done: int, steps_total: int, task_id: str = 'test') -> dict:
    """Build a plan dict with the given step completion counts."""
    steps = []
    for i in range(steps_total):
        steps.append({
            'id': f'step-{i + 1}',
            'description': f'Step {i + 1}',
            'status': 'done' if i < steps_done else 'pending',
            'commit': f'abc{i}' if i < steps_done else None,
        })
    return {
        'task_id': task_id,
        'title': 'Test Task',
        'steps': steps,
    }


def _setup_worktree(base: Path, task_id: str, plan: dict | None = None):
    """Create a fake worktree directory, optionally with a plan."""
    wt = base / task_id
    wt.mkdir(parents=True, exist_ok=True)
    if plan is not None:
        task_dir = wt / '.task'
        task_dir.mkdir(exist_ok=True)
        (task_dir / 'plan.json').write_text(json.dumps(plan))
    return wt


@pytest.mark.asyncio
class TestRecoverCrashedTasks:
    async def test_recover_worktree_with_completed_steps(self, harness: Harness):
        """Worktree with plan (3/5 steps done) -> plan stored in _recovered_plans."""
        plan = _make_plan(steps_done=3, steps_total=5, task_id='35')
        _setup_worktree(harness.git_ops.worktree_base, '35', plan)

        await harness._recover_crashed_tasks()

        assert '35' in harness._recovered_plans
        recovered = harness._recovered_plans['35']
        assert len(recovered['steps']) == 5
        done = [s for s in recovered['steps'] if s['status'] == 'done']
        assert len(done) == 3
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]

    async def test_recover_planless_worktree_cleaned_up(self, harness: Harness):
        """Worktree with no .task/ dir -> cleaned up."""
        wt = _setup_worktree(harness.git_ops.worktree_base, '36')

        await harness._recover_crashed_tasks()

        assert '36' not in harness._recovered_plans
        harness.git_ops.cleanup_worktree.assert_called_once_with(wt, '36')  # type: ignore[attr-defined]

    async def test_recover_plan_no_progress_cleaned_up(self, harness: Harness):
        """Plan with all steps pending -> cleaned up."""
        plan = _make_plan(steps_done=0, steps_total=4)
        wt = _setup_worktree(harness.git_ops.worktree_base, '37', plan)

        await harness._recover_crashed_tasks()

        assert '37' not in harness._recovered_plans
        harness.git_ops.cleanup_worktree.assert_called_once_with(wt, '37')  # type: ignore[attr-defined]

    async def test_recover_corrupt_plan_cleaned_up(self, harness: Harness):
        """Invalid JSON in plan.json -> cleaned up with warning."""
        wt = harness.git_ops.worktree_base / '38'
        wt.mkdir(parents=True)
        task_dir = wt / '.task'
        task_dir.mkdir()
        (task_dir / 'plan.json').write_text('{not valid json!!!')

        await harness._recover_crashed_tasks()

        assert '38' not in harness._recovered_plans
        harness.git_ops.cleanup_worktree.assert_called_once_with(wt, '38')  # type: ignore[attr-defined]

    async def test_recover_no_worktrees_dir_noop(self, harness: Harness):
        """Worktree base doesn't exist -> no-op, no errors."""
        # Don't create the worktree base dir
        assert not harness.git_ops.worktree_base.exists()

        await harness._recover_crashed_tasks()

        assert harness._recovered_plans == {}
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]

    async def test_in_progress_tasks_left_for_reconcile_sweep(self, harness: Harness):
        """_recover_crashed_tasks does NOT reset in-progress tasks to pending.

        Status reconciliation for stranded in-progress tasks is handled by the
        separate _reconcile_stranded_in_progress() sweep that runs immediately
        after this method in Harness.run().
        """
        harness.git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        harness.scheduler.get_tasks.return_value = [  # type: ignore[attr-defined]
            {'id': 10, 'status': 'in-progress', 'title': 'Stuck task'},
            {'id': 11, 'status': 'pending', 'title': 'Normal task'},
            {'id': 12, 'status': 'done', 'title': 'Done task'},
            {'id': 13, 'status': 'in-progress', 'title': 'Another stuck'},
        ]

        await harness._recover_crashed_tasks()

        # set_task_status must NOT be called — status reconciliation is
        # delegated to _reconcile_stranded_in_progress.
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]

    async def test_recovered_plan_injected_in_run_slot(self, harness: Harness):
        """Plan consumed from _recovered_plans and passed as initial_plan."""
        plan = _make_plan(steps_done=3, steps_total=5)
        harness._recovered_plans['42'] = plan

        assignment = MagicMock()
        assignment.task_id = '42'
        assignment.task = {'title': 'Recovered task'}

        sem = MagicMock()
        sem.release = MagicMock()

        with patch('orchestrator.harness.TaskWorkflow') as MockWorkflow:
            mock_wf = AsyncMock()
            mock_wf.run.return_value = MagicMock(value='done')
            mock_wf.metrics = MagicMock(
                total_cost_usd=0.0,
                total_duration_ms=0,
                agent_invocations=0,
            )
            MockWorkflow.return_value = mock_wf

            await harness._run_slot(assignment, sem)

            # Verify TaskWorkflow was created with the recovered plan
            call_kwargs = MockWorkflow.call_args.kwargs
            assert call_kwargs['initial_plan'] is plan

        # Plan should be consumed (popped)
        assert '42' not in harness._recovered_plans

    async def test_no_injection_without_recovered_plan(self, harness: Harness):
        """Without a recovered plan, initial_plan should be None."""
        assignment = MagicMock()
        assignment.task_id = '99'
        assignment.task = {'title': 'Fresh task'}

        sem = MagicMock()
        sem.release = MagicMock()

        with patch('orchestrator.harness.TaskWorkflow') as MockWorkflow:
            mock_wf = AsyncMock()
            mock_wf.run.return_value = MagicMock(value='done')
            mock_wf.metrics = MagicMock(
                total_cost_usd=0.0,
                total_duration_ms=0,
                agent_invocations=0,
            )
            MockWorkflow.return_value = mock_wf

            await harness._run_slot(assignment, sem)

            call_kwargs = MockWorkflow.call_args.kwargs
            assert call_kwargs['initial_plan'] is None

    async def test_recover_plan_task_id_mismatch_cleaned_up(self, harness: Harness):
        """Plan whose task_id doesn't match the worktree dir -> cleaned up."""
        plan = _make_plan(steps_done=3, steps_total=5, task_id='216')
        wt = _setup_worktree(harness.git_ops.worktree_base, '369', plan)

        await harness._recover_crashed_tasks()

        assert '369' not in harness._recovered_plans
        harness.git_ops.cleanup_worktree.assert_called_once_with(wt, '369')  # type: ignore[attr-defined]

    async def test_multiple_worktrees_mixed(self, harness: Harness):
        """Multiple worktrees: one recovered, one cleaned, one no-progress."""
        base = harness.git_ops.worktree_base

        # Task with progress — should be recovered
        plan_good = _make_plan(steps_done=2, steps_total=4, task_id='50')
        _setup_worktree(base, '50', plan_good)

        # Task with no plan — should be cleaned
        wt_noplan = _setup_worktree(base, '51')

        # Task with no progress — should be cleaned
        plan_empty = _make_plan(steps_done=0, steps_total=3)
        wt_noprog = _setup_worktree(base, '52', plan_empty)

        await harness._recover_crashed_tasks()

        assert '50' in harness._recovered_plans
        assert '51' not in harness._recovered_plans
        assert '52' not in harness._recovered_plans

        cleanup_calls = harness.git_ops.cleanup_worktree.call_args_list  # type: ignore[attr-defined]
        cleaned_paths = {str(c.args[0]) for c in cleanup_calls}
        assert str(wt_noplan) in cleaned_paths
        assert str(wt_noprog) in cleaned_paths
        assert len(cleanup_calls) == 2
