"""A dispatch splits its artifacts across the lane root and the task root (task 3155).

PRD boundary case B9: after one cold dispatch the task-lifetime artifacts live
in ``.task-meta/task-<id>/`` and the lane root holds only lane-keyed names.
An external (eval-mode) worktree stays single-root, because parallel trials of
one task id under ``<eval_root>/<id>/`` would otherwise share one task root.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _workflow_helpers import AgentStub, _build_workflow, _init_repo

from orchestrator.agents import sandbox_dispatch
from orchestrator.artifacts import ARTIFACT_ROOTS, ArtifactRoot, TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig, SandboxConfig
from orchestrator.dry_run_unblock import investigation_task_name
from orchestrator.git_ops import GitOps, _run
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify import VerifyResult

_LANE_KEYED = frozenset(name for name, root in ARTIFACT_ROOTS.items() if root is ArtifactRoot.LANE)


@pytest.fixture
def config(tmp_path: Path) -> OrchestratorConfig:
    repo = tmp_path / 'repo'
    repo.mkdir()
    _init_repo(repo)
    return OrchestratorConfig(
        project_root=repo,
        max_concurrent_tasks=1,
        max_execute_iterations=5,
        max_verify_attempts=3,
        max_review_cycles=2,
        git=GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin', worktree_dir='.worktrees',
        ),
        sandbox=SandboxConfig(enabled=False),
    )


@pytest.fixture
def git_ops(config: OrchestratorConfig) -> GitOps:
    return GitOps(config.git, config.project_root)


@pytest.fixture
def task_assignment() -> TaskAssignment:
    return TaskAssignment(
        task_id='42',
        task={
            'id': '42',
            'title': 'Add farewell function',
            'description': 'Add a farewell(name) function to lib.py with tests',
            'status': 'pending',
            'metadata': {'files': ['lib']},
            'dependencies': [],
        },
        modules=['lib'],
    )


@pytest.fixture
def stub(monkeypatch) -> AgentStub:
    stub = AgentStub()
    monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
    monkeypatch.setattr(
        'orchestrator.workflow.run_scoped_verification',
        AsyncMock(return_value=VerifyResult(
            passed=True, test_output='', lint_output='', type_output='',
            summary='All checks passed',
        )),
    )
    return stub


@pytest.mark.asyncio
async def test_cold_dispatch_keeps_task_artifacts_in_the_task_root(
    config, git_ops, task_assignment, stub,
):
    workflow, _ = _build_workflow(config, git_ops, task_assignment, stub)
    observed: dict[str, object] = {}
    lane_names: set[str] = set()
    original_cleanup = git_ops.cleanup_worktree

    async def observe_then_cleanup(worktree: Path, branch: str) -> None:
        task_root = TaskArtifacts.task_root_for(worktree.parent, '42')
        lane_plan = worktree / '.task' / 'plan.json'
        observed.update(
            task_plan=(task_root / 'plan.json').is_file(),
            task_iterations=(task_root / 'iterations.jsonl').is_file(),
            lane_plan_is_symlink=lane_plan.is_symlink(),
            lane_plan_target=lane_plan.resolve(strict=True),
            expected_target=(task_root / 'plan.json').resolve(),
            lane_iterations_target=(worktree / '.task' / 'iterations.jsonl').resolve(strict=True),
            expected_iterations=(task_root / 'iterations.jsonl').resolve(),
        )
        lane_names.update(
            p.name for p in TaskArtifacts.meta_root_for(worktree.parent, worktree.name).iterdir()
        )
        await original_cleanup(worktree, branch)

    git_ops.cleanup_worktree = observe_then_cleanup

    await workflow.run()

    assert observed['task_plan'] is True
    assert observed['task_iterations'] is True
    assert observed['lane_plan_is_symlink'] is True
    assert observed['lane_plan_target'] == observed['expected_target']
    assert observed['lane_iterations_target'] == observed['expected_iterations']
    assert lane_names <= _LANE_KEYED


async def _eval_worktree(config: OrchestratorConfig, tmp_path: Path) -> Path:
    worktree = tmp_path / 'evals' / '42' / 'run-a'
    await _run(['git', 'worktree', 'add', '-b', 'eval/42-a', str(worktree)], cwd=config.project_root)
    return worktree


@pytest.mark.asyncio
async def test_external_worktree_stays_single_root(config, git_ops, task_assignment, stub, tmp_path):
    worktree = await _eval_worktree(config, tmp_path)
    workflow, _ = _build_workflow(config, git_ops, task_assignment, stub)
    workflow.worktree = worktree

    await workflow.run()

    lane = TaskArtifacts.meta_root_for(worktree.parent, worktree.name)
    assert stub.calls[0] == 'architect'
    assert workflow.artifacts is not None
    assert workflow.artifacts.lane_root == workflow.artifacts.task_root == lane
    assert (lane / 'plan.json').is_file()
    assert not TaskArtifacts.task_root_for(worktree.parent, '42').exists()


@pytest.mark.asyncio
async def test_external_worktree_sandbox_grants_no_task_root(
    config, git_ops, task_assignment, stub, tmp_path,
):
    """Parallel eval trials of one id would all share the task root, so a
    single-root dispatch must not be granted it."""
    config.sandbox.enabled = True
    sandbox_dispatch.set_backend('none')
    worktree = await _eval_worktree(config, tmp_path)
    workflow, _ = _build_workflow(config, git_ops, task_assignment, stub)
    workflow.worktree = worktree

    await workflow.run()

    extras = stub.sandbox_extras_by_role['implementer']
    assert extras is not None
    assert str(TaskArtifacts.meta_root_for(worktree.parent, worktree.name).resolve()) in extras
    assert str(TaskArtifacts.task_root_for(worktree.parent, '42').resolve()) not in extras


@pytest.mark.asyncio
async def test_external_worktree_investigation_reads_reviews_from_its_lane_root(
    config, git_ops, task_assignment, stub, tmp_path, monkeypatch,
):
    """The dispatch's own root decision, not a re-derived task root, tells the
    dry-run investigator where this run's reviewers wrote."""
    investigations: list[dict] = []

    async def record_investigation(**investigation) -> None:
        investigations.append(investigation)

    monkeypatch.setattr('orchestrator.workflow.run_dry_run_unblock', record_investigation)
    monkeypatch.setattr(
        'orchestrator.workflow.run_scoped_verification',
        AsyncMock(return_value=VerifyResult(
            passed=False, test_output='FAILED', lint_output='', type_output='',
            summary='tests failed',
        )),
    )
    config.unblock_auto.enabled = True
    worktree = await _eval_worktree(config, tmp_path)
    workflow, _ = _build_workflow(config, git_ops, task_assignment, stub)
    workflow.worktree = worktree

    await workflow.run()
    await asyncio.gather(*(
        t for t in asyncio.all_tasks() if t.get_name() == investigation_task_name('42')
    ))

    lane = TaskArtifacts.meta_root_for(worktree.parent, worktree.name)
    assert [i['reviews_dir'] for i in investigations] == [lane / 'reviews']
