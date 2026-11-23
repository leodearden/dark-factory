"""A TaskWorkflow over a real git repo and a real EventStore, with its agent and scheduler collaborators doubled."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.agents.invoke import AgentResult
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify_cache.replay import CacheAccess
from orchestrator.workflow import TaskWorkflow

SCENE_TASK_ID = '42'


def scene_git(cwd: Path, *argv: str) -> str:
    return subprocess.run(
        ['git', '-c', 'user.email=t@t', '-c', 'user.name=t', *argv],
        cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout.strip()


def scene_commit(cwd: Path, name: str) -> None:
    (cwd / name).write_text(f'{name}\n')
    scene_git(cwd, 'add', '-A')
    scene_git(cwd, 'commit', '-q', '-m', name)


def task_scene_config(tmp_path: Path) -> OrchestratorConfig:
    """A one-commit repo on main under *tmp_path*, and the config a TaskWorkflow over it runs under."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    scene_git(repo, 'init', '-q', '-b', 'main')
    scene_commit(repo, 'lib.py')
    return OrchestratorConfig(
        project_root=repo,
        max_concurrent_tasks=1,
        max_verify_attempts=2,
        escalate_preexisting_main_break=False,
        git=GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin', worktree_dir='.worktrees',
        ),
    )


async def task_workflow(
    config: OrchestratorConfig, git_ops: GitOps, store: EventStore, monkeypatch: pytest.MonkeyPatch,
    *, verify_cache: CacheAccess | None = None,
) -> TaskWorkflow:
    worktree = await git_ops.create_worktree(SCENE_TASK_ID)
    workflow = TaskWorkflow(
        assignment=TaskAssignment(
            task_id=SCENE_TASK_ID,
            task={
                'id': SCENE_TASK_ID, 'title': 'X', 'description': '', 'status': 'pending',
                'metadata': {'files': ['lib']}, 'dependencies': [],
            },
            modules=['lib'],
        ),
        config=config, git_ops=git_ops,
        scheduler=MagicMock(), briefing=MagicMock(), mcp=MagicMock(),
        event_store=store, verify_cache=verify_cache,
    )
    workflow.worktree = worktree.path
    workflow._base_commit = worktree.base_commit
    workflow.artifacts = TaskArtifacts(worktree.path)
    workflow.artifacts.init(SCENE_TASK_ID, 'X', 'desc', base_commit=worktree.base_commit)
    workflow.plan = {'task_id': SCENE_TASK_ID, 'steps': []}
    workflow.briefing.build_debugger_prompt = AsyncMock(return_value='debug')
    monkeypatch.setattr(workflow, '_check_escalations', MagicMock(return_value=[]))
    monkeypatch.setattr(workflow, '_invoke', AsyncMock(return_value=AgentResult(success=True, output='')))
    return workflow


async def forked_task_workflow(
    config: OrchestratorConfig, git_ops: GitOps, store: EventStore, monkeypatch: pytest.MonkeyPatch,
    *, verify_cache: CacheAccess | None = None,
) -> TaskWorkflow:
    """A task branch with one commit of its own, forked one commit behind main's tip."""
    workflow = await task_workflow(config, git_ops, store, monkeypatch, verify_cache=verify_cache)
    assert workflow.worktree is not None
    scene_commit(workflow.worktree, 'task_change.py')
    scene_commit(config.project_root, 'main_change.py')
    return workflow
