"""Tests that _setup_worktree_and_artifacts threads train metadata to create_worktree.

Two cases:
  (a) task.metadata contains 'train' dict → forwarded as train=<dict> kwarg.
  (b) task.metadata absent or empty → train=None is forwarded.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import WorktreeInfo
from orchestrator.workflow import TaskWorkflow


def _make_workflow(
    *,
    project_root: Path,
    task_id: str = '101',
    task_metadata: dict | None = None,
) -> TaskWorkflow:
    """Return a minimal TaskWorkflow with scheduler/git_ops mocked."""
    assignment = MagicMock()
    assignment.task_id = task_id
    task_dict: dict = {'id': task_id, 'title': 'T', 'description': 'd'}
    if task_metadata is not None:
        task_dict['metadata'] = task_metadata
    assignment.task = task_dict
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.project_root = project_root

    scheduler = MagicMock()
    scheduler.set_task_status = AsyncMock(return_value=None)
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.get_status = AsyncMock(return_value=None)

    git_ops = MagicMock()
    git_ops.worktree_base = project_root / '.worktrees'

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    return wf


@pytest.mark.asyncio
async def test_train_metadata_forwarded_to_create_worktree(tmp_path: Path):
    """task.metadata['train'] must be forwarded as train= kwarg to create_worktree."""
    train_meta = {'id': 'T1', 'order': 1, 'members': ['100', '101']}
    wf = _make_workflow(project_root=tmp_path, task_metadata={'train': train_meta})
    base_sha = 'a' * 40

    wf.git_ops.create_worktree = AsyncMock(
        return_value=WorktreeInfo(path=tmp_path, base_commit=base_sha, stale_commits=None)
    )
    wf._sync_worktree_venvs = AsyncMock()

    with patch('orchestrator.workflow._run', new=AsyncMock(return_value=(0, '', ''))):
        await wf._setup_worktree_and_artifacts('task/101')

    call_kwargs = wf.git_ops.create_worktree.await_args.kwargs
    assert call_kwargs.get('train') == train_meta


@pytest.mark.asyncio
async def test_no_train_metadata_passes_none(tmp_path: Path):
    """When task.metadata has no 'train' key, train=None is forwarded."""
    wf = _make_workflow(project_root=tmp_path, task_metadata={})
    base_sha = 'b' * 40

    wf.git_ops.create_worktree = AsyncMock(
        return_value=WorktreeInfo(path=tmp_path, base_commit=base_sha, stale_commits=None)
    )
    wf._sync_worktree_venvs = AsyncMock()

    with patch('orchestrator.workflow._run', new=AsyncMock(return_value=(0, '', ''))):
        await wf._setup_worktree_and_artifacts('task/101')

    call_kwargs = wf.git_ops.create_worktree.await_args.kwargs
    assert call_kwargs.get('train') is None


@pytest.mark.asyncio
async def test_no_metadata_key_at_all_passes_none(tmp_path: Path):
    """When task has no 'metadata' key at all, train=None is forwarded."""
    wf = _make_workflow(project_root=tmp_path, task_metadata=None)
    base_sha = 'c' * 40

    wf.git_ops.create_worktree = AsyncMock(
        return_value=WorktreeInfo(path=tmp_path, base_commit=base_sha, stale_commits=None)
    )
    wf._sync_worktree_venvs = AsyncMock()

    with patch('orchestrator.workflow._run', new=AsyncMock(return_value=(0, '', ''))):
        await wf._setup_worktree_and_artifacts('task/101')

    call_kwargs = wf.git_ops.create_worktree.await_args.kwargs
    assert call_kwargs.get('train') is None
