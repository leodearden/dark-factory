"""Tests that _setup_worktree_and_artifacts writes branch_base_sha to task metadata.

Two paths are exercised:
  (a) create_worktree path — the normal task-start flow where git_ops.create_worktree
      is called and WorktreeInfo.base_commit is used.
  (b) external-worktree path — eval mode where self.worktree is already set and
      the base commit is obtained via `git rev-parse HEAD`.

Both paths must call scheduler.update_task(task_id, {'branch_base_sha': base_commit},
append=True) immediately after self._base_commit is set.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.git_ops import WorktreeInfo
from orchestrator.workflow import TaskWorkflow


def _make_workflow(
    *,
    project_root: Path,
    task_id: str = '101',
) -> TaskWorkflow:
    """Return a minimal TaskWorkflow with scheduler/git_ops mocked."""
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    assignment.modules = ['mod_a']

    config = MagicMock()
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.project_root = project_root

    scheduler = MagicMock()
    scheduler.set_task_status = AsyncMock(return_value=None)
    scheduler.update_task = AsyncMock(return_value=True)

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
async def test_setup_worktree_writes_branch_base_sha(tmp_path: Path):
    """create_worktree path: branch_base_sha is written from WorktreeInfo.base_commit."""
    wf = _make_workflow(project_root=tmp_path)
    base_sha = 'a' * 40

    wf.git_ops.create_worktree = AsyncMock(
        return_value=WorktreeInfo(path=tmp_path, base_commit=base_sha, stale_commits=0)
    )
    wf._sync_worktree_venvs = AsyncMock()

    # Stub the .task/ contamination check in _setup_worktree_and_artifacts
    with patch('orchestrator.workflow._run', new=AsyncMock(return_value=(0, '', ''))):
        await wf._setup_worktree_and_artifacts('task/101')

    # branch_base_sha must be written with append=True so existing metadata
    # (prd, files, modules, etc.) is not clobbered.
    cast(AsyncMock, wf.scheduler.update_task).assert_awaited_once_with(
        '101', {'branch_base_sha': base_sha}, append=True
    )


@pytest.mark.asyncio
async def test_setup_worktree_writes_branch_base_sha_external_worktree(tmp_path: Path):
    """External-worktree (eval) path: branch_base_sha comes from git rev-parse HEAD."""
    wf = _make_workflow(project_root=tmp_path)
    # Pre-set worktree to trigger the external-worktree (eval) path
    wf.worktree = tmp_path
    base_sha = 'b' * 40

    # Mock the git rev-parse HEAD subprocess
    fake_proc = MagicMock()
    fake_proc.communicate = AsyncMock(return_value=(base_sha.encode() + b'\n', b''))

    with patch('asyncio.create_subprocess_exec', new=AsyncMock(return_value=fake_proc)), \
         patch('orchestrator.workflow._run', new=AsyncMock(return_value=(0, '', ''))):
        await wf._setup_worktree_and_artifacts('task/101')

    # branch_base_sha must be written with append=True
    cast(AsyncMock, wf.scheduler.update_task).assert_awaited_once_with(
        '101', {'branch_base_sha': base_sha}, append=True
    )
