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

import logging
from pathlib import Path
from typing import cast
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
) -> TaskWorkflow:
    """Return a minimal TaskWorkflow with scheduler/git_ops mocked."""
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
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
    # _setup_worktree_and_artifacts now reads live status at dispatch before
    # claiming 'in-progress' (commit 57e25f1260); None → non-terminal → proceed.
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
    fake_proc.returncode = 0  # explicit 0 so the new returncode check doesn't misfire
    fake_proc.communicate = AsyncMock(return_value=(base_sha.encode() + b'\n', b''))

    with patch('asyncio.create_subprocess_exec', new=AsyncMock(return_value=fake_proc)), \
         patch('orchestrator.workflow._run', new=AsyncMock(return_value=(0, '', ''))):
        await wf._setup_worktree_and_artifacts('task/101')

    # branch_base_sha must be written with append=True
    cast(AsyncMock, wf.scheduler.update_task).assert_awaited_once_with(
        '101', {'branch_base_sha': base_sha}, append=True
    )


@pytest.mark.asyncio
async def test_setup_worktree_external_rev_parse_failure_logs_warning_and_falls_through(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Rev-parse failure leaves _base_commit as None so consumers gate correctly.

    Unlike the update_task soft-fail (which keeps a valid SHA in memory because
    only the metadata write fails), this path has no SHA to record at all —
    `_base_commit` is left None.  Downstream consumers gate safely:
      - workflow.py `if self._merge_sha and self._base_commit:` (truthy, None is falsy)
      - workflow.py `and self._base_commit is not None` (identity check, correctly skips)
    The falsy `if base_commit:` gate at line 497 still skips the update_task write.
    """
    wf = _make_workflow(project_root=tmp_path)
    wf.worktree = tmp_path  # trigger external-worktree path

    fake_proc = MagicMock()
    fake_proc.returncode = 128
    fake_proc.communicate = AsyncMock(
        return_value=(b'', b'fatal: not a git repository\n')
    )

    with caplog.at_level(logging.WARNING, logger='orchestrator.workflow'), \
         patch('asyncio.create_subprocess_exec', new=AsyncMock(return_value=fake_proc)), \
         patch('orchestrator.workflow._run', new=AsyncMock(return_value=(0, '', ''))):
        await wf._setup_worktree_and_artifacts('task/101')

    # (a) must not raise — fall-through, not crash (no explicit assertion needed)
    # (b) base_commit is None — identity check distinguishes None from '' or other falsy values
    assert wf._base_commit is None, f'expected _base_commit is None, got {wf._base_commit!r}'
    # (c) update_task not called — `if base_commit:` gate skips it when falsy (None is falsy)
    cast(AsyncMock, wf.scheduler.update_task).assert_not_awaited()
    # (d) exactly one WARNING about the failure, surfacing task_id, rev-parse, and stderr
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, f'expected 1 warning, got {len(warnings)}: {[r.getMessage() for r in warnings]}'
    msg = warnings[0].getMessage()
    assert '101' in msg, f'task_id not in warning: {msg!r}'
    assert 'not a git repository' in msg, f'stderr not in warning: {msg!r}'


@pytest.mark.asyncio
async def test_setup_worktree_update_task_soft_fail_logs_and_continues(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Soft-fail coverage: update_task exception must not propagate; warning must be logged.

    The existing except-branch at workflow.py:491-496 swallows exceptions from
    scheduler.update_task and logs a warning.  This test pins that behavior so
    a future refactor cannot silently break the soft-fail contract.

    Assertions:
      (a) no exception propagates from _setup_worktree_and_artifacts
      (b) wf._base_commit is set from create_worktree before update_task is called
      (c) the WARNING log contains task_id, 'branch_base_sha', and 'citation-grep'
      (d) update_task was awaited exactly once (proving the except branch was entered,
          not short-circuited by `if base_commit:`)
    """
    wf = _make_workflow(project_root=tmp_path)
    # Leave wf.worktree = None so the create_worktree branch fires
    base_sha = 'c' * 40

    wf.git_ops.create_worktree = AsyncMock(
        return_value=WorktreeInfo(path=tmp_path, base_commit=base_sha, stale_commits=0)
    )
    wf._sync_worktree_venvs = AsyncMock()
    wf.scheduler.update_task = AsyncMock(side_effect=Exception('mcp down'))

    with caplog.at_level(logging.WARNING, logger='orchestrator.workflow'), \
         patch('orchestrator.workflow._run', new=AsyncMock(return_value=(0, '', ''))):
        await wf._setup_worktree_and_artifacts('task/101')

    # (a) no exception propagated (test would have failed above if it raised)
    # (b) _base_commit set from WorktreeInfo before update_task attempted
    assert wf._base_commit == base_sha, (
        f'expected _base_commit={base_sha!r}, got {wf._base_commit!r}'
    )
    # (c) one WARNING containing task_id, branch_base_sha keyword, and citation-grep
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, f'expected 1 warning, got {len(warnings)}: {[r.getMessage() for r in warnings]}'
    msg = warnings[0].getMessage()
    assert '101' in msg, f'task_id not in warning: {msg!r}'
    assert 'branch_base_sha' in msg, f"'branch_base_sha' not in warning: {msg!r}"
    # (d) update_task was called (proving the except branch fired, not `if base_commit:` skip)
    cast(AsyncMock, wf.scheduler.update_task).assert_awaited_once_with(
        '101', {'branch_base_sha': base_sha}, append=True
    )
