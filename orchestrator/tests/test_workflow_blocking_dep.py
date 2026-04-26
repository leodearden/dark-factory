"""Tests for ``TaskWorkflow._handle_blocking_dep_report`` (Fix B).

The helper processes the architect's ``.task/blocking_dependency.json``
artifact: registers the Taskmaster dependency for non-terminal deps, or
rebases + retries the architect when the named dep is already terminal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


@dataclass
class _Fixture:
    wf: TaskWorkflow
    artifacts: TaskArtifacts
    dispatch_tool: AsyncMock
    set_task_status: AsyncMock
    rebase_onto_main: AsyncMock
    get_main_sha: AsyncMock
    get_status: AsyncMock


def _make(
    *,
    worktree: Path,
    project_root: Path,
    task_id: str = '50',
    dep_status: str = 'pending',
    add_dep_raises: bool = False,
    rebase_succeeds: bool = True,
) -> _Fixture:
    """Build a TaskWorkflow + handles to the AsyncMock objects.

    Returning the mocks separately means assertions like
    ``dispatch_tool.assert_awaited_once_with(...)`` go through a directly-
    typed ``AsyncMock`` variable (pyright-clean) rather than via the
    workflow's Protocol-typed scheduler attribute.
    """
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    assignment.modules = ['mod_a']

    config = MagicMock()
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = project_root

    get_status = AsyncMock(return_value=dep_status)
    set_task_status = AsyncMock()
    if add_dep_raises:
        dispatch_tool = AsyncMock(side_effect=RuntimeError('mcp down'))
    else:
        dispatch_tool = AsyncMock(return_value={})

    scheduler = MagicMock()
    scheduler.get_status = get_status
    scheduler._dispatch_tool = dispatch_tool
    scheduler.set_task_status = set_task_status

    rebase_onto_main = AsyncMock(return_value=rebase_succeeds)
    get_main_sha = AsyncMock(return_value='newmain123')

    git_ops = MagicMock()
    git_ops.rebase_onto_main = rebase_onto_main
    git_ops.get_main_sha = get_main_sha

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )

    worktree.mkdir(parents=True, exist_ok=True)
    artifacts = TaskArtifacts(worktree)
    artifacts.init(task_id, 'T', 'd', base_commit='oldbase')
    wf.artifacts = artifacts
    wf.worktree = worktree

    return _Fixture(
        wf=wf, artifacts=artifacts,
        dispatch_tool=dispatch_tool,
        set_task_status=set_task_status,
        rebase_onto_main=rebase_onto_main,
        get_main_sha=get_main_sha,
        get_status=get_status,
    )


@pytest.mark.asyncio
async def test_non_terminal_dep_registers_and_returns_requeued(tmp_path: Path):
    f = _make(
        worktree=tmp_path / 'wt',
        project_root=tmp_path / 'proj',
        dep_status='pending',
    )
    f.artifacts.write_blocking_dependency('42', 'missing helpers.foo', 'sha-x')

    outcome = await f.wf._handle_blocking_dep_report(rebase_retry_used=False)

    assert outcome == WorkflowOutcome.REQUEUED
    # add_dependency MCP tool was dispatched with the correct args.
    f.dispatch_tool.assert_awaited_once()
    args, kwargs = f.dispatch_tool.await_args
    assert args[0] == 'add_dependency'
    payload = args[1]
    assert payload['id'] == '50'
    assert payload['depends_on'] == '42'
    # Status reset to pending so dep-check keeps task from dispatching.
    f.set_task_status.assert_awaited_once_with('50', 'pending')
    # Artifact cleared so a subsequent architect doesn't see a stale report.
    assert f.artifacts.read_blocking_dependency() is None
    # No rebase happened — the dep is non-terminal.
    f.rebase_onto_main.assert_not_called()


@pytest.mark.asyncio
async def test_terminal_dep_rebases_and_returns_none(tmp_path: Path):
    """Dep already done → rebase + clear, return None so caller retries architect."""
    f = _make(
        worktree=tmp_path / 'wt',
        project_root=tmp_path / 'proj',
        dep_status='done',
    )
    f.artifacts.write_blocking_dependency('42', 'missing helpers.foo', 'old-sha')

    outcome = await f.wf._handle_blocking_dep_report(rebase_retry_used=False)

    assert outcome is None  # caller must retry architect
    f.rebase_onto_main.assert_awaited_once_with(f.wf.worktree)
    # add_dependency was NOT called — the dep is already terminal.
    f.dispatch_tool.assert_not_called()
    f.set_task_status.assert_not_called()
    # Artifact cleared, base commit updated to new main sha.
    assert f.artifacts.read_blocking_dependency() is None
    assert f.artifacts.read_base_commit() == 'newmain123'


@pytest.mark.asyncio
async def test_terminal_dep_cancelled_also_rebases(tmp_path: Path):
    f = _make(
        worktree=tmp_path / 'wt',
        project_root=tmp_path / 'proj',
        dep_status='cancelled',
    )
    f.artifacts.write_blocking_dependency('42', 'r', 'old-sha')

    outcome = await f.wf._handle_blocking_dep_report(rebase_retry_used=False)

    assert outcome is None
    f.rebase_onto_main.assert_awaited_once_with(f.wf.worktree)


@pytest.mark.asyncio
async def test_rebase_failure_still_clears_and_retries(tmp_path: Path):
    """If rebase fails, helper logs but still clears artifact and returns None."""
    f = _make(
        worktree=tmp_path / 'wt',
        project_root=tmp_path / 'proj',
        dep_status='done',
        rebase_succeeds=False,
    )
    f.artifacts.write_blocking_dependency('42', 'r', 'old-sha')

    outcome = await f.wf._handle_blocking_dep_report(rebase_retry_used=False)

    assert outcome is None
    # base_commit not updated (rebase didn't actually happen).
    assert f.artifacts.read_base_commit() == 'oldbase'
    assert f.artifacts.read_blocking_dependency() is None


@pytest.mark.asyncio
async def test_terminal_dep_with_retry_used_blocks(tmp_path: Path):
    """When rebase-retry already used and dep is still terminal → BLOCKED."""
    f = _make(
        worktree=tmp_path / 'wt',
        project_root=tmp_path / 'proj',
        dep_status='done',
    )
    f.artifacts.write_blocking_dependency('42', 'still claims missing', 'old-sha')

    # Avoid touching escalation queue / steward.
    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    f.wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    outcome = await f.wf._handle_blocking_dep_report(rebase_retry_used=True)

    assert outcome == WorkflowOutcome.BLOCKED
    mark_blocked.assert_awaited_once()
    assert mark_blocked.await_args is not None
    args, _ = mark_blocked.await_args
    assert 'rebase-retry already used' in args[0]
    assert '42' in args[0]
    # Artifact cleared even on the rebase-retry-used path.
    assert f.artifacts.read_blocking_dependency() is None
    # No rebase attempted on this terminal-after-retry path.
    f.rebase_onto_main.assert_not_called()


@pytest.mark.asyncio
async def test_malformed_artifact_blocks(tmp_path: Path):
    f = _make(
        worktree=tmp_path / 'wt',
        project_root=tmp_path / 'proj',
    )
    # Write an artifact missing depends_on_task_id by hand.
    (f.artifacts.root / 'blocking_dependency.json').write_text(
        '{"reason": "I forgot the dep id"}\n'
    )
    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    f.wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    outcome = await f.wf._handle_blocking_dep_report(rebase_retry_used=False)

    assert outcome == WorkflowOutcome.BLOCKED
    mark_blocked.assert_awaited_once()
    assert mark_blocked.await_args is not None
    args, _ = mark_blocked.await_args
    assert 'malformed' in args[0].lower()
    assert f.artifacts.read_blocking_dependency() is None


@pytest.mark.asyncio
async def test_add_dependency_failure_still_requeues(tmp_path: Path):
    """If add_dependency MCP call fails, the workflow still requeues — the
    dep was not registered but we don't want to spin in the no-plan loop.
    A subsequent run will re-detect (or the human will sort it out)."""
    f = _make(
        worktree=tmp_path / 'wt',
        project_root=tmp_path / 'proj',
        dep_status='pending',
        add_dep_raises=True,
    )
    f.artifacts.write_blocking_dependency('42', 'r', 'sha')

    outcome = await f.wf._handle_blocking_dep_report(rebase_retry_used=False)

    assert outcome == WorkflowOutcome.REQUEUED
    f.set_task_status.assert_awaited_once_with('50', 'pending')
    assert f.artifacts.read_blocking_dependency() is None
