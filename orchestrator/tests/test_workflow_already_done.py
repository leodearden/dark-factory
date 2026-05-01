"""Tests for ``TaskWorkflow._handle_already_done_report``.

The helper processes the architect's ``.task/already_done.json`` artifact:
validates the cited commit is reachable from main, then sets task status
to ``done`` with provenance.  Validation failures route to ``_mark_blocked``
without escalating to a human — they signal an architect mistake, not an
unworkable task.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome, WorkflowState


@dataclass
class _Fixture:
    wf: TaskWorkflow
    artifacts: TaskArtifacts
    set_task_status: AsyncMock
    is_ancestor: AsyncMock
    get_main_sha: AsyncMock


def _make(
    *,
    worktree: Path,
    project_root: Path,
    task_id: str = '50',
    commit_on_main: bool = True,
    main_sha: str = 'mainsha123',
) -> _Fixture:
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

    set_task_status = AsyncMock()
    scheduler = MagicMock()
    scheduler.set_task_status = set_task_status
    # Fix 1: workflow refreshes metadata.files via update_task before
    # set_task_status('done').  Stub as AsyncMock so the await succeeds.
    scheduler.update_task = AsyncMock(return_value=True)

    is_ancestor = AsyncMock(return_value=commit_on_main)
    get_main_sha = AsyncMock(return_value=main_sha)
    git_ops = MagicMock()
    git_ops.is_ancestor = is_ancestor
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
        set_task_status=set_task_status,
        is_ancestor=is_ancestor,
        get_main_sha=get_main_sha,
    )


@pytest.mark.asyncio
async def test_valid_commit_sets_done_with_provenance(tmp_path: Path):
    f = _make(
        worktree=tmp_path / 'wt',
        project_root=tmp_path / 'proj',
        commit_on_main=True,
    )
    f.artifacts.write_already_done(
        commit='abcdef1234567890',
        evidence='helpers.foo on main at task 42',
    )

    outcome = await f.wf._handle_already_done_report()

    assert outcome == WorkflowOutcome.DONE
    assert f.wf.state == WorkflowState.DONE
    f.set_task_status.assert_awaited_once()
    args, kwargs = f.set_task_status.await_args
    assert args[0] == '50'
    assert args[1] == 'done'
    provenance = kwargs['done_provenance']
    # 2026-04-27 hardening: kind discriminator added; verified_by/evidence
    # folded into the structured note field so the schema accepts the call.
    assert provenance['kind'] == 'found_on_main'
    assert provenance['commit'] == 'abcdef1234567890'
    assert 'architect' in provenance['note']
    assert 'helpers.foo on main at task 42' in provenance['note']
    # Artifact cleared so a re-run doesn't see a stale report.
    assert f.artifacts.read_already_done() is None
    # is_ancestor called with (commit, main_sha) — strictly main-membership check.
    f.is_ancestor.assert_awaited_once_with('abcdef1234567890', 'mainsha123')


@pytest.mark.asyncio
async def test_commit_not_on_main_blocks_without_l1(tmp_path: Path):
    f = _make(
        worktree=tmp_path / 'wt',
        project_root=tmp_path / 'proj',
        commit_on_main=False,
    )
    f.artifacts.write_already_done('feedfacecafebeef', 'evidence text')

    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    f.wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    outcome = await f.wf._handle_already_done_report()

    assert outcome == WorkflowOutcome.BLOCKED
    mark_blocked.assert_awaited_once()
    assert mark_blocked.await_args is not None
    args, kwargs = mark_blocked.await_args
    assert 'not reachable from main' in args[0]
    # Validation failure must NOT escalate to human — this is an architect bug.
    assert kwargs.get('escalate_to_human') is not True
    # Status NOT set to done.
    f.set_task_status.assert_not_called()
    # Artifact cleared even on validation failure (don't loop on stale report).
    assert f.artifacts.read_already_done() is None


@pytest.mark.asyncio
async def test_missing_commit_blocks(tmp_path: Path):
    f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
    # Write artifact directly without commit field.
    (f.artifacts.root / 'already_done.json').write_text(
        '{"evidence": "I forgot the commit"}\n'
    )
    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    f.wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    outcome = await f.wf._handle_already_done_report()

    assert outcome == WorkflowOutcome.BLOCKED
    mark_blocked.assert_awaited_once()
    assert mark_blocked.await_args is not None
    args, kwargs = mark_blocked.await_args
    assert 'malformed' in args[0].lower()
    assert kwargs.get('escalate_to_human') is not True
    f.set_task_status.assert_not_called()
    f.is_ancestor.assert_not_called()
    assert f.artifacts.read_already_done() is None


@pytest.mark.asyncio
async def test_empty_commit_string_blocks(tmp_path: Path):
    f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
    f.artifacts.write_already_done(commit='   ', evidence='blank')
    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    f.wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    outcome = await f.wf._handle_already_done_report()

    assert outcome == WorkflowOutcome.BLOCKED
    mark_blocked.assert_awaited_once()
    f.set_task_status.assert_not_called()
    f.is_ancestor.assert_not_called()
