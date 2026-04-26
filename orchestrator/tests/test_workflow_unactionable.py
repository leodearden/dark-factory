"""Tests for ``TaskWorkflow._handle_unactionable_task_report``.

The helper processes the architect's ``.task/unactionable_task.json``
artifact: stops the steward early (defense-in-depth against a stale L0
race), then short-circuits to ``_mark_blocked(escalate_to_human=True)``
which submits a level-1 escalation directly without invoking the steward.
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
    mark_blocked: AsyncMock


def _make(
    *,
    worktree: Path,
    project_root: Path,
    task_id: str = '50',
    steward: object | None = None,
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

    scheduler = MagicMock()
    scheduler.set_task_status = AsyncMock()

    git_ops = MagicMock()

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
    wf._steward = steward

    # Mock _mark_blocked so we can verify it's called with escalate_to_human=True
    # without setting up the entire escalation queue + L1 plumbing.
    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    return _Fixture(wf=wf, artifacts=artifacts, mark_blocked=mark_blocked)


@pytest.mark.asyncio
async def test_writes_l1_via_escalate_to_human(tmp_path: Path):
    f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
    f.artifacts.write_unactionable_task(
        reason='spec contradicts already-merged refactor',
        evidence='task asks to add foo() but task 31 deleted it',
    )

    outcome = await f.wf._handle_unactionable_task_report()

    assert outcome == WorkflowOutcome.BLOCKED
    f.mark_blocked.assert_awaited_once()
    args, kwargs = f.mark_blocked.await_args
    # Reason is propagated into the block reason.
    assert 'spec contradicts already-merged refactor' in args[0]
    # The escalate_to_human=True flag is the load-bearing signal — it makes
    # _mark_blocked submit an L1 directly and skip the steward.
    assert kwargs['escalate_to_human'] is True
    # Detail carries the architect's evidence.
    assert 'task 31 deleted it' in kwargs['detail']
    # Artifact cleared so a re-run doesn't see a stale report.
    assert f.artifacts.read_unactionable_task() is None


@pytest.mark.asyncio
async def test_stops_steward_early_when_running(tmp_path: Path):
    """Defense-in-depth: handler stops a running steward before submitting L1."""
    steward = MagicMock()
    steward.stop = AsyncMock()
    f = _make(
        worktree=tmp_path / 'wt',
        project_root=tmp_path / 'proj',
        steward=steward,
    )
    f.artifacts.write_unactionable_task('reason', 'evidence')

    outcome = await f.wf._handle_unactionable_task_report()

    assert outcome == WorkflowOutcome.BLOCKED
    # Steward's stop() was awaited.
    steward.stop.assert_awaited_once()
    # And the workflow's _steward pointer cleared so the finally block
    # doesn't try to stop it twice.
    assert f.wf._steward is None
    f.mark_blocked.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_steward_present_does_not_raise(tmp_path: Path):
    """At PLAN time the steward typically hasn't been started — handler must no-op."""
    f = _make(
        worktree=tmp_path / 'wt',
        project_root=tmp_path / 'proj',
        steward=None,
    )
    f.artifacts.write_unactionable_task('reason', 'evidence')

    # Should not raise even though _steward is None.
    outcome = await f.wf._handle_unactionable_task_report()
    assert outcome == WorkflowOutcome.BLOCKED


@pytest.mark.asyncio
async def test_missing_reason_still_escalates_to_human(tmp_path: Path):
    """Malformed artifact: still L1 — the architect is signaling task is broken."""
    f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
    # Write artifact directly without reason field.
    (f.artifacts.root / 'unactionable_task.json').write_text(
        '{"evidence": "I forgot the reason"}\n'
    )

    outcome = await f.wf._handle_unactionable_task_report()

    assert outcome == WorkflowOutcome.BLOCKED
    f.mark_blocked.assert_awaited_once()
    args, kwargs = f.mark_blocked.await_args
    assert 'malformed' in args[0].lower()
    # Even malformed reports must escalate to human — the architect's
    # judgment that the task is unworkable shouldn't be lost to a missing
    # field.
    assert kwargs['escalate_to_human'] is True
    assert f.artifacts.read_unactionable_task() is None


@pytest.mark.asyncio
async def test_empty_reason_string_still_escalates(tmp_path: Path):
    f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
    f.artifacts.write_unactionable_task(reason='   ', evidence='e')

    outcome = await f.wf._handle_unactionable_task_report()

    assert outcome == WorkflowOutcome.BLOCKED
    f.mark_blocked.assert_awaited_once()
    args, kwargs = f.mark_blocked.await_args
    assert kwargs['escalate_to_human'] is True
