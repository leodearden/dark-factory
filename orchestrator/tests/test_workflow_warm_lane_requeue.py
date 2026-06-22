"""workflow.run() maps warm-lane create_worktree exceptions to the right outcomes.

When warm_lane_pool is enabled, create_worktree raises typed exceptions instead
of falling through to the cold path:
  WarmLanePoolExhausted → WorkflowOutcome.REQUEUED  (backpressure)
  WarmLaneDiskPressure  → WorkflowOutcome.REQUEUED  (transient infra)
  RuntimeError          → WorkflowOutcome.BLOCKED   (fault → existing blocked+L1 path)

These tests are RED today because the broad ``except Exception`` in run() maps
every create_worktree raise to BLOCKED.  They turn GREEN in step-10 when
``except WarmLaneRequeue`` is inserted before the broad handler.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import (
    WarmLaneDiskPressure,
    WarmLanePoolExhausted,
)
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


def _make_workflow(*, tmp_path: Path, task_id: str = '1859') -> TaskWorkflow:
    """Build a minimal TaskWorkflow with mocked deps for create_worktree tests.

    Unlike the _make_workflow in test_workflow_worktree_missing.py, this one
    does NOT pre-populate wf.worktree — we want run() to call create_worktree
    so the exception raised there propagates through run()'s exception handlers.
    """
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'Test task', 'description': 'desc'}
    assignment.modules = []  # empty → _resolve_module_configs returns [] cleanly

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = tmp_path / 'proj'
    # worktree_identity_guard_enabled is read inside _setup_worktree_and_artifacts
    # before create_worktree; let MagicMock return a truthy value (the default).

    scheduler = MagicMock()
    # get_status: return 'pending' so the pre-empt check doesn't raise TerminalExitRejection
    scheduler.get_status = AsyncMock(return_value='pending')
    scheduler.set_task_status = AsyncMock()

    git_ops = MagicMock()
    # create_worktree will be configured per-test

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    # Do NOT set wf.worktree — keep it None so run() calls create_worktree.
    return wf


# ---------------------------------------------------------------------------
# WarmLanePoolExhausted → REQUEUED
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pool_exhausted_returns_requeued(tmp_path: Path):
    """WarmLanePoolExhausted raised by create_worktree → run() returns REQUEUED.

    Fails today: broad except Exception maps it to BLOCKED.
    Turns GREEN in step-10 when except WarmLaneRequeue is inserted.
    """
    wf = _make_workflow(tmp_path=tmp_path)
    wf.git_ops.create_worktree = AsyncMock(
        side_effect=WarmLanePoolExhausted(
            "warm-lane pool exhausted for branch '1859'; requeue"
        )
    )
    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    outcome = await wf.run()

    assert outcome == WorkflowOutcome.REQUEUED
    mark_blocked.assert_not_awaited()


# ---------------------------------------------------------------------------
# WarmLaneDiskPressure → REQUEUED
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_disk_pressure_returns_requeued(tmp_path: Path):
    """WarmLaneDiskPressure raised by create_worktree → run() returns REQUEUED.

    Fails today: broad except Exception maps it to BLOCKED.
    Turns GREEN in step-10 when except WarmLaneRequeue is inserted.
    """
    wf = _make_workflow(tmp_path=tmp_path)
    wf.git_ops.create_worktree = AsyncMock(
        side_effect=WarmLaneDiskPressure(
            "warm-lane seed disk pressure for branch '1859'; requeue"
        )
    )
    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    outcome = await wf.run()

    assert outcome == WorkflowOutcome.REQUEUED
    mark_blocked.assert_not_awaited()


# ---------------------------------------------------------------------------
# RuntimeError (FAULT) → BLOCKED
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fault_runtime_error_returns_blocked(tmp_path: Path):
    """RuntimeError from create_worktree → existing broad except → _mark_blocked → BLOCKED.

    This should PASS today and continue to pass after step-10 (RuntimeError must
    NOT be caught by the new except WarmLaneRequeue clause — only WarmLaneRequeue
    subclasses go to REQUEUED; genuine faults stay as BLOCKED+L1).
    """
    wf = _make_workflow(tmp_path=tmp_path)
    wf.git_ops.create_worktree = AsyncMock(
        side_effect=RuntimeError(
            'warm-lane acquire fault for branch 1859 (seed/worktree-add failure)'
        )
    )
    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    outcome = await wf.run()

    assert outcome == WorkflowOutcome.BLOCKED
    mark_blocked.assert_awaited_once()
