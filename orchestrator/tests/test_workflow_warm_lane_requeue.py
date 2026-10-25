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
from unittest.mock import AsyncMock

import pytest
from _workflow_helpers import _make_warmlane_workflow as _make_workflow

from orchestrator.git_ops import (
    WarmLaneDiskPressure,
    WarmLanePoolExhausted,
)
from orchestrator.workflow import WorkflowOutcome

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
    outcome = (await wf.run()).outcome

    assert outcome == WorkflowOutcome.REQUEUED


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
    outcome = (await wf.run()).outcome

    assert outcome == WorkflowOutcome.REQUEUED


# ---------------------------------------------------------------------------
# WarmLanePoolHardDown → REQUEUED (task 2061)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pool_hard_down_returns_requeued(tmp_path: Path):
    """WarmLanePoolHardDown raised by create_worktree → run() returns REQUEUED
    with block_reason='warm_lane_pool_hard_down' — the host-scoped warm-lane
    base-absent condition must requeue (fail-open), never escalate a
    per-task blocked+L1.

    Fails today: WarmLanePoolHardDown does not exist yet (ImportError), and
    once it does, the broad except Exception would map it to BLOCKED until
    the dedicated isinstance branch is added in step-6.
    """
    from orchestrator.git_ops import WarmLanePoolHardDown

    wf = _make_workflow(tmp_path=tmp_path)
    wf.git_ops.create_worktree = AsyncMock(
        side_effect=WarmLanePoolHardDown(
            "warm-lane base absent (host-scoped pool hard-down) for branch "
            "'1859'; requeue"
        )
    )
    report = await wf.run()

    assert report.outcome == WorkflowOutcome.REQUEUED
    assert report.reason == 'warm_lane_pool_hard_down'


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
    # run()'s SM-2 exit check reads get_status back, and the REAL _mark_blocked
    # runs here (it is no longer stubbed out), so this reflects the row it
    # persists. _make_workflow's shared default of 'pending' suits the REQUEUED
    # cases the other tests in this module force.
    wf.scheduler.get_status = AsyncMock(return_value='blocked')

    outcome = (await wf.run()).outcome

    assert outcome == WorkflowOutcome.BLOCKED


# ---------------------------------------------------------------------------
# Task 4930: WarmLaneStealFailed / WarmLaneLockTimeout → REQUEUED
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_steal_failed_returns_requeued(tmp_path: Path):
    """WarmLaneStealFailed → REQUEUED with its own reason_prefix and the cap
    charged.

    The REQUEUED half already passes via the shared WarmLaneRequeue base row;
    the reason_prefix and counts_against_requeue_cap assertions are the RED
    ones. Before task 4930 this condition surfaced as a bare RuntimeError and
    landed the task BLOCKED + L1 with agent_invocations=0 — the 2026-08-29
    incident shape.
    """
    from orchestrator.git_ops import WarmLaneStealFailed

    wf = _make_workflow(tmp_path=tmp_path)
    wf.git_ops.create_worktree = AsyncMock(
        side_effect=WarmLaneStealFailed(
            "warm-lane reclaim-on-exhaustion steal failed for branch '1859': "
            'every stolen lane failed to provision, or no further eligible '
            'victim remained (at most 3 attempts per acquire); requeue'
        )
    )
    report = await wf.run()

    assert report.outcome == WorkflowOutcome.REQUEUED, (
        'a pool-pressure steal failure must requeue, never block + L1'
    )
    assert report.reason.startswith('warm_lane_steal_failed (pool pressure)'), (
        f'expected the steal-failure reason_prefix; got {report.reason!r}'
    )
    assert report.counts_against_requeue_cap is True, (
        'chronic pool pressure does not self-clear — the cap is the bounded '
        'loud path that replaces the per-task BLOCKED+L1 being removed'
    )


@pytest.mark.asyncio
async def test_lane_lock_timeout_returns_requeued(tmp_path: Path):
    """WarmLaneLockTimeout → REQUEUED, and the task is NOT charged for it."""
    from orchestrator.git_ops import WarmLaneLockTimeout

    wf = _make_workflow(tmp_path=tmp_path)
    wf.git_ops.create_worktree = AsyncMock(
        side_effect=WarmLaneLockTimeout(
            "warm-lane seed lost the 30s <lane>.lock wait (rc=124) for branch "
            "'1859'; requeue (transient contention)"
        )
    )
    report = await wf.run()

    assert report.outcome == WorkflowOutcome.REQUEUED, (
        'a lost lock race is transient contention — it must requeue'
    )
    assert report.reason.startswith('warm_lane_lock_timeout (transient infra)'), (
        f'expected the lock-timeout reason_prefix; got {report.reason!r}'
    )
    assert report.counts_against_requeue_cap is False, (
        'the requeued task did not cause the lock race and must not be '
        'charged for it'
    )
