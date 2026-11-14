"""A merge request recovered from the journal at startup owns its task (task 7277).

Reify 8666: the startup reconcile re-pended a task whose merge request step 1c0a
had just recovered into the lane, because the request's submitting process was
dead and so its DB claimant was stale. These tests drive the real harness sweep.

This file imports no ``orchestrator.merge_lane`` or ``orchestrator.git_ops``
module: it reads harness privates, which the merge-lane ratchet
(``scripts/merge_lane_metrics.py``) would count against any lane-importing file.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import (
    wire_clear_claim_then_set_status,
    wire_live_status_from_get_task,
    wire_scheduler_liveness_mock,
)

from orchestrator.harness import Harness

_TID = '40'


def _stale_claimed_row() -> dict:
    return {
        'status': 'in-progress',
        'metadata': {},
        'claimant_run_id': 'run-1/session-1/pid=1',
        'heartbeat_at': (datetime.now(UTC) - timedelta(hours=3)).isoformat(),
    }


@pytest.fixture
def scheduler_double() -> MagicMock:
    scheduler = MagicMock()
    wire_scheduler_liveness_mock(scheduler)
    wire_clear_claim_then_set_status(scheduler)
    wire_live_status_from_get_task(scheduler)
    scheduler.get_statuses = AsyncMock(return_value=({_TID: 'in-progress'}, None))
    scheduler.get_task = AsyncMock(return_value=_stale_claimed_row())
    scheduler.set_task_status = AsyncMock()
    scheduler.mark_done = AsyncMock()
    return scheduler


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config, scheduler_double: MagicMock) -> Harness:
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)
    h.config.convert_to_blocked_enforce = False
    h.scheduler = scheduler_double
    h.git_ops.worktree_base = (tmp_path / '.worktrees').resolve()
    h.git_ops.cleanup_worktree = AsyncMock()
    h.git_ops.is_ancestor = AsyncMock(return_value=False)
    h.git_ops.resolve_branch_sha = AsyncMock(return_value='deadbeef' + 'a' * 32)
    h.git_ops.find_merge_marker = AsyncMock(return_value=None)
    h.git_ops.warm_lane_ref_is_degenerate = AsyncMock(return_value=False)
    return h


def _hold_recovery_slot(h: Harness) -> asyncio.Future:
    future = asyncio.get_running_loop().create_future()
    h._merge_inflight_registry.acquire(
        _TID, _TID, future, request_id='mr-recovered', source='recovery',
    )
    return future


def _pending_writes(scheduler: MagicMock) -> list:
    return [
        c for c in scheduler.set_task_status.await_args_list
        if c.args[:2] == (_TID, 'pending')
    ]


@pytest.mark.asyncio
class TestRecoveredRequestOwnsItsTask:
    @pytest.mark.parametrize('mid_run', [False, True], ids=['startup', 'mid_run'])
    async def test_sweep_leaves_a_task_whose_recovered_request_is_in_flight(
        self, harness: Harness, scheduler_double: MagicMock, mid_run: bool,
    ) -> None:
        _hold_recovery_slot(harness)

        await harness._reconcile_stranded_in_progress(mid_run=mid_run)

        assert _pending_writes(scheduler_double) == []

    async def test_sweep_reverts_once_the_recovered_request_finalizes(
        self, harness: Harness, scheduler_double: MagicMock,
    ) -> None:
        future = _hold_recovery_slot(harness)
        future.set_result(None)
        await asyncio.sleep(0)

        await harness._reconcile_stranded_in_progress()

        assert len(_pending_writes(scheduler_double)) == 1
