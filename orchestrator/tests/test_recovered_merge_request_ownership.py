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
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import (
    wire_clear_claim_then_set_status,
    wire_live_status_from_get_task,
    wire_scheduler_liveness_mock,
)
from escalation.queue import EscalationQueue

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


def _recovered_request() -> SimpleNamespace:
    return SimpleNamespace(
        task_id=_TID, request_id='mr-recovered', branch=SimpleNamespace(bare_id=_TID),
        result=asyncio.get_running_loop().create_future(),
    )


async def _recover(h: Harness, request: SimpleNamespace) -> None:
    report = {
        'recovered': 1, 'dropped': 0, 'coalesced': 0, 'requests': [request],
        'journal_corrupt': False,
    }
    with patch('orchestrator.harness.recover_pending_merges', AsyncMock(return_value=report)):
        await h._recover_pending_merges()


async def _drain(h: Harness) -> None:
    await asyncio.sleep(0)
    await asyncio.gather(*h._background_tasks)


class FakeMergeWorker:
    def __init__(self, *, halted: bool) -> None:
        self.is_wip_halted = halted
        self.halt_owner_esc_id: str | None = None

    def halt_for_wip(self, reason: str) -> None:
        self.is_wip_halted = True
        self.halt_owner_esc_id = None

    def set_halt_owner(self, esc_id: str) -> None:
        assert self.is_wip_halted
        assert self.halt_owner_esc_id is None
        self.halt_owner_esc_id = esc_id

    def is_halt_owner(self, esc_id: str) -> bool:
        return self.halt_owner_esc_id == esc_id

    def unhalt_wip(self, reason: str | None = None) -> None:
        self.is_wip_halted = False


@pytest.mark.asyncio
class TestRecoveredRequestOutcomeHasAnOwner:
    async def test_failed_recovered_request_blocks_the_task_with_an_l1(
        self, harness: Harness, scheduler_double: MagicMock, tmp_path: Path,
    ) -> None:
        harness._escalation_queue = EscalationQueue(tmp_path / 'esc')
        request = _recovered_request()
        await _recover(harness, request)

        request.result.set_exception(RuntimeError('verify host lost'))
        await _drain(harness)

        scheduler_double.set_task_status.assert_any_await(_TID, 'blocked')
        assert _pending_writes(scheduler_double) == []
        [esc] = harness._escalation_queue.get_by_task(_TID, status='pending', level=1)
        assert esc.category == 'merge_error'
        assert esc.agent_role == 'harness-recovered-merge-owner'

    async def test_landed_recovered_request_drives_the_landed_outbox_done_write(
        self, harness: Harness, scheduler_double: MagicMock, tmp_path: Path,
    ) -> None:
        harness._escalation_queue = EscalationQueue(tmp_path / 'esc')
        landed_gate = AsyncMock(return_value=True)
        harness._landed_dispatch_gate = landed_gate
        request = _recovered_request()
        await _recover(harness, request)

        request.result.set_result(SimpleNamespace(status='done', merge_sha='a' * 40, reason=''))
        await _drain(harness)

        landed_gate.assert_awaited_once_with(_TID)
        assert harness._escalation_queue.get_by_task(_TID, status='pending', level=1) == []

    async def test_halting_recovered_request_owns_the_merge_halt(
        self, harness: Harness, scheduler_double: MagicMock, tmp_path: Path,
    ) -> None:
        harness._escalation_queue = EscalationQueue(tmp_path / 'esc')
        worker = FakeMergeWorker(halted=True)
        harness._merge_worker = worker
        request = _recovered_request()
        await _recover(harness, request)

        request.result.set_result(SimpleNamespace(
            status='stash_failed', reason='park failed', dirty_files=['README.md'],
            recovery_branch=None, overlap_files=None, merge_sha=None,
        ))
        await _drain(harness)

        [esc] = harness._escalation_queue.get_by_task(_TID, status='pending', level=1)
        assert esc.category == 'stash_failed'
        assert worker.is_halt_owner(esc.id)
        scheduler_double.set_task_status.assert_any_await(_TID, 'blocked')

        restarted = FakeMergeWorker(halted=False)
        harness._merge_worker = restarted
        assert harness._rehydrate_merge_halt() == esc.id
        assert restarted.is_wip_halted
        assert restarted.is_halt_owner(esc.id)
