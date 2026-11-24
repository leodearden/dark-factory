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
from _landed_rows import record_landing
from _orch_helpers import (
    wire_clear_claim_then_set_status,
    wire_live_status_from_get_task,
    wire_scheduler_liveness_mock,
)
from _startup_harness import (
    MERGE_WORKER_SHUTDOWN_OUTCOME,
    drain_background_tasks,
    harness_with_real_merge_worker,
    journal_merge_request,
    merge_lane_harness_factory,
    recover_requests,
    recovered_request,
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


def _status_writes(scheduler: MagicMock, status: str) -> list:
    return [
        c for c in scheduler.set_task_status.await_args_list
        if c.args[:2] == (_TID, status)
    ]


def _pending_writes(scheduler: MagicMock) -> list:
    return _status_writes(scheduler, 'pending')


def _blocked_writes(scheduler: MagicMock) -> list:
    return _status_writes(scheduler, 'blocked')


@pytest.mark.asyncio
class TestRecoveredRequestOwnsItsTask:
    @pytest.mark.parametrize('mid_run', [False, True], ids=['startup', 'mid_run'])
    async def test_sweep_leaves_a_task_whose_recovered_request_is_in_flight(
        self, harness: Harness, scheduler_double: MagicMock, mid_run: bool,
    ) -> None:
        _hold_recovery_slot(harness)

        await harness._reconcile_stranded_in_progress(mid_run=mid_run)

        assert _pending_writes(scheduler_double) == []

    @pytest.mark.parametrize('mid_run', [False, True], ids=['startup', 'mid_run'])
    async def test_sweep_reverts_once_the_recovered_request_finalizes(
        self, harness: Harness, scheduler_double: MagicMock, mid_run: bool,
    ) -> None:
        future = _hold_recovery_slot(harness)
        future.set_result(None)
        await asyncio.sleep(0)

        await harness._reconcile_stranded_in_progress(mid_run=mid_run)

        assert len(_pending_writes(scheduler_double)) == 1

    @pytest.mark.parametrize('recovered', [True, False], ids=['recovered', 'not_recovered'])
    async def test_sweep_leaves_a_task_whose_journaled_request_startup_recovered(
        self, tmp_path: Path, scheduler_double: MagicMock, recovered: bool,
    ) -> None:
        h = merge_lane_harness_factory(tmp_path)()
        await journal_merge_request(h, _TID)
        if recovered:
            await h._recover_pending_merges()
        h.config.convert_to_blocked_enforce = False
        h.scheduler = scheduler_double

        await Harness._reconcile_stranded_in_progress(h)

        assert len(_pending_writes(scheduler_double)) == (0 if recovered else 1)


@pytest.mark.asyncio
class TestRecoveredRequestOutcomeHasAnOwner:
    async def test_failed_recovered_request_blocks_the_task_with_an_l1(
        self, harness: Harness, scheduler_double: MagicMock, tmp_path: Path,
    ) -> None:
        harness._escalation_queue = EscalationQueue(tmp_path / 'esc')
        request = recovered_request(_TID)
        await recover_requests(harness, request)

        request.result.set_exception(RuntimeError('verify host lost'))
        await drain_background_tasks(harness)

        scheduler_double.set_task_status.assert_any_await(_TID, 'blocked')
        assert _pending_writes(scheduler_double) == []
        [esc] = harness._escalation_queue.get_by_task(_TID, status='pending', level=1)
        assert esc.category == 'merge_error'
        assert esc.agent_role == 'harness-recovered-merge-owner'

    async def test_landed_recovered_request_drives_the_landed_outbox_done_write(
        self, tmp_path: Path, scheduler_double: MagicMock,
    ) -> None:
        harness = await harness_with_real_merge_worker(
            merge_lane_harness_factory(tmp_path), scheduler_double,
        )
        assert harness._merge_worker is not None
        outbox = harness._merge_worker._landed_outbox
        assert outbox is not None
        main_sha = await harness.git_ops.get_main_sha()
        record_landing(outbox, _TID, advanced_sha=main_sha)
        request = recovered_request(_TID)
        await recover_requests(harness, request)

        request.result.set_result(SimpleNamespace(status='done', merge_sha=main_sha, reason=''))
        await drain_background_tasks(harness)

        scheduler_double.mark_done.assert_awaited_once_with(_TID, kind='merged', sha=main_sha)
        assert outbox.lookup(_TID) is None
        assert harness._escalation_queue is not None
        assert harness._escalation_queue.get_by_task(_TID, status='pending', level=1) == []

    async def test_halting_recovered_request_owns_the_merge_halt(
        self, tmp_path: Path, scheduler_double: MagicMock,
    ) -> None:
        make = merge_lane_harness_factory(tmp_path, real_steps=frozenset({'1c0'}))
        harness = await harness_with_real_merge_worker(make, scheduler_double)
        worker = harness._merge_worker
        assert worker is not None
        worker.halt_for_wip('park of the main checkout failed')
        request = recovered_request(_TID)
        await recover_requests(harness, request)

        request.result.set_result(SimpleNamespace(
            status='stash_failed', reason='park failed', dirty_files=['README.md'],
            recovery_branch=None, overlap_files=None, merge_sha=None,
        ))
        await drain_background_tasks(harness)

        assert harness._escalation_queue is not None
        [esc] = harness._escalation_queue.get_by_task(_TID, status='pending', level=1)
        assert esc.category == 'stash_failed'
        assert worker.is_halt_owner(esc.id)
        scheduler_double.set_task_status.assert_any_await(_TID, 'blocked')

        (harness.git_ops.project_root / 'README.md').write_text('# unparked WIP\n')
        restarted = await harness_with_real_merge_worker(make, scheduler_double)
        assert await restarted._rehydrate_merge_halt() == esc.id
        assert restarted._merge_worker is not None
        assert restarted._merge_worker.is_wip_halted
        assert restarted._merge_worker.is_halt_owner(esc.id)

    async def test_shutdown_outcome_leaves_the_task_for_the_next_start(
        self, harness: Harness, scheduler_double: MagicMock, tmp_path: Path,
    ) -> None:
        harness._escalation_queue = EscalationQueue(tmp_path / 'esc')
        first = recovered_request(_TID)
        await recover_requests(harness, first)

        first.result.set_result(MERGE_WORKER_SHUTDOWN_OUTCOME)
        await drain_background_tasks(harness)

        assert _blocked_writes(scheduler_double) == []
        assert harness._escalation_queue.get_by_task(_TID, status='pending', level=1) == []

        second = recovered_request(_TID)
        await recover_requests(harness, second)
        second.result.set_exception(RuntimeError('verify host lost'))
        await drain_background_tasks(harness)

        assert len(_blocked_writes(scheduler_double)) == 1
        [esc] = harness._escalation_queue.get_by_task(_TID, status='pending', level=1)
        assert esc.category == 'merge_error'
        assert 'verify host lost' in esc.detail
