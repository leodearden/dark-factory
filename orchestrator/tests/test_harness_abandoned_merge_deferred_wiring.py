"""The abandoned merge-deferred sweep is the last step of the harness's
stranded-reconcile pass (task 4737).

The harness is built like ``test_reconcile_stranded.py``'s ``harness`` fixture.
The parked-member scheduler fake from ``test_merge_deferred_recovery.py`` is
handed in as the constructed scheduler, and the merge lane is a ``spec_set``
double of ``SpeculativeMergeWorker`` exposing only the lane's real interface.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import ANY, MagicMock, call, patch

import pytest
from test_merge_deferred_recovery import (
    ParkedScheduler,
    _abandoned_records,
    single_request,
    submit_single,
)
from test_reconcile_stranded import _bind_queue

from orchestrator.harness import Harness
from orchestrator.merge_lane import InFlightMergeRegistry, MemberRedriveOutcome, MergeOutcome
from orchestrator.merge_lane.worker import SpeculativeMergeWorker


@pytest.fixture
def sched() -> ParkedScheduler:
    return ParkedScheduler()


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config, sched: ParkedScheduler) -> Harness:
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler', return_value=sched), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)
    h.config.claimant_liveness_ttl_secs = 300.0
    h.git_ops.worktree_base = (tmp_path / '.worktrees').resolve()
    return h


def _parked_lane() -> MagicMock:
    lane = MagicMock(spec_set=SpeculativeMergeWorker)
    lane.train_member_ids_in_custody.return_value = frozenset()
    lane.redrive_abandoned_members.return_value = {'x': MemberRedriveOutcome.NOT_ON_MAIN}
    return lane


def _bind_lane(harness: Harness, lane: SpeculativeMergeWorker | None) -> None:
    """The suite's one write of the harness's private merge-worker slot."""
    harness._merge_worker = lane


def _shared_registry(harness: Harness) -> InFlightMergeRegistry:
    """The suite's one read of the harness's shared in-flight merge registry."""
    return harness._merge_inflight_registry


async def _recover(harness: Harness) -> None:
    """The suite's one call of the harness's private recovery step."""
    await harness._recover_abandoned_merge_deferred()


async def _stranded_reconcile_pass(harness: Harness) -> None:
    """The suite's one call of the stranded-reconcile BackgroundService pass."""
    await harness._run_stranded_reconcile_pass()


@pytest.mark.asyncio
class TestRecoverAbandonedMergeDeferred:
    async def test_the_second_consecutive_sighting_redrives_and_escalates(
        self, harness: Harness, sched: ParkedScheduler, tmp_path: Path,
    ) -> None:
        await sched.park('x')
        queue = _bind_queue(harness, tmp_path / 'esc')
        lane = _parked_lane()
        _bind_lane(harness, lane)

        await _recover(harness)
        lane.redrive_abandoned_members.assert_not_awaited()

        await _recover(harness)

        assert lane.redrive_abandoned_members.await_args_list == [
            call(ANY, ['x'], strip_unlanded_base=True),
        ]
        assert _abandoned_records(queue, 'x') == [(0, 'info')]

    async def test_no_merge_worker_is_a_noop(
        self, harness: Harness, sched: ParkedScheduler, tmp_path: Path,
    ) -> None:
        await sched.park('x')
        queue = _bind_queue(harness, tmp_path / 'esc')
        _bind_lane(harness, None)

        await _recover(harness)
        await _recover(harness)

        assert sched.statuses == {'x': ['merge-deferred']}
        assert queue.get_pending() == []

    async def test_a_sweep_fault_is_logged_and_never_raised(
        self, harness: Harness, sched: ParkedScheduler, tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        await sched.park('x')
        _bind_queue(harness, tmp_path / 'esc')
        lane = _parked_lane()
        lane.train_member_ids_in_custody.side_effect = RuntimeError('custody unreadable')
        _bind_lane(harness, lane)

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            await _recover(harness)

        assert any(
            record.levelno == logging.WARNING and record.exc_info is not None
            for record in caplog.records
        )


@pytest.mark.asyncio
class TestStrandedReconcilePassWiring:
    async def test_two_passes_redrive_an_abandoned_member(
        self, harness: Harness, sched: ParkedScheduler, tmp_path: Path,
    ) -> None:
        await sched.park('x')
        queue = _bind_queue(harness, tmp_path / 'esc')
        lane = _parked_lane()
        _bind_lane(harness, lane)

        await _stranded_reconcile_pass(harness)
        lane.redrive_abandoned_members.assert_not_awaited()

        await _stranded_reconcile_pass(harness)

        assert lane.redrive_abandoned_members.await_args_list == [
            call(ANY, ['x'], strip_unlanded_base=True),
        ]
        assert _abandoned_records(queue, 'x') == [(0, 'info')]
        assert sched.statuses == {'x': ['merge-deferred']}

    async def test_a_live_single_in_the_shared_registry_holds_the_member(
        self, harness: Harness, sched: ParkedScheduler, tmp_path: Path,
    ) -> None:
        await sched.park('x')
        queue = _bind_queue(harness, tmp_path / 'esc')
        lane = _parked_lane()
        _bind_lane(harness, lane)
        req = single_request(harness.config, 'x')
        await submit_single(_shared_registry(harness), req)

        await _stranded_reconcile_pass(harness)
        await _stranded_reconcile_pass(harness)

        lane.redrive_abandoned_members.assert_not_awaited()
        assert sched.statuses == {'x': ['merge-deferred']}
        assert queue.get_pending() == []

        req.result.set_result(MergeOutcome('blocked', reason='main is red'))
        await asyncio.sleep(0)
        await _stranded_reconcile_pass(harness)
        await _stranded_reconcile_pass(harness)

        assert lane.redrive_abandoned_members.await_args_list == [
            call(ANY, ['x'], strip_unlanded_base=True),
        ]
