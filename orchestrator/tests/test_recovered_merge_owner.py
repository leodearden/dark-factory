"""RecoveredMergeOwner settles the outcome of a journal-recovered merge request (task 7277).

Public seams only (``adopt``, ``settle``): this file imports lane types, so the
merge-lane ratchet counts every private read and lane patch target in it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from escalation.queue import EscalationQueue

from orchestrator.event_store import EventType
from orchestrator.merge_halt_escalation import MERGE_HALT_CATEGORY_BY_STATUS
from orchestrator.merge_lane import MERGE_WORKER_SHUTDOWN_REASON, MergeOutcome
from orchestrator.merge_lane.outcome_facts import MergeBlockCategory
from orchestrator.recovered_merge_owner import RecoveredMergeOwner, Settled

_TID = '40'
_REQUEST_ID = 'mr-recovered'
_VERIFY_RED = MergeOutcome(
    status='blocked',
    reason='post-merge verify red: test_x',
    block_category=MergeBlockCategory.POST_MERGE_VERIFY,
)
_SHUTDOWN = MergeOutcome(status='blocked', reason=MERGE_WORKER_SHUTDOWN_REASON)


class FakeTaskStore:
    def __init__(self, status: str | None = 'in-progress', *, held: bool = False) -> None:
        self.status = status
        self.held = held
        self.writes: list[tuple[str, str]] = []

    def is_actively_held(self, tid: str) -> bool:
        return self.held

    async def get_status(self, tid: str) -> str | None:
        return self.status

    async def clear_claim_then_set_status(self, tid: str, status: str) -> None:
        self.writes.append((tid, status))
        self.status = status


class FakeMergeHalt:
    def __init__(
        self, *, is_wip_halted: bool = True, halt_owner_esc_id: str | None = None,
    ) -> None:
        self.is_wip_halted = is_wip_halted
        self.halt_owner_esc_id = halt_owner_esc_id
        self.owners: list[str] = []
        self.unhalt_reasons: list[str | None] = []

    def set_halt_owner(self, esc_id: str) -> None:
        assert self.is_wip_halted
        assert self.halt_owner_esc_id is None
        self.halt_owner_esc_id = esc_id
        self.owners.append(esc_id)

    def is_halt_owner(self, esc_id: str) -> bool:
        return self.halt_owner_esc_id == esc_id

    def unhalt_wip(self, reason: str | None = None) -> None:
        self.is_wip_halted = False
        self.unhalt_reasons.append(reason)


@dataclass(frozen=True)
class Request:
    result: asyncio.Future
    task_id: str = _TID
    request_id: str = _REQUEST_ID


@dataclass
class ScheduleCollector:
    scheduled: list[tuple[Coroutine[Any, Any, Settled], str]] = field(default_factory=list)

    def __call__(self, coro: Coroutine[Any, Any, Settled], *, label: str) -> None:
        self.scheduled.append((coro, label))


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'esc')


@pytest.fixture
def reconcile_landed() -> AsyncMock:
    return AsyncMock(return_value=True)


def _owner(
    store: FakeTaskStore, queue: EscalationQueue, reconcile_landed: AsyncMock,
    event_store: MagicMock | None = None, *, merge_halt: FakeMergeHalt | None = None,
) -> RecoveredMergeOwner:
    return RecoveredMergeOwner(
        tasks=store, escalations=queue, reconcile_landed=reconcile_landed,
        merge_halt=merge_halt, event_store=event_store,
    )


def _request() -> Request:
    return Request(result=asyncio.get_running_loop().create_future())


def _open_l1s(queue: EscalationQueue) -> list:
    return queue.get_by_task(_TID, status='pending', level=1)


@pytest.mark.asyncio
class TestDurableFailureBlocksAndEscalates:
    async def test_in_progress_task_is_blocked_with_one_l1(
        self, queue: EscalationQueue, reconcile_landed: AsyncMock,
    ) -> None:
        store = FakeTaskStore('in-progress')

        settled = await _owner(store, queue, reconcile_landed).settle(_request(), _VERIFY_RED)

        assert settled == Settled.BLOCKED
        assert store.writes == [(_TID, 'blocked')]
        [esc] = _open_l1s(queue)
        assert esc.category == 'post_merge_verify'
        assert esc.agent_role == 'harness-recovered-merge-owner'
        assert esc.severity == 'blocking'
        assert _REQUEST_ID in esc.summary
        assert 'post-merge verify red: test_x' in esc.summary + esc.detail
        reconcile_landed.assert_not_awaited()

    async def test_task_a_sweep_already_re_pended_is_blocked_with_one_l1(
        self, queue: EscalationQueue, reconcile_landed: AsyncMock,
    ) -> None:
        store = FakeTaskStore('pending')

        settled = await _owner(store, queue, reconcile_landed).settle(_request(), _VERIFY_RED)

        assert settled == Settled.BLOCKED
        assert store.writes == [(_TID, 'blocked')]
        assert len(_open_l1s(queue)) == 1

    async def test_a_second_settle_files_no_second_l1(
        self, queue: EscalationQueue, reconcile_landed: AsyncMock,
    ) -> None:
        store = FakeTaskStore('in-progress')
        owner = _owner(store, queue, reconcile_landed)

        await owner.settle(_request(), _VERIFY_RED)
        store.status = 'in-progress'
        await owner.settle(_request(), _VERIFY_RED)

        assert len(_open_l1s(queue)) == 1

    async def test_filing_emits_escalation_created(
        self, queue: EscalationQueue, reconcile_landed: AsyncMock,
    ) -> None:
        event_store = MagicMock()

        await _owner(FakeTaskStore(), queue, reconcile_landed, event_store).settle(
            _request(), _VERIFY_RED,
        )

        [esc] = _open_l1s(queue)
        event_store.emit.assert_called_once()
        assert event_store.emit.call_args.args == (EventType.escalation_created,)
        assert event_store.emit.call_args.kwargs['task_id'] == _TID
        assert event_store.emit.call_args.kwargs['data']['escalation_id'] == esc.id

    async def test_a_task_a_workflow_holds_is_left_alone(
        self, queue: EscalationQueue, reconcile_landed: AsyncMock,
    ) -> None:
        store = FakeTaskStore('in-progress', held=True)

        settled = await _owner(store, queue, reconcile_landed).settle(_request(), _VERIFY_RED)

        assert settled == Settled.WORKFLOW_HELD
        assert store.writes == []
        assert _open_l1s(queue) == []

    @pytest.mark.parametrize('status', ['done', 'blocked', 'cancelled'])
    async def test_a_task_another_owner_moved_is_left_alone(
        self, queue: EscalationQueue, reconcile_landed: AsyncMock, status: str,
    ) -> None:
        store = FakeTaskStore(status)

        settled = await _owner(store, queue, reconcile_landed).settle(_request(), _VERIFY_RED)

        assert settled == Settled.STATUS_MOVED
        assert store.writes == []
        assert _open_l1s(queue) == []


@pytest.mark.asyncio
class TestLandingDrivesTheLandedOutboxDoneWrite:
    @pytest.mark.parametrize('outcome', [
        MergeOutcome(status='done', merge_sha='a' * 40),
        MergeOutcome(status='already_merged', merge_sha='a' * 40),
    ], ids=lambda outcome: outcome.status)
    async def test_landing_calls_reconcile_landed(
        self, queue: EscalationQueue, reconcile_landed: AsyncMock, outcome: MergeOutcome,
    ) -> None:
        store = FakeTaskStore('in-progress')

        settled = await _owner(store, queue, reconcile_landed).settle(_request(), outcome)

        assert settled == Settled.LANDED
        reconcile_landed.assert_awaited_once_with(_TID)
        assert store.writes == []
        assert _open_l1s(queue) == []

    async def test_landing_with_no_outbox_row_is_reported_unrecorded(
        self, queue: EscalationQueue,
    ) -> None:
        store = FakeTaskStore('in-progress')
        no_row = AsyncMock(return_value=False)

        settled = await _owner(store, queue, no_row).settle(
            _request(), MergeOutcome(status='already_merged'),
        )

        assert settled == Settled.LANDING_UNRECORDED
        assert store.writes == []


@pytest.mark.asyncio
class TestTransientOutcomeIsLeftToTheSweep:
    @pytest.mark.parametrize('outcome', [
        MergeOutcome(status='superseded'),
    ], ids=lambda outcome: outcome.status)
    async def test_nothing_happens(
        self, queue: EscalationQueue, reconcile_landed: AsyncMock, outcome: MergeOutcome,
    ) -> None:
        store = FakeTaskStore('in-progress')

        settled = await _owner(store, queue, reconcile_landed).settle(_request(), outcome)

        assert settled == Settled.TRANSIENT
        reconcile_landed.assert_not_awaited()
        assert store.writes == []
        assert _open_l1s(queue) == []


_HALT_OUTCOMES = {
    'wip_halted': MergeOutcome(status='wip_halted', overlap_files=['x.py']),
    'done_wip_recovery': MergeOutcome(
        status='done_wip_recovery', recovery_branch='wip/r', merge_sha='a' * 40,
    ),
    'wip_recovery_no_advance': MergeOutcome(
        status='wip_recovery_no_advance', recovery_branch='wip/r',
    ),
    'unmerged_state': MergeOutcome(status='unmerged_state'),
    'stash_failed': MergeOutcome(status='stash_failed', dirty_files=['README.md']),
}
_STASH_FAILED = _HALT_OUTCOMES['stash_failed']


def _open_l1s_in(queue: EscalationQueue, category: str) -> list:
    return [esc for esc in _open_l1s(queue) if esc.category == category]


@pytest.mark.asyncio
class TestHaltOutcomeOwnsTheMergeHalt:
    @pytest.mark.parametrize('status', sorted(_HALT_OUTCOMES))
    async def test_halt_outcome_files_the_halt_owning_l1(
        self, queue: EscalationQueue, reconcile_landed: AsyncMock, status: str,
    ) -> None:
        store = FakeTaskStore('in-progress')
        halt = FakeMergeHalt()

        settled = await _owner(store, queue, reconcile_landed, merge_halt=halt).settle(
            _request(), _HALT_OUTCOMES[status],
        )

        assert settled == Settled.HALTED
        [esc] = _open_l1s(queue)
        assert esc.category == MERGE_HALT_CATEGORY_BY_STATUS[status]
        assert esc.suggested_action == 'manual_intervention'
        assert _REQUEST_ID in esc.detail
        assert halt.owners == [esc.id]
        if status == 'done_wip_recovery':
            reconcile_landed.assert_awaited_once_with(_TID)
            assert (_TID, 'blocked') not in store.writes
        else:
            assert store.writes == [(_TID, 'blocked')]
            reconcile_landed.assert_not_awaited()

    async def test_an_already_owned_halt_gets_a_non_owning_record(
        self, queue: EscalationQueue, reconcile_landed: AsyncMock,
    ) -> None:
        store = FakeTaskStore('in-progress')
        halt = FakeMergeHalt(halt_owner_esc_id='esc-other')

        await _owner(store, queue, reconcile_landed, merge_halt=halt).settle(
            _request(), _STASH_FAILED,
        )

        assert _open_l1s_in(queue, 'stash_failed') == []
        [esc] = _open_l1s(queue)
        assert esc.category == 'task_failure'
        assert 'esc-other' in esc.detail
        assert halt.owners == []
        assert store.writes == [(_TID, 'blocked')]

    @pytest.mark.parametrize('halt', [None, FakeMergeHalt(is_wip_halted=False)],
                             ids=['no_merge_worker', 'force_unhalted'])
    async def test_no_ownable_halt_files_no_halt_category_record(
        self, queue: EscalationQueue, reconcile_landed: AsyncMock,
        halt: FakeMergeHalt | None,
    ) -> None:
        store = FakeTaskStore('in-progress')

        await _owner(store, queue, reconcile_landed, merge_halt=halt).settle(
            _request(), _STASH_FAILED,
        )

        assert _open_l1s_in(queue, 'stash_failed') == []
        [esc] = _open_l1s(queue)
        assert esc.category == 'task_failure'
        assert store.writes == [(_TID, 'blocked')]

    async def test_a_moved_task_status_still_files_the_halt_owner(
        self, queue: EscalationQueue, reconcile_landed: AsyncMock,
    ) -> None:
        store = FakeTaskStore('blocked')
        halt = FakeMergeHalt()

        await _owner(store, queue, reconcile_landed, merge_halt=halt).settle(
            _request(), _STASH_FAILED,
        )

        [esc] = _open_l1s_in(queue, 'stash_failed')
        assert halt.owners == [esc.id]
        assert store.writes == []

    async def test_a_task_a_workflow_holds_is_left_to_its_handler(
        self, queue: EscalationQueue, reconcile_landed: AsyncMock,
    ) -> None:
        store = FakeTaskStore('in-progress', held=True)
        halt = FakeMergeHalt()

        settled = await _owner(store, queue, reconcile_landed, merge_halt=halt).settle(
            _request(), _STASH_FAILED,
        )

        assert settled == Settled.WORKFLOW_HELD
        assert _open_l1s(queue) == []
        assert halt.owners == []

    async def test_a_second_settle_files_no_second_halt_record(
        self, queue: EscalationQueue, reconcile_landed: AsyncMock,
    ) -> None:
        store = FakeTaskStore('in-progress')
        halt = FakeMergeHalt()
        owner = _owner(store, queue, reconcile_landed, merge_halt=halt)

        await owner.settle(_request(), _STASH_FAILED)
        await owner.settle(_request(), _STASH_FAILED)

        assert len(_open_l1s_in(queue, 'stash_failed')) == 1
        assert len(halt.owners) == 1


@pytest.mark.asyncio
class TestWorkerShutdownOutcomeIsLeftForTheNextStart:
    async def test_shutdown_outcome_writes_and_files_nothing(
        self, queue: EscalationQueue, reconcile_landed: AsyncMock,
    ) -> None:
        store = FakeTaskStore('in-progress')
        halt = FakeMergeHalt()

        settled = await _owner(store, queue, reconcile_landed, merge_halt=halt).settle(
            _request(), _SHUTDOWN,
        )

        assert settled == Settled.WORKER_SHUTDOWN
        assert store.writes == []
        assert _open_l1s(queue) == []
        reconcile_landed.assert_not_awaited()
        assert halt.owners == []

    async def test_the_restored_requests_real_failure_still_gets_its_owner(
        self, queue: EscalationQueue, reconcile_landed: AsyncMock,
    ) -> None:
        store = FakeTaskStore('in-progress')
        owner = _owner(store, queue, reconcile_landed, merge_halt=FakeMergeHalt())

        await owner.settle(_request(), _SHUTDOWN)
        settled = await owner.settle(_request(), _VERIFY_RED)

        assert settled == Settled.BLOCKED
        assert store.writes == [(_TID, 'blocked')]
        [esc] = _open_l1s(queue)
        assert 'post-merge verify red' in esc.summary

    async def test_the_restored_requests_landing_leaves_no_stale_l1(
        self, queue: EscalationQueue, reconcile_landed: AsyncMock,
    ) -> None:
        owner = _owner(
            FakeTaskStore('in-progress'), queue, reconcile_landed, merge_halt=FakeMergeHalt(),
        )

        await owner.settle(_request(), _SHUTDOWN)
        settled = await owner.settle(_request(), MergeOutcome(status='done', merge_sha='a' * 40))

        assert settled == Settled.LANDED
        reconcile_landed.assert_awaited_once_with(_TID)
        assert _open_l1s(queue) == []

    async def test_adopt_schedules_a_settle_that_leaves_the_shutdown_alone(
        self, queue: EscalationQueue, reconcile_landed: AsyncMock,
    ) -> None:
        store = FakeTaskStore('in-progress')
        collector = ScheduleCollector()
        request = _request()
        _owner(store, queue, reconcile_landed, merge_halt=FakeMergeHalt()).adopt(
            [request], schedule=collector,
        )

        request.result.set_result(_SHUTDOWN)
        await asyncio.sleep(0)

        [(coro, _label)] = collector.scheduled
        assert await coro == Settled.WORKER_SHUTDOWN
        assert store.writes == []

    async def test_only_the_exact_shutdown_reason_is_left_alone(
        self, queue: EscalationQueue, reconcile_landed: AsyncMock,
    ) -> None:
        # The journal removes this record too, so nothing retries it at the next start.
        aborted = MergeOutcome(
            status='blocked', reason=f'{MERGE_WORKER_SHUTDOWN_REASON}: verify aborted',
        )

        settled = await _owner(
            FakeTaskStore('in-progress'), queue, reconcile_landed, merge_halt=FakeMergeHalt(),
        ).settle(_request(), aborted)

        assert settled == Settled.BLOCKED
        assert len(_open_l1s(queue)) == 1


@pytest.mark.asyncio
class TestAdoptSchedulesASettleWhenTheFutureFinalizes:
    async def test_resolved_failure_blocks_and_escalates(
        self, queue: EscalationQueue, reconcile_landed: AsyncMock,
    ) -> None:
        store = FakeTaskStore('in-progress')
        collector = ScheduleCollector()
        request = _request()
        _owner(store, queue, reconcile_landed).adopt([request], schedule=collector)

        request.result.set_result(_VERIFY_RED)
        await asyncio.sleep(0)

        [(coro, label)] = collector.scheduled
        assert _TID in label
        assert await coro == Settled.BLOCKED
        assert store.writes == [(_TID, 'blocked')]
        assert len(_open_l1s(queue)) == 1

    async def test_raised_future_is_a_merge_error_failure(
        self, queue: EscalationQueue, reconcile_landed: AsyncMock,
    ) -> None:
        store = FakeTaskStore('in-progress')
        collector = ScheduleCollector()
        request = _request()
        _owner(store, queue, reconcile_landed).adopt([request], schedule=collector)

        request.result.set_exception(RuntimeError('verify host lost'))
        await asyncio.sleep(0)

        [(coro, _label)] = collector.scheduled
        assert await coro == Settled.BLOCKED
        assert store.writes == [(_TID, 'blocked')]
        [esc] = _open_l1s(queue)
        assert esc.category == 'merge_error'
        assert 'verify host lost' in esc.detail

    async def test_cancelled_future_schedules_nothing(
        self, queue: EscalationQueue, reconcile_landed: AsyncMock,
    ) -> None:
        collector = ScheduleCollector()
        request = _request()
        _owner(FakeTaskStore(), queue, reconcile_landed).adopt([request], schedule=collector)

        request.result.cancel()
        await asyncio.sleep(0)

        assert collector.scheduled == []
