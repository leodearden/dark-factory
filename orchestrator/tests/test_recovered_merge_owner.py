"""RecoveredMergeOwner settles the outcome of a journal-recovered merge request (task 7277).

Public seams only (``adopt``, ``settle``): this file imports lane types, so the
merge-lane ratchet counts every private read and lane patch target in it.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.event_store import EventType
from orchestrator.merge_halt_escalation import (
    MERGE_HALT_CATEGORY_BY_STATUS,
    submit_halt_owning_escalation,
)
from orchestrator.merge_lane import (
    MERGE_WORKER_SHUTDOWN_REASON,
    InFlightMergeRegistry,
    MergeBlockCategory,
    MergeOutcome,
    WaiterRecord,
)
from orchestrator.recovered_merge_owner import RecoveredMergeOwner, Settled

_TID = '40'
_REQUEST_ID = 'mr-recovered'
_VERIFY_RED = MergeOutcome(
    status='blocked',
    reason='post-merge verify red: test_x',
    block_category=MergeBlockCategory.POST_MERGE_VERIFY,
)
_SHUTDOWN = MergeOutcome(status='blocked', reason=MERGE_WORKER_SHUTDOWN_REASON)


class StoreDown(Exception):
    pass


class FakeTaskStore:
    def __init__(
        self, status: str | None = 'in-progress', *, held: bool = False,
        reads_fail: bool = False, writes_fail: bool = False,
    ) -> None:
        self.status = status
        self.held = held
        self.reads_fail = reads_fail
        self.writes_fail = writes_fail
        self.writes: list[tuple[str, str]] = []

    def is_actively_held(self, tid: str) -> bool:
        return self.held

    async def get_status(self, tid: str) -> str | None:
        if self.reads_fail:
            raise StoreDown('status read failed')
        return self.status

    async def clear_claim_then_set_status(self, tid: str, status: str) -> None:
        if self.writes_fail:
            raise StoreDown('status write failed')
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
def reconcile_landing() -> AsyncMock:
    return AsyncMock(return_value='marked_done')


def _owner(
    store: FakeTaskStore, queue: EscalationQueue, reconcile_landing: AsyncMock,
    event_store: MagicMock | None = None, *, merge_halt: FakeMergeHalt | None = None,
) -> RecoveredMergeOwner:
    return RecoveredMergeOwner(
        tasks=store, escalations=queue, reconcile_landing=reconcile_landing,
        merge_halt=merge_halt, event_store=event_store,
    )


def _request() -> Request:
    return Request(result=asyncio.get_running_loop().create_future())


def _open_l1s(queue: EscalationQueue) -> list:
    return queue.get_by_task(_TID, status='pending', level=1)


@pytest.mark.asyncio
class TestDurableFailureBlocksAndEscalates:
    async def test_in_progress_task_is_blocked_with_one_l1(
        self, queue: EscalationQueue, reconcile_landing: AsyncMock,
    ) -> None:
        store = FakeTaskStore('in-progress')

        settled = await _owner(store, queue, reconcile_landing).settle(_request(), _VERIFY_RED)

        assert settled == Settled.BLOCKED
        assert store.writes == [(_TID, 'blocked')]
        [esc] = _open_l1s(queue)
        assert esc.category == 'post_merge_verify'
        assert esc.agent_role == 'harness-recovered-merge-owner'
        assert esc.severity == 'blocking'
        assert _REQUEST_ID in esc.summary
        assert 'post-merge verify red: test_x' in esc.summary + esc.detail
        reconcile_landing.assert_not_awaited()

    async def test_task_a_sweep_already_re_pended_is_blocked_with_one_l1(
        self, queue: EscalationQueue, reconcile_landing: AsyncMock,
    ) -> None:
        store = FakeTaskStore('pending')

        settled = await _owner(store, queue, reconcile_landing).settle(_request(), _VERIFY_RED)

        assert settled == Settled.BLOCKED
        assert store.writes == [(_TID, 'blocked')]
        assert len(_open_l1s(queue)) == 1

    async def test_a_second_settle_files_no_second_l1(
        self, queue: EscalationQueue, reconcile_landing: AsyncMock,
    ) -> None:
        store = FakeTaskStore('in-progress')
        owner = _owner(store, queue, reconcile_landing)

        await owner.settle(_request(), _VERIFY_RED)
        store.status = 'in-progress'
        await owner.settle(_request(), _VERIFY_RED)

        assert len(_open_l1s(queue)) == 1

    async def test_filing_emits_escalation_created(
        self, queue: EscalationQueue, reconcile_landing: AsyncMock,
    ) -> None:
        event_store = MagicMock()

        await _owner(FakeTaskStore(), queue, reconcile_landing, event_store).settle(
            _request(), _VERIFY_RED,
        )

        [esc] = _open_l1s(queue)
        event_store.emit.assert_called_once()
        assert event_store.emit.call_args.args == (EventType.escalation_created,)
        assert event_store.emit.call_args.kwargs['task_id'] == _TID
        assert event_store.emit.call_args.kwargs['data']['escalation_id'] == esc.id

    async def test_a_task_a_workflow_holds_is_left_alone(
        self, queue: EscalationQueue, reconcile_landing: AsyncMock,
    ) -> None:
        store = FakeTaskStore('in-progress', held=True)

        settled = await _owner(store, queue, reconcile_landing).settle(_request(), _VERIFY_RED)

        assert settled == Settled.WORKFLOW_HELD
        assert store.writes == []
        assert _open_l1s(queue) == []

    @pytest.mark.parametrize('status', ['done', 'blocked', 'cancelled'])
    async def test_a_task_another_owner_moved_is_left_alone(
        self, queue: EscalationQueue, reconcile_landing: AsyncMock, status: str,
    ) -> None:
        store = FakeTaskStore(status)

        settled = await _owner(store, queue, reconcile_landing).settle(_request(), _VERIFY_RED)

        assert settled == Settled.STATUS_MOVED
        assert store.writes == []
        assert _open_l1s(queue) == []

    async def test_a_block_that_never_reaches_the_store_is_still_escalated_as_unwritten(
        self, queue: EscalationQueue, reconcile_landing: AsyncMock,
    ) -> None:
        store = FakeTaskStore('in-progress', writes_fail=True)

        settled = await _owner(store, queue, reconcile_landing).settle(_request(), _VERIFY_RED)

        assert settled == Settled.BLOCK_UNWRITTEN
        [esc] = _open_l1s(queue)
        assert 'did not reach the store' in esc.detail
        assert '-> blocked' not in esc.detail

    async def test_a_failing_status_read_settles_as_raised_with_nothing_filed(
        self, queue: EscalationQueue, reconcile_landing: AsyncMock,
    ) -> None:
        store = FakeTaskStore(reads_fail=True)

        settled = await _owner(store, queue, reconcile_landing).settle(_request(), _VERIFY_RED)

        assert settled == Settled.RAISED
        assert _open_l1s(queue) == []


@pytest.mark.asyncio
class TestLandingDrivesTheLandedOutboxDoneWrite:
    @pytest.mark.parametrize('outcome', [
        MergeOutcome(status='done', merge_sha='a' * 40),
        MergeOutcome(status='already_merged', merge_sha='a' * 40),
    ], ids=lambda outcome: outcome.status)
    async def test_landing_calls_reconcile_landing(
        self, queue: EscalationQueue, reconcile_landing: AsyncMock, outcome: MergeOutcome,
    ) -> None:
        store = FakeTaskStore('in-progress')

        settled = await _owner(store, queue, reconcile_landing).settle(_request(), outcome)

        assert settled == Settled.LANDED
        reconcile_landing.assert_awaited_once_with(_TID)
        assert store.writes == []
        assert _open_l1s(queue) == []

    async def test_landing_with_no_outbox_row_is_reported_unrecorded(
        self, queue: EscalationQueue,
    ) -> None:
        store = FakeTaskStore('in-progress')
        no_row = AsyncMock(return_value=None)

        settled = await _owner(store, queue, no_row).settle(
            _request(), MergeOutcome(status='already_merged'),
        )

        assert settled == Settled.LANDING_UNRECORDED
        assert store.writes == []

    @pytest.mark.parametrize('disposition,settled', [
        ('marked_done', Settled.LANDED),
        ('already_done_pruned', Settled.LANDED),
        ('skipped', Settled.LANDING_NOT_DONE),
        ('stale_conflict', Settled.LANDING_NOT_DONE),
        ('delivered_checks_withheld', Settled.LANDING_NOT_DONE),
        ('pruned_not_landed', Settled.LANDING_NOT_DONE),
    ])
    async def test_only_a_done_write_or_a_resting_task_settles_as_landed(
        self, queue: EscalationQueue, disposition: str, settled: Settled,
    ) -> None:
        owner = _owner(FakeTaskStore('in-progress'), queue, AsyncMock(return_value=disposition))

        assert await owner.settle(_request(), MergeOutcome(status='done')) == settled
        assert _open_l1s(queue) == []


@pytest.mark.asyncio
class TestTransientOutcomeIsLeftToTheSweep:
    @pytest.mark.parametrize('outcome', [
        MergeOutcome(status='superseded'),
    ], ids=lambda outcome: outcome.status)
    async def test_nothing_happens(
        self, queue: EscalationQueue, reconcile_landing: AsyncMock, outcome: MergeOutcome,
    ) -> None:
        store = FakeTaskStore('in-progress')

        settled = await _owner(store, queue, reconcile_landing).settle(_request(), outcome)

        assert settled == Settled.TRANSIENT
        reconcile_landing.assert_not_awaited()
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


_WORKFLOW_ROLE = 'orchestrator'


async def _workflow_attached_to(
    registry: InFlightMergeRegistry, queue: EscalationQueue, halt: FakeMergeHalt,
) -> None:
    """A workflow peer-attached to the in-flight request, awaiting it as
    ``workflow.py::TaskWorkflow._await_cancellable`` does, then filing its halt
    record as its wip handler does."""
    waiter: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
    assert registry.attach(_TID, WaiterRecord(request_id='mr-workflow', future=waiter, source='workflow'))
    cancel_task = asyncio.create_task(asyncio.Event().wait())
    try:
        await asyncio.wait({waiter, cancel_task}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        cancel_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cancel_task
    outcome = waiter.result()
    submit_halt_owning_escalation(queue, halt, Escalation(
        id=queue.make_id(_TID), task_id=_TID, agent_role=_WORKFLOW_ROLE,
        severity='blocking', category=MERGE_HALT_CATEGORY_BY_STATUS[outcome.status],
        summary=f'merge {outcome.status}', detail='', level=1,
    ))


@pytest.mark.asyncio
class TestHaltOutcomeOwnsTheMergeHalt:
    @pytest.mark.parametrize('status', sorted(_HALT_OUTCOMES))
    async def test_halt_outcome_files_the_halt_owning_l1(
        self, queue: EscalationQueue, reconcile_landing: AsyncMock, status: str,
    ) -> None:
        store = FakeTaskStore('in-progress')
        halt = FakeMergeHalt()

        settled = await _owner(store, queue, reconcile_landing, merge_halt=halt).settle(
            _request(), _HALT_OUTCOMES[status],
        )

        assert settled == Settled.HALTED
        [esc] = _open_l1s(queue)
        assert esc.category == MERGE_HALT_CATEGORY_BY_STATUS[status]
        assert esc.suggested_action == 'manual_intervention'
        assert _REQUEST_ID in esc.detail
        assert halt.owners == [esc.id]
        if status == 'done_wip_recovery':
            reconcile_landing.assert_awaited_once_with(_TID)
            assert (_TID, 'blocked') not in store.writes
        else:
            assert store.writes == [(_TID, 'blocked')]
            reconcile_landing.assert_not_awaited()

    async def test_an_already_owned_halt_gets_a_non_owning_record(
        self, queue: EscalationQueue, reconcile_landing: AsyncMock,
    ) -> None:
        store = FakeTaskStore('in-progress')
        halt = FakeMergeHalt(halt_owner_esc_id='esc-other')

        await _owner(store, queue, reconcile_landing, merge_halt=halt).settle(
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
        self, queue: EscalationQueue, reconcile_landing: AsyncMock,
        halt: FakeMergeHalt | None,
    ) -> None:
        store = FakeTaskStore('in-progress')

        await _owner(store, queue, reconcile_landing, merge_halt=halt).settle(
            _request(), _STASH_FAILED,
        )

        assert _open_l1s_in(queue, 'stash_failed') == []
        [esc] = _open_l1s(queue)
        assert esc.category == 'task_failure'
        assert store.writes == [(_TID, 'blocked')]

    async def test_a_moved_task_status_still_files_the_halt_owner(
        self, queue: EscalationQueue, reconcile_landing: AsyncMock,
    ) -> None:
        store = FakeTaskStore('blocked')
        halt = FakeMergeHalt()

        await _owner(store, queue, reconcile_landing, merge_halt=halt).settle(
            _request(), _STASH_FAILED,
        )

        [esc] = _open_l1s_in(queue, 'stash_failed')
        assert halt.owners == [esc.id]
        assert store.writes == []

    @pytest.mark.parametrize('status', sorted(_HALT_OUTCOMES))
    async def test_a_held_task_is_left_to_its_workflow(
        self, queue: EscalationQueue, reconcile_landing: AsyncMock, status: str,
    ) -> None:
        store = FakeTaskStore('in-progress', held=True)
        halt = FakeMergeHalt()

        settled = await _owner(store, queue, reconcile_landing, merge_halt=halt).settle(
            _request(), _HALT_OUTCOMES[status],
        )

        assert settled == Settled.WORKFLOW_HELD
        assert _open_l1s(queue) == []
        assert halt.owners == []
        assert store.writes == []
        reconcile_landing.assert_not_awaited()

    @pytest.mark.parametrize('status', ['wip_halted', 'done_wip_recovery'])
    async def test_a_workflow_attached_to_the_recovered_request_alone_owns_the_halt(
        self, queue: EscalationQueue, reconcile_landing: AsyncMock, status: str,
    ) -> None:
        registry = InFlightMergeRegistry()
        primary = _request()
        registry.acquire(_TID, _TID, primary.result, request_id=_REQUEST_ID, source='recovery')
        halt = FakeMergeHalt()
        settles: list[asyncio.Task[Settled]] = []
        _owner(FakeTaskStore('in-progress', held=True), queue, reconcile_landing, merge_halt=halt).adopt(
            [primary], schedule=lambda coro, *, label: settles.append(asyncio.create_task(coro)),
        )
        workflow = asyncio.create_task(_workflow_attached_to(registry, queue, halt))
        await asyncio.sleep(0)

        primary.result.set_result(_HALT_OUTCOMES[status])
        await workflow
        assert [await settle for settle in settles] == [Settled.WORKFLOW_HELD]

        [esc] = _open_l1s(queue)
        assert esc.category == MERGE_HALT_CATEGORY_BY_STATUS[status]
        assert esc.agent_role == _WORKFLOW_ROLE
        assert halt.owners == [esc.id]

    async def test_a_second_settle_files_no_second_halt_record(
        self, queue: EscalationQueue, reconcile_landing: AsyncMock,
    ) -> None:
        store = FakeTaskStore('in-progress')
        halt = FakeMergeHalt()
        owner = _owner(store, queue, reconcile_landing, merge_halt=halt)

        await owner.settle(_request(), _STASH_FAILED)
        await owner.settle(_request(), _STASH_FAILED)

        assert len(_open_l1s_in(queue, 'stash_failed')) == 1
        assert len(halt.owners) == 1


@pytest.mark.asyncio
class TestWorkerShutdownOutcomeIsLeftForTheNextStart:
    async def test_shutdown_outcome_writes_and_files_nothing(
        self, queue: EscalationQueue, reconcile_landing: AsyncMock,
    ) -> None:
        store = FakeTaskStore('in-progress')
        halt = FakeMergeHalt()

        settled = await _owner(store, queue, reconcile_landing, merge_halt=halt).settle(
            _request(), _SHUTDOWN,
        )

        assert settled == Settled.WORKER_SHUTDOWN
        assert store.writes == []
        assert _open_l1s(queue) == []
        reconcile_landing.assert_not_awaited()
        assert halt.owners == []

    async def test_the_restored_requests_real_failure_still_gets_its_owner(
        self, queue: EscalationQueue, reconcile_landing: AsyncMock,
    ) -> None:
        store = FakeTaskStore('in-progress')
        owner = _owner(store, queue, reconcile_landing, merge_halt=FakeMergeHalt())

        await owner.settle(_request(), _SHUTDOWN)
        settled = await owner.settle(_request(), _VERIFY_RED)

        assert settled == Settled.BLOCKED
        assert store.writes == [(_TID, 'blocked')]
        [esc] = _open_l1s(queue)
        assert 'post-merge verify red' in esc.summary

    async def test_the_restored_requests_landing_leaves_no_stale_l1(
        self, queue: EscalationQueue, reconcile_landing: AsyncMock,
    ) -> None:
        owner = _owner(
            FakeTaskStore('in-progress'), queue, reconcile_landing, merge_halt=FakeMergeHalt(),
        )

        await owner.settle(_request(), _SHUTDOWN)
        settled = await owner.settle(_request(), MergeOutcome(status='done', merge_sha='a' * 40))

        assert settled == Settled.LANDED
        reconcile_landing.assert_awaited_once_with(_TID)
        assert _open_l1s(queue) == []

    async def test_adopt_schedules_a_settle_that_leaves_the_shutdown_alone(
        self, queue: EscalationQueue, reconcile_landing: AsyncMock,
    ) -> None:
        store = FakeTaskStore('in-progress')
        collector = ScheduleCollector()
        request = _request()
        _owner(store, queue, reconcile_landing, merge_halt=FakeMergeHalt()).adopt(
            [request], schedule=collector,
        )

        request.result.set_result(_SHUTDOWN)
        await asyncio.sleep(0)

        [(coro, _label)] = collector.scheduled
        assert await coro == Settled.WORKER_SHUTDOWN
        assert store.writes == []

    async def test_only_the_exact_shutdown_reason_is_left_alone(
        self, queue: EscalationQueue, reconcile_landing: AsyncMock,
    ) -> None:
        # The journal removes this record too, so nothing retries it at the next start.
        aborted = MergeOutcome(
            status='blocked', reason=f'{MERGE_WORKER_SHUTDOWN_REASON}: verify aborted',
        )

        settled = await _owner(
            FakeTaskStore('in-progress'), queue, reconcile_landing, merge_halt=FakeMergeHalt(),
        ).settle(_request(), aborted)

        assert settled == Settled.BLOCKED
        assert len(_open_l1s(queue)) == 1


@pytest.mark.asyncio
class TestAdoptSchedulesASettleWhenTheFutureFinalizes:
    async def test_resolved_failure_blocks_and_escalates(
        self, queue: EscalationQueue, reconcile_landing: AsyncMock,
    ) -> None:
        store = FakeTaskStore('in-progress')
        collector = ScheduleCollector()
        request = _request()
        _owner(store, queue, reconcile_landing).adopt([request], schedule=collector)

        request.result.set_result(_VERIFY_RED)
        await asyncio.sleep(0)

        [(coro, label)] = collector.scheduled
        assert _TID in label
        assert await coro == Settled.BLOCKED
        assert store.writes == [(_TID, 'blocked')]
        assert len(_open_l1s(queue)) == 1

    async def test_raised_future_is_a_merge_error_failure(
        self, queue: EscalationQueue, reconcile_landing: AsyncMock,
    ) -> None:
        store = FakeTaskStore('in-progress')
        collector = ScheduleCollector()
        request = _request()
        _owner(store, queue, reconcile_landing).adopt([request], schedule=collector)

        request.result.set_exception(RuntimeError('verify host lost'))
        await asyncio.sleep(0)

        [(coro, _label)] = collector.scheduled
        assert await coro == Settled.BLOCKED
        assert store.writes == [(_TID, 'blocked')]
        [esc] = _open_l1s(queue)
        assert esc.category == 'merge_error'
        assert 'verify host lost' in esc.detail

    async def test_cancelled_future_schedules_nothing(
        self, queue: EscalationQueue, reconcile_landing: AsyncMock,
    ) -> None:
        collector = ScheduleCollector()
        request = _request()
        _owner(FakeTaskStore(), queue, reconcile_landing).adopt([request], schedule=collector)

        request.result.cancel()
        await asyncio.sleep(0)

        assert collector.scheduled == []
