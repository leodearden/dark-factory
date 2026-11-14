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
from orchestrator.merge_lane import MergeOutcome
from orchestrator.merge_lane.outcome_facts import MergeBlockCategory
from orchestrator.recovered_merge_owner import RecoveredMergeOwner, Settled

_TID = '40'
_REQUEST_ID = 'mr-recovered'
_VERIFY_RED = MergeOutcome(
    status='blocked',
    reason='post-merge verify red: test_x',
    block_category=MergeBlockCategory.POST_MERGE_VERIFY,
)


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
    event_store: MagicMock | None = None,
) -> RecoveredMergeOwner:
    return RecoveredMergeOwner(
        tasks=store, escalations=queue, reconcile_landed=reconcile_landed,
        event_store=event_store,
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
        MergeOutcome(status='done_wip_recovery', merge_sha='a' * 40),
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
        MergeOutcome(status='wip_halted'),
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
