"""Owns the terminal outcome of a merge request recovered from the journal at startup.

No workflow awaits such a request: the process that submitted it is gone, so
this owner turns its outcome into the task-row and escalation writes a waiting
workflow would have made.
"""

from __future__ import annotations

import asyncio
import enum
import functools
import logging
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from escalation.models import Escalation
from shared.task_statuses import TaskStatus

from orchestrator.event_store import EventType
from orchestrator.merge_lane.types import MergeOutcome
from orchestrator.stranded_verified_green import (
    DURABLE_MERGE_FAILURE_STATUSES,
    SUCCESS_TRANSIENT_MERGE_STATUSES,
)

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue

    from orchestrator.event_store import EventStore

logger = logging.getLogger(__name__)

__all__ = [
    'AGENT_ROLE',
    'LANDED_MERGE_STATUSES',
    'OutcomeTasks',
    'RecoveredMergeOwner',
    'RecoveredRequest',
    'ScheduleCoro',
    'Settled',
]

AGENT_ROLE = 'harness-recovered-merge-owner'

LANDED_MERGE_STATUSES: frozenset[str] = frozenset({
    'done', 'already_merged', 'done_wip_recovery',
})

_BLOCKABLE_STATUSES: frozenset[str] = frozenset({
    TaskStatus.IN_PROGRESS, TaskStatus.PENDING,
})


class RecoveredRequest(Protocol):
    @property
    def task_id(self) -> str: ...
    @property
    def request_id(self) -> str: ...
    @property
    def result(self) -> asyncio.Future[MergeOutcome]: ...


class OutcomeTasks(Protocol):
    """What settling needs of the task store; the Scheduler is one."""

    def is_actively_held(self, tid: str, /) -> bool: ...
    async def get_status(self, tid: str, /) -> str | None: ...
    async def clear_claim_then_set_status(self, tid: str, status: str, /) -> None: ...


class ScheduleCoro(Protocol):
    """Runs a coroutine in the background; ``Harness._schedule_coro_threadsafe`` is one."""

    def __call__(self, coro: Coroutine[Any, Any, Any], *, label: str) -> None: ...


class Settled(enum.StrEnum):
    """What :meth:`RecoveredMergeOwner.settle` did with one outcome."""

    LANDED = 'landed'
    LANDING_UNRECORDED = 'landing_unrecorded'
    BLOCKED = 'blocked'
    WORKFLOW_HELD = 'workflow_held'
    STATUS_MOVED = 'status_moved'
    TRANSIENT = 'transient'
    RAISED = 'raised'


@dataclass(frozen=True)
class RecoveredMergeOwner:
    """Settles each adopted request's outcome once its future finalizes.

    A landing drives *reconcile_landed* (the landed-outbox done-write). A
    durable failure, or any status the lane's partition does not classify,
    blocks the task and files one L1 under the lane's stated block category.
    A transient outcome is left to the stranded sweep.
    """

    tasks: OutcomeTasks
    escalations: EscalationQueue | None
    reconcile_landed: Callable[[str], Awaitable[bool]]
    event_store: EventStore | None = None

    def adopt(self, requests: Iterable[RecoveredRequest], *, schedule: ScheduleCoro) -> None:
        for request in requests:
            logger.info(
                'Recovered merge request %s owns task %s until it finalizes',
                request.request_id, request.task_id,
            )
            request.result.add_done_callback(
                functools.partial(self._on_finalized, request=request, schedule=schedule),
            )

    def _on_finalized(
        self, fut: asyncio.Future[MergeOutcome], *, request: RecoveredRequest,
        schedule: ScheduleCoro,
    ) -> None:
        try:
            if fut.cancelled():
                return
            exc = fut.exception()
            outcome = (
                fut.result() if exc is None
                else MergeOutcome(status='error', reason=f'merge future raised: {exc!r}')
            )
            schedule(
                self.settle(request, outcome),
                label=f'recovered-merge-outcome task {request.task_id}',
            )
        except Exception:
            logger.warning(
                'Recovered merge request %s for task %s: could not schedule its '
                'outcome — nothing settled', request.request_id, request.task_id,
                exc_info=True,
            )

    async def settle(self, request: RecoveredRequest, outcome: MergeOutcome) -> Settled:
        try:
            settled = await self._settle(request, outcome)
        except Exception:
            logger.warning(
                'Recovered merge request %s for task %s: settling outcome %s raised',
                request.request_id, request.task_id, outcome.status, exc_info=True,
            )
            return Settled.RAISED
        logger.info(
            'Recovered merge request %s for task %s finished %s: %s',
            request.request_id, request.task_id, outcome.status, settled.value,
        )
        return settled

    async def _settle(self, request: RecoveredRequest, outcome: MergeOutcome) -> Settled:
        tid = request.task_id
        if self.tasks.is_actively_held(tid):
            return Settled.WORKFLOW_HELD
        if outcome.status in LANDED_MERGE_STATUSES:
            if await self.reconcile_landed(tid):
                return Settled.LANDED
            logger.info(
                'Task %s landed with no outbox row; the stranded sweep marks it '
                'done from main once the request is out of flight', tid,
            )
            return Settled.LANDING_UNRECORDED
        if outcome.status in SUCCESS_TRANSIENT_MERGE_STATUSES:
            return Settled.TRANSIENT
        if outcome.status not in DURABLE_MERGE_FAILURE_STATUSES:
            logger.error(
                'Recovered merge request %s: UNCLASSIFIED merge status %r for task '
                '%s — treating it as a durable failure', request.request_id,
                outcome.status, tid,
            )
        return await self._block_and_escalate(request, outcome)

    async def _block_and_escalate(
        self, request: RecoveredRequest, outcome: MergeOutcome,
    ) -> Settled:
        tid = request.task_id
        live = await self.tasks.get_status(tid)
        if live not in _BLOCKABLE_STATUSES:
            return Settled.STATUS_MOVED
        if self.tasks.is_actively_held(tid):
            return Settled.WORKFLOW_HELD
        try:
            await self.tasks.clear_claim_then_set_status(tid, TaskStatus.BLOCKED)
        except Exception:
            logger.warning(
                'Task %s: blocking after recovered merge request %s failed did '
                'not reach the store; filing the escalation anyway',
                tid, request.request_id, exc_info=True,
            )
        self._file_l1(request, outcome, moved_from=live)
        return Settled.BLOCKED

    def _file_l1(
        self, request: RecoveredRequest, outcome: MergeOutcome, *, moved_from: str | None,
    ) -> None:
        tid = request.task_id
        category = outcome.block_category.value
        if self.escalations is None or self.escalations.has_open_l1(tid, category=category):
            return
        detail_lines = [
            f'Merge request {request.request_id} for task {tid} was recovered from '
            'the merge journal at startup, so no workflow awaited it.',
            f'status: {outcome.status}',
            f'reason: {outcome.reason}',
        ]
        if outcome.failure_category:
            detail_lines.append(f'failure_category: {outcome.failure_category}')
        if outcome.failure_cause_hint:
            detail_lines.append(f'failure_cause_hint: {outcome.failure_cause_hint}')
        detail_lines.append(f'The task was moved {moved_from} -> blocked.')
        esc = Escalation(
            id=self.escalations.make_id(tid),
            task_id=tid,
            agent_role=AGENT_ROLE,
            severity='blocking',
            category=category,
            summary=(
                f'Recovered merge request {request.request_id} for task {tid} '
                f'finished {outcome.status}: {outcome.reason}'
            )[:200],
            detail='\n'.join(detail_lines),
            suggested_action='investigate_and_retry',
            level=1,
        )
        self.escalations.submit(esc)
        if self.event_store is not None:
            self.event_store.emit(
                EventType.escalation_created,
                task_id=tid,
                data={
                    'escalation_id': esc.id,
                    'category': category,
                    'severity': 'blocking',
                    'level': 1,
                    'reason': f'recovered-merge-{outcome.status}',
                },
            )
        logger.warning(
            'Filed L1 %s (%s) for task %s: recovered merge request %s finished %s',
            esc.id, category, tid, request.request_id, outcome.status,
        )
