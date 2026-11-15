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
from orchestrator.merge_halt_escalation import (
    MERGE_HALT_STATUSES,
    MergeHaltControl,
    MergeHaltEscalationText,
    build_merge_halt_escalation_text,
    submit_halt_owning_escalation,
)
from orchestrator.merge_lane import MergeOutcome, is_merge_worker_shutdown
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

LANDED_MERGE_STATUSES: frozenset[str] = frozenset({'done', 'already_merged'})

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

    WORKER_SHUTDOWN = 'worker_shutdown'
    LANDED = 'landed'
    LANDING_UNRECORDED = 'landing_unrecorded'
    HALTED = 'halted'
    BLOCKED = 'blocked'
    WORKFLOW_HELD = 'workflow_held'
    STATUS_MOVED = 'status_moved'
    TRANSIENT = 'transient'
    RAISED = 'raised'


@dataclass(frozen=True)
class RecoveredMergeOwner:
    """Settles each adopted request's outcome once its future finalizes.

    - Landing: *reconcile_landed* makes the landed-outbox done-write.
    - Halt: files the halt-owning L1 and registers it with *merge_halt*; then blocks the task,
      or records the landing for ``done_wip_recovery``.
    - Transient (``superseded``): left to the stranded sweep.
    - Durable or unclassified failure: blocks the task and files one L1 under its block category.

    The merge worker's shutdown outcome is not terminal: the journal keeps the
    request and the next start restores and re-adopts it, so it is left alone.
    """

    tasks: OutcomeTasks
    escalations: EscalationQueue | None
    reconcile_landed: Callable[[str], Awaitable[bool]]
    merge_halt: MergeHaltControl | None
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
        if is_merge_worker_shutdown(outcome):
            return Settled.WORKER_SHUTDOWN
        tid = request.task_id
        if self.tasks.is_actively_held(tid):
            return Settled.WORKFLOW_HELD
        if outcome.status in MERGE_HALT_STATUSES:
            await self._own_halt(request, outcome)
            return Settled.HALTED
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
        moved_from = await self._block_task(request)
        if moved_from is None:
            return Settled.STATUS_MOVED
        self._file_l1(
            request, outcome, extra_lines=(f'The task was moved {moved_from} -> blocked.',),
        )
        return Settled.BLOCKED

    async def _block_task(self, request: RecoveredRequest) -> str | None:
        """Block the task unless another owner moved or holds it; returns the status it left."""
        tid = request.task_id
        live = await self.tasks.get_status(tid)
        if live not in _BLOCKABLE_STATUSES or self.tasks.is_actively_held(tid):
            return None
        try:
            await self.tasks.clear_claim_then_set_status(tid, TaskStatus.BLOCKED)
        except Exception:
            logger.warning(
                'Task %s: blocking after recovered merge request %s finished did '
                'not reach the store', tid, request.request_id, exc_info=True,
            )
        return live

    async def _own_halt(self, request: RecoveredRequest, outcome: MergeOutcome) -> None:
        tid = request.task_id
        self._file_halt_record(
            request, outcome, build_merge_halt_escalation_text(outcome.status, outcome, task_id=tid),
        )
        if outcome.status != 'done_wip_recovery':
            await self._block_task(request)
        elif not await self.reconcile_landed(tid):
            logger.info(
                'Task %s landed with WIP recovery but no outbox row; the stranded '
                'sweep marks it done from main once the request is out of flight', tid,
            )

    def _file_halt_record(
        self, request: RecoveredRequest, outcome: MergeOutcome, text: MergeHaltEscalationText,
    ) -> None:
        tid = request.task_id
        if self.escalations is None:
            logger.warning(
                'Task %s: recovered merge request %s halted the merge queue (%s) '
                'with no escalation queue to file on — orphan halt; release it '
                'with unhalt_merge_queue', tid, request.request_id, outcome.status,
            )
            return
        if self.escalations.has_open_l1(tid, category=text.category):
            return
        halt = self.merge_halt
        if halt is not None and halt.is_wip_halted and halt.halt_owner_esc_id is None:
            esc = Escalation(
                id=self.escalations.make_id(tid),
                task_id=tid,
                agent_role=AGENT_ROLE,
                severity='blocking',
                category=text.category,
                summary=text.summary[:200],
                detail=f'{text.detail}\n\n{_recovered_line(request)}',
                suggested_action='manual_intervention',
                level=1,
            )
            submit_halt_owning_escalation(self.escalations, halt, esc)
            self._emit_created(esc, outcome)
            return
        owner = halt.halt_owner_esc_id if halt is not None else None
        holder = (
            f'the halt is already owned by escalation {owner}' if owner is not None
            else 'no merge halt is engaged'
        )
        self._file_l1(
            request, outcome, category='task_failure',
            extra_lines=(
                f'This outcome halts the merge queue ({text.category}), but {holder}, '
                'so no halt-category record was filed.',
                text.detail,
            ),
        )

    def _file_l1(
        self, request: RecoveredRequest, outcome: MergeOutcome, *,
        category: str | None = None, extra_lines: Iterable[str] = (),
    ) -> None:
        tid = request.task_id
        category = category or outcome.block_category.value
        if self.escalations is None or self.escalations.has_open_l1(tid, category=category):
            return
        detail_lines = [
            _recovered_line(request),
            f'status: {outcome.status}',
            f'reason: {outcome.reason}',
        ]
        if outcome.failure_category:
            detail_lines.append(f'failure_category: {outcome.failure_category}')
        if outcome.failure_cause_hint:
            detail_lines.append(f'failure_cause_hint: {outcome.failure_cause_hint}')
        detail_lines.extend(extra_lines)
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
        self._emit_created(esc, outcome)

    def _emit_created(self, esc: Escalation, outcome: MergeOutcome) -> None:
        if self.event_store is not None:
            self.event_store.emit(
                EventType.escalation_created,
                task_id=esc.task_id,
                data={
                    'escalation_id': esc.id,
                    'category': esc.category,
                    'severity': 'blocking',
                    'level': 1,
                    'reason': f'recovered-merge-{outcome.status}',
                },
            )
        logger.warning(
            'Filed L1 %s (%s) for task %s: recovered merge request finished %s',
            esc.id, esc.category, esc.task_id, outcome.status,
        )


def _recovered_line(request: RecoveredRequest) -> str:
    return (
        f'Merge request {request.request_id} for task {request.task_id} was recovered '
        'from the merge journal at startup, so no workflow awaited it.'
    )
