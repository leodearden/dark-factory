"""Advance a held scheduled self-deploy's ``deploy_state`` (task 7276).

A self-target deterministic deploy goes done when its restart is only scheduled:
a detached systemd-run transient restarts the fleet later, this orchestrator
last.  Its dependents wait on ``deploy_state.phase``
(``orchestrator/src/orchestrator/deploy_state.py::awaits_scheduled_restart``),
and this module advances every phase that holds them, one hop per pass:
SCHEDULED to VERIFIED once the restart landed or to FAILED on failure
evidence; FAILED to ESCALATED, filing the born-at-L2 only when none exists;
ESCALATED to DONE once no own-role escalation is pending and the restart
landed.  Rationale: ``docs/deterministic-runner.md`` §2.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, Protocol

from pydantic import ValidationError
from shared.task_read import TaskAbsent, TaskRead, TaskReadFailed

from orchestrator.deploy_state import (
    RESTART_LANDED_PHASES,
    DeployPhase,
    DeployState,
    awaits_scheduled_restart,
    enforce_transition,
)
from orchestrator.deterministic_runner import (
    DETERMINISTIC_AGENT_ROLE,
    SELF_RESTART_FAILURE_CATEGORY,
    file_deterministic_escalation,
    file_illegal_deploy_transition,
)

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue

__all__ = [
    'ScheduledDeploySettler',
    'ScheduledRestart',
    'restart_landed',
    'settled_phase',
]

logger = logging.getLogger(__name__)

_UNIT_RUNNING_STATES = frozenset({'active', 'activating', 'deactivating', 'reloading', 'refreshing'})

# A restart still unobserved this long after its fire time is reported loudly.
_RESTART_OVERDUE_AFTER = timedelta(hours=6)

_ACTOR = 'ScheduledDeploySettler'

UnitInspector = Callable[[str], Awaitable[dict]]


@dataclass(frozen=True)
class ScheduledRestart:
    """The ``before_done_scheduled_at`` stamp: when the transient was scheduled to fire."""

    scheduled_at: datetime
    transient_unit: str
    fire_delay_secs: int

    @property
    def fires_at(self) -> datetime:
        return self.scheduled_at + timedelta(seconds=self.fire_delay_secs)

    @classmethod
    def from_metadata(cls, metadata: Mapping) -> ScheduledRestart | None:
        """The stamp in *metadata*, or None when it is absent or unreadable."""
        stamp = metadata.get('before_done_scheduled_at')
        if not isinstance(stamp, Mapping):
            return None
        scheduled_at = _aware_datetime(stamp.get('at'))
        unit = stamp.get('transient_unit')
        delay = stamp.get('fire_delay_secs')
        if scheduled_at is None or not isinstance(unit, str) or not unit:
            return None
        if not isinstance(delay, int) or isinstance(delay, bool) or delay < 0:
            return None
        return cls(scheduled_at, unit, delay)


def _aware_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def restart_landed(
    restart: ScheduledRestart, *, process_started_at: datetime, transient_active_state: str,
) -> bool:
    """Whether the scheduled restart ran to completion.

    ``restart-all-orchestrators.sh`` restarts this orchestrator last, so this
    process starting after the transient fired, with the transient finished
    and not failed, means every unit was restarted.  An empty ActiveState is
    the inspector's unreadable sentinel and never counts.
    """
    return (
        process_started_at > restart.fires_at
        and bool(transient_active_state)
        and transient_active_state not in _UNIT_RUNNING_STATES
        and transient_active_state != 'failed'
    )


def settled_phase(
    restart: ScheduledRestart,
    *,
    process_started_at: datetime,
    transient_active_state: str,
    failure_escalations: int,
) -> DeployPhase:
    """The phase the evidence supports: FAILED, VERIFIED, or still SCHEDULED."""
    if failure_escalations > 0 or transient_active_state == 'failed':
        return DeployPhase.FAILED
    if restart_landed(
        restart, process_started_at=process_started_at, transient_active_state=transient_active_state,
    ):
        return DeployPhase.VERIFIED
    return DeployPhase.SCHEDULED


class TaskLookup(Protocol):
    async def find_tasks_by_metadata(
        self, key: str, value: str | None, statuses: Sequence[str], *, include_value: bool = False,
    ) -> tuple[list[dict], Exception | None]: ...


class TaskStore(Protocol):
    async def read_task(self, task_id: str, /) -> TaskRead: ...

    async def update_task(self, task_id: str, metadata: dict, *, metadata_mode: str) -> bool: ...


class ScheduledDeploySettler:
    """One settle pass over every done scheduled self-deploy whose phase holds its dependents."""

    def __init__(
        self,
        store: TaskStore,
        lookup: TaskLookup,
        escalation_queue: EscalationQueue,
        *,
        process_started_at: datetime,
        unit_inspector: UnitInspector,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._lookup = lookup
        self._queue = escalation_queue
        self._process_started_at = process_started_at
        self._inspect = unit_inspector
        self._clock = clock or (lambda: datetime.now(UTC))

    async def settle_pass(self) -> None:
        rows, error = await self._lookup.find_tasks_by_metadata(
            'deploy_state', None, ('done',), include_value=True,
        )
        if error is not None:
            logger.warning(
                'scheduled-deploy settle: candidate lookup failed (%s); no phase advanced this pass',
                error,
            )
            return
        candidates = [str(row['task_id']) for row in rows if _row_may_hold(row)]
        outcomes = await asyncio.gather(
            *(self._settle(task_id) for task_id in candidates), return_exceptions=True,
        )
        for task_id, outcome in zip(candidates, outcomes, strict=True):
            if isinstance(outcome, Exception):
                logger.warning('scheduled-deploy settle: task %s not settled', task_id, exc_info=outcome)

    async def _settle(self, task_id: str) -> None:
        read = await self._store.read_task(task_id)
        if isinstance(read, TaskReadFailed):
            logger.warning(
                'scheduled-deploy settle: task %s could not be re-read (%s); not settled this '
                'pass, its dependents stay held', task_id, read.reason,
            )
            return
        if isinstance(read, TaskAbsent):
            logger.info('scheduled-deploy settle: task %s no longer exists; nothing to settle', task_id)
            return
        task = read.task
        state = _held_state(task_id, task)
        if state is None:
            return
        advance = {
            DeployPhase.SCHEDULED: self._settle_scheduled,
            DeployPhase.FAILED: self._route_failure,
            DeployPhase.ESCALATED: self._release_if_resolved,
        }.get(state.phase)
        if advance is None:
            logger.warning(
                'scheduled-deploy settle: task %s is held at deploy_state.phase=%s, which no writer '
                'advances; its dependents stay held until it is repaired by hand', task_id, state.phase,
            )
            return
        restart = ScheduledRestart.from_metadata(task['metadata'])
        if restart is None:
            logger.warning(
                'scheduled-deploy settle: task %s has no readable before_done_scheduled_at; '
                'its dependents stay held', task_id,
            )
            return
        await advance(task_id, state, restart)

    async def _settle_scheduled(self, task_id: str, state: DeployState, restart: ScheduledRestart) -> None:
        """SCHEDULED -> VERIFIED or FAILED, on the restart evidence."""
        active_state = await self._active_state(restart)
        failures = self._failure_escalations(task_id, restart)
        phase = settled_phase(
            restart,
            process_started_at=self._process_started_at,
            transient_active_state=active_state,
            failure_escalations=len(failures),
        )
        if phase is DeployPhase.SCHEDULED:
            self._report_restart_not_observed(task_id, restart, active_state)
            return
        changes = {'verified_at': self._clock().isoformat()} if phase is DeployPhase.VERIFIED else {}
        if await self._write(task_id, state, phase, changes):
            self._log_settled(task_id, phase, restart, failures)

    async def _route_failure(self, task_id: str, state: DeployState, restart: ScheduledRestart) -> None:
        """FAILED -> ESCALATED: a deploy failure always routes to a born-at-L2 escalation (DS-2)."""
        escalations = self._failure_escalations(task_id, restart) or self._file_failure(task_id, restart)
        if await self._write(task_id, state, DeployPhase.ESCALATED, {'escalated_at': self._clock().isoformat()}):
            logger.info(
                'scheduled-deploy settle: task %s failed restart routed to escalation(s) %s; '
                'its dependents stay held until none is pending',
                task_id, escalations or 'already pending',
            )

    async def _release_if_resolved(self, task_id: str, state: DeployState, restart: ScheduledRestart) -> None:
        """ESCALATED -> DONE once no own-role escalation is pending and the restart landed."""
        pending = self._queue.get_by_task(task_id, status='pending', agent_role=DETERMINISTIC_AGENT_ROLE)
        if pending:
            logger.warning(
                'scheduled-deploy settle: task %s is held at deploy_state.phase=escalated behind '
                'pending escalation(s) %s; its dependents stay held until a human resolves them',
                task_id, [record.id for record in pending],
            )
            return
        active_state = await self._active_state(restart)
        if not restart_landed(
            restart, process_started_at=self._process_started_at, transient_active_state=active_state,
        ):
            self._report_restart_not_observed(task_id, restart, active_state)
            return
        if await self._write(task_id, state, DeployPhase.DONE, {}):
            logger.info(
                'scheduled-deploy settle: task %s released — no escalation pending and transient %s '
                'fired at %s; its dependents are released',
                task_id, restart.transient_unit, restart.fires_at.isoformat(),
            )

    async def _active_state(self, restart: ScheduledRestart) -> str:
        inspection = await self._inspect(restart.transient_unit)
        return str(inspection.get('ActiveState') or '')

    def _report_restart_not_observed(self, task_id: str, restart: ScheduledRestart, active_state: str) -> None:
        """DEBUG while the restart may still land; WARNING once it cannot be judged or is overdue."""
        if not active_state:
            logger.warning(
                'scheduled-deploy settle: task %s restart cannot be judged: transient %s state is '
                'unreadable; its dependents stay held', task_id, restart.transient_unit,
            )
            return
        if self._clock() - restart.fires_at > _RESTART_OVERDUE_AFTER:
            logger.warning(
                'scheduled-deploy settle: task %s restart still not observed %s after its fire time %s '
                '(transient %s reads %r, this orchestrator started at %s); its dependents stay held',
                task_id, _RESTART_OVERDUE_AFTER, restart.fires_at.isoformat(), restart.transient_unit,
                active_state, self._process_started_at.isoformat(),
            )
            return
        logger.debug(
            'scheduled-deploy settle: task %s restart not yet observed (transient %s reads %r)',
            task_id, restart.transient_unit, active_state,
        )

    def _failure_escalations(self, task_id: str, restart: ScheduledRestart) -> list[str]:
        """Ids of own-role self-restart-failure escalations filed for *task_id* since it was scheduled.

        The RP-4 fire-time filing and the runner's own completion-failure
        filing share role and category, so both count: conservative, since a
        FAILED deploy is released once its escalations are resolved.
        """
        records = self._queue.get_by_task(task_id, agent_role=DETERMINISTIC_AGENT_ROLE)
        return [
            record.id for record in records
            if record.category == SELF_RESTART_FAILURE_CATEGORY
            and _filed_at_or_after(record.timestamp, restart.scheduled_at)
        ]

    def _file_failure(self, task_id: str, restart: ScheduledRestart) -> list[str]:
        """File the born-at-L2 the transient's RP-4 wrapper did not; its id, if filed."""
        filed = file_deterministic_escalation(
            self._queue,
            task_id,
            actor=_ACTOR,
            category=SELF_RESTART_FAILURE_CATEGORY,
            summary=f'Scheduled self-restart failed: {restart.transient_unit}',
            detail=(
                f'The scheduled self-restart transient {restart.transient_unit} (fire time '
                f'{restart.fires_at.isoformat()}) failed and filed no fire-time escalation. Task '
                f'{task_id} is done but its dependents are held until this escalation is resolved. '
                'Restart the fleet by hand (scripts/restart-all-orchestrators.sh), then resolve it.'
            ),
        )
        return [filed] if filed else []

    async def _write(
        self, task_id: str, state: DeployState, phase: DeployPhase, changes: Mapping[str, object],
    ) -> bool:
        enforce_transition(
            state.phase, phase, task_id=task_id,
            escalation_sink=partial(file_illegal_deploy_transition, self._queue, actor=_ACTOR),
        )
        new_state = state.model_copy(update={'phase': phase, **changes})
        if await self._store.update_task(task_id, new_state.to_metadata(), metadata_mode='merge'):
            return True
        logger.warning(
            'scheduled-deploy settle: task %s deploy_state.phase=%s write refused; '
            'retried next pass', task_id, phase,
        )
        return False

    def _log_settled(
        self, task_id: str, phase: DeployPhase, restart: ScheduledRestart, failures: list[str],
    ) -> None:
        if phase is DeployPhase.VERIFIED:
            logger.info(
                'scheduled-deploy settle: task %s restart verified — transient %s fired at %s, '
                'this orchestrator started at %s; its dependents are released',
                task_id, restart.transient_unit, restart.fires_at.isoformat(),
                self._process_started_at.isoformat(),
            )
            return
        logger.error(
            'scheduled-deploy settle: task %s restart failure evidence found (transient %s, '
            'escalations %s); its dependents stay held until no escalation for it is pending',
            task_id, restart.transient_unit, failures or 'none — the unit reads failed',
        )


def _row_may_hold(row: dict) -> bool:
    value = row.get('matched_value')
    return not isinstance(value, dict) or value.get('phase') not in RESTART_LANDED_PHASES


def _held_state(task_id: str, task: dict | None) -> DeployState | None:
    """*task*'s deploy state when it is a done scheduled self-deploy that still holds its dependents."""
    if not isinstance(task, dict) or task.get('status') != 'done':
        return None
    metadata = task.get('metadata')
    if not isinstance(metadata, dict) or not awaits_scheduled_restart(metadata):
        return None
    try:
        state = DeployState.from_metadata(metadata)
    except ValidationError:
        state = None
    if state is None:
        logger.warning(
            'scheduled-deploy settle: task %s deploy_state is unreadable; its dependents stay held '
            'until it is repaired by hand', task_id,
        )
    return state


def _filed_at_or_after(timestamp: str, moment: datetime) -> bool:
    """Whether a record was filed at or after *moment*; an unreadable stamp counts."""
    filed = _aware_datetime(timestamp)
    return filed is None or filed >= moment
