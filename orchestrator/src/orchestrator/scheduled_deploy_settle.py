"""Advance a scheduled self-deploy's ``deploy_state`` once its restart is observed (task 7276).

A self-target deterministic deploy goes done when its restart is only scheduled:
a detached systemd-run transient restarts the fleet later, this orchestrator
last.  Its dependents wait on ``deploy_state.phase``
(``orchestrator/src/orchestrator/deploy_state.py::awaits_scheduled_restart``),
and this module is what moves that phase: to VERIFIED once this process started
after the transient fired and the transient has finished, to FAILED on the
transient's fire-time escalation or a failed unit.  Rationale:
``docs/deterministic-runner.md`` §2.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, Protocol

from orchestrator.deploy_state import (
    DeployPhase,
    DeployState,
    awaits_scheduled_restart,
    enforce_transition,
)
from orchestrator.deterministic_runner import (
    DETERMINISTIC_AGENT_ROLE,
    file_illegal_deploy_transition,
)

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue

__all__ = [
    'ScheduledDeploySettler',
    'ScheduledRestart',
    'settled_phase',
]

logger = logging.getLogger(__name__)

_UNIT_RUNNING_STATES = frozenset({'active', 'activating', 'deactivating', 'reloading', 'refreshing'})

# The RP-4 wrapper (orchestrator/src/orchestrator/proc_supervision.py) files a
# fire-time failure under the runner's own role with this category.
_FIRE_FAILURE_CATEGORY = 'infra_issue'

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


def settled_phase(
    restart: ScheduledRestart,
    *,
    process_started_at: datetime,
    transient_active_state: str,
    fire_failures: int,
) -> DeployPhase:
    """The phase the evidence supports: FAILED, VERIFIED, or still SCHEDULED.

    ``restart-all-orchestrators.sh`` restarts this orchestrator last, so this
    process starting after the transient fired, with the transient no longer
    running, means every unit was restarted.  An empty ActiveState is the
    inspector's unreadable sentinel and never verifies.
    """
    if fire_failures > 0 or transient_active_state == 'failed':
        return DeployPhase.FAILED
    if process_started_at <= restart.fires_at:
        return DeployPhase.SCHEDULED
    if not transient_active_state or transient_active_state in _UNIT_RUNNING_STATES:
        return DeployPhase.SCHEDULED
    return DeployPhase.VERIFIED


class TaskLookup(Protocol):
    async def find_tasks_by_metadata(
        self, key: str, value: str | None, statuses: Sequence[str], *, include_value: bool = False,
    ) -> tuple[list[dict], Exception | None]: ...


class TaskStore(Protocol):
    async def get_task(self, task_id: str, /) -> dict | None: ...

    async def update_task(self, task_id: str, metadata: dict, *, metadata_mode: str) -> bool: ...


class ScheduledDeploySettler:
    """One settle pass over every done task whose deploy is still SCHEDULED."""

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
        for row in rows:
            if not _row_may_be_scheduled(row):
                continue
            task_id = str(row['task_id'])
            try:
                await self._settle(task_id)
            except Exception:
                logger.warning('scheduled-deploy settle: task %s not settled', task_id, exc_info=True)

    async def _settle(self, task_id: str) -> None:
        task = await self._store.get_task(task_id)
        state = _held_scheduled_state(task)
        if task is None or state is None:
            return
        restart = ScheduledRestart.from_metadata(task['metadata'])
        if restart is None:
            logger.warning(
                'scheduled-deploy settle: task %s has no readable before_done_scheduled_at; '
                'its dependents stay held', task_id,
            )
            return
        inspection = await self._inspect(restart.transient_unit)
        failures = self._fire_failures(task_id, restart)
        phase = settled_phase(
            restart,
            process_started_at=self._process_started_at,
            transient_active_state=str(inspection.get('ActiveState') or ''),
            fire_failures=len(failures),
        )
        if phase is DeployPhase.SCHEDULED:
            logger.debug(
                'scheduled-deploy settle: task %s restart not yet observed (transient %s reads %r)',
                task_id, restart.transient_unit, inspection.get('ActiveState'),
            )
            return
        await self._write(task_id, state, phase, restart, failures)

    def _fire_failures(self, task_id: str, restart: ScheduledRestart) -> list[str]:
        """Ids of the RP-4 fire-time escalations filed for *task_id* since it was scheduled."""
        records = self._queue.get_by_task(task_id, agent_role=DETERMINISTIC_AGENT_ROLE)
        return [
            record.id for record in records
            if record.category == _FIRE_FAILURE_CATEGORY
            and _filed_at_or_after(record.timestamp, restart.scheduled_at)
        ]

    async def _write(
        self,
        task_id: str,
        state: DeployState,
        phase: DeployPhase,
        restart: ScheduledRestart,
        failures: list[str],
    ) -> None:
        enforce_transition(
            state.phase, phase, task_id=task_id,
            escalation_sink=partial(file_illegal_deploy_transition, self._queue),
        )
        changes: dict[str, object] = {'phase': phase}
        if phase is DeployPhase.VERIFIED:
            changes['verified_at'] = self._clock().isoformat()
        new_state = state.model_copy(update=changes)
        if not await self._store.update_task(task_id, new_state.to_metadata(), metadata_mode='merge'):
            logger.warning(
                'scheduled-deploy settle: task %s deploy_state.phase=%s write refused; '
                'retried next pass', task_id, phase,
            )
            return
        self._log_settled(task_id, phase, restart, failures)

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
            'scheduled-deploy settle: task %s restart FAILED at fire time (transient %s, '
            'escalations %s); its dependents stay held until a human acts on that escalation',
            task_id, restart.transient_unit, failures or 'none — the unit reads failed',
        )


def _row_may_be_scheduled(row: dict) -> bool:
    value = row.get('matched_value')
    return not isinstance(value, dict) or value.get('phase') == DeployPhase.SCHEDULED


def _held_scheduled_state(task: dict | None) -> DeployState | None:
    """*task*'s deploy state when it is a done scheduled self-deploy still at SCHEDULED."""
    if not isinstance(task, dict) or task.get('status') != 'done':
        return None
    metadata = task.get('metadata')
    if not isinstance(metadata, dict) or not awaits_scheduled_restart(metadata):
        return None
    state = DeployState.from_metadata(metadata)
    if state is None or state.phase is not DeployPhase.SCHEDULED:
        return None
    return state


def _filed_at_or_after(timestamp: str, moment: datetime) -> bool:
    """Whether a record was filed at or after *moment*; an unreadable stamp counts."""
    filed = _aware_datetime(timestamp)
    return filed is None or filed >= moment
