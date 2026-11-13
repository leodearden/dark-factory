"""Run orchestrator startup steps so a failing step is reported and, unless fatal, skipped.

Design: plans/startup-hardening-prd.md §3.1-3.2.
"""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from escalation.categories import EscalationCategory
from escalation.file_or_refresh import RefreshableAlarm, file_or_refresh

from orchestrator.event_store import EventStore, EventType

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue

logger = logging.getLogger(__name__)

STARTUP_STEP_SENTINEL_PREFIX = 'orchestrator-startup-'
_STEP_NAME_PATTERN = re.compile(r'[0-9a-z][0-9a-z-]*')
_L1_AGENT_ROLE = 'orchestrator-startup'
_SUMMARY_LIMIT = 200


def sentinel_task_id(step_name: str) -> str:
    return f'{STARTUP_STEP_SENTINEL_PREFIX}{step_name}'


class StepStatus(StrEnum):
    OK = 'ok'
    FAILED = 'failed'
    PARTIAL = 'partial'


@dataclass(frozen=True)
class EntryFailure:
    entry: str
    exception_type: str
    message: str


class EntryFailures:
    """The entries a sweep skipped after they raised; the sweep returns it to the runner."""

    def __init__(self, sweep: str) -> None:
        self._sweep = sweep
        self._failures: list[EntryFailure] = []

    def record(self, entry: str, exc: Exception) -> None:
        self._failures.append(EntryFailure(entry, type(exc).__name__, str(exc)))
        logger.error(
            '%s: entry %s failed, continuing with the next entry: %s: %s',
            self._sweep, entry, type(exc).__name__, exc, exc_info=exc,
        )

    @property
    def failures(self) -> tuple[EntryFailure, ...]:
        return tuple(self._failures)


@dataclass(frozen=True)
class StartupStep:
    name: str
    run: Callable[[], object]
    fatal: bool = False

    def __post_init__(self) -> None:
        if not _STEP_NAME_PATTERN.fullmatch(self.name):
            raise ValueError(
                f'startup step name {self.name!r} is not a lowercase slug '
                f'matching {_STEP_NAME_PATTERN.pattern}'
            )


@dataclass(frozen=True)
class StepOutcome:
    step: str
    status: StepStatus
    exception_type: str | None = None
    message: str | None = None
    failed_entries: tuple[EntryFailure, ...] = ()

    @classmethod
    def ok(cls, step: str) -> StepOutcome:
        return cls(step, StepStatus.OK)

    @classmethod
    def failed(cls, step: str, exc: Exception) -> StepOutcome:
        return cls(step, StepStatus.FAILED, type(exc).__name__, str(exc))

    @classmethod
    def partial(cls, step: str, failures: tuple[EntryFailure, ...]) -> StepOutcome:
        return cls(step, StepStatus.PARTIAL, failed_entries=failures)


@dataclass(frozen=True)
class StepInFlight:
    step: str
    started_at: datetime


class StartupJournal:
    """Which startup step started last, held in memory."""

    def __init__(self) -> None:
        self._in_flight: StepInFlight | None = None

    def record_in_flight(self, step: str, started_at: datetime) -> None:
        self._in_flight = StepInFlight(step, started_at)

    @property
    def in_flight(self) -> StepInFlight | None:
        return self._in_flight


async def run_startup_step(
    step: StartupStep,
    *,
    journal: StartupJournal,
    event_store: EventStore | None,
    escalation_queue: EscalationQueue | None,
    filing_claimant_run_id: str | None,
) -> StepOutcome:
    """Run *step*; report a non-ok outcome; re-raise only a fatal step's exception."""
    journal.record_in_flight(step.name, datetime.now(UTC))
    try:
        result = step.run()
        if inspect.isawaitable(result):
            result = await result
    except Exception as exc:
        outcome = StepOutcome.failed(step.name, exc)
        logger.error(
            'startup step %s failed%s: %s: %s',
            step.name, ' (fatal, startup refused)' if step.fatal else '',
            outcome.exception_type, outcome.message, exc_info=exc,
        )
        _report(outcome, step.fatal, event_store, escalation_queue, filing_claimant_run_id)
        if step.fatal:
            raise
        return outcome
    outcome = _outcome_of_result(step.name, result)
    if outcome.status is StepStatus.PARTIAL:
        logger.error(
            'startup step %s partial: %d entr(ies) failed: %s',
            step.name, len(outcome.failed_entries), _entry_names(outcome),
        )
        _report(outcome, step.fatal, event_store, escalation_queue, filing_claimant_run_id)
    return outcome


def _outcome_of_result(step: str, result: object) -> StepOutcome:
    if isinstance(result, EntryFailures) and result.failures:
        return StepOutcome.partial(step, result.failures)
    return StepOutcome.ok(step)


def _entry_names(outcome: StepOutcome) -> str:
    return ', '.join(failure.entry for failure in outcome.failed_entries)


def _report(
    outcome: StepOutcome,
    fatal: bool,
    event_store: EventStore | None,
    escalation_queue: EscalationQueue | None,
    filing_claimant_run_id: str | None,
) -> None:
    if event_store is not None:
        event_store.emit(EventType.startup_step_failed, data=_event_data(outcome))
    if escalation_queue is None:
        return
    try:
        file_or_refresh(
            escalation_queue,
            _step_alarm(outcome.step),
            summary=_l1_summary(outcome)[:_SUMMARY_LIMIT],
            detail=_l1_detail(outcome, fatal=fatal),
            filing_claimant_run_id=filing_claimant_run_id,
        )
    except Exception:
        logger.warning(
            'startup step %s: filing its L1 escalation failed', outcome.step, exc_info=True,
        )


def _event_data(outcome: StepOutcome) -> dict:
    return {
        'step': outcome.step,
        'status': outcome.status.value,
        'exception_type': outcome.exception_type,
        'message': outcome.message,
        'failed_entries': [asdict(failure) for failure in outcome.failed_entries],
    }


def _step_alarm(step: str) -> RefreshableAlarm:
    return RefreshableAlarm(
        task_id=sentinel_task_id(step),
        level=1,
        agent_role=_L1_AGENT_ROLE,
        severity='blocking',
        category=EscalationCategory.INFRA_ISSUE,
        suggested_action='investigate_startup_step',
    )


def _l1_summary(outcome: StepOutcome) -> str:
    if outcome.status is StepStatus.PARTIAL:
        return (
            f'startup step {outcome.step} partial: {len(outcome.failed_entries)} '
            f'entr(ies) failed ({_entry_names(outcome)})'
        )
    return f'startup step {outcome.step} failed: {outcome.exception_type}: {outcome.message}'


def _l1_detail(outcome: StepOutcome, *, fatal: bool) -> str:
    lines = [f'step: {outcome.step}', f'status: {outcome.status.value}']
    if outcome.exception_type is not None:
        lines.append(f'exception: {outcome.exception_type}: {outcome.message}')
    lines.extend(
        f'failed entry {failure.entry}: {failure.exception_type}: {failure.message}'
        for failure in outcome.failed_entries
    )
    refused = fatal and outcome.status is StepStatus.FAILED
    lines.append('startup refused (fatal step)' if refused else 'startup continued')
    return '\n'.join(lines)
