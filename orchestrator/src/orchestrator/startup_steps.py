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
from pathlib import Path
from typing import TYPE_CHECKING

from escalation.categories import EscalationCategory
from escalation.file_or_refresh import RefreshableAlarm, file_or_refresh
from pydantic import BaseModel, ConfigDict, Field
from shared.safe_io import atomic_write_text

from orchestrator.event_store import EventStore, EventType

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue

logger = logging.getLogger(__name__)

STARTUP_STEP_SENTINEL_PREFIX = 'orchestrator-startup-'
_STEP_NAME_PATTERN = re.compile(r'[0-9a-z][0-9a-z-]*')
_AGENT_ROLE = 'orchestrator-startup'
_SUMMARY_LIMIT = 200
CRASH_LOOP_THRESHOLD = 3
CRASH_LOOP_SENTINEL = 'orchestrator-startup-crash-loop'
_CRASH_LOOP_ALARM = RefreshableAlarm(
    task_id=CRASH_LOOP_SENTINEL,
    level=2,
    agent_role=_AGENT_ROLE,
    severity='critical',
    category=EscalationCategory.INFRA_ISSUE,
    suggested_action='investigate_startup_crash_loop',
)


def sentinel_task_id(step_name: str) -> str:
    return f'{STARTUP_STEP_SENTINEL_PREFIX}{step_name}'


class StepStatus(StrEnum):
    OK = 'ok'
    FAILED = 'failed'
    PARTIAL = 'partial'
    QUARANTINED = 'quarantined'


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

    @classmethod
    def quarantined(cls, step: str) -> StepOutcome:
        return cls(step, StepStatus.QUARANTINED)


@dataclass(frozen=True)
class StepInFlight:
    step: str
    started_at: datetime


class StartupFailureStreak(BaseModel):
    """The step (or crash site) that failed on each of the last *consecutive* starts."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    step: str = Field(pattern=f'^{_STEP_NAME_PATTERN.pattern}$')
    exception_type: str
    message: str
    consecutive: int = Field(ge=1)
    first_seen: datetime
    last_seen: datetime

    @classmethod
    def following(
        cls,
        prior: StartupFailureStreak | None,
        *,
        step: str,
        exception_type: str,
        message: str,
        at: datetime,
    ) -> StartupFailureStreak:
        extends = prior is not None and prior.step == step
        return cls(
            step=step,
            exception_type=exception_type,
            message=message,
            consecutive=prior.consecutive + 1 if extends else 1,
            first_seen=prior.first_seen if extends else at,
            last_seen=at,
        )

    @property
    def reached_threshold(self) -> bool:
        return self.consecutive >= CRASH_LOOP_THRESHOLD


class StartupJournal:
    """This start's step in flight, over the durable failure streak the previous starts left."""

    def __init__(self, path: Path, inherited: StartupFailureStreak | None = None) -> None:
        self._path = path
        self._inherited = inherited
        self._in_flight: StepInFlight | None = None
        self._last_returned: str | None = None
        self._crash_loop_paged = False

    @classmethod
    def load(cls, path: Path) -> StartupJournal:
        return cls(path, _read_streak(path))

    @property
    def path(self) -> Path:
        return self._path

    @property
    def inherited(self) -> StartupFailureStreak | None:
        return self._inherited

    def is_quarantined(self, step: StartupStep) -> bool:
        streak = self._inherited
        return (
            not step.fatal and streak is not None
            and streak.step == step.name and streak.reached_threshold
        )

    def record_in_flight(self, step: str, started_at: datetime) -> None:
        self._in_flight = StepInFlight(step, started_at)

    def record_returned(self, step: str) -> None:
        self._in_flight = None
        self._last_returned = step

    @property
    def in_flight(self) -> StepInFlight | None:
        return self._in_flight

    @property
    def crash_site(self) -> str:
        if self._in_flight is not None:
            return self._in_flight.step
        if self._last_returned is not None:
            return f'after-{self._last_returned}'
        return 'before-first-step'

    def record_failure(self, step: str, exception_type: str, message: str) -> StartupFailureStreak:
        streak = StartupFailureStreak.following(
            self._inherited, step=step, exception_type=exception_type, message=message,
            at=datetime.now(UTC),
        )
        try:
            atomic_write_text(self._path, streak.model_dump_json(), mkdir=True)
        except OSError as exc:
            logger.warning(
                'startup failure journal %s: recording %s failed: %s: %s',
                self._path, step, type(exc).__name__, exc,
            )
        return streak

    def record_crash(self, exc: BaseException) -> str:
        site = self.crash_site
        self.record_failure(site, type(exc).__name__, str(exc))
        return site

    @property
    def crash_loop_paged(self) -> bool:
        return self._crash_loop_paged

    def mark_crash_loop_paged(self) -> None:
        self._crash_loop_paged = True

    def clear(self) -> None:
        try:
            self._path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(
                'startup failure journal %s: clearing failed: %s: %s',
                self._path, type(exc).__name__, exc,
            )


def page_crash_loop(
    journal: StartupJournal,
    escalation_queue: EscalationQueue | None,
    *,
    filing_claimant_run_id: str | None,
) -> None:
    """File or refresh the born-at-L2 crash-loop page once per start, if the inherited streak reached K."""
    streak = journal.inherited
    if escalation_queue is None or journal.crash_loop_paged:
        return
    if streak is None or not streak.reached_threshold:
        return
    try:
        file_or_refresh(
            escalation_queue,
            _CRASH_LOOP_ALARM,
            summary=_crash_loop_summary(streak)[:_SUMMARY_LIMIT],
            detail=_crash_loop_detail(streak, journal.path),
            filing_claimant_run_id=filing_claimant_run_id,
        )
    except Exception:
        logger.warning(
            'startup crash loop at %s: filing its L2 escalation failed', streak.step, exc_info=True,
        )
        return
    journal.mark_crash_loop_paged()


def _crash_loop_summary(streak: StartupFailureStreak) -> str:
    return (
        f'orchestrator startup crash loop: {streak.step} failed on {streak.consecutive} '
        f'consecutive starts ({streak.exception_type}: {streak.message})'
    )


def _crash_loop_detail(streak: StartupFailureStreak, journal_path: Path) -> str:
    return '\n'.join((
        streak.model_dump_json(indent=2),
        '',
        'On this start a non-fatal step named here is skipped; a fatal step, or a crash '
        'site outside any step (before-first-step, after-<step>), still runs. The '
        'finish_startup of a start that comes up clears the journal; this record stays '
        'until resolved.',
        '',
        'Investigate:',
        '- journalctl --user -u orchestrator-<project>.service (the traceback)',
        '- runs.db: SELECT run_id, started_at, startup_completed_at, exit_reason FROM runs '
        'ORDER BY rowid DESC LIMIT 10',
        f'- the streak journal: {journal_path}',
    ))


def _read_streak(path: Path) -> StartupFailureStreak | None:
    try:
        return StartupFailureStreak.model_validate_json(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning(
            'startup failure journal %s is unreadable, inheriting no streak: %s: %s',
            path, type(exc).__name__, exc,
        )
        return None


async def run_startup_step(
    step: StartupStep,
    *,
    journal: StartupJournal,
    event_store: EventStore | None,
    escalation_queue: EscalationQueue | None,
    filing_claimant_run_id: str | None,
) -> StepOutcome:
    """Run *step* unless quarantined; report a non-ok outcome; re-raise only a fatal step's exception.

    A raise leaves the step in flight on *journal*, so the outer handler can blame it.
    """
    streak = journal.inherited
    if streak is not None and journal.is_quarantined(step):
        return _skip_quarantined(step.name, streak)
    journal.record_in_flight(step.name, datetime.now(UTC))
    try:
        result = await _run_body(step)
    except Exception as exc:
        outcome = StepOutcome.failed(step.name, exc)
        logger.error(
            'startup step %s failed%s: %s: %s',
            step.name, ' (fatal, startup refused)' if step.fatal else '',
            outcome.exception_type, outcome.message, exc_info=exc,
        )
        journal.record_failure(step.name, type(exc).__name__, str(exc))
        _report(outcome, step.fatal, event_store, escalation_queue, filing_claimant_run_id)
        if step.fatal:
            raise
        journal.record_returned(step.name)
        return outcome
    outcome = _outcome_of_result(step.name, result)
    if outcome.status is StepStatus.PARTIAL:
        logger.error(
            'startup step %s partial: %d entr(ies) failed: %s',
            step.name, len(outcome.failed_entries), _entry_names(outcome),
        )
        _report(outcome, step.fatal, event_store, escalation_queue, filing_claimant_run_id)
    journal.record_returned(step.name)
    return outcome


async def _run_body(step: StartupStep) -> object:
    result = step.run()
    if inspect.isawaitable(result):
        result = await result
    return result


def _skip_quarantined(step: str, streak: StartupFailureStreak) -> StepOutcome:
    logger.warning(
        'startup step %s quarantined: it failed on %d consecutive starts (last %s: %s); '
        'skipping it on this start',
        step, streak.consecutive, streak.exception_type, streak.message,
    )
    return StepOutcome.quarantined(step)


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
        agent_role=_AGENT_ROLE,
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
