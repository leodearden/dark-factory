"""Measure the fleet for the return brief: what landed, what is stuck, what it cost, what closed itself, how the trials read.

The one invariant (heuristic 10): nothing leaves this module without
``measured_at``, ``source`` and ``status``. ``Measurement`` refuses to exist
without them, and refuses a flagged status that states no shortfall, so an
unavailable store is a flagged zero, never a bare one. That is the mechanism
behind the return brief's "every claim re-checked at generation time and
timestamped": an unstamped claim is unrepresentable.

The runs.db figures are ``orchestrator.digest``'s own readers, called directly
(they import cleanly under the scripts env). They fail open to a zero that
cannot be told from a true zero, so each call is preceded by this module's
probe of the store and of the tables its query reads, and followed by a look
at what the reader swallowed: every failure it answers with a zero it first
logs at WARNING with the exception attached, and that exception flags the
figure ``unreadable``.

Read-only: every store is opened ``mode=ro`` and every queue is read through
the unlocked scan helpers, so nothing here creates a file or a directory.
"""
from __future__ import annotations

import logging
import sqlite3
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Generic, Literal, TypeVar

from _task_db_scan import (
    TABLE_NAMES_SQL,
    TaskDbProblem,
    TaskDbUnreadable,
    connect_ro,
    tasks_db_path,
)
from escalation.models import Escalation
from escalation.queue import iter_all_escalation_paths
from escalation.shadow_ruling import AgreementReport, ClassAgreement, agreement_report
from orchestrator.digest import ModelRoleRow, count_done_in_window, model_role_rollup
from orchestrator.digest import logger as digest_log
from orchestrator.session_registry import (
    DecisionRecord,
    DecisionState,
    decisions_dir,
    normalize_escalations_dir,
    normalize_project_token,
)
from sitting import inventory, payloads
from sitting.inventory import QUEUE_SUBDIRS, Shortfall
from sitting.payloads import RejectedMarker

Status = Literal['ok', 'source_missing', 'unreadable']
STATUSES: tuple[Status, ...] = ('ok', 'source_missing', 'unreadable')

RUNS_DB = Path('data', 'orchestrator', 'runs.db')
LANDED_TABLES = ('events',)
SPEND_TABLES = ('events', 'invocations', 'task_results')

NO_OPEN_ESCALATION = 'blocked with no open escalation'
UNKNOWN_OPEN_ESCALATIONS = 'open escalations unknown: no queue of this project was read'

SHADOW_ONLY = (
    'No standing-policy class is adopted (docs/escalation-standing-policy.md), so nothing was ruled '
    'autonomously: every ruling counted here is a shadow-only proposal, a measurement never applied.'
)

T = TypeVar('T')
Refusal = tuple[Status, Shortfall]
TrialOutcome = Literal['agreed', 'disagreed', 'no_lean', 'unanswered', 'rejected']


@dataclass(frozen=True)
class Window:
    """The measured interval, both bounds inclusive, as digest and ``agreement_report`` read theirs."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError('window bounds must be timezone-aware')
        if self.start > self.end:
            raise ValueError(f'window start {self.start.isoformat()} is after its end {self.end.isoformat()}')

    @classmethod
    def trailing(cls, now: datetime, *, days: float) -> Window:
        return cls(now - timedelta(days=days), now)

    def iso(self) -> tuple[str, str]:
        """Both bounds as the UTC ISO text runs.db timestamps are compared against."""
        return self.start.astimezone(UTC).isoformat(), self.end.astimezone(UTC).isoformat()

    def holds(self, instant: datetime | None) -> bool:
        return instant is not None and self.start <= instant <= self.end


@dataclass(frozen=True)
class Measurement(Generic[T]):
    """One stamped figure. ``shortfalls`` states why a flagged one is flagged, and any partial loss on an ok one."""

    value: T
    measured_at: str
    source: str
    status: Status
    shortfalls: tuple[Shortfall, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f'status must be one of {STATUSES}, got {self.status!r}')
        for name in ('measured_at', 'source'):
            if not getattr(self, name):
                raise ValueError(f'{name} is required: an unstamped measurement is unrepresentable')
        if self.status != 'ok' and not self.shortfalls:
            raise ValueError(f'a {self.status} measurement must state its shortfall')

    @property
    def ok(self) -> bool:
        return self.status == 'ok'


@dataclass(frozen=True)
class OpenEscalation:
    """A pending record; an escalation id is unique only within its ``queue_dir``."""

    escalation_id: str
    queue_dir: str
    category: str
    level: int
    age_days: float | None

    def describe(self) -> str:
        age = 'age unknown' if self.age_days is None else f'{self.age_days:.1f}d old'
        root = inventory.queue_project_root(self.queue_dir)
        queue = self.queue_dir if root is None else str(Path(self.queue_dir).relative_to(root))
        return f'{self.escalation_id} ({self.category}, L{self.level}, {age}, {queue})'


@dataclass(frozen=True)
class StuckRow:
    """A blocked task; ``open_escalations`` is None when no queue of its project could be read."""

    task_id: str
    title: str
    open_escalations: tuple[OpenEscalation, ...] | None

    @property
    def reason(self) -> str:
        if self.open_escalations is None:
            return UNKNOWN_OPEN_ESCALATIONS
        if not self.open_escalations:
            return NO_OPEN_ESCALATION
        return '; '.join(esc.describe() for esc in self.open_escalations)


@dataclass(frozen=True)
class AutonomousClose:
    decision_id: str
    escalation_id: str | None
    project: str
    state: str
    filed_at: str
    closed_at: str
    evidence: str


@dataclass(frozen=True)
class AutonomousCloses:
    """Evidence-carrying closes as three populations.

    ``in_window``: closed inside the window. ``undated``: no parsable
    ``closed_at``, so when they closed is unknown and they are shown rather
    than dropped. ``lifetime``: every evidence-carrying close, dated or not,
    the per-50 denominator of the wrong-close kill criterion.
    """

    in_window: tuple[AutonomousClose, ...]
    undated: tuple[AutonomousClose, ...]
    lifetime: int


@dataclass(frozen=True)
class StandingPolicyRulings:
    """The per-queue shadow agreement reports folded into one over the same window."""

    report: AgreementReport

    @property
    def statement(self) -> str:
        """Always shadow-only: no adopted class has a machine-readable home to read.

        ``escalation.shadow_ruling``'s promotion path replaces its codec on
        adoption rather than extending it, so an adopted class will reach this
        module as a change to what it reads, not as a value to branch on.
        """
        return SHADOW_ONLY


@dataclass(frozen=True)
class TurnsDistribution:
    """``resolution_turns`` over the L2s resolved in the window: the sitting's attention instrument."""

    resolved: int
    with_turns: int
    median: float | None
    total: int


@dataclass(frozen=True)
class PreparerTrial:
    """How often Leo's answer matched the prepared recommendation, over records resolved in the window.

    ``n`` counts records whose ``x_prepared`` marker is readable. The rate's
    denominator is the subset that recommended an option and whose last
    ``x_agreed`` could agree or not; a no-lean and an unanswered preparation
    are counted beside it, never in it. ``rejected`` counts records whose last
    marker of either kind could not be read, excluded from every other count.
    """

    n: int
    numerator: int
    denominator: int
    no_lean: int
    unanswered: int
    rejected: int
    resolution_turns: TurnsDistribution

    @property
    def rate(self) -> float | None:
        return self.numerator / self.denominator if self.denominator else None


@dataclass(frozen=True)
class ProjectState:
    project: str
    root: str
    landed: Measurement[int]
    stuck: Measurement[tuple[StuckRow, ...]]
    spend: Measurement[tuple[ModelRoleRow, ...]]
    standing_policy: Measurement[StandingPolicyRulings]
    trial: Measurement[PreparerTrial]


def runs_db_path(project_root: Path | str) -> Path:
    return Path(project_root) / RUNS_DB


def project_queue_dirs(project_root: Path | str) -> tuple[str, ...]:
    return tuple(normalize_escalations_dir(Path(project_root) / subdir) for subdir in QUEUE_SUBDIRS)


def landed(runs_db: Path | str, window: Window, *, now: datetime) -> Measurement[int]:
    """Tasks completed ``done`` in *window*, per ``orchestrator.digest.count_done_in_window``."""
    return _from_runs_db(Path(runs_db), LANDED_TABLES, 0, lambda db: count_done_in_window(db, *window.iso()), now)


def spend_and_cap_hits(runs_db: Path | str, window: Window, *, now: datetime) -> Measurement[tuple[ModelRoleRow, ...]]:
    """Per-(model, role) cost, cap-hit rate and $/done, per ``orchestrator.digest.model_role_rollup``."""
    return _from_runs_db(
        Path(runs_db), SPEND_TABLES, (), lambda db: tuple(model_role_rollup(db, *window.iso()).rows), now,
    )


def stuck(
    project_root: Path | str,
    escalation_index: Mapping[str, Mapping[str, Path]],
    *,
    now: datetime,
) -> Measurement[tuple[StuckRow, ...]]:
    """One row per blocked task, reasoned from the pending records of this project's queues in *escalation_index*."""
    stamp = _stamp(now)
    db = tasks_db_path(str(project_root))
    blocked = _blocked_tasks(db)
    if isinstance(blocked, tuple):
        status, shortfall = blocked
        return Measurement((), stamp, str(db), status, (shortfall,))
    queues = [queue for queue in project_queue_dirs(project_root) if queue in escalation_index]
    if not queues:
        unreasoned = tuple(StuckRow(task_id, title, None) for task_id, title in blocked.items())
        missing = Shortfall('escalation_queue', str(project_root), 'no queue of this project is in the inventory index')
        return Measurement(unreasoned, stamp, str(db), 'ok', (missing,))
    open_by_task, shortfalls = _open_escalations(queues, escalation_index, set(blocked), now)
    rows = tuple(StuckRow(task_id, title, open_by_task.get(task_id, ())) for task_id, title in blocked.items())
    return Measurement(rows, stamp, str(db), 'ok', tuple(shortfalls))


def autonomous_closes(decisions_root: Path | str, window: Window, *, now: datetime) -> Measurement[AutonomousCloses]:
    """Closed decisions carrying ``closing_evidence``, quoted verbatim, windowed on ``closed_at``.

    ``closed_at`` is the close instant
    ``orchestrator/src/orchestrator/session_registry.py::close_decision_with_evidence``
    stamps; ``filed_at`` would miss a close of a gate filed before the window.
    A close with no parsable ``closed_at`` is ``undated``: shown, never dropped.
    """
    stamp = _stamp(now)
    registry = decisions_dir(decisions_root)
    if not registry.is_dir():
        missing = Shortfall('decision_registry', str(registry), 'no such directory')
        return Measurement(AutonomousCloses((), (), 0), stamp, str(registry), 'source_missing', (missing,))
    records, shortfalls = inventory.read_decisions(decisions_root)
    closed = [record for record in records if record.state != DecisionState.OPEN and record.closing_evidence]
    dated = [(record, instant) for record in closed if (instant := inventory.parse_stamp(record.closed_at))]
    in_window = sorted(
        (_autonomous_close(record) for record, instant in dated if window.holds(instant)),
        key=lambda close: (close.closed_at, close.decision_id),
    )
    undated = sorted(
        (_autonomous_close(record) for record in closed if inventory.parse_stamp(record.closed_at) is None),
        key=lambda close: close.decision_id,
    )
    closes = AutonomousCloses(tuple(in_window), tuple(undated), len(closed))
    return Measurement(closes, stamp, str(registry), 'ok', tuple(shortfalls))


def standing_policy_rulings(
    queue_dirs: Iterable[str], window: Window, *, now: datetime,
) -> Measurement[StandingPolicyRulings]:
    """``escalation.shadow_ruling.agreement_report`` per queue, folded into one report."""
    def measure(queues: Sequence[str]) -> tuple[StandingPolicyRulings, tuple[Shortfall, ...]]:
        reports = [agreement_report(queue, since=window.start, until=window.end) for queue in queues]
        return StandingPolicyRulings(_fold_reports(reports, window)), ()

    return _over_queues(queue_dirs, StandingPolicyRulings(_fold_reports((), window)), measure, now)


def preparer_trial(queue_dirs: Iterable[str], window: Window, *, now: datetime) -> Measurement[PreparerTrial]:
    """The recommend-only trial's agreement and attention instruments, over records resolved in *window*."""
    def measure(queues: Sequence[str]) -> tuple[PreparerTrial, tuple[Shortfall, ...]]:
        resolved, shortfalls = _resolved_in_window(queues, window)
        return _trial(resolved), shortfalls

    return _over_queues(queue_dirs, _trial(()), measure, now)


def measure_projects(
    project_roots: Iterable[Path | str],
    escalation_index: Mapping[str, Mapping[str, Path]],
    *,
    window: Window,
    now: datetime,
) -> Mapping[str, ProjectState]:
    """Every per-project measurement, keyed by canonical project token; a root with no stores is flagged, never dropped."""
    states: dict[str, ProjectState] = {}
    for root in project_roots:
        project = normalize_project_token(Path(root).name)
        if project in states:
            raise ValueError(f'{root} and {states[project].root} both fold to project {project!r}')
        queues = _measured_queues(root)
        runs_db = runs_db_path(root)
        states[project] = ProjectState(
            project=project,
            root=str(root),
            landed=landed(runs_db, window, now=now),
            stuck=stuck(root, escalation_index, now=now),
            spend=spend_and_cap_hits(runs_db, window, now=now),
            standing_policy=standing_policy_rulings(queues, window, now=now),
            trial=preparer_trial(queues, window, now=now),
        )
    return MappingProxyType(states)


def _stamp(now: datetime) -> str:
    if now.tzinfo is None:
        raise ValueError('now must be timezone-aware')
    return now.isoformat()


def _refusal(store: str, path: Path, exc: TaskDbUnreadable | sqlite3.Error) -> Refusal:
    if isinstance(exc, TaskDbUnreadable):
        status: Status = 'source_missing' if exc.reason is TaskDbProblem.ABSENT else 'unreadable'
        return status, Shortfall(store, str(path), exc.reason.value)
    return 'unreadable', Shortfall(store, str(path), str(exc))


def _from_runs_db(runs_db: Path, tables: Sequence[str], empty: T, read: Callable[[Path], T], now: datetime) -> Measurement[T]:
    stamp = _stamp(now)
    refusal = _probe_runs_db(runs_db, tables)
    if refusal is None:
        with _swallowed_by_digest() as swallowed:
            value = read(runs_db)
        refusal = _read_refusal(runs_db, swallowed)
        if refusal is None:
            return Measurement(value, stamp, str(runs_db), 'ok')
    status, shortfall = refusal
    return Measurement(empty, stamp, str(runs_db), status, (shortfall,))


class _ExceptionCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.caught: list[BaseException] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.exc_info and record.exc_info[1] is not None:
            self.caught.append(record.exc_info[1])


@contextmanager
def _swallowed_by_digest() -> Iterator[list[BaseException]]:
    """The exceptions ``orchestrator.digest``'s readers catch and log while the block runs.

    When logging is disabled outright those logs never happen, so the list
    starts with a note saying the read cannot be vouched for.
    """
    collector = _ExceptionCollector()
    saved_level = digest_log.level
    digest_log.setLevel(min(digest_log.getEffectiveLevel(), logging.WARNING))
    digest_log.addHandler(collector)
    if not digest_log.isEnabledFor(logging.WARNING):
        collector.caught.append(RuntimeError('logging is disabled, so a failed read cannot be told from a zero'))
    try:
        yield collector.caught
    finally:
        digest_log.removeHandler(collector)
        digest_log.setLevel(saved_level)


def _read_refusal(runs_db: Path, swallowed: Sequence[BaseException]) -> Refusal | None:
    """Why the digest's answer cannot be stood behind, or None when it can."""
    if not runs_db.exists():
        return 'source_missing', Shortfall('runs_db', str(runs_db), 'vanished during the read')
    if swallowed:
        return 'unreadable', Shortfall('runs_db', str(runs_db),
                                       '; '.join(f'{type(exc).__name__}: {exc}' for exc in swallowed))
    return None


def _probe_runs_db(runs_db: Path, tables: Sequence[str]) -> Refusal | None:
    """Why *runs_db* cannot answer a query over *tables*, or None when it can."""
    try:
        conn = connect_ro(runs_db)
    except (TaskDbUnreadable, sqlite3.Error) as exc:
        return _refusal('runs_db', runs_db, exc)
    try:
        present = {name for (name,) in conn.execute(TABLE_NAMES_SQL)}
    except sqlite3.Error as exc:
        return _refusal('runs_db', runs_db, exc)
    finally:
        conn.close()
    missing = [table for table in tables if table not in present]
    if missing:
        return 'unreadable', Shortfall('runs_db', str(runs_db), f'no {", ".join(missing)} table')
    return None


def _blocked_tasks(db: Path) -> dict[str, str] | Refusal:
    """Blocked task id -> title, in id order, or why tasks.db could not say."""
    try:
        conn = connect_ro(db)
    except (TaskDbUnreadable, sqlite3.Error) as exc:
        return _refusal('tasks_db', db, exc)
    try:
        rows = conn.execute(
            "SELECT id, title FROM tasks WHERE tag = 'master' AND status = 'blocked' ORDER BY id",
        ).fetchall()
    except sqlite3.Error as exc:
        return _refusal('tasks_db', db, exc)
    finally:
        conn.close()
    return {str(task_id): title for task_id, title in rows}


def _open_escalations(
    queues: Iterable[str],
    escalation_index: Mapping[str, Mapping[str, Path]],
    task_ids: set[str],
    now: datetime,
) -> tuple[dict[str, tuple[OpenEscalation, ...]], list[Shortfall]]:
    found: dict[str, list[OpenEscalation]] = defaultdict(list)
    shortfalls: list[Shortfall] = []
    for queue in queues:
        for path in escalation_index[queue].values():
            if path.parent != Path(queue):
                continue
            esc, problem = inventory.read_escalation(path)
            if problem is not None:
                shortfalls.append(problem)
            if esc is not None and esc.status == 'pending' and esc.task_id in task_ids:
                found[esc.task_id].append(OpenEscalation(
                    esc.id, queue, esc.category, esc.level, inventory.age_days(esc.timestamp, now),
                ))
    oldest_first = {
        task_id: tuple(sorted(escs, key=lambda e: (-(e.age_days or 0.0), e.escalation_id)))
        for task_id, escs in found.items()
    }
    return oldest_first, shortfalls


def _autonomous_close(record: DecisionRecord) -> AutonomousClose:
    return AutonomousClose(
        decision_id=record.id,
        escalation_id=record.escalation_id,
        project=normalize_project_token(record.project),
        state=str(record.state),
        filed_at=record.filed_at,
        closed_at=record.closed_at,
        evidence=record.closing_evidence,
    )


def _measured_queues(project_root: Path | str) -> tuple[str, ...]:
    """The project's existing queues; when it has none, all of them, so each absence is stated."""
    candidates = project_queue_dirs(project_root)
    return tuple(queue for queue in candidates if Path(queue).is_dir()) or candidates


def _over_queues(
    queue_dirs: Iterable[str],
    empty: T,
    measure: Callable[[Sequence[str]], tuple[T, tuple[Shortfall, ...]]],
    now: datetime,
) -> Measurement[T]:
    stamp = _stamp(now)
    given = tuple(dict.fromkeys(normalize_escalations_dir(queue) for queue in queue_dirs))
    present = [queue for queue in given if Path(queue).is_dir()]
    absent = tuple(Shortfall('escalation_queue', queue, 'no such queue directory') for queue in given
                   if queue not in present)
    source = ', '.join(given) or 'escalation queues'
    if not present:
        stated = absent or (Shortfall('escalation_queue', '', 'no queue was given'),)
        return Measurement(empty, stamp, source, 'source_missing', stated)
    value, shortfalls = measure(present)
    return Measurement(value, stamp, source, 'ok', absent + shortfalls)


def _fold_reports(reports: Sequence[AgreementReport], window: Window) -> AgreementReport:
    """Sum every count of *reports*, class by class; a count added to either type is summed with no edit here."""
    by_class: dict[str, list[ClassAgreement]] = defaultdict(list)
    for report in reports:
        for row in report.classes:
            by_class[row.ruling_class].append(row)
    class_counts = [f.name for f in fields(ClassAgreement) if f.name != 'ruling_class']
    report_counts = [f.name for f in fields(AgreementReport)
                     if f.name not in {'since', 'until', 'classes', 'resolver_tiers'}]
    tiers: Counter[str] = Counter()
    for report in reports:
        tiers.update(report.resolver_tiers)
    return AgreementReport(
        since=window.start,
        until=window.end,
        classes=tuple(
            ClassAgreement(ruling_class=slug, **{name: sum(getattr(row, name) for row in rows) for name in class_counts})
            for slug, rows in sorted(by_class.items())
        ),
        resolver_tiers=MappingProxyType(dict(sorted(tiers.items()))),
        **{name: sum(getattr(report, name) for report in reports) for name in report_counts},
    )


def _resolved_in_window(queues: Sequence[str], window: Window) -> tuple[list[Escalation], tuple[Shortfall, ...]]:
    resolved: list[Escalation] = []
    shortfalls: list[Shortfall] = []
    for queue in queues:
        for path in iter_all_escalation_paths(Path(queue)):
            esc, problem = inventory.read_escalation(path)
            if problem is not None:
                shortfalls.append(problem)
            if esc is not None and esc.status != 'pending' and window.holds(inventory.parse_stamp(esc.resolved_at)):
                resolved.append(esc)
    return resolved, tuple(shortfalls)


def _trial(resolved: Iterable[Escalation]) -> PreparerTrial:
    outcomes: Counter[TrialOutcome] = Counter()
    level_two: list[Escalation] = []
    for esc in resolved:
        outcome = _trial_outcome(esc.triage_note if isinstance(esc.triage_note, str) else '')
        if outcome is not None:
            outcomes[outcome] += 1
        if esc.level == 2:
            level_two.append(esc)
    counted = sum(outcomes.values()) - outcomes['rejected']
    return PreparerTrial(
        n=counted,
        numerator=outcomes['agreed'],
        denominator=outcomes['agreed'] + outcomes['disagreed'],
        no_lean=outcomes['no_lean'],
        unanswered=outcomes['unanswered'],
        rejected=outcomes['rejected'],
        resolution_turns=_turns(level_two),
    )


def _trial_outcome(note: str) -> TrialOutcome | None:
    """What one record contributes to the trial: None when it was never prepared."""
    prepared = payloads.parse_prepared_marker(note)
    if prepared is None:
        return None
    agreed = payloads.parse_agreed_marker(note)
    if isinstance(prepared, RejectedMarker) or isinstance(agreed, RejectedMarker):
        return 'rejected'
    if agreed is None:
        return 'unanswered'
    if not prepared.recommendation or agreed.agreed is None:
        return 'no_lean'
    return 'agreed' if agreed.agreed else 'disagreed'


def _turns(level_two: Sequence[Escalation]) -> TurnsDistribution:
    turns = [esc.resolution_turns for esc in level_two
             if isinstance(esc.resolution_turns, int) and not isinstance(esc.resolution_turns, bool)]
    return TurnsDistribution(
        resolved=len(level_two),
        with_turns=len(turns),
        median=float(statistics.median(turns)) if turns else None,
        total=sum(turns),
    )
