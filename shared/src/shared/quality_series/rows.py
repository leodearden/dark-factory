"""One LandingRow per landing: what happened to it, attributed over (previous landing, landing].

Contract: plans/quality-dashboard-prd.md §Contract "Per-landing row". A source-bearing field is None
when its source has no row in this store yet or the landing precedes that source's first row
(SourceEpochs), never 0. Unit values (rejections, verifies, flakes) are combined over a verify
unit's members and copied to each; reviews stay per landing.
"""

from __future__ import annotations

import bisect
import json
import os
import sqlite3
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from itertools import groupby
from pathlib import Path
from typing import Any, Generic, TypeVar

from shared.quality_series.git_lines import (
    GitObjectMissing,
    GitTimeout,
    LandedLines,
    ModuleLines,
    landed_lines,
)
from shared.quality_series.landings import Landing, parse_utc

__all__ = [
    'FIRST_UNIT_GIT_FLOOR_S',
    'IMPLEMENTER_CLASS_ROLES',
    'CharterClass',
    'CharterClassSource',
    'Cost',
    'LandingRow',
    'Outcomes',
    'QualityRowInvariantError',
    'ReviewRoundsSource',
    'TURNS_CAP_REASON',
    'SourceEpochs',
    'UnknownCharterClass',
    'module_allocation',
    'rows_for',
    'unit_heads',
]

T = TypeVar('T')


class QualityRowInvariantError(ValueError):
    """A row field combination the contract rules out."""


def _invariant(condition: bool, invariant: str, **values: object) -> None:
    if not condition:
        shown = ', '.join(f'{name}={value!r}' for name, value in values.items())
        raise QualityRowInvariantError(f'{invariant} ({shown})')


@dataclass(frozen=True)
class SourceEpochs:
    """The first row's time of each source in this runs.db; None when it has none."""

    rejections: datetime | None
    merge_verify: datetime | None
    review_completed: datetime | None
    flake_occurrence: datetime | None
    invocation_turns: datetime | None
    invocations: datetime | None


_EPOCH_SQL = {
    # The first categorised block of any category, so a project that never had a post-merge
    # rejection reads 0 rejections per landing, not unknown.
    'rejections': "SELECT MIN(timestamp) FROM events WHERE event_type = 'merge_blocked'"
                  " AND json_extract(data, '$.category') IS NOT NULL",
    'merge_verify': "SELECT MIN(timestamp) FROM events WHERE event_type = 'merge_verify'",
    'review_completed': "SELECT MIN(timestamp) FROM events WHERE event_type = 'review_completed'",
    'flake_occurrence': 'SELECT MIN(observed_at) FROM flake_occurrence',
    'invocation_turns': "SELECT MIN(timestamp) FROM events WHERE event_type = 'invocation_end'"
                        " AND json_extract(data, '$.turns') IS NOT NULL",
    'invocations': 'SELECT MIN(started_at) FROM invocations',
}


def _first_time(conn: sqlite3.Connection, sql: str) -> datetime | None:
    first = conn.execute(sql).fetchone()[0]
    return None if first is None else parse_utc(first)


def _read_epochs(conn: sqlite3.Connection) -> SourceEpochs:
    return SourceEpochs(**{source: _first_time(conn, sql) for source, sql in _EPOCH_SQL.items()})


def _covers(epoch: datetime | None, landing: Landing) -> bool:
    return epoch is not None and landing.landed_at >= epoch


@dataclass(frozen=True)
class Outcomes:
    rejected: bool | None
    rejection_events: int | None
    reviewed: bool | None
    review_blocking: bool | None
    flake_confirmed: bool | None
    flake_real_red: bool | None
    flake_rows: int | None
    reverify_count: int | None
    verify_failed_rows: int | None

    def __post_init__(self) -> None:
        _invariant((self.rejected is None) == (self.rejection_events is None)
                   and (self.rejection_events is None
                        or self.rejected == (self.rejection_events > 0)),
                   'rejected is None iff rejection_events is None, else rejection_events > 0',
                   rejected=self.rejected, rejection_events=self.rejection_events)
        _invariant(self.review_blocking is None or self.reviewed is True,
                   'review_blocking is known only for a reviewed landing',
                   reviewed=self.reviewed, review_blocking=self.review_blocking)
        _invariant(self.reviewed is not True or self.review_blocking is not None,
                   'a reviewed landing says whether a review blocked',
                   reviewed=self.reviewed, review_blocking=self.review_blocking)
        flakes = (self.flake_confirmed, self.flake_real_red, self.flake_rows)
        _invariant(len({value is None for value in flakes}) == 1,
                   'the flake fields are None together', flake_confirmed=self.flake_confirmed,
                   flake_real_red=self.flake_real_red, flake_rows=self.flake_rows)
        _invariant((self.reverify_count is None) == (self.verify_failed_rows is None),
                   'reverify_count and verify_failed_rows are None together',
                   reverify_count=self.reverify_count, verify_failed_rows=self.verify_failed_rows)


IMPLEMENTER_CLASS_ROLES = ('implementer', 'simple_task')


class ReviewRoundsSource(StrEnum):
    REVIEW_COMPLETED = 'review_completed'
    DISK_COUNTER = 'disk_counter'


class CharterClass(StrEnum):
    """Why a task's charter blocked it; task ζ's orchestrator writer imports this vocabulary."""

    UNDER_SPECIFIED = 'under_specified'
    STALE_CHARTER = 'stale_charter'
    PLAN_FILES_UNTOUCHED = 'plan_files_untouched'


class CharterClassSource(StrEnum):
    CHARTER_CLASS = 'charter_class'
    REGEX_BACKFILL = 'regex_backfill'


class UnknownCharterClass(ValueError):
    """task_results.charter_class holds a value outside CharterClass."""


_CHARTER_BACKFILL_PREFIXES: tuple[tuple[str, CharterClass], ...] = (
    # orchestrator/src/orchestrator/workflow.py::TaskWorkflow._handle_unactionable_task_report
    ('Architect reported task unactionable', CharterClass.UNDER_SPECIFIED),
    # orchestrator/src/orchestrator/workflow.py::TaskWorkflow._handle_false_premise_report
    ('Architect reported false RED-test premise', CharterClass.UNDER_SPECIFIED),
    # orchestrator/src/orchestrator/workflow.py::TaskWorkflow._handle_already_done_report
    ('Architect reported task already done', CharterClass.STALE_CHARTER),
    # orchestrator/src/orchestrator/merge_lane/gates.py::PLAN_FILES_NOT_TOUCHED_REASON_PREFIX
    ('Plan files not touched by branch', CharterClass.PLAN_FILES_UNTOUCHED),
)
"""The pre-ζ backfill: block_reason prefixes, the only string matching in this package."""

TURNS_CAP_REASON = 'turns'
"""shared/src/shared/cost_store.py::CapReason.TURNS, restated: that module is not stdlib-only.

shared/tests/test_quality_series_rows.py asserts the two stay equal.
"""


@dataclass(frozen=True)
class Cost:
    """What the landing cost, over invocations started and task_results completed in its window.

    execute_iterations sums the task_results rows of the dispatches that produced the landing
    (see _landing_dispatches), so it includes the landing dispatch's own row, written after it.
    """

    all_role_usd: float | None
    by_role: tuple[tuple[str, float], ...]
    implementer_invocations: int | None
    execute_iterations: int | None
    turns: int | None
    turns_capped: bool
    code_lines: int | None
    src_lines: int | None
    test_lines: int | None
    review_rounds: int | None
    review_rounds_source: ReviewRoundsSource | None
    charter_class: CharterClass | None
    charter_class_source: CharterClassSource | None

    def __post_init__(self) -> None:
        _invariant((self.all_role_usd is None) == (self.implementer_invocations is None),
                   'all_role_usd and implementer_invocations are None together',
                   all_role_usd=self.all_role_usd,
                   implementer_invocations=self.implementer_invocations)
        _invariant((self.all_role_usd is None) == (not self.by_role),
                   'by_role is empty exactly when all_role_usd is None',
                   all_role_usd=self.all_role_usd, by_role=self.by_role)
        _invariant(self.turns is not None or not self.turns_capped,
                   'a turns cap needs a turns count', turns=self.turns,
                   turns_capped=self.turns_capped)
        lines = (self.code_lines, self.src_lines, self.test_lines)
        _invariant(len({value is None for value in lines}) == 1,
                   'the three line fields are None together', code_lines=self.code_lines,
                   src_lines=self.src_lines, test_lines=self.test_lines)
        _invariant((self.review_rounds is None) == (self.review_rounds_source is None),
                   'review_rounds and its source are None together',
                   review_rounds=self.review_rounds,
                   review_rounds_source=self.review_rounds_source)
        _invariant((self.charter_class is None) == (self.charter_class_source is None),
                   'charter_class and its source are None together',
                   charter_class=self.charter_class,
                   charter_class_source=self.charter_class_source)


@dataclass(frozen=True)
class LandingRow:
    landing: Landing
    outcomes: Outcomes
    cost: Cost
    modules: tuple[ModuleLines, ...]
    rule_vintage: SourceEpochs


class _Timeline(Generic[T]):
    """One task's rows of one source, sorted by time."""

    def __init__(self, timed: list[tuple[datetime, T]]) -> None:
        timed.sort(key=lambda pair: pair[0])
        self._times = [at for at, _ in timed]
        self._values = [value for _, value in timed]

    def window(self, landing: Landing) -> list[T]:
        """Values in (prev_landed_at, landed_at] of the landing's task."""
        prev = landing.prev_landed_at
        start = 0 if prev is None else bisect.bisect_right(self._times, prev)
        return self._values[start:bisect.bisect_right(self._times, landing.landed_at)]

    def first_after(self, at: datetime) -> T | None:
        """The first value strictly after ``at``; None when there is none."""
        index = bisect.bisect_right(self._times, at)
        return self._values[index] if index < len(self._values) else None


_CHUNK = 500


def _chunks(task_ids: Sequence[str]) -> Iterator[Sequence[str]]:
    for start in range(0, len(task_ids), _CHUNK):
        yield task_ids[start:start + _CHUNK]


def _timelines(
    conn: sqlite3.Connection,
    sql: str,
    task_ids: Sequence[str],
    value: Callable[[tuple[Any, ...]], T],
) -> dict[str, _Timeline[T]]:
    """Rows of ``sql`` (selecting task_id, timestamp, then value columns) per task id.

    ``sql`` ends in ``task_id IN ({})``, filled with one placeholder per id of each chunk.
    """
    by_task: dict[str, list[tuple[datetime, T]]] = {}
    for chunk in _chunks(task_ids):
        query = sql.format(', '.join('?' for _ in chunk))
        for row in conn.execute(query, tuple(chunk)):
            by_task.setdefault(str(row[0]), []).append((parse_utc(row[1]), value(row[2:])))
    return {task_id: _Timeline(timed) for task_id, timed in by_task.items()}


_REJECTIONS_SQL = """
SELECT task_id, timestamp FROM events
 WHERE event_type = 'merge_blocked' AND json_extract(data, '$.category') = 'post_merge_verify'
   AND task_id IN ({})
"""
_VERIFY_SQL = """
SELECT task_id, timestamp, json_extract(data, '$.passed') FROM events
 WHERE event_type = 'merge_verify' AND task_id IN ({})
"""
_REVIEW_SQL = """
SELECT task_id, timestamp, json_extract(data, '$.cycle'),
       json_extract(data, '$.has_blocking_issues')
  FROM events
 WHERE event_type = 'review_completed' AND task_id IN ({})
"""
_FLAKE_SQL = 'SELECT task_id, observed_at, verdict FROM flake_occurrence WHERE task_id IN ({})'
_INVOCATIONS_SQL = 'SELECT task_id, started_at, role, cost_usd FROM invocations WHERE task_id IN ({})'
_TURNS_SQL = f"""
SELECT task_id, timestamp, json_extract(data, '$.turns'), json_extract(data, '$.capped_reason')
  FROM events
 WHERE event_type = 'invocation_end'
   AND role IN ({', '.join(repr(role) for role in IMPLEMENTER_CLASS_ROLES)})
   AND json_extract(data, '$.turns') IS NOT NULL
   AND task_id IN ({{}})
"""
_DISPATCH_SQL = """
SELECT task_id, completed_at, outcome, execute_iterations FROM task_results
 WHERE completed_at IS NOT NULL AND completed_at != '' AND task_id IN ({})
"""
_CHARTER_SQL = """
SELECT task_id, completed_at, task_id, {charter_column}, block_reason FROM task_results
 WHERE completed_at IS NOT NULL AND completed_at != '' AND task_id IN ({{}})
"""


def _has_charter_column(conn: sqlite3.Connection) -> bool:
    return any(row[1] == 'charter_class' for row in conn.execute('PRAGMA table_info(task_results)'))


@dataclass(frozen=True)
class _Charter:
    charter_class: CharterClass
    source: CharterClassSource


def _charter(values: tuple[Any, ...]) -> _Charter | None:
    task_id, column, block_reason = values
    if column is not None:
        try:
            return _Charter(CharterClass(column), CharterClassSource.CHARTER_CLASS)
        except ValueError as exc:
            raise UnknownCharterClass(
                f'task_results.charter_class={column!r} for task_id={task_id!r} is not one of '
                f'{sorted(c.value for c in CharterClass)}'
            ) from exc
    for prefix, charter_class in _CHARTER_BACKFILL_PREFIXES:
        if (block_reason or '').startswith(prefix):
            return _Charter(charter_class, CharterClassSource.REGEX_BACKFILL)
    return None


DISPATCH_DONE_OUTCOME = 'done'
"""orchestrator/src/orchestrator/workflow_types.py::WorkflowOutcome.DONE, restated.

shared may not import orchestrator.
"""


@dataclass(frozen=True)
class _Dispatch:
    """One task_results row: a dispatch of the task, and whether it ended reporting done."""

    done: bool
    execute_iterations: int


# orchestrator/src/orchestrator/flake_ledger.py::FlakeVerdict, restated: shared may not import it.
FLAKE_CONFIRMED = 'passes_in_isolation'
FLAKE_REAL_RED = 'fails_in_isolation'


@dataclass(frozen=True)
class _Review:
    cycle: int
    blocking: bool


@dataclass(frozen=True)
class _Invocation:
    role: str
    cost_usd: float


@dataclass(frozen=True)
class _Turns:
    turns: int
    capped: bool


@dataclass(frozen=True)
class _Sources:
    """Everything rows_for reads from runs.db, loaded once per call."""

    epochs: SourceEpochs
    rejections: dict[str, _Timeline[bool]]
    verifies: dict[str, _Timeline[bool]]
    reviews: dict[str, _Timeline[_Review]]
    flakes: dict[str, _Timeline[str]]
    invocations: dict[str, _Timeline[_Invocation]]
    turns: dict[str, _Timeline[_Turns]]
    dispatches: dict[str, _Timeline[_Dispatch]]
    charters: dict[str, _Timeline[_Charter | None]]

    @classmethod
    def load(cls, conn: sqlite3.Connection, task_ids: Sequence[str]) -> _Sources:
        return cls(
            epochs=_read_epochs(conn),
            rejections=_timelines(conn, _REJECTIONS_SQL, task_ids, lambda _: True),
            verifies=_timelines(conn, _VERIFY_SQL, task_ids, lambda v: bool(v[0])),
            reviews=_timelines(conn, _REVIEW_SQL, task_ids,
                               lambda v: _Review(int(v[0]), bool(v[1]))),
            flakes=_timelines(conn, _FLAKE_SQL, task_ids, lambda v: str(v[0])),
            invocations=_timelines(conn, _INVOCATIONS_SQL, task_ids,
                                   lambda v: _Invocation(str(v[0]), float(v[1]))),
            turns=_timelines(conn, _TURNS_SQL, task_ids,
                             lambda v: _Turns(int(v[0]), v[1] == TURNS_CAP_REASON)),
            dispatches=_timelines(
                conn, _DISPATCH_SQL, task_ids,
                lambda v: _Dispatch(v[0] == DISPATCH_DONE_OUTCOME, int(v[1] or 0)),
            ),
            charters=_timelines(conn, _CHARTER_SQL.format(
                charter_column='charter_class' if _has_charter_column(conn) else 'NULL',
            ), task_ids, _charter),
        )


def _window(timelines: dict[str, _Timeline[T]], landing: Landing) -> list[T]:
    """The landing's window of its task's timeline; empty when the task has no rows."""
    timeline = timelines.get(landing.task_id)
    return [] if timeline is None else timeline.window(landing)


def _members_window(timelines: dict[str, _Timeline[T]], unit: Sequence[Landing]) -> list[T]:
    return [value for member in unit for value in _window(timelines, member)]


@dataclass(frozen=True)
class _UnitOutcomes:
    rejection_events: int | None
    flake_verdicts: tuple[str, ...] | None
    verify_passed: tuple[bool, ...] | None


def _unit_outcomes(sources: _Sources, unit: Sequence[Landing]) -> _UnitOutcomes:
    head = unit[0]
    epochs = sources.epochs
    return _UnitOutcomes(
        rejection_events=(len(_members_window(sources.rejections, unit))
                          if _covers(epochs.rejections, head) else None),
        flake_verdicts=(tuple(_members_window(sources.flakes, unit))
                        if _covers(epochs.flake_occurrence, head) else None),
        verify_passed=(tuple(_members_window(sources.verifies, unit))
                       if _covers(epochs.merge_verify, head) else None),
    )


def _outcomes(sources: _Sources, unit: _UnitOutcomes, landing: Landing) -> Outcomes:
    reviews = (_window(sources.reviews, landing)
               if _covers(sources.epochs.review_completed, landing) else None)
    verdicts = unit.flake_verdicts
    passed = unit.verify_passed
    return Outcomes(
        rejected=None if unit.rejection_events is None else unit.rejection_events > 0,
        rejection_events=unit.rejection_events,
        reviewed=None if reviews is None else bool(reviews),
        review_blocking=any(r.blocking for r in reviews) if reviews else None,
        flake_confirmed=None if verdicts is None else FLAKE_CONFIRMED in verdicts,
        flake_real_red=None if verdicts is None else FLAKE_REAL_RED in verdicts,
        flake_rows=None if verdicts is None else len(verdicts),
        reverify_count=None if passed is None else max(len(passed) - 1, 0),
        verify_failed_rows=None if passed is None else sum(1 for ok in passed if not ok),
    )


def _usd_by_role(invocations: list[_Invocation]) -> tuple[tuple[str, float], ...]:
    by_role: dict[str, float] = {}
    for invocation in invocations:
        by_role[invocation.role] = by_role.get(invocation.role, 0.0) + invocation.cost_usd
    return tuple(sorted(by_role.items()))


def _disk_review_cycles(repo: str | os.PathLike[str], task_id: str) -> int | None:
    """The task's lifetime review counter; None when the file is absent, unreadable or malformed.

    The task-keyed directory (orchestrator/src/orchestrator/artifacts.py::TaskArtifacts.task_root_for)
    is review_state.json's current home, so it wins over the legacy bare-id directory.
    """
    meta = Path(repo) / '.worktrees' / '.task-meta'
    for dirname in (f'task-{task_id}', task_id):
        state = meta / dirname / 'review_state.json'
        if not state.exists():
            continue
        try:
            document = json.loads(state.read_text(encoding='utf-8'))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        total = document.get('review_cycles_total') if isinstance(document, dict) else None
        return total if isinstance(total, int) and not isinstance(total, bool) else None
    return None


def _review_rounds(
    sources: _Sources, landing: Landing, repo: str | os.PathLike[str]
) -> tuple[int | None, ReviewRoundsSource | None]:
    """Rounds from review_completed cycles; before that source's epoch, the lifetime disk counter."""
    if _covers(sources.epochs.review_completed, landing):
        cycles = [review.cycle for review in _window(sources.reviews, landing)]
        return (max(cycles), ReviewRoundsSource.REVIEW_COMPLETED) if cycles else (None, None)
    total = _disk_review_cycles(repo, landing.task_id)
    return (total + 1, ReviewRoundsSource.DISK_COUNTER) if total is not None else (None, None)


def _landing_dispatches(sources: _Sources, landing: Landing) -> list[_Dispatch]:
    """The task_results rows of the dispatches that produced the landing.

    The window's rows that did not report done (a done row there is the previous landing's own
    dispatch row), plus the landing dispatch's own row: task_results is written when a dispatch
    ends, after its merge lands, so it is the task's first row after the landing, when done.
    """
    produced = [dispatch for dispatch in _window(sources.dispatches, landing) if not dispatch.done]
    timeline = sources.dispatches.get(landing.task_id)
    landing_dispatch = None if timeline is None else timeline.first_after(landing.landed_at)
    if landing_dispatch is not None and landing_dispatch.done:
        produced.append(landing_dispatch)
    return produced


def _charter_class(sources: _Sources, landing: Landing) -> _Charter | None:
    classified = [c for c in _window(sources.charters, landing) if c is not None]
    return classified[-1] if classified else None


def _cost(
    sources: _Sources, landing: Landing, repo: str | os.PathLike[str], lines: LandedLines | None
) -> Cost:
    epochs = sources.epochs
    invocations = (_window(sources.invocations, landing)
                   if _covers(epochs.invocations, landing) else [])
    turns = (_window(sources.turns, landing)
             if _covers(epochs.invocation_turns, landing) else [])
    dispatches = _landing_dispatches(sources, landing)
    by_role = _usd_by_role(invocations)
    implementer = sum(1 for i in invocations if i.role in IMPLEMENTER_CLASS_ROLES)
    review_rounds, review_rounds_source = _review_rounds(sources, landing, repo)
    charter = _charter_class(sources, landing)
    return Cost(
        all_role_usd=sum(usd for _, usd in by_role) if invocations else None,
        by_role=by_role,
        implementer_invocations=implementer if invocations else None,
        execute_iterations=(sum(d.execute_iterations for d in dispatches)
                            if dispatches else None),
        turns=sum(t.turns for t in turns) if turns else None,
        turns_capped=any(t.capped for t in turns),
        code_lines=None if lines is None else lines.code_lines,
        src_lines=None if lines is None else lines.src_lines,
        test_lines=None if lines is None else lines.test_lines,
        review_rounds=review_rounds,
        review_rounds_source=review_rounds_source,
        charter_class=None if charter is None else charter.charter_class,
        charter_class_source=None if charter is None else charter.source,
    )


def _units(landings: Sequence[Landing]) -> list[tuple[Landing, ...]]:
    """The input's units in order; each must be one complete run of consecutive landings."""
    units = [tuple(run) for _, run in groupby(landings, key=lambda landing: landing.unit_id)]
    started: set[str] = set()
    for unit in units:
        head = unit[0]
        _invariant(len(unit) == head.unit_size,
                   "a unit's members arrive as one run of unit_size landings",
                   unit_id=head.unit_id, unit_size=head.unit_size,
                   task_ids=tuple(member.task_id for member in unit))
        _invariant(head.unit_id not in started,
                   'a unit_id starts only one run of consecutive landings', unit_id=head.unit_id)
        started.add(head.unit_id)
    return units


FIRST_UNIT_GIT_FLOOR_S = 60.0
"""The git timeout the first unit of a call always gets, so a spent budget still makes progress."""


class _BudgetSpent(Exception):
    pass


def _unit_lines(
    repo: str | os.PathLike[str], base_sha: str, sha: str, timeout_s: float
) -> LandedLines | None:
    """The unit's landed lines; None when a named commit is missing, so they never will be."""
    try:
        return landed_lines(repo, base_sha, sha, timeout_s)
    except GitObjectMissing:
        return None


def _git_timeout(first: bool, deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if first:
        return max(FIRST_UNIT_GIT_FLOOR_S, remaining)
    if remaining <= 0:
        raise _BudgetSpent
    return remaining


_MeasuredUnit = tuple[tuple[Landing, ...], LandedLines | None]


def _measure(
    units: Sequence[tuple[Landing, ...]], repo: str | os.PathLike[str], git_budget_s: float
) -> list[_MeasuredUnit]:
    """The longest prefix of ``units`` whose lines fit the budget, each with its lines."""
    deadline = time.monotonic() + git_budget_s
    measured: list[_MeasuredUnit] = []
    for index, unit in enumerate(units):
        head = unit[0]
        if head.base_sha is None:
            measured.append((unit, None))
            continue
        try:
            timeout_s = _git_timeout(index == 0, deadline)
            measured.append((unit, _unit_lines(repo, head.base_sha, head.merge_sha, timeout_s)))
        except (_BudgetSpent, GitTimeout):
            break
    return measured


def rows_for(
    landings: Sequence[Landing],
    runs_db: sqlite3.Connection,
    repo: str | os.PathLike[str],
    git_budget_s: float,
) -> list[LandingRow]:
    """LandingRows for the longest unit-atomic prefix of ``landings`` measurable in the budget.

    ``landings`` must carry each unit's members consecutive and complete, as landings() returns
    them and as any slice at a unit boundary keeps them; the whole input is checked before any
    git work, and a split, short or repeated unit raises QualityRowInvariantError. Rows come
    back in input order. Units are measured in that order; the first always gets at least
    FIRST_UNIT_GIT_FLOOR_S of git time, each later one the remaining budget. A spent budget or
    a GitTimeout ends the call before that unit, never inside it, and no row is ever emitted
    with lines missing for budget reasons: the caller resumes with ``landings[len(rows):]``.
    runs.db is read after git, and only for the measured prefix's tasks.
    """
    measured = _measure(_units(landings), repo, git_budget_s)
    sources = _Sources.load(runs_db, sorted({m.task_id for unit, _ in measured for m in unit}))
    rows: list[LandingRow] = []
    for unit, lines in measured:
        unit_outcomes = _unit_outcomes(sources, unit)
        rows.extend(
            LandingRow(
                landing=member,
                outcomes=_outcomes(sources, unit_outcomes, member),
                cost=_cost(sources, member, repo, lines),
                modules=() if lines is None else lines.modules,
                rule_vintage=sources.epochs,
            )
            for member in unit
        )
    return rows


def module_allocation(row: LandingRow) -> tuple[tuple[str, float], ...]:
    """Each module's share of the row's src lines; empty when it touched no src lines."""
    total = sum(module.src_lines for module in row.modules)
    if total == 0:
        return ()
    return tuple((module.module, module.src_lines / total) for module in row.modules)


def unit_heads(rows: Sequence[LandingRow]) -> list[LandingRow]:
    """One row per unit_id: the member with the lowest ordinal."""
    heads: dict[str, LandingRow] = {}
    for row in rows:
        current = heads.get(row.landing.unit_id)
        if current is None or row.landing.ordinal < current.landing.ordinal:
            heads[row.landing.unit_id] = row
    return sorted(heads.values(), key=lambda row: row.landing.ordinal)
