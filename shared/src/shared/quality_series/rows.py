"""One LandingRow per landing: what happened to it, attributed over (previous landing, landing].

Contract: plans/quality-dashboard-prd.md §Contract "Per-landing row". A source-bearing field is None
when its source has no row in this store yet or the landing precedes that source's first row
(SourceEpochs), never 0. Unit values (rejections, verifies, flakes) are combined over a verify
unit's members and copied to each; reviews stay per landing.
"""

from __future__ import annotations

import bisect
import os
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generic, TypeVar

from shared.quality_series.git_lines import ModuleLines
from shared.quality_series.landings import Landing, parse_utc

__all__ = [
    'LandingRow',
    'Outcomes',
    'QualityRowInvariantError',
    'SourceEpochs',
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
    """The first row's timestamp of each source in this runs.db; None when it has none."""

    rejections: str | None
    merge_verify: str | None
    review_completed: str | None
    flake_occurrence: str | None
    invocation_turns: str | None
    invocations: str | None


_EPOCH_SQL = {
    'rejections': "SELECT MIN(timestamp) FROM events WHERE event_type = 'merge_blocked'"
                  " AND json_extract(data, '$.category') = 'post_merge_verify'",
    'merge_verify': "SELECT MIN(timestamp) FROM events WHERE event_type = 'merge_verify'",
    'review_completed': "SELECT MIN(timestamp) FROM events WHERE event_type = 'review_completed'",
    'flake_occurrence': 'SELECT MIN(observed_at) FROM flake_occurrence',
    'invocation_turns': "SELECT MIN(timestamp) FROM events WHERE event_type = 'invocation_end'"
                        " AND json_extract(data, '$.turns') IS NOT NULL",
    'invocations': 'SELECT MIN(started_at) FROM invocations',
}


def _read_epochs(conn: sqlite3.Connection) -> SourceEpochs:
    return SourceEpochs(**{
        source: conn.execute(sql).fetchone()[0] for source, sql in _EPOCH_SQL.items()
    })


def _covers(epoch: str | None, landing: Landing) -> bool:
    return epoch is not None and landing.landed_time >= parse_utc(epoch)


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


@dataclass(frozen=True)
class LandingRow:
    landing: Landing
    outcomes: Outcomes
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
        prev = landing.prev_landed_time
        start = 0 if prev is None else bisect.bisect_right(self._times, prev)
        return self._values[start:bisect.bisect_right(self._times, landing.landed_time)]


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

FLAKE_CONFIRMED = 'passes_in_isolation'
FLAKE_REAL_RED = 'fails_in_isolation'


@dataclass(frozen=True)
class _Review:
    cycle: int
    blocking: bool


@dataclass(frozen=True)
class _Sources:
    """Everything rows_for reads from runs.db, loaded once per call."""

    epochs: SourceEpochs
    rejections: dict[str, _Timeline[bool]]
    verifies: dict[str, _Timeline[bool]]
    reviews: dict[str, _Timeline[_Review]]
    flakes: dict[str, _Timeline[str]]

    @classmethod
    def load(cls, conn: sqlite3.Connection, task_ids: Sequence[str]) -> _Sources:
        return cls(
            epochs=_read_epochs(conn),
            rejections=_timelines(conn, _REJECTIONS_SQL, task_ids, lambda _: True),
            verifies=_timelines(conn, _VERIFY_SQL, task_ids, lambda v: bool(v[0])),
            reviews=_timelines(conn, _REVIEW_SQL, task_ids,
                               lambda v: _Review(int(v[0]), bool(v[1]))),
            flakes=_timelines(conn, _FLAKE_SQL, task_ids, lambda v: str(v[0])),
        )


def _members_window(
    timelines: dict[str, _Timeline[T]], unit: Sequence[Landing]
) -> list[T]:
    empty: _Timeline[T] = _Timeline([])
    return [value for member in unit
            for value in timelines.get(member.task_id, empty).window(member)]


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
    reviews: list[_Review] | None = None
    if _covers(sources.epochs.review_completed, landing):
        timeline = sources.reviews.get(landing.task_id)
        reviews = timeline.window(landing) if timeline is not None else []
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


def _units(landings: Sequence[Landing]) -> list[tuple[Landing, ...]]:
    members: dict[str, list[Landing]] = {}
    for landing in landings:
        members.setdefault(landing.unit_id, []).append(landing)
    return [tuple(unit) for unit in members.values()]


def rows_for(
    landings: Sequence[Landing],
    runs_db: sqlite3.Connection,
    repo: str | os.PathLike[str],
    git_budget_s: float,
) -> list[LandingRow]:
    """One LandingRow per landing, in the order given (ordinal order from landings())."""
    sources = _Sources.load(runs_db, sorted({landing.task_id for landing in landings}))
    rows: list[LandingRow] = []
    for unit in _units(landings):
        unit_outcomes = _unit_outcomes(sources, unit)
        rows.extend(
            LandingRow(
                landing=member,
                outcomes=_outcomes(sources, unit_outcomes, member),
                modules=(),
                rule_vintage=sources.epochs,
            )
            for member in unit
        )
    return rows


def unit_heads(rows: Sequence[LandingRow]) -> list[LandingRow]:
    """One row per unit_id: the member with the lowest ordinal."""
    heads: dict[str, LandingRow] = {}
    for row in rows:
        current = heads.get(row.landing.unit_id)
        if current is None or row.landing.ordinal < current.landing.ordinal:
            heads[row.landing.unit_id] = row
    return sorted(heads.values(), key=lambda row: row.landing.ordinal)
