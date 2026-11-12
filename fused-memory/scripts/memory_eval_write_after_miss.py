"""E7 write-after-miss: an agent searched, was not shown the entry, and wrote a duplicate.

The metric ``write-after-miss`` is ``incidents / adjudicated`` over one
evaluation period. It is a proportion and ``higher_is_worse``.

WINDOW. A caller's window is ``(project_id, caller agent id)``. On a write row
the caller is the ``agent_id`` column; on a search row it is
``params.caller_agent_id``. The journal leaves ``session_id`` NULL on both row
kinds, so the window cannot be a session.

ELIGIBILITY. A write is adjudicated when all of these hold:
- its category is Mem0-primary;
- it landed at least one memory id;
- its window holds at least one search strictly before it, inside the period.
Every other write is counted under a skip reason, so
``writes = adjudicated + skipped``.

INCIDENT. The pure near-duplicate guard function
(``fused_memory/server/near_duplicate_guard.py::find_near_duplicate_memory``),
at its configured threshold (``resolve_near_dup_threshold``), finds an existing
entry in the write's category that existed before the write. If no prior search
in the window showed that entry, directly or through its grouped parent, the
write is one incident.

THE PREDICATE IS A YARDSTICK, NOT PRODUCTION'S DECISION. While
``write_triage.enabled`` is on, ``fused_memory/server/tools.py::add_memory``
routes every explicit Mem0-primary write through write triage, whose calibrated
bands judge its own candidates, and never runs this guard. An incident means
"the guard function calls this a near-duplicate of an entry the caller was not
shown", not "production rejected or attached the write".

CANDIDATES COME FROM A READ-ONLY LIVE SEARCH. The journal holds no cosine
between a written content and existing entries: a search row's ``store_score``
is relative to that search's query. So a guard-shaped search (Mem0 only, the
write's category, no topic anchoring) is run with the write's stored content,
with no ``causation_id``, which means nothing is journalled. Three narrowings:
- entries deleted since the write cannot be seen, a stated undercount;
- entries written after it are dropped; when they still fill a search at
  ``CANDIDATE_FETCH_CEILING`` and nothing matched, the write is skipped as
  ``candidates_crowded_out`` rather than called clean;
- candidates with no aware ``created_at`` cannot be placed before the write;
  they are dropped and counted as ``undated_candidates``.

PERIODS tile on run stamps: ``[previous e7 artifact's stamp, this stamp)``.
Consecutive runs are therefore disjoint samples. ``--initial-period-hours``
bootstraps the first run only. A period longer than ``--max-period-hours``
keeps only its latest hours, so one outage cannot grow every later run past the
runner's timeout; the dropped hours are disclosed as uncovered.

The script never writes the corpus or the journal, and it asserts no alarm
limit: the scheduled runner judges the series (G6).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from operator import attrgetter
from pathlib import Path
from types import MappingProxyType
from typing import Any

from shared import safe_io
from shared.memory_eval_metrics import (
    SCHEMA_VERSION,
    Corpus,
    Metric,
    MetricSeries,
    canonical_json_text,
    eval_dir,
    load_series_window,
    run_stamp,
    stamp_datetime,
    validate_stamp,
    write_metric_series,
)
from shared.memory_eval_metrics import render_report as render_series_report

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.models.enums import MEM0_PRIMARY, MemoryCategory
from fused_memory.models.memory import MemoryResult
from fused_memory.server.grouped_read import child_parent_id
from fused_memory.server.near_duplicate_guard import (
    cosine_of,
    find_near_duplicate_memory,
    resolve_near_dup_threshold,
)
from fused_memory.services.memory_service import MemoryService
from fused_memory.services.read_telemetry import SEARCH_TELEMETRY_SCHEMA_VERSION
from fused_memory.services.write_journal import WriteJournal

EVAL_ID = 'e7-write-after-miss'
METRIC_ID = 'write-after-miss'

REPO_ROOT = Path(__file__).resolve().parents[2]
"""fused-memory.service's WorkingDirectory: the anchor of a relative ``reconciliation.data_dir``."""

_DEFAULT_METRICS_ROOT = REPO_ROOT / 'fused-memory' / 'data' / 'memory-evals'

INITIAL_PERIOD_HOURS = 24.0
"""The first run's period length. Every later period runs from the previous run's stamp."""

MAX_PERIOD_HOURS = 72.0
"""The longest period one run adjudicates; older hours since the previous run are left uncovered.

Adjudication costs one store read and one embedding search per eligible write,
so an unbounded period after an outage could outrun the runner's per-producer
timeout, fail, and widen the next period further.
"""

CANDIDATE_DEPTH = 5
"""How many entries that predate the write a replay judges: the guard function's search depth.

The ``limit=5`` of the guard's pre-write search in
``fused_memory/server/tools.py::add_memory``. The first search asks for this
many plus the write's own ids, because those are filtered out.
"""

CANDIDATE_FETCH_CEILING = 64
"""The deepest search a replay runs to find ``CANDIDATE_DEPTH`` entries that predate the write.

Entries written after the write, such as a loop's own later restatements, can
outrank every earlier one. The replay doubles its depth past them up to here.
"""

JOURNAL_BUSY_TIMEOUT_SECS = 30.0
"""How long a read waits on the live journal's writer lock (sqlite3's ``timeout``).

The live journal is appended to continuously, so contention is expected: a read
that waits beats one that refuses. Same value and reasoning as
``harvest_production_queries.py::JOURNAL_CONNECT_TIMEOUT``.
"""

_SELECT_PERIOD_ROWS = (
    'SELECT id, operation, agent_id, params, result_summary, created_at FROM write_ops '
    'WHERE project_id = ? AND created_at >= ? AND created_at < ? '
    "AND operation IN ('search', 'add_memory') ORDER BY created_at"
)


class JournalUnavailableError(RuntimeError):
    """The journal is absent, unreadable, or not a write journal.

    Raised instead of returning an empty window, so a missing journal is never
    read as "zero incidents".
    """


class MemoryStoreUnavailableError(RuntimeError):
    """The live memory store could not be opened for reads."""


class _UnparsedRowError(ValueError):
    """A journal row whose blobs do not have the shape their producer writes."""


@dataclass(frozen=True)
class Period:
    """The half-open evaluation period ``[start, end)``, in aware datetimes."""

    start: datetime
    end: datetime
    uncovered_start: datetime | None = None
    """Where ``[uncovered_start, start)`` begins: hours since the previous run that the cap left unevaluated."""

    def __post_init__(self) -> None:
        bounds = (self.start, self.end, self.uncovered_start or self.start)
        if any(bound.tzinfo is None for bound in bounds):
            raise ValueError(
                f'Period bounds must be timezone-aware: start={self.start!r}, end={self.end!r}, '
                f'uncovered_start={self.uncovered_start!r}'
            )
        if not self.start < self.end:
            raise ValueError(
                f'Period start {self.start.isoformat()} is not before end {self.end.isoformat()}'
            )
        if self.uncovered_start is not None and not self.uncovered_start < self.start:
            raise ValueError(
                f'uncovered_start {self.uncovered_start.isoformat()} is not before '
                f'start {self.start.isoformat()}'
            )

    @property
    def uncovered_seconds(self) -> int:
        if self.uncovered_start is None:
            return 0
        return round((self.start - self.uncovered_start).total_seconds())


@dataclass(frozen=True)
class WriteRow:
    row_id: str
    project_id: str
    caller_agent_id: str | None
    category: str
    memory_ids: tuple[str, ...]
    created_at: datetime


@dataclass(frozen=True)
class SearchRow:
    row_id: str
    project_id: str
    caller_agent_id: str
    shown_ids: frozenset[str]
    complete: bool
    """False when the row cannot prove what it did NOT show.

    That is: no telemetry envelope, one of another schema version, a telemetry
    fault, or a truncated result list.
    """
    created_at: datetime


@dataclass(frozen=True)
class JournalWindow:
    writes: tuple[WriteRow, ...]
    searches: tuple[SearchRow, ...]
    unattributed_searches: int
    unparsed_rows: int


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Open *db_path* read-only: ``mode=ro`` at the VFS layer, ``query_only`` per statement.

    Copied from ``harvest_production_queries.py::_connect_readonly``; scripts
    keep such helpers private rather than importing a sibling script.
    """
    if not db_path.exists():
        raise JournalUnavailableError(f'write journal not found: {db_path}')
    try:
        connection = sqlite3.connect(
            f'{db_path.resolve().as_uri()}?mode=ro', uri=True, timeout=JOURNAL_BUSY_TIMEOUT_SECS,
        )
    except sqlite3.Error as exc:
        raise JournalUnavailableError(f'cannot open {db_path} read-only: {exc}') from exc
    connection.execute('PRAGMA query_only=ON')
    return connection


def _journal_timestamp(moment: datetime) -> str:
    """*moment* rendered as ``WriteJournal`` stamps ``created_at``, so SQL string order is time order."""
    return moment.astimezone(UTC).isoformat()


def _period_rows(db_path: Path, project_id: str, period: Period) -> Iterator[tuple]:
    connection = _connect_readonly(db_path)
    try:
        present = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='write_ops'"
        ).fetchone()
        if present is None:
            raise JournalUnavailableError(f'{db_path} has no write_ops table')
        yield from connection.execute(
            _SELECT_PERIOD_ROWS,
            (project_id, _journal_timestamp(period.start), _journal_timestamp(period.end)),
        )
    except sqlite3.Error as exc:
        raise JournalUnavailableError(f'cannot read write_ops from {db_path}: {exc}') from exc
    finally:
        connection.close()


def _decode(blob: str | None) -> Any:
    if blob is None:
        return None
    try:
        return json.loads(blob)
    except (TypeError, ValueError) as exc:
        raise _UnparsedRowError(f'not JSON: {blob[:80]!r}') from exc


def _aware_time(created_at: object) -> datetime:
    try:
        moment = datetime.fromisoformat(created_at) if isinstance(created_at, str) else None
    except ValueError:
        moment = None
    if moment is None or moment.tzinfo is None:
        raise _UnparsedRowError(f'created_at {created_at!r} is not an aware ISO timestamp')
    return moment


def _string_list(value: object, *, what: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise _UnparsedRowError(f'{what} is not a list of strings: {value!r}')
    return tuple(value)


def _write_row(project_id: str, row: tuple) -> WriteRow:
    row_id, _operation, agent_id, params_blob, summary_blob, created_at = row
    params = _decode(params_blob)
    summary = _decode(summary_blob)
    if not isinstance(params, dict) or not isinstance(params.get('category'), str):
        raise _UnparsedRowError(f'write {row_id}: params carry no category')
    if not isinstance(summary, dict):
        raise _UnparsedRowError(f'write {row_id}: result_summary is not an object')
    return WriteRow(
        row_id=row_id,
        project_id=project_id,
        caller_agent_id=agent_id if isinstance(agent_id, str) and agent_id else None,
        category=params['category'],
        memory_ids=_string_list(summary.get('memory_ids'), what=f'write {row_id} memory_ids'),
        created_at=_aware_time(created_at),
    )


def _shown_ids(summary: dict) -> frozenset[str]:
    shown: set[str] = set()
    for entry in summary.get('results') or ():
        if not isinstance(entry, dict):
            raise _UnparsedRowError(f'search result entry is not an object: {entry!r}')
        if isinstance(entry.get('id'), str):
            shown.add(entry['id'])
        shown.update(_string_list(entry.get('folded_child_ids', []), what='folded_child_ids'))
    return frozenset(shown)


def _search_row(project_id: str, row: tuple) -> SearchRow | None:
    """The parsed search, or ``None`` when it names no caller (it belongs to no window)."""
    row_id, _operation, _agent_id, params_blob, summary_blob, created_at = row
    params = _decode(params_blob)
    if not isinstance(params, dict):
        raise _UnparsedRowError(f'search {row_id}: params is not an object')
    caller = params.get('caller_agent_id')
    if not isinstance(caller, str) or not caller:
        return None
    summary = _decode(summary_blob)
    if summary is not None and not isinstance(summary, dict):
        raise _UnparsedRowError(f'search {row_id}: result_summary is not an object')
    summary = summary or {}
    return SearchRow(
        row_id=row_id,
        project_id=project_id,
        caller_agent_id=caller,
        shown_ids=_shown_ids(summary),
        complete=(
            summary.get('schema_version') == SEARCH_TELEMETRY_SCHEMA_VERSION
            and not summary.get('telemetry_error')
            and not summary.get('results_truncated')
        ),
        created_at=_aware_time(created_at),
    )


def read_journal(db_path: str | Path, *, project_id: str, period: Period) -> JournalWindow:
    """Every ``add_memory`` and ``search`` row of *project_id* in *period*, parsed, read-only."""
    writes: list[WriteRow] = []
    searches: list[SearchRow] = []
    unattributed = 0
    unparsed = 0
    for row in _period_rows(Path(db_path), project_id, period):
        try:
            if row[1] == 'add_memory':
                writes.append(_write_row(project_id, row))
            elif (search := _search_row(project_id, row)) is None:
                unattributed += 1
            else:
                searches.append(search)
        except _UnparsedRowError:
            unparsed += 1
    return JournalWindow(
        writes=tuple(writes),
        searches=tuple(searches),
        unattributed_searches=unattributed,
        unparsed_rows=unparsed,
    )


class SkipReason(StrEnum):
    """Why a write in the period was not adjudicated. Each one is counted, never dropped."""

    NO_CALLER = 'no_caller'
    """The write row names no agent, so it has no window to search in."""
    NO_PRIOR_SEARCH = 'no_prior_search'
    CATEGORY_NOT_COMPARABLE = 'category_not_comparable'
    NO_MEMORY_ID = 'no_memory_id'
    CONTENT_UNAVAILABLE = 'content_unavailable'
    CANDIDATE_SEARCH_DEGRADED = 'candidate_search_degraded'
    SHOWN_SET_INCOMPLETE = 'shown_set_incomplete'
    """The match is outside every known shown set and some prior search cannot prove absence."""
    CANDIDATES_CROWDED_OUT = 'candidates_crowded_out'
    """Nothing matched, but later entries still filled the ceiling-depth search: an earlier match may lie deeper."""


class Exposure(StrEnum):
    """Whether a write's window was shown the entry the guard matched."""

    SHOWN = 'shown'
    NOT_SHOWN = 'not_shown'
    UNKNOWN = 'unknown'


@dataclass(frozen=True)
class Incident:
    """One write whose guard-selected near-duplicate no prior search in its window showed."""

    write_row_id: str
    search_row_ids: tuple[str, ...]
    project_id: str
    caller_agent_id: str
    matched_memory_id: str
    similarity: float
    threshold: float
    category: str
    write_created_at: datetime


@dataclass(frozen=True)
class Adjudication:
    incidents: tuple[Incident, ...]
    adjudicated: int
    skipped: Mapping[SkipReason, int]
    """Every reason, present at zero too."""
    undated_candidates: int
    """Candidates dropped from adjudicated writes because they carry no aware ``created_at``."""


SearchesByWindow = Mapping[tuple[str, str], tuple[SearchRow, ...]]

_created_at = attrgetter('created_at')


def searches_by_window(searches: Sequence[SearchRow]) -> SearchesByWindow:
    """Each ``(project_id, caller)`` window's searches in time order, grouped once per run."""
    grouped: defaultdict[tuple[str, str], list[SearchRow]] = defaultdict(list)
    for search in searches:
        grouped[(search.project_id, search.caller_agent_id)].append(search)
    return MappingProxyType({
        window: tuple(sorted(rows, key=_created_at)) for window, rows in grouped.items()
    })


def prior_searches(write: WriteRow, by_window: SearchesByWindow) -> tuple[SearchRow, ...]:
    """The write's window: same project and caller, strictly before the write."""
    if write.caller_agent_id is None:
        return ()
    rows = by_window.get((write.project_id, write.caller_agent_id), ())
    return rows[:bisect_left(rows, write.created_at, key=_created_at)]


def ineligibility(write: WriteRow, prior: Sequence[SearchRow]) -> SkipReason | None:
    """Why *write* cannot be adjudicated before any store read, or ``None`` when it can."""
    if write.category not in MEM0_PRIMARY:
        return SkipReason.CATEGORY_NOT_COMPARABLE
    if not write.memory_ids:
        return SkipReason.NO_MEMORY_ID
    if write.caller_agent_id is None:
        return SkipReason.NO_CALLER
    if not prior:
        return SkipReason.NO_PRIOR_SEARCH
    return None


async def _stored_content(memory: Any, write: WriteRow, project_id: str) -> str | None:
    for memory_id in write.memory_ids:
        record = await memory.get_memory_by_id(project_id, memory_id)
        if record is not None and record.get('content'):
            return record['content']
    return None


@dataclass(frozen=True)
class CandidateWindow:
    """What one candidate search found for a write, split by when each entry was created."""

    admissible: tuple[MemoryResult, ...]
    """Entries that already existed when the write landed, excluding its own ids."""
    undated: int
    """Entries with no aware ``created_at``, which cannot be placed before or after the write."""
    crowded: bool = False
    """The ceiling-depth search was full yet held fewer than ``CANDIDATE_DEPTH`` admissible entries."""


def _created_at_of(result: MemoryResult) -> datetime | None:
    try:
        created = datetime.fromisoformat(result.created_at) if result.created_at else None
    except ValueError:
        return None
    return created if created is not None and created.tzinfo is not None else None


def candidate_window(results: Sequence[MemoryResult], write: WriteRow) -> CandidateWindow:
    """*results* without the write's own ids, split into entries that predate it and undated ones."""
    others = [result for result in results if result.id not in write.memory_ids]
    created = [_created_at_of(result) for result in others]
    return CandidateWindow(
        admissible=tuple(
            result for result, moment in zip(others, created, strict=True)
            if moment is not None and moment < write.created_at
        ),
        undated=sum(moment is None for moment in created),
    )


async def fetch_candidates(
    memory: Any, write: WriteRow, *, project_id: str,
) -> CandidateWindow | SkipReason:
    """Search the write's stored content for entries that predate it, journalling nothing.

    No ``causation_id`` is passed, so ``MemoryService.search`` writes no journal
    row. ``anchor_topics=False`` matches the guard function's search: a pinned
    canonical carries no cosine and would only spend a candidate slot. The depth
    doubles while later entries crowd out earlier ones, up to
    ``CANDIDATE_FETCH_CEILING``.
    """
    content = await _stored_content(memory, write, project_id)
    if content is None:
        return SkipReason.CONTENT_UNAVAILABLE
    limit = CANDIDATE_DEPTH + len(write.memory_ids)
    while True:
        results = await memory.search(
            query=content,
            project_id=project_id,
            categories=[write.category],
            stores=['mem0'],
            limit=limit,
            anchor_topics=False,
        )
        if getattr(results, 'degraded', False):
            return SkipReason.CANDIDATE_SEARCH_DEGRADED
        window = candidate_window(results, write)
        if len(window.admissible) >= CANDIDATE_DEPTH or len(results) < limit:
            return window
        if limit >= CANDIDATE_FETCH_CEILING:
            return replace(window, crowded=True)
        limit = min(limit * 2, CANDIDATE_FETCH_CEILING)


def was_shown(match: MemoryResult, searches: Sequence[SearchRow]) -> Exposure:
    """Whether *match*, or the grouped parent it folds under, was in some search's shown set.

    UNKNOWN when it is in no known shown set but some search is incomplete: that
    search may have shown it, so absence is not proven.
    """
    shown = frozenset().union(*(search.shown_ids for search in searches))
    if match.id in shown or child_parent_id(match.metadata) in shown:
        return Exposure.SHOWN
    if all(search.complete for search in searches):
        return Exposure.NOT_SHOWN
    return Exposure.UNKNOWN


@dataclass(frozen=True)
class _WriteVerdict:
    outcome: Incident | SkipReason | None
    """The write's incident, the reason it was skipped, or ``None`` for a clean adjudication."""
    undated_candidates: int = 0


async def _adjudicate_write(
    write: WriteRow,
    prior: Sequence[SearchRow],
    memory: Any,
    *,
    project_id: str,
    threshold: float,
) -> _WriteVerdict:
    reason = ineligibility(write, prior)
    if reason is not None:
        return _WriteVerdict(reason)
    candidates = await fetch_candidates(memory, write, project_id=project_id)
    if isinstance(candidates, SkipReason):
        return _WriteVerdict(candidates)
    return _WriteVerdict(
        _judge(write, prior, candidates, threshold=threshold),
        undated_candidates=candidates.undated,
    )


def _judge(
    write: WriteRow,
    prior: Sequence[SearchRow],
    candidates: CandidateWindow,
    *,
    threshold: float,
) -> Incident | SkipReason | None:
    match = find_near_duplicate_memory(
        list(candidates.admissible), threshold, category=MemoryCategory(write.category),
    )
    if match is None:
        return SkipReason.CANDIDATES_CROWDED_OUT if candidates.crowded else None
    exposure = was_shown(match, prior)
    if exposure is Exposure.SHOWN:
        return None
    if exposure is Exposure.UNKNOWN:
        return SkipReason.SHOWN_SET_INCOMPLETE
    return Incident(
        write_row_id=write.row_id,
        search_row_ids=tuple(search.row_id for search in prior),
        project_id=write.project_id,
        caller_agent_id=prior[0].caller_agent_id,
        matched_memory_id=match.id,
        similarity=cosine_of(match) or 0.0,
        threshold=threshold,
        category=write.category,
        write_created_at=write.created_at,
    )


async def adjudicate(window: JournalWindow, memory: Any, *, project_id: str) -> Adjudication:
    """Adjudicate every write in *window* with the near-duplicate guard function at its configured threshold."""
    threshold = resolve_near_dup_threshold(memory)
    by_window = searches_by_window(window.searches)
    incidents: list[Incident] = []
    skipped: Counter[SkipReason] = Counter()
    adjudicated = 0
    undated = 0
    for write in window.writes:
        verdict = await _adjudicate_write(
            write, prior_searches(write, by_window), memory,
            project_id=project_id, threshold=threshold,
        )
        if isinstance(verdict.outcome, SkipReason):
            skipped[verdict.outcome] += 1
            continue
        adjudicated += 1
        undated += verdict.undated_candidates
        if verdict.outcome is not None:
            incidents.append(verdict.outcome)
    return Adjudication(
        incidents=tuple(incidents),
        adjudicated=adjudicated,
        skipped=MappingProxyType({reason: skipped[reason] for reason in SkipReason}),
        undated_candidates=undated,
    )


def incidents_filename(stamp: str) -> str:
    """``incidents-<STAMP>.json``: the evidence file's name beside the metrics, and its ``details_path``.

    A bare filename, resolved against the metrics file's directory, as the
    sibling producers stamp theirs.
    """
    return f'incidents-{validate_stamp(stamp, what="incidents stamp")}.json'


def corpus_counts(
    adjudication: Adjudication, window: JournalWindow, period: Period,
) -> dict[str, int]:
    """Every narrowing between the period's rows and the metric's denominator, at zero too."""
    return {
        'uncovered_period_seconds': period.uncovered_seconds,
        'writes_in_period': len(window.writes),
        'searches_in_period': len(window.searches),
        'unattributed_searches': window.unattributed_searches,
        'unparsed_rows': window.unparsed_rows,
        'adjudicated': adjudication.adjudicated,
        'undated_candidates': adjudication.undated_candidates,
        'incidents': len(adjudication.incidents),
        **{f'skipped_{reason.value}': count for reason, count in adjudication.skipped.items()},
    }


def build_series(
    adjudication: Adjudication,
    window: JournalWindow,
    *,
    period: Period,
    project_id: str,
    stamp: str,
    details_path: str,
    eval_id: str = EVAL_ID,
) -> MetricSeries:
    """The run's M1 series. The proportion is ABSENT when nothing was adjudicated, never 0/0."""
    metrics = []
    if adjudication.adjudicated > 0:
        metrics.append(Metric(
            metric_id=METRIC_ID,
            kind='proportion',
            value=len(adjudication.incidents) / adjudication.adjudicated,
            n=adjudication.adjudicated,
            denominator=adjudication.adjudicated,
            direction='higher_is_worse',
            details_path=details_path,
        ))
    return MetricSeries(
        schema_version=SCHEMA_VERSION,
        eval_id=eval_id,
        run_stamp=stamp,
        corpus=Corpus(project_id=project_id, counts=corpus_counts(adjudication, window, period)),
        metrics=metrics,
    )


def _incident_record(incident: Incident) -> dict[str, Any]:
    return {
        'write_row_id': incident.write_row_id,
        'search_row_ids': list(incident.search_row_ids),
        'project_id': incident.project_id,
        'caller_agent_id': incident.caller_agent_id,
        'matched_memory_id': incident.matched_memory_id,
        'similarity': incident.similarity,
        'threshold': incident.threshold,
        'category': incident.category,
        'write_created_at': incident.write_created_at.isoformat(),
    }


def incidents_payload(
    adjudication: Adjudication, *, project_id: str, stamp: str, period: Period, eval_id: str = EVAL_ID,
) -> dict[str, Any]:
    """The structured INV-2 evidence: each incident's fields, the period, and the skip tally."""
    return {
        'eval_id': eval_id,
        'run_stamp': stamp,
        'project_id': project_id,
        'period_start': period.start.isoformat(),
        'period_end': period.end.isoformat(),
        'uncovered_start': period.uncovered_start.isoformat() if period.uncovered_start else None,
        'adjudicated': adjudication.adjudicated,
        'skipped': {reason.value: count for reason, count in adjudication.skipped.items()},
        'incidents': [_incident_record(incident) for incident in adjudication.incidents],
    }


def _measurement_line(adjudication: Adjudication, writes: int) -> str:
    if adjudication.adjudicated == 0:
        return (
            f'{METRIC_ID}: not measured: 0 of {writes} write(s) in the period were '
            'adjudicated (see the skipped_* counts above)'
        )
    return (
        f'{METRIC_ID}: {len(adjudication.incidents)} incident(s) in '
        f'{adjudication.adjudicated} adjudicated write(s)'
    )


def _incident_line(incident: Incident) -> str:
    return (
        f'  - write {incident.write_row_id}: caller {incident.caller_agent_id}, '
        f'matched {incident.matched_memory_id}, similarity {incident.similarity:.3f} '
        f'vs threshold {incident.threshold:.3f}, '
        f'searches {", ".join(incident.search_row_ids)}'
    )


def render_report(
    series: MetricSeries, adjudication: Adjudication, *, period: Period, writes: int,
) -> str:
    """The shared series report, then the period, the measurement, and each incident."""
    lines = [
        render_series_report(series).rstrip('\n'),
        '',
        f'period: [{period.start.isoformat()}, {period.end.isoformat()})',
    ]
    if period.uncovered_start is not None:
        lines.append(
            f'uncovered: [{period.uncovered_start.isoformat()}, {period.start.isoformat()}) '
            'since the previous run, past --max-period-hours; not evaluated'
        )
    lines.append(_measurement_line(adjudication, writes))
    if adjudication.incidents:
        lines.append(f'incidents ({len(adjudication.incidents)}):')
        lines.extend(_incident_line(incident) for incident in adjudication.incidents)
    return '\n'.join(lines) + '\n'


@dataclass(frozen=True)
class WriteAfterMissOutcome:
    series: MetricSeries
    adjudication: Adjudication
    report: str
    metrics_path: Path
    report_path: Path
    incidents_path: Path


def _write_artifact(path: Path, text: str) -> None:
    safe_io.atomic_write_text(path, text, encoding='utf-8', mode=0o600)


async def run_write_after_miss(
    memory: Any,
    *,
    journal_path: str | Path,
    out_root: str | Path,
    project_id: str,
    stamp: str,
    period: Period,
    eval_id: str = EVAL_ID,
) -> WriteAfterMissOutcome:
    """Read the period, adjudicate it, and write the incidents, metrics and report artifacts."""
    window = read_journal(journal_path, project_id=project_id, period=period)
    return await produce_write_after_miss(
        window, memory, out_root=out_root, project_id=project_id, stamp=stamp, period=period,
        eval_id=eval_id,
    )


async def produce_write_after_miss(
    window: JournalWindow,
    memory: Any,
    *,
    out_root: str | Path,
    project_id: str,
    stamp: str,
    period: Period,
    eval_id: str = EVAL_ID,
) -> WriteAfterMissOutcome:
    """Adjudicate an already-read *window* and write the incidents, metrics and report artifacts."""
    adjudication = await adjudicate(window, memory, project_id=project_id)
    incidents_path = eval_dir(out_root, eval_id) / incidents_filename(stamp)
    series = build_series(
        adjudication, window, period=period,
        project_id=project_id, stamp=stamp, details_path=incidents_path.name, eval_id=eval_id,
    )
    incidents_path.parent.mkdir(parents=True, exist_ok=True)
    _write_artifact(incidents_path, canonical_json_text(incidents_payload(
        adjudication, project_id=project_id, stamp=stamp, period=period, eval_id=eval_id,
    )))
    metrics_path, report_path = write_metric_series(series, out_root)
    report = render_report(series, adjudication, period=period, writes=len(window.writes))
    _write_artifact(report_path, report)
    return WriteAfterMissOutcome(
        series=series,
        adjudication=adjudication,
        report=report,
        metrics_path=metrics_path,
        report_path=report_path,
        incidents_path=incidents_path,
    )


def evaluation_period(
    root: str | Path,
    eval_id: str,
    stamp: str,
    *,
    initial_period_hours: float,
    max_period_hours: float,
) -> Period:
    """``[previous run's stamp, stamp)``, so consecutive runs are disjoint samples.

    With no earlier artifact, the period is the *initial_period_hours* before
    *stamp*. A period longer than *max_period_hours* starts that long before
    *stamp* instead, and names the previous stamp as its ``uncovered_start``.
    """
    end = stamp_datetime(stamp)
    previous = load_series_window(root, eval_id, limit=1, before_stamp=stamp)
    if not previous:
        return Period(end - timedelta(hours=initial_period_hours), end)
    start = stamp_datetime(previous[-1].run_stamp)
    earliest = end - timedelta(hours=max_period_hours)
    if start < earliest:
        return Period(earliest, end, uncovered_start=start)
    return Period(start, end)


def build_parser() -> argparse.ArgumentParser:
    """The runner's ``--metrics-root/--eval-id/--project-id`` plus this producer's inputs.

    There is no ``--apply`` and no ``--no-metrics``: the producer only reads,
    and a run that cannot write its artifact must fail rather than go quiet.
    """
    parser = argparse.ArgumentParser(description=(
        'Read-only E7 write-after-miss producer: the share of searched-then-written '
        'Mem0 writes whose near-duplicate the caller was never shown.'
    ))
    parser.add_argument('--project-id', dest='project_id', required=True, help='Project id to measure')
    parser.add_argument(
        '--metrics-root', dest='metrics_root', default=str(_DEFAULT_METRICS_ROOT),
        help=f'Root for M1 metric artifacts (default: {_DEFAULT_METRICS_ROOT})',
    )
    parser.add_argument(
        '--eval-id', dest='eval_id', default=EVAL_ID,
        help=f'Artifact directory name under --metrics-root (default: {EVAL_ID})',
    )
    parser.add_argument(
        '--journal', default=None,
        help='Write journal to read (default: <reconciliation.data_dir>/write_journal.db '
             'from the fused-memory config, a relative data_dir anchored at the repo root)',
    )
    parser.add_argument(
        '--initial-period-hours', dest='initial_period_hours', type=float,
        default=INITIAL_PERIOD_HOURS,
        help='First-run bootstrap only: the period length when no earlier artifact exists. '
             'Later periods run from the previous run stamp (default: %(default)s)',
    )
    parser.add_argument(
        '--max-period-hours', dest='max_period_hours', type=float, default=MAX_PERIOD_HOURS,
        help='The longest period one run adjudicates; earlier hours since the previous run '
             'are reported as uncovered (default: %(default)s)',
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to fused-memory config file (sets CONFIG_PATH env var)',
    )
    return parser


def _default_journal(config: FusedMemoryConfig) -> Path:
    data_dir = Path(config.reconciliation.data_dir)
    return WriteJournal(data_dir if data_dir.is_absolute() else REPO_ROOT / data_dir).db_path


MemoryFactory = Callable[[FusedMemoryConfig], Awaitable[Any]]


async def open_memory(
    config: FusedMemoryConfig,
    *,
    service: Callable[[FusedMemoryConfig], Any] = MemoryService,
) -> Any:
    """The live service ``main`` uses by default, opened for reads only. The caller closes it.

    ``initialize_for_reads`` starts no durable write queue and runs no Graphiti
    startup maintenance. The full ``initialize`` would reset the live server's
    in-flight queued writes to pending and execute them from this process.
    """
    memory = service(config)
    try:
        await memory.initialize_for_reads()
    except (OSError, sqlite3.Error) as exc:
        await memory.close()
        raise MemoryStoreUnavailableError(
            f'the memory store could not be opened ({type(exc).__name__}): {exc}'
        ) from exc
    return memory


async def _run(args: argparse.Namespace, open_memory: MemoryFactory) -> int:
    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)
    stamp = run_stamp()
    period = evaluation_period(
        args.metrics_root, args.eval_id, stamp,
        initial_period_hours=args.initial_period_hours, max_period_hours=args.max_period_hours,
    )
    config = FusedMemoryConfig()
    journal_path = Path(args.journal) if args.journal else _default_journal(config)
    window = read_journal(journal_path, project_id=args.project_id, period=period)
    memory = await open_memory(config)
    try:
        outcome = await produce_write_after_miss(
            window,
            memory,
            out_root=args.metrics_root,
            project_id=args.project_id,
            stamp=stamp,
            period=period,
            eval_id=args.eval_id,
        )
    finally:
        await memory.close()
    print(outcome.report, end='')
    return 0


def main(argv: Sequence[str] | None = None, *, open_memory: MemoryFactory = open_memory) -> int:
    """Exit non-zero on an unreadable journal, so the runner files a produce failure."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_run(args, open_memory))
    except (JournalUnavailableError, MemoryStoreUnavailableError) as exc:
        print(f'{args.eval_id}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
