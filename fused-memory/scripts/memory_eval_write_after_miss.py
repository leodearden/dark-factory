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

INCIDENT. The production near-duplicate guard
(``fused_memory/server/near_duplicate_guard.py::find_near_duplicate_memory``),
run at production's resolved threshold, finds an existing entry in the write's
category that existed before the write. If no prior search in the window showed
that entry, directly or through its grouped parent, the write is one incident.

CANDIDATES COME FROM A READ-ONLY LIVE SEARCH. The journal holds no cosine
between a written content and existing entries. A search row's ``store_score``
is relative to that search's query, and the guard's own pre-write search is not
journalled. So the guard's search is re-run against the write's stored content,
with no ``causation_id``, which means nothing is journalled. Entries deleted
since the write cannot be seen, a stated undercount.

PERIODS tile on run stamps: ``[previous e7 artifact's stamp, this stamp)``.
Consecutive runs are therefore disjoint samples. ``--initial-period-hours``
bootstraps the first run only.

The script never writes the corpus or the journal, and it asserts no alarm
limit: the scheduled runner judges the series (G6).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EVAL_ID = 'e7-write-after-miss'
METRIC_ID = 'write-after-miss'

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


class _UnparsedRowError(ValueError):
    """A journal row whose blobs do not have the shape their producer writes."""


@dataclass(frozen=True)
class Period:
    """The half-open evaluation period ``[start, end)``, in aware datetimes."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError(
                f'Period bounds must be timezone-aware: start={self.start!r}, end={self.end!r}'
            )
        if not self.start < self.end:
            raise ValueError(
                f'Period start {self.start.isoformat()} is not before end {self.end.isoformat()}'
            )


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

    That is: no telemetry envelope, a telemetry fault, or a truncated result list.
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
    except sqlite3.Error as exc:  # pragma: no cover - OS-level failure
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
            'schema_version' in summary
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
