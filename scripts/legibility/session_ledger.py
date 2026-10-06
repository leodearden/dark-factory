#!/usr/bin/env python3
"""Host-local ledger of successfully coded sessions.

The nightly trickle and the census each record every session they coded
successfully; the census skips ledgered sessions when it mines, so no
session is coded twice and a capped run resumes where it stopped. The
contract (location, columns, pruning, window) is
plans/census-incremental-prd.md §4.2 (C2).

Stdlib-only, like :mod:`legibility.trickle_state`, whose per-project state
directory it shares.
"""
from __future__ import annotations

import sqlite3
import sys
from collections.abc import Iterable, Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legibility import trickle_state

LEDGER_FILENAME = 'coded-sessions.sqlite'

_CONNECT_TIMEOUT_SECS = 30.0
_UNUSABLE_SESSIONS = frozenset({'', 'unknown'})


class CodedBy(StrEnum):
    TRICKLE = 'trickle'
    CENSUS = 'census'


class Outcome(StrEnum):
    MATCHED = 'matched'
    CANDIDATE = 'candidate'
    EMPTY = 'empty'


class LedgerState(StrEnum):
    OK = 'ok'
    CREATED = 'created'
    ABSENT = 'absent'
    UNREADABLE = 'unreadable'


class LedgerError(Exception):
    """A ledger write failed; the message names the ledger path."""


def _sql_values(enum: type[StrEnum]) -> str:
    return ', '.join(f"'{member.value}'" for member in enum)


_DDL = f"""
CREATE TABLE IF NOT EXISTS coded_sessions (
    session TEXT PRIMARY KEY,
    instrument_version INTEGER NOT NULL,
    coded_by TEXT NOT NULL CHECK (coded_by IN ({_sql_values(CodedBy)})),
    run_ref TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ({_sql_values(Outcome)})),
    coded_at TEXT NOT NULL
)
"""

_INSERT = """
INSERT INTO coded_sessions
    (session, instrument_version, coded_by, run_ref, outcome, coded_at)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(session) DO NOTHING
"""

_SELECT = 'SELECT session, coded_by FROM coded_sessions'
_PRUNE = 'DELETE FROM coded_sessions WHERE coded_at < ?'


def ledger_path(project_id: str) -> Path:
    return trickle_state.project_state_dir(project_id) / LEDGER_FILENAME


def _is_usable_session(value: object) -> bool:
    return isinstance(value, str) and value not in _UNUSABLE_SESSIONS


def _is_aware(value: object) -> bool:
    return isinstance(value, datetime) and value.utcoffset() is not None


def _utc_text(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec='seconds')


def _require(ok: bool, field_name: str, value: object) -> None:
    if not ok:
        raise ValueError(f'LedgerRow.{field_name} is invalid: {value!r}')


@dataclass(frozen=True)
class LedgerRow:
    session: str
    instrument_version: int
    coded_by: CodedBy
    run_ref: str
    outcome: Outcome
    coded_at: datetime

    def __post_init__(self) -> None:
        _require(_is_usable_session(self.session), 'session', self.session)
        _require(
            isinstance(self.instrument_version, int)
            and not isinstance(self.instrument_version, bool),
            'instrument_version', self.instrument_version,
        )
        _require(isinstance(self.coded_by, CodedBy), 'coded_by', self.coded_by)
        _require(isinstance(self.run_ref, str) and bool(self.run_ref), 'run_ref', self.run_ref)
        _require(isinstance(self.outcome, Outcome), 'outcome', self.outcome)
        _require(_is_aware(self.coded_at), 'coded_at', self.coded_at)

    def as_params(self) -> tuple[str, int, str, str, str, str]:
        return (
            self.session,
            self.instrument_version,
            self.coded_by.value,
            self.run_ref,
            self.outcome.value,
            _utc_text(self.coded_at),
        )


def outcome_of(record: Mapping[str, Any]) -> Outcome:
    if record.get('candidates'):
        return Outcome.CANDIDATE
    if record.get('matches'):
        return Outcome.MATCHED
    return Outcome.EMPTY


def rows_for(
    records: Iterable[Mapping[str, Any]],
    *,
    coded_by: CodedBy,
    run_ref: str,
    instrument_version: int,
    coded_at: datetime,
) -> tuple[LedgerRow, ...]:
    """One row per coding record; records with no usable session are skipped."""
    return tuple(
        LedgerRow(
            session=record['session'],
            instrument_version=instrument_version,
            coded_by=coded_by,
            run_ref=run_ref,
            outcome=outcome_of(record),
            coded_at=coded_at,
        )
        for record in records
        if _is_usable_session(record.get('session'))
    )


@dataclass(frozen=True)
class LedgerSnapshot:
    path: Path
    state: LedgerState
    sessions: frozenset[str] = frozenset()
    rows_by_coded_by: Mapping[CodedBy, int] = field(
        default_factory=lambda: MappingProxyType({}),
    )
    pruned: int = 0
    error: str | None = None

    @property
    def total_rows(self) -> int | None:
        """``None`` when the ledger could not be counted — never 0."""
        if self.state in (LedgerState.ABSENT, LedgerState.UNREADABLE):
            return None
        return sum(self.rows_by_coded_by.values())

    @property
    def has_census_rows(self) -> bool:
        return self.rows_by_coded_by.get(CodedBy.CENSUS, 0) > 0


def _connect(path: Path, *, read_only: bool) -> sqlite3.Connection:
    if read_only:
        return sqlite3.connect(
            f'{path.absolute().as_uri()}?mode=ro',
            uri=True,
            timeout=_CONNECT_TIMEOUT_SECS,
        )
    return sqlite3.connect(path, timeout=_CONNECT_TIMEOUT_SECS, isolation_level=None)


@contextmanager
def _immediate_transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """Take the write lock up front so two writers wait instead of failing."""
    conn.execute('BEGIN IMMEDIATE')
    try:
        yield
    except BaseException:
        conn.execute('ROLLBACK')
        raise
    conn.execute('COMMIT')


def _read_snapshot(
    conn: sqlite3.Connection, path: Path, state: LedgerState, *, pruned: int = 0,
) -> LedgerSnapshot:
    rows = conn.execute(_SELECT).fetchall()
    counts = {member: 0 for member in CodedBy}
    for _, coded_by in rows:
        counts[CodedBy(coded_by)] += 1
    return LedgerSnapshot(
        path=path,
        state=state,
        sessions=frozenset(session for session, _ in rows),
        rows_by_coded_by=MappingProxyType(counts),
        pruned=pruned,
    )


def _unreadable(path: Path, exc: Exception) -> LedgerSnapshot:
    return LedgerSnapshot(path=path, state=LedgerState.UNREADABLE, error=f'{path}: {exc}')


def read_ledger(path: str | Path) -> LedgerSnapshot:
    """Read the ledger without ever creating or modifying it."""
    path = Path(path)
    if not path.exists():
        return LedgerSnapshot(path=path, state=LedgerState.ABSENT)
    try:
        with closing(_connect(path, read_only=True)) as conn:
            return _read_snapshot(conn, path, LedgerState.OK)
    except (sqlite3.Error, OSError, ValueError) as exc:
        return _unreadable(path, exc)


def open_for_census(path: str | Path, *, prune_before: datetime) -> LedgerSnapshot:
    """Create the ledger if absent, prune rows coded before *prune_before*,
    and read it. A file sqlite cannot open is reported UNREADABLE and left
    exactly as found."""
    path = Path(path)
    state = LedgerState.OK if path.exists() else LedgerState.CREATED
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(_connect(path, read_only=False)) as conn:
            with _immediate_transaction(conn):
                conn.execute(_DDL)
                pruned = conn.execute(_PRUNE, (_utc_text(prune_before),)).rowcount
            return _read_snapshot(conn, path, state, pruned=pruned)
    except (sqlite3.Error, OSError, ValueError) as exc:
        return _unreadable(path, exc)


def record_codings(path: str | Path, rows: Iterable[LedgerRow]) -> int:
    """Insert *rows*, creating the ledger if absent; a session already
    ledgered keeps its first row. Returns how many rows were inserted."""
    path = Path(path)
    params = [row.as_params() for row in rows]
    if not params:
        return 0
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with (
            closing(_connect(path, read_only=False)) as conn,
            _immediate_transaction(conn),
        ):
            conn.execute(_DDL)
            return conn.executemany(_INSERT, params).rowcount
    except (sqlite3.Error, OSError) as exc:
        raise LedgerError(f'session ledger {path}: {exc}') from exc
