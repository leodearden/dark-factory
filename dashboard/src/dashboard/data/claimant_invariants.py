"""Per-root claimant-invariant census read straight from a project's ``tasks.db``.

The record shape is contract E-1 of ``docs/prds/claimant-invariant-detection.md``
and its blind state is D-2: a root the gauge could not read reports ``blind``,
never a zero count. Which rows belong to which tier is decided wholly by the
predicates in ``shared.task_claimant``; this module only fetches rows and counts.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import aiosqlite
from shared.task_claimant import (
    DEFAULT_CLAIMANT_HEARTBEAT_TTL,
    is_hygiene_tier_claimant,
    is_stale_hygiene_tier_claimant,
    ledger_covers_claimant,
    violates_terminal_claimant_invariant,
)
from shared.task_metadata_wire import coerce_task_metadata

from dashboard.data.mcp_fanout import project_label

_COUNT_SQL = 'SELECT COUNT(*) FROM tasks'
# A necessary-condition prefilter: every shared tier predicate requires a carried claimant.
# The predicates still decide each fetched row (a blank-string claimant is fetched, then rejected).
_CANDIDATES_SQL = (
    'SELECT tag, id, status, claimant_run_id, heartbeat_at, metadata '
    'FROM tasks WHERE claimant_run_id IS NOT NULL ORDER BY tag, id'
)


class ScanStatus(StrEnum):
    OK = 'ok'
    BLIND = 'blind'


@dataclass(frozen=True)
class TerminalViolation:
    tag: str
    task_id: int
    status: str
    claimant_run_id: str
    heartbeat_at: str | None
    ledgered: bool


@dataclass(frozen=True)
class TerminalTier:
    violations: tuple[TerminalViolation, ...]

    @property
    def count(self) -> int:
        return len(self.violations)

    @property
    def unledgered_count(self) -> int:
        return sum(1 for violation in self.violations if not violation.ledgered)

    def to_wire(self) -> dict[str, Any]:
        return {
            'count': self.count,
            'unledgered_count': self.unledgered_count,
            'violations': [asdict(violation) for violation in self.violations],
        }


@dataclass(frozen=True)
class HygieneTier:
    count: int
    stale_count: int


@dataclass(frozen=True)
class RootInvariants:
    root: Path
    status: ScanStatus
    measured_at: datetime
    rows_scanned: int | None
    error: str | None
    terminal_tier: TerminalTier | None
    hygiene_tier: HygieneTier | None

    def __post_init__(self) -> None:
        problem = _blind_problem(self) if self.status is ScanStatus.BLIND else _ok_problem(self)
        if problem is not None:
            raise ValueError(f'RootInvariants status {self.status.value!r}: {problem}')

    def to_wire(self) -> dict[str, Any]:
        return {
            'root': str(self.root),
            'project': project_label(self.root),
            'status': self.status.value,
            'error': self.error,
            'rows_scanned': self.rows_scanned,
            'measured_at': self.measured_at.isoformat(),
            'terminal_tier': None if self.terminal_tier is None else self.terminal_tier.to_wire(),
            'hygiene_tier': None if self.hygiene_tier is None else asdict(self.hygiene_tier),
        }


def _blind_problem(record: RootInvariants) -> str | None:
    if not record.error:
        return f'error must be a non-empty str, got {record.error!r}'
    if record.rows_scanned not in (None, 0):
        return f'rows_scanned must be None or 0, got {record.rows_scanned!r}'
    if record.terminal_tier is not None:
        return 'terminal_tier must be None: a blind root makes no count claim'
    if record.hygiene_tier is not None:
        return 'hygiene_tier must be None: a blind root makes no count claim'
    return None


def _ok_problem(record: RootInvariants) -> str | None:
    if record.error is not None:
        return f'error must be None, got {record.error!r}'
    if record.rows_scanned is None or record.rows_scanned <= 0:
        return f'rows_scanned must be > 0, got {record.rows_scanned!r}'
    if record.terminal_tier is None:
        return 'terminal_tier must be present'
    if record.hygiene_tier is None:
        return 'hygiene_tier must be present'
    return None


def _blind(root: Path, now: datetime, error: str, rows_scanned: int | None = None) -> RootInvariants:
    return RootInvariants(
        root=root,
        status=ScanStatus.BLIND,
        measured_at=now,
        rows_scanned=rows_scanned,
        error=error,
        terminal_tier=None,
        hygiene_tier=None,
    )


def _unopened_reason(db_path: Path) -> str:
    if not db_path.exists():
        return f'tasks.db not found at {db_path}'
    return f'tasks.db at {db_path} could not be opened read-only (see dashboard log)'


def _as_task(row: Mapping[str, Any]) -> dict[str, Any]:
    return {**row, 'metadata': coerce_task_metadata(row['metadata']) or {}}


def _violation(task: Mapping[str, Any]) -> TerminalViolation:
    return TerminalViolation(
        tag=task['tag'],
        task_id=task['id'],
        status=task['status'],
        claimant_run_id=task['claimant_run_id'],
        heartbeat_at=task['heartbeat_at'],
        ledgered=ledger_covers_claimant(task),
    )


def classify_candidates(
    rows: Iterable[Mapping[str, Any]], *, now: datetime, ttl: timedelta
) -> tuple[TerminalTier, HygieneTier]:
    tasks = [_as_task(row) for row in rows]
    terminal = TerminalTier(
        tuple(_violation(task) for task in tasks if violates_terminal_claimant_invariant(task))
    )
    hygiene = HygieneTier(
        count=sum(1 for task in tasks if is_hygiene_tier_claimant(task)),
        stale_count=sum(1 for task in tasks if is_stale_hygiene_tier_claimant(task, now, ttl)),
    )
    return terminal, hygiene


async def _read_tasks(conn: aiosqlite.Connection) -> tuple[int, list[dict[str, Any]]]:
    (count_row,) = await conn.execute_fetchall(_COUNT_SQL)
    candidates = [dict(row) for row in await conn.execute_fetchall(_CANDIDATES_SQL)]
    return count_row[0], candidates


async def scan_root_invariants(
    root: Path,
    db_path: Path,
    conn: aiosqlite.Connection | None,
    *,
    now: datetime,
    ttl: timedelta = DEFAULT_CLAIMANT_HEARTBEAT_TTL,
) -> RootInvariants:
    if conn is None:
        return _blind(root, now, _unopened_reason(db_path))
    try:
        rows_scanned, candidates = await _read_tasks(conn)
    except (sqlite3.Error, OSError) as exc:
        return _blind(root, now, f'{type(exc).__name__}: {exc}')
    if rows_scanned == 0:
        return _blind(root, now, 'scanned 0 rows: nothing scanned is not clean', rows_scanned=0)
    terminal, hygiene = classify_candidates(candidates, now=now, ttl=ttl)
    return RootInvariants(
        root=root,
        status=ScanStatus.OK,
        measured_at=now,
        rows_scanned=rows_scanned,
        error=None,
        terminal_tier=terminal,
        hygiene_tier=hygiene,
    )
