"""The per-root E-1 record of ``/api/v2/dashboard/invariants`` and its D-2 blind state.

Contract E-1 and decision D-2 are in ``docs/prds/claimant-invariant-detection.md``:
a root the gauge could not read reports ``blind``, never a zero count. The record
is composed from the sections defined below it, each deciding its own rows; this
module only reads them from a root's ``tasks.db`` and decides ok versus blind.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import aiosqlite
from shared.task_claimant import DEFAULT_CLAIMANT_HEARTBEAT_TTL

from dashboard.data.claimant_invariants import (
    HygieneTier,
    TerminalTier,
    classify_candidates,
    read_claimant_candidates,
)
from dashboard.data.escalation_corpus import PendingEscalations
from dashboard.data.recurrence_gauge import ChainEntry, classify_chains, read_chain_rows
from dashboard.project_labels import ProjectLabels

_COUNT_SQL = 'SELECT COUNT(*) FROM tasks'


class ScanStatus(StrEnum):
    OK = 'ok'
    BLIND = 'blind'


@dataclass(frozen=True)
class RootInvariants:
    root: Path
    status: ScanStatus
    measured_at: datetime
    rows_scanned: int | None
    error: str | None
    terminal_tier: TerminalTier | None
    hygiene_tier: HygieneTier | None
    chains: tuple[ChainEntry, ...] | None

    def __post_init__(self) -> None:
        problem = _blind_problem(self) if self.status is ScanStatus.BLIND else _ok_problem(self)
        if problem is not None:
            raise ValueError(f'RootInvariants status {self.status.value!r}: {problem}')

    def to_wire(self, labels: ProjectLabels) -> dict[str, Any]:
        return {
            'root': str(self.root),
            'project': labels.of(self.root),
            'status': self.status.value,
            'error': self.error,
            'rows_scanned': self.rows_scanned,
            'measured_at': self.measured_at.isoformat(),
            'terminal_tier': None if self.terminal_tier is None else self.terminal_tier.to_wire(),
            'hygiene_tier': None if self.hygiene_tier is None else self.hygiene_tier.to_wire(),
            'chains': None if self.chains is None else [chain.to_wire() for chain in self.chains],
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
    if record.chains is not None:
        return 'chains must be None: a blind root makes no chain claim'
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
    if record.chains is None:
        return 'chains must be present'
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
        chains=None,
    )


def _unopened_reason(db_path: Path) -> str:
    if not db_path.exists():
        return f'tasks.db not found at {db_path}'
    return f'tasks.db at {db_path} could not be opened read-only (see dashboard log)'


async def _read_row_count(conn: aiosqlite.Connection) -> int:
    (count_row,) = await conn.execute_fetchall(_COUNT_SQL)
    return count_row[0]


async def scan_root_invariants(
    root: Path,
    db_path: Path,
    conn: aiosqlite.Connection | None,
    *,
    now: datetime,
    pending_escalations: PendingEscalations | None,
    ttl: timedelta = DEFAULT_CLAIMANT_HEARTBEAT_TTL,
) -> RootInvariants:
    """Census one root. Never raises for a failed read: that root alone reports blind.

    The failure need not be a sqlite or OS error; aiosqlite raises ``ValueError`` on
    a pooled connection closed under the scan. A caller fanning out over the
    roster therefore never loses the other roots' records to one root's failure.

    *pending_escalations* is the root's escalation queue read by task; None is unknown.
    """
    if conn is None:
        return _blind(root, now, _unopened_reason(db_path))
    try:
        rows_scanned = await _read_row_count(conn)
        candidates = await read_claimant_candidates(conn)
        chain_rows = await read_chain_rows(conn)
    except Exception as exc:
        return _blind(root, now, f'{type(exc).__name__}: {exc}')
    if rows_scanned == 0:
        return _blind(root, now, 'scanned 0 rows: nothing scanned is not clean', rows_scanned=0)
    terminal, hygiene = classify_candidates(candidates, now=now, ttl=ttl)
    chains = classify_chains(chain_rows, now=now, pending_escalations=pending_escalations)
    return RootInvariants(
        root=root,
        status=ScanStatus.OK,
        measured_at=now,
        rows_scanned=rows_scanned,
        error=None,
        terminal_tier=terminal,
        hygiene_tier=hygiene,
        chains=chains,
    )
