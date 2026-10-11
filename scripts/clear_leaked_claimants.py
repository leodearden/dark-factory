#!/usr/bin/env python3
"""Census and repair of task-store claimants that own nothing (task 4625).

Two tiers, per docs/prds/claimant-invariant-enforcement.md D3: TERMINAL rows
carrying any claimant, and HYGIENE rows (an allowlisted non-terminal status)
carrying a stale one. A dry run is the default: it reports both tiers and never
dials the MCP server. --apply clears the terminal tier, plus the hygiene tier
with --repair-hygiene, through set_task_claimant only. The PRD's D6 and its
task zeta give the rationale for each gate.

Usage:
    python scripts/clear_leaked_claimants.py --project-root ROOT [--project-root ROOT ...]
    python scripts/clear_leaked_claimants.py --project-root ROOT --apply [--repair-hygiene]

Run it directly, never via `python -m`: it imports flat siblings in scripts/.
"""
from __future__ import annotations

import sqlite3
import sys
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from _task_db_scan import connect_ro, decode_metadata, tasks_db_path

# Bind shared/ and fused-memory/ to THIS checkout rather than whichever copy an
# editable install put on sys.path (scripts/audit_manifest_descriptor_drift.py::_SHARED_SRC).
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _src in (_REPO_ROOT / 'shared' / 'src', _REPO_ROOT / 'fused-memory' / 'src'):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from shared.task_claimant import (  # noqa: E402
    DEFAULT_CLAIMANT_HEARTBEAT_TTL,
    is_stale_hygiene_tier_claimant,
    violates_terminal_claimant_invariant,
)


class Tier(StrEnum):
    TERMINAL = 'terminal'
    HYGIENE = 'hygiene'


def classify_tier(task: Mapping, now: datetime) -> Tier | None:
    # A claimant with no heartbeat and no pid (reify 5225's shape) needs no special case: the
    # terminal predicate takes no TTL, and a missing heartbeat is stranded per
    # shared/src/shared/task_claimant.py::_claimant_liveness_stranded, so it never fails closed.
    if violates_terminal_claimant_invariant(task):
        return Tier.TERMINAL
    if is_stale_hygiene_tier_claimant(task, now, DEFAULT_CLAIMANT_HEARTBEAT_TTL):
        return Tier.HYGIENE
    return None


_ROW_COLUMNS = ('tag', 'id', 'status', 'claimant_run_id', 'heartbeat_at', 'metadata')


def _read_claimant_rows(
    conn: sqlite3.Connection, where: str, params: Sequence[Any] = (),
) -> list[dict[str, Any]]:
    sql = f"SELECT {', '.join(_ROW_COLUMNS)} FROM tasks WHERE {where} ORDER BY tag, id"
    rows = []
    for values in conn.execute(sql, params):
        row = dict(zip(_ROW_COLUMNS, values, strict=True))
        row['metadata'] = decode_metadata(row['metadata'])
        rows.append(row)
    return rows


@dataclass(frozen=True)
class Violation:
    tier: Tier
    tag: str
    task_id: int
    status: str
    claimant_run_id: str
    heartbeat_at: str | None
    measured_at: datetime

    @classmethod
    def from_row(cls, row: Mapping[str, Any], tier: Tier, measured_at: datetime) -> Violation:
        return cls(
            tier=tier,
            tag=row['tag'],
            task_id=row['id'],
            status=row['status'],
            claimant_run_id=row['claimant_run_id'],
            heartbeat_at=row['heartbeat_at'],
            measured_at=measured_at,
        )

    @property
    def key(self) -> tuple[str, int]:
        return (self.tag, self.task_id)

    def as_json(self) -> dict[str, Any]:
        return {
            'status': self.status,
            'tier': self.tier.value,
            'task_id': self.task_id,
            'tag': self.tag,
            'claimant_run_id': self.claimant_run_id,
            'heartbeat_at': self.heartbeat_at,
            'measured_at': self.measured_at.isoformat(),
        }


@dataclass(frozen=True)
class Census:
    project_root: str
    measured_at: datetime
    rows_in_store: int
    violations: tuple[Violation, ...]

    def totals(self) -> dict[str, dict[Tier, Any]]:
        by_tier = {tier: 0 for tier in Tier}
        by_tier_status: dict[Tier, dict[str, int]] = {tier: {} for tier in Tier}
        for violation in self.violations:
            by_tier[violation.tier] += 1
            statuses = by_tier_status[violation.tier]
            statuses[violation.status] = statuses.get(violation.status, 0) + 1
        return {'by_tier': by_tier, 'by_tier_status': by_tier_status}


def take_census(project_root: str, now: datetime) -> Census:
    with closing(connect_ro(tasks_db_path(project_root))) as conn:
        (rows_in_store,) = conn.execute('SELECT count(*) FROM tasks').fetchone()
        rows = _read_claimant_rows(conn, 'claimant_run_id IS NOT NULL')
    violations = tuple(
        Violation.from_row(row, tier, now)
        for row in rows
        if (tier := classify_tier(row, now)) is not None
    )
    return Census(project_root, now, rows_in_store, violations)
