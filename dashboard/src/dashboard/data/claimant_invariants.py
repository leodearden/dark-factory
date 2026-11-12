"""The claimant tiers of a root's E-1 record, read straight from its ``tasks.db``.

Contract E-1 of ``docs/prds/claimant-invariant-detection.md`` names two tiers: the
terminal tier (a claimant left on a terminal row) and the hygiene tier. Which rows
belong to which tier is decided wholly by the predicates in ``shared.task_claimant``;
this module only fetches candidate rows and counts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

import aiosqlite
from shared.task_claimant import (
    is_hygiene_tier_claimant,
    is_stale_hygiene_tier_claimant,
    ledger_covers_claimant,
    violates_terminal_claimant_invariant,
)
from shared.task_metadata_wire import coerce_task_metadata

# A necessary-condition prefilter: every shared tier predicate requires a carried claimant.
# The predicates still decide each fetched row (a blank-string claimant is fetched, then rejected).
_CANDIDATES_SQL = (
    'SELECT tag, id, status, claimant_run_id, heartbeat_at, metadata '
    'FROM tasks WHERE claimant_run_id IS NOT NULL ORDER BY tag, id'
)


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

    def to_wire(self) -> dict[str, Any]:
        return {'count': self.count, 'stale_count': self.stale_count}


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


async def read_claimant_candidates(conn: aiosqlite.Connection) -> list[dict[str, Any]]:
    return [dict(row) for row in await conn.execute_fetchall(_CANDIDATES_SQL)]
