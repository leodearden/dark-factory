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

import argparse
import asyncio
import json
import sqlite3
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any

from _task_db_scan import (
    UNREADABLE_STORE_ERRORS,
    connect_ro,
    decode_metadata,
    format_kv_line,
    tasks_db_path,
)

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


@dataclass(frozen=True)
class ProjectReport:
    census: Census
    apply: object | None = None


@dataclass(frozen=True)
class SweepResult:
    applied: bool
    repair_tiers: tuple[Tier, ...]
    projects: tuple[ProjectReport, ...]
    unreadable: tuple[str, ...]

    @property
    def mode(self) -> str:
        return 'apply' if self.applied else 'dry-run'


async def sweep(
    roots: Iterable[str],
    *,
    apply: bool,
    tiers: frozenset[Tier],
    client: object | None,
    clock: Callable[[], datetime],
) -> SweepResult:
    projects: list[ProjectReport] = []
    unreadable: list[str] = []
    for root in roots:
        try:
            census = take_census(root, clock())
        except UNREADABLE_STORE_ERRORS as exc:
            print(f'warning: skipping unreadable project {root}: {exc}', file=sys.stderr)
            unreadable.append(root)
            continue
        projects.append(ProjectReport(census))
    return SweepResult(
        applied=apply,
        repair_tiers=tuple(tier for tier in Tier if tier in tiers),
        projects=tuple(projects),
        unreadable=tuple(unreadable),
    )


def _census_lines(census: Census) -> list[str]:
    totals = census.totals()
    lines = [
        format_kv_line([('rows_in_store', census.rows_in_store)]),
        format_kv_line(list(totals['by_tier'].items()), indent='  totals: '),
    ]
    for tier, statuses in totals['by_tier_status'].items():
        lines.append(format_kv_line(sorted(statuses.items()), indent=f'  {tier}: ').rstrip())
    lines.extend(_violation_line(violation) for violation in census.violations)
    return lines


def _violation_line(violation: Violation, *, indent: str = '    ') -> str:
    return format_kv_line(
        [
            ('tier', violation.tier),
            ('task_id', violation.task_id),
            ('tag', violation.tag),
            ('status', violation.status),
            ('claimant_run_id', violation.claimant_run_id),
            ('heartbeat_at', violation.heartbeat_at),
        ],
        indent=indent,
    )


def render_text(result: SweepResult) -> str:
    lines: list[str] = []
    for project in result.projects:
        census = project.census
        lines.append(format_kv_line(
            [
                ('project', census.project_root),
                ('mode', result.mode),
                ('measured_at', census.measured_at.isoformat()),
            ],
            indent='',
        ))
        lines.extend(_census_lines(census))
        if not result.applied:
            would_clear = ' '.join(result.repair_tiers)
            lines.append(f'  dry-run: nothing was written; --apply would clear tiers: {would_clear}')
    lines.extend(f'unreadable: {root}' for root in result.unreadable)
    return '\n'.join(lines)


def _census_json(census: Census) -> dict[str, Any]:
    return {
        'measured_at': census.measured_at.isoformat(),
        'rows_in_store': census.rows_in_store,
        'totals': census.totals(),
        'violations': [violation.as_json() for violation in census.violations],
    }


def render_json(result: SweepResult) -> str:
    return json.dumps(
        {
            'mode': result.mode,
            'repair_tiers': list(result.repair_tiers),
            'projects': [
                {
                    'project_root': project.census.project_root,
                    'census': _census_json(project.census),
                    'apply': None,
                }
                for project in result.projects
            ],
            'unreadable': list(result.unreadable),
        },
        indent=2,
    )


class ExitCode(IntEnum):
    OK = 0
    REPAIR_FAILED = 1
    USAGE = 2
    NOTHING_READ = 3
    SERVER_UNREACHABLE = 4
    HOLDER_REFUSED = 5


_EXIT_CODE_MEANINGS = {
    ExitCode.OK: 'ran and no repair failed; a dry run is always 0, whatever it found',
    ExitCode.REPAIR_FAILED: 'at least one row landed in the failed bucket',
    ExitCode.USAGE: 'bad arguments',
    ExitCode.NOTHING_READ: 'every --project-root was unreadable, so nothing was examined',
    ExitCode.SERVER_UNREACHABLE: '--apply, but the MCP server could not be reached; nothing was written',
    ExitCode.HOLDER_REFUSED: (
        '--apply refused a project because a selected row is a current lock holder; '
        'that project was not written'
    ),
}


def exit_code(result: SweepResult) -> ExitCode:
    if not result.projects:
        return ExitCode.NOTHING_READ
    return ExitCode.OK


DEFAULT_SERVER_URL = 'http://127.0.0.1:8002'


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Census, and with --apply repair, of task claimants that own nothing. '
            'Dry run by default: nothing is written and the MCP server is never dialled.'
        ),
        epilog='exit codes:\n' + '\n'.join(
            f'  {code.value}  {meaning}' for code, meaning in _EXIT_CODE_MEANINGS.items()
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--project-root', dest='project_roots', action='append', required=True,
        help='Project whose <root>/.taskmaster/tasks/tasks.db to examine. Repeatable.',
    )
    parser.add_argument(
        '--apply', action='store_true',
        help='Clear the selected tiers through set_task_claimant. Without it nothing is written.',
    )
    parser.add_argument(
        '--repair-hygiene', action='store_true',
        help='Also clear the hygiene tier on --apply. The terminal tier is always selected.',
    )
    parser.add_argument('--json', action='store_true', help='Emit one JSON object.')
    parser.add_argument(
        '--server-url', default=DEFAULT_SERVER_URL,
        help=f'Fused-memory MCP server URL (default: {DEFAULT_SERVER_URL}).',
    )
    return parser


def _utcnow() -> datetime:
    return datetime.now(UTC)


async def main_async(args: argparse.Namespace) -> int:
    tiers = frozenset({Tier.TERMINAL, Tier.HYGIENE} if args.repair_hygiene else {Tier.TERMINAL})
    result = await sweep(args.project_roots, apply=False, tiers=tiers, client=None, clock=_utcnow)
    code = exit_code(result)
    if code is ExitCode.NOTHING_READ:
        print(f'error: {_EXIT_CODE_MEANINGS[code]} (not a clean result)', file=sys.stderr)
    print(render_json(result) if args.json else render_text(result))
    return code


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(main_async(_build_parser().parse_args(argv)))


if __name__ == '__main__':
    sys.exit(main())
