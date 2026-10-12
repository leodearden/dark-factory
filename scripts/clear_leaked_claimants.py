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
import contextlib
import json
import sqlite3
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any, Protocol

import httpx
from _task_db_scan import (
    UNREADABLE_STORE_ERRORS,
    connect_ro,
    decode_metadata,
    format_kv_line,
    tasks_db_path,
)
from migrate_metadata_modules_to_files import FusedMemoryClient, write_failure_reason

# Bind shared/ and fused-memory/ to THIS checkout rather than whichever copy an
# editable install put on sys.path (scripts/audit_manifest_descriptor_drift.py::_SHARED_SRC).
# Bare sys.path.insert statements, which ruff's E402 exempts, so the imports below need no noqa.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'fused-memory' / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'shared' / 'src'))

# A stdlib-only module under empty package __init__s, so the scripts env can import it.
from fused_memory.mcp_tools.scheduler_state import read_scheduler_state
from shared.task_claimant import (
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
class CensusTotals:
    by_tier: Mapping[Tier, int]
    by_tier_status: Mapping[Tier, Mapping[str, int]]

    def as_json(self) -> dict[str, Any]:
        return {
            'by_tier': dict(self.by_tier),
            'by_tier_status': {tier: dict(statuses) for tier, statuses in self.by_tier_status.items()},
        }


@dataclass(frozen=True)
class Census:
    project_root: str
    measured_at: datetime
    rows_in_store: int
    violations: tuple[Violation, ...]

    def totals(self) -> CensusTotals:
        by_tier = {tier: 0 for tier in Tier}
        by_tier_status: dict[Tier, dict[str, int]] = {tier: {} for tier in Tier}
        for violation in self.violations:
            by_tier[violation.tier] += 1
            statuses = by_tier_status[violation.tier]
            statuses[violation.status] = statuses.get(violation.status, 0) + 1
        return CensusTotals(by_tier=by_tier, by_tier_status=by_tier_status)


def _ordered(tiers: frozenset[Tier]) -> tuple[Tier, ...]:
    return tuple(tier for tier in Tier if tier in tiers)


def take_census(project_root: str, now: datetime) -> Census:
    with contextlib.closing(connect_ro(tasks_db_path(project_root))) as conn:
        (rows_in_store,) = conn.execute('SELECT count(*) FROM tasks').fetchone()
        rows = _read_claimant_rows(conn, 'claimant_run_id IS NOT NULL')
    violations = tuple(
        Violation.from_row(row, tier, now)
        for row in rows
        if (tier := classify_tier(row, now)) is not None
    )
    return Census(project_root, now, rows_in_store, violations)


class ToolClient(Protocol):
    async def call_tool(self, name: str, arguments: dict) -> dict: ...


class Bucket(StrEnum):
    REPAIRED = 'repaired'
    SKIPPED_CHANGED_UNDER_US = 'skipped_changed_under_us'
    ARRIVED_DURING_WINDOW = 'arrived_during_window'
    FAILED = 'failed'


Delta = Mapping[str, tuple[Any, Any]]


@dataclass(frozen=True)
class RowOutcome:
    violation: Violation
    bucket: Bucket
    delta: Delta | None = None
    detail: str | None = None


@dataclass(frozen=True)
class ApplyReport:
    project_root: str
    tiers: tuple[Tier, ...]
    refused: bool
    holder_intersection: tuple[int, ...]
    outcomes: tuple[RowOutcome, ...]
    verification_census: Census | None
    verification_error: str | None

    def counts(self) -> dict[Bucket, int]:
        counts = {bucket: 0 for bucket in Bucket}
        for outcome in self.outcomes:
            counts[outcome.bucket] += 1
        return counts


@dataclass(frozen=True)
class _Attempt:
    """delta: the re-read declined the row. error: the re-read or the write failed. Neither: written."""

    delta: Delta | None = None
    error: str | None = None


STILL_VIOLATING_DETAIL = 'still violating at verification census'
REREAD_FAILED_DETAIL = 're-read failed, nothing was written'
UNVERIFIED_DETAIL = 'written, but the verification census could not read the store'


def _describe(exc: BaseException) -> str:
    return f'{type(exc).__name__}: {exc}'


async def _clear_claimant(client: ToolClient, project_root: str, violation: Violation) -> str | None:
    arguments = {
        'id': str(violation.task_id),
        'tag': violation.tag,
        'project_root': project_root,
        'claimant_run_id': None,
        'heartbeat_at': None,
    }
    try:
        reply = await client.call_tool('set_task_claimant', arguments)
    except Exception as exc:
        return _describe(exc)
    return write_failure_reason(reply)


def holder_task_ids(project_root: str) -> frozenset[str]:
    state = read_scheduler_state(Path(project_root))
    holders = state.get('current_holders') if isinstance(state, Mapping) else None
    if not isinstance(holders, Mapping):
        return frozenset()
    return frozenset(str(task_id) for task_id in holders.values())


def observed_delta(
    violation: Violation, live: Mapping[str, Any] | None, now: datetime,
) -> dict[str, tuple[Any, Any]] | None:
    if live is None:
        return {'row': ('present', 'absent')}
    census_values = {
        'status': violation.status,
        'claimant_run_id': violation.claimant_run_id,
        'heartbeat_at': violation.heartbeat_at,
    }
    delta = {
        field: (census_value, live[field])
        for field, census_value in census_values.items()
        if census_value != live[field]
    }
    live_tier = classify_tier(live, now)
    if live_tier is not violation.tier:
        delta['tier'] = (violation.tier.value, None if live_tier is None else live_tier.value)
    return delta or None


def _read_live_row(project_root: str, key: tuple[str, int]) -> dict[str, Any] | None:
    with contextlib.closing(connect_ro(tasks_db_path(project_root))) as conn:
        rows = _read_claimant_rows(conn, 'tag = ? AND id = ?', key)
    return rows[0] if rows else None


async def _attempt_repair(
    client: ToolClient, project_root: str, violation: Violation, now: datetime,
) -> _Attempt:
    try:
        live = _read_live_row(project_root, violation.key)
    except UNREADABLE_STORE_ERRORS as exc:
        return _Attempt(error=f'{REREAD_FAILED_DETAIL}: {_describe(exc)}')
    delta = observed_delta(violation, live, now)
    if delta is not None:
        return _Attempt(delta=delta)
    return _Attempt(error=await _clear_claimant(client, project_root, violation))


def _outcome(
    violation: Violation, attempt: _Attempt, still_violating: frozenset[tuple[str, int]] | None,
) -> RowOutcome:
    if attempt.delta is not None:
        return RowOutcome(violation, Bucket.SKIPPED_CHANGED_UNDER_US, delta=attempt.delta)
    if attempt.error is not None:
        return RowOutcome(violation, Bucket.FAILED, detail=attempt.error)
    if still_violating is None:
        return RowOutcome(violation, Bucket.FAILED, detail=UNVERIFIED_DETAIL)
    if violation.key in still_violating:
        return RowOutcome(violation, Bucket.FAILED, detail=STILL_VIOLATING_DETAIL)
    return RowOutcome(violation, Bucket.REPAIRED)


def bucket_outcomes(
    selected: Sequence[Violation],
    attempts: Mapping[tuple[str, int], _Attempt],
    verification: Census | None,
    tiers: frozenset[Tier],
) -> tuple[RowOutcome, ...]:
    if verification is None:
        return tuple(_outcome(v, attempts[v.key], None) for v in selected)
    remaining = tuple(v for v in verification.violations if v.tier in tiers)
    still_violating = frozenset(v.key for v in remaining)
    enumerated = frozenset(v.key for v in selected)
    return (
        *(_outcome(v, attempts[v.key], still_violating) for v in selected),
        *(RowOutcome(v, Bucket.ARRIVED_DURING_WINDOW) for v in remaining if v.key not in enumerated),
    )


async def apply_repairs(
    census: Census,
    client: ToolClient | None,
    *,
    tiers: frozenset[Tier],
    clock: Callable[[], datetime],
) -> ApplyReport:
    if client is None:
        raise ValueError('apply_repairs writes through an MCP client; none was given')
    root = census.project_root
    selected = tuple(v for v in census.violations if v.tier in tiers)
    holders = holder_task_ids(root)
    held = tuple(sorted(v.task_id for v in selected if str(v.task_id) in holders))
    if held:
        return ApplyReport(
            project_root=root,
            tiers=_ordered(tiers),
            refused=True,
            holder_intersection=held,
            outcomes=(),
            verification_census=None,
            verification_error=None,
        )
    attempts = {
        violation.key: await _attempt_repair(client, root, violation, clock())
        for violation in selected
    }
    try:
        verification, verification_error = take_census(root, clock()), None
    except UNREADABLE_STORE_ERRORS as exc:
        verification, verification_error = None, _describe(exc)
    return ApplyReport(
        project_root=root,
        tiers=_ordered(tiers),
        refused=False,
        holder_intersection=(),
        outcomes=bucket_outcomes(selected, attempts, verification, tiers),
        verification_census=verification,
        verification_error=verification_error,
    )


@dataclass(frozen=True)
class ProjectReport:
    census: Census
    apply: ApplyReport | None = None


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
    client: ToolClient | None,
    clock: Callable[[], datetime],
) -> SweepResult:
    if apply and client is None:
        raise ValueError('an applying sweep writes through an MCP client; none was given')
    projects: list[ProjectReport] = []
    unreadable: list[str] = []
    for root in roots:
        try:
            census = take_census(root, clock())
        except UNREADABLE_STORE_ERRORS as exc:
            print(f'warning: skipping unreadable project {root}: {exc}', file=sys.stderr)
            unreadable.append(root)
            continue
        report = await apply_repairs(census, client, tiers=tiers, clock=clock) if apply else None
        projects.append(ProjectReport(census, report))
    return SweepResult(
        applied=apply,
        repair_tiers=_ordered(tiers),
        projects=tuple(projects),
        unreadable=tuple(unreadable),
    )


def _census_lines(census: Census) -> list[str]:
    totals = census.totals()
    lines = [
        format_kv_line([('rows_in_store', census.rows_in_store)]),
        format_kv_line(list(totals.by_tier.items()), indent='  totals: '),
    ]
    for tier, statuses in totals.by_tier_status.items():
        lines.append(format_kv_line(sorted(statuses.items()), indent=f'  {tier}: ').rstrip())
    lines.extend(format_kv_line(_violation_pairs(v), indent='    ') for v in census.violations)
    return lines


def _violation_pairs(violation: Violation) -> list[tuple[str, Any]]:
    return [
        ('tier', violation.tier),
        ('task_id', violation.task_id),
        ('tag', violation.tag),
        ('status', violation.status),
        ('claimant_run_id', violation.claimant_run_id),
        ('heartbeat_at', violation.heartbeat_at),
    ]


def _outcome_line(outcome: RowOutcome) -> str:
    delta = outcome.delta or {}
    pairs = [
        ('bucket', outcome.bucket),
        *_violation_pairs(outcome.violation),
        *((f'delta.{field}', f'{before}->{after}') for field, (before, after) in delta.items()),
    ]
    if outcome.detail is not None:
        pairs.append(('detail', outcome.detail))
    return format_kv_line(pairs, indent='    ')


def _apply_lines(report: ApplyReport) -> list[str]:
    if report.refused:
        held = ' '.join(str(task_id) for task_id in report.holder_intersection)
        return [f'  apply: refused, nothing was written; current lock holders among the selected rows: {held}']
    lines = [format_kv_line(list(report.counts().items()), indent='  buckets: ')]
    lines.extend(_outcome_line(outcome) for outcome in report.outcomes)
    if report.verification_census is not None:
        verification = report.verification_census
        lines.append(f'  verification_census: measured_at={verification.measured_at.isoformat()}')
        lines.extend(_census_lines(verification))
    if report.verification_error is not None:
        lines.append(f'  verification_census: unavailable, {report.verification_error}')
    return lines


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
        if project.apply is not None:
            lines.extend(_apply_lines(project.apply))
        elif not result.applied:
            would_clear = ' '.join(result.repair_tiers)
            lines.append(f'  dry-run: nothing was written; --apply would clear tiers: {would_clear}')
    lines.extend(f'unreadable: {root}' for root in result.unreadable)
    return '\n'.join(lines)


def _census_json(census: Census) -> dict[str, Any]:
    return {
        'measured_at': census.measured_at.isoformat(),
        'rows_in_store': census.rows_in_store,
        'totals': census.totals().as_json(),
        'violations': [violation.as_json() for violation in census.violations],
    }


def _outcome_json(outcome: RowOutcome) -> dict[str, Any]:
    row = outcome.violation.as_json()
    if outcome.delta is not None:
        row['delta'] = dict(outcome.delta)
    if outcome.detail is not None:
        row['detail'] = outcome.detail
    return row


def _apply_json(report: ApplyReport) -> dict[str, Any]:
    verification = report.verification_census
    return {
        'refused': report.refused,
        'holder_intersection': list(report.holder_intersection),
        'counts': report.counts(),
        'buckets': {
            bucket: [_outcome_json(o) for o in report.outcomes if o.bucket is bucket]
            for bucket in Bucket
        },
        'verification_census': None if verification is None else _census_json(verification),
        'verification_error': report.verification_error,
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
                    'apply': None if project.apply is None else _apply_json(project.apply),
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


def exit_conditions(result: SweepResult) -> tuple[ExitCode, ...]:
    """Every non-OK condition present, highest precedence first."""
    if not result.projects:
        return (ExitCode.NOTHING_READ,)
    reports = [project.apply for project in result.projects if project.apply is not None]
    conditions = {
        ExitCode.REPAIR_FAILED: any(report.counts()[Bucket.FAILED] for report in reports),
        ExitCode.HOLDER_REFUSED: any(report.refused for report in reports),
    }
    return tuple(code for code, present in conditions.items() if present)


def exit_code(result: SweepResult) -> ExitCode:
    conditions = exit_conditions(result)
    return conditions[0] if conditions else ExitCode.OK


def emit_report(result: SweepResult, *, as_json: bool) -> ExitCode:
    print(render_json(result) if as_json else render_text(result))
    for condition in exit_conditions(result):
        print(f'error: {_EXIT_CODE_MEANINGS[condition]}', file=sys.stderr)
    return exit_code(result)


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


CLIENT_NAME = 'clear-leaked-claimants-4625'


class ClaimantRepairClient(FusedMemoryClient):
    _client_name = CLIENT_NAME


_HANDSHAKE_ERRORS = (httpx.HTTPError, OSError, RuntimeError)


async def _applying_sweep(
    server_url: str, roots: Sequence[str], tiers: frozenset[Tier],
) -> SweepResult | None:
    async with contextlib.AsyncExitStack() as stack:
        try:
            client = await stack.enter_async_context(ClaimantRepairClient(server_url))
        except _HANDSHAKE_ERRORS as exc:
            print(
                f'error: could not reach the fused-memory MCP server at {server_url}: '
                f'{_describe(exc)}; nothing was written',
                file=sys.stderr,
            )
            return None
        return await sweep(roots, apply=True, tiers=tiers, client=client, clock=_utcnow)


async def main_async(args: argparse.Namespace) -> int:
    tiers = frozenset({Tier.TERMINAL, Tier.HYGIENE} if args.repair_hygiene else {Tier.TERMINAL})
    if args.apply:
        result = await _applying_sweep(args.server_url, args.project_roots, tiers)
        if result is None:
            return ExitCode.SERVER_UNREACHABLE
    else:
        result = await sweep(args.project_roots, apply=False, tiers=tiers, client=None, clock=_utcnow)
    return emit_report(result, as_json=args.json)


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(main_async(_build_parser().parse_args(argv)))


if __name__ == '__main__':
    sys.exit(main())
