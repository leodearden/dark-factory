"""``python -m shared.quality_series``: print a project's per-landing rows, and optionally a week summary."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path

from shared.project_layout import runs_db
from shared.quality_series.git_lines import GitObjectMissing, GitTimeout, GitUnavailable
from shared.quality_series.landings import (
    CursorNotOnMain,
    Landing,
    UnrecognisedTrainEvent,
    landings,
)
from shared.quality_series.rows import LandingRow, ReviewRoundsSource, rows_for, unit_heads
from shared.tasks_db_readonly import TaskDbUnreadable, connect_ro

__all__ = ['KOfN', 'WeekSummary', 'main', 'week_summary']

_COLUMNS: tuple[tuple[str, Callable[[LandingRow], object]], ...] = (
    ('ordinal', lambda r: r.landing.ordinal),
    ('landed_at', lambda r: r.landing.landed_at),
    ('task_id', lambda r: r.landing.task_id),
    ('merge_sha', lambda r: r.landing.merge_sha),
    ('source', lambda r: r.landing.source),
    ('unit_id', lambda r: r.landing.unit_id),
    ('unit_size', lambda r: r.landing.unit_size),
    ('git_first_parent', lambda r: r.landing.git_first_parent),
    ('lines_base', lambda r: r.landing.lines_base),
    ('rejected', lambda r: r.outcomes.rejected),
    ('rejection_events', lambda r: r.outcomes.rejection_events),
    ('reviewed', lambda r: r.outcomes.reviewed),
    ('review_blocking', lambda r: r.outcomes.review_blocking),
    ('flake_confirmed', lambda r: r.outcomes.flake_confirmed),
    ('flake_real_red', lambda r: r.outcomes.flake_real_red),
    ('flake_rows', lambda r: r.outcomes.flake_rows),
    ('reverify_count', lambda r: r.outcomes.reverify_count),
    ('verify_failed_rows', lambda r: r.outcomes.verify_failed_rows),
    ('all_role_usd', lambda r: r.cost.all_role_usd),
    ('implementer_invocations', lambda r: r.cost.implementer_invocations),
    ('turns', lambda r: r.cost.turns),
    ('turns_capped', lambda r: r.cost.turns_capped),
    ('execute_iterations', lambda r: r.cost.execute_iterations),
    ('review_rounds', lambda r: r.cost.review_rounds),
    ('review_rounds_source', lambda r: r.cost.review_rounds_source),
    ('charter_class', lambda r: r.cost.charter_class),
    ('code_lines', lambda r: r.cost.code_lines),
    ('src_lines', lambda r: r.cost.src_lines),
    ('test_lines', lambda r: r.cost.test_lines),
)

ANCHORING = (
    '# anchoring: landed_at is the merge_finalized done or train_merged event timestamp; '
    'every source is attributed over (previous landing, landing] of the same task'
)


def _render(value: object) -> str:
    if value is None:
        return 'none'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, float):
        return f'{value:.4f}'
    return str(value)


@dataclass(frozen=True)
class KOfN:
    k: int
    n: int

    def __str__(self) -> str:
        return f'{self.k}/{self.n}'


@dataclass(frozen=True)
class WeekSummary:
    """One ISO week's figures; a landing whose value is unknown (None) drops out of that figure.

    Event counts (rejection events, re-verifies) are summed once per unit over unit heads, since a
    unit's value is copied to every member; every rate's denominator is the week's landings whose
    value is known.
    """

    week: str
    units: int
    landings: int
    rejection_events: KOfN
    rejected_landings: KOfN
    flake_confirmed: KOfN
    flake_real_red: KOfN
    review_rounds_ge2: KOfN
    review_rounds_sources: tuple[tuple[ReviewRoundsSource, int], ...]
    reverifies: KOfN

    def lines(self) -> list[str]:
        sources = ', '.join(f'{source.value}={count}' for source, count in self.review_rounds_sources)
        return [
            f'# week {self.week}: {self.units} units, {self.landings} landings',
            '#   rejections per 100 landings: '
            f'{_per(self.rejection_events, 100, 1)} '
            f'({self.rejection_events.k} events over {self.rejection_events.n} landings)',
            f'#   landings with >= 1 rejection: {self.rejected_landings}',
            f'#   confirmed-flake landings: {self.flake_confirmed}',
            f'#   real-red landings: {self.flake_real_red}',
            f'#   review rounds >= 2: {self.review_rounds_ge2} ({sources})',
            f'#   re-verifies per landing: {_per(self.reverifies, 1, 2)} '
            f'({self.reverifies.k} over {self.reverifies.n} landings)',
        ]


def _per(ratio: KOfN, scale: int, digits: int) -> str:
    return 'none' if ratio.n == 0 else f'{scale * ratio.k / ratio.n:.{digits}f}'


def _week_bounds(iso_week: str) -> tuple[datetime, datetime]:
    monday = datetime.strptime(f'{iso_week}-1', '%G-W%V-%u').replace(tzinfo=UTC)
    return monday, monday + timedelta(days=7)


def _count(values: Sequence[bool | None]) -> KOfN:
    known = [value for value in values if value is not None]
    return KOfN(sum(1 for value in known if value), len(known))


def _unit_events_over_landings(
    week: Sequence[LandingRow], value: Callable[[LandingRow], int | None]
) -> KOfN:
    events = [value(head) for head in unit_heads(week)]
    known_landings = sum(1 for row in week if value(row) is not None)
    return KOfN(sum(event for event in events if event is not None), known_landings)


def week_summary(rows: Sequence[LandingRow], iso_week: str) -> WeekSummary:
    start, end = _week_bounds(iso_week)
    week = [row for row in rows if start <= row.landing.landed_time < end]
    heads = unit_heads(week)
    rounds = [row.cost for row in week if row.cost.review_rounds is not None]
    return WeekSummary(
        week=iso_week,
        units=len(heads),
        landings=len(week),
        rejection_events=_unit_events_over_landings(week, lambda r: r.outcomes.rejection_events),
        rejected_landings=_count([row.outcomes.rejected for row in week]),
        flake_confirmed=_count([row.outcomes.flake_confirmed for row in week]),
        flake_real_red=_count([row.outcomes.flake_real_red for row in week]),
        review_rounds_ge2=KOfN(sum(1 for cost in rounds if (cost.review_rounds or 0) >= 2),
                               len(rounds)),
        review_rounds_sources=tuple(
            (source, sum(1 for cost in rounds if cost.review_rounds_source is source))
            for source in ReviewRoundsSource
        ),
        reverifies=_unit_events_over_landings(week, lambda r: r.outcomes.reverify_count),
    )


def _iso_week(text: str) -> str:
    try:
        _week_bounds(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'{text!r} is not an ISO week like 2026-W44') from exc
    return text


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m shared.quality_series', description=__doc__)
    parser.add_argument('--root', required=True, type=Path, help='project root (the git repo)')
    parser.add_argument('--since', default=None, help='cursor sha on main; omit for full history')
    parser.add_argument('--week', type=_iso_week, default=None, help='ISO week, e.g. 2026-W44')
    parser.add_argument('--git-budget-s', type=float, default=600.0)
    parser.add_argument('--main-ref', default='main')
    return parser


def _measure_all(
    pending: Sequence[Landing], conn: sqlite3.Connection, root: Path, git_budget_s: float
) -> tuple[list[LandingRow], int]:
    measured: list[LandingRow] = []
    while pending:
        rows = rows_for(pending, conn, root, git_budget_s)
        if not rows:
            break
        measured.extend(rows)
        pending = pending[len(rows):]
    return measured, len(pending)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        conn = connect_ro(runs_db(args.root))
    except TaskDbUnreadable as exc:
        print(f'quality_series: runs.db is unreadable: {exc}', file=sys.stderr)
        return 2
    try:
        batch = landings(conn, args.root, since_cursor=args.since or None, main_ref=args.main_ref)
        rows, deferred = _measure_all(batch.landings, conn, args.root, args.git_budget_s)
    except (
        GitUnavailable, GitObjectMissing, GitTimeout, CursorNotOnMain, UnrecognisedTrainEvent,
    ) as exc:
        print(f'quality_series: {exc}', file=sys.stderr)
        return 2
    finally:
        conn.close()

    print('# columns: ' + '\t'.join(name for name, _ in _COLUMNS))
    for row in rows:
        print('\t'.join(_render(value(row)) for _, value in _COLUMNS))
    check = batch.git_check
    print(f'# git first-parent disagreement: {check.disagreement} '
          f'(off_first_parent={check.off_first_parent}, '
          f'merges_without_landing={check.merges_without_landing}, '
          f'late_claims={check.late_claims})')
    print(ANCHORING)
    if deferred:
        print(f'# deferred: {deferred}')
    if args.week is not None:
        for line in week_summary(rows, args.week).lines():
            print(line)
    return 0
