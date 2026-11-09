"""``python -m shared.quality_series``: print a project's per-landing rows, and optionally a week summary."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
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
from shared.quality_series.rows import (
    LandingRow,
    QualityRowInvariantError,
    UnknownCharterClass,
    rows_for,
)
from shared.quality_series.summary import week_bounds, week_summary
from shared.tasks_db_readonly import TaskDbUnreadable, connect_ro

__all__ = ['main']

_REFUSALS = (
    GitUnavailable, GitObjectMissing, GitTimeout, CursorNotOnMain, UnrecognisedTrainEvent,
    UnknownCharterClass, QualityRowInvariantError, sqlite3.Error,
)
"""Every typed refusal of the producer, and sqlite's own errors: a reason on stderr and exit 2."""

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
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _iso_week(text: str) -> str:
    try:
        week_bounds(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'{text!r} is not an ISO week like 2026-W44') from exc
    return text


def _project_root(text: str) -> Path:
    """The root resolved, since the project id derives from its basename (``.`` has none)."""
    return Path(text).resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m shared.quality_series', description=__doc__)
    parser.add_argument('--root', required=True, type=_project_root,
                        help='project root (the git repo); a relative path is resolved')
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
    except (TaskDbUnreadable, sqlite3.Error) as exc:
        print(f'quality_series: runs.db is unreadable: {exc}', file=sys.stderr)
        return 2
    try:
        batch = landings(conn, args.root, since_cursor=args.since or None, main_ref=args.main_ref)
        rows, deferred = _measure_all(batch.landings, conn, args.root, args.git_budget_s)
    except _REFUSALS as exc:
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
    print(f'# next cursor: {_render(batch.next_cursor)}')
    if deferred:
        print(f'# deferred: {deferred}')
    if args.week is not None:
        for line in week_summary(rows, args.week).lines():
            print(line)
    return 0
