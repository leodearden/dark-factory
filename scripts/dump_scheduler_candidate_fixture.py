#!/usr/bin/env python3
"""Dump the live pending population as the scheduler's dispatch scorer sees it.

Writes one JSON fixture: a provenance block, then every pending task of one tag
with the four inputs the C2 score reads besides the clock — id, tier, the
verbatim ``pending_since`` anchor (null when absent) and the transitive
dependent count D (plans/scheduler-dispatch-scoring-and-lock-layer-prd.md).
D comes from the scheduler's own ``Scheduler._compute_transitive_counts``, so
the fixture's D is by definition the D a live tick scores on.

Strictly READ-ONLY against the store; refuses loudly, writing nothing, when the
store cannot be read or the selected population is empty.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import _checkout as _checkout
from _task_db_scan import TaskDbUnreadable, connect_ro, decode_metadata, tasks_db_path
from orchestrator.config import coerce_tier
from orchestrator.scheduler import Scheduler
from tasks_db_schema import EXIT_OK, EXIT_UNREADABLE, MainCheckoutUnresolved, resolve_live_db_path

POPULATION_NOTE = 'every pending task of the tag, anchorless rows included'


class EmptyPopulation(Exception):
    def __init__(self, db_path: Path, tag: str) -> None:
        super().__init__(f'{db_path}: no pending tasks under tag {tag!r}; refusing to write an empty fixture.')


def _read_tag(conn: sqlite3.Connection, tag: str) -> list[dict]:
    """Every task of *tag*, shaped as the scheduler's task dicts (str ids)."""
    dependencies: dict[str, list[str]] = {}
    for task_id, depends_on in conn.execute(
        'SELECT task_id, depends_on FROM dependencies WHERE tag = ?', (tag,)
    ):
        dependencies.setdefault(str(task_id), []).append(str(depends_on))
    return [
        {
            'id': str(task_id),
            'status': status,
            'priority': priority,
            'metadata': decode_metadata(metadata),
            'dependencies': dependencies.get(str(task_id), []),
        }
        for task_id, status, priority, metadata in conn.execute(
            'SELECT id, status, priority, metadata FROM tasks WHERE tag = ?', (tag,)
        )
    ]


def build_fixture(conn: sqlite3.Connection, db_path: Path, tag: str) -> dict:
    tasks = _read_tag(conn, tag)
    transitive_counts = Scheduler._compute_transitive_counts(
        {t['id']: t for t in tasks},
        Scheduler._build_reverse_index(tasks),
        {t['id']: t['status'] for t in tasks},
    )
    pending = sorted((t for t in tasks if t['status'] == 'pending'), key=lambda t: int(t['id']))
    if not pending:
        raise EmptyPopulation(db_path, tag)
    return {
        'provenance': {
            'captured_at': datetime.now(UTC).isoformat(),
            'source_db': str(db_path.resolve()),
            'tag': tag,
            'row_count': len(pending),
            'population': POPULATION_NOTE,
        },
        'candidates': [
            {
                'id': t['id'],
                'tier': coerce_tier(t['priority']),
                'pending_since': t['metadata'].get('pending_since'),
                'transitive_count': transitive_counts[t['id']],
            }
            for t in pending
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Dump the live pending population as the dispatch scorer sees it (read-only).'
    )
    where = parser.add_mutually_exclusive_group()
    where.add_argument('--db', help='a store to read; default: the live one for this checkout')
    where.add_argument('--project-root', help='a project root, whose .taskmaster/tasks/tasks.db is read')
    parser.add_argument('--tag', default='master', help='the task tag to dump (default: master)')
    parser.add_argument('--out', required=True, help='where to write the JSON fixture')
    return parser


def _resolve_db_path(args: argparse.Namespace) -> Path:
    if args.db is not None:
        return Path(args.db)
    if args.project_root is not None:
        return tasks_db_path(args.project_root)
    return resolve_live_db_path(Path.cwd())


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        db_path = _resolve_db_path(args)
        conn = connect_ro(db_path)
        try:
            fixture = build_fixture(conn, db_path, args.tag)
        finally:
            conn.close()
    except (MainCheckoutUnresolved, TaskDbUnreadable, EmptyPopulation) as refusal:
        print(refusal, file=sys.stderr)
        return EXIT_UNREADABLE

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, indent=1) + '\n')
    print(f"{out}: {fixture['provenance']['row_count']} candidates from {db_path.resolve()}")
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
