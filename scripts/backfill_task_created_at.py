#!/usr/bin/env python3
"""Fill ``tasks.created_at`` for the rows that predate the column (task 8558).

PRD ``plans/instruments-honesty-prd.md`` decision 3. ``SqliteTaskBackend.add_task``
stamps every row it creates with ``created_at_source = 'stamped'``. This script
fills only the rows whose ``created_at`` is still NULL. Each of the three
proxies below is an upper bound on when the task was filed, because a task
exists before any event, pending landing or update touches it. So the script
stores the earliest usable one and records its source; the list order only
breaks an exact tie:

1. ``first_event``: the earliest runs.db event for the task id. Events carry
   no tag, so only tag ``master`` rows are matched to them. A long-waiting
   task's earliest event is often the starvation watchdog's alarm, not its
   filing, which is why the event cannot simply win.
2. ``pending_since``: ``metadata.pending_since``. One marked
   ``pending_since_backfilled`` is the v4 -> v5 migration's copy of
   ``updated_at``, so it counts as an ``updated_at`` reading.
3. ``updated_at``.

A blank or unparseable value is not usable. A row with no usable source
stays NULL and is counted. Every value is normalised by
``shared.task_timestamp.format_task_timestamp``.

All rows are filled in one ``BEGIN IMMEDIATE`` transaction, and every UPDATE
is guarded by ``created_at IS NULL``, so a row the live backend stamps
meanwhile is never overwritten. A second run therefore writes nothing.

Usage:
    scripts/backfill_task_created_at.py --project-root R [--dry-run]

    --dry-run rolls the transaction back and reports what would be written.
    The JSON report goes to stdout.

Exit codes:
    0  completed.
    2  refused, and nothing was written: the tasks.db is absent, empty or not
       a database; it lacks the created_at columns (fused-memory adds them
       when it next opens the store); a runs.db exists but cannot be read,
       which must not silently degrade every row to a weaker source; or the
       fill failed inside sqlite, typically "database is locked" because
       fused-memory held the write lock past the busy timeout. Re-run it.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import _checkout as _checkout
from shared.project_layout import runs_db, tasks_db
from shared.task_created_at import CreatedAtSource
from shared.task_metadata_wire import coerce_task_metadata
from shared.task_timestamp import format_task_timestamp
from shared.tasks_db_readonly import TaskDbUnreadable, connect_ro

EXIT_OK = 0
EXIT_REFUSED = 2

_EVENT_TAG = 'master'
_BUSY_TIMEOUT_MS = 5000
_CREATED_AT_COLUMNS = ('created_at', 'created_at_source')
_BACKFILL_SOURCES = (
    CreatedAtSource.FIRST_EVENT,
    CreatedAtSource.PENDING_SINCE,
    CreatedAtSource.UPDATED_AT,
)
_FILL = (
    'UPDATE tasks SET created_at = ?, created_at_source = ? '
    'WHERE tag = ? AND id = ? AND created_at IS NULL'
)


class BackfillRefused(Exception):
    """The backfill was refused; nothing was written."""


@dataclass(frozen=True)
class FirstEvents:
    by_task_id: Mapping[str, datetime]
    unparseable_timestamps: int


@dataclass(frozen=True)
class Fills:
    updates: tuple[tuple[str, str, object, object], ...]
    counts: Mapping[CreatedAtSource, int]
    skipped: int


def _instant(raw: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _candidates(
    row: Mapping[str, object], first_event: datetime | None
) -> tuple[tuple[CreatedAtSource, datetime | None], ...]:
    """*row*'s filing-time proxies in rank order; None marks an unusable one."""
    metadata = coerce_task_metadata(row['metadata']) or {}
    pending_source = (
        CreatedAtSource.UPDATED_AT
        if 'pending_since_backfilled' in metadata
        else CreatedAtSource.PENDING_SINCE
    )
    return (
        (CreatedAtSource.FIRST_EVENT, first_event),
        (pending_source, _instant(metadata.get('pending_since'))),
        (CreatedAtSource.UPDATED_AT, _instant(row['updated_at'])),
    )


def choose_created_at(
    row: Mapping[str, object], first_event: datetime | None
) -> tuple[CreatedAtSource, str] | None:
    """The earliest usable candidate for *row*'s filing time, formatted; None if none is usable.

    Every proxy is an upper bound on the filing time; ties go to rank order.
    """
    usable = [
        (source, instant)
        for source, instant in _candidates(row, first_event)
        if instant is not None
    ]
    if not usable:
        return None
    source, instant = min(usable, key=lambda candidate: candidate[1])
    return source, format_task_timestamp(instant)


def earliest_events(conn: sqlite3.Connection) -> FirstEvents:
    """Each task id's earliest event instant, compared as parsed instants, not TEXT.

    TEXT order matches instant order only while every timestamp carries one offset.
    """
    earliest: dict[str, datetime] = {}
    unparseable = 0
    for task_id, timestamp in conn.execute(
        'SELECT task_id, timestamp FROM events WHERE task_id IS NOT NULL'
    ):
        instant = _instant(timestamp)
        if instant is None:
            unparseable += 1
        elif task_id not in earliest or instant < earliest[task_id]:
            earliest[task_id] = instant
    return FirstEvents(earliest, unparseable)


def plan_fills(rows: Iterable[Mapping[str, object]], events: FirstEvents | None) -> Fills:
    updates: list[tuple[str, str, object, object]] = []
    counts: Counter[CreatedAtSource] = Counter()
    skipped = 0
    for row in rows:
        first_event = (
            events.by_task_id.get(str(row['id']))
            if events is not None and row['tag'] == _EVENT_TAG
            else None
        )
        chosen = choose_created_at(row, first_event)
        if chosen is None:
            skipped += 1
            continue
        source, created_at = chosen
        counts[source] += 1
        updates.append((created_at, source.value, row['tag'], row['id']))
    return Fills(tuple(updates), counts, skipped)


def _require_created_at_columns(store: Path) -> None:
    try:
        conn = connect_ro(store)
    except (TaskDbUnreadable, sqlite3.Error) as exc:
        raise BackfillRefused(str(exc)) from exc
    try:
        columns = {row[1] for row in conn.execute('PRAGMA table_info(tasks)')}
    finally:
        conn.close()
    missing = [column for column in _CREATED_AT_COLUMNS if column not in columns]
    if missing:
        raise BackfillRefused(
            f'{store}: the tasks table has no {" or ".join(missing)} column. '
            'fused-memory adds both when it opens the store at schema v6, so '
            'let fused-memory open the store first (any get_tasks call for '
            'this project root), then re-run.'
        )


def _read_first_events(path: Path) -> FirstEvents | None:
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(f'{path.resolve().as_uri()}?mode=ro', uri=True)
        try:
            return earliest_events(conn)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise BackfillRefused(
            f'{path}: runs.db exists but cannot be read ({exc}). Refusing rather '
            'than filling every row from a weaker source.'
        ) from exc


def _fill(store: Path, events: FirstEvents | None, *, dry_run: bool) -> tuple[Fills, int]:
    conn = sqlite3.connect(f'{store.resolve().as_uri()}?mode=rw', uri=True, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(f'PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}')
        conn.execute('BEGIN IMMEDIATE')
        try:
            fills = plan_fills(
                conn.execute(
                    'SELECT tag, id, metadata, updated_at FROM tasks WHERE created_at IS NULL'
                ),
                events,
            )
            rows_written = conn.executemany(_FILL, fills.updates).rowcount if fills.updates else 0
        except BaseException:
            conn.execute('ROLLBACK')
            raise
        conn.execute('ROLLBACK' if dry_run else 'COMMIT')
    finally:
        conn.close()
    return fills, rows_written


def _fill_or_refuse(
    store: Path, events: FirstEvents | None, *, dry_run: bool
) -> tuple[Fills, int]:
    """:func:`_fill`, with any sqlite failure refused: the transaction never committed."""
    try:
        return _fill(store, events, dry_run=dry_run)
    except sqlite3.OperationalError as exc:
        raise BackfillRefused(
            f'{store}: the fill failed ({exc}); nothing was written. A "database is '
            f'locked" means another writer held the store past {_BUSY_TIMEOUT_MS} ms; '
            're-run.'
        ) from exc


def run(project_root: Path, *, dry_run: bool) -> dict[str, object]:
    """Fill every NULL ``created_at`` under *project_root*; return the report.

    Raises:
        BackfillRefused: the backfill was refused; nothing was written.
    """
    store = tasks_db(project_root)
    _require_created_at_columns(store)
    events = _read_first_events(runs_db(project_root))
    fills, rows_written = _fill_or_refuse(store, events, dry_run=dry_run)
    return {
        'project_root': str(project_root),
        'tasks_db': str(store),
        'runs_db': 'absent' if events is None else 'present',
        'dry_run': dry_run,
        'counts': {source.value: fills.counts[source] for source in _BACKFILL_SOURCES},
        'skipped_no_usable_source': fills.skipped,
        'unparseable_event_timestamps': 0 if events is None else events.unparseable_timestamps,
        'rows_written': rows_written,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Fill tasks.created_at for rows that predate the column.',
    )
    parser.add_argument(
        '--project-root', required=True, type=Path,
        help='the project whose .taskmaster/tasks/tasks.db is filled',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='report what would be written, and write nothing',
    )
    args = parser.parse_args(argv)
    try:
        report = run(args.project_root, dry_run=args.dry_run)
    except BackfillRefused as refusal:
        print(f'backfill_task_created_at: {refusal}', file=sys.stderr)
        return EXIT_REFUSED
    print(json.dumps(report, indent=2))
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
