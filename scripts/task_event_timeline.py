#!/usr/bin/env python3
"""List one task's events from its project's orchestrator event log, across every run.

A multi-event failure sequence is counted from these rows rather than narrated
from memory. The habit this serves is stated in
``skills/_shared/counting-failure-events.md``.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

from _task_db_scan import connect_ro, decode_metadata
from audit_wiped_metadata_files import runs_db_path

EXIT_OK = 0


class TimelineEvent(NamedTuple):
    """One ``events`` row; field names are the column names they are read from."""

    id: int
    timestamp: str
    run_id: str
    event_type: str
    phase: str | None
    role: str | None
    cost_usd: float | None
    duration_ms: int | None
    data: dict

    @property
    def subtype(self) -> str | None:
        subtype = self.data.get("subtype")
        return subtype if isinstance(subtype, str) else None


class OutcomeCount(NamedTuple):
    """How many ``invocation_end`` rows share one (role, subtype)."""

    role: str | None
    subtype: str | None
    count: int


_TIMELINE_SQL = f"SELECT {', '.join(TimelineEvent._fields)} FROM events WHERE task_id = ?"


def read_timeline(
    conn: sqlite3.Connection, task_id: str, event_types: Sequence[str] = ()
) -> tuple[TimelineEvent, ...]:
    """Every event of *task_id* in id order, whichever orchestrator run wrote it."""
    sql = _TIMELINE_SQL
    params = [task_id]
    if event_types:
        sql += f" AND event_type IN ({', '.join('?' * len(event_types))})"
        params += event_types
    rows = conn.execute(f"{sql} ORDER BY id", params)
    return tuple(TimelineEvent(*row[:-1], data=decode_metadata(row[-1])) for row in rows)


def run_ids(events: Sequence[TimelineEvent]) -> tuple[str, ...]:
    """The runs *events* came from, in first-seen order."""
    return tuple(dict.fromkeys(event.run_id for event in events))


def invocation_outcomes(events: Sequence[TimelineEvent]) -> tuple[OutcomeCount, ...]:
    """``invocation_end`` rows tallied by (role, subtype), in first-occurrence order."""
    tally = Counter(
        (event.role, event.subtype) for event in events if event.event_type == "invocation_end"
    )
    return tuple(OutcomeCount(role, subtype, count) for (role, subtype), count in tally.items())


def render_json(db_path: Path, task_id: str, events: Sequence[TimelineEvent]) -> str:
    return json.dumps(
        {
            "db": str(db_path),
            "task_id": task_id,
            "runs": list(run_ids(events)),
            "events": [event._asdict() for event in events],
            "invocation_outcomes": [outcome._asdict() for outcome in invocation_outcomes(events)],
        },
        indent=2,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "List every event one task left in its project's orchestrator event log "
            "(runs.db), across all orchestrator runs, and count the invocation outcomes."
        ),
        epilog=(
            "Strictly READ-ONLY: the log is opened mode=ro and this tool never writes, "
            "and never creates a missing log."
        ),
    )
    parser.add_argument("task_id", help="the task whose events to list")
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--db", help="the runs.db to read")
    where.add_argument(
        "--project-root",
        help="the MAIN checkout of the task's project, whose data/orchestrator/runs.db is read",
    )
    parser.add_argument(
        "--event-type",
        action="append",
        help="list only this event type; repeat to list several",
    )
    parser.add_argument("--json", action="store_true", help="emit one JSON object")
    return parser


def _resolve_db_path(args: argparse.Namespace) -> Path:
    if args.db is not None:
        return Path(args.db)
    return runs_db_path(args.project_root)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    db_path = _resolve_db_path(args).resolve()
    conn = connect_ro(db_path)
    try:
        events = read_timeline(conn, args.task_id)
    finally:
        conn.close()

    if args.json:
        print(render_json(db_path, args.task_id, events))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
