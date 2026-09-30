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

from _task_db_scan import TaskDbUnreadable, connect_ro, decode_metadata
from audit_wiped_metadata_files import runs_db_path

EXIT_OK = 0
EXIT_NO_EVENTS = 1
# 3, as in tasks_db_schema.py::EXIT_UNREADABLE: nothing was read. 2 is argparse's usage error.
EXIT_UNREADABLE = 3


class EventLogUnreadable(Exception):
    """*path* is not a readable orchestrator event log, for the reason in *detail*."""

    def __init__(self, path: Path, detail: str) -> None:
        self.path = path
        self.detail = detail
        super().__init__(
            f"{path}: not a readable orchestrator event log ({detail}). The log lives "
            f"in the MAIN checkout of the TASK's project, at "
            f"<main checkout>/data/orchestrator/runs.db. data/ is gitignored, so it is "
            f"absent from worktrees; `git worktree list --porcelain` names the main "
            f"checkout on its first line; the data/runs.db beside the real log is a "
            f"0-byte decoy; and each project keeps its own runs.db."
        )


def open_event_log(path: Path) -> sqlite3.Connection:
    """Open *path* read-only, or refuse with :class:`EventLogUnreadable`."""
    try:
        conn = connect_ro(path)
    except TaskDbUnreadable as refusal:
        raise EventLogUnreadable(refusal.path, refusal.reason.value) from refusal
    has_events = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'events'"
    ).fetchone()
    if has_events is None:
        conn.close()
        raise EventLogUnreadable(Path(path).resolve(), "no events table")
    return conn


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


def _event_line(event: TimelineEvent) -> str:
    labelled = (("role", event.role), ("phase", event.phase), ("subtype", event.subtype))
    extras = "".join(f" {label}={value}" for label, value in labelled if value is not None)
    cost = f" cost={event.cost_usd:.2f}" if event.cost_usd is not None else ""
    return f"#{event.id} {event.timestamp} {event.run_id} {event.event_type}{extras}{cost}"


def _summary_line(task_id: str, events: Sequence[TimelineEvent]) -> str:
    runs = run_ids(events)
    return (
        f"{len(events)} events for task {task_id} across {len(runs)} orchestrator runs: "
        f"{', '.join(runs)}"
    )


def _outcome_lines(events: Sequence[TimelineEvent]) -> list[str]:
    outcomes = invocation_outcomes(events)
    if not outcomes:
        return ["invocation_end outcomes: none"]
    return [
        "invocation_end outcomes:",
        *(f"  {o.count} x {o.role or '-'} {o.subtype or '-'}" for o in outcomes),
    ]


def render_text(db_path: Path, task_id: str, events: Sequence[TimelineEvent]) -> str:
    """The log read, one line per event, then the event/run count and the outcome tally."""
    return "\n".join(
        [
            str(db_path),
            *(_event_line(event) for event in events),
            "",
            _summary_line(task_id, events),
            *_outcome_lines(events),
        ]
    )


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
            "and never creates a missing log. Exit 0: events listed. Exit 1: this log "
            "holds no such events for that task, so the id or the project is wrong. "
            "Exit 3: the path is not a readable event log."
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


def _no_events_message(db_path: Path, task_id: str, event_types: Sequence[str]) -> str:
    of_types = f" of type {', '.join(event_types)}" if event_types else ""
    return (
        f"{db_path}: no events{of_types} for task {task_id}. Each project keeps its own "
        f"runs.db, so a task id from another project finds nothing here."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """List the task's events; refuse loudly, with nothing on stdout, when there are none."""
    args = _build_parser().parse_args(argv)
    db_path = _resolve_db_path(args).resolve()
    event_types = tuple(args.event_type or ())
    try:
        conn = open_event_log(db_path)
    except EventLogUnreadable as refusal:
        print(refusal, file=sys.stderr)
        return EXIT_UNREADABLE
    try:
        events = read_timeline(conn, args.task_id, event_types)
    finally:
        conn.close()

    if not events:
        print(_no_events_message(db_path, args.task_id, event_types), file=sys.stderr)
        return EXIT_NO_EVENTS
    render = render_json if args.json else render_text
    print(render(db_path, args.task_id, events))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
