"""A project's orchestrator runs.db, written by tests the way the orchestrator lays it out.

The DDL is copied, not imported: the dashboard does not depend on the orchestrator
package. ``EVENTS_DDL`` is ``orchestrator/src/orchestrator/event_store.py::_SCHEMA``
and ``TASK_RESULTS_DDL`` is the ``task_results`` table of
``orchestrator/src/orchestrator/run_store.py::_SCHEMA``.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

RUNS_DB = Path('data') / 'orchestrator' / 'runs.db'
"""Where a project keeps its runs.db, relative to the project root."""

EVENTS_DDL = """\
CREATE TABLE events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    run_id      TEXT    NOT NULL,
    task_id     TEXT,
    event_type  TEXT    NOT NULL,
    phase       TEXT,
    role        TEXT,
    data        TEXT    DEFAULT '{}',
    cost_usd    REAL,
    duration_ms INTEGER
);

CREATE INDEX idx_events_run  ON events(run_id);
CREATE INDEX idx_events_task ON events(run_id, task_id);
CREATE INDEX idx_events_type ON events(event_type);
CREATE INDEX idx_events_ts   ON events(timestamp);
"""

TASK_RESULTS_DDL = """\
CREATE TABLE task_results (
    run_id              TEXT NOT NULL,
    task_id             TEXT NOT NULL,
    project_id          TEXT NOT NULL,
    title               TEXT,
    outcome             TEXT NOT NULL,
    cost_usd            REAL DEFAULT 0.0,
    duration_ms         INTEGER DEFAULT 0,
    agent_invocations   INTEGER DEFAULT 0,
    execute_iterations  INTEGER DEFAULT 0,
    verify_attempts     INTEGER DEFAULT 0,
    review_cycles       INTEGER DEFAULT 0,
    steward_cost_usd    REAL DEFAULT 0.0,
    steward_invocations INTEGER DEFAULT 0,
    completed_at        TEXT,
    block_reason        TEXT,
    block_phase         TEXT,
    PRIMARY KEY (run_id, task_id)
);
"""

EventRow = tuple[datetime | str, str | None, str, Mapping[str, Any] | str]
"""(timestamp, task_id, event_type, data): a mapping is JSON-encoded, a string is stored verbatim."""

TaskResultRow = tuple[str, str, datetime | str | None]
"""(task_id, outcome, completed_at)."""


def runs_db_path(root: Path) -> Path:
    return root / RUNS_DB


def write_runs_db(
    root: Path,
    events: Iterable[EventRow] = (),
    task_results: Iterable[TaskResultRow] = (),
    *,
    with_task_results_table: bool = True,
) -> Path:
    """Create *root*'s runs.db holding *events* and *task_results*; return its path."""
    store = runs_db_path(root)
    store.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(store)) as conn:
        conn.executescript(EVENTS_DDL)
        conn.executemany(
            'INSERT INTO events (timestamp, run_id, task_id, event_type, data) VALUES (?, ?, ?, ?, ?)',
            [
                (_text(timestamp), f'run-{task_id}', task_id, event_type, _data(data))
                for timestamp, task_id, event_type, data in events
            ],
        )
        if with_task_results_table:
            conn.executescript(TASK_RESULTS_DDL)
            conn.executemany(
                'INSERT INTO task_results (run_id, task_id, project_id, outcome, completed_at) '
                "VALUES (?, ?, 'fixture', ?, ?)",
                [
                    (f'run-{task_id}', task_id, outcome, _text(completed_at))
                    for task_id, outcome, completed_at in task_results
                ],
            )
        conn.commit()
    return store


def _text(instant: datetime | str | None) -> str | None:
    return instant.isoformat() if isinstance(instant, datetime) else instant


def _data(data: Mapping[str, Any] | str) -> str:
    return data if isinstance(data, str) else json.dumps(dict(data))
