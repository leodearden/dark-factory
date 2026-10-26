"""Fixture tasks.db and runs.db stores under a root, in their producers' shape, without importing the producers."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from shared.project_layout import runs_db, tasks_db

# fused-memory/src/fused_memory/backends/sqlite_task_backend.py::_SCHEMA_SQL
TASKS_DDL: Final = """
CREATE TABLE IF NOT EXISTS tasks (
    tag             TEXT NOT NULL DEFAULT 'master',
    id              INTEGER NOT NULL,
    title           TEXT NOT NULL,
    description     TEXT,
    details         TEXT,
    test_strategy   TEXT,
    status          TEXT NOT NULL,
    priority        TEXT,
    metadata        TEXT,
    updated_at      TEXT NOT NULL,
    claimant_run_id TEXT,
    heartbeat_at    TEXT,
    candidate_key   TEXT,
    PRIMARY KEY (tag, id)
);

CREATE INDEX IF NOT EXISTS ix_tasks_status ON tasks (tag, status);

CREATE TABLE IF NOT EXISTS dependencies (
    tag        TEXT NOT NULL DEFAULT 'master',
    task_id    INTEGER NOT NULL,
    depends_on INTEGER NOT NULL,
    PRIMARY KEY (tag, task_id, depends_on)
);
"""

# orchestrator/src/orchestrator/event_store.py::_SCHEMA
EVENTS_DDL: Final = """\
CREATE TABLE IF NOT EXISTS events (
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

CREATE INDEX IF NOT EXISTS idx_events_run  ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_task ON events(run_id, task_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_ts   ON events(timestamp);
"""

# orchestrator/src/orchestrator/run_store.py::_SCHEMA (runs, task_results)
RUN_RESULTS_DDL: Final = """\
CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT PRIMARY KEY,
    project_id     TEXT NOT NULL,
    prd_path       TEXT,
    started_at     TEXT NOT NULL,
    completed_at   TEXT,
    total_tasks    INTEGER DEFAULT 0,
    completed      INTEGER DEFAULT 0,
    blocked        INTEGER DEFAULT 0,
    escalated      INTEGER DEFAULT 0,
    total_cost_usd REAL DEFAULT 0.0,
    paused_for_cap INTEGER DEFAULT 0,
    cap_pause_secs REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS task_results (
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
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

CREATE INDEX IF NOT EXISTS idx_task_results_project
    ON task_results(project_id, completed_at);
"""

RUNS_DDL: Final = EVENTS_DDL + RUN_RESULTS_DDL

_PROJECT_ID: Final = 'fixture'


def _create(path: Path, ddl: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(ddl)
    finally:
        conn.close()


def _write(path: Path, statements: Iterable[tuple[str, tuple[Any, ...]]]) -> int:
    """Run `statements` in one transaction on a fresh connection; the last insert's row id."""
    conn = sqlite3.connect(path)
    try:
        row_id = 0
        for sql, params in statements:
            row_id = conn.execute(sql, params).lastrowid or row_id
        conn.commit()
        return row_id
    finally:
        conn.close()


def _json_text(value: dict | str | None) -> str | None:
    """A str is written raw, so a malformed blob can be stored; a mapping is serialized."""
    return value if value is None or isinstance(value, str) else json.dumps(value)


@dataclass(frozen=True)
class RunsDb:
    path: Path

    @classmethod
    def create(cls, root: Path) -> RunsDb:
        path = runs_db(root)
        _create(path, RUNS_DDL)
        return cls(path)

    def emit(
        self,
        event_type: str,
        *,
        at: datetime | str,
        task_id: str | int | None = None,
        phase: str | None = None,
        data: dict | str | None = None,
    ) -> int:
        """One events row as EventStore.emit writes it; a str `at` is written raw. Returns the row id."""
        timestamp = at if isinstance(at, str) else at.isoformat()
        return _write(
            self.path,
            [(
                'INSERT INTO events (timestamp, run_id, task_id, event_type, phase, data)'
                ' VALUES (?, ?, ?, ?, ?, ?)',
                (
                    timestamp,
                    'run-fixture',
                    None if task_id is None else str(task_id),
                    event_type,
                    phase,
                    _json_text(data) or '{}',
                ),
            )],
        )

    def task_result(self, task_id: str | int, *, review_cycles: int, run_id: str = 'run-1') -> None:
        _write(
            self.path,
            [
                (
                    'INSERT OR IGNORE INTO runs (run_id, project_id, started_at) VALUES (?, ?, ?)',
                    (run_id, _PROJECT_ID, '2026-10-01T00:00:00+00:00'),
                ),
                (
                    'INSERT INTO task_results (run_id, task_id, project_id, outcome, review_cycles)'
                    ' VALUES (?, ?, ?, ?, ?)',
                    (run_id, str(task_id), _PROJECT_ID, 'done', review_cycles),
                ),
            ],
        )


@dataclass(frozen=True)
class TasksDb:
    path: Path

    @classmethod
    def create(cls, root: Path) -> TasksDb:
        path = tasks_db(root)
        _create(path, TASKS_DDL)
        return cls(path)

    def add(
        self,
        task_id: int,
        *,
        title: str = 't',
        status: str = 'pending',
        updated_at: str,
        tag: str = 'master',
        description: str | None = None,
        details: str | None = None,
        test_strategy: str | None = None,
        metadata: dict | str | None = None,
        depends_on: Iterable[int] = (),
    ) -> None:
        task = (
            'INSERT INTO tasks (tag, id, title, description, details, test_strategy, status,'
            ' metadata, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                tag, task_id, title, description, details, test_strategy, status,
                _json_text(metadata), updated_at,
            ),
        )
        dependencies = [
            ('INSERT INTO dependencies (tag, task_id, depends_on) VALUES (?, ?, ?)',
             (tag, task_id, dependency))
            for dependency in depends_on
        ]
        _write(self.path, [task, *dependencies])
