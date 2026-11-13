"""Fixture tasks.db, runs.db and escalation-queue stores under a root, in their producers' shape, without importing the producers."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from shared.project_layout import escalation_queue_dir, runs_db, tasks_db
from shared.task_store_schema import TASK_STORE_DDL

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
        _create(path, TASK_STORE_DDL)
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


@dataclass(frozen=True)
class EscalationQueueDir:
    path: Path

    @classmethod
    def create(cls, root: Path) -> EscalationQueueDir:
        path = escalation_queue_dir(root)
        path.mkdir(parents=True, exist_ok=True)
        return cls(path)

    def write(
        self,
        escalation_id: str,
        *,
        task_id: str | None,
        level: object = 0,
        status: str = 'pending',
        members: Iterable[str] = (),
        **fields: object,
    ) -> Path:
        """One root record; `level` is any JSON value, so a malformed level can be stored."""
        # escalation/src/escalation/models.py::Escalation.to_json
        record = {
            'id': escalation_id,
            'task_id': task_id,
            'agent_role': 'steward',
            'severity': 'blocking',
            'category': 'design_concern',
            'summary': 'SECRET-SUMMARY /home/leo/x',
            'detail': 'SECRET-DETAIL',
            'timestamp': '2026-11-01T00:00:00+00:00',
            'status': status,
            'resolution': None,
            'level': level,
            'resolved_at': None,
            'members': list(members),
            **fields,
        }
        return self.write_raw(f'{escalation_id}.json', json.dumps(record, indent=2))

    def archive(self, escalation_id: str, *, status: str = 'resolved', on: str = '2026-11-01') -> None:
        """Stamp the record closed and move it under archive/<on>/, as EscalationQueue.resolve does."""
        record_path = self.path / f'{escalation_id}.json'
        record = json.loads(record_path.read_text())
        record.update(status=status, resolved_at=f'{on}T00:00:00+00:00')
        archived = self.path / 'archive' / on / record_path.name
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_text(json.dumps(record, indent=2))
        record_path.unlink()

    def write_raw(self, name: str, text: str) -> Path:
        """Any file at the root: a corrupt or legacy record, or a sidecar."""
        path = self.path / name
        path.write_text(text)
        return path
