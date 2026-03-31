"""SQLite persistence for orchestrator run results."""

from __future__ import annotations

import logging
import sqlite3
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.harness import HarnessReport

logger = logging.getLogger(__name__)

_SCHEMA = """\
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
    PRIMARY KEY (run_id, task_id)
);

CREATE INDEX IF NOT EXISTS idx_task_results_project
    ON task_results(project_id, completed_at);
"""


class RunStore:
    """Synchronous SQLite writer for orchestrator run results."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.executescript(_SCHEMA)
        finally:
            conn.close()

    def save_run(
        self,
        report: HarnessReport,
        project_id: str,
        prd_path: str | None = None,
        run_id: str | None = None,
    ) -> str:
        """Persist a HarnessReport and its TaskReports to SQLite.

        When *run_id* is provided it is used directly; otherwise a new
        ``run-{uuid12}`` identifier is generated.

        Returns the run_id used.
        """
        if run_id is None:
            run_id = f'run-{uuid.uuid4().hex[:12]}'
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(
                'INSERT INTO runs '
                '(run_id, project_id, prd_path, started_at, completed_at, '
                ' total_tasks, completed, blocked, escalated, '
                ' total_cost_usd, paused_for_cap, cap_pause_secs) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    run_id,
                    project_id,
                    prd_path,
                    report.started_at,
                    report.completed_at,
                    report.total_tasks,
                    report.completed,
                    report.blocked,
                    report.escalated,
                    report.total_cost_usd,
                    int(report.paused_for_cap),
                    report.cap_pause_duration_secs,
                ),
            )
            for tr in report.task_reports:
                conn.execute(
                    'INSERT INTO task_results '
                    '(run_id, task_id, project_id, title, outcome, '
                    ' cost_usd, duration_ms, agent_invocations, '
                    ' execute_iterations, verify_attempts, review_cycles, '
                    ' steward_cost_usd, steward_invocations, completed_at) '
                    'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (
                        run_id,
                        tr.task_id,
                        project_id,
                        tr.title,
                        tr.outcome.value,
                        tr.cost_usd,
                        tr.duration_ms,
                        tr.agent_invocations,
                        tr.execute_iterations,
                        tr.verify_attempts,
                        tr.review_cycles,
                        tr.steward_cost_usd,
                        tr.steward_invocations,
                        tr.completed_at,
                    ),
                )
            conn.commit()
        finally:
            conn.close()

        logger.info(
            f'Persisted run {run_id}: {len(report.task_reports)} task results '
            f'for project {project_id}'
        )
        return run_id
