"""SQLite persistence for orchestrator run results."""

from __future__ import annotations

import logging
import sqlite3
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.harness import HarnessReport, TaskReport

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

    def start_run(
        self,
        run_id: str,
        project_id: str,
        started_at: str,
        prd_path: str | None = None,
    ) -> None:
        """Insert an initial runs row at orchestrator startup.

        The row is updated with final aggregates via :meth:`finish_run`
        when the run completes.
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(
                'INSERT OR IGNORE INTO runs '
                '(run_id, project_id, prd_path, started_at) '
                'VALUES (?, ?, ?, ?)',
                (run_id, project_id, prd_path, started_at),
            )
            conn.commit()
        finally:
            conn.close()
        logger.info(f'Started run {run_id} for project {project_id}')

    def save_task_result(
        self,
        run_id: str,
        task_report: TaskReport,
        project_id: str,
    ) -> None:
        """Persist a single TaskReport row immediately on task completion."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(
                'INSERT OR REPLACE INTO task_results '
                '(run_id, task_id, project_id, title, outcome, '
                ' cost_usd, duration_ms, agent_invocations, '
                ' execute_iterations, verify_attempts, review_cycles, '
                ' steward_cost_usd, steward_invocations, completed_at) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    run_id,
                    task_report.task_id,
                    project_id,
                    task_report.title,
                    task_report.outcome.value,
                    task_report.cost_usd,
                    task_report.duration_ms,
                    task_report.agent_invocations,
                    task_report.execute_iterations,
                    task_report.verify_attempts,
                    task_report.review_cycles,
                    task_report.steward_cost_usd,
                    task_report.steward_invocations,
                    task_report.completed_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        logger.debug(
            f'Persisted task result {task_report.task_id} '
            f'({task_report.outcome.value}) for run {run_id}'
        )

    def get_task_cost(self, run_id: str, task_id: str) -> float:
        """Return latest persisted ``cost_usd + steward_cost_usd`` for a task.

        Only the most recent ``save_task_result`` row is retained per
        ``(run_id, task_id)`` — prior requeue attempts are overwritten.
        Returns ``0.0`` when no row exists.
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            cur = conn.execute(
                'SELECT cost_usd, steward_cost_usd FROM task_results '
                'WHERE run_id = ? AND task_id = ?',
                (run_id, task_id),
            )
            row = cur.fetchone()
        finally:
            conn.close()
        if row is None:
            return 0.0
        cost_usd, steward_cost_usd = row
        return float(cost_usd or 0.0) + float(steward_cost_usd or 0.0)

    def finish_run(
        self,
        run_id: str,
        report: HarnessReport,
    ) -> None:
        """Update the runs row with final aggregates at shutdown."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(
                'UPDATE runs SET '
                'completed_at = ?, total_tasks = ?, completed = ?, '
                'blocked = ?, escalated = ?, total_cost_usd = ?, '
                'paused_for_cap = ?, cap_pause_secs = ? '
                'WHERE run_id = ?',
                (
                    report.completed_at,
                    report.total_tasks,
                    report.completed,
                    report.blocked,
                    report.escalated,
                    report.total_cost_usd,
                    int(report.paused_for_cap),
                    report.cap_pause_duration_secs,
                    run_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        logger.info(
            f'Finished run {run_id}: {len(report.task_reports)} task results'
        )

    def save_run(
        self,
        report: HarnessReport,
        project_id: str,
        prd_path: str | None = None,
    ) -> str:
        """Persist a HarnessReport and its TaskReports to SQLite.

        Returns the generated run_id.
        """
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
