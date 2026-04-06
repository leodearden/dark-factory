"""Append-only SQLite event store for orchestrator observability.

Captures structured events throughout execution — agent invocations, phase
transitions, escalations, waste, and infrastructure events.  All writes are
fire-and-forget: failures are logged but never propagate to callers.

Modeled on fused-memory's WriteJournal pattern.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA = """\
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


class EventType(StrEnum):
    """Orchestrator event taxonomy."""

    # Agent invocations
    invocation_end = 'invocation_end'

    # Workflow phases
    phase_enter = 'phase_enter'
    phase_exit = 'phase_exit'

    # Escalations
    escalation_created = 'escalation_created'
    escalation_resolved = 'escalation_resolved'

    # Waste detection
    waste_detected = 'waste_detected'

    # Infrastructure
    cap_hit = 'cap_hit'
    lock_acquired = 'lock_acquired'
    lock_released = 'lock_released'
    merge_attempt = 'merge_attempt'
    speculative_merge = 'speculative_merge'
    speculative_discard = 'speculative_discard'

    # Task lifecycle
    task_started = 'task_started'
    task_completed = 'task_completed'


class EventStore:
    """Append-only SQLite event store.

    One instance per orchestrator run, created in the harness and propagated
    to workflows, merge worker, steward, and scheduler.  Every ``emit()``
    call is fire-and-forget — failures log a warning but never raise.
    """

    def __init__(self, db_path: Path, run_id: str):
        self.db_path = db_path
        self.run_id = run_id
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        try:
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.execute('PRAGMA journal_mode=WAL')
                conn.executescript(_SCHEMA)
            finally:
                conn.close()
        except Exception:
            logger.warning('event_store: schema init failed', exc_info=True)

    def emit(
        self,
        event_type: EventType,
        *,
        task_id: str | None = None,
        phase: str | None = None,
        role: str | None = None,
        data: dict | None = None,
        cost_usd: float | None = None,
        duration_ms: int | None = None,
    ) -> None:
        """Write a single event row.  Never raises."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.execute(
                    'INSERT INTO events '
                    '(timestamp, run_id, task_id, event_type, phase, role, '
                    ' data, cost_usd, duration_ms) '
                    'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (
                        datetime.now(UTC).isoformat(),
                        self.run_id,
                        task_id,
                        event_type.value,
                        phase,
                        role,
                        json.dumps(data) if data else '{}',
                        cost_usd,
                        duration_ms,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            logger.warning('event_store.emit failed', exc_info=True)
