"""Non-fixture test helpers for dashboard tests.

Lives outside conftest.py to avoid the `sys.modules['conftest']` collision
that arises when root-level pytest loads multiple subprojects' conftests in
the same process.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite

RECONCILIATION_SCHEMA = """
CREATE TABLE IF NOT EXISTS watermarks (
    project_id TEXT PRIMARY KEY,
    last_full_run_id TEXT,
    last_full_run_completed TEXT,
    last_episode_timestamp TEXT,
    last_memory_timestamp TEXT,
    last_task_change_timestamp TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    run_type TEXT NOT NULL,
    trigger_reason TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    events_processed INTEGER DEFAULT 0,
    stage_reports TEXT DEFAULT '{}',
    status TEXT DEFAULT 'running'
);
CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project_id);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);

CREATE TABLE IF NOT EXISTS journal_entries (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    stage TEXT,
    timestamp TEXT NOT NULL,
    operation TEXT NOT NULL,
    target_system TEXT NOT NULL,
    before_state TEXT,
    after_state TEXT,
    reasoning TEXT DEFAULT '',
    evidence TEXT DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_journal_run ON journal_entries(run_id);

CREATE TABLE IF NOT EXISTS judge_verdicts (
    run_id TEXT PRIMARY KEY,
    reviewed_at TEXT NOT NULL,
    severity TEXT NOT NULL,
    findings TEXT DEFAULT '[]',
    action_taken TEXT DEFAULT 'none'
);
CREATE INDEX IF NOT EXISTS idx_verdicts_reviewed ON judge_verdicts(reviewed_at);

CREATE TABLE IF NOT EXISTS event_buffer (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_source TEXT NOT NULL,
    agent_id TEXT,
    timestamp TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'buffered'
);
CREATE INDEX IF NOT EXISTS idx_eb_project_status ON event_buffer(project_id, status);
CREATE INDEX IF NOT EXISTS idx_eb_agent_timestamp ON event_buffer(agent_id, timestamp)
    WHERE agent_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS reconciliation_locks (
    project_id TEXT PRIMARY KEY,
    instance_id TEXT NOT NULL,
    acquired_at TEXT NOT NULL,
    heartbeat_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS burst_state (
    agent_id TEXT PRIMARY KEY,
    state TEXT NOT NULL DEFAULT 'idle',
    last_write_at TEXT NOT NULL,
    burst_started_at TEXT
);

CREATE TABLE IF NOT EXISTS chunk_boundaries (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    run_id TEXT,
    events_count INTEGER,
    status TEXT DEFAULT 'processing',
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunk_project ON chunk_boundaries(project_id);

CREATE TABLE IF NOT EXISTS run_actions (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    target TEXT NOT NULL,
    operation TEXT NOT NULL,
    detail TEXT DEFAULT '{}',
    causation_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ra_run ON run_actions(run_id);
"""

assert 'started_at TEXT NOT NULL' in RECONCILIATION_SCHEMA, (
    "RECONCILIATION_SCHEMA must contain 'started_at TEXT NOT NULL'; "
    "update RELAXED_RECONCILIATION_SCHEMA derivation if the schema changes."
)

RELAXED_RECONCILIATION_SCHEMA = RECONCILIATION_SCHEMA.replace(
    'started_at TEXT NOT NULL', 'started_at TEXT'
)


@asynccontextmanager
async def make_recon_db(
    tmp_path: Path,
    inserts: Sequence[str | tuple[str, Any]],
    *,
    name: str = 'test.db',
    schema: str | None = None,
):
    """Async context manager that creates a temporary SQLite reconciliation DB.

    Creates a DB at ``tmp_path / name``, applies ``schema`` (defaults to
    ``RECONCILIATION_SCHEMA``), executes each statement in ``inserts``, then
    yields an :class:`aiosqlite.Connection` with ``row_factory`` set to
    :class:`aiosqlite.Row`.  The connection is closed on context exit.
    """
    if schema is None:
        schema = RECONCILIATION_SCHEMA

    db_path = tmp_path / name
    sync_conn = sqlite3.connect(str(db_path))
    sync_conn.executescript(schema)
    for stmt in inserts:
        if isinstance(stmt, str):
            sync_conn.execute(stmt)
        else:
            sql, params = stmt
            sync_conn.execute(sql, params)
    sync_conn.commit()
    sync_conn.close()

    async with aiosqlite.connect(str(db_path)) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn
