"""Shared test fixtures for dashboard tests."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest
from starlette.testclient import TestClient

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


@pytest.fixture()
def dashboard_config(tmp_path):
    """Create a DashboardConfig with tmp_path-based project_root."""
    from dashboard.config import DashboardConfig

    return DashboardConfig(project_root=tmp_path)


@pytest.fixture()
def client():
    """Create a TestClient for the dashboard FastAPI app."""
    from dashboard.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def reconciliation_db(tmp_path):
    """Create a temporary SQLite reconciliation DB with schema and sample data.

    Returns the path to the database file.
    """
    db_path = tmp_path / 'reconciliation.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(RECONCILIATION_SCHEMA)

    now = datetime.now(UTC)

    # Watermark row
    conn.execute(
        """INSERT INTO watermarks
           (project_id, last_full_run_id, last_full_run_completed,
            last_episode_timestamp, last_memory_timestamp, last_task_change_timestamp)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            'dark_factory',
            'run-001',
            (now - timedelta(hours=1)).isoformat(),
            (now - timedelta(minutes=30)).isoformat(),
            (now - timedelta(minutes=20)).isoformat(),
            (now - timedelta(minutes=10)).isoformat(),
        ),
    )

    # Runs: one completed, one still running
    completed_started = now - timedelta(hours=2)
    completed_finished = completed_started + timedelta(minutes=5)
    conn.execute(
        """INSERT INTO runs
           (id, project_id, run_type, trigger_reason, started_at, completed_at,
            events_processed, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            'run-001',
            'dark_factory',
            'full',
            'staleness_timer',
            completed_started.isoformat(),
            completed_finished.isoformat(),
            7,
            'completed',
        ),
    )
    running_started = now - timedelta(minutes=10)
    conn.execute(
        """INSERT INTO runs
           (id, project_id, run_type, trigger_reason, started_at, completed_at,
            events_processed, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            'run-002',
            'dark_factory',
            'incremental',
            'event_threshold',
            running_started.isoformat(),
            None,
            3,
            'running',
        ),
    )

    # Event buffer: 3 events with varying timestamps
    for i, minutes_ago in enumerate([60, 30, 5]):
        conn.execute(
            """INSERT INTO event_buffer
               (id, project_id, event_type, event_source, agent_id, timestamp, payload, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                f'evt-{i + 1:03d}',
                'dark_factory',
                'memory_write',
                'interceptor',
                f'agent-{i + 1}',
                (now - timedelta(minutes=minutes_ago)).isoformat(),
                '{}',
                'buffered',
            ),
        )

    # Burst state: 2 rows
    conn.execute(
        """INSERT INTO burst_state (agent_id, state, last_write_at, burst_started_at)
           VALUES (?, ?, ?, ?)""",
        (
            'agent-1',
            'bursting',
            (now - timedelta(minutes=2)).isoformat(),
            (now - timedelta(minutes=10)).isoformat(),
        ),
    )
    conn.execute(
        """INSERT INTO burst_state (agent_id, state, last_write_at, burst_started_at)
           VALUES (?, ?, ?, ?)""",
        (
            'agent-2',
            'idle',
            (now - timedelta(hours=1)).isoformat(),
            None,
        ),
    )

    # Journal entries for run-001
    conn.execute(
        """INSERT INTO journal_entries
           (id, run_id, stage, timestamp, operation, target_system,
            before_state, after_state, reasoning, evidence)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            'je-001',
            'run-001',
            'memory_consolidation',
            (now - timedelta(hours=1, minutes=55)).isoformat(),
            'consolidate',
            'mem0',
            '{"count": 5}',
            '{"count": 3}',
            'Merged duplicate memories',
            '[{"source": "mem0", "id": "m-1"}]',
        ),
    )
    conn.execute(
        """INSERT INTO journal_entries
           (id, run_id, stage, timestamp, operation, target_system,
            before_state, after_state, reasoning, evidence)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            'je-002',
            'run-001',
            'task_knowledge_sync',
            (now - timedelta(hours=1, minutes=50)).isoformat(),
            'sync',
            'graphiti',
            None,
            '{"entities": 2}',
            '',
            '[]',
        ),
    )

    # Judge verdict
    conn.execute(
        """INSERT INTO judge_verdicts (run_id, reviewed_at, severity, findings, action_taken)
           VALUES (?, ?, ?, ?, ?)""",
        (
            'run-001',
            (now - timedelta(hours=1)).isoformat(),
            'low',
            '[{"issue": "minor drift"}]',
            'logged',
        ),
    )

    conn.commit()
    conn.close()

    yield db_path


@pytest.fixture()
def empty_reconciliation_db(tmp_path):
    """Create a reconciliation DB with schema but no data.

    Returns the path to the database file.
    """
    db_path = tmp_path / 'empty_reconciliation.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(RECONCILIATION_SCHEMA)
    conn.commit()
    conn.close()

    yield db_path


@pytest.fixture()
def missing_db_path(tmp_path):
    """Return a path to a non-existent database file."""
    return tmp_path / 'nonexistent' / 'reconciliation.db'
