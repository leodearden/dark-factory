"""Tests for dashboard.data.costs — cost query functions."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import aiosqlite
import pytest

# ---------------------------------------------------------------------------
# Schema — all four tables that coexist in runs.db
# ---------------------------------------------------------------------------

COSTS_SCHEMA = """\
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

CREATE TABLE IF NOT EXISTS invocations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL,
    task_id             TEXT,
    project_id          TEXT NOT NULL,
    account_name        TEXT NOT NULL,
    model               TEXT NOT NULL,
    role                TEXT NOT NULL,
    cost_usd            REAL NOT NULL DEFAULT 0.0,
    input_tokens        INTEGER,
    output_tokens       INTEGER,
    cache_read_tokens   INTEGER,
    cache_create_tokens INTEGER,
    duration_ms         INTEGER NOT NULL DEFAULT 0,
    capped              INTEGER NOT NULL DEFAULT 0,
    started_at          TEXT NOT NULL,
    completed_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_inv_project
    ON invocations(project_id);

CREATE INDEX IF NOT EXISTS idx_inv_account
    ON invocations(account_name);

CREATE INDEX IF NOT EXISTS idx_inv_run
    ON invocations(run_id);

CREATE TABLE IF NOT EXISTS account_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    account_name TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    project_id   TEXT,
    run_id       TEXT,
    details      TEXT,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_acct_evt_account
    ON account_events(account_name);
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def costs_db(tmp_path):
    """Populated runs.db with invocations and account events across two projects."""
    db_path = tmp_path / 'runs.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(COSTS_SCHEMA)

    now = datetime.now(UTC)

    # Runs
    conn.executemany(
        'INSERT INTO runs (run_id, project_id, started_at, completed_at) VALUES (?, ?, ?, ?)',
        [
            ('run-001', 'dark_factory',
             (now - timedelta(hours=3)).isoformat(),
             (now - timedelta(hours=0, minutes=30)).isoformat()),
            ('run-002', 'reify',
             (now - timedelta(hours=4)).isoformat(),
             (now - timedelta(hours=2)).isoformat()),
        ],
    )

    # Task results (needed for get_run_cost_breakdown JOIN)
    conn.executemany(
        'INSERT INTO task_results (run_id, task_id, project_id, title, outcome, completed_at) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        [
            ('run-001', '101', 'dark_factory', 'Widget A', 'done',
             (now - timedelta(hours=2)).isoformat()),
            ('run-001', '102', 'dark_factory', 'Widget B', 'done',
             (now - timedelta(hours=1, minutes=30)).isoformat()),
            ('run-001', '103', 'dark_factory', 'Refactor C', 'done',
             (now - timedelta(hours=1)).isoformat()),
            ('run-002', '201', 'reify', 'Reify task', 'done',
             (now - timedelta(hours=3)).isoformat()),
        ],
    )

    # Invocations
    # dark_factory — run-001:
    #   task 101: max-a/opus/implementer (1.00) + max-a/sonnet/reviewer (0.50)
    #   task 102: max-b/opus/implementer (0.80) + max-b/sonnet/reviewer (0.30)
    #   task 103: max-a/opus/debugger    (0.60, capped)
    # reify — run-002:
    #   task 201: max-a/sonnet/implementer (0.40)
    invocations = [
        ('run-001', '101', 'dark_factory', 'max-a', 'claude-opus-4-5',
         'implementer', 1.00, 0,
         (now - timedelta(hours=2, minutes=5)).isoformat(),
         (now - timedelta(hours=2)).isoformat()),
        ('run-001', '101', 'dark_factory', 'max-a', 'claude-sonnet-4-5',
         'reviewer', 0.50, 0,
         (now - timedelta(hours=1, minutes=50)).isoformat(),
         (now - timedelta(hours=1, minutes=45)).isoformat()),
        ('run-001', '102', 'dark_factory', 'max-b', 'claude-opus-4-5',
         'implementer', 0.80, 0,
         (now - timedelta(hours=1, minutes=40)).isoformat(),
         (now - timedelta(hours=1, minutes=35)).isoformat()),
        ('run-001', '102', 'dark_factory', 'max-b', 'claude-sonnet-4-5',
         'reviewer', 0.30, 0,
         (now - timedelta(hours=1, minutes=30)).isoformat(),
         (now - timedelta(hours=1, minutes=25)).isoformat()),
        ('run-001', '103', 'dark_factory', 'max-a', 'claude-opus-4-5',
         'debugger', 0.60, 1,
         (now - timedelta(hours=1, minutes=10)).isoformat(),
         (now - timedelta(hours=1)).isoformat()),
        ('run-002', '201', 'reify', 'max-a', 'claude-sonnet-4-5',
         'implementer', 0.40, 0,
         (now - timedelta(hours=3, minutes=10)).isoformat(),
         (now - timedelta(hours=3)).isoformat()),
    ]
    conn.executemany(
        'INSERT INTO invocations '
        '(run_id, task_id, project_id, account_name, model, role, '
        ' cost_usd, capped, started_at, completed_at) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        invocations,
    )

    # Account events:
    #   max-a: one cap_hit in dark_factory (no resumed) → status=capped
    #   max-b: cap_hit then resumed in dark_factory     → status=active
    account_events = [
        ('max-b', 'cap_hit', 'dark_factory', 'run-001', None,
         (now - timedelta(hours=2)).isoformat()),
        ('max-b', 'resumed', 'dark_factory', 'run-001', None,
         (now - timedelta(hours=1, minutes=30)).isoformat()),
        ('max-a', 'cap_hit', 'dark_factory', 'run-001', None,
         (now - timedelta(minutes=30)).isoformat()),
    ]
    conn.executemany(
        'INSERT INTO account_events '
        '(account_name, event_type, project_id, run_id, details, created_at) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        account_events,
    )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
def empty_costs_db(tmp_path):
    """Empty runs.db with schema only — no data."""
    db_path = tmp_path / 'empty_runs.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(COSTS_SCHEMA)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
async def costs_conn(costs_db):
    async with aiosqlite.connect(str(costs_db)) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn


@pytest.fixture()
async def empty_costs_conn(empty_costs_db):
    async with aiosqlite.connect(str(empty_costs_db)) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn
