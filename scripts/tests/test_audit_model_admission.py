"""Tests for scripts/audit_model_admission.py.

Hermetic: every test runs against a synthetic runs.db materialised into
tmp_path, never against the 181 MB live store the script defaults to.

The schema below is a VERBATIM copy of the three tables the audit reads,
captured with

    sqlite3 data/orchestrator/runs.db ".schema events invocations account_events"

Copied rather than imported because scripts/tests/ is collected by
`uv run --project shared pytest` and imports NO first-party package
(dark-factory-orchestrator.yaml:111-112) — so the orchestrator's own event
store, which owns this DDL, is out of reach here. Re-capture with that command
rather than hand-editing if the writer's schema moves.
"""
import json
import sqlite3

import pytest

RUNS_DB_SCHEMA = """
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
CREATE TABLE invocations (
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
CREATE TABLE account_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    account_name TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    project_id   TEXT,
    run_id       TEXT,
    details      TEXT,
    created_at   TEXT NOT NULL
);
"""


@pytest.fixture
def runs_db_path(tmp_path):
    """Path to a fresh, empty runs.db carrying :data:`RUNS_DB_SCHEMA`."""
    path = tmp_path / 'runs.db'
    conn = sqlite3.connect(path)
    try:
        conn.executescript(RUNS_DB_SCHEMA)
        conn.commit()
    finally:
        conn.close()
    return path


@pytest.fixture
def runs_db(runs_db_path):
    """A WRITABLE connection on :func:`runs_db_path`, for seeding scenarios.

    The audit's scan functions take an open connection, so a test normally
    seeds through this fixture and hands the same connection straight to the
    function under test. Tests that exercise the read-only connection factory
    or the CLI take ``runs_db_path`` instead — both name the same file.
    """
    conn = sqlite3.connect(runs_db_path)
    try:
        yield conn
    finally:
        conn.close()


def _payload(value):
    """JSON-encode a dict/list payload; pass a str or None through VERBATIM.

    The pass-through is what lets a test seed a deliberately malformed payload
    — the live store holds an ``account_events.details`` of the bare string
    ``'Escalation watcher (auto)'`` — so the audit's tolerant-parse paths are
    exercised against the real shape rather than a hypothetical one. Same
    convention as ``make_tasks_db``'s ``metadata`` handling in conftest.py.
    """
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value)


def _event(
    conn,
    timestamp,
    event_type,
    *,
    run_id='run-1',
    task_id=None,
    phase=None,
    role=None,
    data=None,
    cost_usd=None,
    duration_ms=None,
):
    """Insert one `events` row, stating its payload as a dict rather than JSON text."""
    conn.execute(
        'INSERT INTO events (timestamp, run_id, task_id, event_type, phase, role, '
        'data, cost_usd, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (
            timestamp,
            run_id,
            task_id,
            event_type,
            phase,
            role,
            _payload({} if data is None else data),
            cost_usd,
            duration_ms,
        ),
    )
    conn.commit()


def _invocation(
    conn,
    *,
    model,
    role,
    started_at,
    completed_at,
    run_id='run-1',
    task_id=None,
    project_id='dark_factory',
    account_name='max-a',
    cost_usd=0.0,
    input_tokens=None,
    output_tokens=None,
    cache_read_tokens=None,
    cache_create_tokens=None,
    duration_ms=0,
    capped=0,
):
    """Insert one `invocations` row."""
    conn.execute(
        'INSERT INTO invocations (run_id, task_id, project_id, account_name, model, role, '
        'cost_usd, input_tokens, output_tokens, cache_read_tokens, cache_create_tokens, '
        'duration_ms, capped, started_at, completed_at) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (
            run_id,
            task_id,
            project_id,
            account_name,
            model,
            role,
            cost_usd,
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_create_tokens,
            duration_ms,
            capped,
            started_at,
            completed_at,
        ),
    )
    conn.commit()


def _account_event(
    conn,
    *,
    account_name,
    event_type,
    created_at,
    details=None,
    project_id='dark_factory',
    run_id='run-1',
):
    """Insert one `account_events` row; *details* follows :func:`_payload`."""
    conn.execute(
        'INSERT INTO account_events (account_name, event_type, project_id, run_id, '
        'details, created_at) VALUES (?, ?, ?, ?, ?, ?)',
        (account_name, event_type, project_id, run_id, _payload(details), created_at),
    )
    conn.commit()
