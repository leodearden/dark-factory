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


# ---------------------------------------------------------------------------
# Tests: get_cost_summary
# ---------------------------------------------------------------------------

from dashboard.data.costs import (  # noqa: E402
    get_cost_by_account,
    get_cost_by_project,
    get_cost_by_role,
    get_cost_summary,
    get_cost_trend,
)


class TestCostSummary:
    @pytest.mark.asyncio
    async def test_populated(self, costs_conn):
        result = await get_cost_summary(costs_conn)

        assert 'dark_factory' in result
        df = result['dark_factory']
        assert df['total_spend'] == pytest.approx(3.2, abs=1e-6)
        assert df['avg_cost_per_task'] == pytest.approx(3.2 / 3, abs=1e-6)
        assert df['active_accounts'] == 2  # max-a and max-b
        assert df['cap_events'] == 2        # 2 cap_hit events in dark_factory

        assert 'reify' in result
        r = result['reify']
        assert r['total_spend'] == pytest.approx(0.4, abs=1e-6)
        assert r['avg_cost_per_task'] == pytest.approx(0.4, abs=1e-6)
        assert r['active_accounts'] == 1   # max-a only
        assert r['cap_events'] == 0         # no cap events for reify

    @pytest.mark.asyncio
    async def test_none_db(self):
        result = await get_cost_summary(None)
        assert result == {}

    @pytest.mark.asyncio
    async def test_empty_db(self, empty_costs_conn):
        result = await get_cost_summary(empty_costs_conn)
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: get_cost_by_project
# ---------------------------------------------------------------------------

class TestCostByProject:
    @pytest.mark.asyncio
    async def test_populated(self, costs_conn):
        result = await get_cost_by_project(costs_conn)

        assert 'dark_factory' in result
        df_models = {entry['model']: entry['total'] for entry in result['dark_factory']}
        # opus: 1.00 + 0.80 + 0.60 = 2.40; sonnet: 0.50 + 0.30 = 0.80
        assert df_models['claude-opus-4-5'] == pytest.approx(2.4, abs=1e-6)
        assert df_models['claude-sonnet-4-5'] == pytest.approx(0.8, abs=1e-6)

        assert 'reify' in result
        r_models = {entry['model']: entry['total'] for entry in result['reify']}
        assert r_models['claude-sonnet-4-5'] == pytest.approx(0.4, abs=1e-6)

    @pytest.mark.asyncio
    async def test_none_db(self):
        result = await get_cost_by_project(None)
        assert result == {}

    @pytest.mark.asyncio
    async def test_multi_project(self, costs_conn):
        result = await get_cost_by_project(costs_conn)
        assert 'dark_factory' in result
        assert 'reify' in result
        assert len(result['dark_factory']) == 2   # opus and sonnet
        assert len(result['reify']) == 1           # sonnet only


# ---------------------------------------------------------------------------
# Tests: get_cost_by_account
# ---------------------------------------------------------------------------

class TestCostByAccount:
    @pytest.mark.asyncio
    async def test_populated(self, costs_conn):
        result = await get_cost_by_account(costs_conn)

        # max-a: invocations in dark_factory (1.00+0.50+0.60) + reify (0.40) = 2.50
        assert 'max-a' in result
        ma = result['max-a']
        assert ma['spend'] == pytest.approx(2.5, abs=1e-6)
        assert ma['invocations'] == 4
        assert ma['cap_events'] == 1   # one cap_hit event for max-a
        assert ma['last_cap'] is not None
        assert ma['status'] == 'capped'  # cap_hit with no subsequent resumed

        # max-b: invocations 0.80+0.30 = 1.10; cap_hit then resumed → active
        assert 'max-b' in result
        mb = result['max-b']
        assert mb['spend'] == pytest.approx(1.1, abs=1e-6)
        assert mb['invocations'] == 2
        assert mb['cap_events'] == 1   # one cap_hit event for max-b
        assert mb['last_cap'] is not None
        assert mb['status'] == 'active'  # resumed is most recent event

    @pytest.mark.asyncio
    async def test_none_db(self):
        result = await get_cost_by_account(None)
        assert result == {}

    @pytest.mark.asyncio
    async def test_account_with_no_caps(self, tmp_path):
        """Account that has invocations but no cap events has cap_events=0, last_cap=None."""
        db_path = tmp_path / 'nocap.db'
        conn = __import__('sqlite3').connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)
        now = datetime.now(UTC)
        conn.execute(
            'INSERT INTO invocations '
            '(run_id, task_id, project_id, account_name, model, role, '
            ' cost_usd, capped, started_at, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            ('r1', 't1', 'proj', 'max-z', 'claude-opus-4-5', 'implementer',
             0.5, 0, (now - timedelta(hours=1)).isoformat(), now.isoformat()),
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as aconn:
            aconn.row_factory = aiosqlite.Row
            result = await get_cost_by_account(aconn)

        assert 'max-z' in result
        mz = result['max-z']
        assert mz['cap_events'] == 0
        assert mz['last_cap'] is None
        assert mz['status'] == 'active'


# ---------------------------------------------------------------------------
# Tests: get_cost_by_role
# ---------------------------------------------------------------------------

class TestCostByRole:
    @pytest.mark.asyncio
    async def test_populated(self, costs_conn):
        result = await get_cost_by_role(costs_conn)

        assert 'dark_factory' in result
        df = result['dark_factory']

        # implementer → opus: 1.00 (task 101 max-a) + 0.80 (task 102 max-b) = 1.80
        assert df['implementer']['claude-opus-4-5'] == pytest.approx(1.8, abs=1e-6)
        # reviewer → sonnet: 0.50 (task 101 max-a) + 0.30 (task 102 max-b) = 0.80
        assert df['reviewer']['claude-sonnet-4-5'] == pytest.approx(0.8, abs=1e-6)
        # debugger → opus: 0.60 (task 103 max-a)
        assert df['debugger']['claude-opus-4-5'] == pytest.approx(0.6, abs=1e-6)

        assert 'reify' in result
        r = result['reify']
        # implementer → sonnet: 0.40
        assert r['implementer']['claude-sonnet-4-5'] == pytest.approx(0.4, abs=1e-6)

    @pytest.mark.asyncio
    async def test_none_db(self):
        result = await get_cost_by_role(None)
        assert result == {}

    @pytest.mark.asyncio
    async def test_single_role(self, empty_costs_conn, tmp_path):
        """Single role — structure should still be {project_id: {role: {model: total}}}."""
        db_path = tmp_path / 'single.db'
        conn = __import__('sqlite3').connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)
        now = datetime.now(UTC)
        conn.execute(
            'INSERT INTO invocations '
            '(run_id, task_id, project_id, account_name, model, role, '
            ' cost_usd, capped, started_at, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            ('r1', 't1', 'solo', 'acc', 'claude-opus-4-5', 'implementer',
             1.5, 0, (now - timedelta(hours=1)).isoformat(), now.isoformat()),
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as aconn:
            aconn.row_factory = aiosqlite.Row
            result = await get_cost_by_role(aconn)

        assert result == {'solo': {'implementer': {'claude-opus-4-5': pytest.approx(1.5)}}}


# ---------------------------------------------------------------------------
# Tests: get_cost_trend
# ---------------------------------------------------------------------------

class TestCostTrend:
    @pytest.mark.asyncio
    async def test_populated(self, costs_conn):
        result = await get_cost_trend(costs_conn, days=7)

        assert 'dark_factory' in result
        df = result['dark_factory']
        assert isinstance(df, list)
        # All entries should have day and total keys
        for entry in df:
            assert 'day' in entry
            assert 'total' in entry
            assert isinstance(entry['total'], float)

        # There should be exactly 7 days in the result
        assert len(df) == 7

        # Entries should be in chronological order
        days_sorted = [entry['day'] for entry in df]
        assert days_sorted == sorted(days_sorted)

        # Total for today's day should include fixture invocations
        # (all invocations were within last few hours, so all on today)
        total_all = sum(entry['total'] for entry in df)
        assert total_all == pytest.approx(3.2, abs=1e-6)  # dark_factory total

        assert 'reify' in result
        total_reify = sum(entry['total'] for entry in result['reify'])
        assert total_reify == pytest.approx(0.4, abs=1e-6)

    @pytest.mark.asyncio
    async def test_none_db(self):
        result = await get_cost_trend(None)
        assert result == {}

    @pytest.mark.asyncio
    async def test_fills_gaps(self, costs_conn):
        """All days in the window should appear, even those with no spending."""
        result = await get_cost_trend(costs_conn, days=7)
        assert 'dark_factory' in result
        df = result['dark_factory']
        # 7 days total; only today has spending; others should be 0.0
        zero_days = [e for e in df if e['total'] == 0.0]
        assert len(zero_days) == 6  # 6 days back have no data
