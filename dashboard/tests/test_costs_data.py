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
    get_account_events,
    get_cost_by_account,
    get_cost_by_project,
    get_cost_by_role,
    get_cost_summary,
    get_cost_trend,
    get_run_cost_breakdown,
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

    @pytest.mark.asyncio
    async def test_days_window_boundary(self, tmp_path):
        """_cutoff(days) must exclude invocations older than the window.

        Insert two invocations: one 2 hours ago (recent) and one 20 days ago
        (old). days=1 should include only the recent one; days=30 should
        include both, producing a higher total_spend.
        """
        db_path = tmp_path / 'window.db'
        conn = __import__('sqlite3').connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)
        now = datetime.now(UTC)
        conn.executemany(
            'INSERT INTO invocations '
            '(run_id, task_id, project_id, account_name, model, role, '
            ' cost_usd, capped, started_at, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            [
                # Recent: 2 hours ago — inside any reasonable window
                ('r1', 't1', 'proj', 'acc', 'claude-opus-4-5', 'implementer',
                 0.50, 0,
                 (now - timedelta(hours=2, minutes=1)).isoformat(),
                 (now - timedelta(hours=2)).isoformat()),
                # Old: 20 days ago — outside 7d window, inside 30d window
                ('r2', 't2', 'proj', 'acc', 'claude-opus-4-5', 'implementer',
                 1.50, 0,
                 (now - timedelta(days=20, hours=1)).isoformat(),
                 (now - timedelta(days=20)).isoformat()),
            ],
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as aconn:
            aconn.row_factory = aiosqlite.Row
            narrow = await get_cost_summary(aconn, days=1)
            wide = await get_cost_summary(aconn, days=30)

        # days=1: only the 2h-ago invocation is included
        assert 'proj' in narrow
        assert narrow['proj']['total_spend'] == pytest.approx(0.50, abs=1e-6)

        # days=30: both invocations are included
        assert 'proj' in wide
        assert wide['proj']['total_spend'] == pytest.approx(2.00, abs=1e-6)


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
    async def test_last_cap_is_cap_hit_timestamp(self, tmp_path):
        """last_cap must hold the cap_hit timestamp, not the resumed timestamp.

        When an account has cap_hit at T1 then resumed at T2 (T2 > T1),
        last_cap must equal T1 (cap_hit). The current MAX(created_at)
        implementation returns T2 (resumed), which is wrong.
        """
        db_path = tmp_path / 'cap_order.db'
        conn = __import__('sqlite3').connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)

        now = datetime.now(UTC)
        cap_ts = (now - timedelta(hours=4)).isoformat()
        resumed_ts = (now - timedelta(hours=2)).isoformat()  # more recent

        conn.executemany(
            'INSERT INTO account_events '
            '(account_name, event_type, project_id, run_id, details, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            [
                ('test-acc', 'cap_hit', 'proj', 'r1', None, cap_ts),
                ('test-acc', 'resumed', 'proj', 'r1', None, resumed_ts),
            ],
        )
        conn.execute(
            'INSERT INTO invocations '
            '(run_id, task_id, project_id, account_name, model, role, '
            ' cost_usd, capped, started_at, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            ('r1', 't1', 'proj', 'test-acc', 'claude-opus-4-5', 'implementer',
             1.0, 0,
             (now - timedelta(hours=5)).isoformat(),
             (now - timedelta(hours=1)).isoformat()),
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as aconn:
            aconn.row_factory = aiosqlite.Row
            result = await get_cost_by_account(aconn)

        assert 'test-acc' in result
        ta = result['test-acc']
        # last_cap must be the cap_hit timestamp, not the resumed timestamp
        assert ta['last_cap'] == cap_ts, (
            f"expected last_cap={cap_ts!r} (cap_hit), got {ta['last_cap']!r}"
        )
        # Status: resumed is the most recent in-window event → active
        assert ta['status'] == 'active'

    @pytest.mark.asyncio
    async def test_out_of_window_events(self, tmp_path):
        """Status and last_cap respect the look-back window boundary.

        acct-x: old 'resumed' (outside window) + recent 'cap_hit' (inside) → capped.
        acct-y: two old events (outside) + recent 'cap_hit' (inside) → capped.
        Verifies the correlated subquery uses in-window events for status derivation.
        """
        db_path = tmp_path / 'window_events.db'
        conn = __import__('sqlite3').connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)

        now = datetime.now(UTC)
        acct_x_cap_ts = (now - timedelta(days=2)).isoformat()
        acct_y_cap_ts = (now - timedelta(days=1)).isoformat()

        conn.executemany(
            'INSERT INTO account_events '
            '(account_name, event_type, project_id, run_id, details, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            [
                # acct-x: resumed at -30d (outside window), cap_hit at -2d (inside)
                ('acct-x', 'resumed', 'proj', 'r1', None,
                 (now - timedelta(days=30)).isoformat()),
                ('acct-x', 'cap_hit', 'proj', 'r2', None, acct_x_cap_ts),
                # acct-y: cap_hit at -30d (outside), resumed at -20d (outside),
                #         cap_hit at -1d (inside)
                ('acct-y', 'cap_hit', 'proj', 'r3', None,
                 (now - timedelta(days=30)).isoformat()),
                ('acct-y', 'resumed', 'proj', 'r4', None,
                 (now - timedelta(days=20)).isoformat()),
                ('acct-y', 'cap_hit', 'proj', 'r5', None, acct_y_cap_ts),
            ],
        )
        # Invocations so both accounts appear in inv_rows
        conn.executemany(
            'INSERT INTO invocations '
            '(run_id, task_id, project_id, account_name, model, role, '
            ' cost_usd, capped, started_at, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            [
                ('r2', 't1', 'proj', 'acct-x', 'claude-opus-4-5', 'implementer',
                 0.5, 0,
                 (now - timedelta(days=2, hours=1)).isoformat(),
                 (now - timedelta(days=2)).isoformat()),
                ('r5', 't2', 'proj', 'acct-y', 'claude-opus-4-5', 'implementer',
                 0.5, 0,
                 (now - timedelta(days=1, hours=1)).isoformat(),
                 (now - timedelta(days=1)).isoformat()),
            ],
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as aconn:
            aconn.row_factory = aiosqlite.Row
            result = await get_cost_by_account(aconn, days=7)

        # acct-x: in-window cap_hit → capped; last_cap = cap_hit timestamp
        assert 'acct-x' in result
        ax = result['acct-x']
        assert ax['status'] == 'capped'
        assert ax['last_cap'] == acct_x_cap_ts

        # acct-y: only in-window event is cap_hit → capped
        assert 'acct-y' in result
        ay = result['acct-y']
        assert ay['status'] == 'capped'
        assert ay['last_cap'] == acct_y_cap_ts

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
        # 7 days total; only today (UTC) has spending; others should be 0.0.
        # Use a dynamic assertion so this passes near UTC midnight when fixture
        # invocations might span two calendar days.
        zero_days = [e for e in df if e['total'] == 0.0]
        # Verify gap-filling actually works: exactly 7 days, at least one with 0.0.
        assert len(df) == 7
        assert len(zero_days) > 0


# ---------------------------------------------------------------------------
# Tests: get_account_events
# ---------------------------------------------------------------------------

class TestAccountEvents:
    @pytest.mark.asyncio
    async def test_populated(self, costs_conn):
        result = await get_account_events(costs_conn)

        # Should return a list of dicts
        assert isinstance(result, list)
        assert len(result) == 3  # 3 account events in fixture

        # Required keys on every entry
        for entry in result:
            assert 'account_name' in entry
            assert 'event_type' in entry
            assert 'project_id' in entry
            assert 'run_id' in entry
            assert 'details' in entry
            assert 'created_at' in entry

        # Ordered by created_at DESC — most recent first
        timestamps = [e['created_at'] for e in result]
        assert timestamps == sorted(timestamps, reverse=True)

        # Event types present: cap_hit (x2) and resumed (x1)
        event_types = [e['event_type'] for e in result]
        assert event_types.count('cap_hit') == 2
        assert event_types.count('resumed') == 1

    @pytest.mark.asyncio
    async def test_none_db(self):
        result = await get_account_events(None)
        assert result == []

    @pytest.mark.asyncio
    async def test_filters_by_window(self, tmp_path):
        """Events older than the window should be excluded."""
        db_path = tmp_path / 'old_events.db'
        conn = __import__('sqlite3').connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)
        now = datetime.now(UTC)
        # One recent event and one very old event
        conn.executemany(
            'INSERT INTO account_events '
            '(account_name, event_type, project_id, run_id, details, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            [
                ('max-a', 'cap_hit', 'proj', 'r1', None, now.isoformat()),
                ('max-b', 'cap_hit', 'proj', 'r2', None,
                 (now - timedelta(days=30)).isoformat()),  # outside 7-day window
            ],
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as aconn:
            aconn.row_factory = aiosqlite.Row
            result = await get_account_events(aconn, days=7)

        assert len(result) == 1
        assert result[0]['account_name'] == 'max-a'


# ---------------------------------------------------------------------------
# Tests: get_run_cost_breakdown
# ---------------------------------------------------------------------------

class TestRunCostBreakdown:
    @pytest.mark.asyncio
    async def test_populated(self, costs_conn):
        """Returns list of runs with per-task breakdown and task titles from JOIN."""
        result = await get_run_cost_breakdown(costs_conn)

        # Should return a list
        assert isinstance(result, list)
        assert len(result) == 2  # run-001 (dark_factory) and run-002 (reify)

        # Find run-001
        run001 = next((r for r in result if r['run_id'] == 'run-001'), None)
        assert run001 is not None
        assert run001['project_id'] == 'dark_factory'
        assert run001['total_cost'] == pytest.approx(3.2, abs=1e-6)

        # run-001 has 3 tasks
        tasks = {t['task_id']: t for t in run001['tasks']}
        assert '101' in tasks
        assert '102' in tasks
        assert '103' in tasks

        # task 101: 1.00 + 0.50 = 1.50; title from task_results = 'Widget A'
        t101 = tasks['101']
        assert t101['cost'] == pytest.approx(1.5, abs=1e-6)
        assert t101['title'] == 'Widget A'
        assert isinstance(t101['invocations'], list)
        assert len(t101['invocations']) == 2  # implementer + reviewer

        # task 102: 0.80 + 0.30 = 1.10; title = 'Widget B'
        t102 = tasks['102']
        assert t102['cost'] == pytest.approx(1.1, abs=1e-6)
        assert t102['title'] == 'Widget B'

        # task 103: 0.60; title = 'Refactor C'
        t103 = tasks['103']
        assert t103['cost'] == pytest.approx(0.6, abs=1e-6)
        assert t103['title'] == 'Refactor C'

        # Find run-002
        run002 = next((r for r in result if r['run_id'] == 'run-002'), None)
        assert run002 is not None
        assert run002['project_id'] == 'reify'
        assert run002['total_cost'] == pytest.approx(0.4, abs=1e-6)
        assert len(run002['tasks']) == 1
        assert run002['tasks'][0]['task_id'] == '201'
        assert run002['tasks'][0]['title'] == 'Reify task'

    @pytest.mark.asyncio
    async def test_none_db(self):
        result = await get_run_cost_breakdown(None)
        assert result == []

    @pytest.mark.asyncio
    async def test_null_task_id(self, tmp_path):
        """Multiple NULL-task_id invocations are grouped into a single task entry.

        Two orchestrator invocations with task_id=NULL (run-level billing) must
        collapse into one task dict (task_id=None), with costs accumulated and
        both invocation rows preserved in the detail list.
        """
        db_path = tmp_path / 'null_task.db'
        conn = __import__('sqlite3').connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)
        now = datetime.now(UTC)

        # Insert a run
        conn.execute(
            'INSERT INTO runs (run_id, project_id, started_at, completed_at) '
            'VALUES (?, ?, ?, ?)',
            ('r1', 'proj', (now - timedelta(hours=1)).isoformat(), now.isoformat()),
        )
        # Insert TWO invocations with NULL task_id (run-level billing)
        conn.executemany(
            'INSERT INTO invocations '
            '(run_id, task_id, project_id, account_name, model, role, '
            ' cost_usd, capped, started_at, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            [
                ('r1', None, 'proj', 'acc', 'claude-sonnet-4-5', 'orchestrator',
                 0.25, 0,
                 (now - timedelta(minutes=40)).isoformat(),
                 (now - timedelta(minutes=30)).isoformat()),
                ('r1', None, 'proj', 'acc', 'claude-sonnet-4-5', 'orchestrator',
                 0.15, 0,
                 (now - timedelta(minutes=20)).isoformat(),
                 (now - timedelta(minutes=10)).isoformat()),
            ],
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as aconn:
            aconn.row_factory = aiosqlite.Row
            result = await get_run_cost_breakdown(aconn)

        assert len(result) == 1
        run = result[0]
        assert run['run_id'] == 'r1'
        assert run['total_cost'] == pytest.approx(0.40, abs=1e-6)

        # Both NULLs must be grouped into exactly ONE task entry
        assert len(run['tasks']) == 1
        t = run['tasks'][0]
        assert t['task_id'] is None
        assert t['title'] is None
        assert t['cost'] == pytest.approx(0.40, abs=1e-6)
        # Both invocation rows must be preserved in the detail list
        assert len(t['invocations']) == 2
