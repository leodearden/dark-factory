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

CREATE INDEX IF NOT EXISTS idx_inv_completed_at
    ON invocations(completed_at);

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
    async def test_task_count_present(self, costs_conn):
        """task_count must be present in each project dict and equal distinct task IDs."""
        result = await get_cost_summary(costs_conn)

        # dark_factory has 3 distinct task IDs: 101, 102, 103
        assert 'dark_factory' in result
        df = result['dark_factory']
        assert 'task_count' in df, "task_count key missing from dark_factory summary"
        assert df['task_count'] == 3

        # reify has 1 distinct task ID: 201
        assert 'reify' in result
        r = result['reify']
        assert 'task_count' in r, "task_count key missing from reify summary"
        assert r['task_count'] == 1

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
    async def test_cap_only_account(self, tmp_path):
        """Account with cap_hit events but zero invocations IS present in results.

        get_cost_by_account performs a second pass over caps.keys() after the
        inv_rows loop to emit any account that has cap events but no invocations
        in the window. This makes capped-but-idle accounts visible in the
        dashboard, which is important for ops awareness.

        Expected: spend=0.0, invocations=0, cap_events=2, status='capped',
        last_cap set to the most recent cap_hit timestamp.
        """
        db_path = tmp_path / 'cap_only.db'
        conn = __import__('sqlite3').connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)
        now = datetime.now(UTC)

        cap_ts_1 = (now - timedelta(hours=3)).isoformat()
        cap_ts_2 = (now - timedelta(hours=1)).isoformat()

        # Insert cap_hit events only — no corresponding invocations
        conn.executemany(
            'INSERT INTO account_events '
            '(account_name, event_type, project_id, run_id, details, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            [
                ('max-cap-only', 'cap_hit', 'proj', 'r1', None, cap_ts_1),
                ('max-cap-only', 'cap_hit', 'proj', 'r2', None, cap_ts_2),
            ],
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as aconn:
            aconn.row_factory = aiosqlite.Row
            result = await get_cost_by_account(aconn)

        # Account with only cap events must be present — capped-but-idle accounts
        # are operationally significant and must appear in the dashboard.
        assert 'max-cap-only' in result, (
            "cap-only account must appear in get_cost_by_account result"
        )
        mco = result['max-cap-only']
        assert mco['spend'] == 0.0
        assert mco['invocations'] == 0
        assert mco['cap_events'] == 2
        assert mco['status'] == 'capped'
        assert mco['last_cap'] is not None

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

    @pytest.mark.asyncio
    async def test_cap_only_account_with_out_of_window_invocations(self, tmp_path):
        """Account with in-window cap event but only out-of-window invocations appears in result.

        This is a more realistic production edge case than test_cap_only_account:
        the account ('max-ghost') has invocations that aged out of the look-back
        window, so inv_rows yields zero rows for it. However, it still has a
        cap_hit event inside the window. The second-pass gap-fill (costs.py line 195)
        must surface this account with spend=0.0 and invocations=0.

        Setup:
          - 2 invocations for 'max-ghost' at 20d ago (outside days=7)
          - 1 cap_hit event for 'max-ghost' at 2h ago (inside days=7)

        Expected result for 'max-ghost':
          spend=0.0, invocations=0, cap_events=1, status='capped',
          last_cap == cap_hit timestamp
        """
        db_path = tmp_path / 'ghost_cap.db'
        conn = __import__('sqlite3').connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)
        now = datetime.now(UTC)

        cap_ts = (now - timedelta(hours=2)).isoformat()

        # Two invocations at 20d ago — outside days=7 window
        conn.executemany(
            'INSERT INTO invocations '
            '(run_id, task_id, project_id, account_name, model, role, '
            ' cost_usd, capped, started_at, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            [
                ('r1', 't1', 'proj', 'max-ghost', 'claude-opus-4-5', 'implementer',
                 1.00, 0,
                 (now - timedelta(days=20, hours=1)).isoformat(),
                 (now - timedelta(days=20)).isoformat()),
                ('r1', 't2', 'proj', 'max-ghost', 'claude-sonnet-4-5', 'reviewer',
                 0.50, 0,
                 (now - timedelta(days=20, hours=2)).isoformat(),
                 (now - timedelta(days=20, hours=1)).isoformat()),
            ],
        )
        # One cap_hit event at 2h ago — inside days=7 window
        conn.execute(
            'INSERT INTO account_events '
            '(account_name, event_type, project_id, run_id, details, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            ('max-ghost', 'cap_hit', 'proj', 'r1', None, cap_ts),
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as aconn:
            aconn.row_factory = aiosqlite.Row
            result = await get_cost_by_account(aconn, days=7)

        # 'max-ghost' must appear via the second-pass gap-fill even though it
        # has no in-window invocations.
        assert 'max-ghost' in result, (
            "account with out-of-window invocations but in-window cap event "
            "must appear in get_cost_by_account result via second-pass gap-fill"
        )
        mg = result['max-ghost']
        assert mg['spend'] == pytest.approx(0.0, abs=1e-6)
        assert mg['invocations'] == 0
        assert mg['cap_events'] == 1
        assert mg['status'] == 'capped'
        assert mg['last_cap'] == cap_ts, (
            f"expected last_cap={cap_ts!r}, got {mg['last_cap']!r}"
        )

    @pytest.mark.asyncio
    async def test_unrelated_event_after_cap_hit_stays_capped(self, tmp_path):
        """cap_hit followed by auth_failed/failover must stay 'capped'.

        Regression: previous logic picked the latest event of ANY type and
        treated a non-cap_hit latest event as 'active', which wrongly flipped
        accounts back to active when the only thing that had happened was a
        later auth failure or failover.  Only a 'resumed' event clears a cap.
        """
        db_path = tmp_path / 'post_cap_events.db'
        conn = sqlite3.connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)

        now = datetime.now(UTC)
        cap_ts = (now - timedelta(hours=4)).isoformat()
        auth_fail_ts = (now - timedelta(hours=2)).isoformat()  # later, non-cap
        failover_ts = (now - timedelta(hours=1)).isoformat()   # later, non-cap

        conn.executemany(
            'INSERT INTO account_events '
            '(account_name, event_type, project_id, run_id, details, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            [
                ('test-acc', 'cap_hit', 'proj', 'r1', None, cap_ts),
                ('test-acc', 'auth_failed', 'proj', 'r1', None, auth_fail_ts),
                ('test-acc', 'failover', 'proj', 'r1', None, failover_ts),
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

        ta = result['test-acc']
        assert ta['status'] == 'capped', (
            'cap_hit with later auth_failed/failover but no resumed must stay capped'
        )
        assert ta['last_cap'] == cap_ts

    @pytest.mark.asyncio
    async def test_resets_at_extracted_from_details(self, tmp_path):
        """resets_at JSON field in cap_hit details is surfaced on the row."""
        db_path = tmp_path / 'resets_at.db'
        conn = sqlite3.connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)

        now = datetime.now(UTC)
        cap_ts = (now - timedelta(hours=1)).isoformat()
        reset_iso = (now + timedelta(hours=3)).isoformat()
        details = '{"reason": "You\'ve hit your limit", "resets_at": "' + reset_iso + '"}'

        conn.execute(
            'INSERT INTO account_events '
            '(account_name, event_type, project_id, run_id, details, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            ('test-acc', 'cap_hit', 'proj', 'r1', details, cap_ts),
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as aconn:
            aconn.row_factory = aiosqlite.Row
            result = await get_cost_by_account(aconn)

        ta = result['test-acc']
        assert ta['status'] == 'capped'
        assert ta['resets_at'] == reset_iso

    @pytest.mark.asyncio
    async def test_resets_at_none_for_legacy_details(self, tmp_path):
        """Legacy cap_hit rows with an unparseable reason expose resets_at=None.

        The transitional fallback (removes on 2026-04-28) only recognises
        specific time-wording formats; a bare "hit your limit" has nothing to
        extract and should not synthesize a bogus countdown.
        """
        db_path = tmp_path / 'legacy_details.db'
        conn = sqlite3.connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)

        now = datetime.now(UTC)
        cap_ts = (now - timedelta(hours=1)).isoformat()
        # Legacy shape: only reason, no resets_at, no parseable time info
        legacy_details = '{"reason": "You\'ve hit your limit"}'

        conn.execute(
            'INSERT INTO account_events '
            '(account_name, event_type, project_id, run_id, details, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            ('test-acc', 'cap_hit', 'proj', 'r1', legacy_details, cap_ts),
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as aconn:
            aconn.row_factory = aiosqlite.Row
            result = await get_cost_by_account(aconn)

        ta = result['test-acc']
        assert ta['status'] == 'capped'
        assert ta['resets_at'] is None

    @pytest.mark.asyncio
    async def test_resets_at_fallback_parses_relative(self, tmp_path):
        """Transitional fallback parses 'resets in Xh' from legacy rows.

        Remove this test with the fallback on 2026-04-28.
        """
        db_path = tmp_path / 'fallback_relative.db'
        conn = sqlite3.connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)

        now = datetime.now(UTC)
        cap_ts = (now - timedelta(minutes=1)).isoformat()
        legacy_details = '{"reason": "You\'ve hit your limit — resets in 3h"}'

        conn.execute(
            'INSERT INTO account_events '
            '(account_name, event_type, project_id, run_id, details, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            ('test-acc', 'cap_hit', 'proj', 'r1', legacy_details, cap_ts),
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as aconn:
            aconn.row_factory = aiosqlite.Row
            result = await get_cost_by_account(aconn)

        ta = result['test-acc']
        assert ta['status'] == 'capped'
        parsed = datetime.fromisoformat(ta['resets_at'])
        expected = datetime.now(UTC) + timedelta(hours=3)
        delta_s = abs((parsed - expected).total_seconds())
        assert delta_s < 60, (
            f'expected ~3h ahead, got {ta["resets_at"]!r} '
            f'(delta {delta_s:.0f}s from expected)'
        )

    @pytest.mark.asyncio
    async def test_resets_at_fallback_parses_absolute(self, tmp_path):
        """Transitional fallback parses 'resets Xpm (tz)' from legacy rows.

        Remove this test with the fallback on 2026-04-28.
        """
        db_path = tmp_path / 'fallback_absolute.db'
        conn = sqlite3.connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)

        now = datetime.now(UTC)
        cap_ts = (now - timedelta(minutes=1)).isoformat()
        legacy_details = (
            '{"reason": "You\'ve hit your limit \\u00b7 resets 7pm (Europe/London)"}'
        )

        conn.execute(
            'INSERT INTO account_events '
            '(account_name, event_type, project_id, run_id, details, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            ('test-acc', 'cap_hit', 'proj', 'r1', legacy_details, cap_ts),
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as aconn:
            aconn.row_factory = aiosqlite.Row
            result = await get_cost_by_account(aconn)

        ta = result['test-acc']
        assert ta['status'] == 'capped'
        # Fallback produces a UTC ISO string that parses and is in the future.
        # Full exact-match would couple to local date/DST; not needed.
        parsed = datetime.fromisoformat(ta['resets_at'])
        assert parsed > datetime.now(UTC), (
            f'expected future resets_at, got {ta["resets_at"]!r}'
        )

    @pytest.mark.asyncio
    async def test_resets_at_persisted_wins_over_fallback(self, tmp_path):
        """When both persisted field and parseable reason exist, the persisted
        field is authoritative (the fallback is strictly a legacy bridge).

        Remove this test with the fallback on 2026-04-28.
        """
        db_path = tmp_path / 'both.db'
        conn = sqlite3.connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)

        now = datetime.now(UTC)
        cap_ts = (now - timedelta(minutes=1)).isoformat()
        persisted = (now + timedelta(hours=7)).isoformat()
        # Reason would parse to +3h; persisted says +7h; persisted must win.
        details = ('{"reason": "You\'ve hit your limit — resets in 3h", '
                   f'"resets_at": "{persisted}"}}')

        conn.execute(
            'INSERT INTO account_events '
            '(account_name, event_type, project_id, run_id, details, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            ('test-acc', 'cap_hit', 'proj', 'r1', details, cap_ts),
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as aconn:
            aconn.row_factory = aiosqlite.Row
            result = await get_cost_by_account(aconn)

        assert result['test-acc']['resets_at'] == persisted

    @pytest.mark.asyncio
    async def test_resets_at_none_when_active(self, tmp_path):
        """An account that ended 'active' exposes resets_at=None even if its
        latest cap_hit carried a resets_at — the value is only meaningful for
        currently-capped accounts.
        """
        db_path = tmp_path / 'active_resets_at.db'
        conn = sqlite3.connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)

        now = datetime.now(UTC)
        cap_ts = (now - timedelta(hours=3)).isoformat()
        resumed_ts = (now - timedelta(hours=1)).isoformat()  # later → active
        reset_iso = (now + timedelta(hours=3)).isoformat()
        details = '{"reason": "limit", "resets_at": "' + reset_iso + '"}'

        conn.executemany(
            'INSERT INTO account_events '
            '(account_name, event_type, project_id, run_id, details, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            [
                ('test-acc', 'cap_hit', 'proj', 'r1', details, cap_ts),
                ('test-acc', 'resumed', 'proj', 'r1', None, resumed_ts),
            ],
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as aconn:
            aconn.row_factory = aiosqlite.Row
            result = await get_cost_by_account(aconn)

        ta = result['test-acc']
        assert ta['status'] == 'active'
        assert ta['resets_at'] is None

    @pytest.mark.asyncio
    async def test_auth_failed_only_is_capped(self, tmp_path):
        """Account with only an auth_failed event (no cap_hit, no invocations)
        is reported as 'capped' and surfaces a parsed resets_at from the
        reason text.

        This is the dominant production case since Anthropic switched to
        returning "out of extra usage" as HTTP 429 — UsageGate routes 4xx
        through _handle_auth_failure, which writes auth_failed (not cap_hit).
        Without this case the dashboard would mis-report the account as
        'active' while the gate has it locked out.
        """
        db_path = tmp_path / 'auth_failed_only.db'
        conn = sqlite3.connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)

        now = datetime.now(UTC)
        auth_fail_ts = (now - timedelta(minutes=10)).isoformat()
        # Same reason wording the orchestrator persists for HTTP 429 today.
        details = (
            '{"reason": "HTTP 429: You\'re out of extra usage '
            '\\u00b7 resets in 3h"}'
        )

        conn.execute(
            'INSERT INTO account_events '
            '(account_name, event_type, project_id, run_id, details, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            ('test-acc', 'auth_failed', 'proj', 'r1', details, auth_fail_ts),
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as aconn:
            aconn.row_factory = aiosqlite.Row
            result = await get_cost_by_account(aconn)

        ta = result['test-acc']
        assert ta['status'] == 'capped', (
            'auth_failed without recovery must report capped'
        )
        assert ta['last_auth_fail'] == auth_fail_ts
        assert ta['last_cap'] is None
        # Reason fallback parses "resets in 3h" → ~3h ahead.
        parsed = datetime.fromisoformat(ta['resets_at'])
        delta_s = abs((parsed - (now + timedelta(hours=3))).total_seconds())
        assert delta_s < 60

    @pytest.mark.asyncio
    async def test_auth_resumed_clears_auth_failed(self, tmp_path):
        """auth_failed followed by a later auth_resumed clears unavailability."""
        db_path = tmp_path / 'auth_failed_then_resumed.db'
        conn = sqlite3.connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)

        now = datetime.now(UTC)
        auth_fail_ts = (now - timedelta(hours=2)).isoformat()
        auth_resumed_ts = (now - timedelta(hours=1)).isoformat()

        conn.executemany(
            'INSERT INTO account_events '
            '(account_name, event_type, project_id, run_id, details, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            [
                ('test-acc', 'auth_failed', 'proj', 'r1', None, auth_fail_ts),
                ('test-acc', 'auth_resumed', 'proj', 'r1', None, auth_resumed_ts),
            ],
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as aconn:
            aconn.row_factory = aiosqlite.Row
            result = await get_cost_by_account(aconn)

        ta = result['test-acc']
        assert ta['status'] == 'active'
        assert ta['resets_at'] is None

    @pytest.mark.asyncio
    async def test_resumed_after_auth_failed_clears(self, tmp_path):
        """A regular 'resumed' event later than auth_failed also clears.

        Defensive: usage_gate today only writes auth_resumed for the auth-fail
        recovery path, but the dashboard should not depend on that asymmetry.
        Any recovery event (resumed OR auth_resumed) clears any unavailability
        event (cap_hit OR auth_failed).
        """
        db_path = tmp_path / 'resumed_clears_auth.db'
        conn = sqlite3.connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)

        now = datetime.now(UTC)
        auth_fail_ts = (now - timedelta(hours=2)).isoformat()
        resumed_ts = (now - timedelta(hours=1)).isoformat()

        conn.executemany(
            'INSERT INTO account_events '
            '(account_name, event_type, project_id, run_id, details, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            [
                ('test-acc', 'auth_failed', 'proj', 'r1', None, auth_fail_ts),
                ('test-acc', 'resumed', 'proj', 'r1', None, resumed_ts),
            ],
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as aconn:
            aconn.row_factory = aiosqlite.Row
            result = await get_cost_by_account(aconn)

        assert result['test-acc']['status'] == 'active'

    @pytest.mark.asyncio
    async def test_auth_failed_after_resumed_recaptures(self, tmp_path):
        """An older cap_hit/resumed pair followed by a fresh auth_failed must
        flip back to 'capped'. Mirrors the production timeline for max-e/max-f
        on 2026-05-01: stale cap_hit/resumed pair from a prior day, then a
        new HTTP 429 → auth_failed today.
        """
        db_path = tmp_path / 'auth_after_resumed.db'
        conn = sqlite3.connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)

        now = datetime.now(UTC)
        cap_ts = (now - timedelta(days=2)).isoformat()
        resumed_ts = (now - timedelta(days=2, hours=-1)).isoformat()
        auth_fail_ts = (now - timedelta(hours=1)).isoformat()
        details = (
            '{"reason": "HTTP 429: You\'re out of extra usage '
            '\\u00b7 resets in 2h"}'
        )

        conn.executemany(
            'INSERT INTO account_events '
            '(account_name, event_type, project_id, run_id, details, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            [
                ('test-acc', 'cap_hit', 'proj', 'r1', None, cap_ts),
                ('test-acc', 'resumed', 'proj', 'r1', None, resumed_ts),
                ('test-acc', 'auth_failed', 'proj', 'r1', details, auth_fail_ts),
            ],
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as aconn:
            aconn.row_factory = aiosqlite.Row
            result = await get_cost_by_account(aconn)

        ta = result['test-acc']
        assert ta['status'] == 'capped'
        # resets_at comes from the auth_failed (newer) event's reason text.
        parsed = datetime.fromisoformat(ta['resets_at'])
        delta_s = abs((parsed - (now + timedelta(hours=2))).total_seconds())
        assert delta_s < 60


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
        assert len(df) == 7
        zero_days = [e for e in df if e['total'] == 0.0]
        non_zero_days = [e for e in df if e['total'] > 0]
        # Fixture: all dark_factory invocations are within ~2h of 'now'.
        # In a 7-day window only today has spend (possibly yesterday near
        # UTC midnight). Hard-coded from fixture knowledge, not derived
        # from the actual result.
        assert len(non_zero_days) >= 1, "fixture must produce at least one day with spend"
        assert len(non_zero_days) <= 2, "fixture spend spans at most 2 calendar days"
        assert len(zero_days) >= 5, "gap-filling must produce at least 5 zero-value days"

    @pytest.mark.asyncio
    async def test_trend_days_align_with_cutoff(self, tmp_path):
        """First and last day in the trend must derive from the same `now`.

        Regression guard for the clock-skew race: _cutoff(days) calls
        datetime.now(UTC) internally, then get_cost_trend calls datetime.now(UTC)
        again to build all_days. If these two calls straddle UTC midnight the
        day list and the SQL cutoff window are misaligned (off-by-one).

        The fix captures `now` once and derives both `since` and `all_days` from
        it. This test validates the invariant: with days=5, the result must
        cover exactly 5 days; the first day must be
        `(now - timedelta(days=4)).strftime('%Y-%m-%d')` and the last must be
        today; and an invocation inserted at exactly the first-day boundary must
        appear in the output.
        """
        db_path = tmp_path / 'trend_align.db'
        conn = __import__('sqlite3').connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)
        now = datetime.now(UTC)

        days = 5
        # Place an invocation exactly at the start of the first day in the window
        # (days-1 days before today's midnight equivalent).
        first_day = (now - timedelta(days=days - 1)).strftime('%Y-%m-%d')
        last_day = now.strftime('%Y-%m-%d')
        # Use noon of the first day to avoid midnight ambiguity
        boundary_ts = (
            datetime.now(UTC).replace(
                hour=12, minute=0, second=0, microsecond=0
            ) - timedelta(days=days - 1)
        ).isoformat()

        conn.execute(
            'INSERT INTO invocations '
            '(run_id, task_id, project_id, account_name, model, role, '
            ' cost_usd, capped, started_at, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            ('r1', 't1', 'proj', 'acc', 'claude-sonnet-4-5', 'implementer',
             0.75, 0, boundary_ts, boundary_ts),
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as aconn:
            aconn.row_factory = aiosqlite.Row
            result = await get_cost_trend(aconn, days=days)

        assert 'proj' in result
        proj_days = result['proj']

        # Exactly `days` entries
        assert len(proj_days) == days, (
            f"expected {days} day entries, got {len(proj_days)}: "
            f"{[e['day'] for e in proj_days]}"
        )

        # Chronologically ordered
        day_strs = [e['day'] for e in proj_days]
        assert day_strs == sorted(day_strs)

        # First day aligns with (now - timedelta(days=days-1))
        assert day_strs[0] == first_day, (
            f"expected first day={first_day!r}, got {day_strs[0]!r}"
        )
        # Last day is today
        assert day_strs[-1] == last_day, (
            f"expected last day={last_day!r}, got {day_strs[-1]!r}"
        )

        # The boundary invocation must appear in the correct day slot
        day_totals = {e['day']: e['total'] for e in proj_days}
        assert day_totals[first_day] == pytest.approx(0.75, abs=1e-6), (
            f"boundary invocation missing from first day {first_day!r}"
        )


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

        Three invocations with task_id=NULL but DIFFERENT model/role values
        (claude-sonnet-4-5/orchestrator × 2, claude-opus-4-5/debugger × 1)
        must all collapse into one task dict (task_id=None), with costs
        accumulated and all three invocation rows preserved in the detail list.

        This tests that heterogeneous model/role attributes don't accidentally
        produce separate task entries — the grouping key is task_id only.
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
        # Insert THREE invocations with NULL task_id (run-level billing):
        #   inv-1: sonnet / orchestrator  → 0.25
        #   inv-2: sonnet / orchestrator  → 0.15
        #   inv-3: opus   / debugger      → 0.10  (different model AND role)
        conn.executemany(
            'INSERT INTO invocations '
            '(run_id, task_id, project_id, account_name, model, role, '
            ' cost_usd, capped, started_at, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            [
                ('r1', None, 'proj', 'acc', 'claude-sonnet-4-5', 'orchestrator',
                 0.25, 0,
                 (now - timedelta(minutes=50)).isoformat(),
                 (now - timedelta(minutes=40)).isoformat()),
                ('r1', None, 'proj', 'acc', 'claude-sonnet-4-5', 'orchestrator',
                 0.15, 0,
                 (now - timedelta(minutes=30)).isoformat(),
                 (now - timedelta(minutes=20)).isoformat()),
                ('r1', None, 'proj', 'acc', 'claude-opus-4-5', 'debugger',
                 0.10, 0,
                 (now - timedelta(minutes=15)).isoformat(),
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
        assert run['total_cost'] == pytest.approx(0.50, abs=1e-6)

        # All three NULLs must be grouped into exactly ONE task entry
        assert len(run['tasks']) == 1
        t = run['tasks'][0]
        assert t['task_id'] is None
        assert t['title'] is None
        assert t['cost'] == pytest.approx(0.50, abs=1e-6)
        # All three invocation rows must be preserved in the detail list
        assert len(t['invocations']) == 3

        # Verify each invocation's model/role/cost_usd are individually correct.
        # Sort by cost_usd descending: 0.25, 0.15, 0.10.
        invs = sorted(t['invocations'], key=lambda x: x['cost_usd'], reverse=True)
        assert invs[0]['cost_usd'] == pytest.approx(0.25, abs=1e-6)
        assert invs[0]['model'] == 'claude-sonnet-4-5'
        assert invs[0]['role'] == 'orchestrator'
        assert invs[1]['cost_usd'] == pytest.approx(0.15, abs=1e-6)
        assert invs[1]['model'] == 'claude-sonnet-4-5'
        assert invs[1]['role'] == 'orchestrator'
        assert invs[2]['cost_usd'] == pytest.approx(0.10, abs=1e-6)
        assert invs[2]['model'] == 'claude-opus-4-5'
        assert invs[2]['role'] == 'debugger'


# ---------------------------------------------------------------------------
# Tests: days parameter filtering (cross-function parametric)
# ---------------------------------------------------------------------------

class TestDaysParameter:
    """Parametric coverage of the _cutoff(days) window filter across query functions.

    Each test case exercises a different query function with days=1, days=7, and
    days=30 to verify that the window correctly excludes/includes a 20-day-old
    invocation that falls outside the narrow window but inside the wide window.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("fn_name", [
        "summary",
        "by_project",
        "by_account",
        "by_role",
        "account_events",
    ])
    async def test_days_controls_lookback_window(self, tmp_path, fn_name):
        """days parameter controls look-back for each query function.

        DB setup: one invocation at 2h ago (cost=0.50) and one at 20d ago
        (cost=1.50), plus one cap_hit event at each offset.

        - days=1 and days=7 see only the 2h-ago record (narrow window).
        - days=30 sees both records (wide window).
        """
        db_path = tmp_path / f'days_{fn_name}.db'
        conn = __import__('sqlite3').connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)
        now = datetime.now(UTC)
        recent_ts = (now - timedelta(hours=2)).isoformat()
        old_ts = (now - timedelta(days=20)).isoformat()

        conn.executemany(
            'INSERT INTO invocations '
            '(run_id, task_id, project_id, account_name, model, role, '
            ' cost_usd, capped, started_at, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            [
                ('r1', 't1', 'proj', 'acc', 'claude-opus-4-5', 'implementer',
                 0.50, 0,
                 (now - timedelta(hours=2, minutes=5)).isoformat(),
                 recent_ts),
                ('r2', 't2', 'proj', 'acc', 'claude-opus-4-5', 'implementer',
                 1.50, 0,
                 (now - timedelta(days=20, hours=1)).isoformat(),
                 old_ts),
            ],
        )
        conn.executemany(
            'INSERT INTO account_events '
            '(account_name, event_type, project_id, run_id, details, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            [
                ('acc', 'cap_hit', 'proj', 'r1', None, recent_ts),
                ('acc', 'cap_hit', 'proj', 'r2', None, old_ts),
            ],
        )
        conn.commit()
        conn.close()

        fn_map = {
            'summary': get_cost_summary,
            'by_project': get_cost_by_project,
            'by_account': get_cost_by_account,
            'by_role': get_cost_by_role,
            'account_events': get_account_events,
        }
        fn = fn_map[fn_name]

        async with aiosqlite.connect(str(db_path)) as aconn:
            aconn.row_factory = aiosqlite.Row
            narrow1 = await fn(aconn, days=1)
            narrow7 = await fn(aconn, days=7)
            wide30 = await fn(aconn, days=30)

        def _metric(result, name):
            """Extract the key numeric metric for each function's result."""
            if name == 'summary':
                return result.get('proj', {}).get('total_spend', 0.0)
            if name == 'by_project':
                return sum(e['total'] for e in result.get('proj', []))
            if name == 'by_account':
                return result.get('acc', {}).get('invocations', 0)
            if name == 'by_role':
                proj = result.get('proj', {})
                return sum(
                    v for role_data in proj.values() for v in role_data.values()
                )
            if name == 'account_events':
                return len(result)
            return 0

        # Integer metrics (invocation/event counts)
        if fn_name in ('by_account', 'account_events'):
            narrow_expected: int | float = 1
            wide_expected: int | float = 2
        else:
            narrow_expected = 0.50
            wide_expected = 2.00

        assert _metric(narrow1, fn_name) == pytest.approx(narrow_expected, abs=1e-6), (
            f"{fn_name}: days=1 narrow expected {narrow_expected}, "
            f"got {_metric(narrow1, fn_name)}"
        )
        assert _metric(narrow7, fn_name) == pytest.approx(narrow_expected, abs=1e-6), (
            f"{fn_name}: days=7 narrow expected {narrow_expected}, "
            f"got {_metric(narrow7, fn_name)}"
        )
        assert _metric(wide30, fn_name) == pytest.approx(wide_expected, abs=1e-6), (
            f"{fn_name}: days=30 wide expected {wide_expected}, "
            f"got {_metric(wide30, fn_name)}"
        )


# ---------------------------------------------------------------------------
# Tests: Schema index validation
# ---------------------------------------------------------------------------

class TestSchema:
    def test_completed_at_index_exists(self, tmp_path):
        """COSTS_SCHEMA must create idx_inv_completed_at on invocations.

        Every query in costs.py filters WHERE completed_at >= ?, so the
        index is critical for performance. This test verifies the test-schema
        constant has the index DDL.
        """
        db_path = tmp_path / 'schema_check.db'
        conn = sqlite3.connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)
        conn.commit()

        # PRAGMA index_list returns rows: (seq, name, unique, origin, partial)
        rows = conn.execute("PRAGMA index_list('invocations')").fetchall()
        index_names = [row[1] for row in rows]
        conn.close()

        assert 'idx_inv_completed_at' in index_names, (
            f"idx_inv_completed_at missing from COSTS_SCHEMA indexes: {index_names}"
        )

    def test_cost_store_schema_has_completed_at_index(self, tmp_path):
        """_SCHEMA in shared.cost_store must create idx_inv_completed_at.

        The production schema must mirror the test schema for this performance-
        critical index. This test creates a fresh DB using _SCHEMA (production)
        and asserts the index is present.
        """
        import sys  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        # shared is not a dashboard venv dependency; locate it via relative path
        # parents[2] = worktree root (e.g. .worktrees/317)
        shared_src = str(
            Path(__file__).resolve().parents[2] / 'shared' / 'src'
        )
        sys.path.insert(0, shared_src)
        try:
            from shared.cost_store import (  # pyright: ignore[reportMissingImports]
                _SCHEMA as PROD_SCHEMA,  # noqa: PLC0415
            )
        finally:
            sys.path.remove(shared_src)

        db_path = tmp_path / 'prod_schema_check.db'
        conn = sqlite3.connect(str(db_path))
        conn.executescript(PROD_SCHEMA)
        conn.commit()

        rows = conn.execute("PRAGMA index_list('invocations')").fetchall()
        index_names = [row[1] for row in rows]
        conn.close()

        assert 'idx_inv_completed_at' in index_names, (
            f"idx_inv_completed_at missing from cost_store._SCHEMA indexes: {index_names}"
        )


# ---------------------------------------------------------------------------
# Tests: get_account_events limit parameter
# ---------------------------------------------------------------------------

class TestAccountEventsLimit:
    @pytest.mark.asyncio
    async def test_limit_restricts_results(self, tmp_path):
        """limit=3 returns only the 3 most-recent events (DESC order preserved).

        Insert 10 account events with known timestamps, call get_account_events
        with limit=3, assert exactly 3 rows returned and they are the 3 most
        recent (highest created_at values).
        """
        db_path = tmp_path / 'ae_limit.db'
        conn = __import__('sqlite3').connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)
        now = datetime.now(UTC)

        # Insert 10 events at 10 different timestamps (10m ago, 9m ago, ... 1m ago)
        events = [
            ('acc', 'cap_hit', 'proj', 'r1', None,
             (now - timedelta(minutes=10 - i)).isoformat())
            for i in range(10)
        ]
        conn.executemany(
            'INSERT INTO account_events '
            '(account_name, event_type, project_id, run_id, details, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            events,
        )
        conn.commit()
        conn.close()

        # Expected: the 3 most recent = minutes 0, 1, 2 ago (DESC)
        expected_ts = sorted(
            [(now - timedelta(minutes=10 - i)).isoformat() for i in range(10)],
            reverse=True,
        )[:3]

        async with aiosqlite.connect(str(db_path)) as aconn:
            aconn.row_factory = aiosqlite.Row
            result = await get_account_events(aconn, limit=3)

        assert len(result) == 3, f"expected 3 rows, got {len(result)}"
        result_ts = [e['created_at'] for e in result]
        # Verify DESC order is preserved
        assert result_ts == sorted(result_ts, reverse=True)
        # Verify these are the 3 most recent
        assert result_ts == expected_ts

    @pytest.mark.asyncio
    async def test_default_limit_returns_all_when_under_threshold(self, tmp_path):
        """Default limit (200) returns all rows when count < 200.

        Insert 5 events, call without explicit limit, assert all 5 returned.
        """
        db_path = tmp_path / 'ae_default_limit.db'
        conn = __import__('sqlite3').connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)
        now = datetime.now(UTC)

        events = [
            ('acc', 'cap_hit', 'proj', 'r1', None,
             (now - timedelta(minutes=i)).isoformat())
            for i in range(5)
        ]
        conn.executemany(
            'INSERT INTO account_events '
            '(account_name, event_type, project_id, run_id, details, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            events,
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as aconn:
            aconn.row_factory = aiosqlite.Row
            result = await get_account_events(aconn)

        assert len(result) == 5, f"expected 5 rows with default limit, got {len(result)}"


# ---------------------------------------------------------------------------
# Tests: get_run_cost_breakdown limit parameter
# ---------------------------------------------------------------------------

class TestRunCostBreakdownLimit:
    @pytest.mark.asyncio
    async def test_limit_restricts_invocation_rows(self, tmp_path):
        """limit=3 caps invocation rows at the SQL level to at most 3.

        Insert 6 invocations across 2 runs/tasks, call get_run_cost_breakdown
        with limit=3, assert the assembled result reflects at most 3 invocation
        rows total (which may result in fewer runs/tasks assembled).
        """
        db_path = tmp_path / 'rcb_limit.db'
        conn = __import__('sqlite3').connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)
        now = datetime.now(UTC)

        conn.execute(
            'INSERT INTO runs (run_id, project_id, started_at, completed_at) '
            'VALUES (?, ?, ?, ?)',
            ('r1', 'proj', (now - timedelta(hours=2)).isoformat(), now.isoformat()),
        )
        conn.execute(
            'INSERT INTO task_results (run_id, task_id, project_id, title, outcome, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            ('r1', 't1', 'proj', 'Task One', 'done', now.isoformat()),
        )
        # Insert 6 invocations for the same run/task
        invocations = [
            ('r1', 't1', 'proj', 'acc', 'claude-opus-4-5', 'implementer',
             0.10, 0,
             (now - timedelta(minutes=6 - i)).isoformat(),
             (now - timedelta(minutes=5 - i)).isoformat())
            for i in range(6)
        ]
        conn.executemany(
            'INSERT INTO invocations '
            '(run_id, task_id, project_id, account_name, model, role, '
            ' cost_usd, capped, started_at, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            invocations,
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as aconn:
            aconn.row_factory = aiosqlite.Row
            result = await get_run_cost_breakdown(aconn, limit=3)

        # Total invocations across all tasks in all runs must be <= 3
        total_invocations = sum(
            len(task['invocations'])
            for run in result
            for task in run['tasks']
        )
        assert total_invocations <= 3, (
            f"expected <= 3 invocation rows with limit=3, got {total_invocations}"
        )

    @pytest.mark.asyncio
    async def test_default_limit_returns_all_when_under_threshold(self, tmp_path):
        """Default limit (500) returns all rows when count < 500.

        Insert 4 invocations, call without explicit limit, assert all 4 returned.
        """
        db_path = tmp_path / 'rcb_default_limit.db'
        conn = __import__('sqlite3').connect(str(db_path))
        conn.executescript(COSTS_SCHEMA)
        now = datetime.now(UTC)

        conn.execute(
            'INSERT INTO runs (run_id, project_id, started_at, completed_at) '
            'VALUES (?, ?, ?, ?)',
            ('r1', 'proj', (now - timedelta(hours=1)).isoformat(), now.isoformat()),
        )
        invocations = [
            ('r1', f't{i}', 'proj', 'acc', 'claude-opus-4-5', 'implementer',
             0.10, 0,
             (now - timedelta(minutes=i + 1)).isoformat(),
             (now - timedelta(minutes=i)).isoformat())
            for i in range(4)
        ]
        conn.executemany(
            'INSERT INTO invocations '
            '(run_id, task_id, project_id, account_name, model, role, '
            ' cost_usd, capped, started_at, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            invocations,
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as aconn:
            aconn.row_factory = aiosqlite.Row
            result = await get_run_cost_breakdown(aconn)

        total_invocations = sum(
            len(task['invocations'])
            for run in result
            for task in run['tasks']
        )
        assert total_invocations == 4, (
            f"expected 4 invocations with default limit, got {total_invocations}"
        )


# ===========================================================================
# Multi-DB aggregation tests
# ===========================================================================

# Minimal schema — invocations + account_events only (no runs/task_results).
# Matches DBs like dark-factory and reify that never used RunStore.
MINIMAL_SCHEMA = """\
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

from dashboard.data.costs import (  # noqa: E402
    aggregate_account_events,
    aggregate_cost_by_account,
    aggregate_cost_by_project,
    aggregate_cost_by_role,
    aggregate_cost_summary,
    aggregate_cost_trend,
    aggregate_run_cost_breakdown,
)


@pytest.fixture()
def two_project_dbs(tmp_path):
    """Two separate SQLite DBs with different project_ids and a shared account."""
    now = datetime.now(UTC)

    # --- DB A: dark_factory (full schema with task_results) ---
    db_a = tmp_path / 'a.db'
    conn = sqlite3.connect(str(db_a))
    conn.executescript(COSTS_SCHEMA)
    conn.execute(
        'INSERT INTO runs (run_id, project_id, started_at) VALUES (?, ?, ?)',
        ('run-a1', 'dark_factory', now.isoformat()),
    )
    conn.execute(
        'INSERT INTO task_results (run_id, task_id, project_id, title, outcome) '
        'VALUES (?, ?, ?, ?, ?)',
        ('run-a1', '10', 'dark_factory', 'Task A', 'done'),
    )
    conn.executemany(
        'INSERT INTO invocations '
        '(run_id, task_id, project_id, account_name, model, role, '
        ' cost_usd, capped, started_at, completed_at) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        [
            ('run-a1', '10', 'dark_factory', 'max-a', 'opus', 'implementer',
             2.0, 0, (now - timedelta(hours=1)).isoformat(), now.isoformat()),
            ('run-a1', '10', 'dark_factory', 'max-a', 'sonnet', 'reviewer',
             0.5, 0, (now - timedelta(minutes=30)).isoformat(), now.isoformat()),
        ],
    )
    conn.execute(
        'INSERT INTO account_events '
        '(account_name, event_type, project_id, run_id, details, created_at) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        ('max-a', 'cap_hit', 'dark_factory', 'run-a1', None, now.isoformat()),
    )
    conn.commit()
    conn.close()

    # --- DB B: reify (minimal schema — no task_results) ---
    db_b = tmp_path / 'b.db'
    conn = sqlite3.connect(str(db_b))
    conn.executescript(MINIMAL_SCHEMA)
    conn.executemany(
        'INSERT INTO invocations '
        '(run_id, task_id, project_id, account_name, model, role, '
        ' cost_usd, capped, started_at, completed_at) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        [
            ('run-b1', '20', 'reify', 'max-a', 'opus', 'implementer',
             3.0, 0, (now - timedelta(hours=2)).isoformat(), now.isoformat()),
            ('run-b1', '21', 'reify', 'max-b', 'sonnet', 'reviewer',
             1.0, 0, (now - timedelta(hours=1)).isoformat(), now.isoformat()),
        ],
    )
    conn.execute(
        'INSERT INTO account_events '
        '(account_name, event_type, project_id, run_id, details, created_at) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        ('max-a', 'resumed', 'reify', 'run-b1', None,
         (now + timedelta(minutes=5)).isoformat()),
    )
    conn.commit()
    conn.close()

    return db_a, db_b


@pytest.fixture()
async def two_conns(two_project_dbs):
    db_a, db_b = two_project_dbs
    async with aiosqlite.connect(str(db_a)) as a, aiosqlite.connect(str(db_b)) as b:
        a.row_factory = aiosqlite.Row
        b.row_factory = aiosqlite.Row
        yield [a, b]


class TestAggregateCostSummary:
    @pytest.mark.asyncio
    async def test_merges_two_projects(self, two_conns):
        result = await aggregate_cost_summary(two_conns, days=30)
        assert 'dark_factory' in result
        assert 'reify' in result
        assert result['dark_factory']['total_spend'] == pytest.approx(2.5)
        assert result['reify']['total_spend'] == pytest.approx(4.0)

    @pytest.mark.asyncio
    async def test_empty_list(self):
        result = await aggregate_cost_summary([], days=7)
        assert result == {}

    @pytest.mark.asyncio
    async def test_none_connections(self):
        result = await aggregate_cost_summary([None, None], days=7)
        assert result == {}


class TestAggregateCostByAccount:
    @pytest.mark.asyncio
    async def test_shared_account_summed(self, two_conns):
        result = await aggregate_cost_by_account(two_conns, days=30)
        # max-a appears in both DBs — spend should be summed
        assert 'max-a' in result
        assert result['max-a']['spend'] == pytest.approx(2.0 + 0.5 + 3.0)
        assert result['max-a']['invocations'] == 3

    @pytest.mark.asyncio
    async def test_cross_db_resumed_clears_cap(self, two_conns):
        """A resumed event in one DB clears a cap observed in another.

        The fixture has max-a cap_hit in DB A at *now* and resumed in DB B at
        *now + 5min*. Accounts are global (same OAuth token), so the later
        resumed clears the cap even though no single DB sees both events. The
        aggregator must compute global ordering, not latch on per-DB status.
        Regression: prior code latched 'capped' on any source, masking the
        globally-newer resumed.
        """
        result = await aggregate_cost_by_account(two_conns, days=30)
        assert result['max-a']['status'] == 'active'
        assert result['max-a']['cap_events'] == 1
        # resets_at is meaningless once we know the cap was resumed.
        assert result['max-a']['resets_at'] is None

    @pytest.mark.asyncio
    async def test_stale_db_cap_without_resumed_wins(self, tmp_path):
        """If the globally-newest cap_hit has no later resumed anywhere,
        status is capped — even if another DB shows an older resumed.

        Setup:
          - DB A: max-a resumed at T1
          - DB B: max-a cap_hit at T2 (T2 > T1), no resumed
        Expected: capped (the T2 cap is newer than any resumed).
        """
        now = datetime.now(UTC)
        t1 = (now - timedelta(hours=3)).isoformat()
        t2 = (now - timedelta(hours=1)).isoformat()

        db_a = tmp_path / 'stale_a.db'
        db_b = tmp_path / 'stale_b.db'
        for p in (db_a, db_b):
            conn = sqlite3.connect(str(p))
            conn.executescript(MINIMAL_SCHEMA)
            conn.close()

        conn = sqlite3.connect(str(db_a))
        conn.execute(
            'INSERT INTO account_events '
            '(account_name, event_type, project_id, run_id, details, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            ('max-a', 'resumed', 'pa', 'r1', None, t1),
        )
        conn.execute(
            'INSERT INTO invocations '
            '(run_id, task_id, project_id, account_name, model, role, '
            ' cost_usd, capped, started_at, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            ('r1', 't1', 'pa', 'max-a', 'opus', 'implementer',
             0.1, 0, t1, t1),
        )
        conn.commit()
        conn.close()

        conn = sqlite3.connect(str(db_b))
        conn.execute(
            'INSERT INTO account_events '
            '(account_name, event_type, project_id, run_id, details, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            ('max-a', 'cap_hit', 'pb', 'r2', '{"reason": "limit"}', t2),
        )
        conn.execute(
            'INSERT INTO invocations '
            '(run_id, task_id, project_id, account_name, model, role, '
            ' cost_usd, capped, started_at, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            ('r2', 't2', 'pb', 'max-a', 'opus', 'implementer',
             0.1, 0, t2, t2),
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_a)) as a, aiosqlite.connect(str(db_b)) as b:
            a.row_factory = aiosqlite.Row
            b.row_factory = aiosqlite.Row
            result = await aggregate_cost_by_account([a, b], days=7)

        assert result['max-a']['status'] == 'capped'
        assert result['max-a']['last_cap'] == t2

    @pytest.mark.asyncio
    async def test_resets_at_travels_with_winning_last_cap(self, tmp_path):
        """When one DB supplies the globally-newest cap_hit, its resets_at
        (and only its) attaches to the merged row.
        """
        now = datetime.now(UTC)
        t_old = (now - timedelta(hours=3)).isoformat()
        t_new = (now - timedelta(hours=1)).isoformat()
        new_resets_at = (now + timedelta(hours=2)).isoformat()

        db_a = tmp_path / 'resets_a.db'
        db_b = tmp_path / 'resets_b.db'
        for p in (db_a, db_b):
            conn = sqlite3.connect(str(p))
            conn.executescript(MINIMAL_SCHEMA)
            conn.close()

        old_details = '{"reason": "old limit", "resets_at": "' \
            + (now - timedelta(hours=1)).isoformat() + '"}'
        new_details = '{"reason": "new limit", "resets_at": "' + new_resets_at + '"}'

        for p, cap_ts, details in [(db_a, t_old, old_details), (db_b, t_new, new_details)]:
            conn = sqlite3.connect(str(p))
            conn.execute(
                'INSERT INTO account_events '
                '(account_name, event_type, project_id, run_id, details, created_at) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                ('max-a', 'cap_hit', 'p', f'r-{cap_ts}', details, cap_ts),
            )
            conn.execute(
                'INSERT INTO invocations '
                '(run_id, task_id, project_id, account_name, model, role, '
                ' cost_usd, capped, started_at, completed_at) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (f'r-{cap_ts}', 't', 'p', 'max-a', 'opus', 'implementer',
                 0.1, 0, cap_ts, cap_ts),
            )
            conn.commit()
            conn.close()

        async with aiosqlite.connect(str(db_a)) as a, aiosqlite.connect(str(db_b)) as b:
            a.row_factory = aiosqlite.Row
            b.row_factory = aiosqlite.Row
            result = await aggregate_cost_by_account([a, b], days=7)

        assert result['max-a']['status'] == 'capped'
        assert result['max-a']['last_cap'] == t_new
        assert result['max-a']['resets_at'] == new_resets_at

    @pytest.mark.asyncio
    async def test_unique_account(self, two_conns):
        result = await aggregate_cost_by_account(two_conns, days=30)
        assert 'max-b' in result
        assert result['max-b']['spend'] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_cross_db_auth_failed_wins(self, tmp_path):
        """An auth_failed event in one DB makes the account 'capped' globally
        even when the other DB has only a stale cap_hit/resumed pair.

        Mirrors the production timeline that motivated the auth_failed→
        unavailable mapping: max-e/max-f had old cap_hit/resumed pairs in
        the reify DB (showing 'active') and only auth_failed events in
        know-live (with no later auth_resumed). The aggregator must report
        capped because the most recent unavailability event globally is the
        auth_failed.
        """
        now = datetime.now(UTC)
        old_cap = (now - timedelta(days=2)).isoformat()
        old_resumed = (now - timedelta(days=2, hours=-1)).isoformat()
        new_auth_fail = (now - timedelta(hours=1)).isoformat()

        db_a = tmp_path / 'cross_auth_a.db'
        db_b = tmp_path / 'cross_auth_b.db'
        for p in (db_a, db_b):
            conn = sqlite3.connect(str(p))
            conn.executescript(MINIMAL_SCHEMA)
            conn.close()

        # DB A: stale cap_hit + later resumed (looks 'active' in isolation)
        conn = sqlite3.connect(str(db_a))
        conn.executemany(
            'INSERT INTO account_events '
            '(account_name, event_type, project_id, run_id, details, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            [
                ('max-a', 'cap_hit', 'pa', 'r1',
                 '{"reason": "old"}', old_cap),
                ('max-a', 'resumed', 'pa', 'r1', None, old_resumed),
            ],
        )
        conn.execute(
            'INSERT INTO invocations '
            '(run_id, task_id, project_id, account_name, model, role, '
            ' cost_usd, capped, started_at, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            ('r1', 't1', 'pa', 'max-a', 'opus', 'implementer',
             0.1, 0, old_cap, old_cap),
        )
        conn.commit()
        conn.close()

        # DB B: fresh auth_failed with no recovery
        conn = sqlite3.connect(str(db_b))
        auth_details = (
            '{"reason": "HTTP 429: You\'re out of extra usage '
            '\\u00b7 resets in 2h"}'
        )
        conn.execute(
            'INSERT INTO account_events '
            '(account_name, event_type, project_id, run_id, details, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            ('max-a', 'auth_failed', 'pb', 'r2', auth_details, new_auth_fail),
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_a)) as a, aiosqlite.connect(str(db_b)) as b:
            a.row_factory = aiosqlite.Row
            b.row_factory = aiosqlite.Row
            result = await aggregate_cost_by_account([a, b], days=7)

        assert result['max-a']['status'] == 'capped'
        assert result['max-a']['last_auth_fail'] == new_auth_fail
        # Reason fallback parses "resets in 2h" → ~2h ahead.
        parsed = datetime.fromisoformat(result['max-a']['resets_at'])
        delta_s = abs((parsed - (now + timedelta(hours=2))).total_seconds())
        assert delta_s < 60


class TestAggregateAccountEvents:
    @pytest.mark.asyncio
    async def test_merged_sorted_and_limited(self, two_conns):
        result = await aggregate_account_events(two_conns, days=30, limit=1)
        assert len(result) == 1
        # Most recent event should come first (resumed from DB B)
        assert result[0]['event_type'] == 'resumed'

    @pytest.mark.asyncio
    async def test_all_events_present(self, two_conns):
        result = await aggregate_account_events(two_conns, days=30)
        assert len(result) == 2


class TestAggregateRunCostBreakdown:
    @pytest.mark.asyncio
    async def test_runs_from_both_dbs(self, two_conns):
        result = await aggregate_run_cost_breakdown(two_conns, days=30)
        run_ids = {r['run_id'] for r in result}
        assert 'run-a1' in run_ids
        assert 'run-b1' in run_ids

    @pytest.mark.asyncio
    async def test_missing_task_results_table(self, two_conns):
        """DB B lacks task_results — query should still succeed with title=None."""
        result = await aggregate_run_cost_breakdown(two_conns, days=30)
        reify_runs = [r for r in result if r['project_id'] == 'reify']
        assert len(reify_runs) == 1
        for task in reify_runs[0]['tasks']:
            assert task['title'] is None


class TestAggregateCostByProject:
    @pytest.mark.asyncio
    async def test_both_projects(self, two_conns):
        result = await aggregate_cost_by_project(two_conns, days=30)
        assert 'dark_factory' in result
        assert 'reify' in result


class TestAggregateCostByRole:
    @pytest.mark.asyncio
    async def test_roles_by_project(self, two_conns):
        result = await aggregate_cost_by_role(two_conns, days=30)
        assert 'implementer' in result['dark_factory']
        assert 'reviewer' in result['reify']


class TestAggregateCostTrend:
    @pytest.mark.asyncio
    async def test_both_projects_have_series(self, two_conns):
        result = await aggregate_cost_trend(two_conns, days=7)
        assert 'dark_factory' in result
        assert 'reify' in result
        # Each project should have 7 days of data (gap-filled)
        assert len(result['dark_factory']) == 7
        assert len(result['reify']) == 7
