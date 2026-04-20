"""Tests for dashboard.data.performance — orchestrator performance queries."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite
import pytest

from dashboard.data.performance import (
    aggregate_completion_paths,
    aggregate_escalation_rates,
    aggregate_loop_histograms,
    aggregate_time_centiles,
    get_completion_paths,
    get_escalation_rates,
    get_loop_histograms,
    get_time_centiles,
)

# ---------------------------------------------------------------------------
# Schema (matches orchestrator/src/orchestrator/run_store.py)
# ---------------------------------------------------------------------------

RUNS_SCHEMA = """\
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def runs_db(tmp_path):
    """Populated runs.db with diverse task results across two projects."""
    db_path = tmp_path / 'runs.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(RUNS_SCHEMA)

    now = datetime.now(UTC)

    # Run for dark_factory
    conn.execute(
        'INSERT INTO runs (run_id, project_id, prd_path, started_at, completed_at, '
        ' total_tasks, completed, blocked) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        ('run-001', 'dark_factory', '/prd.md', (now - timedelta(hours=2)).isoformat(),
         (now - timedelta(hours=1)).isoformat(), 6, 5, 1),
    )

    # Task results for dark_factory
    tasks = [
        # one-pass: outcome=done, review_cycles=0, steward=0
        ('run-001', '101', 'dark_factory', 'Widget A', 'done', 0.5, 300_000,
         3, 1, 0, 0, 0.0, 0, (now - timedelta(hours=1, minutes=50)).isoformat()),
        # one-pass: another clean task
        ('run-001', '102', 'dark_factory', 'Widget B', 'done', 0.4, 200_000,
         2, 1, 1, 0, 0.0, 0, (now - timedelta(hours=1, minutes=40)).isoformat()),
        # multi-pass: review_cycles > 0, no steward
        ('run-001', '103', 'dark_factory', 'Refactor C', 'done', 0.8, 600_000,
         5, 3, 2, 2, 0.0, 0, (now - timedelta(hours=1, minutes=30)).isoformat()),
        # via-steward: steward_invocations > 0
        ('run-001', '104', 'dark_factory', 'Fix D', 'done', 1.0, 900_000,
         7, 2, 3, 1, 0.5, 2, (now - timedelta(hours=1, minutes=20)).isoformat()),
        # blocked: outcome=blocked
        ('run-001', '105', 'dark_factory', 'Feature E', 'blocked', 0.3, 150_000,
         2, 1, 1, 0, 0.0, 0, (now - timedelta(hours=1, minutes=10)).isoformat()),
        # via-interactive: outcome=done (will be matched by escalation file with level=1)
        ('run-001', '106', 'dark_factory', 'Auth F', 'done', 0.6, 450_000,
         4, 2, 1, 0, 0.0, 0, (now - timedelta(hours=1, minutes=5)).isoformat()),
    ]
    for t in tasks:
        conn.execute(
            'INSERT INTO task_results '
            '(run_id, task_id, project_id, title, outcome, cost_usd, duration_ms, '
            ' agent_invocations, execute_iterations, verify_attempts, review_cycles, '
            ' steward_cost_usd, steward_invocations, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', t,
        )

    # Run for reify project
    conn.execute(
        'INSERT INTO runs (run_id, project_id, prd_path, started_at, completed_at, '
        ' total_tasks, completed) VALUES (?, ?, ?, ?, ?, ?, ?)',
        ('run-002', 'reify', '/reify/prd.md', (now - timedelta(hours=3)).isoformat(),
         (now - timedelta(hours=2)).isoformat(), 2, 2),
    )
    conn.execute(
        'INSERT INTO task_results '
        '(run_id, task_id, project_id, title, outcome, cost_usd, duration_ms, '
        ' agent_invocations, execute_iterations, verify_attempts, review_cycles, '
        ' steward_cost_usd, steward_invocations, completed_at) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        ('run-002', '201', 'reify', 'Reify task', 'done', 0.3, 180_000,
         2, 1, 0, 0, 0.0, 0, (now - timedelta(hours=2, minutes=30)).isoformat()),
    )
    conn.execute(
        'INSERT INTO task_results '
        '(run_id, task_id, project_id, title, outcome, cost_usd, duration_ms, '
        ' agent_invocations, execute_iterations, verify_attempts, review_cycles, '
        ' steward_cost_usd, steward_invocations, completed_at) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        ('run-002', '202', 'reify', 'Reify task 2', 'done', 0.4, 240_000,
         3, 2, 1, 1, 0.0, 0, (now - timedelta(hours=2, minutes=20)).isoformat()),
    )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
def empty_runs_db(tmp_path):
    db_path = tmp_path / 'empty_runs.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(RUNS_SCHEMA)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
async def runs_conn(runs_db):
    async with aiosqlite.connect(str(runs_db)) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn


@pytest.fixture()
async def empty_runs_conn(empty_runs_db):
    async with aiosqlite.connect(str(empty_runs_db)) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn


@pytest.fixture()
def escalations_dir(tmp_path):
    """Escalation directory with sample files including a level-1 for task 106."""
    esc_dir = tmp_path / 'escalations'
    esc_dir.mkdir()

    # Level-0 resolved (steward handled)
    (esc_dir / 'esc-104-1.json').write_text(json.dumps({
        'id': 'esc-104-1', 'task_id': '104', 'agent_role': 'debugger',
        'severity': 'blocking', 'category': 'infra_issue',
        'summary': 'Test infra down', 'level': 0,
        'status': 'resolved', 'resolution': 'Fixed by steward',
        'resolved_by': 'steward', 'resolution_turns': 5,
    }))

    # Level-1 resolved (interactive — matches task 106)
    (esc_dir / 'esc-106-1.json').write_text(json.dumps({
        'id': 'esc-106-1', 'task_id': '106', 'agent_role': 'implementer',
        'severity': 'blocking', 'category': 'scope_violation',
        'summary': 'Needs design decision', 'level': 1,
        'status': 'resolved', 'resolution': 'Human resolved it',
        'resolved_by': 'interactive', 'resolution_turns': 3,
    }))

    # Level-1 still pending (should not count)
    (esc_dir / 'esc-999-1.json').write_text(json.dumps({
        'id': 'esc-999-1', 'task_id': '999', 'agent_role': 'debugger',
        'severity': 'blocking', 'category': 'design_concern',
        'summary': 'Still pending', 'level': 1,
        'status': 'pending',
    }))

    return esc_dir


@pytest.fixture()
def empty_escalations_dir(tmp_path):
    esc_dir = tmp_path / 'empty_escalations'
    esc_dir.mkdir()
    return esc_dir


# ---------------------------------------------------------------------------
# Tests: get_completion_paths
# ---------------------------------------------------------------------------

class TestCompletionPaths:
    @pytest.mark.asyncio
    async def test_populated(self, runs_conn, escalations_dir):
        result = await get_completion_paths(runs_conn, escalations_dir)
        assert 'dark_factory' in result

        paths = {p['path']: p for p in result['dark_factory']}
        assert paths['one-pass']['count'] == 2
        assert paths['multi-pass']['count'] == 1
        assert paths['via-steward']['count'] == 1
        assert paths['via-interactive']['count'] == 1
        assert paths['blocked']['count'] == 1

    @pytest.mark.asyncio
    async def test_multi_project(self, runs_conn, escalations_dir):
        result = await get_completion_paths(runs_conn, escalations_dir)
        assert 'reify' in result
        paths = {p['path']: p for p in result['reify']}
        assert paths['one-pass']['count'] == 1
        assert paths['multi-pass']['count'] == 1

    @pytest.mark.asyncio
    async def test_empty_db(self, empty_runs_conn, empty_escalations_dir):
        result = await get_completion_paths(empty_runs_conn, empty_escalations_dir)
        assert result == {}

    @pytest.mark.asyncio
    async def test_missing_db(self, tmp_path, empty_escalations_dir):
        result = await get_completion_paths(None, empty_escalations_dir)
        assert result == {}

    @pytest.mark.asyncio
    async def test_percentages_sum(self, runs_conn, escalations_dir):
        result = await get_completion_paths(runs_conn, escalations_dir)
        for _project_id, paths in result.items():
            total_pct = sum(p['pct'] for p in paths)
            assert abs(total_pct - 100.0) < 1.0  # within rounding


# ---------------------------------------------------------------------------
# Tests: get_escalation_rates
# ---------------------------------------------------------------------------

class TestEscalationRates:
    @pytest.mark.asyncio
    async def test_populated(self, runs_conn, escalations_dir):
        result = await get_escalation_rates(runs_conn, escalations_dir)
        df = result['dark_factory']
        assert df['total_tasks'] == 6
        assert df['steward_count'] == 1  # task 104
        assert df['interactive_count'] == 1  # task 106
        assert df['steward_rate'] == pytest.approx(16.7, abs=0.1)
        assert df['interactive_rate'] == pytest.approx(16.7, abs=0.1)

    @pytest.mark.asyncio
    async def test_human_attention(self, runs_conn, escalations_dir):
        result = await get_escalation_rates(runs_conn, escalations_dir)
        attention = result['dark_factory']['human_attention']
        assert attention['significant'] == 1  # task 106: 3 turns >= 3
        assert attention['minimal'] == 0
        assert attention['zero'] == 0

    @pytest.mark.asyncio
    async def test_empty(self, empty_runs_conn, empty_escalations_dir):
        result = await get_escalation_rates(empty_runs_conn, empty_escalations_dir)
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: get_loop_histograms
# ---------------------------------------------------------------------------

class TestLoopHistograms:
    @pytest.mark.asyncio
    async def test_populated(self, runs_conn):
        result = await get_loop_histograms(runs_conn)
        df = result['dark_factory']

        # Outer loop (review_cycles): tasks 101(0), 102(0), 103(2), 104(1), 106(0)
        # = [3, 1, 1, 0] for bins 0,1,2,3+
        outer = df['outer']
        assert outer['labels'] == ['0', '1', '2', '3+']
        assert outer['values'] == [3, 1, 1, 0]

        # Inner loop (verify_attempts): tasks 101(0), 102(1), 103(2), 104(3), 106(1)
        # = [1, 2, 1, 1, 0, 0] for bins 0,1,2,3,4,5+
        inner = df['inner']
        assert inner['labels'] == ['0', '1', '2', '3', '4', '5+']
        assert inner['values'] == [1, 2, 1, 1, 0, 0]

    @pytest.mark.asyncio
    async def test_reify_project(self, runs_conn):
        result = await get_loop_histograms(runs_conn)
        reify = result['reify']
        # task 201: rc=0, va=0; task 202: rc=1, va=1
        assert reify['outer']['values'] == [1, 1, 0, 0]
        assert reify['inner']['values'] == [1, 1, 0, 0, 0, 0]

    @pytest.mark.asyncio
    async def test_empty(self, empty_runs_conn):
        result = await get_loop_histograms(empty_runs_conn)
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: get_time_centiles
# ---------------------------------------------------------------------------

class TestTimeCentiles:
    @pytest.mark.asyncio
    async def test_populated(self, runs_conn):
        result = await get_time_centiles(runs_conn)
        df = result['dark_factory']
        # 5 done tasks: 200k, 300k, 450k, 600k, 900k ms
        assert df['count'] == 5
        assert df['p50'] > 0
        assert df['p50'] <= df['p75'] <= df['p90'] <= df['p95']

    @pytest.mark.asyncio
    async def test_single_task(self, tmp_path):
        """Single task: all centiles should equal that task's duration."""
        db_path = tmp_path / 'single.db'
        conn = sqlite3.connect(str(db_path))
        conn.executescript(RUNS_SCHEMA)
        now = datetime.now(UTC)
        conn.execute(
            'INSERT INTO task_results '
            '(run_id, task_id, project_id, outcome, duration_ms, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            ('r1', 't1', 'proj', 'done', 42_000, now.isoformat()),
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as aconn:
            aconn.row_factory = aiosqlite.Row
            result = await get_time_centiles(aconn)
        assert result['proj']['p50'] == 42_000
        assert result['proj']['p95'] == 42_000
        assert result['proj']['count'] == 1

    @pytest.mark.asyncio
    async def test_empty(self, empty_runs_conn):
        result = await get_time_centiles(empty_runs_conn)
        assert result == {}

    @pytest.mark.asyncio
    async def test_missing_db(self, tmp_path):
        result = await get_time_centiles(None)
        assert result == {}

    @pytest.mark.asyncio
    async def test_single_db_parity_with_aggregate(self, runs_conn):
        """get_time_centiles and aggregate_time_centiles must return identical
        percentiles for the same single database.

        Both functions share _durations_by_project as the single SQL source.
        For a single DB the results must be exactly equal — this is the core
        invariant the shared helper exists to preserve.
        """
        single = await get_time_centiles(runs_conn, days=7)
        agg = await aggregate_time_centiles([runs_conn], days=7)
        assert single == agg


# ---------------------------------------------------------------------------
# Helpers for aggregate tests
# ---------------------------------------------------------------------------

def _make_runs_db(tmp_path, name: str, tasks: list[tuple]) -> Path:
    """Create a runs.db at tmp_path/name with provided task_results rows."""
    db_path = Path(tmp_path) / name
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(RUNS_SCHEMA)
    now = datetime.now(UTC)

    # Insert a minimal run row for each distinct project
    seen_runs: set = set()
    for t in tasks:
        run_id, project_id = t[0], t[2]
        if run_id not in seen_runs:
            conn.execute(
                'INSERT INTO runs (run_id, project_id, prd_path, started_at) '
                'VALUES (?, ?, ?, ?)',
                (run_id, project_id, '/prd.md',
                 (now - timedelta(hours=2)).isoformat()),
            )
            seen_runs.add(run_id)
        conn.execute(
            'INSERT INTO task_results '
            '(run_id, task_id, project_id, title, outcome, cost_usd, duration_ms, '
            ' agent_invocations, execute_iterations, verify_attempts, review_cycles, '
            ' steward_cost_usd, steward_invocations, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', t,
        )
    conn.commit()
    conn.close()
    return db_path


def _make_escalations_dir(tmp_path, name: str,
                          esc_list: list[dict]) -> Path:
    """Create an escalations directory with JSON files."""
    import json as _json
    esc_dir = Path(tmp_path) / name
    esc_dir.mkdir(parents=True, exist_ok=True)
    for i, esc in enumerate(esc_list):
        (esc_dir / f'esc-{i}.json').write_text(_json.dumps(esc))
    return esc_dir


# ---------------------------------------------------------------------------
# Tests: aggregate_completion_paths
# ---------------------------------------------------------------------------

class TestAggregateCompletionPaths:
    """Tests for aggregate_completion_paths across multiple DB connections."""

    @pytest.fixture()
    async def two_project_conns(self, tmp_path):
        """Two runs.dbs, both with dark_factory tasks (and DB2 also has reify)."""
        now = datetime.now(UTC)
        ts1 = (now - timedelta(minutes=30)).isoformat()
        ts2 = (now - timedelta(minutes=20)).isoformat()

        # DB1: dark_factory — 1 one-pass task, 1 via-steward
        db1_path = _make_runs_db(tmp_path / 'db1', 'runs.db', [
            ('run-a', 't1', 'dark_factory', 'T1', 'done',
             0.5, 300_000, 2, 1, 0, 0, 0.0, 0, ts1),
            ('run-a', 't2', 'dark_factory', 'T2', 'done',
             0.5, 400_000, 3, 1, 0, 0, 0.2, 2, ts2),
        ])

        # DB2: dark_factory — 2 more one-pass tasks; reify — 1 one-pass
        db2_path = _make_runs_db(tmp_path / 'db2', 'runs.db', [
            ('run-b', 't3', 'dark_factory', 'T3', 'done',
             0.3, 200_000, 2, 1, 0, 0, 0.0, 0, ts1),
            ('run-b', 't4', 'dark_factory', 'T4', 'done',
             0.3, 250_000, 2, 1, 0, 0, 0.0, 0, ts2),
            ('run-b', 't5', 'reify', 'T5', 'done',
             0.2, 150_000, 1, 1, 0, 0, 0.0, 0, ts1),
        ])

        # No escalations (via-interactive = 0 in both)
        edir1 = _make_escalations_dir(tmp_path / 'esc1', 'e', [])
        edir2 = _make_escalations_dir(tmp_path / 'esc2', 'e', [])

        async with (
            aiosqlite.connect(str(db1_path)) as c1,
            aiosqlite.connect(str(db2_path)) as c2,
        ):
            c1.row_factory = aiosqlite.Row
            c2.row_factory = aiosqlite.Row
            yield [c1, c2], [edir1, edir2]

    @pytest.mark.asyncio
    async def test_sums_counts_for_same_project(self, two_project_conns):
        """Counts for the same project_id across DBs are summed."""
        (dbs, edirs) = two_project_conns
        result = await aggregate_completion_paths(dbs, edirs, days=1)

        df = result['dark_factory']
        paths = {p['path']: p['count'] for p in df}
        # DB1: 1 one-pass + 1 via-steward; DB2: 2 one-pass
        assert paths.get('one-pass', 0) == 3
        assert paths.get('via-steward', 0) == 1

    @pytest.mark.asyncio
    async def test_both_project_ids_surface(self, two_project_conns):
        """Both dark_factory and reify appear in the result."""
        (dbs, edirs) = two_project_conns
        result = await aggregate_completion_paths(dbs, edirs, days=1)
        assert 'dark_factory' in result
        assert 'reify' in result

    @pytest.mark.asyncio
    async def test_empty_dbs_list(self, tmp_path):
        """`dbs=[]` returns empty dict."""
        result = await aggregate_completion_paths([], [], days=7)
        assert result == {}

    @pytest.mark.asyncio
    async def test_none_dbs_returns_empty(self, tmp_path):
        """`dbs=[None, None]` (no valid connections) returns empty dict."""
        edir = _make_escalations_dir(tmp_path, 'esc', [])
        result = await aggregate_completion_paths(
            [None, None], [edir, edir], days=7,
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_missing_escalations_dir_no_crash(self, tmp_path):
        """A non-existent escalations_dir does not crash; via-interactive=0."""
        now = datetime.now(UTC)
        ts = (now - timedelta(minutes=10)).isoformat()
        db_path = _make_runs_db(tmp_path, 'runs.db', [
            ('run-x', 'tx1', 'proj', 'T', 'done',
             0.3, 100_000, 1, 1, 0, 0, 0.0, 0, ts),
        ])
        missing_dir = tmp_path / 'does_not_exist'

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await aggregate_completion_paths(
                [conn], [missing_dir], days=7,
            )

        assert 'proj' in result
        paths = {p['path']: p['count'] for p in result['proj']}
        assert paths.get('via-interactive', 0) == 0


# ---------------------------------------------------------------------------
# Tests: aggregate_escalation_rates
# ---------------------------------------------------------------------------

class TestAggregateEscalationRates:
    """Tests for aggregate_escalation_rates across multiple DB connections."""

    @pytest.fixture()
    async def two_esc_conns(self, tmp_path):
        """Two runs.dbs each with dark_factory tasks; DB2 also has reify."""
        now = datetime.now(UTC)
        ts1 = (now - timedelta(minutes=40)).isoformat()
        ts2 = (now - timedelta(minutes=30)).isoformat()
        ts3 = (now - timedelta(minutes=20)).isoformat()

        # DB1: dark_factory — 3 tasks, 1 has steward_invocations > 0
        db1_path = _make_runs_db(tmp_path / 'db1', 'runs.db', [
            ('run-a', 't1', 'dark_factory', 'T1', 'done', 0.5, 300_000, 2, 1, 0, 0, 0.0, 0, ts1),
            ('run-a', 't2', 'dark_factory', 'T2', 'done', 0.5, 400_000, 3, 1, 0, 0, 0.2, 2, ts2),
            ('run-a', 't3', 'dark_factory', 'T3', 'blocked', 0.3, 150_000, 1, 1, 0, 0, 0.0, 0, ts3),
        ])
        # Escalation: task t2 had a level-1 interactive escalation with 4 turns
        esc1 = _make_escalations_dir(tmp_path / 'esc1', 'e', [
            {'id': 'e1', 'task_id': 't2', 'level': 1, 'status': 'resolved',
             'resolution_turns': 4, 'resolved_by': 'interactive'},
        ])

        # DB2: dark_factory — 2 more tasks; reify — 1 task
        db2_path = _make_runs_db(tmp_path / 'db2', 'runs.db', [
            ('run-b', 't4', 'dark_factory', 'T4', 'done', 0.3, 200_000, 2, 1, 0, 0, 0.0, 0, ts1),
            ('run-b', 't5', 'dark_factory', 'T5', 'done', 0.3, 250_000, 2, 1, 0, 0, 0.3, 3, ts2),
            ('run-b', 't6', 'reify', 'T6', 'done', 0.2, 100_000, 1, 1, 0, 0, 0.0, 0, ts3),
        ])
        esc2 = _make_escalations_dir(tmp_path / 'esc2', 'e', [])

        async with (
            aiosqlite.connect(str(db1_path)) as c1,
            aiosqlite.connect(str(db2_path)) as c2,
        ):
            c1.row_factory = aiosqlite.Row
            c2.row_factory = aiosqlite.Row
            yield [c1, c2], [esc1, esc2]

    @pytest.mark.asyncio
    async def test_sums_task_counts(self, two_esc_conns):
        """total_tasks, steward_count, interactive_count summed across DBs."""
        (dbs, edirs) = two_esc_conns
        result = await aggregate_escalation_rates(dbs, edirs, days=1)
        df = result['dark_factory']
        # DB1: 3 tasks (1 steward, 1 interactive); DB2: 2 tasks (1 steward)
        assert df['total_tasks'] == 5
        assert df['steward_count'] == 2  # t2 from DB1 + t5 from DB2
        assert df['interactive_count'] == 1  # t2 from DB1 only (has level-1 esc)

    @pytest.mark.asyncio
    async def test_human_attention_buckets_summed(self, two_esc_conns):
        """human_attention buckets are summed across DBs."""
        (dbs, edirs) = two_esc_conns
        result = await aggregate_escalation_rates(dbs, edirs, days=1)
        attention = result['dark_factory']['human_attention']
        # t2: 4 resolution_turns → 'significant' (> 2)
        assert attention['significant'] == 1
        assert attention['minimal'] == 0
        assert attention['zero'] == 0

    @pytest.mark.asyncio
    async def test_rates_recomputed_from_merged_totals(self, two_esc_conns):
        """steward_rate and interactive_rate derived from merged totals (not averaged)."""
        (dbs, edirs) = two_esc_conns
        result = await aggregate_escalation_rates(dbs, edirs, days=1)
        df = result['dark_factory']
        # 2 steward_count / 5 total = 40.0%
        assert df['steward_rate'] == pytest.approx(40.0, abs=0.1)
        # 1 interactive_count / 5 total = 20.0%
        assert df['interactive_rate'] == pytest.approx(20.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_empty_dbs_list(self, tmp_path):
        """`dbs=[]` returns empty dict."""
        result = await aggregate_escalation_rates([], [], days=7)
        assert result == {}

    @pytest.mark.asyncio
    async def test_none_dbs_returns_empty(self, tmp_path):
        """`dbs=[None, None]` returns empty dict."""
        edir = _make_escalations_dir(tmp_path, 'esc', [])
        result = await aggregate_escalation_rates(
            [None, None], [edir, edir], days=7,
        )
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: aggregate_loop_histograms
# ---------------------------------------------------------------------------

class TestAggregateLoopHistograms:
    """Tests for aggregate_loop_histograms across multiple DB connections."""

    @pytest.fixture()
    async def two_hist_conns(self, tmp_path):
        """Two runs.dbs each with dark_factory tasks (and DB2 has reify too)."""
        now = datetime.now(UTC)
        ts1 = (now - timedelta(minutes=40)).isoformat()
        ts2 = (now - timedelta(minutes=30)).isoformat()
        ts3 = (now - timedelta(minutes=20)).isoformat()

        # DB1: dark_factory — rc=0,va=0 and rc=2,va=3
        db1_path = _make_runs_db(tmp_path / 'db1', 'runs.db', [
            ('run-a', 't1', 'dark_factory', 'T1', 'done',
             0.5, 300_000, 2, 1, 0, 0, 0.0, 0, ts1),   # rc=0, va=0
            ('run-a', 't2', 'dark_factory', 'T2', 'done',
             0.5, 400_000, 3, 1, 3, 2, 0.0, 0, ts2),   # rc=2, va=3
        ])

        # DB2: dark_factory — rc=1,va=1 and rc=0,va=2
        # reify — rc=0,va=0
        db2_path = _make_runs_db(tmp_path / 'db2', 'runs.db', [
            ('run-b', 't3', 'dark_factory', 'T3', 'done',
             0.3, 200_000, 2, 1, 1, 1, 0.0, 0, ts1),   # rc=1, va=1
            ('run-b', 't4', 'dark_factory', 'T4', 'done',
             0.3, 250_000, 2, 1, 2, 0, 0.0, 0, ts2),   # rc=0, va=2
            ('run-b', 't5', 'reify', 'T5', 'done',
             0.2, 150_000, 1, 1, 0, 0, 0.0, 0, ts3),   # rc=0, va=0
        ])

        async with (
            aiosqlite.connect(str(db1_path)) as c1,
            aiosqlite.connect(str(db2_path)) as c2,
        ):
            c1.row_factory = aiosqlite.Row
            c2.row_factory = aiosqlite.Row
            yield [c1, c2]

    @pytest.mark.asyncio
    async def test_outer_bins_summed_same_project(self, two_hist_conns):
        """Outer (review_cycles) bins summed across DBs for same project_id."""
        dbs = two_hist_conns
        result = await aggregate_loop_histograms(dbs, days=1)
        df = result['dark_factory']
        outer = df['outer']
        # DB1: rc=[0,2] → bins [1,0,1,0]; DB2: rc=[1,0] → bins [1,1,0,0]
        # merged: [2, 1, 1, 0]
        assert outer['values'] == [2, 1, 1, 0]

    @pytest.mark.asyncio
    async def test_inner_bins_summed_same_project(self, two_hist_conns):
        """Inner (verify_attempts) bins summed across DBs for same project_id."""
        dbs = two_hist_conns
        result = await aggregate_loop_histograms(dbs, days=1)
        df = result['dark_factory']
        inner = df['inner']
        # DB1: va=[0,3] → bins [1,0,0,1,0,0]; DB2: va=[1,2] → bins [0,1,1,0,0,0]
        # merged: [1,1,1,1,0,0]
        assert inner['values'] == [1, 1, 1, 1, 0, 0]

    @pytest.mark.asyncio
    async def test_labels_preserved(self, two_hist_conns):
        """Labels are the canonical fixed lists for outer and inner loops."""
        dbs = two_hist_conns
        result = await aggregate_loop_histograms(dbs, days=1)
        df = result['dark_factory']
        assert df['outer']['labels'] == ['0', '1', '2', '3+']
        assert df['inner']['labels'] == ['0', '1', '2', '3', '4', '5+']

    @pytest.mark.asyncio
    async def test_two_projects_surfaced(self, two_hist_conns):
        """Both dark_factory and reify appear when each DB has a distinct project."""
        dbs = two_hist_conns
        result = await aggregate_loop_histograms(dbs, days=1)
        assert 'dark_factory' in result
        assert 'reify' in result

    @pytest.mark.asyncio
    async def test_empty_dbs_list(self):
        """`dbs=[]` returns empty dict."""
        result = await aggregate_loop_histograms([], days=7)
        assert result == {}

    @pytest.mark.asyncio
    async def test_none_dbs_returns_empty(self):
        """`dbs=[None, None]` returns empty dict."""
        result = await aggregate_loop_histograms([None, None], days=7)
        assert result == {}

    @pytest.mark.asyncio
    async def test_mismatched_histogram_longer_incoming_warns_and_merges(
        self, monkeypatch, caplog,
    ):
        """Longer incoming label list triggers WARNING and is merged by label key.

        First DB (baseline): outer has canonical 4-label list and values [1,2,3,4].
        Second DB: outer has a 6-label list (['0','1','2','3','4','5+']),
        which differs from the baseline's ['0','1','2','3+'].

        Label-dict merge appends new labels to the accumulator order and sums
        values for shared labels; the baseline '3+' label carries its value
        unchanged while incoming extra labels '3','4','5+' are appended.
        inner (canonical match across both DBs) is summed normally.
        """
        call_count = 0

        async def _fake_get_loop_histograms(db, *, days):  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Baseline (first DB)
                return {
                    'proj': {
                        'outer': {'labels': ['0', '1', '2', '3+'], 'values': [1, 2, 3, 4]},
                        'inner': {
                            'labels': ['0', '1', '2', '3', '4', '5+'],
                            'values': [1, 1, 1, 1, 1, 1],
                        },
                    },
                }
            else:
                # Second DB: outer has different 6-label list (mismatch), inner matches
                return {
                    'proj': {
                        'outer': {
                            'labels': ['0', '1', '2', '3', '4', '5+'],
                            'values': [5, 6, 7, 8, 9, 10],
                        },
                        'inner': {
                            'labels': ['0', '1', '2', '3', '4', '5+'],
                            'values': [2, 3, 4, 5, 6, 7],
                        },
                    },
                }

        monkeypatch.setattr(
            'dashboard.data.performance.get_loop_histograms',
            _fake_get_loop_histograms,
        )

        with caplog.at_level(logging.WARNING, logger='dashboard.data.performance'):
            result = await aggregate_loop_histograms([None, None], days=7)

        # (a) No exception — we got here
        assert 'proj' in result

        # (b) At least one WARNING record mentions 'mismatch' and 'proj'
        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any('mismatch' in t and 'proj' in t for t in warning_texts), (
            f'Expected mismatch warning mentioning proj; got: {warning_texts}'
        )

        # (c) Label-dict merge: shared labels summed, new labels appended
        #   '0'→1+5=6, '1'→2+6=8, '2'→3+7=10, '3+'→4 (not in incoming)
        #   new from incoming: '3'→8, '4'→9, '5+'→10
        outer = result['proj']['outer']
        assert outer['labels'] == ['0', '1', '2', '3+', '3', '4', '5+']
        assert outer['values'] == [6, 8, 10, 4, 8, 9, 10]

        # (d) Matched inner merged normally: [1+2, 1+3, 1+4, 1+5, 1+6, 1+7]
        assert result['proj']['inner']['values'] == [3, 4, 5, 6, 7, 8]

    @pytest.mark.asyncio
    async def test_mismatched_histogram_shorter_incoming_warns_and_merges(
        self, monkeypatch, caplog,
    ):
        """Shorter incoming label list triggers WARNING and is merged by label key.

        First DB (baseline): outer has canonical ['0','1','2','3+'] with values [1,2,3,4].
        Second DB: outer has only 3 labels ['0','1','2+'] (shorter, partly overlapping).

        Label-dict merge: shared labels '0','1' summed; baseline-only '2','3+'
        retain their values; incoming-only '2+' appended with its value.
        inner (canonical match) is summed normally.
        """
        call_count = 0

        async def _fake_get_loop_histograms(db, *, days):  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    'proj': {
                        'outer': {'labels': ['0', '1', '2', '3+'], 'values': [1, 2, 3, 4]},
                        'inner': {
                            'labels': ['0', '1', '2', '3', '4', '5+'],
                            'values': [1, 1, 1, 1, 1, 1],
                        },
                    },
                }
            else:
                # Second DB: outer has 3 non-identical labels, inner matches
                return {
                    'proj': {
                        'outer': {'labels': ['0', '1', '2+'], 'values': [5, 6, 7]},
                        'inner': {
                            'labels': ['0', '1', '2', '3', '4', '5+'],
                            'values': [2, 3, 4, 5, 6, 7],
                        },
                    },
                }

        monkeypatch.setattr(
            'dashboard.data.performance.get_loop_histograms',
            _fake_get_loop_histograms,
        )

        with caplog.at_level(logging.WARNING, logger='dashboard.data.performance'):
            result = await aggregate_loop_histograms([None, None], days=7)

        # (a) No exception raised
        assert 'proj' in result

        # (b) At least one WARNING record mentions 'mismatch' and 'proj'
        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any('mismatch' in t and 'proj' in t for t in warning_texts), (
            f'Expected mismatch warning mentioning proj; got: {warning_texts}'
        )

        # (c) Label-dict merge: shared labels summed, baseline-only retained,
        #   incoming-only appended
        #   '0'→1+5=6, '1'→2+6=8, '2'→3 (baseline-only), '3+'→4 (baseline-only)
        #   '2+' is incoming-only → appended with value 7
        outer = result['proj']['outer']
        assert outer['labels'] == ['0', '1', '2', '3+', '2+']
        assert outer['values'] == [6, 8, 3, 4, 7]

        # (d) Matched inner merged normally: [1+2, 1+3, 1+4, 1+5, 1+6, 1+7]
        assert result['proj']['inner']['values'] == [3, 4, 5, 6, 7, 8]

    @pytest.mark.asyncio
    async def test_divergent_values_length_does_not_crash(
        self, monkeypatch, caplog,
    ):
        """Mismatched label lists are merged by label key; no IndexError, WARNING emitted.

        DB1 outer: 4 canonical labels ['0','1','2','3+'] with values [1,1,0,0].
        DB2 outer: 3 labels ['0','1','2'] with values [2,1,0] (shorter, non-canonical).
        Both DBs share a canonical inner histogram.

        Label-dict merge must:
        (a) not raise an exception
        (b) keep 'p' in the result
        (c) emit a WARNING mentioning the mismatch and project 'p'
        (d) merge outer by label: '0'→1+2=3, '1'→1+1=2, '2'→0+0=0, '3+'→0
            so outer.values == [3, 2, 0, 0], not the skip-preserved baseline [1, 1, 0, 0]
        """
        call_count = 0

        async def _fake_get_loop_histograms(db, *, days):  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            canonical_inner = {
                'labels': ['0', '1', '2', '3', '4', '5+'],
                'values': [2, 2, 0, 0, 0, 0],
            }
            if call_count == 1:
                return {
                    'p': {
                        'outer': {
                            'labels': ['0', '1', '2', '3+'],
                            'values': [1, 1, 0, 0],
                        },
                        'inner': canonical_inner,
                    },
                }
            else:
                return {
                    'p': {
                        'outer': {
                            'labels': ['0', '1', '2'],
                            'values': [2, 1, 0],
                        },
                        'inner': canonical_inner,
                    },
                }

        monkeypatch.setattr(
            'dashboard.data.performance.get_loop_histograms',
            _fake_get_loop_histograms,
        )

        with caplog.at_level(logging.WARNING, logger='dashboard.data.performance'):
            result = await aggregate_loop_histograms([None, None], days=7)

        # (a) No exception raised — we got here
        assert 'p' in result

        # (b) At least one WARNING mentioning mismatch and project 'p'
        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any('mismatch' in t and 'p' in t for t in warning_texts), (
            f'Expected mismatch warning mentioning project p; got: {warning_texts}'
        )

        # (c) outer is merged by label (not skipped): '0'→3, '1'→2, '2'→0, '3+'→0
        outer = result['p']['outer']
        assert outer['labels'] == ['0', '1', '2', '3+']
        assert outer['values'] == [3, 2, 0, 0], (
            f'Expected label-dict merge [3,2,0,0], got {outer["values"]!r}; '
            f'skip-on-mismatch would leave baseline [1,1,0,0]'
        )

        # (d) inner (canonical match) is element-wise summed: [2+2, 2+2, 0, 0, 0, 0]
        assert result['p']['inner']['values'] == [4, 4, 0, 0, 0, 0]

    @pytest.mark.asyncio
    async def test_three_dbs_canonical_mismatched_canonical_two_warnings(
        self, monkeypatch, caplog,
    ):
        """Three DBs (canonical → extended → canonical) emit two WARNINGs.

        DB1 outer: canonical ['0','1','2','3+'] → [1,2,3,4].
        DB2 outer: extended  ['0','1','2','3+','4+'] → [5,6,7,8,9].
          Mismatch #1 vs accumulator → dict-merge; accumulator becomes
          labels=['0','1','2','3+','4+'], values=[6,8,10,12,9].
        DB3 outer: canonical ['0','1','2','3+'] → [1,1,1,1].
          Mismatch #2 vs extended accumulator → dict-merge; '4+' retains 9.
          Final: labels=['0','1','2','3+','4+'], values=[7,9,11,13,9].

        All DBs share identical canonical inner (values [1,1,1,1,1,1]) →
        no inner WARNINGs; inner fast-path sums to [3,3,3,3,3,3].
        """
        call_count = 0
        canonical_inner = {
            'labels': ['0', '1', '2', '3', '4', '5+'],
            'values': [1, 1, 1, 1, 1, 1],
        }

        async def _fake_get_loop_histograms(db, *, days):  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    'proj': {
                        'outer': {'labels': ['0', '1', '2', '3+'], 'values': [1, 2, 3, 4]},
                        'inner': canonical_inner,
                    },
                }
            elif call_count == 2:
                return {
                    'proj': {
                        'outer': {
                            'labels': ['0', '1', '2', '3+', '4+'],
                            'values': [5, 6, 7, 8, 9],
                        },
                        'inner': canonical_inner,
                    },
                }
            else:
                return {
                    'proj': {
                        'outer': {'labels': ['0', '1', '2', '3+'], 'values': [1, 1, 1, 1]},
                        'inner': canonical_inner,
                    },
                }

        monkeypatch.setattr(
            'dashboard.data.performance.get_loop_histograms',
            _fake_get_loop_histograms,
        )

        with caplog.at_level(logging.WARNING, logger='dashboard.data.performance'):
            result = await aggregate_loop_histograms([None, None, None], days=7)

        # (a) No exception raised — we got here
        assert 'proj' in result

        # (b) Exactly two WARNINGs for outer mismatch (DB2 and DB3)
        mismatch_warns = [
            r.message
            for r in caplog.records
            if r.levelno >= logging.WARNING and 'mismatch' in r.message and 'proj' in r.message
        ]
        assert len(mismatch_warns) == 2, (
            f'Expected 2 outer-mismatch warnings, got {len(mismatch_warns)}: {mismatch_warns}'
        )

        # (c) outer: label-dict merges accumulate correctly across all three DBs
        outer = result['proj']['outer']
        assert outer['labels'] == ['0', '1', '2', '3+', '4+']
        assert outer['values'] == [7, 9, 11, 13, 9], (
            f'Expected [7,9,11,13,9] after three-DB merge, got {outer["values"]!r}'
        )

        # (d) inner: no mismatch — fast-path element-wise sum → [3,3,3,3,3,3]
        assert result['proj']['inner']['values'] == [3, 3, 3, 3, 3, 3]


# ---------------------------------------------------------------------------
# Tests: aggregate_time_centiles
# ---------------------------------------------------------------------------

class TestAggregateTimeCentiles:
    """Tests for aggregate_time_centiles across multiple DB connections."""

    @pytest.fixture()
    async def two_centile_conns(self, tmp_path):
        """Two runs.dbs: DB1 has dark_factory with short durations,
        DB2 has dark_factory with longer durations (and reify too)."""
        now = datetime.now(UTC)
        ts1 = (now - timedelta(minutes=40)).isoformat()
        ts2 = (now - timedelta(minutes=30)).isoformat()
        ts3 = (now - timedelta(minutes=20)).isoformat()

        # DB1: dark_factory — durations 100k, 200k ms
        db1_path = _make_runs_db(tmp_path / 'db1', 'runs.db', [
            ('run-a', 't1', 'dark_factory', 'T1', 'done',
             0.5, 100_000, 2, 1, 0, 0, 0.0, 0, ts1),
            ('run-a', 't2', 'dark_factory', 'T2', 'done',
             0.5, 200_000, 2, 1, 0, 0, 0.0, 0, ts2),
        ])

        # DB2: dark_factory — durations 300k, 400k ms; reify — 500k ms
        db2_path = _make_runs_db(tmp_path / 'db2', 'runs.db', [
            ('run-b', 't3', 'dark_factory', 'T3', 'done',
             0.3, 300_000, 2, 1, 0, 0, 0.0, 0, ts1),
            ('run-b', 't4', 'dark_factory', 'T4', 'done',
             0.3, 400_000, 2, 1, 0, 0, 0.0, 0, ts2),
            ('run-b', 't5', 'reify', 'T5', 'done',
             0.2, 500_000, 1, 1, 0, 0, 0.0, 0, ts3),
        ])

        async with (
            aiosqlite.connect(str(db1_path)) as c1,
            aiosqlite.connect(str(db2_path)) as c2,
        ):
            c1.row_factory = aiosqlite.Row
            c2.row_factory = aiosqlite.Row
            yield [c1, c2]

    @pytest.mark.asyncio
    async def test_count_sums(self, two_centile_conns):
        """count is the sum of task counts across DBs."""
        dbs = two_centile_conns
        result = await aggregate_time_centiles(dbs, days=1)
        # DB1: 2 tasks, DB2: 2 dark_factory tasks = 4 total
        assert result['dark_factory']['count'] == 4

    @pytest.mark.asyncio
    async def test_percentiles_from_unified_sample(self, two_centile_conns):
        """p50 is the true median of all 4 durations [100k,200k,300k,400k]."""
        dbs = two_centile_conns
        result = await aggregate_time_centiles(dbs, days=1)
        # True median of [100k,200k,300k,400k] = 250k
        df = result['dark_factory']
        assert df['p50'] == pytest.approx(250_000, rel=0.05)
        assert df['p50'] <= df['p75'] <= df['p90'] <= df['p95']

    @pytest.mark.asyncio
    async def test_reify_project_surfaced(self, two_centile_conns):
        """reify project from DB2 appears in result."""
        dbs = two_centile_conns
        result = await aggregate_time_centiles(dbs, days=1)
        assert 'reify' in result
        assert result['reify']['count'] == 1
        assert result['reify']['p50'] == 500_000

    @pytest.mark.asyncio
    async def test_empty_dbs_list(self):
        """`dbs=[]` returns empty dict."""
        result = await aggregate_time_centiles([], days=7)
        assert result == {}

    @pytest.mark.asyncio
    async def test_none_dbs_returns_empty(self):
        """`dbs=[None, None]` returns empty dict."""
        result = await aggregate_time_centiles([None, None], days=7)
        assert result == {}
