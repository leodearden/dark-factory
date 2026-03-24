"""Tests for dashboard.data.performance — orchestrator performance queries."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from dashboard.data.performance import (
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
    async def test_populated(self, runs_db, escalations_dir):
        result = await get_completion_paths(runs_db, escalations_dir)
        assert 'dark_factory' in result

        paths = {p['path']: p for p in result['dark_factory']}
        assert paths['one-pass']['count'] == 2
        assert paths['multi-pass']['count'] == 1
        assert paths['via-steward']['count'] == 1
        assert paths['via-interactive']['count'] == 1
        assert paths['blocked']['count'] == 1

    @pytest.mark.asyncio
    async def test_multi_project(self, runs_db, escalations_dir):
        result = await get_completion_paths(runs_db, escalations_dir)
        assert 'reify' in result
        paths = {p['path']: p for p in result['reify']}
        assert paths['one-pass']['count'] == 1
        assert paths['multi-pass']['count'] == 1

    @pytest.mark.asyncio
    async def test_empty_db(self, empty_runs_db, empty_escalations_dir):
        result = await get_completion_paths(empty_runs_db, empty_escalations_dir)
        assert result == {}

    @pytest.mark.asyncio
    async def test_missing_db(self, tmp_path, empty_escalations_dir):
        missing = tmp_path / 'nonexistent.db'
        result = await get_completion_paths(missing, empty_escalations_dir)
        assert result == {}

    @pytest.mark.asyncio
    async def test_percentages_sum(self, runs_db, escalations_dir):
        result = await get_completion_paths(runs_db, escalations_dir)
        for _project_id, paths in result.items():
            total_pct = sum(p['pct'] for p in paths)
            assert abs(total_pct - 100.0) < 1.0  # within rounding


# ---------------------------------------------------------------------------
# Tests: get_escalation_rates
# ---------------------------------------------------------------------------

class TestEscalationRates:
    @pytest.mark.asyncio
    async def test_populated(self, runs_db, escalations_dir):
        result = await get_escalation_rates(runs_db, escalations_dir)
        df = result['dark_factory']
        assert df['total_tasks'] == 6
        assert df['steward_count'] == 1  # task 104
        assert df['interactive_count'] == 1  # task 106
        assert df['steward_rate'] == pytest.approx(16.7, abs=0.1)
        assert df['interactive_rate'] == pytest.approx(16.7, abs=0.1)

    @pytest.mark.asyncio
    async def test_human_attention(self, runs_db, escalations_dir):
        result = await get_escalation_rates(runs_db, escalations_dir)
        attention = result['dark_factory']['human_attention']
        assert attention['significant'] == 1  # task 106: 3 turns >= 3
        assert attention['minimal'] == 0
        assert attention['zero'] == 0

    @pytest.mark.asyncio
    async def test_empty(self, empty_runs_db, empty_escalations_dir):
        result = await get_escalation_rates(empty_runs_db, empty_escalations_dir)
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: get_loop_histograms
# ---------------------------------------------------------------------------

class TestLoopHistograms:
    @pytest.mark.asyncio
    async def test_populated(self, runs_db):
        result = await get_loop_histograms(runs_db)
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
    async def test_reify_project(self, runs_db):
        result = await get_loop_histograms(runs_db)
        reify = result['reify']
        # task 201: rc=0, va=0; task 202: rc=1, va=1
        assert reify['outer']['values'] == [1, 1, 0, 0]
        assert reify['inner']['values'] == [1, 1, 0, 0, 0, 0]

    @pytest.mark.asyncio
    async def test_empty(self, empty_runs_db):
        result = await get_loop_histograms(empty_runs_db)
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: get_time_centiles
# ---------------------------------------------------------------------------

class TestTimeCentiles:
    @pytest.mark.asyncio
    async def test_populated(self, runs_db):
        result = await get_time_centiles(runs_db)
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

        result = await get_time_centiles(db_path)
        assert result['proj']['p50'] == 42_000
        assert result['proj']['p95'] == 42_000
        assert result['proj']['count'] == 1

    @pytest.mark.asyncio
    async def test_empty(self, empty_runs_db):
        result = await get_time_centiles(empty_runs_db)
        assert result == {}

    @pytest.mark.asyncio
    async def test_missing_db(self, tmp_path):
        missing = tmp_path / 'nope.db'
        result = await get_time_centiles(missing)
        assert result == {}
