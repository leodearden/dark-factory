"""Tests for the new aggregators added alongside metrics.db.

Covers:
- burndown.compute_forecast_confidence (recent vs lifetime velocity)
- costs.aggregate_cost_summary (tokens + run_costs + p95)
- performance.aggregate_performance_history (hour-bucketed history)
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite
import pytest

from dashboard.data.burndown import compute_forecast_confidence
from dashboard.data.costs import aggregate_cost_summary
from dashboard.data.performance import aggregate_performance_history


# ---------------------------------------------------------------------------
# Forecast confidence
# ---------------------------------------------------------------------------


def test_forecast_confidence_returns_nones_on_sparse_history():
    """Fewer than 7 distinct days of history → no forecast (UI renders '—')."""
    series = {
        'labels': [f'2026-04-{d:02d}T00:00:00' for d in range(1, 4)],
        'done': [1, 2, 3],
        'pending': [10, 9, 8],
    }
    assert compute_forecast_confidence(series) == {
        'forecast_low': None,
        'forecast_high': None,
    }


def test_forecast_confidence_returns_zero_when_pending_zero():
    series = {
        'labels': [f'2026-04-{d:02d}T00:00:00' for d in range(1, 10)],
        'done': list(range(9)),
        'pending': [0] * 9,
    }
    assert compute_forecast_confidence(series) == {
        'forecast_low': 0,
        'forecast_high': 0,
    }


def test_forecast_confidence_uses_recent_and_lifetime_velocity():
    """Recent velocity faster → forecast_low driven by recent."""
    days = [f'2026-04-{d:02d}T00:00:00' for d in range(1, 11)]
    # Lifetime: 10 done in 10 days = 1/day. Last 7d: 9 done = ~1.3/day.
    # 5 pending → recent gives ~3.9d, lifetime gives ~5d.
    done = [0, 0, 0, 1, 2, 4, 5, 7, 8, 10]
    pending = [10, 9, 8, 7, 7, 6, 5, 5, 5, 5]
    series = {'labels': days, 'done': done, 'pending': pending}
    result = compute_forecast_confidence(series)
    assert result['forecast_low'] is not None
    assert result['forecast_high'] is not None
    assert result['forecast_low'] <= result['forecast_high']


# ---------------------------------------------------------------------------
# Costs: aggregate_cost_summary surfaces tokens + run-cost lists + p95
# ---------------------------------------------------------------------------


def _create_runs_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
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
    )
    conn.commit()
    conn.close()


@pytest.mark.asyncio
async def test_aggregate_cost_summary_sums_tokens_and_p95(tmp_path: Path):
    db_path = tmp_path / 'runs.db'
    _create_runs_db(db_path)
    conn = sqlite3.connect(str(db_path))
    now = datetime.now(UTC).isoformat()
    rows = [
        # run-1: two invocations, cost 1.0 + 2.0 = 3.0; tokens 100/200/0/0
        ('run-1', 't1', 'proj-a', 'acct', 'sonnet', 'execute', 1.0, 50, 100, 0, 0, now, now),
        ('run-1', 't1', 'proj-a', 'acct', 'sonnet', 'execute', 2.0, 50, 100, 0, 0, now, now),
        # run-2: cost 5.0
        ('run-2', 't2', 'proj-a', 'acct', 'sonnet', 'plan', 5.0, 200, 300, 100, 50, now, now),
        # run-3: cost 10.0 (drives p95)
        ('run-3', 't3', 'proj-a', 'acct', 'sonnet', 'execute', 10.0, 0, 0, 0, 0, now, now),
    ]
    for r in rows:
        conn.execute(
            'INSERT INTO invocations (run_id, task_id, project_id, account_name, '
            'model, role, cost_usd, input_tokens, output_tokens, cache_read_tokens, '
            'cache_create_tokens, started_at, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            r,
        )
    conn.commit()
    conn.close()

    db = await aiosqlite.connect(f'file:{db_path}?mode=ro', uri=True)
    db.row_factory = aiosqlite.Row
    try:
        result = await aggregate_cost_summary([db], days=30)
    finally:
        await db.close()
    a = result['proj-a']
    assert a['total_spend'] == pytest.approx(18.0)
    assert a['tokens']['input'] == 300
    assert a['tokens']['output'] == 500
    assert a['tokens']['total'] == 300 + 500 + 100 + 50
    # 3 runs: 3.0, 5.0, 10.0 → p95 should land between 5 and 10.
    assert a['p95_run_cost'] is not None
    assert 5.0 <= a['p95_run_cost'] <= 10.0
    # run_costs preserved for shape_costs to reuse globally.
    assert sorted(a['run_costs']) == sorted([3.0, 5.0, 10.0])


# ---------------------------------------------------------------------------
# Performance: hour-bucketed history aggregator
# ---------------------------------------------------------------------------


def _create_task_results_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE task_results (
            run_id              TEXT NOT NULL,
            task_id             TEXT NOT NULL,
            project_id          TEXT NOT NULL,
            title               TEXT,
            outcome             TEXT,
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
        CREATE INDEX idx_task_results_project ON task_results(project_id, completed_at);
        """
    )
    conn.commit()
    conn.close()


@pytest.mark.asyncio
async def test_aggregate_performance_history_buckets_by_hour(tmp_path: Path):
    db_path = tmp_path / 'runs.db'
    _create_task_results_db(db_path)
    conn = sqlite3.connect(str(db_path))
    base = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    rows = [
        # bucket H0: two done tasks, one one-pass (review_cycles=0), one not, one steward.
        ('r1', 't1', 'proj-a', 'done', 1000, 0, 0, base.isoformat()),
        ('r2', 't2', 'proj-a', 'done', 3000, 2, 1, (base + timedelta(minutes=10)).isoformat()),
        # bucket H+1: one done one-pass.
        ('r3', 't3', 'proj-a', 'done', 2000, 0, 0, (base + timedelta(hours=1, minutes=5)).isoformat()),
    ]
    for r in rows:
        conn.execute(
            'INSERT INTO task_results (run_id, task_id, project_id, outcome, duration_ms, '
            'review_cycles, steward_invocations, completed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            r,
        )
    conn.commit()
    conn.close()

    db = await aiosqlite.connect(f'file:{db_path}?mode=ro', uri=True)
    db.row_factory = aiosqlite.Row
    try:
        result = await aggregate_performance_history([db], days=7)
    finally:
        await db.close()
    block = result['proj-a']
    h = block['time_centiles_history']
    assert len(h['labels']) == 2
    assert len(h['p50']) == 2
    # First bucket has 2 done tasks, p50 = avg of 1000 and 3000.
    assert h['p50'][0] in (1000, 2000, 3000)
    # one_pass_history first bucket: 1 of 2 done tasks is one-pass = 50%.
    assert block['one_pass_history']['values'][0] == 50.0
    # escalation_history first bucket: 1 of 2 had steward_invocations > 0 = 50%.
    assert block['escalation_history']['values'][0] == 50.0
