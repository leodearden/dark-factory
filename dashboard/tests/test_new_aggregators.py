"""Tests for the new aggregators added alongside metrics.db.

Covers:
- burndown.compute_forecast_confidence (recent vs lifetime velocity)
- burndown.aggregate_forecast_confidence (the same forecast folded over
  per-project measured series)
- costs.aggregate_cost_summary (tokens + run_costs + p95)
- performance.aggregate_performance_history (hour-bucketed history)
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite
import pytest

from dashboard.data.burndown import (
    aggregate_forecast_confidence,
    compute_forecast_confidence,
)
from dashboard.data.costs import aggregate_cost_summary
from dashboard.data.performance import aggregate_performance_history

# ---------------------------------------------------------------------------
# Forecast confidence
# ---------------------------------------------------------------------------

NO_FORECAST = {'forecast_low': None, 'forecast_high': None}

SPARSE_SERIES = {
    'labels': [f'2026-04-{d:02d}T00:00:00' for d in range(1, 4)],
    'done': [1, 2, 3],
    'pending': [10, 9, 8],
}

ZERO_PENDING_SERIES = {
    'labels': [f'2026-04-{d:02d}T00:00:00' for d in range(1, 10)],
    'done': list(range(9)),
    'pending': [0] * 9,
}

# Lifetime: 10 done in 10 days = 1/day. Last 7d: 9 done = ~1.3/day.
# 5 pending → recent gives ~3.9d, lifetime gives ~5d.
RECENT_FASTER_SERIES = {
    'labels': [f'2026-04-{d:02d}T00:00:00' for d in range(1, 11)],
    'done': [0, 0, 0, 1, 2, 4, 5, 7, 8, 10],
    'pending': [10, 9, 8, 7, 7, 6, 5, 5, 5, 5],
}


def _daily(first_day, done, pending):
    """A series with one row per day from 2026-04-*first_day*."""
    return {
        'labels': [f'2026-04-{first_day + i:02d}T00:00:00' for i in range(len(done))],
        'done': list(done),
        'pending': list(pending),
    }


def test_forecast_confidence_returns_nones_on_sparse_history():
    """Fewer than 7 distinct days of history → no forecast (UI renders '—')."""
    assert compute_forecast_confidence(SPARSE_SERIES) == NO_FORECAST


def test_forecast_confidence_returns_zero_when_pending_zero():
    assert compute_forecast_confidence(ZERO_PENDING_SERIES) == {
        'forecast_low': 0,
        'forecast_high': 0,
    }


def test_forecast_confidence_uses_recent_and_lifetime_velocity():
    """Recent velocity faster → forecast_low driven by recent."""
    result = compute_forecast_confidence(RECENT_FASTER_SERIES)
    assert result['forecast_low'] is not None
    assert result['forecast_high'] is not None
    assert result['forecast_low'] <= result['forecast_high']


@pytest.mark.parametrize(
    'series',
    [SPARSE_SERIES, ZERO_PENDING_SERIES, RECENT_FASTER_SERIES],
    ids=['sparse', 'zero-pending', 'recent-faster'],
)
def test_single_series_forecast_is_the_fold_of_one_series(series):
    """The per-project forecast and the aggregate fold agree on one project."""
    assert compute_forecast_confidence(series) == aggregate_forecast_confidence([series])


def test_fold_does_not_read_a_mid_window_entry_as_velocity():
    """B's first measured row (done 4000, day 6) is where B starts, not 4000 completions.

    Velocity comes from A's measured gains alone; pending is both projects'
    last measured pending (20 + 5). A delta over the summed ``done`` series
    would read B's arrival as a 4000-task jump and forecast ~0 days.
    """
    a = _daily(1, done=range(10), pending=[20] * 10)
    b = _daily(6, done=[4000] * 5, pending=[5] * 5)

    result = aggregate_forecast_confidence([a, b])

    assert result == {'forecast_low': 27.8, 'forecast_high': 29.2}
    a_with_both_pendings = {**a, 'pending': [*a['pending'][:-1], 25]}
    assert result == compute_forecast_confidence(a_with_both_pendings)
    summed = {
        'labels': a['labels'],
        'done': [*a['done'][:5], *(x + 4000 for x in a['done'][5:])],
        'pending': [*a['pending'][:5], *([25] * 5)],
    }
    assert compute_forecast_confidence(summed)['forecast_low'] == 0.0


def test_fold_sums_each_projects_last_measured_pending():
    """A's last row is day 8, B's is day 7: pending is 12 + 8, not A's 12 alone."""
    a = _daily(1, done=range(8), pending=[10] * 7 + [12])
    b = _daily(1, done=[0, 1, 1, 1, 1, 1, 2], pending=[3] * 6 + [8])

    # v_recent = (6 + 1) / 7 = 1.0 → 20.0d; v_lifetime = (7 + 2) / 8 → 17.8d.
    assert aggregate_forecast_confidence([a, b]) == {
        'forecast_low': 17.8,
        'forecast_high': 20.0,
    }


def test_fold_refuses_a_project_whose_lists_disagree_in_length():
    """One ragged project makes the whole fold unknown, never a partial sum."""
    a = _daily(1, done=range(10), pending=[20] * 10)
    ragged = {**_daily(1, done=range(10), pending=[5] * 10), 'done': list(range(9))}
    assert aggregate_forecast_confidence([a, ragged]) == NO_FORECAST


def test_fold_counts_distinct_days_over_the_union_of_labels():
    """Six days across two projects is sparse; seven across two is enough."""
    six = [_daily(1, done=[0, 1, 2], pending=[5] * 3), _daily(4, done=[0, 1, 2], pending=[5] * 3)]
    assert aggregate_forecast_confidence(six) == NO_FORECAST

    seven = [_daily(1, done=[0, 1, 2, 3], pending=[5] * 4), _daily(4, done=[0, 1, 2, 3], pending=[5] * 4)]
    # 3 + 3 gained over 7 days, 10 pending → 10 / (6 / 7) = 11.7d either window.
    assert aggregate_forecast_confidence(seven) == {'forecast_low': 11.7, 'forecast_high': 11.7}


@pytest.mark.parametrize(
    'empty',
    [{'labels': [], 'done': [], 'pending': []}, {}],
    ids=['empty-lists', 'no-keys'],
)
def test_fold_ignores_an_empty_series(empty):
    """A project with nothing measured contributes nothing — it does not void the fold."""
    assert aggregate_forecast_confidence([RECENT_FASTER_SERIES, empty]) == (
        compute_forecast_confidence(RECENT_FASTER_SERIES)
    )


def test_fold_of_nothing_is_no_forecast():
    assert aggregate_forecast_confidence([]) == NO_FORECAST
    assert aggregate_forecast_confidence([{}]) == NO_FORECAST


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
