"""Tests for dashboard.data.load — /api/load data module."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import aiosqlite
import pytest

from dashboard.data.load import (
    KNOWN_METRICS,
    LOAD_SAMPLES_SCHEMA,
    get_load_metrics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def populated_load_db(tmp_path: Path) -> Path:
    """Build a tmp_path samples DB with a mix of known and unknown metrics.

    Inserts:
    - 3 rows for occt_queue_depth (ts=100/110/120, value=1/2/3; latest has window_mean=2.0, window_max=3.0)
    - 2 rows for psi_cpu_some_avg10 (ts=100/120, value=5/7.5; window_mean=NULL, window_max=NULL)
    - 1 row for bogus_metric (unknown, should be filtered out)
    """
    db_path = tmp_path / 'load-samples.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(LOAD_SAMPLES_SCHEMA)

    # occt_queue_depth — process metric, last row carries window_mean/window_max
    conn.executemany(
        'INSERT INTO samples (ts, metric, value, window_mean, window_max) VALUES (?, ?, ?, ?, ?)',
        [
            (100, 'occt_queue_depth', 1.0, None, None),
            (110, 'occt_queue_depth', 2.0, None, None),
            (120, 'occt_queue_depth', 3.0, 2.0, 3.0),
        ],
    )

    # psi_cpu_some_avg10 — PSI metric, window cols always NULL
    conn.executemany(
        'INSERT INTO samples (ts, metric, value, window_mean, window_max) VALUES (?, ?, ?, ?, ?)',
        [
            (100, 'psi_cpu_some_avg10', 5.0, None, None),
            (120, 'psi_cpu_some_avg10', 7.5, None, None),
        ],
    )

    # Unknown metric — should not appear in result
    conn.execute(
        'INSERT INTO samples (ts, metric, value, window_mean, window_max) VALUES (?, ?, ?, ?, ?)',
        (100, 'bogus_metric', 42.0, None, None),
    )

    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Tests: core shape contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_all_known_keys_with_values_or_placeholders(populated_load_db: Path) -> None:
    """All 9 KNOWN_METRICS keys present; populated and placeholder metrics correct."""
    async with aiosqlite.connect(str(populated_load_db)) as conn:
        conn.row_factory = aiosqlite.Row
        result = await get_load_metrics(conn)

    # All 9 known metrics present
    assert set(result.keys()) == set(KNOWN_METRICS)

    # Populated process metric with window data on the latest row
    assert result['occt_queue_depth'] == {
        'current': 3.0,
        'sparkline': [1.0, 2.0, 3.0],
        'window_mean': 2.0,
        'window_max': 3.0,
    }

    # PSI metric: window cols always NULL
    assert result['psi_cpu_some_avg10'] == {
        'current': 7.5,
        'sparkline': [5.0, 7.5],
        'window_mean': None,
        'window_max': None,
    }

    # Unpopulated known metric → placeholder shape
    assert result['verify_concurrency'] == {
        'current': None,
        'sparkline': [],
        'window_mean': None,
        'window_max': None,
    }

    # Unknown metric must be absent
    assert 'bogus_metric' not in result


# ---------------------------------------------------------------------------
# Tests: empty and None DB
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_db_returns_all_placeholders(tmp_path: Path) -> None:
    """Empty DB (schema exists, no rows) returns placeholder for every known metric."""
    db_path = tmp_path / 'empty-load.db'
    conn_sync = sqlite3.connect(str(db_path))
    conn_sync.executescript(LOAD_SAMPLES_SCHEMA)
    conn_sync.commit()
    conn_sync.close()

    async with aiosqlite.connect(str(db_path)) as conn:
        conn.row_factory = aiosqlite.Row
        result = await get_load_metrics(conn)

    placeholder = {'current': None, 'sparkline': [], 'window_mean': None, 'window_max': None}

    assert len(result) == len(KNOWN_METRICS) == 9
    assert set(result.keys()) == set(KNOWN_METRICS)
    for metric in KNOWN_METRICS:
        assert result[metric] == placeholder, f'{metric} not placeholder: {result[metric]}'


@pytest.mark.asyncio
async def test_none_db_returns_all_placeholders() -> None:
    """None connection (DB absent/unavailable) returns placeholder for every known metric."""
    result = await get_load_metrics(None)

    placeholder = {'current': None, 'sparkline': [], 'window_mean': None, 'window_max': None}

    assert len(result) == len(KNOWN_METRICS) == 9
    assert set(result.keys()) == set(KNOWN_METRICS)
    for metric in KNOWN_METRICS:
        assert result[metric] == placeholder, f'{metric} not placeholder: {result[metric]}'


# ---------------------------------------------------------------------------
# Tests: sparkline cap and ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sparkline_clamped_to_60_and_chronological(tmp_path: Path) -> None:
    """75 rows for verify_concurrency → sparkline has exactly 60 entries (oldest 60..75th retained),
    in ascending ts order. current == 75.0."""
    db_path = tmp_path / 'capped-load.db'
    conn_sync = sqlite3.connect(str(db_path))
    conn_sync.executescript(LOAD_SAMPLES_SCHEMA)
    conn_sync.executemany(
        'INSERT INTO samples (ts, metric, value, window_mean, window_max) VALUES (?, ?, ?, ?, ?)',
        [(ts, 'verify_concurrency', float(ts), None, None) for ts in range(1, 76)],
    )
    conn_sync.commit()
    conn_sync.close()

    async with aiosqlite.connect(str(db_path)) as conn:
        conn.row_factory = aiosqlite.Row
        result = await get_load_metrics(conn)

    sparkline = result['verify_concurrency']['sparkline']
    assert len(sparkline) == 60
    # Oldest kept row (ts=16) is first; newest (ts=75) is last
    assert sparkline[0] == 16.0
    assert sparkline[-1] == 75.0
    assert result['verify_concurrency']['current'] == 75.0


# ---------------------------------------------------------------------------
# Tests: DashboardConfig.load_samples_db property
# ---------------------------------------------------------------------------


def test_load_samples_db_property_points_to_canonical_path(tmp_path: Path) -> None:
    """DashboardConfig.load_samples_db returns <project_root>/data/load-samples.db."""
    from dashboard.config import DashboardConfig

    config = DashboardConfig(project_root=tmp_path)
    # tmp_path is already resolved; __post_init__ resolves project_root too
    assert config.load_samples_db == config.project_root / 'data' / 'load-samples.db'
