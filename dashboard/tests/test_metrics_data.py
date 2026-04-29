"""Tests for dashboard.data.metrics — schema, samplers, and read aggregators."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite
import pytest

from dashboard.data.metrics import (
    METRICS_SCHEMA,
    _split_queue_stats,
    _split_status,
    downsample_metrics,
    get_memory_24h_ago,
    get_memory_sparks,
    get_merge_active_series,
    get_orchestrators_running_series,
    get_queue_pending_series,
    get_recon_sparks,
)


def _create_metrics_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.executescript(METRICS_SCHEMA)
    conn.commit()
    return conn


@pytest.fixture
def metrics_db_path(tmp_path: Path) -> Path:
    db_path = tmp_path / 'metrics.db'
    conn = _create_metrics_db(db_path)
    conn.close()
    return db_path


@pytest.fixture
async def ro_db(metrics_db_path: Path):
    conn = await aiosqlite.connect(f'file:{metrics_db_path}?mode=ro', uri=True)
    conn.row_factory = aiosqlite.Row
    yield conn
    await conn.close()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_split_status_offline_returns_empty():
    pairs, queue = _split_status({'offline': True, 'error': 'x'})
    assert pairs == []
    assert queue is None


def test_split_status_extracts_per_project_and_queue():
    payload = {
        'projects': {
            'a': {'graphiti_nodes': 10, 'mem0_memories': 5},
            'b': {'graphiti_nodes': 1, 'mem0_memories': 0},
            'bad': 'not-a-dict',
        },
        'queue': {'counts': {'pending': 2}},
    }
    pairs, queue = _split_status(payload)
    assert {pid for pid, _ in pairs} == {'a', 'b'}
    assert queue == {'counts': {'pending': 2}}


def test_split_queue_stats_offline_returns_nones():
    p, r, d = _split_queue_stats({'offline': True})
    assert (p, r, d) == (None, None, None)


def test_split_queue_stats_pulls_counts():
    p, r, d = _split_queue_stats({'counts': {'pending': 3, 'retry': 1, 'dead': 0}})
    assert (p, r, d) == (3, 1, 0)


# ---------------------------------------------------------------------------
# Read aggregators
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrators_series_handles_none_db():
    series = await get_orchestrators_running_series(None)
    assert series == {'labels': [], 'values': []}


@pytest.mark.asyncio
async def test_orchestrators_series_groups_per_timestamp(metrics_db_path: Path):
    now = datetime.now(UTC)
    conn = sqlite3.connect(str(metrics_db_path))
    for offset, rows in (
        (10, [('proj-a', 1), ('proj-b', 2)]),
        (5, [('proj-a', 3), ('proj-b', 0)]),
    ):
        ts = (now - timedelta(minutes=offset)).isoformat()
        for pid, count in rows:
            conn.execute(
                'INSERT INTO orchestrator_snapshots (ts, project_id, running_count) VALUES (?, ?, ?)',
                (ts, pid, count),
            )
    conn.commit()
    conn.close()

    db = await aiosqlite.connect(f'file:{metrics_db_path}?mode=ro', uri=True)
    db.row_factory = aiosqlite.Row
    try:
        series = await get_orchestrators_running_series(db, days=1)
    finally:
        await db.close()
    assert series['values'] == [3, 3]  # newest first-bucket sum is 3, then 3


@pytest.mark.asyncio
async def test_memory_24h_ago_picks_closest_within_tolerance(metrics_db_path: Path):
    now = datetime.now(UTC)
    target = now - timedelta(hours=24)
    conn = sqlite3.connect(str(metrics_db_path))
    rows = [
        ('proj-a', target - timedelta(minutes=30), 100, 200),  # within ±2h
        ('proj-a', target + timedelta(hours=2, minutes=30), 150, 250),  # outside ±2h
        ('proj-b', now - timedelta(hours=1), 5, 5),  # 23h from target → drop
    ]
    for pid, ts, gn, mm in rows:
        conn.execute(
            'INSERT INTO memory_snapshots (ts, project_id, graphiti_nodes, mem0_memories) '
            'VALUES (?, ?, ?, ?)',
            (ts.isoformat(), pid, gn, mm),
        )
    conn.commit()
    conn.close()

    db = await aiosqlite.connect(f'file:{metrics_db_path}?mode=ro', uri=True)
    db.row_factory = aiosqlite.Row
    try:
        result = await get_memory_24h_ago(db)
    finally:
        await db.close()
    # proj-a has a row 30min before target; in tolerance.
    assert result['proj-a']['graphiti_nodes'] == 100
    # proj-b's only row is far from the target → omitted entirely so UI renders '—'.
    assert 'proj-b' not in result


@pytest.mark.asyncio
async def test_memory_sparks_sums_across_projects(metrics_db_path: Path):
    now = datetime.now(UTC)
    conn = sqlite3.connect(str(metrics_db_path))
    ts = now.isoformat()
    conn.executemany(
        'INSERT INTO memory_snapshots (ts, project_id, graphiti_nodes, mem0_memories) '
        'VALUES (?, ?, ?, ?)',
        [(ts, 'a', 100, 200), (ts, 'b', 50, 0)],
    )
    conn.commit()
    conn.close()

    db = await aiosqlite.connect(f'file:{metrics_db_path}?mode=ro', uri=True)
    db.row_factory = aiosqlite.Row
    try:
        sparks = await get_memory_sparks(db, days=1)
    finally:
        await db.close()
    assert sparks['graphiti_nodes']['values'] == [150]
    assert sparks['mem0_memories']['values'] == [200]


@pytest.mark.asyncio
async def test_recon_and_queue_sparks(metrics_db_path: Path):
    now = datetime.now(UTC)
    conn = sqlite3.connect(str(metrics_db_path))
    ts = now.isoformat()
    conn.execute(
        'INSERT INTO recon_snapshots (ts, buffered_count, active_agents) VALUES (?, ?, ?)',
        (ts, 7, 3),
    )
    conn.execute(
        'INSERT INTO queue_snapshots (ts, pending, retry, dead) VALUES (?, ?, ?, ?)',
        (ts, 2, 0, 0),
    )
    conn.commit()
    conn.close()

    db = await aiosqlite.connect(f'file:{metrics_db_path}?mode=ro', uri=True)
    db.row_factory = aiosqlite.Row
    try:
        recon = await get_recon_sparks(db, days=1)
        queue = await get_queue_pending_series(db, days=1)
    finally:
        await db.close()
    assert recon['buffered_count']['values'] == [7]
    assert recon['active_agents']['values'] == [3]
    assert queue['values'] == [2]


@pytest.mark.asyncio
async def test_merge_active_series_per_project_filter(metrics_db_path: Path):
    now = datetime.now(UTC)
    conn = sqlite3.connect(str(metrics_db_path))
    ts = now.isoformat()
    conn.executemany(
        'INSERT INTO merge_snapshots (ts, project_id, active_count) VALUES (?, ?, ?)',
        [(ts, 'a', 4), (ts, 'b', 1)],
    )
    conn.commit()
    conn.close()

    db = await aiosqlite.connect(f'file:{metrics_db_path}?mode=ro', uri=True)
    db.row_factory = aiosqlite.Row
    try:
        agg = await get_merge_active_series(db, days=1)
        scoped = await get_merge_active_series(db, project_id='a', days=1)
    finally:
        await db.close()
    assert agg['values'] == [5]
    assert scoped['values'] == [4]


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_downsample_keeps_latest_per_hour_after_7d(metrics_db_path: Path):
    now = datetime.now(UTC)
    old = now - timedelta(days=10)
    conn = sqlite3.connect(str(metrics_db_path))
    # Two rows in the same hour, project_id='proj' — the older should be culled.
    conn.execute(
        'INSERT INTO orchestrator_snapshots (ts, project_id, running_count) VALUES (?, ?, ?)',
        ((old + timedelta(minutes=5)).isoformat(), 'proj', 1),
    )
    conn.execute(
        'INSERT INTO orchestrator_snapshots (ts, project_id, running_count) VALUES (?, ?, ?)',
        ((old + timedelta(minutes=55)).isoformat(), 'proj', 9),
    )
    # System-wide tables: two same-hour rows, latest wins.
    conn.execute(
        'INSERT INTO recon_snapshots (ts, buffered_count, active_agents) VALUES (?, ?, ?)',
        ((old + timedelta(minutes=10)).isoformat(), 1, 1),
    )
    conn.execute(
        'INSERT INTO recon_snapshots (ts, buffered_count, active_agents) VALUES (?, ?, ?)',
        ((old + timedelta(minutes=50)).isoformat(), 9, 9),
    )
    conn.commit()
    conn.close()

    rw = await aiosqlite.connect(str(metrics_db_path))
    try:
        await downsample_metrics(rw)
    finally:
        await rw.close()

    inspect = sqlite3.connect(str(metrics_db_path))
    cnt = inspect.execute('SELECT running_count FROM orchestrator_snapshots').fetchall()
    rec = inspect.execute('SELECT buffered_count FROM recon_snapshots').fetchall()
    inspect.close()
    assert cnt == [(9,)]
    assert rec == [(9,)]
