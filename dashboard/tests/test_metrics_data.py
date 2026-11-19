"""Tests for dashboard.data.metrics — schema, samplers, and read aggregators."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import aiosqlite
import httpx
import pytest

from dashboard.data.metrics import (
    METRICS_SCHEMA,
    _split_queue_stats,
    _split_status,
    downsample_metrics,
    get_curator_sparks,
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


def test_split_queue_stats_warns_on_missing_counts_keys(caplog):
    """counts present but missing some of pending/retry/dead -> returns partial values
    AND emits a WARNING from dashboard.data.metrics (shape drift).

    Fails today because the code silently uses counts.get(...) with no warning.
    """
    import logging

    with caplog.at_level(logging.WARNING, logger='dashboard.data.metrics'):
        p, r, d = _split_queue_stats({'counts': {'pending': 3}})

    # Partial values are returned (pending was present).
    assert p == 3
    # Missing keys return None.
    assert r is None
    assert d is None
    # A WARNING must have been emitted.
    warning_records = [rec for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert warning_records, (
        'expected a WARNING for counts missing retry/dead keys, but none was emitted'
    )


def test_split_queue_stats_warns_on_invalid_counts(caplog):
    """counts not a dict (e.g. {'foo': 1} has no 'counts' key) -> (None, None, None)
    AND emits a WARNING from dashboard.data.metrics.

    Fails today because the code silently coerces missing/invalid counts to {} and
    returns (None, None, None) without any warning.
    """
    import logging

    with caplog.at_level(logging.WARNING, logger='dashboard.data.metrics'):
        p, r, d = _split_queue_stats({'foo': 1})

    assert (p, r, d) == (None, None, None)
    warning_records = [rec for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert warning_records, (
        'expected a WARNING for missing/invalid counts, but none was emitted'
    )


def test_split_queue_stats_offline_no_warning(caplog):
    """Offline marker stays silent — no WARNING should be emitted for a known offline state."""
    import logging

    with caplog.at_level(logging.WARNING, logger='dashboard.data.metrics'):
        p, r, d = _split_queue_stats({'offline': True})

    assert (p, r, d) == (None, None, None)
    warning_records = [rec for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert not warning_records, (
        f'expected no WARNINGs for offline marker, got: {[r.message for r in warning_records]}'
    )


def test_split_queue_stats_valid_full_counts_no_warning(caplog):
    """Valid full counts dict -> values returned AND no WARNING emitted."""
    import logging

    with caplog.at_level(logging.WARNING, logger='dashboard.data.metrics'):
        p, r, d = _split_queue_stats({'counts': {'pending': 1, 'retry': 0, 'dead': 0}})

    assert (p, r, d) == (1, 0, 0)
    warning_records = [rec for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert not warning_records, (
        f'expected no WARNINGs for valid counts, got: {[r.message for r in warning_records]}'
    )


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


_EVENTS_SCHEMA = """
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    run_id TEXT NOT NULL,
    task_id TEXT,
    event_type TEXT NOT NULL,
    phase TEXT,
    role TEXT,
    data TEXT DEFAULT '{}',
    cost_usd REAL,
    duration_ms INTEGER
);
"""


def _refusing_transport() -> httpx.MockTransport:
    """Every fused-memory sampler fails fast, so only the merge sampler writes."""
    def _refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError('refused', request=request)
    return httpx.MockTransport(_refuse)


@pytest.mark.asyncio
async def test_the_merge_sampler_records_the_live_probe(tmp_path: Path, monkeypatch):
    """merge_snapshots is the history of the reading the In-queue-now tile shows.

    Only a reachable probe is a measurement: the unreachable project and the
    project with no probe get no row. proj-a's runs.db holds fresh
    merge_queued events, so a sampler still counting the events table would
    record 5, not the probe's 2.
    """
    import dashboard.data.metrics as metrics_mod
    from dashboard.config import DashboardConfig
    from dashboard.data.memory import reset_sessions
    from dashboard.data.metrics import collect_metrics_snapshot

    roots = {name: tmp_path / name for name in ('proj-a', 'proj-b', 'proj-c')}
    for root in roots.values():
        root.mkdir()
    escalation_urls = {'proj-a': 'http://127.0.0.1:9/mcp', 'proj-b': 'http://127.0.0.1:10/mcp'}
    config = DashboardConfig(
        project_root=roots['proj-a'],
        known_project_roots=[roots['proj-b'], roots['proj-c']],
        fused_memory_urls=['http://127.0.0.1:11'],
        escalation_urls=escalation_urls,
    )

    probed: list[dict] = []

    async def _live(client, urls, **_kwargs):
        probed.append(dict(urls))
        entry = {'task_id': '1', 'branch': 'task/1', 'state': 'queued',
                 'age_secs': 3.0, 'position': 0, 'waiter_alive': True}
        return {
            'proj-a': {'entries': [entry, {**entry, 'task_id': '2'}],
                       'reachable': True, 'metrics': None},
            'proj-b': {'entries': [], 'reachable': False, 'error': 'connect refused',
                       'metrics': None},
        }

    monkeypatch.setattr(metrics_mod, 'fetch_live_merge_queues', _live)
    monkeypatch.setattr(metrics_mod, 'find_running_orchestrators', lambda: [])

    runs_path = tmp_path / 'runs.db'
    runs_sync = sqlite3.connect(str(runs_path))
    runs_sync.executescript(_EVENTS_SCHEMA)
    queued_at = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    runs_sync.executemany(
        "INSERT INTO events (timestamp, run_id, task_id, event_type) VALUES (?, ?, ?, 'merge_queued')",
        [(queued_at, f'run-{i}', str(i)) for i in range(5)],
    )
    runs_sync.commit()
    runs_sync.close()

    metrics_conn = await aiosqlite.connect(str(tmp_path / 'metrics.db'))
    await metrics_conn.executescript(METRICS_SCHEMA)
    await metrics_conn.commit()
    reset_sessions()
    try:
        async with httpx.AsyncClient(transport=_refusing_transport()) as http_client:
            runs_conn = await aiosqlite.connect(str(runs_path))
            runs_conn.row_factory = aiosqlite.Row
            try:
                await collect_metrics_snapshot(
                    conn=metrics_conn,
                    config=config,
                    http_client=http_client,
                    recon_db=None,
                    merge_dbs=[
                        (str(roots['proj-a']), runs_conn),
                        (str(roots['proj-b']), None),
                        (str(roots['proj-c']), None),
                    ],
                )
            finally:
                await runs_conn.close()

        async with metrics_conn.execute(
            'SELECT project_id, active_count FROM merge_snapshots'
        ) as cur:
            rows = [tuple(row) for row in await cur.fetchall()]
        series = await get_merge_active_series(
            metrics_conn, project_id=str(roots['proj-a']), now=datetime.now(UTC),
        )
    finally:
        await metrics_conn.close()
        reset_sessions()

    assert probed == [escalation_urls], 'one probe, over the configured URLs'
    assert rows == [(str(roots['proj-a']), 2)]
    assert series['values'][-1] == 2


# ---------------------------------------------------------------------------
# now-threading (task 2281) — mirrors test_costs_data.py Test_Cutoff:
# fixed-now determinism per function, plus one no-now real-clock bracket.
#
# FIXED_NOW is anchored months away from the real wall clock on purpose: if a
# function silently ignored its `now=` argument and fell back to
# datetime.now(UTC) internally, the "inside window" row (offset from
# FIXED_NOW) would land far outside whatever cutoff the real clock produces
# and the assertions below would fail loudly instead of passing by accident.
# ---------------------------------------------------------------------------

FIXED_NOW = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_orchestrators_series_uses_provided_now(metrics_db_path: Path):
    conn = sqlite3.connect(str(metrics_db_path))
    inside = (FIXED_NOW - timedelta(hours=12)).isoformat()
    outside = (FIXED_NOW - timedelta(days=2)).isoformat()
    conn.execute(
        'INSERT INTO orchestrator_snapshots (ts, project_id, running_count) VALUES (?, ?, ?)',
        (inside, 'proj-a', 4),
    )
    conn.execute(
        'INSERT INTO orchestrator_snapshots (ts, project_id, running_count) VALUES (?, ?, ?)',
        (outside, 'proj-a', 99),
    )
    conn.commit()
    conn.close()

    db = await aiosqlite.connect(f'file:{metrics_db_path}?mode=ro', uri=True)
    db.row_factory = aiosqlite.Row
    try:
        series = await get_orchestrators_running_series(db, days=1, now=FIXED_NOW)
    finally:
        await db.close()
    assert series['values'] == [4]


@pytest.mark.asyncio
async def test_memory_sparks_uses_provided_now(metrics_db_path: Path):
    conn = sqlite3.connect(str(metrics_db_path))
    inside = (FIXED_NOW - timedelta(hours=12)).isoformat()
    outside = (FIXED_NOW - timedelta(days=2)).isoformat()
    conn.executemany(
        'INSERT INTO memory_snapshots (ts, project_id, graphiti_nodes, mem0_memories) '
        'VALUES (?, ?, ?, ?)',
        [(inside, 'a', 10, 20), (outside, 'a', 999, 999)],
    )
    conn.commit()
    conn.close()

    db = await aiosqlite.connect(f'file:{metrics_db_path}?mode=ro', uri=True)
    db.row_factory = aiosqlite.Row
    try:
        sparks = await get_memory_sparks(db, days=1, now=FIXED_NOW)
    finally:
        await db.close()
    assert sparks['graphiti_nodes']['values'] == [10]
    assert sparks['mem0_memories']['values'] == [20]


@pytest.mark.asyncio
async def test_memory_24h_ago_uses_provided_now(metrics_db_path: Path):
    target = FIXED_NOW - timedelta(hours=24)
    conn = sqlite3.connect(str(metrics_db_path))
    conn.execute(
        'INSERT INTO memory_snapshots (ts, project_id, graphiti_nodes, mem0_memories) '
        'VALUES (?, ?, ?, ?)',
        ((target - timedelta(minutes=30)).isoformat(), 'proj-a', 100, 200),
    )
    # Anchored to the *real* current clock — this is the row that would win
    # "closest to 24h ago" if the function ignored `now=FIXED_NOW` and read
    # datetime.now(UTC) internally instead.
    conn.execute(
        'INSERT INTO memory_snapshots (ts, project_id, graphiti_nodes, mem0_memories) '
        'VALUES (?, ?, ?, ?)',
        ((datetime.now(UTC) - timedelta(hours=24)).isoformat(), 'proj-a', 999, 999),
    )
    conn.commit()
    conn.close()

    db = await aiosqlite.connect(f'file:{metrics_db_path}?mode=ro', uri=True)
    db.row_factory = aiosqlite.Row
    try:
        result = await get_memory_24h_ago(db, now=FIXED_NOW)
    finally:
        await db.close()
    assert result['proj-a']['graphiti_nodes'] == 100


_TARGET_24H_AGO = FIXED_NOW - timedelta(hours=24)
_TOLERANCE = timedelta(hours=2)
_NEAR_TARGET_PROJECTS = [f'proj-{p}' for p in range(5)]


def _insert_memory_snapshots(
    db_path: Path, rows: list[tuple[str, datetime, int, int]]
) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.executemany(
        'INSERT INTO memory_snapshots (ts, project_id, graphiti_nodes, mem0_memories) '
        'VALUES (?, ?, ?, ?)',
        [(ts.isoformat(), pid, gn, mm) for pid, ts, gn, mm in rows],
    )
    conn.commit()
    conn.close()


def _rows_near_the_24h_target() -> list[tuple[str, datetime, int, int]]:
    return [
        (pid, _TARGET_24H_AGO + timedelta(minutes=offset), p * 100 + offset, p * 200 + offset)
        for p, pid in enumerate(_NEAR_TARGET_PROJECTS)
        for offset in (-90, -10, 40)
    ]


def _rows_outside_the_tolerance_window() -> list[tuple[str, datetime, int, int]]:
    older = [
        _TARGET_24H_AGO - timedelta(days=3) - i * (timedelta(days=86) / 200)
        for i in range(200)
    ]
    newer = [
        _TARGET_24H_AGO + timedelta(hours=3) + i * (timedelta(hours=21) / 200)
        for i in range(200)
    ]
    return [(pid, ts, 999, 999) for pid in _NEAR_TARGET_PROJECTS for ts in older + newer]


async def _memory_24h_ago_and_vm_steps(
    db_path: Path, now: datetime
) -> tuple[dict[str, dict], int]:
    steps = 0

    def count_step() -> int:
        nonlocal steps
        steps += 1
        return 0

    db = await aiosqlite.connect(f'file:{db_path}?mode=ro', uri=True)
    try:
        await db.set_progress_handler(count_step, 1)
        result = await get_memory_24h_ago(db, now=now)
    finally:
        await db.close()
    return result, steps


@pytest.mark.asyncio
async def test_memory_24h_ago_cost_does_not_grow_with_out_of_window_history(
    metrics_db_path: Path,
):
    _insert_memory_snapshots(metrics_db_path, _rows_near_the_24h_target())
    near_only, near_only_steps = await _memory_24h_ago_and_vm_steps(metrics_db_path, FIXED_NOW)

    _insert_memory_snapshots(metrics_db_path, _rows_outside_the_tolerance_window())
    with_history, with_history_steps = await _memory_24h_ago_and_vm_steps(
        metrics_db_path, FIXED_NOW
    )

    assert near_only == {
        pid: {'graphiti_nodes': p * 100 - 10, 'mem0_memories': p * 200 - 10}
        for p, pid in enumerate(_NEAR_TARGET_PROJECTS)
    }
    assert with_history == near_only
    assert with_history_steps <= 1.5 * near_only_steps, (near_only_steps, with_history_steps)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'now',
    [FIXED_NOW, FIXED_NOW + timedelta(milliseconds=500)],
    ids=['whole-second-now', 'sub-second-now'],
)
async def test_memory_24h_ago_keeps_rows_at_the_exact_tolerance_edges(
    metrics_db_path: Path, now: datetime
):
    _insert_memory_snapshots(
        metrics_db_path,
        [
            ('at-lower-edge', _TARGET_24H_AGO - _TOLERANCE, 1, 1),
            ('at-upper-edge', _TARGET_24H_AGO + _TOLERANCE, 2, 2),
            ('past-upper-edge', _TARGET_24H_AGO + _TOLERANCE + timedelta(seconds=1), 3, 3),
        ],
    )

    result, _ = await _memory_24h_ago_and_vm_steps(metrics_db_path, now)

    assert set(result) == {'at-lower-edge', 'at-upper-edge'}


@pytest.mark.asyncio
async def test_memory_24h_ago_reads_a_non_utc_now_as_the_same_instant(metrics_db_path: Path):
    _insert_memory_snapshots(metrics_db_path, _rows_near_the_24h_target())
    now_at_plus_five = FIXED_NOW.astimezone(timezone(timedelta(hours=5)))

    from_utc, _ = await _memory_24h_ago_and_vm_steps(metrics_db_path, FIXED_NOW)
    from_plus_five, _ = await _memory_24h_ago_and_vm_steps(metrics_db_path, now_at_plus_five)

    assert from_utc
    assert from_plus_five == from_utc


@pytest.mark.asyncio
async def test_queue_pending_series_uses_provided_now(metrics_db_path: Path):
    conn = sqlite3.connect(str(metrics_db_path))
    inside = (FIXED_NOW - timedelta(hours=12)).isoformat()
    outside = (FIXED_NOW - timedelta(days=2)).isoformat()
    conn.execute(
        'INSERT INTO queue_snapshots (ts, pending, retry, dead) VALUES (?, ?, ?, ?)',
        (inside, 7, 1, 0),
    )
    conn.execute(
        'INSERT INTO queue_snapshots (ts, pending, retry, dead) VALUES (?, ?, ?, ?)',
        (outside, 999, 0, 0),
    )
    conn.commit()
    conn.close()

    db = await aiosqlite.connect(f'file:{metrics_db_path}?mode=ro', uri=True)
    db.row_factory = aiosqlite.Row
    try:
        series = await get_queue_pending_series(db, days=1, now=FIXED_NOW)
    finally:
        await db.close()
    assert series['values'] == [7]


@pytest.mark.asyncio
async def test_queue_pending_series_no_now_brackets_real_clock(metrics_db_path: Path):
    """Without an explicit `now`, the days=1 cutoff still derives from the real
    UTC clock (via resolve_now), not a frozen/ignored value.

    Mirrors Test_Cutoff.test_cutoff_no_now_uses_current_time: brackets the
    write with a real `datetime.now(UTC)` read and a small tolerance rather
    than freezing time.
    """
    before = datetime.now(UTC)
    conn = sqlite3.connect(str(metrics_db_path))
    inside = (before - timedelta(days=1) + timedelta(minutes=5)).isoformat()
    outside = (before - timedelta(days=2)).isoformat()
    conn.execute(
        'INSERT INTO queue_snapshots (ts, pending, retry, dead) VALUES (?, ?, ?, ?)',
        (inside, 42, 0, 0),
    )
    conn.execute(
        'INSERT INTO queue_snapshots (ts, pending, retry, dead) VALUES (?, ?, ?, ?)',
        (outside, 99, 0, 0),
    )
    conn.commit()
    conn.close()

    db = await aiosqlite.connect(f'file:{metrics_db_path}?mode=ro', uri=True)
    db.row_factory = aiosqlite.Row
    try:
        series = await get_queue_pending_series(db, days=1)
    finally:
        await db.close()
    assert series['values'] == [42]


@pytest.mark.asyncio
async def test_recon_sparks_uses_provided_now(metrics_db_path: Path):
    conn = sqlite3.connect(str(metrics_db_path))
    inside = (FIXED_NOW - timedelta(hours=12)).isoformat()
    outside = (FIXED_NOW - timedelta(days=2)).isoformat()
    conn.execute(
        'INSERT INTO recon_snapshots (ts, buffered_count, active_agents) VALUES (?, ?, ?)',
        (inside, 5, 2),
    )
    conn.execute(
        'INSERT INTO recon_snapshots (ts, buffered_count, active_agents) VALUES (?, ?, ?)',
        (outside, 999, 999),
    )
    conn.commit()
    conn.close()

    db = await aiosqlite.connect(f'file:{metrics_db_path}?mode=ro', uri=True)
    db.row_factory = aiosqlite.Row
    try:
        recon = await get_recon_sparks(db, days=1, now=FIXED_NOW)
    finally:
        await db.close()
    assert recon['buffered_count']['values'] == [5]
    assert recon['active_agents']['values'] == [2]


@pytest.mark.asyncio
async def test_curator_sparks_uses_provided_now(metrics_db_path: Path):
    conn = sqlite3.connect(str(metrics_db_path))
    inside = (FIXED_NOW - timedelta(hours=12)).isoformat()
    outside = (FIXED_NOW - timedelta(days=2)).isoformat()
    conn.execute(
        'INSERT INTO curator_snapshots '
        '(ts, pending_total, capped_now, p50_active_ms, p90_active_ms, p99_active_ms) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        (inside, 3, 0, 100, 200, 300),
    )
    conn.execute(
        'INSERT INTO curator_snapshots '
        '(ts, pending_total, capped_now, p50_active_ms, p90_active_ms, p99_active_ms) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        (outside, 999, 1, 999, 999, 999),
    )
    conn.commit()
    conn.close()

    db = await aiosqlite.connect(f'file:{metrics_db_path}?mode=ro', uri=True)
    db.row_factory = aiosqlite.Row
    try:
        sparks = await get_curator_sparks(db, days=1, now=FIXED_NOW)
    finally:
        await db.close()
    assert sparks['pending']['values'] == [3]
    assert sparks['p50']['values'] == [100]


@pytest.mark.asyncio
async def test_merge_active_series_uses_provided_now(metrics_db_path: Path):
    conn = sqlite3.connect(str(metrics_db_path))
    inside = (FIXED_NOW - timedelta(hours=12)).isoformat()
    outside = (FIXED_NOW - timedelta(days=2)).isoformat()
    conn.executemany(
        'INSERT INTO merge_snapshots (ts, project_id, active_count) VALUES (?, ?, ?)',
        [(inside, 'a', 4), (outside, 'a', 999)],
    )
    conn.commit()
    conn.close()

    db = await aiosqlite.connect(f'file:{metrics_db_path}?mode=ro', uri=True)
    db.row_factory = aiosqlite.Row
    try:
        series = await get_merge_active_series(db, days=1, now=FIXED_NOW)
    finally:
        await db.close()
    assert series['values'] == [4]


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------


def _metrics_table_names() -> list[str]:
    """Every table METRICS_SCHEMA creates, so a table left out of downsampling fails."""
    conn = sqlite3.connect(':memory:')
    try:
        conn.executescript(METRICS_SCHEMA)
        return [name for (name,) in conn.execute(
            "SELECT name FROM sqlite_master"
            " WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )]
    finally:
        conn.close()


def _not_null_by_column(conn: sqlite3.Connection, table: str) -> dict[str, bool]:
    return {row[1]: bool(row[3]) for row in conn.execute(f'PRAGMA table_info({table})')}


def _insert_metrics_row(
    conn: sqlite3.Connection,
    table: str,
    not_null_by_column: dict[str, bool],
    project_id: str | None,
    ts: str,
) -> None:
    values: dict[str, object] = {'ts': ts}
    if 'project_id' in not_null_by_column:
        values['project_id'] = project_id
    for column, not_null in not_null_by_column.items():
        if not_null and column not in values:
            values[column] = 0
    conn.execute(
        f'INSERT INTO {table} ({", ".join(values)}) VALUES ({", ".join("?" * len(values))})',
        tuple(values.values()),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('table', _metrics_table_names())
async def test_downsample_keeps_one_row_per_old_hour(metrics_db_path: Path, table: str):
    later_old_hour = (FIXED_NOW - timedelta(days=10)).replace(minute=0, second=0, microsecond=0)
    old_hours = (later_old_hour - timedelta(hours=1), later_old_hour)
    conn = sqlite3.connect(str(metrics_db_path))
    not_null_by_column = _not_null_by_column(conn, table)
    per_project = 'project_id' in not_null_by_column
    projects = ('a', 'b') if per_project else (None,)
    recent = (projects[0], (FIXED_NOW - timedelta(hours=1)).isoformat())
    for project in projects:
        for hour in old_hours:
            for minute in (10, 50):
                ts = (hour + timedelta(minutes=minute)).isoformat()
                _insert_metrics_row(conn, table, not_null_by_column, project, ts)
    expired_ts = (FIXED_NOW - timedelta(days=100)).isoformat()
    _insert_metrics_row(conn, table, not_null_by_column, projects[0], expired_ts)
    _insert_metrics_row(conn, table, not_null_by_column, *recent)
    conn.commit()
    conn.close()

    rw = await aiosqlite.connect(str(metrics_db_path))
    try:
        await downsample_metrics(rw, now=FIXED_NOW)
    finally:
        await rw.close()

    project_column = 'project_id' if per_project else 'NULL'
    inspect = sqlite3.connect(str(metrics_db_path))
    got = inspect.execute(
        f'SELECT {project_column}, ts FROM {table} ORDER BY ts, {project_column}'
    ).fetchall()
    inspect.close()
    newest_of_each_old_hour = [
        (project, (hour + timedelta(minutes=50)).isoformat())
        for hour in old_hours
        for project in projects
    ]
    expected = [*newest_of_each_old_hour, recent]
    assert got == expected, f'{table}: expected survivors {expected}, got {got}'
