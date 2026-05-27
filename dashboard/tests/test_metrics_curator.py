"""Tests for curator_snapshots table + metrics sampler + reader.

Built incrementally — one step at a time in TDD order.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import httpx
import pytest

from dashboard.app import _metrics_loop, _MetricsStore
from dashboard.config import DashboardConfig
from dashboard.data.db import DbPool
from dashboard.data.metrics import (
    METRICS_SCHEMA,
    collect_metrics_snapshot,
    downsample_metrics,
)

# ---------------------------------------------------------------------------
# Shared schemas (minimal for tests)
# ---------------------------------------------------------------------------

TICKETS_SCHEMA = """\
CREATE TABLE IF NOT EXISTS tickets (
    id          TEXT PRIMARY KEY,
    project_id  TEXT,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    resolved_at TEXT
);
"""

ACCOUNT_EVENTS_SCHEMA = """\
CREATE TABLE IF NOT EXISTS account_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    account_name TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    project_id   TEXT,
    run_id       TEXT,
    details      TEXT,
    created_at   TEXT NOT NULL
);
"""

# ---------------------------------------------------------------------------
# MCP mock helpers (mirrors test_memory.py pattern)
# ---------------------------------------------------------------------------


def _make_mcp_response(inner_dict: dict, request_id: int = 1) -> httpx.Response:
    body = {
        'jsonrpc': '2.0',
        'id': request_id,
        'result': {
            'content': [
                {'type': 'text', 'text': json.dumps(inner_dict)},
            ],
        },
    }
    return httpx.Response(
        200,
        json=body,
        headers={'mcp-session-id': 'test-session-id'},
    )


def _make_init_response(request_id: int = 1) -> httpx.Response:
    body = {
        'jsonrpc': '2.0',
        'id': request_id,
        'result': {
            'protocolVersion': '2025-03-26',
            'capabilities': {'tools': {}},
            'serverInfo': {'name': 'test', 'version': '0.1'},
        },
    }
    return httpx.Response(
        200,
        json=body,
        headers={'mcp-session-id': 'test-session-id'},
    )


def _make_notify_response() -> httpx.Response:
    return httpx.Response(202, headers={'mcp-session-id': 'test-session-id'})


class _ListTicketsHandler:
    """Mock MCP handler that returns a fixed count for list_tickets calls."""

    def __init__(self, count: int = 2, project_id: str = 'test-project'):
        self.count = count
        self.project_id = project_id
        self.calls: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        method = body.get('method', '')
        request_id = body.get('id', 1)

        if method == 'initialize':
            return _make_init_response(request_id)
        if method.startswith('notifications/'):
            return _make_notify_response()

        # tools/call
        self.calls.append(body)
        return _make_mcp_response(
            {'count': self.count, 'tickets': [], 'project_id': self.project_id},
            request_id,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def metrics_db_path(tmp_path: Path) -> Path:
    db_path = tmp_path / 'metrics.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(METRICS_SCHEMA)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def tickets_db_path(tmp_path: Path) -> Path:
    db_path = tmp_path / 'tickets.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(TICKETS_SCHEMA)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def runs_db_path(tmp_path: Path) -> Path:
    db_path = tmp_path / 'runs.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(ACCOUNT_EVENTS_SCHEMA)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def config(tmp_path: Path):
    from dashboard.config import DashboardConfig

    return DashboardConfig(
        project_root=tmp_path,
        fused_memory_urls=['http://localhost:18765'],
        known_project_roots=[tmp_path],
    )


# ---------------------------------------------------------------------------
# Step-1: Schema migration — curator_snapshots exists after executescript
# ---------------------------------------------------------------------------


def test_curator_snapshots_table_exists_after_schema_migration(tmp_path: Path):
    """curator_snapshots must be created by METRICS_SCHEMA."""
    db_path = tmp_path / 'test_schema.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(METRICS_SCHEMA)
    conn.commit()

    # Check table exists
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='curator_snapshots'"
    ).fetchone()
    assert row is not None, 'curator_snapshots table was not created by METRICS_SCHEMA'

    # Check expected columns
    col_info = conn.execute('PRAGMA table_info(curator_snapshots)').fetchall()
    cols = {r[1] for r in col_info}
    expected = {
        'ts',
        'pending_total',
        'capped_now',
        'p50_active_ms',
        'p90_active_ms',
        'p99_active_ms',
    }
    assert expected <= cols, f'Missing columns: {expected - cols}'

    # Check ts is the PRIMARY KEY (pk=1 in PRAGMA table_info)
    pk_cols = {r[1] for r in col_info if r[5] == 1}
    assert 'ts' in pk_cols, 'ts is not the PRIMARY KEY of curator_snapshots'

    conn.close()


# ---------------------------------------------------------------------------
# Step-3: get_curator_sparks(None) returns 4-key empty shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_curator_sparks_none_returns_empty_shape():
    """get_curator_sparks(None) must return 4-key empty shape."""
    from dashboard.data.metrics import get_curator_sparks

    result = await get_curator_sparks(None, days=1)
    assert set(result.keys()) == {'pending', 'p50', 'p90', 'p99'}
    for key in ('pending', 'p50', 'p90', 'p99'):
        assert result[key] == {'labels': [], 'values': []}, (
            f"Key '{key}' expected empty series, got {result[key]}"
        )


# ---------------------------------------------------------------------------
# Step-5: get_curator_sparks with rows — filters by days, ordered by ts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_curator_sparks_with_rows(metrics_db_path: Path):
    """get_curator_sparks returns only in-window rows, ordered by ts."""
    from dashboard.data.metrics import get_curator_sparks

    now = datetime.now(UTC)
    in_window_ts = (now - timedelta(hours=6)).isoformat()
    old_ts = (now - timedelta(days=5)).isoformat()

    conn_sync = sqlite3.connect(str(metrics_db_path))
    conn_sync.execute(
        'INSERT INTO curator_snapshots '
        '(ts, pending_total, capped_now, p50_active_ms, p90_active_ms, p99_active_ms) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        (in_window_ts, 3, 0, 100, 200, 300),
    )
    conn_sync.execute(
        'INSERT INTO curator_snapshots '
        '(ts, pending_total, capped_now, p50_active_ms, p90_active_ms, p99_active_ms) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        (old_ts, 99, 1, 999, 999, 999),
    )
    conn_sync.commit()
    conn_sync.close()

    db = await aiosqlite.connect(f'file:{metrics_db_path}?mode=ro', uri=True)
    db.row_factory = aiosqlite.Row
    try:
        result = await get_curator_sparks(db, days=1)
    finally:
        await db.close()

    assert set(result.keys()) == {'pending', 'p50', 'p90', 'p99'}
    assert result['pending']['values'] == [3]
    assert result['p50']['values'] == [100]
    assert result['p90']['values'] == [200]
    assert result['p99']['values'] == [300]
    assert result['pending']['labels'] == [in_window_ts]


@pytest.mark.asyncio
async def test_get_curator_sparks_order_by_ts(metrics_db_path: Path):
    """get_curator_sparks returns rows ordered by ts ascending."""
    from dashboard.data.metrics import get_curator_sparks

    now = datetime.now(UTC)
    ts1 = (now - timedelta(hours=10)).isoformat()
    ts2 = (now - timedelta(hours=5)).isoformat()

    conn_sync = sqlite3.connect(str(metrics_db_path))
    # Insert in reverse order to test ORDER BY
    conn_sync.execute(
        'INSERT INTO curator_snapshots '
        '(ts, pending_total, capped_now, p50_active_ms, p90_active_ms, p99_active_ms) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        (ts2, 2, 0, None, None, None),
    )
    conn_sync.execute(
        'INSERT INTO curator_snapshots '
        '(ts, pending_total, capped_now, p50_active_ms, p90_active_ms, p99_active_ms) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        (ts1, 1, 0, None, None, None),
    )
    conn_sync.commit()
    conn_sync.close()

    db = await aiosqlite.connect(f'file:{metrics_db_path}?mode=ro', uri=True)
    db.row_factory = aiosqlite.Row
    try:
        result = await get_curator_sparks(db, days=2)
    finally:
        await db.close()

    assert result['pending']['values'] == [1, 2]  # ascending ts order


# ---------------------------------------------------------------------------
# Step-7: downsample_metrics — curator_snapshots respects downsampling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_downsample_curator_snapshots_keeps_latest_per_hour(metrics_db_path: Path):
    """After downsample_metrics, only the latest-per-hour row survives for >7d-old rows."""
    now = datetime.now(UTC)
    old = now - timedelta(days=10)

    conn_sync = sqlite3.connect(str(metrics_db_path))
    ts_early = (old + timedelta(minutes=5)).isoformat()
    ts_late = (old + timedelta(minutes=55)).isoformat()
    conn_sync.execute(
        'INSERT INTO curator_snapshots '
        '(ts, pending_total, capped_now, p50_active_ms, p90_active_ms, p99_active_ms) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        (ts_early, 1, 0, None, None, None),
    )
    conn_sync.execute(
        'INSERT INTO curator_snapshots '
        '(ts, pending_total, capped_now, p50_active_ms, p90_active_ms, p99_active_ms) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        (ts_late, 9, 0, None, None, None),
    )
    # Very old row (100 days) — should be dropped entirely
    very_old = (now - timedelta(days=100)).isoformat()
    conn_sync.execute(
        'INSERT INTO curator_snapshots '
        '(ts, pending_total, capped_now, p50_active_ms, p90_active_ms, p99_active_ms) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        (very_old, 99, 1, None, None, None),
    )
    conn_sync.commit()
    conn_sync.close()

    rw = await aiosqlite.connect(str(metrics_db_path))
    try:
        await downsample_metrics(rw)
    finally:
        await rw.close()

    inspect = sqlite3.connect(str(metrics_db_path))
    rows = inspect.execute('SELECT pending_total FROM curator_snapshots ORDER BY ts').fetchall()
    inspect.close()

    assert rows == [(9,)], f'Expected [(9,)], got {rows}'


# ---------------------------------------------------------------------------
# Step-9: DashboardConfig.tickets_db property
# ---------------------------------------------------------------------------


def test_dashboard_config_tickets_db_property(tmp_path: Path):
    """tickets_db property returns <project_root>/data/reconciliation/tickets.db."""
    from dashboard.config import DashboardConfig

    config = DashboardConfig(project_root=tmp_path)
    expected = tmp_path.resolve() / 'data' / 'reconciliation' / 'tickets.db'
    assert config.tickets_db == expected


# ---------------------------------------------------------------------------
# Step-11: _sample_curator happy-path test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sample_curator_happy_path(tmp_path: Path, config):
    """_sample_curator returns correct pending_total, capped_now=0, and centiles."""
    from dashboard.data.memory import reset_sessions
    from dashboard.data.metrics import _sample_curator
    from dashboard.data.stats_utils import percentile

    # Use real current time so read_cap_intervals' 1-day internal cutoff is satisfied
    now = datetime.now(UTC)
    t_base = now - timedelta(minutes=30)

    # tickets.db: 3 terminal tickets with durations 100ms, 200ms, 300ms
    tickets_path = tmp_path / 'tickets.db'
    conn_sync = sqlite3.connect(str(tickets_path))
    conn_sync.executescript(TICKETS_SCHEMA)
    conn_sync.execute(
        'INSERT INTO tickets (id, project_id, status, created_at, resolved_at) VALUES (?, ?, ?, ?, ?)',
        (
            't1',
            'proj',
            'created',
            t_base.isoformat(),
            (t_base + timedelta(milliseconds=100)).isoformat(),
        ),
    )
    conn_sync.execute(
        'INSERT INTO tickets (id, project_id, status, created_at, resolved_at) VALUES (?, ?, ?, ?, ?)',
        (
            't2',
            'proj',
            'combined',
            t_base.isoformat(),
            (t_base + timedelta(milliseconds=200)).isoformat(),
        ),
    )
    conn_sync.execute(
        'INSERT INTO tickets (id, project_id, status, created_at, resolved_at) VALUES (?, ?, ?, ?, ?)',
        (
            't3',
            'proj',
            'failed',
            t_base.isoformat(),
            (t_base + timedelta(milliseconds=300)).isoformat(),
        ),
    )
    # Ticket D: too old (> 1 hour ago), should not be counted
    old_t = now - timedelta(hours=2)
    conn_sync.execute(
        'INSERT INTO tickets (id, project_id, status, created_at, resolved_at) VALUES (?, ?, ?, ?, ?)',
        (
            't4',
            'proj',
            'failed',
            old_t.isoformat(),
            (old_t + timedelta(milliseconds=500)).isoformat(),
        ),
    )
    conn_sync.commit()
    conn_sync.close()

    # runs.db: no cap events
    runs_path = tmp_path / 'runs.db'
    conn_sync = sqlite3.connect(str(runs_path))
    conn_sync.executescript(ACCOUNT_EVENTS_SCHEMA)
    conn_sync.commit()
    conn_sync.close()

    handler = _ListTicketsHandler(count=2)
    transport = httpx.MockTransport(handler)

    reset_sessions()
    async with httpx.AsyncClient(transport=transport) as http_client:
        tickets_conn = await aiosqlite.connect(str(tickets_path))
        tickets_conn.row_factory = aiosqlite.Row
        runs_conn = await aiosqlite.connect(str(runs_path))
        runs_conn.row_factory = aiosqlite.Row
        try:
            result = await _sample_curator(
                http_client,
                config,
                tickets_conn,
                [runs_conn],
                now=now,
            )
        finally:
            await tickets_conn.close()
            await runs_conn.close()

    assert result is not None
    # One fused_memory_url × one unique project_root (tmp_path) → count=2
    assert result['pending_total'] == 2
    assert result['capped_now'] == 0
    assert result['p50_active_ms'] == int(percentile([100.0, 200.0, 300.0], 50))
    assert result['p90_active_ms'] == int(percentile([100.0, 200.0, 300.0], 90))
    assert result['p99_active_ms'] == int(percentile([100.0, 200.0, 300.0], 99))


# ---------------------------------------------------------------------------
# Step-13: Cap-overlap subtraction test for _sample_curator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sample_curator_cap_overlap_subtraction(tmp_path: Path, config):
    """active_ms is reduced by capped overlap time."""
    from dashboard.data.memory import reset_sessions
    from dashboard.data.metrics import _sample_curator

    # Use real current time so read_cap_intervals' 1-day internal cutoff is satisfied
    now = datetime.now(UTC)
    t_base = now - timedelta(minutes=30)

    # One ticket: 300ms duration
    tickets_path = tmp_path / 'tickets.db'
    conn_sync = sqlite3.connect(str(tickets_path))
    conn_sync.executescript(TICKETS_SCHEMA)
    ticket_start = t_base
    ticket_end = t_base + timedelta(milliseconds=300)
    conn_sync.execute(
        'INSERT INTO tickets (id, project_id, status, created_at, resolved_at) VALUES (?, ?, ?, ?, ?)',
        ('t1', 'proj', 'created', ticket_start.isoformat(), ticket_end.isoformat()),
    )
    conn_sync.commit()
    conn_sync.close()

    # runs.db: cap_hit + resumed covering 100ms of ticket's window
    runs_path = tmp_path / 'runs.db'
    conn_sync = sqlite3.connect(str(runs_path))
    conn_sync.executescript(ACCOUNT_EVENTS_SCHEMA)
    cap_start = t_base + timedelta(milliseconds=50)
    cap_end = t_base + timedelta(milliseconds=150)  # 100ms of overlap
    conn_sync.execute(
        'INSERT INTO account_events (account_name, event_type, created_at) VALUES (?, ?, ?)',
        ('acc1', 'cap_hit', cap_start.isoformat()),
    )
    conn_sync.execute(
        'INSERT INTO account_events (account_name, event_type, created_at) VALUES (?, ?, ?)',
        ('acc1', 'resumed', cap_end.isoformat()),
    )
    conn_sync.commit()
    conn_sync.close()

    handler = _ListTicketsHandler(count=0)
    transport = httpx.MockTransport(handler)

    reset_sessions()
    async with httpx.AsyncClient(transport=transport) as http_client:
        tickets_conn = await aiosqlite.connect(str(tickets_path))
        tickets_conn.row_factory = aiosqlite.Row
        runs_conn = await aiosqlite.connect(str(runs_path))
        runs_conn.row_factory = aiosqlite.Row
        try:
            result = await _sample_curator(
                http_client,
                config,
                tickets_conn,
                [runs_conn],
                now=now,
            )
        finally:
            await tickets_conn.close()
            await runs_conn.close()

    assert result is not None
    # ticket active_ms = 300ms - 100ms (overlap) = 200ms
    assert result['p50_active_ms'] == 200
    assert result['p90_active_ms'] == 200
    assert result['p99_active_ms'] == 200


# ---------------------------------------------------------------------------
# Step-15: capped_now=1/0 tests for _sample_curator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sample_curator_capped_now_true(tmp_path: Path, config):
    """capped_now=1 when merge_all_accounts_capped yields an open-ended window."""
    from dashboard.data.memory import reset_sessions
    from dashboard.data.metrics import _sample_curator

    # Use real current time so read_cap_intervals' 1-day internal cutoff is satisfied
    now = datetime.now(UTC)

    tickets_path = tmp_path / 'tickets.db'
    conn_sync = sqlite3.connect(str(tickets_path))
    conn_sync.executescript(TICKETS_SCHEMA)
    conn_sync.commit()
    conn_sync.close()

    # one cap_hit 30min ago, no resumed → open-ended interval
    runs_path = tmp_path / 'runs.db'
    conn_sync = sqlite3.connect(str(runs_path))
    conn_sync.executescript(ACCOUNT_EVENTS_SCHEMA)
    cap_time = now - timedelta(minutes=30)
    conn_sync.execute(
        'INSERT INTO account_events (account_name, event_type, created_at) VALUES (?, ?, ?)',
        ('acc1', 'cap_hit', cap_time.isoformat()),
    )
    conn_sync.commit()
    conn_sync.close()

    handler = _ListTicketsHandler(count=0)
    transport = httpx.MockTransport(handler)

    reset_sessions()
    async with httpx.AsyncClient(transport=transport) as http_client:
        tickets_conn = await aiosqlite.connect(str(tickets_path))
        tickets_conn.row_factory = aiosqlite.Row
        runs_conn = await aiosqlite.connect(str(runs_path))
        runs_conn.row_factory = aiosqlite.Row
        try:
            result = await _sample_curator(
                http_client,
                config,
                tickets_conn,
                [runs_conn],
                now=now,
            )
        finally:
            await tickets_conn.close()
            await runs_conn.close()

    assert result is not None
    assert result['capped_now'] == 1


@pytest.mark.asyncio
async def test_sample_curator_capped_now_false_when_closed(tmp_path: Path, config):
    """capped_now=0 when cap_hit + resumed pair (closed interval)."""
    from dashboard.data.memory import reset_sessions
    from dashboard.data.metrics import _sample_curator

    # Use real current time so read_cap_intervals' 1-day internal cutoff is satisfied
    now = datetime.now(UTC)

    tickets_path = tmp_path / 'tickets.db'
    conn_sync = sqlite3.connect(str(tickets_path))
    conn_sync.executescript(TICKETS_SCHEMA)
    conn_sync.commit()
    conn_sync.close()

    runs_path = tmp_path / 'runs.db'
    conn_sync = sqlite3.connect(str(runs_path))
    conn_sync.executescript(ACCOUNT_EVENTS_SCHEMA)
    cap_time = now - timedelta(minutes=30)
    resumed_time = now - timedelta(minutes=10)
    conn_sync.execute(
        'INSERT INTO account_events (account_name, event_type, created_at) VALUES (?, ?, ?)',
        ('acc1', 'cap_hit', cap_time.isoformat()),
    )
    conn_sync.execute(
        'INSERT INTO account_events (account_name, event_type, created_at) VALUES (?, ?, ?)',
        ('acc1', 'resumed', resumed_time.isoformat()),
    )
    conn_sync.commit()
    conn_sync.close()

    handler = _ListTicketsHandler(count=0)
    transport = httpx.MockTransport(handler)

    reset_sessions()
    async with httpx.AsyncClient(transport=transport) as http_client:
        tickets_conn = await aiosqlite.connect(str(tickets_path))
        tickets_conn.row_factory = aiosqlite.Row
        runs_conn = await aiosqlite.connect(str(runs_path))
        runs_conn.row_factory = aiosqlite.Row
        try:
            result = await _sample_curator(
                http_client,
                config,
                tickets_conn,
                [runs_conn],
                now=now,
            )
        finally:
            await tickets_conn.close()
            await runs_conn.close()

    assert result is not None
    assert result['capped_now'] == 0


# ---------------------------------------------------------------------------
# Step-17: collect_metrics_snapshot integration test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_metrics_snapshot_writes_curator_row(tmp_path: Path, config):
    """collect_metrics_snapshot writes one curator_snapshots row."""
    from dashboard.data.memory import reset_sessions
    from dashboard.data.stats_utils import percentile

    # collect_metrics_snapshot uses datetime.now(UTC) internally, so tickets
    # must be resolved within the last hour of the real current time.
    t_base = datetime.now(UTC) - timedelta(minutes=30)

    metrics_path = tmp_path / 'metrics.db'
    metrics_conn = await aiosqlite.connect(str(metrics_path))
    await metrics_conn.executescript(METRICS_SCHEMA)
    await metrics_conn.commit()

    # 3 terminal tickets
    tickets_path = tmp_path / 'tickets.db'
    conn_sync = sqlite3.connect(str(tickets_path))
    conn_sync.executescript(TICKETS_SCHEMA)
    for i, ms in enumerate([100, 200, 300], 1):
        conn_sync.execute(
            'INSERT INTO tickets (id, project_id, status, created_at, resolved_at) VALUES (?, ?, ?, ?, ?)',
            (
                f't{i}',
                'proj',
                'created',
                t_base.isoformat(),
                (t_base + timedelta(milliseconds=ms)).isoformat(),
            ),
        )
    conn_sync.commit()
    conn_sync.close()

    # runs.db: no events
    runs_path = tmp_path / 'runs.db'
    conn_sync = sqlite3.connect(str(runs_path))
    conn_sync.executescript(ACCOUNT_EVENTS_SCHEMA)
    conn_sync.commit()
    conn_sync.close()

    handler = _ListTicketsHandler(count=2)
    transport = httpx.MockTransport(handler)

    reset_sessions()
    async with httpx.AsyncClient(transport=transport) as http_client:
        tickets_conn = await aiosqlite.connect(str(tickets_path))
        tickets_conn.row_factory = aiosqlite.Row
        runs_conn = await aiosqlite.connect(str(runs_path))
        runs_conn.row_factory = aiosqlite.Row
        try:
            await collect_metrics_snapshot(
                conn=metrics_conn,
                config=config,
                http_client=http_client,
                recon_db=None,
                merge_dbs=[('proj', runs_conn)],
                tickets_db=tickets_conn,
            )
        finally:
            await tickets_conn.close()
            await runs_conn.close()

    async with metrics_conn.execute(
        'SELECT pending_total, capped_now, p50_active_ms, p90_active_ms, p99_active_ms '
        'FROM curator_snapshots'
    ) as cur:
        rows = list(await cur.fetchall())

    await metrics_conn.close()

    assert len(rows) == 1, f'Expected 1 row, got {len(rows)}'
    pending_total, capped_now, p50, p90, p99 = rows[0]
    assert pending_total == 2
    assert capped_now == 0
    assert p50 == int(percentile([100.0, 200.0, 300.0], 50))
    assert p90 == int(percentile([100.0, 200.0, 300.0], 90))
    assert p99 == int(percentile([100.0, 200.0, 300.0], 99))


# ---------------------------------------------------------------------------
# Step-19: Metrics-loop kwarg test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_loop_passes_tickets_db_kwarg(tmp_path: Path):
    """_metrics_loop._run_once passes tickets_db kwarg to collect_metrics_snapshot.

    Calls _metrics_loop directly against a hand-built app-state stub so the
    real lifespan — and its _burndown_loop side task — are never started.
    Only _metrics_loop is exercised; collect_metrics_snapshot is patched to
    record the call and set an event, then the task is cancelled cleanly.

    tickets.db is created on disk at the canonical path so DbPool.get() returns
    a real connection.  This pins the path-to-connection wiring: a wrong-but-
    missing path would make get() return None, failing the is-not-None assertion.
    """
    called_event = asyncio.Event()

    async def _side_effect(*args, **kwargs):
        called_event.set()

    mock_collect = AsyncMock(side_effect=_side_effect)

    # Minimal app-state stub — no real lifespan, no side tasks.
    fixed_config = DashboardConfig(
        project_root=tmp_path,
        fused_memory_urls=['http://localhost:18765'],
        known_project_roots=[],
    )
    # Create tickets.db at the canonical path so DbPool.get() returns a real
    # connection.  This validates path-to-connection wiring, not just key presence.
    tickets_db_path = fixed_config.tickets_db
    tickets_db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(tickets_db_path)):
        pass

    pool = DbPool()
    mock_app = MagicMock()
    mock_app.state.config = fixed_config
    mock_app.state.db = pool
    mock_app.state.http_client = (
        MagicMock()
    )  # collect_metrics_snapshot is patched; client never used

    # _metrics_loop now takes a _MetricsStore, not a raw aiosqlite.Connection.
    metrics_store = _MetricsStore(tmp_path / 'metrics.db', busy_timeout_ms=5000)
    await metrics_store.open()

    expected_conn = None
    loop_opened = False
    try:
        with patch('dashboard.app.collect_metrics_snapshot', mock_collect):
            # _metrics_loop calls _run_once() immediately before entering the
            # aligned-sleep loop.  We cancel the task once the event fires.
            task = asyncio.create_task(_metrics_loop(metrics_store, mock_app))
            try:
                # 2 s is generous for a single fast AsyncMock _run_once() cycle.
                await asyncio.wait_for(called_event.wait(), timeout=2.0)
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        # Verify the loop opened the connection BEFORE calling pool.get() ourselves.
        # pool.open_count > 0 means _metrics_loop._run_once() actually called pool.get();
        # without this check, pool.get() below would lazily open the connection and mask
        # a regression where the loop stops using the pool.
        loop_opened = pool.open_count > 0
        # Snapshot the cached connection via the public API before close_all().
        # DbPool.get() is idempotent: returns the same cached object on repeat call.
        expected_conn = await pool.get(fixed_config.tickets_db)
    finally:
        await metrics_store.close()
        await pool.close_all()

    assert mock_collect.called, 'collect_metrics_snapshot was never called'
    call_kwargs = mock_collect.call_args.kwargs if mock_collect.call_args else {}
    assert 'tickets_db' in call_kwargs, (
        f'tickets_db not in kwargs: {call_kwargs}. All calls: {mock_collect.call_args_list}'
    )
    assert loop_opened, (
        f'_metrics_loop._run_once() never called pool.get(); '
        f'pool.open_count was 0 after the task ran — '
        f'check that _run_once() uses the pool for {fixed_config.tickets_db}'
    )
    # tickets.db exists on disk so DbPool.get() returns the real connection it
    # cached during _metrics_loop._run_once().  We assert object identity to pin
    # path-to-connection wiring: a wrong-but-existing path yields a different
    # connection object; a wrong-but-missing path leaves expected_conn as None.
    assert expected_conn is not None, (
        f'DbPool.get() returned None for {fixed_config.tickets_db}; '
        f'file was created in test setup — check if the pool connection opened correctly'
    )
    assert call_kwargs['tickets_db'] is expected_conn, (
        f'tickets_db should be the DbPool connection for {fixed_config.tickets_db} '
        f'but got {call_kwargs["tickets_db"]!r}. '
        f'All calls: {mock_collect.call_args_list}'
    )


# ---------------------------------------------------------------------------
# Step-1 (task-1279): wait_for ceiling — timeout returns partial pending_total
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sample_curator_http_timeout_partial_pending_total(
    tmp_path: Path,
    monkeypatch,
):
    """_sample_curator wraps each per-(url,root) mcp_tool_call in asyncio.wait_for.

    Setup: two distinct project roots (r1, r2), one fused_memory_url.
    The first call returns count=3 immediately; the second awaits asyncio.sleep(5)
    — longer than the real timeout — which proves wait_for is in place (not just
    a pre-cooked TimeoutError).

    Monkeypatching _HTTP_SAMPLER_TIMEOUT_SECONDS to 0.05 keeps the test fast.

    Assertions:
    - no exception propagates
    - result['pending_total'] == 3 (partial accumulation from r1 only)
    - mcp_tool_call was called at least twice (loop reached r2)
    - wall-clock elapsed < 3s (timeout was honoured, not the un-capped 5s sleep)
    """
    import time

    import dashboard.data.metrics as metrics_mod
    from dashboard.config import DashboardConfig
    from dashboard.data.metrics import _sample_curator

    # Patch the timeout constant to keep the test fast
    monkeypatch.setattr(metrics_mod, '_HTTP_SAMPLER_TIMEOUT_SECONDS', 0.05)

    # Build config with two distinct roots
    r1 = tmp_path / 'r1'
    r2 = tmp_path / 'r2'
    r1.mkdir()
    r2.mkdir()
    cfg = DashboardConfig(
        project_root=r1,
        fused_memory_urls=['http://localhost:18765'],
        known_project_roots=[r2],
    )

    call_count = 0

    async def _side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {'count': 3}
        # Subsequent calls hang — they should be cancelled by wait_for
        await asyncio.sleep(5)
        return {'count': 99}  # unreachable

    mock_mcp = AsyncMock(side_effect=_side_effect)

    now = datetime.now(UTC)
    # A no-op transport so httpx.AsyncClient builds without error
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))
    t0 = time.monotonic()
    with patch('dashboard.data.metrics.mcp_tool_call', mock_mcp):
        async with httpx.AsyncClient(transport=transport) as http_client:
            result = await _sample_curator(
                http_client,
                cfg,
                tickets_db=None,
                runs_dbs=[],
                now=now,
            )
    elapsed = time.monotonic() - t0

    assert result is not None
    assert result['pending_total'] == 3, f'Expected 3, got {result["pending_total"]}'
    assert call_count >= 2, f'Expected at least 2 mcp_tool_call invocations, got {call_count}'
    assert elapsed < 3.0, (
        f'Elapsed {elapsed:.3f}s ≥ 3s — wait_for ceiling does not appear to be in place'
    )


# ---------------------------------------------------------------------------
# Step-1 (task-1299): fan_out_list_tickets failover — first URL raises HTTPError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fan_out_list_tickets_failover_first_url_http_error(tmp_path: Path):
    """fan_out_list_tickets falls back to the second URL when the first raises HTTPError.

    Setup: single root (tmp_path), two fused_memory_urls ['http://bad', 'http://good'].
    The mock raises httpx.HTTPError for 'http://bad' and returns count=3 for 'http://good'.

    Assertions:
    - pending_total == 3 (counted ONCE via failover, not 6 — proves the per-url
      exception is swallowed and the break prevents double-counting)
    - mock_mcp.call_count == 2 (both URLs were attempted)
    - tickets == [] (no ticket rows in the mock response)
    """
    from dashboard.data.metrics import fan_out_list_tickets

    cfg = DashboardConfig(
        project_root=tmp_path,
        fused_memory_urls=['http://bad', 'http://good'],
        known_project_roots=[],
    )

    async def _side_effect(http_client, url, *args, **kwargs):
        if url == 'http://bad':
            raise httpx.HTTPError('boom')
        return {'count': 3, 'tickets': [], 'project_id': 'p'}

    mock_mcp = AsyncMock(side_effect=_side_effect)
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))

    with patch('dashboard.data.metrics.mcp_tool_call', mock_mcp):
        async with httpx.AsyncClient(transport=transport) as http_client:
            tickets, pending_total = await fan_out_list_tickets(http_client, cfg)

    assert pending_total == 3, f'Expected pending_total=3, got {pending_total}'
    assert mock_mcp.call_count == 2, (
        f'Expected 2 mcp_tool_call invocations (both URLs tried), got {mock_mcp.call_count}'
    )
    assert tickets == [], f'Expected empty tickets list, got {tickets}'


# ---------------------------------------------------------------------------
# Step-2 (task-1299): fan_out_list_tickets saturation warning when count == limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fan_out_list_tickets_warns_when_count_at_limit(tmp_path: Path, caplog):
    """fan_out_list_tickets logs a WARNING when list_tickets returns count == limit.

    Setup: single root (tmp_path), one URL. Mock returns count=2000 == limit=2000.

    Assertions:
    - pending_total == 2000 (count is NOT suppressed by the warning)
    - at least one WARNING log record on the dashboard.data.metrics logger whose
      message mentions both 'list_tickets' and '2000'

    Replaces a prior integration-level _sample_curator saturation test; pending_total propagation is now covered by test_sample_curator_http_timeout_partial_pending_total.
    """
    from dashboard.data.metrics import fan_out_list_tickets

    cfg = DashboardConfig(
        project_root=tmp_path,
        fused_memory_urls=['http://localhost:18765'],
        known_project_roots=[],
    )

    mock_mcp = AsyncMock(return_value={'count': 2000, 'tickets': [], 'project_id': 'p'})
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))

    with (
        caplog.at_level(logging.WARNING, logger='dashboard.data.metrics'),
        patch('dashboard.data.metrics.mcp_tool_call', mock_mcp),
    ):
        async with httpx.AsyncClient(transport=transport) as http_client:
            tickets, pending_total = await fan_out_list_tickets(
                http_client,
                cfg,
                limit=2000,
            )

    assert pending_total == 2000, f'Expected pending_total=2000, got {pending_total}'
    warning_records = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING
        and 'list_tickets' in r.getMessage()
        and '2000' in r.getMessage()
    ]
    assert warning_records, (
        "Expected at least one WARNING log mentioning 'list_tickets' and '2000' "
        f'for saturation, got records: {[(r.levelname, r.getMessage()) for r in caplog.records]}'
    )


# ---------------------------------------------------------------------------
# Step-1 (task-1280): _LIST_TICKETS_LIMIT constant flows into fan_out_list_tickets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fan_out_list_tickets_uses_LIST_TICKETS_LIMIT_constant(
    tmp_path: Path, caplog, monkeypatch
):
    """_LIST_TICKETS_LIMIT constant must flow into the outgoing request limit.

    Monkeypatches _LIST_TICKETS_LIMIT to 50 and calls fan_out_list_tickets
    WITHOUT a limit kwarg so
    the function default is exercised.  count=50 == limit=50 triggers the
    saturation WARNING.

    Assertions:
    - handler.calls[0]['params']['arguments']['limit'] == 50  (constant flowed in)
    - pending_total == 50
    - at least one WARNING record on the dashboard.data.metrics logger whose
      getMessage() contains '50' (saturation warning uses the patched constant)

    Failing baseline: _LIST_TICKETS_LIMIT doesn't exist today AND
    fan_out_list_tickets's default is the literal 2000, so the patched value
    never reaches the payload — handler.calls[0]['params']['arguments']['limit']
    is 2000, not 50.
    """
    import dashboard.data.metrics as _metrics_mod
    from dashboard.data.memory import reset_sessions
    from dashboard.data.metrics import fan_out_list_tickets

    monkeypatch.setattr(_metrics_mod, '_LIST_TICKETS_LIMIT', 50)

    cfg = DashboardConfig(
        project_root=tmp_path,
        fused_memory_urls=['http://localhost:18765'],
        known_project_roots=[],
    )
    handler = _ListTicketsHandler(count=50)
    transport = httpx.MockTransport(handler)

    reset_sessions()
    with caplog.at_level(logging.WARNING, logger='dashboard.data.metrics'):
        async with httpx.AsyncClient(transport=transport) as http_client:
            _, pending_total = await fan_out_list_tickets(http_client, cfg)

    # (a) The patched constant flowed into the outgoing payload
    assert handler.calls, 'fan_out_list_tickets made no tool/call requests'
    assert handler.calls[0]['params']['arguments']['limit'] == 50, (
        f'Expected limit=50 in outgoing payload, got '
        f'{handler.calls[0]["params"]["arguments"].get("limit")}'
    )
    # (b) pending_total reflects the mocked count
    assert pending_total == 50, f'Expected pending_total=50, got {pending_total}'
    # (c) Saturation WARNING mentions the patched constant value
    warning_records = [
        r for r in caplog.records if r.levelno >= logging.WARNING and '50' in r.getMessage()
    ]
    assert warning_records, (
        "Expected at least one WARNING mentioning '50' (saturation at patched limit), "
        f'got: {[(r.levelname, r.getMessage()) for r in caplog.records]}'
    )


# ---------------------------------------------------------------------------
# Step-3 (task-1280): TimeoutError in fan_out_list_tickets logs WARNING with root+url
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fan_out_list_tickets_timeout_logs_warning_with_root_and_url(tmp_path: Path, caplog):
    """fan_out_list_tickets must log a WARNING (not DEBUG) for TimeoutError.

    Patches mcp_tool_call with AsyncMock(side_effect=TimeoutError()) so the
    error is deterministic.  Single root (tmp_path), single URL
    'http://localhost:18765'.

    Assertions:
    - no exception propagates (TimeoutError is swallowed)
    - pending_total == 0
    - at least one WARNING record whose getMessage() contains BOTH
      str(tmp_path) AND 'http://localhost:18765'

    Failing baseline: today's fan_out_list_tickets catches TimeoutError inside
    the broad except (httpx.HTTPError, TimeoutError, ValueError) block at DEBUG
    level — no WARNING record is emitted.
    """
    from dashboard.data.memory import reset_sessions
    from dashboard.data.metrics import fan_out_list_tickets

    cfg = DashboardConfig(
        project_root=tmp_path,
        fused_memory_urls=['http://localhost:18765'],
        known_project_roots=[],
    )

    mock_mcp = AsyncMock(side_effect=TimeoutError())
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))

    reset_sessions()
    with (
        caplog.at_level(logging.WARNING, logger='dashboard.data.metrics'),
        patch('dashboard.data.metrics.mcp_tool_call', mock_mcp),
    ):
        async with httpx.AsyncClient(transport=transport) as http_client:
            _, pending_total = await fan_out_list_tickets(http_client, cfg)

    assert pending_total == 0, f'Expected pending_total=0, got {pending_total}'

    root_str = str(tmp_path)
    warning_records = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING
        and root_str in r.getMessage()
        and 'http://localhost:18765' in r.getMessage()
    ]
    assert warning_records, (
        f"Expected at least one WARNING containing both '{root_str}' and "
        f"'http://localhost:18765', got: {[(r.levelname, r.getMessage()) for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# Step-11 (task-1280): _sample_curator delegates to compute_capped_now_and_windows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sample_curator_delegates_to_compute_capped_now_and_windows(tmp_path: Path, config):
    """_sample_curator must call compute_capped_now_and_windows and use its return.

    Setup: cap_hit 30 min ago (open-ended interval), empty tickets.db.
    Patches dashboard.data.metrics.compute_capped_now_and_windows with a
    MagicMock returning sentinel (7, []) — capped_now=7 is impossible from
    the real helper so its presence in the result proves the mock was used.

    Assertions:
    - result['capped_now'] == 7  (sentinel flowed through)
    - mock was called exactly once
    - mock's first positional arg is a non-empty list of CapInterval objects

    Failing baseline: _sample_curator computes capped_now inline (line ~289),
    never calling compute_capped_now_and_windows, so result['capped_now'] is
    1 (real open-ended interval), not 7.
    """
    from dashboard.data.cap_history import CapInterval
    from dashboard.data.memory import reset_sessions
    from dashboard.data.metrics import _sample_curator

    now = datetime.now(UTC)

    # runs.db: one open-ended cap_hit 30 min ago
    runs_path = tmp_path / 'runs.db'
    conn_sync = sqlite3.connect(str(runs_path))
    conn_sync.executescript(ACCOUNT_EVENTS_SCHEMA)
    conn_sync.execute(
        'INSERT INTO account_events (account_name, event_type, created_at) VALUES (?, ?, ?)',
        ('acc1', 'cap_hit', (now - timedelta(minutes=30)).isoformat()),
    )
    conn_sync.commit()
    conn_sync.close()

    # tickets.db: empty
    tickets_path = tmp_path / 'tickets.db'
    conn_sync = sqlite3.connect(str(tickets_path))
    conn_sync.executescript(TICKETS_SCHEMA)
    conn_sync.commit()
    conn_sync.close()

    handler = _ListTicketsHandler(count=0)
    transport = httpx.MockTransport(handler)
    mock_helper = MagicMock(return_value=(7, []))

    reset_sessions()
    with patch('dashboard.data.metrics.compute_capped_now_and_windows', mock_helper):
        async with httpx.AsyncClient(transport=transport) as http_client:
            tickets_conn = await aiosqlite.connect(str(tickets_path))
            tickets_conn.row_factory = aiosqlite.Row
            runs_conn = await aiosqlite.connect(str(runs_path))
            runs_conn.row_factory = aiosqlite.Row
            try:
                result = await _sample_curator(
                    http_client,
                    config,
                    tickets_conn,
                    [runs_conn],
                    now=now,
                )
            finally:
                await tickets_conn.close()
                await runs_conn.close()

    # (a) Sentinel capped_now flowed through
    assert result['capped_now'] == 7, (
        f'Expected capped_now=7 (sentinel from mock), got {result["capped_now"]}. '
        'This fails if _sample_curator computes capped_now inline instead of delegating.'
    )
    # (b) Mock was called exactly once
    assert mock_helper.call_count == 1, (
        f'Expected compute_capped_now_and_windows called once, got {mock_helper.call_count}'
    )
    # (c) First arg is a non-empty list of CapInterval
    call_arg = mock_helper.call_args.args[0]
    assert len(call_arg) >= 1, f'Expected non-empty intervals arg, got {call_arg}'
    assert all(isinstance(iv, CapInterval) for iv in call_arg), (
        f'Expected list of CapInterval, got {call_arg}'
    )


# ---------------------------------------------------------------------------
# Step-7 (task-1280): long-running open-ended cap is subtracted (days=7)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sample_curator_long_running_open_ended_cap_overlap_subtracted(
    tmp_path: Path, config
):
    """A 2-day-old open-ended cap should be included after bumping lookback to days=7.

    Setup: cap_hit 2 days ago, no resumed (open-ended). One ticket that ran
    for 60 seconds entirely inside the open-ended cap window.

    With days=1 (current): SQL cutoff filters out the 2-day-old cap_hit →
    intervals=[], capped_windows=[], overlap=0, active_ms=60_000.
    With days=7 (target): cap_hit is within the window → full 60_000ms
    overlap is subtracted → p50_active_ms == 0.

    Failing baseline: _sample_curator calls read_cap_intervals(days=1),
    so the 2-day-old cap_hit is excluded → p50_active_ms == 60_000.
    """
    from dashboard.data.memory import reset_sessions
    from dashboard.data.metrics import _sample_curator

    now = datetime.now(UTC)

    # runs.db: open-ended cap 2 days ago
    runs_path = tmp_path / 'runs.db'
    conn_sync = sqlite3.connect(str(runs_path))
    conn_sync.executescript(ACCOUNT_EVENTS_SCHEMA)
    cap_start = now - timedelta(days=2)
    conn_sync.execute(
        'INSERT INTO account_events (account_name, event_type, created_at) VALUES (?, ?, ?)',
        ('acc1', 'cap_hit', cap_start.isoformat()),
    )
    conn_sync.commit()
    conn_sync.close()

    # tickets.db: one 60s ticket fully inside the open-ended cap window
    tickets_path = tmp_path / 'tickets.db'
    conn_sync = sqlite3.connect(str(tickets_path))
    conn_sync.executescript(TICKETS_SCHEMA)
    ticket_start = now - timedelta(minutes=5)
    ticket_end = now - timedelta(minutes=4)
    conn_sync.execute(
        'INSERT INTO tickets (id, project_id, status, created_at, resolved_at) VALUES (?, ?, ?, ?, ?)',
        ('t1', 'proj', 'created', ticket_start.isoformat(), ticket_end.isoformat()),
    )
    conn_sync.commit()
    conn_sync.close()

    handler = _ListTicketsHandler(count=0)
    transport = httpx.MockTransport(handler)

    reset_sessions()
    async with httpx.AsyncClient(transport=transport) as http_client:
        tickets_conn = await aiosqlite.connect(str(tickets_path))
        tickets_conn.row_factory = aiosqlite.Row
        runs_conn = await aiosqlite.connect(str(runs_path))
        runs_conn.row_factory = aiosqlite.Row
        try:
            result = await _sample_curator(
                http_client,
                config,
                tickets_conn,
                [runs_conn],
                now=now,
            )
        finally:
            await tickets_conn.close()
            await runs_conn.close()

    # With days=7: cap_hit is within window → 60s ticket fully capped → p50=0
    assert result['p50_active_ms'] == 0, (
        f'Expected p50_active_ms=0 (cap subtracted full 60s), got {result["p50_active_ms"]}. '
        'This fails if _sample_curator uses days=1 (excludes 2-day-old cap_hit).'
    )
    assert result['capped_now'] == 1, (
        f'Expected capped_now=1 (open-ended interval), got {result["capped_now"]}'
    )


# ---------------------------------------------------------------------------
# task-1298 step-3: fan_out_list_tickets runs roots concurrently
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fan_out_list_tickets_runs_roots_concurrently(tmp_path: Path):
    """fan_out_list_tickets runs per-root coroutines in parallel.

    Two-event handshake:
    - r1's mock sets event_r1_started then awaits event_r2_started (2 s ceiling).
    - r2's mock sets event_r2_started then awaits event_r1_started (2 s ceiling).

    Concurrent execution: both fire concurrently → each sets its own event before
    awaiting the other → both wait_for calls return within ms → pending_total == 2.

    Sequential execution: r1 sets event_r1 then waits for event_r2 (which is
    never set because r2 hasn't started yet) → r1 times out after 2 s →
    TimeoutError swallowed at WARNING → r1 contributes 0; r2 then runs, sets
    event_r2, awaits event_r1 (already set) → r2 contributes 1 → pending_total == 1.

    pending_total == 2 is the definitive concurrent signal.
    """
    from dashboard.data.metrics import fan_out_list_tickets

    r1 = tmp_path / 'r1'
    r2 = tmp_path / 'r2'
    r1.mkdir()
    r2.mkdir()

    cfg = DashboardConfig(
        project_root=r1,
        fused_memory_urls=['http://localhost:18765'],
        known_project_roots=[r2],
    )

    event_r1_started = asyncio.Event()
    event_r2_started = asyncio.Event()

    async def _side_effect(http_client, url, tool, arguments):
        root = arguments.get('project_root', '')
        if root == str(r1):
            event_r1_started.set()
            await asyncio.wait_for(event_r2_started.wait(), timeout=2.0)
        else:
            event_r2_started.set()
            await asyncio.wait_for(event_r1_started.wait(), timeout=2.0)
        return {'project_id': 'p', 'count': 1, 'tickets': []}

    mock_mcp = AsyncMock(side_effect=_side_effect)
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))

    with patch('dashboard.data.metrics.mcp_tool_call', mock_mcp):
        async with httpx.AsyncClient(transport=transport) as http_client:
            tickets, pending_total = await fan_out_list_tickets(
                http_client,
                cfg,
                limit=2000,
                timeout=5.0,
            )

    assert pending_total == 2, (
        f'pending_total=={pending_total}, expected 2 — '
        'roots did not run concurrently (sequential gives 1: r1 times out, r2 succeeds)'
    )


# ---------------------------------------------------------------------------
# task-1298 step-4: URL fallback regression — first URL success skips second
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fan_out_list_tickets_first_url_succeeds_skips_second(tmp_path: Path):
    """When URL1 succeeds, URL2 must never be called (break short-circuit holds).

    Setup: ONE root (tmp_path), TWO fused_memory_urls; BOTH return count=5.

    Assertions:
    - mock_mcp.call_count == 1  (URL2 must NOT be attempted after URL1 succeeds)
    - pending_total == 5        (URL1's count only — NOT 10 from double-counting)

    This distinguishes the correct sequential-URL-fallback from a regression that
    parallelises URLs within a root: parallel would call both URLs concurrently,
    giving call_count == 2 and pending_total == 10.

    The existing test_fan_out_list_tickets_failover_first_url_http_error covers
    the complementary case (URL1 fails → URL2 is tried).  Together they pin both
    halves of the sequential-fallback contract.
    """
    from dashboard.data.metrics import fan_out_list_tickets

    cfg = DashboardConfig(
        project_root=tmp_path,
        fused_memory_urls=['http://url1', 'http://url2'],
        known_project_roots=[],
    )

    mock_mcp = AsyncMock(return_value={'project_id': 'p', 'count': 5, 'tickets': []})
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))

    with patch('dashboard.data.metrics.mcp_tool_call', mock_mcp):
        async with httpx.AsyncClient(transport=transport) as http_client:
            tickets, pending_total = await fan_out_list_tickets(
                http_client,
                cfg,
                limit=2000,
            )

    assert mock_mcp.call_count == 1, (
        f'mock_mcp.call_count=={mock_mcp.call_count}, expected 1 — '
        'URL2 must not be called when URL1 succeeds (break short-circuit)'
    )
    assert pending_total == 5, (
        f'pending_total=={pending_total}, expected 5 — '
        'should count URL1 only, not double-count both URLs'
    )
    assert tickets == []


# ---------------------------------------------------------------------------
# task-1510 step-3(B): mixed availability → capped_now=0 in _sample_curator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sample_curator_mixed_availability_returns_capped_now_zero(
    tmp_path: Path, config
):
    """capped_now=0 when one account is capped and another is uncapped.

    Setup: two accounts in runs.db:
    - acc-a: open-ended cap_hit (still capped)
    - acc-b: cap_hit + resumed closed pair (currently uncapped)

    With all-accounts-capped semantics: total=2, capped=1, available=1 → capped_now=0.
    Under old any-account semantics acc-a is open → capped_now=1 (the bug).

    RED baseline: old any-account semantics; GREEN after all-accounts-capped fix.
    """
    from dashboard.data.memory import reset_sessions
    from dashboard.data.metrics import _sample_curator

    now = datetime.now(UTC)

    tickets_path = tmp_path / 'tickets.db'
    conn_sync = sqlite3.connect(str(tickets_path))
    conn_sync.executescript(TICKETS_SCHEMA)
    conn_sync.commit()
    conn_sync.close()

    runs_path = tmp_path / 'runs.db'
    conn_sync = sqlite3.connect(str(runs_path))
    conn_sync.executescript(ACCOUNT_EVENTS_SCHEMA)

    # acc-a: open-ended cap (still capped)
    cap_time_a = now - timedelta(minutes=30)
    conn_sync.execute(
        'INSERT INTO account_events (account_name, event_type, created_at) VALUES (?, ?, ?)',
        ('acc-a', 'cap_hit', cap_time_a.isoformat()),
    )
    # acc-b: closed cap (currently uncapped)
    cap_time_b = now - timedelta(minutes=60)
    resumed_time_b = now - timedelta(minutes=20)
    conn_sync.execute(
        'INSERT INTO account_events (account_name, event_type, created_at) VALUES (?, ?, ?)',
        ('acc-b', 'cap_hit', cap_time_b.isoformat()),
    )
    conn_sync.execute(
        'INSERT INTO account_events (account_name, event_type, created_at) VALUES (?, ?, ?)',
        ('acc-b', 'resumed', resumed_time_b.isoformat()),
    )
    conn_sync.commit()
    conn_sync.close()

    handler = _ListTicketsHandler(count=0)
    transport = httpx.MockTransport(handler)

    reset_sessions()
    async with httpx.AsyncClient(transport=transport) as http_client:
        tickets_conn = await aiosqlite.connect(str(tickets_path))
        tickets_conn.row_factory = aiosqlite.Row
        runs_conn = await aiosqlite.connect(str(runs_path))
        runs_conn.row_factory = aiosqlite.Row
        try:
            result = await _sample_curator(
                http_client,
                config,
                tickets_conn,
                [runs_conn],
                now=now,
            )
        finally:
            await tickets_conn.close()
            await runs_conn.close()

    assert result is not None
    assert result['capped_now'] == 0, (
        f'Expected capped_now=0 (acc-b is uncapped so not all accounts are capped), '
        f'got {result["capped_now"]}. '
        f'Bug: old any-account semantics returns 1 because acc-a is still open.'
    )


# ---------------------------------------------------------------------------
# task-1329 step-3: CancelledError from a CHILD coroutine propagates out of
# fan_out_list_tickets (buggy reducer swallows it silently)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fan_out_list_tickets_propagates_child_cancelled_error(tmp_path: Path):
    """fan_out_list_tickets must propagate CancelledError raised in a child coroutine.

    RED baseline: the current reducer
        per_root_results = [r if not isinstance(r, BaseException) else (0, []) ...]
    coerces a child-level CancelledError captured in raw_results into (0, []).
    fan_out_list_tickets then returns ([], 0) silently, pytest.raises sees no
    exception → the test FAILS (no raise observed).

    The fix is to replace the reducer with
        safe_gather_result(r, default, 'curator/fan_out_root')
    which re-raises non-Exception BaseException (CancelledError/KeyboardInterrupt/
    SystemExit) and substitutes the default only for Exception.

    IMPORTANT — why child-level CancelledError and not outer-task cancel:
    This test deliberately raises CancelledError from INSIDE a child coroutine
    (mcp_tool_call side_effect) rather than cancelling the outer task that runs
    fan_out_list_tickets.  Under asyncio.gather(return_exceptions=True), a
    child-level CancelledError is captured INTO raw_results as a BaseException
    instance — the gather itself is NOT cancelled.  Cancelling the outer task
    instead makes `await asyncio.gather(...)` raise CancelledError directly
    (bypassing the reducer), which would pass even with the buggy code and yield
    no RED.  The reducer bug only manifests when CancelledError is aggregated
    into raw_results via a child-level raise.
    """
    from dashboard.data.metrics import fan_out_list_tickets

    cfg = DashboardConfig(
        project_root=tmp_path,
        fused_memory_urls=['http://localhost:18765'],
        known_project_roots=[],
    )
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))

    with patch(
        'dashboard.data.metrics.mcp_tool_call', new=AsyncMock(side_effect=asyncio.CancelledError())
    ):
        async with httpx.AsyncClient(transport=transport) as http_client:
            with pytest.raises(asyncio.CancelledError):
                await fan_out_list_tickets(http_client, cfg, limit=2000, timeout=5.0)
