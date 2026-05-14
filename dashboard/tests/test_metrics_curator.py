"""Tests for curator_snapshots table + metrics sampler + reader.

Built incrementally — one step at a time in TDD order.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiosqlite
import httpx
import pytest

from dashboard.data.metrics import (
    METRICS_SCHEMA,
    downsample_metrics,
    collect_metrics_snapshot,
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
        200, json=body,
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
        200, json=body,
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
    assert row is not None, "curator_snapshots table was not created by METRICS_SCHEMA"

    # Check expected columns
    col_info = conn.execute('PRAGMA table_info(curator_snapshots)').fetchall()
    cols = {r[1] for r in col_info}
    expected = {'ts', 'pending_total', 'capped_now', 'p50_active_ms', 'p90_active_ms', 'p99_active_ms'}
    assert expected <= cols, f"Missing columns: {expected - cols}"

    # Check ts is the PRIMARY KEY (pk=1 in PRAGMA table_info)
    pk_cols = {r[1] for r in col_info if r[5] == 1}
    assert 'ts' in pk_cols, "ts is not the PRIMARY KEY of curator_snapshots"

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
        assert result[key] == {'labels': [], 'values': []}, \
            f"Key '{key}' expected empty series, got {result[key]}"


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
    rows = inspect.execute(
        'SELECT pending_total FROM curator_snapshots ORDER BY ts'
    ).fetchall()
    inspect.close()

    assert rows == [(9,)], f"Expected [(9,)], got {rows}"


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

    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    t_base = now - timedelta(minutes=30)

    # tickets.db: 3 terminal tickets with durations 100ms, 200ms, 300ms
    tickets_path = tmp_path / 'tickets.db'
    conn_sync = sqlite3.connect(str(tickets_path))
    conn_sync.executescript(TICKETS_SCHEMA)
    conn_sync.execute(
        'INSERT INTO tickets (id, project_id, status, created_at, resolved_at) VALUES (?, ?, ?, ?, ?)',
        ('t1', 'proj', 'created',
         t_base.isoformat(),
         (t_base + timedelta(milliseconds=100)).isoformat()),
    )
    conn_sync.execute(
        'INSERT INTO tickets (id, project_id, status, created_at, resolved_at) VALUES (?, ?, ?, ?, ?)',
        ('t2', 'proj', 'combined',
         t_base.isoformat(),
         (t_base + timedelta(milliseconds=200)).isoformat()),
    )
    conn_sync.execute(
        'INSERT INTO tickets (id, project_id, status, created_at, resolved_at) VALUES (?, ?, ?, ?, ?)',
        ('t3', 'proj', 'failed',
         t_base.isoformat(),
         (t_base + timedelta(milliseconds=300)).isoformat()),
    )
    # Ticket D: too old (> 1 hour ago), should not be counted
    old_t = now - timedelta(hours=2)
    conn_sync.execute(
        'INSERT INTO tickets (id, project_id, status, created_at, resolved_at) VALUES (?, ?, ?, ?, ?)',
        ('t4', 'proj', 'failed',
         old_t.isoformat(),
         (old_t + timedelta(milliseconds=500)).isoformat()),
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
                http_client, config, tickets_conn, [runs_conn], now=now,
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

    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
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
                http_client, config, tickets_conn, [runs_conn], now=now,
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

    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

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
                http_client, config, tickets_conn, [runs_conn], now=now,
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

    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

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
                http_client, config, tickets_conn, [runs_conn], now=now,
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

    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    t_base = now - timedelta(minutes=30)

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
            (f't{i}', 'proj', 'created',
             t_base.isoformat(),
             (t_base + timedelta(milliseconds=ms)).isoformat()),
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
        rows = await cur.fetchall()

    await metrics_conn.close()

    assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"
    pending_total, capped_now, p50, p90, p99 = rows[0]
    assert pending_total == 2
    assert capped_now == 0
    assert p50 == int(percentile([100.0, 200.0, 300.0], 50))
    assert p90 == int(percentile([100.0, 200.0, 300.0], 90))
    assert p99 == int(percentile([100.0, 200.0, 300.0], 99))


# ---------------------------------------------------------------------------
# Step-19: App-wiring test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_app_wiring_tickets_db_passed_to_collect_metrics_snapshot(tmp_path: Path):
    """_metrics_loop._run_once passes tickets_db kwarg to collect_metrics_snapshot."""
    import asyncio
    from dashboard.app import app, lifespan

    mock_collect = AsyncMock()

    with patch('dashboard.app.collect_metrics_snapshot', mock_collect):
        async with lifespan(app):
            await asyncio.sleep(0.05)

    assert mock_collect.called, 'collect_metrics_snapshot was never called'
    call_kwargs = mock_collect.call_args.kwargs if mock_collect.call_args else {}
    assert 'tickets_db' in call_kwargs, (
        f"tickets_db not in kwargs: {call_kwargs}. "
        f"All calls: {mock_collect.call_args_list}"
    )
