"""Tests for GET /api/v2/dashboard/curator endpoint + shape_curator.

TDD steps:
  step-1  envelope shape test (RED before route exists)
  step-3  pending list fan-out test (RED before real fan-out)
  step-5  latency_spark from metrics.db (RED before spark wiring)
  step-7  capped_spark from runs.db (RED before cap interval wiring)
  step-9  capped_now open-ended interval + pure shape_curator unit test
"""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from dashboard.data.metrics import METRICS_SCHEMA

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# fan_out_list_tickets (in metrics.py) resolves mcp_tool_call from the
# dashboard.data.metrics namespace, so patch there — not dashboard.data.memory.
_PATCH_TARGET = 'dashboard.data.metrics.mcp_tool_call'

# Healthy (unpaused) curator-gate dict returned by get_curator_state when the
# MCP helper is mocked.  Most endpoint tests do not care about paused_reason and
# just need a deterministic, server-free response.
_HEALTHY_GATE = {'paused': False, 'paused_reason': None, 'soonest_open_at': None, 'account_count': 0}

# Convenience patch target for get_curator_state (resolved in the app namespace).
_PATCH_CURATOR_STATE = 'dashboard.app.memory_data.get_curator_state'

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


def _make_config(tmp_path: Path, *, fused_memory_urls=None, known_project_roots=None):
    """Build a DashboardConfig pointing at tmp_path."""
    from dashboard.config import DashboardConfig

    return DashboardConfig(
        project_root=tmp_path,
        fused_memory_urls=fused_memory_urls or ['http://localhost:18765'],
        known_project_roots=known_project_roots or [],
    )


@contextmanager
def _override_client(config):
    """Context manager: yield a TestClient with app.state.config overridden."""
    from dashboard.app import app

    with TestClient(app) as c:
        app.state.config = config
        yield c


# ---------------------------------------------------------------------------
# step-1: envelope shape test (RED — route does not exist yet)
# ---------------------------------------------------------------------------


def test_curator_endpoint_returns_envelope_shape(client):
    """GET /api/v2/dashboard/curator returns 200 with correct envelope keys.

    mcp_tool_call is mocked to return empty tickets so the fan-out succeeds
    without a real server. latency_spark / capped_spark default to empty series
    because no DB is seeded in this test.

    get_curator_state is mocked to a healthy (unpaused) state so the test is
    deterministic without a real fused-memory server.  paused_reason should be
    None in the response (healthy gate has no pause reason).
    """
    mcp_result = {'project_id': 'p', 'count': 0, 'tickets': []}
    with (
        patch(_PATCH_TARGET, new=AsyncMock(return_value=mcp_result)),
        patch(_PATCH_CURATOR_STATE, new=AsyncMock(return_value=_HEALTHY_GATE)),
    ):
        resp = client.get('/api/v2/dashboard/curator')

    assert resp.status_code == 200
    data = resp.json()
    assert 'CURATOR_STATE' in data
    cs = data['CURATOR_STATE']
    assert isinstance(cs.get('pending'), list)
    ls = cs.get('latency_spark', {})
    assert 'labels' in ls
    assert 'p50' in ls
    assert 'p90' in ls
    assert 'p99' in ls
    ps = cs.get('pending_spark', {})
    assert 'labels' in ps
    assert 'values' in ps
    capped = cs.get('capped_spark', {})
    assert 'labels' in capped
    assert 'values' in capped
    state = cs.get('state', {})
    assert 'capped_now' in state
    assert 'paused_reason' in state
    assert state['paused_reason'] is None  # healthy gate → paused_reason is None
    assert 'pending_total' in state


# ---------------------------------------------------------------------------
# step-3: pending list fan-out (RED — skeleton still returns hardcoded empty)
# ---------------------------------------------------------------------------


def test_curator_pending_list_from_list_tickets_fanout(tmp_path: Path):
    """CURATOR_STATE.pending is built from the list_tickets MCP fan-out.

    Config has one root (tmp_path). The mock returns one ticket for that root.
    Assertions: pending list has length 1; row fields are correct;
    state.pending_total == 1.
    """
    now = datetime.now(UTC)
    created_at_dt = now - timedelta(seconds=120)
    created_at = created_at_dt.isoformat()

    ticket_row = {
        'ticket_id': 'tkt_a',
        'candidate_title': 'task A',
        'created_at': created_at,
    }
    mcp_result = {'project_id': 'proj_p', 'count': 1, 'tickets': [ticket_row]}

    config = _make_config(tmp_path)
    with (
        _override_client(config) as c,
        patch(_PATCH_TARGET, new=AsyncMock(return_value=mcp_result)),
        patch(_PATCH_CURATOR_STATE, new=AsyncMock(return_value=_HEALTHY_GATE)),
    ):
        resp = c.get('/api/v2/dashboard/curator')

    assert resp.status_code == 200
    cs = resp.json()['CURATOR_STATE']
    assert len(cs['pending']) == 1
    row = cs['pending'][0]
    assert row['ticket_id'] == 'tkt_a'
    assert row['project_id'] == 'proj_p'
    assert row['title'] == 'task A'
    assert row['files'] == []
    assert row['created_at'] == created_at
    assert isinstance(row['age_seconds'], (int, float))
    assert abs(row['age_seconds'] - 120) < 10  # allow small tolerance

    assert cs['state']['pending_total'] == 1


# ---------------------------------------------------------------------------
# step-5: latency_spark from metrics.db (RED — sparks still empty)
# ---------------------------------------------------------------------------


def test_curator_latency_spark_from_metrics_db(tmp_path: Path):
    """latency_spark is populated from curator_snapshots in metrics.db."""
    now = datetime.now(UTC)
    ts1 = (now - timedelta(hours=1)).isoformat()
    ts2 = (now - timedelta(minutes=30)).isoformat()

    # seed metrics.db
    metrics_dir = tmp_path / 'data' / 'burndown'
    metrics_dir.mkdir(parents=True)
    metrics_db_path = metrics_dir / 'metrics.db'
    conn = sqlite3.connect(str(metrics_db_path))
    conn.executescript(METRICS_SCHEMA)
    conn.execute(
        'INSERT INTO curator_snapshots '
        '(ts, pending_total, capped_now, p50_active_ms, p90_active_ms, p99_active_ms) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        (ts1, 1, 0, 100, 200, 300),
    )
    conn.execute(
        'INSERT INTO curator_snapshots '
        '(ts, pending_total, capped_now, p50_active_ms, p90_active_ms, p99_active_ms) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        (ts2, 2, 0, 110, 210, 310),
    )
    conn.commit()
    conn.close()

    config = _make_config(tmp_path)
    empty_result = AsyncMock(return_value={'project_id': 'p', 'count': 0, 'tickets': []})
    with (
        _override_client(config) as c,
        patch(_PATCH_TARGET, new=empty_result),
        patch(_PATCH_CURATOR_STATE, new=AsyncMock(return_value=_HEALTHY_GATE)),
    ):
        resp = c.get('/api/v2/dashboard/curator')

    assert resp.status_code == 200
    ls = resp.json()['CURATOR_STATE']['latency_spark']
    assert ls['labels'] == [ts1, ts2]
    assert ls['p50'] == [100, 110]
    assert ls['p90'] == [200, 210]
    assert ls['p99'] == [300, 310]


# ---------------------------------------------------------------------------
# step-7: capped_spark from runs.db (RED — still passing empty capped_spark)
# ---------------------------------------------------------------------------


def test_curator_capped_spark_from_runs_db(tmp_path: Path):
    """capped_spark is computed from account_events in runs.db.

    Inserts one closed cap interval (4 min) well within 24h window.
    Expected: 144-bucket sparkline with at least one '1' value;
              state.capped_now == 0 (interval is closed).
    """
    now = datetime.now(UTC)
    # Use 12-min ago → 3-min ago so the bucket right-edge at now-10min falls
    # within [start, end): bucketise_cap_sparkline samples at 600s right-edges
    # and the [start, end) condition requires start <= right_edge < end.
    cap_hit_time = now - timedelta(minutes=12)
    resumed_time = now - timedelta(minutes=3)

    # seed runs.db
    runs_dir = tmp_path / 'data' / 'orchestrator'
    runs_dir.mkdir(parents=True)
    runs_db_path = runs_dir / 'runs.db'
    conn = sqlite3.connect(str(runs_db_path))
    conn.executescript(ACCOUNT_EVENTS_SCHEMA)
    conn.execute(
        'INSERT INTO account_events (account_name, event_type, created_at) VALUES (?, ?, ?)',
        ('acc1', 'cap_hit', cap_hit_time.isoformat()),
    )
    conn.execute(
        'INSERT INTO account_events (account_name, event_type, created_at) VALUES (?, ?, ?)',
        ('acc1', 'resumed', resumed_time.isoformat()),
    )
    conn.commit()
    conn.close()

    config = _make_config(tmp_path)
    empty_result = AsyncMock(return_value={'project_id': 'p', 'count': 0, 'tickets': []})
    with (
        _override_client(config) as c,
        patch(_PATCH_TARGET, new=empty_result),
        patch(_PATCH_CURATOR_STATE, new=AsyncMock(return_value=_HEALTHY_GATE)),
    ):
        resp = c.get('/api/v2/dashboard/curator')

    assert resp.status_code == 200
    cs = resp.json()['CURATOR_STATE']
    capped = cs['capped_spark']
    assert len(capped['labels']) == 144  # 24h / 600s buckets
    assert len(capped['values']) == 144
    assert 1 in capped['values']  # at least one bucket overlaps the interval
    assert cs['state']['capped_now'] == 0  # closed interval → not capped now


# ---------------------------------------------------------------------------
# step-9a: capped_now == 1 for open-ended cap interval
# ---------------------------------------------------------------------------


def test_curator_capped_now_open_ended_interval(tmp_path: Path):
    """capped_now == 1 when a cap_hit has no matching resumed (open-ended)."""
    now = datetime.now(UTC)
    cap_hit_time = now - timedelta(minutes=10)

    runs_dir = tmp_path / 'data' / 'orchestrator'
    runs_dir.mkdir(parents=True)
    runs_db_path = runs_dir / 'runs.db'
    conn = sqlite3.connect(str(runs_db_path))
    conn.executescript(ACCOUNT_EVENTS_SCHEMA)
    conn.execute(
        'INSERT INTO account_events (account_name, event_type, created_at) VALUES (?, ?, ?)',
        ('acc1', 'cap_hit', cap_hit_time.isoformat()),
    )
    conn.commit()
    conn.close()

    config = _make_config(tmp_path)
    empty_result = AsyncMock(return_value={'project_id': 'p', 'count': 0, 'tickets': []})
    with (
        _override_client(config) as c,
        patch(_PATCH_TARGET, new=empty_result),
        patch(_PATCH_CURATOR_STATE, new=AsyncMock(return_value=_HEALTHY_GATE)),
    ):
        resp = c.get('/api/v2/dashboard/curator')

    assert resp.status_code == 200
    assert resp.json()['CURATOR_STATE']['state']['capped_now'] == 1


# ---------------------------------------------------------------------------
# step-13: api_curator delegates to compute_capped_now_and_windows
# ---------------------------------------------------------------------------


def test_api_curator_delegates_to_compute_capped_now_and_windows(tmp_path: Path):
    """api_curator uses compute_capped_now_and_windows; sentinel return value flows through.

    Sentinel capped_now == 7 is impossible from the real helper (returns 0 or 1)
    so its appearance in the response proves api_curator is using the helper's
    return value.  Failing baseline: app.py computes capped_now inline — the mock
    is never called and capped_now is 1 (real open-ended interval), not 7.
    """
    from dashboard.data.cap_history import CapInterval

    now = datetime.now(UTC)
    cap_hit_time = now - timedelta(minutes=10)

    runs_dir = tmp_path / 'data' / 'orchestrator'
    runs_dir.mkdir(parents=True)
    runs_db_path = runs_dir / 'runs.db'
    conn = sqlite3.connect(str(runs_db_path))
    conn.executescript(ACCOUNT_EVENTS_SCHEMA)
    conn.execute(
        'INSERT INTO account_events (account_name, event_type, created_at) VALUES (?, ?, ?)',
        ('acc1', 'cap_hit', cap_hit_time.isoformat()),
    )
    conn.commit()
    conn.close()

    config = _make_config(tmp_path)
    empty_result = AsyncMock(return_value={'project_id': 'p', 'count': 0, 'tickets': []})
    mock_helper = MagicMock(return_value=(7, []))
    with (
        _override_client(config) as c,
        patch(_PATCH_TARGET, new=empty_result),
        patch('dashboard.app.compute_capped_now_and_windows', new=mock_helper),
        patch(_PATCH_CURATOR_STATE, new=AsyncMock(return_value=_HEALTHY_GATE)),
    ):
        resp = c.get('/api/v2/dashboard/curator')

    assert resp.status_code == 200
    # Sentinel capped_now == 7 proves the helper's return value flowed through.
    assert resp.json()['CURATOR_STATE']['state']['capped_now'] == 7
    # Helper was called exactly once.
    assert mock_helper.call_count == 1
    # The intervals list forwarded to the helper contains CapInterval instances.
    forwarded = mock_helper.call_args.args[0]
    assert isinstance(forwarded, list)
    assert len(forwarded) >= 1
    assert all(isinstance(iv, CapInterval) for iv in forwarded)


# ---------------------------------------------------------------------------
# step-9b: pure unit test for shape_curator
# ---------------------------------------------------------------------------


def test_shape_curator_pure_function():
    """shape_curator(…) returns the correct envelope structure."""
    from dashboard.data.redux_api import shape_curator

    pending = [
        {
            'ticket_id': 'x',
            'project_id': 'p',
            'title': 't',
            'files': ['a.py'],
            'created_at': 'iso',
            'age_seconds': 5,
        }
    ]
    curator_sparks = {
        'p50': {'labels': ['t1'], 'values': [50]},
        'p90': {'labels': ['t1'], 'values': [90]},
        'p99': {'labels': ['t1'], 'values': [99]},
        'pending': {'labels': ['t1'], 'values': [1]},
    }
    capped_spark = {'labels': ['l1'], 'values': [1]}

    result = shape_curator(
        pending=pending,
        curator_sparks=curator_sparks,
        capped_spark=capped_spark,
        capped_now=1,
        paused_reason='manual',
        pending_total=42,
    )

    assert 'CURATOR_STATE' in result
    cs = result['CURATOR_STATE']

    # pending list round-trips
    assert cs['pending'] == pending

    # latency_spark collapses 4-key dict to flat shape
    ls = cs['latency_spark']
    assert ls['labels'] == ['t1']
    assert ls['p50'] == [50]
    assert ls['p90'] == [90]
    assert ls['p99'] == [99]

    # pending_spark surfaces the pending series from curator_sparks
    assert cs['pending_spark'] == {'labels': ['t1'], 'values': [1]}

    # capped_spark passes through
    assert cs['capped_spark'] == capped_spark

    # state block
    state = cs['state']
    assert state['capped_now'] == 1
    assert state['paused_reason'] == 'manual'
    assert state['pending_total'] == 42


# ---------------------------------------------------------------------------
# step-1300-3: pending_spark end-to-end from metrics.db
# ---------------------------------------------------------------------------


def test_curator_pending_spark_from_metrics_db(tmp_path: Path):
    """pending_spark is populated from curator_snapshots.pending_total in metrics.db.

    Mirrors test_curator_latency_spark_from_metrics_db — seeds two rows with
    distinct ts and pending_total values, GETs the endpoint, and asserts the
    full round-trip: sampler → snapshot → get_curator_sparks → shape_curator →
    CURATOR_STATE.pending_spark.
    """
    now = datetime.now(UTC)
    ts1 = (now - timedelta(hours=1)).isoformat()
    ts2 = (now - timedelta(minutes=30)).isoformat()

    metrics_dir = tmp_path / 'data' / 'burndown'
    metrics_dir.mkdir(parents=True)
    metrics_db_path = metrics_dir / 'metrics.db'
    conn = sqlite3.connect(str(metrics_db_path))
    conn.executescript(METRICS_SCHEMA)
    conn.execute(
        'INSERT INTO curator_snapshots '
        '(ts, pending_total, capped_now, p50_active_ms, p90_active_ms, p99_active_ms) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        (ts1, 5, 0, 100, 200, 300),
    )
    conn.execute(
        'INSERT INTO curator_snapshots '
        '(ts, pending_total, capped_now, p50_active_ms, p90_active_ms, p99_active_ms) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        (ts2, 7, 0, 110, 210, 310),
    )
    conn.commit()
    conn.close()

    config = _make_config(tmp_path)
    empty_result = AsyncMock(return_value={'project_id': 'p', 'count': 0, 'tickets': []})
    with (
        _override_client(config) as c,
        patch(_PATCH_TARGET, new=empty_result),
        patch(_PATCH_CURATOR_STATE, new=AsyncMock(return_value=_HEALTHY_GATE)),
    ):
        resp = c.get('/api/v2/dashboard/curator')

    assert resp.status_code == 200
    ps = resp.json()['CURATOR_STATE']['pending_spark']
    assert ps['labels'] == [ts1, ts2]
    assert ps['values'] == [5, 7]


# ---------------------------------------------------------------------------
# step-1299-4: bad created_at values → age_seconds=None, no 500
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'created_at_value', ['not-an-iso-date', ''], ids=['malformed_iso', 'empty_string']
)
def test_curator_endpoint_bad_created_at_returns_age_seconds_none(
    tmp_path: Path, created_at_value: str
):
    """Bad created_at is preserved verbatim; age_seconds is None; no 500.

    Two code paths are covered by the two parameter values:
    - 'not-an-iso-date': app.py raises ValueError/TypeError in the parse block,
      which is caught and sets age_seconds=None.
    - '': the `if created_at_str:` guard (app.py) skips the parse block
      entirely, leaving age_seconds=None implicitly.

    In both cases the row survives, the bad string round-trips unchanged, and
    the endpoint returns 200 (not 500).
    """
    ticket_row = {
        'ticket_id': 'tkt_x',
        'candidate_title': 'task X',
        'created_at': created_at_value,
    }
    mcp_result = {'project_id': 'proj_p', 'count': 1, 'tickets': [ticket_row]}

    config = _make_config(tmp_path)
    with (
        _override_client(config) as c,
        patch(_PATCH_TARGET, new=AsyncMock(return_value=mcp_result)),
        patch(_PATCH_CURATOR_STATE, new=AsyncMock(return_value=_HEALTHY_GATE)),
    ):
        resp = c.get('/api/v2/dashboard/curator')

    assert resp.status_code == 200, f'Expected 200, got {resp.status_code}'
    cs = resp.json()['CURATOR_STATE']
    assert len(cs['pending']) == 1, f'Expected 1 pending row, got {len(cs["pending"])}'
    row = cs['pending'][0]
    assert row['ticket_id'] == 'tkt_x'
    assert row['created_at'] == created_at_value, (
        f'created_at should be preserved verbatim, got {row["created_at"]!r}'
    )
    assert row['age_seconds'] is None, (
        f'age_seconds should be None for bad created_at, got {row["age_seconds"]!r}'
    )


# ---------------------------------------------------------------------------
# task-1298 step-1: fan_out and DB queries run concurrently in api_curator
# ---------------------------------------------------------------------------


def test_api_curator_runs_fanout_and_db_concurrently(tmp_path: Path):
    """fan_out_list_tickets runs concurrently with get_curator_sparks in api_curator.

    Concurrency is verified via an asyncio.Event handshake:
    - mcp_tool_call (inside fan_out) awaits sparks_started (with 2 s ceiling).
    - get_curator_sparks sets sparks_started and returns immediately.

    If they run concurrently: sparks_started is set while mcp_tool_call is
    waiting → wait_for returns normally → mcp_timed_out stays False.

    If they run sequentially (fan_out first): get_curator_sparks hasn't started
    yet → sparks_started is never set → 2 s wait_for fires → mcp_timed_out = True.

    fan_out_list_tickets swallows TimeoutError at WARNING, so the endpoint
    returns 200 in BOTH cases — status_code alone cannot distinguish sequential
    from concurrent.  The mcp_timed_out flag is the only definitive signal.

    NOTE: patch target for get_curator_sparks is dashboard.app.get_curator_sparks
    (not dashboard.data.metrics.get_curator_sparks) because api_curator looks up
    the name bound in the dashboard.app module namespace (imported at line ~61-73).
    """
    config = _make_config(tmp_path)
    sparks_started = asyncio.Event()
    mcp_timed_out = False

    _empty_sparks: dict = {
        'pending': {'labels': [], 'values': []},
        'p50': {'labels': [], 'values': []},
        'p90': {'labels': [], 'values': []},
        'p99': {'labels': [], 'values': []},
    }

    async def _mock_sparks(db, *, days):
        sparks_started.set()
        return _empty_sparks

    async def _mock_mcp(http_client, url, tool, arguments):
        nonlocal mcp_timed_out
        try:
            await asyncio.wait_for(sparks_started.wait(), timeout=2.0)
        except TimeoutError:
            mcp_timed_out = True
            raise
        return {'project_id': 'p', 'count': 0, 'tickets': []}

    with (
        _override_client(config) as c,
        patch('dashboard.app.get_curator_sparks', new=_mock_sparks),
        patch(_PATCH_TARGET, new=_mock_mcp),
        patch(_PATCH_CURATOR_STATE, new=AsyncMock(return_value=_HEALTHY_GATE)),
    ):
        resp = c.get('/api/v2/dashboard/curator')

    assert mcp_timed_out is False, (
        'mcp_timed_out is True — fan_out ran before get_curator_sparks started; '
        'api_curator must run fan_out concurrently with DB queries'
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# task-1336 step-1: _LIST_TICKETS_LIMIT constant flows through api_curator
# ---------------------------------------------------------------------------


def test_api_curator_uses_LIST_TICKETS_LIMIT_constant(tmp_path: Path, monkeypatch):
    """api_curator must not pass a magic literal limit to fan_out_list_tickets.

    Monkeypatches _LIST_TICKETS_LIMIT to 50. Patches mcp_tool_call to capture
    the outgoing arguments['limit']. GETs /api/v2/dashboard/curator. Asserts the
    captured limit equals 50 (the monkeypatched constant).

    Failing baseline: app.py passes limit=2000 to fan_out_list_tickets. Since
    limit is non-None, effective_limit stays 2000 regardless of the monkeypatch.
    The captured payload limit is 2000, not 50 — test fails with a clear message
    pointing at app.py.
    """
    import dashboard.data.metrics as _metrics_mod

    monkeypatch.setattr(_metrics_mod, '_LIST_TICKETS_LIMIT', 50)

    captured: dict = {}

    async def _mock_mcp(client, base_url, tool_name, arguments):
        captured['limit'] = arguments.get('limit')
        return {'project_id': 'p', 'count': 0, 'tickets': []}

    config = _make_config(tmp_path)
    with (
        _override_client(config) as c,
        patch(_PATCH_TARGET, new=_mock_mcp),
        patch(_PATCH_CURATOR_STATE, new=AsyncMock(return_value=_HEALTHY_GATE)),
    ):
        resp = c.get('/api/v2/dashboard/curator')

    assert resp.status_code == 200
    assert captured.get('limit') == 50, (
        f'Expected captured limit == 50 (the monkeypatched _LIST_TICKETS_LIMIT), '
        f'got {captured.get("limit")!r}. A non-50 value (e.g. 2000) means '
        'app.py is passing a magic literal that overrides the constant — '
        'remove the limit=2000 kwarg from the fan_out_list_tickets call in app.py.'
    )


# ---------------------------------------------------------------------------
# task-1329 step-1: per-leg DB degradation — OSError from pool.get stays
# contained to the sparks/intervals legs (fan-out leg unaffected)
# ---------------------------------------------------------------------------


def test_api_curator_db_resolution_failure_degrades_per_leg(tmp_path: Path):
    """DB resolution failure must degrade only sparks/intervals legs, not 500.

    RED baseline: `metrics_db = await pool.get(...)` and
    `cost_dbs_raw = await _cost_dbs(...)` run BEFORE asyncio.gather with no
    error handling.  An OSError from pool.get propagates out of api_curator
    → 500.  The fix moves these resolutions into per-leg wrapper coroutines
    inside the gather so safe_gather_result absorbs the failure per-leg.

    NOTE: This is a *structural* regression guard, not a production-realistic
    failure scenario.  In production, DbPool.get already catches OSError
    internally (db.py:55) and returns None, so OSError from pool.get cannot
    reach api_curator.  The test patches the entire DbPool.get method, bypassing
    its own error handling, to verify that any unguarded exception raised by the
    DB resolution code (from whatever cause) stays contained to the affected leg
    rather than propagating as a 500.

    DbPool.get is patched at the class (dashboard.data.db.DbPool.get) because
    api_curator resolves pool.get on the app.state.db instance — patching the
    class method covers metrics_db AND _cost_dbs→_project_scoped_dbs in one shot.

    Expected post-fix behaviour:
    - status 200 (no 500)
    - cs['pending'] has the one row returned by the fan-out mock (DB failure
      must NOT affect the fan-out leg which uses mcp_tool_call, not pool.get)
    - cs['state']['pending_total'] == 1
    - cs['latency_spark'], cs['pending_spark'], cs['capped_spark'] are all
      empty series (DB leg failures → safe_gather_result returns defaults)
    - cs['state']['capped_now'] == 0
    """
    now = datetime.now(UTC)
    created_at = (now - timedelta(seconds=60)).isoformat()
    ticket_row = {
        'ticket_id': 'tkt_a',
        'candidate_title': 'A',
        'created_at': created_at,
        'project_id': 'proj_p',
    }
    mcp_result = {'project_id': 'proj_p', 'count': 1, 'tickets': [ticket_row]}

    config = _make_config(tmp_path)
    with (
        _override_client(config) as c,
        patch(_PATCH_TARGET, new=AsyncMock(return_value=mcp_result)),
        patch(
            'dashboard.data.db.DbPool.get',
            new=AsyncMock(side_effect=OSError('simulated transient FS error')),
        ),
        patch(_PATCH_CURATOR_STATE, new=AsyncMock(return_value=_HEALTHY_GATE)),
    ):
        resp = c.get('/api/v2/dashboard/curator')

    assert resp.status_code == 200, (
        f'Expected 200 (per-leg degradation), got {resp.status_code}. '
        'pool.get OSError is propagating out of api_curator before the gather.'
    )
    cs = resp.json()['CURATOR_STATE']

    # Fan-out leg is unaffected (uses mcp_tool_call, not pool.get).
    assert len(cs['pending']) == 1
    assert cs['pending'][0]['ticket_id'] == 'tkt_a'
    assert cs['state']['pending_total'] == 1

    # DB legs degrade gracefully to empty series.
    ls = cs['latency_spark']
    assert ls['labels'] == []
    assert ls['p50'] == []
    assert ls['p90'] == []
    assert ls['p99'] == []
    ps = cs['pending_spark']
    assert ps['labels'] == []
    assert ps['values'] == []
    # capped_spark always generates the full 24h time-bucket grid (144 labels)
    # even for empty intervals; what matters is that no bucket shows a cap event
    # and capped_now is 0 (no open-ended interval from a DB that failed to load).
    capped = cs['capped_spark']
    assert 1 not in capped['values'], 'Expected no cap events (all zeros) when intervals leg failed'
    assert cs['state']['capped_now'] == 0


# ---------------------------------------------------------------------------
# step-11: paused_reason flows from MCP get_curator_state tool
# ---------------------------------------------------------------------------


def test_curator_endpoint_flows_paused_reason_from_mcp(tmp_path: Path):
    """paused_reason from get_curator_state helper appears in curator state envelope.

    Patches both the list_tickets fan-out (metrics.mcp_tool_call) and the
    get_curator_state helper so the endpoint resolves without a real server.
    Verifies that paused_reason from the gate reaches CURATOR_STATE.state.
    """
    config = _make_config(tmp_path, fused_memory_urls=['http://localhost:18765'])
    mcp_tickets = {'project_id': 'p', 'count': 0, 'tickets': []}
    curator_state_payload = {
        'paused': True,
        'paused_reason': 'All accounts capped (last: synthetic cap)',
        'soonest_open_at': '2026-06-01T00:00:00+00:00',
        'account_count': 1,
    }

    with (
        patch(_PATCH_TARGET, new=AsyncMock(return_value=mcp_tickets)),
        patch('dashboard.app.memory_data.get_curator_state',
              new=AsyncMock(return_value=curator_state_payload)),
        _override_client(config) as client,
    ):
        resp = client.get('/api/v2/dashboard/curator')

    assert resp.status_code == 200
    data = resp.json()
    state = data['CURATOR_STATE']['state']
    assert state['paused_reason'] == 'All accounts capped (last: synthetic cap)', (
        f'Expected paused_reason to flow through from MCP tool; got {state["paused_reason"]!r}'
    )


def test_curator_endpoint_paused_reason_none_when_helper_offline(tmp_path: Path):
    """paused_reason is None when the get_curator_state helper is offline.

    Ensures a failing curator-state leg degrades gracefully to paused_reason=None
    rather than propagating an error or returning an unexpected value.
    """
    config = _make_config(tmp_path, fused_memory_urls=['http://localhost:18765'])
    mcp_tickets = {'project_id': 'p', 'count': 0, 'tickets': []}
    offline_payload = {'offline': True, 'error': 'all unreachable'}

    with (
        patch(_PATCH_TARGET, new=AsyncMock(return_value=mcp_tickets)),
        patch('dashboard.app.memory_data.get_curator_state',
              new=AsyncMock(return_value=offline_payload)),
        _override_client(config) as client,
    ):
        resp = client.get('/api/v2/dashboard/curator')

    assert resp.status_code == 200
    data = resp.json()
    state = data['CURATOR_STATE']['state']
    assert state['paused_reason'] is None, (
        f'Expected paused_reason=None when gate helper offline; got {state["paused_reason"]!r}'
    )


# ---------------------------------------------------------------------------
# step-5 (task-1510): shape_curator accounts_summary kwarg
# ---------------------------------------------------------------------------


def test_shape_curator_passes_accounts_summary():
    """shape_curator propagates accounts_summary kwarg into the state dict.

    RED baseline: shape_curator does not accept accounts_summary -> TypeError.
    """
    from dashboard.data.redux_api import shape_curator

    curator_sparks = {
        'p50': {'labels': [], 'values': []},
        'p90': {'labels': [], 'values': []},
        'p99': {'labels': [], 'values': []},
        'pending': {'labels': [], 'values': []},
    }
    capped_spark = {'labels': [], 'values': []}
    accounts_summary = {'total': 3, 'capped': 1, 'available': 2, 'capped_accounts': ['a'], 'account_names': ['a']}

    result = shape_curator(
        pending=[],
        curator_sparks=curator_sparks,
        capped_spark=capped_spark,
        capped_now=0,
        paused_reason=None,
        pending_total=0,
        accounts_summary=accounts_summary,
    )

    assert 'CURATOR_STATE' in result
    state = result['CURATOR_STATE']['state']
    assert state['accounts_summary'] == {'total': 3, 'capped': 1, 'available': 2, 'capped_accounts': ['a'], 'account_names': ['a']}, (
        f'Expected accounts_summary to flow through shape_curator, got {state.get("accounts_summary")!r}'
    )


def test_shape_curator_default_accounts_summary():
    """Omitting accounts_summary kwarg yields the empty default dict."""
    from dashboard.data.redux_api import shape_curator

    curator_sparks = {
        'p50': {'labels': [], 'values': []},
        'p90': {'labels': [], 'values': []},
        'p99': {'labels': [], 'values': []},
        'pending': {'labels': [], 'values': []},
    }
    capped_spark = {'labels': [], 'values': []}

    result = shape_curator(
        pending=[],
        curator_sparks=curator_sparks,
        capped_spark=capped_spark,
        capped_now=0,
        paused_reason=None,
        pending_total=0,
        # accounts_summary deliberately omitted
    )

    state = result['CURATOR_STATE']['state']
    assert state['accounts_summary'] == {'total': 0, 'capped': 0, 'available': 0, 'capped_accounts': [], 'account_names': []}, (
        f'Expected default empty accounts_summary when kwarg omitted, got {state.get("accounts_summary")!r}'
    )


# ---------------------------------------------------------------------------
# step-7 (task-1510): end-to-end account_count denominator + days=7 tests
# ---------------------------------------------------------------------------


def _seed_runs_db(tmp_path: Path, events: list[tuple[str, str, datetime]]) -> Path:
    """Create runs.db at the canonical path and insert account_events; return path."""
    runs_dir = tmp_path / 'data' / 'orchestrator'
    runs_dir.mkdir(parents=True, exist_ok=True)
    runs_db_path = runs_dir / 'runs.db'
    conn = sqlite3.connect(str(runs_db_path))
    conn.executescript(ACCOUNT_EVENTS_SCHEMA)
    for account, event_type, ts in events:
        conn.execute(
            'INSERT INTO account_events (account_name, event_type, created_at) VALUES (?, ?, ?)',
            (account, event_type, ts.isoformat()),
        )
    conn.commit()
    conn.close()
    return runs_db_path


def test_api_curator_account_count_denominator_not_capped(tmp_path: Path):
    """THE bug-fix test: 1 open cap of 4 configured accounts → capped_now=0.

    Scenario: ONE account has a stale open-ended cap_hit while 3 healthy accounts
    have NO recent cap events (so they never appear in intervals).

    Under the OLD any-account semantics: capped_now=1 (acc-a is open → bug).
    Under the NEW all-accounts-capped semantics with account_count=4:
    total=4, capped=1, available=3 → capped_now=0.

    RED baseline: app.py calls compute_capped_now_and_windows(intervals) without
    total_accounts, so total=1, available=0 → capped_now=1 (bug not fixed yet).
    """
    now = datetime.now(UTC)
    cap_hit_time = now - timedelta(minutes=10)

    _seed_runs_db(tmp_path, [('acc-a', 'cap_hit', cap_hit_time)])

    gate = {
        'paused': False,
        'paused_reason': None,
        'soonest_open_at': None,
        'account_count': 4,
    }
    config = _make_config(tmp_path)
    empty_result = AsyncMock(return_value={'project_id': 'p', 'count': 0, 'tickets': []})

    with (
        _override_client(config) as c,
        patch(_PATCH_TARGET, new=empty_result),
        patch(_PATCH_CURATOR_STATE, new=AsyncMock(return_value=gate)),
    ):
        resp = c.get('/api/v2/dashboard/curator')

    assert resp.status_code == 200
    state = resp.json()['CURATOR_STATE']['state']
    assert state['capped_now'] == 0, (
        f'1 open cap of 4 configured accounts must NOT read Capped; '
        f'account_count is the denominator. capped_now={state["capped_now"]} means '
        f'the fix ignored account_count — the original 3-uncapped-accounts bug.'
    )
    assert state['accounts_summary'] == {
        'total': 4,
        'capped': 1,
        'available': 3,
        'capped_accounts': ['acc-a'],
        'account_names': ['acc-a'],
    }, f'Expected accounts_summary with total=4, got {state.get("accounts_summary")!r}'


def test_api_curator_all_accounts_capped_pins_one(tmp_path: Path):
    """Both accounts capped → capped_now=1 and accounts_summary shows zero available."""
    now = datetime.now(UTC)
    cap_time_a = now - timedelta(minutes=15)
    cap_time_b = now - timedelta(minutes=10)

    _seed_runs_db(tmp_path, [
        ('acc-a', 'cap_hit', cap_time_a),
        ('acc-b', 'cap_hit', cap_time_b),
    ])

    gate = {
        'paused': False,
        'paused_reason': None,
        'soonest_open_at': None,
        'account_count': 2,
    }
    config = _make_config(tmp_path)
    empty_result = AsyncMock(return_value={'project_id': 'p', 'count': 0, 'tickets': []})

    with (
        _override_client(config) as c,
        patch(_PATCH_TARGET, new=empty_result),
        patch(_PATCH_CURATOR_STATE, new=AsyncMock(return_value=gate)),
    ):
        resp = c.get('/api/v2/dashboard/curator')

    assert resp.status_code == 200
    state = resp.json()['CURATOR_STATE']['state']
    assert state['capped_now'] == 1, (
        f'Both accounts capped → expected capped_now=1, got {state["capped_now"]}'
    )
    summary = state['accounts_summary']
    assert summary['available'] == 0, f'Expected available=0, got {summary}'
    assert summary['capped'] == 2, f'Expected capped=2, got {summary}'
    assert summary['total'] == 2, f'Expected total=2, got {summary}'


def test_api_curator_uses_days_seven_lookback(tmp_path: Path):
    """Cap from 2 days ago must be visible (requires days=7 lookback).

    RED baseline: with days=1 the 2-day-old cap_hit is filtered out
    (intervals=[]) so capped=0/available=1/capped_now=0.

    The days=7 bump in step-8 includes the 2-day-old event so capped_now=1.
    """
    now = datetime.now(UTC)
    old_cap_time = now - timedelta(days=2)  # outside 1-day window, inside 7-day window

    _seed_runs_db(tmp_path, [('acc-a', 'cap_hit', old_cap_time)])

    gate = {
        'paused': False,
        'paused_reason': None,
        'soonest_open_at': None,
        'account_count': 1,
    }
    config = _make_config(tmp_path)
    empty_result = AsyncMock(return_value={'project_id': 'p', 'count': 0, 'tickets': []})

    with (
        _override_client(config) as c,
        patch(_PATCH_TARGET, new=empty_result),
        patch(_PATCH_CURATOR_STATE, new=AsyncMock(return_value=gate)),
    ):
        resp = c.get('/api/v2/dashboard/curator')

    assert resp.status_code == 200
    state = resp.json()['CURATOR_STATE']['state']
    assert state['capped_now'] == 1, (
        f'2-day-old open-ended cap must be visible with days=7 lookback. '
        f'capped_now={state["capped_now"]} means the cap was excluded — '
        f'days=1 cutoff is still in place (the 1-day lookback bug).'
    )
    assert state['accounts_summary'] == {
        'total': 1,
        'capped': 1,
        'available': 0,
        'capped_accounts': ['acc-a'],
        'account_names': ['acc-a'],
    }, f'Expected accounts_summary with total=1, got {state.get("accounts_summary")!r}'


# ---------------------------------------------------------------------------
# step-17 (task-1814): accounts_summary exception must log at WARNING
# ---------------------------------------------------------------------------


def test_api_curator_accounts_summary_failure_logs_warning(tmp_path: Path, caplog):
    """When summarize_accounts raises, api_curator logs a WARNING (not DEBUG).

    Fails today because the except block calls logger.debug (invisible at INFO).
    After the fix, a WARNING containing 'accounts_summary' must appear in caplog.
    The response must still be 200 with an empty accounts_summary (graceful degrade).
    """
    import logging

    config = _make_config(tmp_path)
    empty_result = AsyncMock(return_value={'project_id': 'p', 'count': 0, 'tickets': []})

    with (
        _override_client(config) as c,
        patch(_PATCH_TARGET, new=empty_result),
        patch(_PATCH_CURATOR_STATE, new=AsyncMock(return_value=_HEALTHY_GATE)),
        patch(
            'dashboard.app.summarize_accounts',
            side_effect=RuntimeError('simulated summarize_accounts failure'),
        ),
        caplog.at_level(logging.WARNING, logger='dashboard.app'),
    ):
        resp = c.get('/api/v2/dashboard/curator')

    assert resp.status_code == 200, (
        f'Expected 200 (graceful degrade), got {resp.status_code}'
    )
    state = resp.json()['CURATOR_STATE']['state']
    # Degrade to empty accounts_summary.
    assert state['accounts_summary'] == {
        'total': 0,
        'capped': 0,
        'available': 0,
        'capped_accounts': [],
        'account_names': [],
    }, f'Expected empty accounts_summary on failure, got {state.get("accounts_summary")!r}'

    # A WARNING mentioning accounts_summary must have been emitted.
    warning_records = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and 'accounts_summary' in r.message
    ]
    assert warning_records, (
        'Expected a WARNING mentioning accounts_summary when summarize_accounts raises, '
        f'but only found: {[r.message for r in caplog.records]}'
    )
