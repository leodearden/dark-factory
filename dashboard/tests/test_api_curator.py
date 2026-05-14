"""Tests for GET /api/v2/dashboard/curator endpoint + shape_curator.

TDD steps:
  step-1  envelope shape test (RED before route exists)
  step-3  pending list fan-out test (RED before real fan-out)
  step-5  latency_spark from metrics.db (RED before spark wiring)
  step-7  capped_spark from runs.db (RED before cap interval wiring)
  step-9  capped_now open-ended interval + pure shape_curator unit test
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

from dashboard.data.metrics import METRICS_SCHEMA

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# fan_out_list_tickets (in metrics.py) resolves mcp_tool_call from the
# dashboard.data.metrics namespace, so patch there — not dashboard.data.memory.
_PATCH_TARGET = 'dashboard.data.metrics.mcp_tool_call'

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
    """
    mcp_result = {'project_id': 'p', 'count': 0, 'tickets': []}
    with patch(_PATCH_TARGET, new=AsyncMock(return_value=mcp_result)):
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
    capped = cs.get('capped_spark', {})
    assert 'labels' in capped
    assert 'values' in capped
    state = cs.get('state', {})
    assert 'capped_now' in state
    assert 'paused_reason' in state
    assert state['paused_reason'] is None  # pinned to None; see app.py paused_reason TODO
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
    with _override_client(config) as c, patch(_PATCH_TARGET, new=AsyncMock(return_value=mcp_result)):
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
    with _override_client(config) as c, patch(_PATCH_TARGET, new=empty_result):
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
    with _override_client(config) as c, patch(_PATCH_TARGET, new=empty_result):
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
    with _override_client(config) as c, patch(_PATCH_TARGET, new=empty_result):
        resp = c.get('/api/v2/dashboard/curator')

    assert resp.status_code == 200
    assert resp.json()['CURATOR_STATE']['state']['capped_now'] == 1


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

    # capped_spark passes through
    assert cs['capped_spark'] == capped_spark

    # state block
    state = cs['state']
    assert state['capped_now'] == 1
    assert state['paused_reason'] == 'manual'
    assert state['pending_total'] == 42


# ---------------------------------------------------------------------------
# step-1299-4: bad created_at values → age_seconds=None, no 500
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('created_at_value', ['not-an-iso-date', ''])
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
    with _override_client(config) as c, patch(_PATCH_TARGET, new=AsyncMock(return_value=mcp_result)):
        resp = c.get('/api/v2/dashboard/curator')

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    cs = resp.json()['CURATOR_STATE']
    assert len(cs['pending']) == 1, f"Expected 1 pending row, got {len(cs['pending'])}"
    row = cs['pending'][0]
    assert row['ticket_id'] == 'tkt_x'
    assert row['created_at'] == created_at_value, (
        f"created_at should be preserved verbatim, got {row['created_at']!r}"
    )
    assert row['age_seconds'] is None, (
        f"age_seconds should be None for bad created_at, got {row['age_seconds']!r}"
    )
