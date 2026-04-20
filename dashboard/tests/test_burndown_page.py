"""Tests for the burndown page — routes, templates, and window parameters."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# _BURNDOWN_WINDOWS mapping
# ---------------------------------------------------------------------------


class TestBurndownWindows:
    def test_mapping_has_four_keys(self):
        from dashboard.app import _BURNDOWN_WINDOWS
        assert set(_BURNDOWN_WINDOWS.keys()) == {'24h', '7d', '30d', '90d'}

    def test_24h_maps_to_1(self):
        from dashboard.app import _BURNDOWN_WINDOWS
        assert _BURNDOWN_WINDOWS['24h'] == 1

    def test_7d_maps_to_7(self):
        from dashboard.app import _BURNDOWN_WINDOWS
        assert _BURNDOWN_WINDOWS['7d'] == 7

    def test_30d_maps_to_30(self):
        from dashboard.app import _BURNDOWN_WINDOWS
        assert _BURNDOWN_WINDOWS['30d'] == 30

    def test_90d_maps_to_90(self):
        from dashboard.app import _BURNDOWN_WINDOWS
        assert _BURNDOWN_WINDOWS['90d'] == 90


# ---------------------------------------------------------------------------
# Route tests
# ---------------------------------------------------------------------------


class TestBurndownRoute:
    def test_burndown_page_returns_200(self, client):
        resp = client.get('/burndown')
        assert resp.status_code == 200

    def test_burndown_page_contains_title(self, client):
        resp = client.get('/burndown')
        assert 'Burndown' in resp.text

    def test_burndown_page_default_window(self, client):
        resp = client.get('/burndown')
        # Alpine store should be initialized with '7d'
        assert '7d' in resp.text

    def test_burndown_page_custom_window(self, client):
        resp = client.get('/burndown?window=30d')
        assert resp.status_code == 200

    def test_burndown_page_invalid_window_defaults(self, client):
        resp = client.get('/burndown?window=invalid')
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Nav link
# ---------------------------------------------------------------------------


class TestNavLink:
    def test_burndown_link_in_nav(self, client):
        resp = client.get('/burndown')
        assert 'href="/burndown"' in resp.text

    def test_burndown_link_on_index(self, client):
        resp = client.get('/')
        assert 'href="/burndown"' in resp.text

    def test_burndown_link_on_costs(self, client):
        resp = client.get('/costs')
        assert 'href="/burndown"' in resp.text


# ---------------------------------------------------------------------------
# Helpers for multi-project burndown route tests
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402 (after class definitions)
import sqlite3  # noqa: E402
from datetime import UTC, datetime, timedelta  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import patch  # noqa: E402

from starlette.testclient import TestClient  # noqa: E402


def _make_burndown_db(db_path: Path, project_id: str) -> None:
    """Create a burndown.db at *db_path* with one recent snapshot for *project_id*."""
    from dashboard.data.burndown import _INSERT_SNAPSHOT_SQL, BURNDOWN_SCHEMA

    db_path.parent.mkdir(parents=True, exist_ok=True)
    ts = (datetime.now(UTC) - timedelta(minutes=30)).isoformat()
    conn = sqlite3.connect(str(db_path))
    conn.executescript(BURNDOWN_SCHEMA)
    conn.execute(_INSERT_SNAPSHOT_SQL, (project_id, ts, 0, 0, 0, 0, 0, 3))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Multi-project burndown charts route tests
# ---------------------------------------------------------------------------


class TestBurndownChartsRouteMultiProject:
    """Route-level tests for /burndown/partials/charts with multiple project roots.

    Before step-16 the route queries only the main burndown.db and misses the
    peer project.  After step-16 it calls the aggregate helpers and surfaces
    both project_ids.
    """

    def test_both_project_ids_in_html(self, tmp_path):
        """Both project_ids appear in HTML when two burndown DBs exist.

        FAILS before step-16 (route only queries main burndown.db).
        PASSES after step-16 (route calls aggregate helpers across both DBs).
        """
        from dashboard.app import app
        from dashboard.config import DashboardConfig

        main_root = tmp_path / 'main'
        peer_root = tmp_path / 'peer'

        _make_burndown_db(main_root / 'data' / 'burndown' / 'burndown.db', 'dark_factory')
        _make_burndown_db(peer_root / 'data' / 'burndown' / 'burndown.db', 'peer_project')

        config = DashboardConfig(
            project_root=main_root,
            known_project_roots=[peer_root],
        )

        with patch('dashboard.app.DashboardConfig.from_env', return_value=config), TestClient(app) as c:
            html = c.get('/burndown/partials/charts?window=30d').text

        assert 'dark_factory' in html
        assert 'peer_project' in html

    def test_single_project_unchanged(self, tmp_path):
        """Single-project behavior is unchanged when known_project_roots=[]."""
        from dashboard.app import app
        from dashboard.config import DashboardConfig

        main_root = tmp_path / 'main'
        _make_burndown_db(main_root / 'data' / 'burndown' / 'burndown.db', 'dark_factory')

        config = DashboardConfig(project_root=main_root, known_project_roots=[])

        with patch('dashboard.app.DashboardConfig.from_env', return_value=config), TestClient(app) as c:
            html = c.get('/burndown/partials/charts?window=30d').text

        assert 'dark_factory' in html


# ---------------------------------------------------------------------------
# Parallelism sanity test for burndown_partials_charts
# ---------------------------------------------------------------------------


class TestBurndownChartsRouteParallelism:
    """Verify that burndown_partials_charts fetches series concurrently across projects.

    The test patches aggregate_burndown_series with a fake that tracks how many
    calls are in-flight simultaneously.  If the handler gathers all projects, the
    counter reaches 3; if it awaits sequentially, the counter never exceeds 1.
    This tests the real concurrency property directly without depending on the
    handler's except-clause behavior or a slow timeout.
    """

    def test_per_project_calls_are_concurrent(self, tmp_path):
        """Max simultaneous in-flight calls equals the number of projects.

        FAILS before the gather fix (sequential await never has more than 1
        in-flight call at once, max_concurrent stays at 1).
        PASSES after the gather fix (all three coroutines overlap, max_concurrent
        reaches 3).
        """
        from dashboard.app import app
        from dashboard.config import DashboardConfig

        # Three project roots, each with its own burndown.db and project_id.
        roots = []
        for name in ('alpha', 'beta', 'gamma'):
            root = tmp_path / name
            _make_burndown_db(root / 'data' / 'burndown' / 'burndown.db', name)
            roots.append(root)

        config = DashboardConfig(
            project_root=roots[0],
            known_project_roots=roots[1:],
        )

        # Shared state (lists because Python closures can't rebind bare ints).
        in_flight = [0]
        max_concurrent = [0]

        async def fake_aggregate_burndown_series(dbs, project_id, *, days):
            in_flight[0] += 1
            max_concurrent[0] = max(max_concurrent[0], in_flight[0])
            # Yield control several times so other coroutines get scheduled.
            for _ in range(5):
                await asyncio.sleep(0)
            in_flight[0] -= 1
            # Return a valid empty-series dict so the template still renders.
            return {'labels': [], 'done': [], 'cancelled': [], 'blocked': [],
                    'deferred': [], 'in_progress': [], 'pending': []}

        with (
            patch('dashboard.app.DashboardConfig.from_env', return_value=config),
            patch('dashboard.app.aggregate_burndown_series', new=fake_aggregate_burndown_series),
            TestClient(app) as c,
        ):
            resp = c.get('/burndown/partials/charts?window=30d')

        assert resp.status_code == 200
        assert max_concurrent[0] == 3, (
            f'per-project aggregate_burndown_series calls did not run concurrently '
            f'— max simultaneous in-flight: {max_concurrent[0]}, expected 3 '
            f'(sequential await would give max_concurrent=1)'
        )
