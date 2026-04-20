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

import sqlite3  # noqa: E402 (after class definitions)
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


import asyncio  # noqa: E402
import pytest  # noqa: E402


class TestBurndownChartsRouteParallelism:
    """Verify that burndown_partials_charts fetches series concurrently across projects.

    The test patches aggregate_burndown_series with a fake that awaits an
    asyncio.Barrier(3).  If the handler loops sequentially, only one coroutine
    reaches the barrier at a time and the wait_for raises TimeoutError — the
    AssertionError propagates to the handler's try/except, series becomes {},
    and none of the project IDs appear in the response HTML.  If asyncio.gather
    is used, all three reach the barrier simultaneously, it releases within the
    timeout, each fake returns valid data, and all project IDs appear in the HTML.
    """

    def test_per_project_calls_are_concurrent(self, tmp_path):
        """All project IDs appear in HTML only when calls run concurrently via gather.

        FAILS before the gather fix (sequential await causes barrier timeout,
        handler catches the error, series={}, HTML has no project data).
        PASSES after the gather fix (all three coroutines reach the barrier
        simultaneously, valid data returned, HTML contains all project IDs).
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

        n_projects = 3
        barrier = asyncio.Barrier(n_projects)

        async def fake_aggregate_burndown_series(dbs, project_id, *, days):
            try:
                await asyncio.wait_for(barrier.wait(), timeout=2.0)
            except (asyncio.TimeoutError, asyncio.BrokenBarrierError) as exc:
                raise AssertionError(
                    'per-project aggregate_burndown_series calls did not run '
                    'concurrently — expected asyncio.gather across projects '
                    f'(barrier raised {type(exc).__name__})'
                ) from exc
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
        # All three project IDs must appear in the rendered HTML.
        # In the sequential case the barrier times out, the handler's try/except
        # catches the AssertionError, series={}, and the HTML has no project data.
        # In the parallel (gather) case all three succeed and the template renders
        # each project_id in an <h3> tag via the project_name filter.
        html = resp.text
        assert 'alpha' in html, (
            'project "alpha" not in HTML — per-project calls likely ran sequentially '
            '(barrier timed out for the first call, handler caught the error, series={})'
        )
        assert 'beta' in html, 'project "beta" not in HTML — per-project calls likely ran sequentially'
        assert 'gamma' in html, 'project "gamma" not in HTML — per-project calls likely ran sequentially'
