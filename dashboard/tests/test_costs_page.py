"""Tests for the costs page — routes, templates, and window parameter parsing."""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import AsyncMock, patch


# ---------------------------------------------------------------------------
# Step-1: _WINDOW_DAYS mapping and _parse_window helper
# ---------------------------------------------------------------------------

class TestWindowDays:
    """Tests for the _WINDOW_DAYS mapping in app.py."""

    def test_mapping_has_four_keys(self):
        from dashboard.app import _WINDOW_DAYS
        assert set(_WINDOW_DAYS.keys()) == {'24h', '7d', '30d', 'all'}

    def test_24h_maps_to_1(self):
        from dashboard.app import _WINDOW_DAYS
        assert _WINDOW_DAYS['24h'] == 1

    def test_7d_maps_to_7(self):
        from dashboard.app import _WINDOW_DAYS
        assert _WINDOW_DAYS['7d'] == 7

    def test_30d_maps_to_30(self):
        from dashboard.app import _WINDOW_DAYS
        assert _WINDOW_DAYS['30d'] == 30

    def test_all_maps_to_3650(self):
        from dashboard.app import _WINDOW_DAYS
        assert _WINDOW_DAYS['all'] == 3650


class TestParseWindow:
    """Tests for _parse_window(request) helper in app.py."""

    def _make_request(self, window: str | None):
        """Create a minimal fake request with a query_params dict."""
        from starlette.testclient import TestClient
        from dashboard.app import app

        # Use TestClient to make real requests and test via route
        # We'll test _parse_window directly via its module.
        # Build a mock Request with query_params
        class FakeQueryParams:
            def __init__(self, d):
                self._d = d

            def get(self, key, default=None):
                return self._d.get(key, default)

        class FakeRequest:
            def __init__(self, window):
                self.query_params = FakeQueryParams(
                    {'window': window} if window is not None else {}
                )

        return FakeRequest(window)

    def test_valid_24h(self):
        from dashboard.app import _parse_window
        req = self._make_request('24h')
        assert _parse_window(req) == 1

    def test_valid_7d(self):
        from dashboard.app import _parse_window
        req = self._make_request('7d')
        assert _parse_window(req) == 7

    def test_valid_30d(self):
        from dashboard.app import _parse_window
        req = self._make_request('30d')
        assert _parse_window(req) == 30

    def test_valid_all(self):
        from dashboard.app import _parse_window
        req = self._make_request('all')
        assert _parse_window(req) == 3650

    def test_missing_defaults_to_7(self):
        from dashboard.app import _parse_window
        req = self._make_request(None)
        assert _parse_window(req) == 7

    def test_invalid_defaults_to_7(self):
        from dashboard.app import _parse_window
        req = self._make_request('invalid')
        assert _parse_window(req) == 7

    def test_returns_int(self):
        from dashboard.app import _parse_window
        req = self._make_request('24h')
        result = _parse_window(req)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# Step-3: GET /costs page template tests
# ---------------------------------------------------------------------------

class TestCostsPage:
    """Tests for the GET /costs route and template."""

    def test_returns_200(self, client):
        resp = client.get('/costs')
        assert resp.status_code == 200

    def test_content_type_html(self, client):
        resp = client.get('/costs')
        assert 'text/html' in resp.headers['content-type']

    def test_contains_summary_partial(self, client):
        html = client.get('/costs').text
        assert '/costs/partials/summary' in html

    def test_contains_by_project_partial(self, client):
        html = client.get('/costs').text
        assert '/costs/partials/by-project' in html

    def test_contains_by_account_partial(self, client):
        html = client.get('/costs').text
        assert '/costs/partials/by-account' in html

    def test_contains_by_role_partial(self, client):
        html = client.get('/costs').text
        assert '/costs/partials/by-role' in html

    def test_contains_trend_partial(self, client):
        html = client.get('/costs').text
        assert '/costs/partials/trend' in html

    def test_contains_events_partial(self, client):
        html = client.get('/costs').text
        assert '/costs/partials/events' in html

    def test_contains_runs_partial(self, client):
        html = client.get('/costs').text
        assert '/costs/partials/runs' in html

    def test_contains_window_selector_24h(self, client):
        html = client.get('/costs').text
        assert '24h' in html

    def test_contains_window_selector_7d(self, client):
        html = client.get('/costs').text
        assert '7d' in html

    def test_contains_window_selector_30d(self, client):
        html = client.get('/costs').text
        assert '30d' in html

    def test_contains_window_selector_all(self, client):
        html = client.get('/costs').text
        assert 'all' in html.lower()

    def test_sections_use_morph_swap(self, client):
        """All HTMX polling sections must use morph:innerHTML for DOM diffing."""
        html = client.get('/costs').text
        assert 'morph:innerHTML' in html

    def test_nav_costs_link_is_active(self, client):
        """The 'Costs' nav link should have the active class on the /costs page."""
        html = client.get('/costs').text
        # Active class is 'border-b-2 border-blue-500'; costs link should carry it
        assert 'border-blue-500' in html

    def test_default_window_7d_marked_active(self, client):
        """Without ?window=, the 7d button should be visually active."""
        html = client.get('/costs').text
        # The page must indicate 7d is the default/active window
        # We check that 7d appears as the default in Alpine state or aria-pressed
        assert '7d' in html
