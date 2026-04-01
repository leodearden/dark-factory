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
