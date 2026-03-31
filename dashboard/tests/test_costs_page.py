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
