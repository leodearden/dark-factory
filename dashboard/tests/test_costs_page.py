"""Tests for the costs page — routes, templates, and window parameter parsing."""

from __future__ import annotations

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
        # Build a duck-typed fake request with query_params for _parse_window
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
        """All 7 HTMX polling sections must use morph:innerHTML for DOM diffing."""
        html = client.get('/costs').text
        assert html.count('morph:innerHTML') == 7

    def test_nav_costs_link_is_active(self, client):
        """The 'Costs' nav link should have the active class on the /costs page."""
        html = client.get('/costs').text
        # Active class is 'border-b-2 border-blue-500'; costs link should carry it
        assert 'border-blue-500' in html

    def test_default_window_7d_marked_active(self, client):
        """Without ?window=, the 7d button should be visually active."""
        html = client.get('/costs').text
        # The page must indicate 7d is the default/active window via Alpine.store init
        assert "Alpine.store('costs', { window: \"7d\" })" in html


# ---------------------------------------------------------------------------
# Step-5: GET /costs/partials/summary route tests
# ---------------------------------------------------------------------------

_MOCK_SUMMARY = {
    'dark_factory': {
        'total_spend': 12.34,
        'task_count': 22,
        'avg_cost_per_task': 0.56,
        'active_accounts': 3,
        'cap_events': 2,
    },
    'other_project': {
        'total_spend': 5.00,
        'task_count': 20,
        'avg_cost_per_task': 0.25,
        'active_accounts': 1,
        'cap_events': 0,
    },
}


def _patch_summary(return_value=_MOCK_SUMMARY):
    return patch(
        'dashboard.app.get_cost_summary',
        new_callable=AsyncMock,
        return_value=return_value,
    )


class TestCostsSummaryPartial:
    """Tests for GET /costs/partials/summary."""

    def test_returns_200(self, client):
        with _patch_summary():
            resp = client.get('/costs/partials/summary')
        assert resp.status_code == 200

    def test_content_type_html(self, client):
        with _patch_summary():
            resp = client.get('/costs/partials/summary')
        assert 'text/html' in resp.headers['content-type']

    def test_renders_total_spend(self, client):
        with _patch_summary():
            html = client.get('/costs/partials/summary').text
        # Should show a dollar amount for combined spend (12.34 + 5.00 = 17.34 or per-project)
        assert '$' in html

    def test_renders_active_accounts(self, client):
        with _patch_summary():
            html = client.get('/costs/partials/summary').text
        # Should show account count
        assert 'account' in html.lower()

    def test_renders_cap_events(self, client):
        with _patch_summary():
            html = client.get('/costs/partials/summary').text
        # Should show cap event info
        assert 'cap' in html.lower()

    def test_handles_empty_data(self, client):
        with _patch_summary(return_value={}):
            resp = client.get('/costs/partials/summary')
        assert resp.status_code == 200

    def test_respects_window_param(self, client):
        """Window query param should be forwarded to get_cost_summary via days arg."""
        with _patch_summary() as mock_fn:
            client.get('/costs/partials/summary?window=24h')
        mock_fn.assert_called_once()
        _, kwargs = mock_fn.call_args
        assert kwargs.get('days') == 1

    def test_default_window_7d(self, client):
        """Without ?window=, days=7 should be used."""
        with _patch_summary() as mock_fn:
            client.get('/costs/partials/summary')
        _, kwargs = mock_fn.call_args
        assert kwargs.get('days') == 7


# ---------------------------------------------------------------------------
# Step-7: GET /costs/partials/by-project route tests
# ---------------------------------------------------------------------------

_MOCK_BY_PROJECT = {
    'dark_factory': [
        {'model': 'claude-sonnet', 'total': 8.50},
        {'model': 'claude-opus', 'total': 3.84},
    ],
    'other_project': [
        {'model': 'claude-haiku', 'total': 1.20},
    ],
}


def _patch_by_project(return_value=_MOCK_BY_PROJECT):
    return patch(
        'dashboard.app.get_cost_by_project',
        new_callable=AsyncMock,
        return_value=return_value,
    )


class TestCostsByProjectPartial:
    """Tests for GET /costs/partials/by-project."""

    def test_returns_200(self, client):
        with _patch_by_project():
            resp = client.get('/costs/partials/by-project')
        assert resp.status_code == 200

    def test_content_type_html(self, client):
        with _patch_by_project():
            resp = client.get('/costs/partials/by-project')
        assert 'text/html' in resp.headers['content-type']

    def test_renders_chart_canvas(self, client):
        """By-project partial must contain a Chart.js canvas element."""
        with _patch_by_project():
            html = client.get('/costs/partials/by-project').text
        assert 'canvas' in html.lower()

    def test_renders_chart_js_config(self, client):
        """Chart.js dataset config must appear in the rendered script."""
        with _patch_by_project():
            html = client.get('/costs/partials/by-project').text
        assert 'new Chart' in html

    def test_uses_color_palette(self, client):
        """Must use the standard 10-color palette."""
        with _patch_by_project():
            html = client.get('/costs/partials/by-project').text
        assert '#60a5fa' in html  # first color from palette

    def test_handles_empty_data(self, client):
        with _patch_by_project(return_value={}):
            resp = client.get('/costs/partials/by-project')
        assert resp.status_code == 200

    def test_respects_window_param(self, client):
        with _patch_by_project() as mock_fn:
            client.get('/costs/partials/by-project?window=30d')
        _, kwargs = mock_fn.call_args
        assert kwargs.get('days') == 30


# ---------------------------------------------------------------------------
# Step-11: GET /costs/partials/by-role route tests
# ---------------------------------------------------------------------------

_MOCK_BY_ROLE = {
    'dark_factory': {
        'implementer': {'claude-sonnet': 5.20, 'claude-opus': 1.50},
        'reviewer': {'claude-opus': 3.10},
    },
    'other_project': {
        'implementer': {'claude-haiku': 0.80},
    },
}


def _patch_by_role(return_value=_MOCK_BY_ROLE):
    return patch(
        'dashboard.app.get_cost_by_role',
        new_callable=AsyncMock,
        return_value=return_value,
    )


class TestCostsByRolePartial:
    """Tests for GET /costs/partials/by-role."""

    def test_returns_200(self, client):
        with _patch_by_role():
            resp = client.get('/costs/partials/by-role')
        assert resp.status_code == 200

    def test_content_type_html(self, client):
        with _patch_by_role():
            resp = client.get('/costs/partials/by-role')
        assert 'text/html' in resp.headers['content-type']

    def test_renders_chart_canvas(self, client):
        """By-role partial must contain a Chart.js canvas element."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        assert 'canvas' in html.lower()

    def test_renders_chart_js_config(self, client):
        """Chart.js dataset config must appear in the rendered script."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        assert 'new Chart' in html

    def test_uses_color_palette(self, client):
        """Must use the standard 10-color palette."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        assert '#60a5fa' in html  # first color from palette

    def test_role_names_appear_in_output(self, client):
        """Role names from the data should appear in the rendered HTML."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        assert 'implementer' in html

    def test_handles_empty_data(self, client):
        with _patch_by_role(return_value={}):
            resp = client.get('/costs/partials/by-role')
        assert resp.status_code == 200

    def test_respects_window_param(self, client):
        with _patch_by_role() as mock_fn:
            client.get('/costs/partials/by-role?window=30d')
        _, kwargs = mock_fn.call_args
        assert kwargs.get('days') == 30


# ---------------------------------------------------------------------------
# Step-13: GET /costs/partials/trend route tests
# ---------------------------------------------------------------------------

_MOCK_TREND = {
    'dark_factory': [
        {'day': '2026-03-25', 'total': 2.10},
        {'day': '2026-03-26', 'total': 3.40},
        {'day': '2026-03-27', 'total': 1.80},
    ],
    'other_project': [
        {'day': '2026-03-25', 'total': 0.50},
        {'day': '2026-03-26', 'total': 0.70},
        {'day': '2026-03-27', 'total': 0.30},
    ],
}


def _patch_trend(return_value=_MOCK_TREND):
    return patch(
        'dashboard.app.get_cost_trend',
        new_callable=AsyncMock,
        return_value=return_value,
    )


class TestCostsTrendPartial:
    """Tests for GET /costs/partials/trend."""

    def test_returns_200(self, client):
        with _patch_trend():
            resp = client.get('/costs/partials/trend')
        assert resp.status_code == 200

    def test_content_type_html(self, client):
        with _patch_trend():
            resp = client.get('/costs/partials/trend')
        assert 'text/html' in resp.headers['content-type']

    def test_renders_chart_canvas(self, client):
        """Trend partial must contain a Chart.js canvas element."""
        with _patch_trend():
            html = client.get('/costs/partials/trend').text
        assert 'canvas' in html.lower()

    def test_renders_line_chart(self, client):
        """Chart type must be 'line'."""
        with _patch_trend():
            html = client.get('/costs/partials/trend').text
        assert "'line'" in html or '"line"' in html

    def test_renders_chart_js_config(self, client):
        """Chart.js config with new Chart must appear."""
        with _patch_trend():
            html = client.get('/costs/partials/trend').text
        assert 'new Chart' in html

    def test_day_labels_present(self, client):
        """Day strings from the data should appear in the rendered output."""
        with _patch_trend():
            html = client.get('/costs/partials/trend').text
        assert '2026-03-25' in html

    def test_handles_empty_data(self, client):
        with _patch_trend(return_value={}):
            resp = client.get('/costs/partials/trend')
        assert resp.status_code == 200

    def test_respects_window_param(self, client):
        with _patch_trend() as mock_fn:
            client.get('/costs/partials/trend?window=24h')
        _, kwargs = mock_fn.call_args
        assert kwargs.get('days') == 1


# ---------------------------------------------------------------------------
# Step-15: GET /costs/partials/events route tests
# ---------------------------------------------------------------------------

_MOCK_EVENTS = [
    {
        'account_name': 'account-A',
        'event_type': 'cap_hit',
        'project_id': 'dark_factory',
        'run_id': 'run-123',
        'details': None,
        'created_at': '2026-03-30T10:00:00+00:00',
    },
    {
        'account_name': 'account-B',
        'event_type': 'resumed',
        'project_id': 'dark_factory',
        'run_id': 'run-456',
        'details': None,
        'created_at': '2026-03-30T11:30:00+00:00',
    },
]


def _patch_events(return_value=_MOCK_EVENTS):
    return patch(
        'dashboard.app.get_account_events',
        new_callable=AsyncMock,
        return_value=return_value,
    )


class TestCostsEventsPartial:
    """Tests for GET /costs/partials/events."""

    def test_returns_200(self, client):
        with _patch_events():
            resp = client.get('/costs/partials/events')
        assert resp.status_code == 200

    def test_content_type_html(self, client):
        with _patch_events():
            resp = client.get('/costs/partials/events')
        assert 'text/html' in resp.headers['content-type']

    def test_renders_account_names(self, client):
        """Account names from the events should appear in the output."""
        with _patch_events():
            html = client.get('/costs/partials/events').text
        assert 'account-A' in html
        assert 'account-B' in html

    def test_cap_hit_uses_red(self, client):
        """cap_hit events should have red styling."""
        with _patch_events():
            html = client.get('/costs/partials/events').text
        assert 'red' in html.lower()

    def test_resumed_uses_green(self, client):
        """resumed events should have green styling."""
        with _patch_events():
            html = client.get('/costs/partials/events').text
        assert 'green' in html.lower()

    def test_handles_empty_list(self, client):
        with _patch_events(return_value=[]):
            resp = client.get('/costs/partials/events')
        assert resp.status_code == 200

    def test_respects_window_param(self, client):
        with _patch_events() as mock_fn:
            client.get('/costs/partials/events?window=all')
        _, kwargs = mock_fn.call_args
        assert kwargs.get('days') == 3650


# ---------------------------------------------------------------------------
# Step-17: GET /costs/partials/runs route tests
# ---------------------------------------------------------------------------

_MOCK_RUNS = [
    {
        'run_id': 'run-abc-123',
        'project_id': 'dark_factory',
        'total_cost': 4.20,
        'tasks': [
            {
                'task_id': '42',
                'title': 'Implement feature X',
                'cost': 3.50,
                'invocations': [
                    {
                        'model': 'claude-sonnet',
                        'role': 'implementer',
                        'cost_usd': 2.00,
                        'account_name': 'account-A',
                        'duration_ms': 15000,
                        'capped': False,
                    },
                    {
                        'model': 'claude-opus',
                        'role': 'reviewer',
                        'cost_usd': 1.50,
                        'account_name': 'account-B',
                        'duration_ms': 8000,
                        'capped': True,
                    },
                ],
            },
            {
                'task_id': None,
                'title': None,
                'cost': 0.70,
                'invocations': [
                    {
                        'model': 'claude-haiku',
                        'role': 'orchestrator',
                        'cost_usd': 0.70,
                        'account_name': 'account-A',
                        'duration_ms': 3000,
                        'capped': False,
                    },
                ],
            },
        ],
    },
]


def _patch_runs(return_value=_MOCK_RUNS):
    return patch(
        'dashboard.app.get_run_cost_breakdown',
        new_callable=AsyncMock,
        return_value=return_value,
    )


class TestCostsRunsPartial:
    """Tests for GET /costs/partials/runs."""

    def test_returns_200(self, client):
        with _patch_runs():
            resp = client.get('/costs/partials/runs')
        assert resp.status_code == 200

    def test_content_type_html(self, client):
        with _patch_runs():
            resp = client.get('/costs/partials/runs')
        assert 'text/html' in resp.headers['content-type']

    def test_renders_run_id(self, client):
        """Run ID should appear in the rendered output."""
        with _patch_runs():
            html = client.get('/costs/partials/runs').text
        assert 'run-abc-123' in html

    def test_renders_total_cost(self, client):
        """Total cost for runs should appear."""
        with _patch_runs():
            html = client.get('/costs/partials/runs').text
        assert '$' in html

    def test_renders_task_title(self, client):
        """Task titles should appear in the expandable details."""
        with _patch_runs():
            html = client.get('/costs/partials/runs').text
        assert 'Implement feature X' in html

    def test_renders_capped_badge(self, client):
        """Capped invocations should show a 'capped' badge."""
        with _patch_runs():
            html = client.get('/costs/partials/runs').text
        assert 'capped' in html.lower()

    def test_handles_null_task_id(self, client):
        """Runs with null task_id should still render without errors."""
        with _patch_runs():
            resp = client.get('/costs/partials/runs')
        assert resp.status_code == 200

    def test_handles_empty_list(self, client):
        with _patch_runs(return_value=[]):
            resp = client.get('/costs/partials/runs')
        assert resp.status_code == 200

    def test_respects_window_param(self, client):
        with _patch_runs() as mock_fn:
            client.get('/costs/partials/runs?window=7d')
        _, kwargs = mock_fn.call_args
        assert kwargs.get('days') == 7


# ---------------------------------------------------------------------------
# Step-19: Window selector integration tests
# ---------------------------------------------------------------------------


class TestWindowSelectorIntegration:
    """Integration tests for window parameter propagation."""

    def test_costs_page_with_24h_window(self, client):
        """GET /costs?window=24h returns 200."""
        resp = client.get('/costs?window=24h')
        assert resp.status_code == 200

    def test_costs_page_with_all_window(self, client):
        """GET /costs?window=all returns 200."""
        resp = client.get('/costs?window=all')
        assert resp.status_code == 200

    def test_summary_partial_24h_calls_data_fn_with_days_1(self, client):
        """?window=24h → get_cost_summary called with days=1."""
        with _patch_summary() as mock_fn:
            client.get('/costs/partials/summary?window=24h')
        _, kwargs = mock_fn.call_args
        assert kwargs.get('days') == 1

    def test_by_project_partial_30d_calls_data_fn_with_days_30(self, client):
        """?window=30d → get_cost_by_project called with days=30."""
        with _patch_by_project() as mock_fn:
            client.get('/costs/partials/by-project?window=30d')
        _, kwargs = mock_fn.call_args
        assert kwargs.get('days') == 30

    def test_by_account_partial_all_calls_data_fn_with_days_3650(self, client):
        """?window=all → get_cost_by_account called with days=3650."""
        with _patch_by_account() as mock_fn:
            client.get('/costs/partials/by-account?window=all')
        _, kwargs = mock_fn.call_args
        assert kwargs.get('days') == 3650

    def test_by_role_partial_7d_calls_data_fn_with_days_7(self, client):
        """?window=7d → get_cost_by_role called with days=7."""
        with _patch_by_role() as mock_fn:
            client.get('/costs/partials/by-role?window=7d')
        _, kwargs = mock_fn.call_args
        assert kwargs.get('days') == 7

    def test_trend_partial_24h_calls_data_fn_with_days_1(self, client):
        """?window=24h → get_cost_trend called with days=1."""
        with _patch_trend() as mock_fn:
            client.get('/costs/partials/trend?window=24h')
        _, kwargs = mock_fn.call_args
        assert kwargs.get('days') == 1

    def test_events_partial_30d_calls_data_fn_with_days_30(self, client):
        """?window=30d → get_account_events called with days=30."""
        with _patch_events() as mock_fn:
            client.get('/costs/partials/events?window=30d')
        _, kwargs = mock_fn.call_args
        assert kwargs.get('days') == 30

    def test_runs_partial_all_calls_data_fn_with_days_3650(self, client):
        """?window=all → get_run_cost_breakdown called with days=3650."""
        with _patch_runs() as mock_fn:
            client.get('/costs/partials/runs?window=all')
        _, kwargs = mock_fn.call_args
        assert kwargs.get('days') == 3650

    def test_invalid_window_defaults_to_7(self, client):
        """Invalid ?window=bogus → days=7 (default) for any partial."""
        with _patch_summary() as mock_fn:
            client.get('/costs/partials/summary?window=bogus')
        _, kwargs = mock_fn.call_args
        assert kwargs.get('days') == 7


# ---------------------------------------------------------------------------
# Step-21: Partial failure resilience tests
# ---------------------------------------------------------------------------


def _patch_raises(target: str):
    """Patch a data function to raise RuntimeError."""
    return patch(target, new_callable=AsyncMock, side_effect=RuntimeError('db unavailable'))


class TestPartialFailureResilience:
    """Test that partial routes return 200 even when data functions raise."""

    def test_summary_returns_200_on_data_error(self, client):
        with _patch_raises('dashboard.app.get_cost_summary'):
            resp = client.get('/costs/partials/summary')
        assert resp.status_code == 200

    def test_summary_shows_fallback_content_on_error(self, client):
        """On error, summary should render the empty-data fallback message."""
        with _patch_raises('dashboard.app.get_cost_summary'):
            html = client.get('/costs/partials/summary').text
        assert 'No cost data available for this window.' in html

    def test_by_project_returns_200_on_data_error(self, client):
        with _patch_raises('dashboard.app.get_cost_by_project'):
            resp = client.get('/costs/partials/by-project')
        assert resp.status_code == 200

    def test_by_account_returns_200_on_data_error(self, client):
        with _patch_raises('dashboard.app.get_cost_by_account'):
            resp = client.get('/costs/partials/by-account')
        assert resp.status_code == 200

    def test_by_role_returns_200_on_data_error(self, client):
        with _patch_raises('dashboard.app.get_cost_by_role'):
            resp = client.get('/costs/partials/by-role')
        assert resp.status_code == 200

    def test_trend_returns_200_on_data_error(self, client):
        with _patch_raises('dashboard.app.get_cost_trend'):
            resp = client.get('/costs/partials/trend')
        assert resp.status_code == 200

    def test_events_returns_200_on_data_error(self, client):
        with _patch_raises('dashboard.app.get_account_events'):
            resp = client.get('/costs/partials/events')
        assert resp.status_code == 200

    def test_runs_returns_200_on_data_error(self, client):
        with _patch_raises('dashboard.app.get_run_cost_breakdown'):
            resp = client.get('/costs/partials/runs')
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Step-9: GET /costs/partials/by-account route tests
# ---------------------------------------------------------------------------

_MOCK_BY_ACCOUNT = {
    'account-A': {
        'spend': 9.50,
        'invocations': 50,
        'cap_events': 1,
        'last_cap': '2026-03-30T10:00:00+00:00',
        'status': 'capped',
    },
    'account-B': {
        'spend': 3.20,
        'invocations': 20,
        'cap_events': 0,
        'last_cap': None,
        'status': 'active',
    },
}


def _patch_by_account(return_value=_MOCK_BY_ACCOUNT):
    return patch(
        'dashboard.app.get_cost_by_account',
        new_callable=AsyncMock,
        return_value=return_value,
    )


class TestCostsByAccountPartial:
    """Tests for GET /costs/partials/by-account."""

    def test_returns_200(self, client):
        with _patch_by_account():
            resp = client.get('/costs/partials/by-account')
        assert resp.status_code == 200

    def test_renders_doughnut_chart(self, client):
        """Must include a canvas for the doughnut chart."""
        with _patch_by_account():
            html = client.get('/costs/partials/by-account').text
        assert 'canvas' in html.lower()
        assert 'doughnut' in html.lower()

    def test_renders_account_table(self, client):
        """Must include a table with account data."""
        with _patch_by_account():
            html = client.get('/costs/partials/by-account').text
        assert 'account-A' in html
        assert 'account-B' in html

    def test_capped_status_badge(self, client):
        """Capped accounts should show red status badge."""
        with _patch_by_account():
            html = client.get('/costs/partials/by-account').text
        # Red color class for capped accounts
        assert 'red' in html.lower()

    def test_active_status_badge(self, client):
        """Active accounts should show green status badge."""
        with _patch_by_account():
            html = client.get('/costs/partials/by-account').text
        assert 'green' in html.lower()

    def test_handles_empty_data(self, client):
        with _patch_by_account(return_value={}):
            resp = client.get('/costs/partials/by-account')
        assert resp.status_code == 200

    def test_respects_window_param(self, client):
        with _patch_by_account() as mock_fn:
            client.get('/costs/partials/by-account?window=7d')
        _, kwargs = mock_fn.call_args
        assert kwargs.get('days') == 7


# ---------------------------------------------------------------------------
# Step-23: Alpine v3 store pattern tests
# ---------------------------------------------------------------------------


class TestAlpineV3StorePattern:
    """Verify costs.html uses Alpine v3 store API, not the deprecated Alpine v2 __x API."""

    def test_alpine_store_costs_registration_present(self, client):
        """costs.html must register an Alpine.store('costs', ...) store."""
        html = client.get('/costs').text
        assert "Alpine.store('costs'" in html or 'Alpine.store("costs"' in html

    def test_no_alpine_v2_private_api(self, client):
        """The __x private Alpine v2 API must not appear anywhere in the page."""
        html = client.get('/costs').text
        assert '__x.$data' not in html

    def test_no_alpine_v2_null_guard(self, client):
        """The '__x &&' null-guard pattern from Alpine v2 must not appear."""
        html = client.get('/costs').text
        assert '__x &&' not in html

    def test_hx_vals_use_alpine_store(self, client):
        """All hx-vals expressions must read from Alpine.store(\"costs\").window."""
        html = client.get('/costs').text
        # Count occurrences — there are 7 sections, each with one hx-vals
        assert 'Alpine.store("costs").window' in html or "Alpine.store('costs').window" in html

    def test_hx_vals_count(self, client):
        """All 7 section hx-vals must use the store (not the old __x expression)."""
        html = client.get('/costs').text
        # Old pattern used getElementById + __x; new pattern uses Alpine.store
        old_pattern_count = html.count('__x.$data.currentWindow')
        assert old_pattern_count == 0
        # All 7 sections should have Alpine.store("costs").window in hx-vals
        new_pattern_count = html.count('Alpine.store("costs").window')
        assert new_pattern_count == 7

    def test_button_click_writes_to_store(self, client):
        """@click handlers on window selector buttons must write to Alpine.store('costs').window."""
        html = client.get('/costs').text
        # Must contain store write; old pattern was `currentWindow = '...'`
        assert "Alpine.store('costs').window" in html or 'Alpine.store("costs").window' in html

    def test_button_class_binding_reads_from_store(self, client):
        """Button :class bindings must read from $store.costs.window (not currentWindow)."""
        html = client.get('/costs').text
        assert '$store.costs.window' in html

    def test_no_x_data_currentwindow(self, client):
        """The x-data component with currentWindow must not be present (state is in global store)."""
        html = client.get('/costs').text
        assert 'currentWindow' not in html

    def test_alpine_init_listener_present(self, client):
        """The store must be initialized via an alpine:init event listener."""
        html = client.get('/costs').text
        assert "alpine:init" in html

    def test_store_default_window_7d(self, client):
        """The store initialization must default to '7d'."""
        html = client.get('/costs').text
        # The store init should embed the window value; default is 7d
        assert "'7d'" in html or '"7d"' in html
