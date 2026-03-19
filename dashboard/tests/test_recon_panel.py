"""Tests for the reconciliation panel route and supporting utilities."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch


class TestFormatTriggerFilter:
    """Tests for the format_trigger Jinja2 filter function."""

    def test_none_returns_empty_string(self):
        from dashboard.app import format_trigger

        assert format_trigger(None) == ''

    def test_no_colon_returns_value_unchanged(self):
        from dashboard.app import format_trigger

        assert format_trigger('manual') == 'manual'

    def test_quiescent_with_value(self):
        from dashboard.app import format_trigger

        assert format_trigger('quiescent:6') == 'quiescent (6)'

    def test_buffer_size_maps_to_buffer(self):
        from dashboard.app import format_trigger

        assert format_trigger('buffer_size:10') == 'buffer (10)'

    def test_max_staleness_maps_to_staleness_with_relative_time(self):
        from dashboard.app import format_trigger

        ts = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        result = format_trigger(f'max_staleness:{ts}')
        assert result == 'staleness (1h ago)'

    def test_unknown_type_uses_generic_format(self):
        from dashboard.app import format_trigger

        assert format_trigger('foo:bar') == 'foo (bar)'

    def test_filter_registered_on_jinja2_env(self):
        from dashboard.app import format_trigger, templates

        assert 'format_trigger' in templates.env.filters
        assert templates.env.filters['format_trigger'] is format_trigger


class TestTimeagoFilter:
    """Tests for the timeago Jinja2 filter function."""

    def test_none_returns_never(self):
        from dashboard.app import timeago

        assert timeago(None) == 'never'

    def test_minutes_ago(self):
        from dashboard.app import timeago

        ts = (datetime.now(UTC) - timedelta(minutes=3)).isoformat()
        assert timeago(ts) == '3m ago'

    def test_hours_ago(self):
        from dashboard.app import timeago

        ts = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        assert timeago(ts) == '2h ago'

    def test_days_ago(self):
        from dashboard.app import timeago

        ts = (datetime.now(UTC) - timedelta(days=2)).isoformat()
        assert timeago(ts) == '2d ago'

    def test_invalid_string_returns_never(self):
        from dashboard.app import timeago

        assert timeago('not-a-date') == 'never'

    def test_empty_string_returns_never(self):
        from dashboard.app import timeago

        assert timeago('') == 'never'

    def test_filter_registered_on_jinja2_env(self):
        from dashboard.app import templates, timeago

        assert 'timeago' in templates.env.filters
        assert templates.env.filters['timeago'] is timeago


# --- Mock data for route tests ---

MOCK_BUFFER_STATS = {'buffered_count': 3, 'oldest_event_age_seconds': 600.0}

MOCK_BURST_STATE = [
    {
        'agent_id': 'agent-1',
        'state': 'bursting',
        'last_write_at': '2026-03-19T00:00:00+00:00',
        'burst_started_at': '2026-03-19T00:00:00+00:00',
    },
    {
        'agent_id': 'agent-2',
        'state': 'idle',
        'last_write_at': '2026-03-19T00:00:00+00:00',
        'burst_started_at': None,
    },
]

MOCK_WATERMARKS = {
    'last_full_run_completed': '2026-03-19T10:00:00+00:00',
    'last_episode_timestamp': '2026-03-19T10:30:00+00:00',
    'last_memory_timestamp': '2026-03-19T10:40:00+00:00',
    'last_task_change_timestamp': '2026-03-19T10:50:00+00:00',
}

MOCK_VERDICT = {
    'run_id': 'run-001',
    'severity': 'ok',
    'action_taken': 'none',
    'reviewed_at': '2026-03-19T10:00:00+00:00',
}

MOCK_RUNS = [
    {
        'id': 'run-001',
        'run_type': 'full',
        'trigger_reason': 'staleness_timer',
        'started_at': '2026-03-19T08:00:00+00:00',
        'completed_at': '2026-03-19T08:05:00+00:00',
        'events_processed': 7,
        'status': 'completed',
        'duration_seconds': 300.0,
    }
]


_UNSET = object()


def _patch_recon_data(buffer_stats=_UNSET, burst_state=_UNSET, watermarks=_UNSET,
                      verdict=_UNSET, runs=_UNSET):
    """Return an ExitStack that patches all 5 recon data functions."""
    stack = ExitStack()
    stack.enter_context(patch(
        'dashboard.app.get_buffer_stats',
        new_callable=AsyncMock,
        return_value=buffer_stats if buffer_stats is not _UNSET else MOCK_BUFFER_STATS,
    ))
    stack.enter_context(patch(
        'dashboard.app.get_burst_state',
        new_callable=AsyncMock,
        return_value=burst_state if burst_state is not _UNSET else MOCK_BURST_STATE,
    ))
    stack.enter_context(patch(
        'dashboard.app.get_watermarks',
        new_callable=AsyncMock,
        return_value=watermarks if watermarks is not _UNSET else MOCK_WATERMARKS,
    ))
    stack.enter_context(patch(
        'dashboard.app.get_latest_verdict',
        new_callable=AsyncMock,
        return_value=verdict if verdict is not _UNSET else MOCK_VERDICT,
    ))
    stack.enter_context(patch(
        'dashboard.app.get_recent_runs',
        new_callable=AsyncMock,
        return_value=runs if runs is not _UNSET else MOCK_RUNS,
    ))
    return stack


class TestReconRoute:
    """Tests for GET /partials/recon with populated data."""

    def test_returns_200(self, client):
        with _patch_recon_data():
            resp = client.get('/partials/recon')
        assert resp.status_code == 200

    def test_content_type_html(self, client):
        with _patch_recon_data():
            resp = client.get('/partials/recon')
        assert 'text/html' in resp.headers['content-type']

    def test_buffer_stats_displayed(self, client):
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert '3 events buffered' in html

    def test_burst_agents_displayed(self, client):
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'agent-1' in html
        assert 'agent-2' in html

    def test_burst_state_badges(self, client):
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'bursting' in html
        assert 'idle' in html

    def test_watermark_labels(self, client):
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'Last full run' in html
        assert 'Last episode' in html
        assert 'Last memory' in html
        assert 'Last task change' in html

    def test_verdict_severity_badge(self, client):
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'bg-green-600' in html  # ok severity → green badge

    def test_verdict_action(self, client):
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'none' in html  # action_taken

    def test_runs_table_content(self, client):
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'staleness_timer' in html
        assert 'completed' in html

    def test_grid_layout(self, client):
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'grid-cols-1' in html
        assert 'lg:grid-cols-2' in html

    def test_responsive_grid_stacks(self, client):
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'grid grid-cols-1 lg:grid-cols-2' in html

    def test_runs_table_overflow_container(self, client):
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'overflow-x-auto' in html

    def test_trigger_formatted_in_html(self, client):
        runs_with_colon = [
            {
                'id': 'run-002',
                'run_type': 'full',
                'trigger_reason': 'max_staleness:2026-03-19T08:00:00+00:00',
                'started_at': '2026-03-19T08:00:00+00:00',
                'completed_at': '2026-03-19T08:05:00+00:00',
                'events_processed': 3,
                'status': 'completed',
                'duration_seconds': 300.0,
            }
        ]
        with _patch_recon_data(runs=runs_with_colon):
            html = client.get('/partials/recon').text
        assert 'staleness' in html
        assert 'title="max_staleness:2026-03-19T08:00:00+00:00"' in html


# --- Empty data constants ---

EMPTY_BUFFER_STATS = {'buffered_count': 0, 'oldest_event_age_seconds': None}


class TestReconRouteEmpty:
    """Tests for GET /partials/recon with empty/default data."""

    def test_returns_200(self, client):
        with _patch_recon_data(
            buffer_stats=EMPTY_BUFFER_STATS,
            burst_state=[],
            watermarks={},
            verdict=None,
            runs=[],
        ):
            resp = client.get('/partials/recon')
        assert resp.status_code == 200

    def test_zero_events_buffered(self, client):
        with _patch_recon_data(
            buffer_stats=EMPTY_BUFFER_STATS,
            burst_state=[],
            watermarks={},
            verdict=None,
            runs=[],
        ):
            html = client.get('/partials/recon').text
        assert '0 events buffered' in html

    def test_no_agents_fallback(self, client):
        with _patch_recon_data(
            buffer_stats=EMPTY_BUFFER_STATS,
            burst_state=[],
            watermarks={},
            verdict=None,
            runs=[],
        ):
            html = client.get('/partials/recon').text
        assert 'No agents' in html

    def test_no_verdicts_fallback(self, client):
        with _patch_recon_data(
            buffer_stats=EMPTY_BUFFER_STATS,
            burst_state=[],
            watermarks={},
            verdict=None,
            runs=[],
        ):
            html = client.get('/partials/recon').text
        assert 'No verdicts yet' in html

    def test_no_runs_fallback(self, client):
        with _patch_recon_data(
            buffer_stats=EMPTY_BUFFER_STATS,
            burst_state=[],
            watermarks={},
            verdict=None,
            runs=[],
        ):
            html = client.get('/partials/recon').text
        assert 'No reconciliation runs yet' in html

    def test_no_badges_when_empty(self, client):
        with _patch_recon_data(
            buffer_stats=EMPTY_BUFFER_STATS,
            burst_state=[],
            watermarks={},
            verdict=None,
            runs=[],
        ):
            html = client.get('/partials/recon').text
        assert 'bg-green-600' not in html
        assert 'bg-red-600' not in html
