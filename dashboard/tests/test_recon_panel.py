"""Tests for the reconciliation panel route and supporting utilities."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest


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


def _patch_recon_data(buffer_stats=None, burst_state=None, watermarks=None,
                      verdict=None, runs=None):
    """Return an ExitStack that patches all 5 recon data functions."""
    stack = ExitStack()
    stack.enter_context(patch(
        'dashboard.app.get_buffer_stats',
        new_callable=AsyncMock,
        return_value=buffer_stats if buffer_stats is not None else MOCK_BUFFER_STATS,
    ))
    stack.enter_context(patch(
        'dashboard.app.get_burst_state',
        new_callable=AsyncMock,
        return_value=burst_state if burst_state is not None else MOCK_BURST_STATE,
    ))
    stack.enter_context(patch(
        'dashboard.app.get_watermarks',
        new_callable=AsyncMock,
        return_value=watermarks if watermarks is not None else MOCK_WATERMARKS,
    ))
    stack.enter_context(patch(
        'dashboard.app.get_latest_verdict',
        new_callable=AsyncMock,
        return_value=verdict if verdict is not None else MOCK_VERDICT,
    ))
    stack.enter_context(patch(
        'dashboard.app.get_recent_runs',
        new_callable=AsyncMock,
        return_value=runs if runs is not None else MOCK_RUNS,
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
        assert 'grid grid-cols-2' in html
