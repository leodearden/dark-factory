"""Tests for the reconciliation panel route and supporting utilities."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch


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


class TestReconRoute:
    """Tests for GET /partials/recon with populated data."""

    @patch('dashboard.app.get_recent_runs', new_callable=AsyncMock, return_value=MOCK_RUNS)
    @patch('dashboard.app.get_latest_verdict', new_callable=AsyncMock, return_value=MOCK_VERDICT)
    @patch('dashboard.app.get_watermarks', new_callable=AsyncMock, return_value=MOCK_WATERMARKS)
    @patch('dashboard.app.get_burst_state', new_callable=AsyncMock, return_value=MOCK_BURST_STATE)
    @patch('dashboard.app.get_buffer_stats', new_callable=AsyncMock, return_value=MOCK_BUFFER_STATS)
    def test_returns_200(self, *_mocks, client):
        resp = client.get('/partials/recon')
        assert resp.status_code == 200

    @patch('dashboard.app.get_recent_runs', new_callable=AsyncMock, return_value=MOCK_RUNS)
    @patch('dashboard.app.get_latest_verdict', new_callable=AsyncMock, return_value=MOCK_VERDICT)
    @patch('dashboard.app.get_watermarks', new_callable=AsyncMock, return_value=MOCK_WATERMARKS)
    @patch('dashboard.app.get_burst_state', new_callable=AsyncMock, return_value=MOCK_BURST_STATE)
    @patch('dashboard.app.get_buffer_stats', new_callable=AsyncMock, return_value=MOCK_BUFFER_STATS)
    def test_content_type_html(self, *_mocks, client):
        resp = client.get('/partials/recon')
        assert 'text/html' in resp.headers['content-type']

    @patch('dashboard.app.get_recent_runs', new_callable=AsyncMock, return_value=MOCK_RUNS)
    @patch('dashboard.app.get_latest_verdict', new_callable=AsyncMock, return_value=MOCK_VERDICT)
    @patch('dashboard.app.get_watermarks', new_callable=AsyncMock, return_value=MOCK_WATERMARKS)
    @patch('dashboard.app.get_burst_state', new_callable=AsyncMock, return_value=MOCK_BURST_STATE)
    @patch('dashboard.app.get_buffer_stats', new_callable=AsyncMock, return_value=MOCK_BUFFER_STATS)
    def test_buffer_stats_displayed(self, *_mocks, client):
        resp = client.get('/partials/recon')
        html = resp.text
        assert '3 events buffered' in html

    @patch('dashboard.app.get_recent_runs', new_callable=AsyncMock, return_value=MOCK_RUNS)
    @patch('dashboard.app.get_latest_verdict', new_callable=AsyncMock, return_value=MOCK_VERDICT)
    @patch('dashboard.app.get_watermarks', new_callable=AsyncMock, return_value=MOCK_WATERMARKS)
    @patch('dashboard.app.get_burst_state', new_callable=AsyncMock, return_value=MOCK_BURST_STATE)
    @patch('dashboard.app.get_buffer_stats', new_callable=AsyncMock, return_value=MOCK_BUFFER_STATS)
    def test_burst_agents_displayed(self, *_mocks, client):
        resp = client.get('/partials/recon')
        html = resp.text
        assert 'agent-1' in html
        assert 'agent-2' in html

    @patch('dashboard.app.get_recent_runs', new_callable=AsyncMock, return_value=MOCK_RUNS)
    @patch('dashboard.app.get_latest_verdict', new_callable=AsyncMock, return_value=MOCK_VERDICT)
    @patch('dashboard.app.get_watermarks', new_callable=AsyncMock, return_value=MOCK_WATERMARKS)
    @patch('dashboard.app.get_burst_state', new_callable=AsyncMock, return_value=MOCK_BURST_STATE)
    @patch('dashboard.app.get_buffer_stats', new_callable=AsyncMock, return_value=MOCK_BUFFER_STATS)
    def test_burst_state_badges(self, *_mocks, client):
        resp = client.get('/partials/recon')
        html = resp.text
        assert 'bursting' in html
        assert 'idle' in html

    @patch('dashboard.app.get_recent_runs', new_callable=AsyncMock, return_value=MOCK_RUNS)
    @patch('dashboard.app.get_latest_verdict', new_callable=AsyncMock, return_value=MOCK_VERDICT)
    @patch('dashboard.app.get_watermarks', new_callable=AsyncMock, return_value=MOCK_WATERMARKS)
    @patch('dashboard.app.get_burst_state', new_callable=AsyncMock, return_value=MOCK_BURST_STATE)
    @patch('dashboard.app.get_buffer_stats', new_callable=AsyncMock, return_value=MOCK_BUFFER_STATS)
    def test_watermark_labels(self, *_mocks, client):
        resp = client.get('/partials/recon')
        html = resp.text
        assert 'Last full run' in html
        assert 'Last episode' in html
        assert 'Last memory' in html
        assert 'Last task change' in html

    @patch('dashboard.app.get_recent_runs', new_callable=AsyncMock, return_value=MOCK_RUNS)
    @patch('dashboard.app.get_latest_verdict', new_callable=AsyncMock, return_value=MOCK_VERDICT)
    @patch('dashboard.app.get_watermarks', new_callable=AsyncMock, return_value=MOCK_WATERMARKS)
    @patch('dashboard.app.get_burst_state', new_callable=AsyncMock, return_value=MOCK_BURST_STATE)
    @patch('dashboard.app.get_buffer_stats', new_callable=AsyncMock, return_value=MOCK_BUFFER_STATS)
    def test_verdict_severity_badge(self, *_mocks, client):
        resp = client.get('/partials/recon')
        html = resp.text
        assert 'bg-green-600' in html  # ok severity → green badge

    @patch('dashboard.app.get_recent_runs', new_callable=AsyncMock, return_value=MOCK_RUNS)
    @patch('dashboard.app.get_latest_verdict', new_callable=AsyncMock, return_value=MOCK_VERDICT)
    @patch('dashboard.app.get_watermarks', new_callable=AsyncMock, return_value=MOCK_WATERMARKS)
    @patch('dashboard.app.get_burst_state', new_callable=AsyncMock, return_value=MOCK_BURST_STATE)
    @patch('dashboard.app.get_buffer_stats', new_callable=AsyncMock, return_value=MOCK_BUFFER_STATS)
    def test_verdict_action(self, *_mocks, client):
        resp = client.get('/partials/recon')
        html = resp.text
        assert 'none' in html  # action_taken

    @patch('dashboard.app.get_recent_runs', new_callable=AsyncMock, return_value=MOCK_RUNS)
    @patch('dashboard.app.get_latest_verdict', new_callable=AsyncMock, return_value=MOCK_VERDICT)
    @patch('dashboard.app.get_watermarks', new_callable=AsyncMock, return_value=MOCK_WATERMARKS)
    @patch('dashboard.app.get_burst_state', new_callable=AsyncMock, return_value=MOCK_BURST_STATE)
    @patch('dashboard.app.get_buffer_stats', new_callable=AsyncMock, return_value=MOCK_BUFFER_STATS)
    def test_runs_table_content(self, *_mocks, client):
        resp = client.get('/partials/recon')
        html = resp.text
        assert 'staleness_timer' in html
        assert 'completed' in html

    @patch('dashboard.app.get_recent_runs', new_callable=AsyncMock, return_value=MOCK_RUNS)
    @patch('dashboard.app.get_latest_verdict', new_callable=AsyncMock, return_value=MOCK_VERDICT)
    @patch('dashboard.app.get_watermarks', new_callable=AsyncMock, return_value=MOCK_WATERMARKS)
    @patch('dashboard.app.get_burst_state', new_callable=AsyncMock, return_value=MOCK_BURST_STATE)
    @patch('dashboard.app.get_buffer_stats', new_callable=AsyncMock, return_value=MOCK_BUFFER_STATS)
    def test_grid_layout(self, *_mocks, client):
        resp = client.get('/partials/recon')
        html = resp.text
        assert 'grid grid-cols-2' in html
