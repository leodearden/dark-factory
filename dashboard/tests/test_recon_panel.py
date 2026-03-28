"""Tests for the reconciliation panel route and supporting utilities."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch


class TestProjectNameFilter:
    """Tests for the project_name Jinja2 filter function."""

    def test_path_returns_basename(self):
        from dashboard.app import project_name

        assert project_name('/home/leo/src/dark-factory') == 'dark-factory'

    def test_non_path_returned_unchanged(self):
        from dashboard.app import project_name

        assert project_name('dark_factory') == 'dark_factory'

    def test_none_returns_empty_string(self):
        from dashboard.app import project_name

        assert project_name(None) == ''

    def test_empty_string_returns_empty_string(self):
        from dashboard.app import project_name

        assert project_name('') == ''

    def test_trailing_slash_returns_basename(self):
        from dashboard.app import project_name

        assert project_name('/home/leo/src/dark-factory/') == 'dark-factory'

    def test_filter_registered_on_jinja2_env(self):
        from dashboard.app import project_name, templates

        assert 'project_name' in templates.env.filters
        assert templates.env.filters['project_name'] is project_name


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

    def test_just_now_for_zero_minutes(self):
        from dashboard.app import timeago

        ts = (datetime.now(UTC) - timedelta(seconds=5)).isoformat()
        assert timeago(ts) == 'just now'

    def test_filter_registered_on_jinja2_env(self):
        from dashboard.app import templates, timeago

        assert 'timeago' in templates.env.filters
        assert templates.env.filters['timeago'] is timeago


class TestPartitionBurstState:
    """Tests for the partition_burst_state helper."""

    def test_bursting_agent_is_active(self):
        from dashboard.data.reconciliation import partition_burst_state

        agents = [{'agent_id': 'a1', 'state': 'bursting', 'last_write_at': '2020-01-01T00:00:00+00:00'}]
        active, idle = partition_burst_state(agents)
        assert len(active) == 1
        assert len(idle) == 0

    def test_idle_old_agent_is_idle(self):
        from dashboard.data.reconciliation import partition_burst_state

        agents = [{'agent_id': 'a1', 'state': 'idle', 'last_write_at': '2020-01-01T00:00:00+00:00'}]
        active, idle = partition_burst_state(agents)
        assert len(active) == 0
        assert len(idle) == 1

    def test_idle_recent_agent_is_active(self):
        from dashboard.data.reconciliation import partition_burst_state

        recent_ts = (datetime.now(UTC) - timedelta(minutes=30)).isoformat()
        agents = [{'agent_id': 'a1', 'state': 'idle', 'last_write_at': recent_ts}]
        active, idle = partition_burst_state(agents)
        assert len(active) == 1
        assert len(idle) == 0

    def test_mixed_partition(self):
        from dashboard.data.reconciliation import partition_burst_state

        old_ts = '2020-01-01T00:00:00+00:00'
        recent_ts = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
        agents = [
            {'agent_id': 'bursting', 'state': 'bursting', 'last_write_at': old_ts},
            {'agent_id': 'idle-old', 'state': 'idle', 'last_write_at': old_ts},
            {'agent_id': 'idle-recent', 'state': 'idle', 'last_write_at': recent_ts},
        ]
        active, idle = partition_burst_state(agents)
        assert [a['agent_id'] for a in active] == ['bursting', 'idle-recent']
        assert [a['agent_id'] for a in idle] == ['idle-old']

    def test_empty_list(self):
        from dashboard.data.reconciliation import partition_burst_state

        active, idle = partition_burst_state([])
        assert active == []
        assert idle == []

    def test_invalid_timestamp_treated_as_idle(self):
        from dashboard.data.reconciliation import partition_burst_state

        agents = [{'agent_id': 'a1', 'state': 'idle', 'last_write_at': 'not-a-date'}]
        active, idle = partition_burst_state(agents)
        assert len(active) == 0
        assert len(idle) == 1

    def test_custom_threshold(self):
        from dashboard.data.reconciliation import partition_burst_state

        ts = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
        agents = [{'agent_id': 'a1', 'state': 'idle', 'last_write_at': ts}]
        # 5 min old, threshold 3 min → idle
        active, idle = partition_burst_state(agents, active_threshold_seconds=180)
        assert len(active) == 0
        assert len(idle) == 1
        # 5 min old, threshold 10 min → active
        active, idle = partition_burst_state(agents, active_threshold_seconds=600)
        assert len(active) == 1
        assert len(idle) == 0


class TestFormatDuration:
    """Tests for the format_duration Jinja2 filter (accepts seconds)."""

    def test_under_60s_returns_seconds_only(self):
        from dashboard.app import format_duration

        assert format_duration(45) == '45s'

    def test_zero_seconds_returns_dash(self):
        from dashboard.app import format_duration

        assert format_duration(0) == '-'

    def test_exactly_59s_returns_seconds(self):
        from dashboard.app import format_duration

        assert format_duration(59) == '59s'

    def test_exactly_60s_returns_minutes_seconds(self):
        from dashboard.app import format_duration

        assert format_duration(60) == '1m 0s'

    def test_600s_returns_10m_0s(self):
        from dashboard.app import format_duration

        assert format_duration(600) == '10m 0s'

    def test_90s_returns_1m_30s(self):
        from dashboard.app import format_duration

        assert format_duration(90) == '1m 30s'

    def test_3599s_returns_minutes_seconds(self):
        from dashboard.app import format_duration

        assert format_duration(3599) == '59m 59s'

    def test_exactly_3600s_returns_hours_minutes(self):
        from dashboard.app import format_duration

        assert format_duration(3600) == '1h 0m'

    def test_large_value_returns_hours_minutes(self):
        from dashboard.app import format_duration

        # 62736 seconds = 17h 25m 36s → '17h 25m'
        assert format_duration(62736) == '17h 25m'

    def test_float_input_is_handled(self):
        from dashboard.app import format_duration

        assert format_duration(600.0) == '10m 0s'

    def test_filter_registered_on_jinja2_env(self):
        from dashboard.app import format_duration, templates

        assert 'format_duration' in templates.env.filters
        assert templates.env.filters['format_duration'] is format_duration

    def test_negative_value_returns_dash(self):
        from dashboard.app import format_duration

        assert format_duration(-30) == '-'

    def test_non_numeric_returns_dash(self):
        from dashboard.app import format_duration

        assert format_duration('not_a_number') == '-'


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

MOCK_WATERMARKS = [
    {
        'project_id': 'dark_factory',
        'last_full_run_completed': '2026-03-19T10:00:00+00:00',
        'last_episode_timestamp': '2026-03-19T10:30:00+00:00',
        'last_memory_timestamp': '2026-03-19T10:40:00+00:00',
        'last_task_change_timestamp': '2026-03-19T10:50:00+00:00',
    },
]

MOCK_VERDICT = {
    'run_id': 'run-001',
    'severity': 'ok',
    'action_taken': 'none',
    'reviewed_at': '2026-03-19T10:00:00+00:00',
}

MOCK_RUNS = [
    {
        'id': 'run-001',
        'project_id': 'dark_factory',
        'run_type': 'full',
        'trigger_reason': 'staleness_timer',
        'started_at': '2026-03-19T08:00:00+00:00',
        'completed_at': '2026-03-19T08:05:00+00:00',
        'events_processed': 7,
        'status': 'completed',
        'duration_seconds': 300.0,
        'journal_entry_count': 3,
    }
]

MOCK_LAST_ATTEMPTED = {
    'dark_factory': {
        'id': 'run-002',
        'status': 'failed',
        'started_at': '2026-03-19T09:00:00+00:00',
        'completed_at': '2026-03-19T09:01:00+00:00',
    },
}


_UNSET = object()


def _patch_recon_data(buffer_stats=_UNSET, burst_state=_UNSET, watermarks=_UNSET,
                      verdict=_UNSET, runs=_UNSET, last_attempted=_UNSET):
    """Return an ExitStack that patches all 6 recon data functions."""
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
    stack.enter_context(patch(
        'dashboard.app.get_last_attempted_run',
        new_callable=AsyncMock,
        return_value=last_attempted if last_attempted is not _UNSET else MOCK_LAST_ATTEMPTED,
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
        assert 'Last successful run' in html
        assert 'Last episode' in html
        assert 'Last memory' in html
        assert 'Last task change' in html

    def test_last_attempted_run_displayed(self, client):
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'Last attempted run' in html
        assert 'aria-label="Last attempted run status: failed"' in html

    def test_last_successful_run_label(self, client):
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'Last successful run' in html

    def test_last_attempted_empty_shows_never(self, client):
        with _patch_recon_data(last_attempted={}):
            html = client.get('/partials/recon').text
        assert 'Last attempted run' in html
        # The "never" text should appear in the last attempted run row
        assert 'never' in html

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
                'project_id': 'dark_factory',
                'run_type': 'full',
                'trigger_reason': 'max_staleness:2026-03-19T08:00:00+00:00',
                'started_at': '2026-03-19T08:00:00+00:00',
                'completed_at': '2026-03-19T08:05:00+00:00',
                'events_processed': 3,
                'status': 'completed',
                'duration_seconds': 300.0,
                'journal_entry_count': 0,
            }
        ]
        with _patch_recon_data(runs=runs_with_colon):
            html = client.get('/partials/recon').text
        assert 'staleness' in html
        assert 'title="max_staleness:2026-03-19T08:00:00+00:00"' in html

    def test_recon_heading_present(self, client):
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert '<h2' in html
        assert 'Reconciliation' in html


# --- Empty data constants ---

EMPTY_BUFFER_STATS = {'buffered_count': 0, 'oldest_event_age_seconds': None}


class TestReconRouteEmpty:
    """Tests for GET /partials/recon with empty/default data."""

    def test_returns_200(self, client):
        with _patch_recon_data(
            buffer_stats=EMPTY_BUFFER_STATS,
            burst_state=[],
            watermarks=[],
            verdict=None,
            runs=[],
            last_attempted={},
        ):
            resp = client.get('/partials/recon')
        assert resp.status_code == 200

    def test_zero_events_buffered(self, client):
        with _patch_recon_data(
            buffer_stats=EMPTY_BUFFER_STATS,
            burst_state=[],
            watermarks=[],
            verdict=None,
            runs=[],
            last_attempted={},
        ):
            html = client.get('/partials/recon').text
        assert '0 events buffered' in html

    def test_no_agents_fallback(self, client):
        with _patch_recon_data(
            buffer_stats=EMPTY_BUFFER_STATS,
            burst_state=[],
            watermarks=[],
            verdict=None,
            runs=[],
            last_attempted={},
        ):
            html = client.get('/partials/recon').text
        assert 'No agents' in html

    def test_no_verdicts_fallback(self, client):
        with _patch_recon_data(
            buffer_stats=EMPTY_BUFFER_STATS,
            burst_state=[],
            watermarks=[],
            verdict=None,
            runs=[],
            last_attempted={},
        ):
            html = client.get('/partials/recon').text
        assert 'No verdicts yet' in html

    def test_no_runs_fallback(self, client):
        with _patch_recon_data(
            buffer_stats=EMPTY_BUFFER_STATS,
            burst_state=[],
            watermarks=[],
            verdict=None,
            runs=[],
            last_attempted={},
        ):
            html = client.get('/partials/recon').text
        assert 'No reconciliation runs yet' in html

    def test_no_badges_when_empty(self, client):
        with _patch_recon_data(
            buffer_stats=EMPTY_BUFFER_STATS,
            burst_state=[],
            watermarks=[],
            verdict=None,
            runs=[],
            last_attempted={},
        ):
            html = client.get('/partials/recon').text
        assert 'bg-green-600' not in html
        assert 'bg-red-600' not in html


class TestReconBadgeAriaLabels:
    """Tests for ARIA labels on recon status badges."""

    def test_burst_state_bursting_aria_label(self, client):
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'aria-label="Burst state: bursting"' in html

    def test_burst_state_idle_aria_label(self, client):
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'aria-label="Burst state: idle"' in html

    def test_verdict_severity_ok_aria_label(self, client):
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'aria-label="Verdict severity: ok"' in html

    def test_verdict_severity_minor_aria_label(self, client):
        verdict_minor = {
            'run_id': 'run-001',
            'severity': 'minor',
            'action_taken': 'none',
            'reviewed_at': '2026-03-19T10:00:00+00:00',
        }
        with _patch_recon_data(verdict=verdict_minor):
            html = client.get('/partials/recon').text
        assert 'aria-label="Verdict severity: minor"' in html

    def test_verdict_severity_moderate_aria_label(self, client):
        verdict_mod = {
            'run_id': 'run-001',
            'severity': 'moderate',
            'action_taken': 'repair',
            'reviewed_at': '2026-03-19T10:00:00+00:00',
        }
        with _patch_recon_data(verdict=verdict_mod):
            html = client.get('/partials/recon').text
        assert 'aria-label="Verdict severity: moderate"' in html

    def test_verdict_severity_serious_aria_label(self, client):
        verdict_serious = {
            'run_id': 'run-001',
            'severity': 'serious',
            'action_taken': 'rollback',
            'reviewed_at': '2026-03-19T10:00:00+00:00',
        }
        with _patch_recon_data(verdict=verdict_serious):
            html = client.get('/partials/recon').text
        assert 'aria-label="Verdict severity: serious"' in html

    def test_run_status_completed_aria_label(self, client):
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'aria-label="Run status: completed"' in html

    def test_run_status_running_aria_label(self, client):
        runs_running = [
            {
                'id': 'run-002',
                'project_id': 'dark_factory',
                'run_type': 'full',
                'trigger_reason': 'manual',
                'started_at': '2026-03-19T08:00:00+00:00',
                'completed_at': None,
                'events_processed': 0,
                'status': 'running',
                'duration_seconds': None,
                'journal_entry_count': 0,
            }
        ]
        with _patch_recon_data(runs=runs_running):
            html = client.get('/partials/recon').text
        assert 'aria-label="Run status: running"' in html

    def test_run_status_failed_aria_label(self, client):
        runs_failed = [
            {
                'id': 'run-003',
                'project_id': 'dark_factory',
                'run_type': 'full',
                'trigger_reason': 'manual',
                'started_at': '2026-03-19T08:00:00+00:00',
                'completed_at': '2026-03-19T08:01:00+00:00',
                'events_processed': 0,
                'status': 'failed',
                'duration_seconds': 60.0,
                'journal_entry_count': 0,
            }
        ]
        with _patch_recon_data(runs=runs_failed):
            html = client.get('/partials/recon').text
        assert 'aria-label="Run status: failed"' in html

    def test_run_status_rolled_back_aria_label(self, client):
        runs_rb = [
            {
                'id': 'run-004',
                'project_id': 'dark_factory',
                'run_type': 'full',
                'trigger_reason': 'manual',
                'started_at': '2026-03-19T08:00:00+00:00',
                'completed_at': '2026-03-19T08:01:00+00:00',
                'events_processed': 0,
                'status': 'rolled_back',
                'duration_seconds': 60.0,
                'journal_entry_count': 0,
            }
        ]
        with _patch_recon_data(runs=runs_rb):
            html = client.get('/partials/recon').text
        assert 'aria-label="Run status: rolled_back"' in html

    def test_run_status_circuit_breaker_aria_label(self, client):
        runs_cb = [
            {
                'id': 'run-005',
                'project_id': 'dark_factory',
                'run_type': 'full',
                'trigger_reason': 'manual',
                'started_at': '2026-03-19T08:00:00+00:00',
                'completed_at': '2026-03-19T08:01:00+00:00',
                'events_processed': 0,
                'status': 'circuit_breaker',
                'duration_seconds': 60.0,
                'journal_entry_count': 0,
            }
        ]
        with _patch_recon_data(runs=runs_cb):
            html = client.get('/partials/recon').text
        assert 'aria-label="Run status: circuit_breaker"' in html


class TestReconJournalBadge:
    """Tests for the journal entry count badge in runs table."""

    def test_badge_shown_when_count_positive(self, client):
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        # MOCK_RUNS has journal_entry_count=3
        assert 'data-testid="journal-badge"' in html
        assert '>3<' in html or '>\n                    3\n' in html or '3</button>' in html

    def test_badge_hidden_when_count_zero(self, client):
        runs_no_journal = [
            {
                **MOCK_RUNS[0],
                'journal_entry_count': 0,
            }
        ]
        with _patch_recon_data(runs=runs_no_journal):
            html = client.get('/partials/recon').text
        assert 'data-testid="journal-badge"' not in html

    def test_detail_column_header_present(self, client):
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert '>Detail<' in html or 'Detail</th>' in html

    def test_badge_is_button_element(self, client):
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert '<button' in html and 'data-testid="journal-badge"' in html
        # Extract the badge element and verify it's a button, not a span
        badge_start = html.find('data-testid="journal-badge"')
        assert badge_start != -1
        # Walk back to find the opening tag
        tag_start = html.rfind('<', 0, badge_start)
        assert html[tag_start:tag_start + 7] == '<button'
        assert '</button>' in html

    def test_badge_has_aria_label(self, client):
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        # MOCK_RUNS has journal_entry_count=3
        assert 'aria-label="Show 3 journal entries"' in html

    def test_badge_has_hover_style(self, client):
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'hover:bg-blue-800' in html

    def test_badge_has_focus_ring_styles(self, client):
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'focus:ring-2' in html
        assert 'focus:ring-blue-400' in html

    def test_badge_no_cursor_pointer(self, client):
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        badge_start = html.find('data-testid="journal-badge"')
        assert badge_start != -1
        tag_start = html.rfind('<', 0, badge_start)
        # Find closing > of opening tag
        tag_end = html.find('>', tag_start)
        opening_tag = html[tag_start:tag_end + 1]
        assert 'cursor-pointer' not in opening_tag

    def test_badge_aria_label_dynamic_count(self, client):
        runs_5 = [
            {
                **MOCK_RUNS[0],
                'journal_entry_count': 5,
            }
        ]
        with _patch_recon_data(runs=runs_5):
            html = client.get('/partials/recon').text
        assert 'aria-label="Show 5 journal entries"' in html


class TestReconBufferAgeDisplay:
    """Integration tests for the buffer age display using format_duration filter."""

    def test_large_age_displays_hours_and_minutes(self, client):
        # 62736 seconds = 17h 25m 36s → should render '17h 25m'
        stats = {'buffered_count': 5, 'oldest_event_age_seconds': 62736}
        with _patch_recon_data(buffer_stats=stats):
            html = client.get('/partials/recon').text
        assert '17h 25m' in html

    def test_small_age_displays_minutes_and_seconds(self, client):
        # 600 seconds = 10m 0s
        stats = {'buffered_count': 3, 'oldest_event_age_seconds': 600}
        with _patch_recon_data(buffer_stats=stats):
            html = client.get('/partials/recon').text
        assert '10m 0s' in html

    def test_large_age_does_not_show_raw_minutes_seconds(self, client):
        # Old template would produce '1045m 36s' for 62736s — must not appear
        stats = {'buffered_count': 5, 'oldest_event_age_seconds': 62736}
        with _patch_recon_data(buffer_stats=stats):
            html = client.get('/partials/recon').text
        assert '1045m' not in html

    def test_sub_60s_age_displays_seconds_only(self, client):
        stats = {'buffered_count': 1, 'oldest_event_age_seconds': 45}
        with _patch_recon_data(buffer_stats=stats):
            html = client.get('/partials/recon').text
        assert '45s' in html


class TestProjectNameFilterInTemplate:
    """Tests that project_name filter is applied in the recon template."""

    def test_watermarks_heading_shows_basename_not_full_path(self, client):
        """When project_id is a filesystem path, watermarks heading shows basename."""
        path_watermarks = [
            {
                'project_id': '/home/leo/src/dark-factory',
                'last_full_run_completed': '2026-03-19T10:00:00+00:00',
                'last_episode_timestamp': '2026-03-19T10:30:00+00:00',
                'last_memory_timestamp': '2026-03-19T10:40:00+00:00',
                'last_task_change_timestamp': '2026-03-19T10:50:00+00:00',
            },
            {
                'project_id': '/home/leo/src/other-project',
                'last_full_run_completed': '2026-03-19T09:00:00+00:00',
                'last_episode_timestamp': None,
                'last_memory_timestamp': None,
                'last_task_change_timestamp': None,
            },
        ]
        path_last_attempted = {
            '/home/leo/src/dark-factory': {
                'id': 'run-001',
                'status': 'completed',
                'started_at': '2026-03-19T10:00:00+00:00',
                'completed_at': '2026-03-19T10:05:00+00:00',
            },
        }
        with _patch_recon_data(watermarks=path_watermarks, last_attempted=path_last_attempted):
            html = client.get('/partials/recon').text
        assert 'dark-factory' in html
        assert 'other-project' in html
        assert '/home/leo/src/dark-factory' not in html
        assert '/home/leo/src/other-project' not in html

    def test_runs_table_shows_basename_not_full_path(self, client):
        """When run.project_id is a filesystem path, runs table shows basename."""
        path_runs = [
            {
                'id': 'run-001',
                'project_id': '/home/leo/src/dark-factory',
                'run_type': 'full',
                'trigger_reason': 'staleness_timer',
                'started_at': '2026-03-19T08:00:00+00:00',
                'completed_at': '2026-03-19T08:05:00+00:00',
                'events_processed': 7,
                'status': 'completed',
                'duration_seconds': 300.0,
                'journal_entry_count': 0,
            },
            {
                'id': 'run-002',
                'project_id': '/home/leo/src/other-project',
                'run_type': 'full',
                'trigger_reason': 'manual',
                'started_at': '2026-03-19T09:00:00+00:00',
                'completed_at': '2026-03-19T09:05:00+00:00',
                'events_processed': 3,
                'status': 'completed',
                'duration_seconds': 300.0,
                'journal_entry_count': 0,
            },
        ]
        with _patch_recon_data(runs=path_runs):
            html = client.get('/partials/recon').text
        assert 'dark-factory' in html
        assert 'other-project' in html
        assert '/home/leo/src/dark-factory' not in html
        assert '/home/leo/src/other-project' not in html


class TestReconRunDetailRoute:
    """Tests for GET /partials/recon/run/{run_id}."""

    def test_returns_200(self, client):
        with patch(
            'dashboard.app.get_journal_entries',
            new_callable=AsyncMock,
            return_value=[],
        ):
            resp = client.get('/partials/recon/run/run-001')
        assert resp.status_code == 200

    def test_entries_displayed(self, client):
        mock_entries = [
            {
                'id': 'je-001',
                'stage': 'memory_consolidation',
                'timestamp': '2026-03-19T08:00:00+00:00',
                'operation': 'consolidate',
                'target_system': 'mem0',
                'before_state': {'count': 5},
                'after_state': {'count': 3},
                'reasoning': 'Merged duplicate memories',
                'evidence': [],
            }
        ]
        with patch(
            'dashboard.app.get_journal_entries',
            new_callable=AsyncMock,
            return_value=mock_entries,
        ):
            html = client.get('/partials/recon/run/run-001').text
        assert 'consolidate' in html
        assert 'mem0' in html
        assert 'memory_consolidation' in html
        assert 'Merged duplicate memories' in html

    def test_empty_fallback(self, client):
        with patch(
            'dashboard.app.get_journal_entries',
            new_callable=AsyncMock,
            return_value=[],
        ):
            html = client.get('/partials/recon/run/run-001').text
        assert 'No journal entries for this run.' in html
