"""Tests for the reconciliation panel route and supporting utilities."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

from .test_helpers import _get_opening_tag


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

    def test_none_returns_dash(self):
        from dashboard.app import format_duration

        assert format_duration(None) == '-'

    def test_negative_value_returns_dash(self):
        from dashboard.app import format_duration

        assert format_duration(-30) == '-'

    def test_non_numeric_returns_dash(self):
        from dashboard.app import format_duration

        assert format_duration('not_a_number') == '-'

    def test_infinity_returns_dash(self):
        from dashboard.app import format_duration

        assert format_duration(float('inf')) == '-'

    def test_sub_second_positive_rounds_up(self):
        from dashboard.app import format_duration

        assert format_duration(0.7) == '1s'


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

    def test_large_age_displays_yellow_color(self, client):
        with _patch_recon_data(buffer_stats={'buffered_count': 3, 'oldest_event_age_seconds': 600.0}):
            html = client.get('/partials/recon').text
        assert 'text-yellow-400' in html

    def test_exactly_300s_age_displays_gray_color(self, client):
        with _patch_recon_data(buffer_stats={'buffered_count': 1, 'oldest_event_age_seconds': 300.0}):
            html = client.get('/partials/recon').text
        assert 'text-gray-400' in html
        assert 'text-yellow-400' not in html

    def test_301s_age_displays_yellow_color(self, client):
        with _patch_recon_data(buffer_stats={'buffered_count': 1, 'oldest_event_age_seconds': 301.0}):
            html = client.get('/partials/recon').text
        assert 'text-yellow-400' in html


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
        tag = _get_opening_tag(html, 'data-testid="journal-badge"')
        assert tag.startswith('<button')
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
        opening_tag = _get_opening_tag(html, 'data-testid="journal-badge"')
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


@pytest.fixture(scope='class')
def recon_layout_html():
    """Class-scoped HTML fixture: fetches /partials/recon once for all layout tests."""
    from dashboard.app import app

    with TestClient(app) as c, _patch_recon_data():
        return c.get('/partials/recon').text


class TestReconRightColumnLayout:
    """Tests for right-column layout: no absolute positioning, natural grid flow."""

    def test_no_lg_absolute_in_html(self, recon_layout_html):
        assert 'lg:absolute' not in recon_layout_html

    def test_no_lg_inset_0_in_html(self, recon_layout_html):
        assert 'lg:inset-0' not in recon_layout_html

    def test_right_column_card_has_flex_col_h_full(self, recon_layout_html):
        assert 'flex flex-col h-full' in recon_layout_html

    def test_no_min_h_400_in_html(self, recon_layout_html):
        assert 'min-h-[400px]' not in recon_layout_html

    def test_no_lg_relative_in_html(self, recon_layout_html):
        assert 'lg:relative' not in recon_layout_html


class TestReconTableHeaderNoWrap:
    """Test that thead tr has whitespace-nowrap to prevent header truncation."""

    def test_thead_row_has_whitespace_nowrap(self, client):
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'border-b border-gray-700 whitespace-nowrap' in html


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


class TestVerdictAlertBanner:
    """Tests for the serious verdict alert banner in recon.html."""

    def test_serious_verdict_shows_alert_banner(self, client):
        verdict_serious = {
            'run_id': 'run-001',
            'severity': 'serious',
            'action_taken': 'rollback',
            'reviewed_at': '2026-03-19T10:00:00+00:00',
        }
        with _patch_recon_data(verdict=verdict_serious):
            html = client.get('/partials/recon').text
        assert 'data-testid="verdict-alert"' in html
        assert 'bg-red-900' in html
        assert 'serious' in html
        assert 'rollback' in html

    def test_serious_verdict_banner_contains_explanation(self, client):
        verdict_serious = {
            'run_id': 'run-001',
            'severity': 'serious',
            'action_taken': 'halt',
            'reviewed_at': '2026-03-19T10:00:00+00:00',
        }
        with _patch_recon_data(verdict=verdict_serious):
            html = client.get('/partials/recon').text
        assert 'Reconciliation detected a serious issue requiring attention' in html

    def test_ok_verdict_no_alert_banner(self, client):
        # MOCK_VERDICT has severity='ok'
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'data-testid="verdict-alert"' not in html

    def test_minor_verdict_no_alert_banner(self, client):
        verdict_minor = {
            'run_id': 'run-001',
            'severity': 'minor',
            'action_taken': 'none',
            'reviewed_at': '2026-03-19T10:00:00+00:00',
        }
        with _patch_recon_data(verdict=verdict_minor):
            html = client.get('/partials/recon').text
        assert 'data-testid="verdict-alert"' not in html

    def test_none_verdict_no_alert_banner(self, client):
        with _patch_recon_data(verdict=None):
            html = client.get('/partials/recon').text
        assert 'data-testid="verdict-alert"' not in html


class TestVerdictCardExplanation:
    """Tests for the human-readable explanation in the Latest Verdict card."""

    def test_halt_action_shows_explanation(self, client):
        verdict_halt = {
            'run_id': 'run-001',
            'severity': 'serious',
            'action_taken': 'halt',
            'reviewed_at': '2026-03-19T10:00:00+00:00',
        }
        with _patch_recon_data(verdict=verdict_halt):
            html = client.get('/partials/recon').text
        assert 'data-testid="verdict-explanation"' in html
        assert 'Reconciliation halted' in html
        assert 'manual review required' in html

    def test_rollback_action_shows_explanation(self, client):
        verdict_rollback = {
            'run_id': 'run-001',
            'severity': 'serious',
            'action_taken': 'rollback',
            'reviewed_at': '2026-03-19T10:00:00+00:00',
        }
        with _patch_recon_data(verdict=verdict_rollback):
            html = client.get('/partials/recon').text
        assert 'data-testid="verdict-explanation"' in html
        assert 'Changes were rolled back' in html

    def test_repair_action_shows_explanation(self, client):
        verdict_repair = {
            'run_id': 'run-001',
            'severity': 'moderate',
            'action_taken': 'repair',
            'reviewed_at': '2026-03-19T10:00:00+00:00',
        }
        with _patch_recon_data(verdict=verdict_repair):
            html = client.get('/partials/recon').text
        assert 'data-testid="verdict-explanation"' in html
        assert 'Automatic repair was attempted' in html

    def test_none_action_shows_explanation(self, client):
        # MOCK_VERDICT has action_taken='none'
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'data-testid="verdict-explanation"' in html
        assert 'No action needed' in html

    def test_no_verdict_no_explanation(self, client):
        with _patch_recon_data(verdict=None):
            html = client.get('/partials/recon').text
        assert 'data-testid="verdict-explanation"' not in html


class TestRunPanelJournalBadgeRegression:
    """Regression tests: badge behaviours must survive the Alpine component refactor."""

    def test_badge_visible_with_journal_entries(self, client):
        """Badge renders when journal_entry_count > 0."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'data-testid="journal-badge"' in html

    def test_badge_hidden_with_zero_entries(self, client):
        """Badge not rendered when journal_entry_count == 0."""
        runs_zero = [{**MOCK_RUNS[0], 'journal_entry_count': 0}]
        with _patch_recon_data(runs=runs_zero):
            html = client.get('/partials/recon').text
        assert 'data-testid="journal-badge"' not in html

    def test_badge_aria_label_contains_count(self, client):
        """aria-label reflects the entry count."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'aria-label="Show 3 journal entries"' in html

    def test_badge_is_button_element(self, client):
        """Badge is a <button>, not a <span> or <div>."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        tag = _get_opening_tag(html, 'data-testid="journal-badge"')
        assert tag.startswith('<button')

    def test_badge_has_blue_bg_class(self, client):
        """Badge retains its blue background CSS class."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'bg-blue-900' in html

    def test_badge_has_hover_style(self, client):
        """Badge retains hover style."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'hover:bg-blue-800' in html

    def test_badge_has_focus_ring(self, client):
        """Badge retains focus-ring accessibility styles."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'focus:ring-2' in html
        assert 'focus:ring-blue-400' in html

    def test_x_show_open_still_on_detail_row(self, client):
        """Detail row still uses x-show=\"open\" for show/hide."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'x-show="open"' in html

    def test_x_cloak_still_on_detail_row(self, client):
        """Detail row still has x-cloak to prevent flash of unstyled content."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'x-cloak' in html


class TestPartitionBurstStateNoneGuard:
    """Direct classification test for partition_burst_state with None last_write_at."""

    def test_none_last_write_at_classified_as_idle(self):
        """Agent with last_write_at=None should be classified as idle, not active."""
        from dashboard.data.reconciliation import partition_burst_state

        agent = {'agent_id': 'a1', 'state': 'idle', 'last_write_at': None}
        active, idle = partition_burst_state([agent])
        assert idle == [agent]
        assert active == []


class TestRunPanelAlpineComponent:
    """Tests for the named Alpine.data('runPanel') component in recon.html."""

    def test_alpine_data_run_panel_definition_present(self, client):
        """Alpine.data('runPanel') script block is rendered when journal entries exist."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert "Alpine.data('runPanel'" in html

    def test_run_panel_script_not_inside_alpine_init(self, client):
        """Registration must NOT be wrapped in alpine:init (partial loads after init)."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert "alpine:init" not in html

    def test_tbody_uses_named_component(self, client):
        """tbody uses x-data=\"runPanel\" (named component, not inline object)."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'x-data="runPanel"' in html

    def test_inline_open_false_not_present(self, client):
        """Old inline x-data=\"{ open: false }\" pattern is removed."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'x-data="{ open: false }"' not in html

    def test_badge_button_uses_toggle_detail_handler(self, client):
        """Badge button uses @click=\"toggleDetail()\" not inline open = !open."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert '@click="toggleDetail()"' in html

    def test_old_inline_click_handler_gone(self, client):
        """Old @click=\"open = !open\" handler is removed from the badge button."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert '@click="open = !open"' not in html

    def test_detail_div_uses_custom_trigger(self, client):
        """Detail div uses hx-trigger=\"load-detail\" custom event (not \"revealed\")."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'hx-trigger="load-detail"' in html

    def test_revealed_trigger_removed(self, client):
        """Old hx-trigger=\"revealed\" is no longer present in the recon partial."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'hx-trigger="revealed"' not in html

    def test_detail_div_has_x_ref(self, client):
        """Detail div has x-ref=\"detail\" for Alpine $refs access."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'x-ref="detail"' in html

    def test_detail_div_has_after_swap_handler(self, client):
        """Detail div has hx-on::after-swap to set dataset.loaded flag."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'hx-on::after-swap' in html

    def test_after_swap_sets_loaded_flag(self, client):
        """hx-on::after-swap sets this.dataset.loaded='true'."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert "this.dataset.loaded = 'true'" in html

    def test_after_swap_removes_loading_flag(self, client):
        """hx-on::after-swap deletes this.dataset.loading."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'delete this.dataset.loading' in html

    def test_toggle_detail_guards_against_loaded(self, client):
        """toggleDetail() checks detail.dataset.loaded before dispatching event."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'detail.dataset.loaded' in html

    def test_toggle_detail_guards_against_loading(self, client):
        """toggleDetail() checks detail.dataset.loading before dispatching event."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'detail.dataset.loading' in html

    def test_toggle_detail_sets_loading_synchronously(self, client):
        """toggleDetail() sets detail.dataset.loading='true' before HTMX trigger."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert "detail.dataset.loading = 'true'" in html

    def test_toggle_detail_dispatches_load_detail_event(self, client):
        """toggleDetail() calls htmx.trigger(detail, 'load-detail')."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert "htmx.trigger(detail, 'load-detail')" in html

    def test_toggle_detail_early_returns_when_closing(self, client):
        """toggleDetail() early-returns when open becomes false (closing)."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'if (!this.open)' in html

    def test_toggle_detail_uses_refs_detail(self, client):
        """toggleDetail() uses this.$refs.detail to access the detail element."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'this.$refs.detail' in html

    def test_toggle_detail_null_guards_refs_detail(self, client):
        """toggleDetail() guards against undefined $refs.detail with 'if (!detail) return'."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'if (!detail) return' in html

    def test_null_guard_appears_before_dataset_access(self, client):
        """Null guard appears immediately after $refs lookup, before dataset access."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        refs_pos = html.index('const detail = this.$refs.detail')
        null_guard_pos = html.index('if (!detail) return')
        dataset_pos = html.index('detail.dataset.loaded')
        assert refs_pos < null_guard_pos < dataset_pos

    def test_detail_div_has_after_request_handler(self, client):
        """Detail div has hx-on::after-request to clear loading flag on failure."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        assert 'hx-on::after-request' in html

    def test_after_request_clears_loading_on_failure(self, client):
        """hx-on::after-request clears dataset.loading when request fails."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        # Handler must check for failure via event.detail.failed or !event.detail.successful
        assert 'event.detail.failed' in html or '!event.detail.successful' in html

    def test_after_request_deletes_loading_on_failure(self, client):
        """hx-on::after-request deletes dataset.loading to allow retry after failure."""
        with _patch_recon_data():
            html = client.get('/partials/recon').text
        # The failure handler must delete this.dataset.loading so retry is possible
        # Extract the after-request handler value from the HTML
        assert 'hx-on::after-request' in html
        after_req_idx = html.index('hx-on::after-request')
        # Ensure delete this.dataset.loading appears near the after-request handler
        segment = html[after_req_idx:after_req_idx + 200]
        assert 'delete this.dataset.loading' in segment
