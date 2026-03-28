"""Tests for GET /partials/performance route and template rendering."""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

# --- Mock data ---

MOCK_PATHS = {
    'dark_factory': [
        {'path': 'one-pass', 'count': 10, 'pct': 50.0},
        {'path': 'multi-pass', 'count': 4, 'pct': 20.0},
        {'path': 'via-steward', 'count': 3, 'pct': 15.0},
        {'path': 'via-interactive', 'count': 1, 'pct': 5.0},
        {'path': 'blocked', 'count': 2, 'pct': 10.0},
    ],
}

MOCK_ESCALATIONS = {
    'dark_factory': {
        'total_tasks': 20,
        'steward_count': 3,
        'interactive_count': 1,
        'steward_rate': 15.0,
        'interactive_rate': 5.0,
        'human_attention': {'zero': 0, 'minimal': 0, 'significant': 1},
    },
}

MOCK_HISTOGRAMS = {
    'dark_factory': {
        'outer': {'labels': ['0', '1', '2', '3+'], 'values': [10, 5, 3, 2]},
        'inner': {'labels': ['0', '1', '2', '3', '4', '5+'], 'values': [8, 6, 3, 2, 1, 0]},
    },
}

MOCK_TTC = {
    'dark_factory': {'p50': 300_000, 'p75': 450_000, 'p90': 600_000, 'p95': 900_000, 'count': 18},
}

_UNSET = object()


def _patch_perf_data(paths=_UNSET, escalations=_UNSET, histograms=_UNSET, ttc=_UNSET):
    """Return an ExitStack that patches all 4 performance data functions."""
    stack = ExitStack()
    stack.enter_context(patch(
        'dashboard.app.get_completion_paths',
        new_callable=AsyncMock,
        return_value=paths if paths is not _UNSET else MOCK_PATHS,
    ))
    stack.enter_context(patch(
        'dashboard.app.get_escalation_rates',
        new_callable=AsyncMock,
        return_value=escalations if escalations is not _UNSET else MOCK_ESCALATIONS,
    ))
    stack.enter_context(patch(
        'dashboard.app.get_loop_histograms',
        new_callable=AsyncMock,
        return_value=histograms if histograms is not _UNSET else MOCK_HISTOGRAMS,
    ))
    stack.enter_context(patch(
        'dashboard.app.get_time_centiles',
        new_callable=AsyncMock,
        return_value=ttc if ttc is not _UNSET else MOCK_TTC,
    ))
    return stack


class TestPerformanceRoute:
    """Tests for GET /partials/performance with populated data."""

    def test_returns_200(self, client):
        with _patch_perf_data():
            resp = client.get('/partials/performance')
        assert resp.status_code == 200

    def test_content_type_html(self, client):
        with _patch_perf_data():
            resp = client.get('/partials/performance')
        assert 'text/html' in resp.headers['content-type']

    def test_heading_present(self, client):
        with _patch_perf_data():
            html = client.get('/partials/performance').text
        assert 'Performance' in html

    def test_project_section(self, client):
        with _patch_perf_data():
            html = client.get('/partials/performance').text
        assert 'dark_factory' in html
        assert 'data-testid="perf-project-dark_factory"' in html

    def test_completion_paths_labels(self, client):
        with _patch_perf_data():
            html = client.get('/partials/performance').text
        assert 'one-pass' in html
        assert 'multi-pass' in html
        assert 'via-steward' in html
        assert 'via-interactive' in html
        assert 'blocked' in html

    def test_completion_paths_counts(self, client):
        with _patch_perf_data():
            html = client.get('/partials/performance').text
        assert '50.0%' in html
        assert '20.0%' in html

    def test_escalation_rates(self, client):
        with _patch_perf_data():
            html = client.get('/partials/performance').text
        assert '15.0%' in html  # steward_rate
        assert '5.0%' in html   # interactive_rate
        assert 'Steward' in html
        assert 'Interactive' in html

    def test_human_attention_significant(self, client):
        with _patch_perf_data():
            html = client.get('/partials/performance').text
        assert '1 significant' in html

    def test_histogram_canvas_ids(self, client):
        with _patch_perf_data():
            html = client.get('/partials/performance').text
        assert 'outerHistChart-dark_factory' in html
        assert 'innerHistChart-dark_factory' in html

    def test_histogram_labels(self, client):
        with _patch_perf_data():
            html = client.get('/partials/performance').text
        assert 'Review Cycles (outer loop)' in html
        assert 'Verify Attempts (inner loop)' in html

    def test_time_centiles(self, client):
        with _patch_perf_data():
            html = client.get('/partials/performance').text
        assert 'p50' in html
        assert 'p75' in html
        assert 'p90' in html
        assert 'p95' in html
        # 300_000ms = 5m
        assert '5m' in html

    def test_task_count_in_centiles(self, client):
        with _patch_perf_data():
            html = client.get('/partials/performance').text
        assert '18 tasks' in html

    def test_path_chart_canvas(self, client):
        with _patch_perf_data():
            html = client.get('/partials/performance').text
        assert 'pathChart-dark_factory' in html

    def test_chart_js_script(self, client):
        with _patch_perf_data():
            html = client.get('/partials/performance').text
        assert 'new Chart' in html
        assert 'pathColors' in html


class TestPerformanceRouteEmpty:
    """Tests for GET /partials/performance with empty data."""

    def test_returns_200(self, client):
        with _patch_perf_data(paths={}, escalations={}, histograms={}, ttc={}):
            resp = client.get('/partials/performance')
        assert resp.status_code == 200

    def test_empty_state_message(self, client):
        with _patch_perf_data(paths={}, escalations={}, histograms={}, ttc={}):
            html = client.get('/partials/performance').text
        assert 'No orchestrator run data yet' in html

    def test_no_chart_script_when_empty(self, client):
        with _patch_perf_data(paths={}, escalations={}, histograms={}, ttc={}):
            html = client.get('/partials/performance').text
        assert 'new Chart' not in html


class TestHistogramUniformColor:
    """Tests for histogram uniform bar color (not per-bar multi-color)."""

    def test_histogram_does_not_use_per_bar_palette_coloring(self, client):
        """Histogram bars must NOT use histPalette[i] per-bar coloring pattern."""
        with _patch_perf_data():
            html = client.get('/partials/performance').text
        # The old per-bar coloring pattern should be absent
        assert 'histPalette[i % histPalette.length]' not in html
        assert 'histPalette[i]' not in html

    def test_histogram_uses_uniform_color(self, client):
        """Histogram bar backgroundColor should be a single uniform color string."""
        with _patch_perf_data():
            html = client.get('/partials/performance').text
        # A single color like '#60a5fa' should be used for all histogram bars
        assert '#60a5fa' in html


class TestCompletionPathsDatalabels:
    """Tests for datalabels plugin config on completion-paths doughnut chart."""

    def test_completion_paths_doughnut_has_datalabels(self, client):
        """Completion-paths doughnut chart script must contain datalabels configuration."""
        with _patch_perf_data():
            html = client.get('/partials/performance').text
        assert 'datalabels' in html

    def test_completion_paths_doughnut_has_percentage_formatter(self, client):
        """Completion-paths doughnut chart datalabels must include a percentage formatter."""
        with _patch_perf_data():
            html = client.get('/partials/performance').text
        assert 'formatter' in html
        assert '%' in html or 'toFixed' in html


class TestChartInitTiming:
    """Tests for chart initialization timing fix (htmx:afterSettle instead of double-rAF)."""

    def test_no_request_animation_frame(self, client):
        """Double requestAnimationFrame pattern must be absent from rendered HTML."""
        with _patch_perf_data():
            html = client.get('/partials/performance').text
        assert 'requestAnimationFrame' not in html

    def test_htmx_after_settle_listener(self, client):
        """htmx:afterSettle one-shot event listener must be present in rendered HTML."""
        with _patch_perf_data():
            html = client.get('/partials/performance').text
        assert 'htmx:afterSettle' in html

    def test_render_all_function_present(self, client):
        """renderAll function must still be present after the timing fix."""
        with _patch_perf_data():
            html = client.get('/partials/performance').text
        assert 'renderAll' in html


class TestFormatDurationMs:
    """Tests for the format_duration_ms Jinja2 filter."""

    def test_none_returns_dash(self):
        from dashboard.app import format_duration_ms
        assert format_duration_ms(None) == '-'

    def test_zero_returns_dash(self):
        from dashboard.app import format_duration_ms
        assert format_duration_ms(0) == '-'

    def test_seconds(self):
        from dashboard.app import format_duration_ms
        assert format_duration_ms(45_000) == '45s'

    def test_minutes(self):
        from dashboard.app import format_duration_ms
        assert format_duration_ms(300_000) == '5m'

    def test_hours(self):
        from dashboard.app import format_duration_ms
        assert format_duration_ms(7_200_000) == '2.0h'

    def test_filter_registered(self):
        from dashboard.app import format_duration_ms, templates
        assert templates.env.filters['format_duration_ms'] is format_duration_ms
