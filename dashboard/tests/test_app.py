"""Integration tests for the full dashboard application."""

from __future__ import annotations

import asyncio
import os
import re
import runpy
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

import pytest

from dashboard.app import _safe_gather_result

PARTIAL_URLS = (
    "/partials/memory",
    "/partials/recon",
    "/partials/orchestrators",
    "/partials/performance",
    "/partials/memory-graphs",
)

SECTION_TIMEOUTS = (
    ("/partials/memory", 8000),
    ("/partials/recon", 12000),
    ("/partials/orchestrators", 8000),
    ("/partials/performance", 12000),
    ("/partials/memory-graphs", 10000),
)


def _get_section_window(
    html: str,
    partial_url: str,
    *,
    # before=200: captures the <section> tag and data-section attribute that precede hx-get
    # after=500: captures hx-trigger, hx-swap, hx-request (with timeout JSON), and aria-live
    # that follow hx-get in the polling_section macro attribute order:
    # data-section → hx-get → hx-trigger → hx-swap → hx-request → aria-live
    before: int = 200,
    after: int = 500,
) -> str:
    """Return an HTML substring window centred on hx-get="<partial_url>".

    Raises AssertionError (not ValueError) if the URL is not found, giving a
    clear diagnostic message rather than a bare traceback from str.index().
    """
    hx_get = f'hx-get="{partial_url}"'
    idx = html.find(hx_get)
    assert idx != -1, f'hx-get for {partial_url} not found in HTML'
    return html[max(0, idx - before):idx + after]


def _get_nav_link(html: str, href: str) -> str:
    """Return the opening <a> tag for a nav link with the given href.

    Uses re.search to extract the tag, enabling scoped class assertions —
    e.g. asserting 'border-blue-500' is present in the *active* link or
    absent from the *inactive* link without false positives from other elements.
    Raises AssertionError with a diagnostic message if the link is not found.
    """
    pattern = r'<a\s+[^>]*href="' + re.escape(href) + r'"[^>]*>'
    m = re.search(pattern, html)
    assert m is not None, f'No <a href="{href}"> found in HTML'
    return m.group(0)


class TestGetSectionWindow:
    """Tests for the _get_section_window helper function."""

    def test_returns_correct_window_around_known_hx_get(self):
        # Simulate HTML with hx-get near the start of a section block
        html = 'X' * 300 + 'hx-get="/partials/memory"' + 'Y' * 600
        window = _get_section_window(html, '/partials/memory')
        assert 'hx-get="/partials/memory"' in window

    def test_raises_assertion_error_when_not_found(self):
        html = '<div>some html without the url</div>'
        with pytest.raises(AssertionError, match='/partials/missing'):
            _get_section_window(html, '/partials/missing')

    def test_clamps_to_string_start_boundary(self):
        # hx-get is near the very beginning of the string (idx < before)
        html = 'hx-get="/partials/memory"' + 'Y' * 600
        window = _get_section_window(html, '/partials/memory')
        # Should not raise IndexError and should include the hx-get
        assert 'hx-get="/partials/memory"' in window

    def test_respects_custom_before_after_overrides(self):
        html = 'A' * 50 + 'hx-get="/partials/memory"' + 'B' * 50
        window = _get_section_window(html, '/partials/memory', before=10, after=10)
        hx_get = 'hx-get="/partials/memory"'
        idx = html.find(hx_get)
        expected = html[idx - 10:idx + 10]
        assert window == expected


class TestIdiomorphExtension:
    """Tests for idiomorph extension setup in base.html."""

    def test_idiomorph_script_tag(self, client):
        html = client.get('/').text
        assert 'unpkg.com/idiomorph@0.3.0/dist/idiomorph-ext.min.js' in html

    def test_body_has_morph_extension(self, client):
        html = client.get('/').text
        assert 'hx-ext="morph"' in html


class TestMorphSwap:
    """Tests that all polling sections use morph:innerHTML swap strategy."""

    @pytest.mark.parametrize('partial_url', PARTIAL_URLS)
    def test_section_uses_morph_swap(self, client, partial_url):
        html = client.get('/').text
        window = _get_section_window(html, partial_url)
        assert 'hx-swap="morph:innerHTML"' in window

    def test_no_plain_innerhtml_on_polling_sections(self, client):
        html = client.get('/').text
        assert 'hx-swap="innerHTML"' not in html


class TestLifespan:
    """Tests for the app lifespan — verifies http_client configuration."""

    def test_lifespan_sets_follow_redirects(self, client):
        """The http_client created during lifespan has follow_redirects=True."""
        from dashboard.app import app

        assert app.state.http_client.follow_redirects is True, (
            'http_client must be created with follow_redirects=True '
            'to handle Starlette 307 redirects from /mcp/ to /mcp'
        )


class TestIndex:
    """Tests for GET / — the main dashboard page."""

    def test_get_root_returns_200(self, client):
        resp = client.get('/')
        assert resp.status_code == 200

    def test_get_root_content_type_html(self, client):
        resp = client.get('/')
        assert 'text/html' in resp.headers['content-type']

    def test_get_root_contains_dark_factory(self, client):
        html = client.get('/').text
        assert 'Dark Factory' in html

    def test_loading_skeletons_present(self, client):
        html = client.get('/').text
        assert 'animate-pulse' in html
        assert '<p class="text-gray-400">Loading...</p>' not in html


class TestHealthIntegration:
    """Tests for GET /api/health."""

    def test_health_returns_200(self, client):
        resp = client.get('/api/health')
        assert resp.status_code == 200

    def test_health_content_type_json(self, client):
        resp = client.get('/api/health')
        assert 'application/json' in resp.headers['content-type']

    def test_health_body(self, client):
        resp = client.get('/api/health')
        assert resp.json() == {'status': 'ok'}


class TestMemoryPartialIntegration:
    """Integration tests for GET /partials/memory."""

    def test_memory_online(self, client):
        mock_status = {
            'graphiti': {'connected': True},
            'mem0': {'connected': True},
            'projects': {
                'dark_factory': {'graphiti_nodes': 42, 'mem0_memories': 128},
            },
        }
        mock_queue = {
            'counts': {'pending': 0, 'retry': 0, 'dead': 0},
            'oldest_pending_age_seconds': None,
        }
        with (
            patch(
                'dashboard.data.memory.get_memory_status',
                new_callable=AsyncMock,
                return_value=mock_status,
            ),
            patch(
                'dashboard.data.memory.get_queue_stats',
                new_callable=AsyncMock,
                return_value=mock_queue,
            ),
        ):
            resp = client.get('/partials/memory')
            assert resp.status_code == 200
            assert 'text/html' in resp.headers['content-type']
            html = resp.text
            assert 'Graphiti' in html
            assert 'Mem0' in html
            assert 'Taskmaster' in html
            assert 'Write Queue' in html

    def test_memory_offline(self, client):
        mock_status = {'offline': True, 'error': 'Connection refused'}
        mock_queue = {'offline': True, 'error': 'Connection refused'}
        with (
            patch(
                'dashboard.data.memory.get_memory_status',
                new_callable=AsyncMock,
                return_value=mock_status,
            ),
            patch(
                'dashboard.data.memory.get_queue_stats',
                new_callable=AsyncMock,
                return_value=mock_queue,
            ),
        ):
            resp = client.get('/partials/memory')
            assert resp.status_code == 200
            html = resp.text
            assert 'Offline' in html


_UNSET = object()


def _patch_recon_data(
    buffer_stats=_UNSET,
    burst_state=_UNSET,
    watermarks=_UNSET,
    verdict=_UNSET,
    runs=_UNSET,
    last_attempted=_UNSET,
):
    """Return an ExitStack that patches all 6 recon data functions."""
    defaults = {
        'buffer_stats': buffer_stats if buffer_stats is not _UNSET else {
            'buffered_count': 3, 'oldest_event_age_seconds': 600.0,
        },
        'burst_state': burst_state if burst_state is not _UNSET else [
            {
                'agent_id': 'agent-1',
                'state': 'bursting',
                'last_write_at': '2026-03-19T00:00:00+00:00',
                'burst_started_at': '2026-03-19T00:00:00+00:00',
            },
        ],
        'watermarks': watermarks if watermarks is not _UNSET else [
            {
                'project_id': 'dark_factory',
                'last_full_run_completed': '2026-03-19T10:00:00+00:00',
                'last_episode_timestamp': None,
                'last_memory_timestamp': None,
                'last_task_change_timestamp': None,
            },
        ],
        'verdict': verdict if verdict is not _UNSET else None,
        'runs': runs if runs is not _UNSET else [
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
                'journal_entry_count': 0,
            },
        ],
        'last_attempted': last_attempted if last_attempted is not _UNSET else {
            'dark_factory': {
                'id': 'run-002',
                'status': 'failed',
                'started_at': '2026-03-19T09:00:00+00:00',
                'completed_at': '2026-03-19T09:01:00+00:00',
            },
        },
    }
    stack = ExitStack()
    stack.enter_context(patch(
        'dashboard.app.get_buffer_stats',
        new_callable=AsyncMock,
        return_value=defaults['buffer_stats'],
    ))
    stack.enter_context(patch(
        'dashboard.app.get_burst_state',
        new_callable=AsyncMock,
        return_value=defaults['burst_state'],
    ))
    stack.enter_context(patch(
        'dashboard.app.get_watermarks',
        new_callable=AsyncMock,
        return_value=defaults['watermarks'],
    ))
    stack.enter_context(patch(
        'dashboard.app.get_latest_verdict',
        new_callable=AsyncMock,
        return_value=defaults['verdict'],
    ))
    stack.enter_context(patch(
        'dashboard.app.get_recent_runs',
        new_callable=AsyncMock,
        return_value=defaults['runs'],
    ))
    stack.enter_context(patch(
        'dashboard.app.get_last_attempted_run',
        new_callable=AsyncMock,
        return_value=defaults['last_attempted'],
    ))
    return stack


class TestReconPartialIntegration:
    """Integration tests for GET /partials/recon."""

    def test_recon_with_data(self, client):
        with _patch_recon_data():
            resp = client.get('/partials/recon')
            assert resp.status_code == 200
            assert 'text/html' in resp.headers['content-type']
            html = resp.text
            assert 'staleness_timer' in html
            assert 'completed' in html

    def test_recon_empty(self, client):
        with _patch_recon_data(
            buffer_stats={'buffered_count': 0, 'oldest_event_age_seconds': None},
            burst_state=[],
            watermarks=[],
            runs=[],
            last_attempted={},
        ):
            resp = client.get('/partials/recon')
            assert resp.status_code == 200
            html = resp.text
            assert 'No reconciliation runs' in html

    @pytest.mark.parametrize('failing_fn', [
        'get_buffer_stats',
        'get_burst_state',
        'get_watermarks',
        'get_latest_verdict',
        'get_recent_runs',
        'get_last_attempted_run',
    ])
    def test_recon_partial_failure(self, client, failing_fn):
        """One failing recon coroutine should not cause a 500."""
        with _patch_recon_data(), patch(
            f'dashboard.app.{failing_fn}',
            new_callable=AsyncMock,
            side_effect=RuntimeError('injected error'),
        ):
            resp = client.get('/partials/recon')
            assert resp.status_code == 200
            html = resp.text
            # When runs fails, the other data (staleness_timer) still renders
            if failing_fn != 'get_recent_runs':
                assert 'staleness_timer' in html
            # When runs fails, at least the page renders without a 500
            else:
                assert 'No reconciliation runs' in html


_MOCK_ORCHESTRATOR = {
    'pids': [1234],
    'prd': '/home/leo/src/dark-factory/prd/dashboard.md',
    'running': True,
    'started': 'Mar18',
    'tasks': [
        {'id': 1, 'title': 'Setup infra', 'status': 'done', 'priority': 'high', 'dependencies': [], 'metadata': {}},
        {'id': 2, 'title': 'Build API', 'status': 'in-progress', 'priority': 'medium', 'dependencies': [1], 'metadata': {}},
    ],
    'worktrees': {
        '1': {
            'phase': 'DONE',
            'plan_progress': {'done': 3, 'total': 3},
            'iteration_count': 2,
            'review_summary': '2/2 passed',
            'modules': ['infra/'],
        },
    },
    'summary': {
        'total': 2,
        'done': 1,
        'in_progress': 1,
        'blocked': 0,
        'pending': 0,
    },
}


class TestOrchestratorsPartialIntegration:
    """Integration tests for GET /partials/orchestrators."""

    def test_orchestrators_with_data(self, client):
        with patch('dashboard.app.discover_orchestrators', return_value=[_MOCK_ORCHESTRATOR]):
            resp = client.get('/partials/orchestrators')
            assert resp.status_code == 200
            assert 'text/html' in resp.headers['content-type']
            html = resp.text
            assert '1234' in html
            assert 'dashboard.md' in html
            assert '1 done' in html

    def test_orchestrators_empty(self, client):
        with patch('dashboard.app.discover_orchestrators', return_value=[]):
            resp = client.get('/partials/orchestrators')
            assert resp.status_code == 200
            html = resp.text
            assert 'No orchestrators detected' in html


_PERF_PATHS = {
    'dark_factory': [
        {'path': 'one-pass', 'count': 10, 'pct': 50.0},
        {'path': 'multi-pass', 'count': 4, 'pct': 20.0},
        {'path': 'via-steward', 'count': 3, 'pct': 15.0},
    ],
}
_PERF_ESCALATIONS = {
    'dark_factory': {
        'total_tasks': 20,
        'steward_count': 3,
        'interactive_count': 1,
        'steward_rate': 15.0,
        'interactive_rate': 5.0,
        'human_attention': {'zero': 0, 'minimal': 0, 'significant': 1},
    },
}
_PERF_HISTOGRAMS = {
    'dark_factory': {
        'outer': {'labels': ['0', '1', '2', '3+'], 'values': [10, 5, 3, 2]},
        'inner': {'labels': ['0', '1', '2', '3', '4', '5+'], 'values': [8, 6, 3, 2, 1, 0]},
    },
}
_PERF_TTC = {
    'dark_factory': {'p50': 300_000, 'p75': 450_000, 'p90': 600_000, 'p95': 900_000, 'count': 18},
}


def _patch_perf_integration(paths=_UNSET, escalations=_UNSET, histograms=_UNSET, ttc=_UNSET):
    """Return an ExitStack that patches all 4 performance data functions."""
    defaults = {
        'paths': paths if paths is not _UNSET else _PERF_PATHS,
        'escalations': escalations if escalations is not _UNSET else _PERF_ESCALATIONS,
        'histograms': histograms if histograms is not _UNSET else _PERF_HISTOGRAMS,
        'ttc': ttc if ttc is not _UNSET else _PERF_TTC,
    }
    stack = ExitStack()
    stack.enter_context(patch(
        'dashboard.app.get_completion_paths',
        new_callable=AsyncMock,
        return_value=defaults['paths'],
    ))
    stack.enter_context(patch(
        'dashboard.app.get_escalation_rates',
        new_callable=AsyncMock,
        return_value=defaults['escalations'],
    ))
    stack.enter_context(patch(
        'dashboard.app.get_loop_histograms',
        new_callable=AsyncMock,
        return_value=defaults['histograms'],
    ))
    stack.enter_context(patch(
        'dashboard.app.get_time_centiles',
        new_callable=AsyncMock,
        return_value=defaults['ttc'],
    ))
    return stack


class TestPerformancePartialIntegration:
    """Integration tests for GET /partials/performance."""

    def test_performance_with_data(self, client):
        with _patch_perf_integration():
            resp = client.get('/partials/performance')
            assert resp.status_code == 200
            assert 'text/html' in resp.headers['content-type']
            html = resp.text
            assert 'Performance' in html
            assert 'dark_factory' in html
            assert 'one-pass' in html
            assert 'Steward' in html
            assert 'p50' in html
            assert '5m' in html
            assert 'new Chart' in html

    def test_performance_empty(self, client):
        with _patch_perf_integration(paths={}, escalations={}, histograms={}, ttc={}):
            resp = client.get('/partials/performance')
            assert resp.status_code == 200
            html = resp.text
            assert 'No orchestrator run data yet' in html

    def test_performance_backend_error_degrades_gracefully(self, client):
        # With return_exceptions=True, other results are preserved even when one fails.
        # get_completion_paths failing → paths={} but ttc still has data, so else branch renders.
        with _patch_perf_integration(), patch(
            'dashboard.app.get_completion_paths',
            new_callable=AsyncMock,
            side_effect=RuntimeError('db unavailable'),
        ):
            resp = client.get('/partials/performance')
            assert resp.status_code == 200

    @pytest.mark.parametrize('failing_fn', [
        'get_completion_paths',
        'get_escalation_rates',
        'get_loop_histograms',
        'get_time_centiles',
    ])
    def test_performance_partial_failure(self, client, failing_fn):
        """One failing coroutine should not discard its siblings' data."""
        with _patch_perf_integration(), patch(
            f'dashboard.app.{failing_fn}',
            new_callable=AsyncMock,
            side_effect=RuntimeError('injected error'),
        ):
            resp = client.get('/partials/performance')
            assert resp.status_code == 200
            html = resp.text
            # When paths fails, the template renders empty (all content is paths-keyed)
            # but we still get a 200 — no 500 from one failed gather coroutine.
            if failing_fn == 'get_completion_paths':
                # No 500, template still renders (empty else block since ttc non-empty)
                assert 'Performance' in html
            # When escalations fail, paths data (one-pass) still renders
            if failing_fn == 'get_escalation_rates':
                assert 'one-pass' in html
            # When histograms fail, paths and escalations still render
            if failing_fn == 'get_loop_histograms':
                assert 'one-pass' in html
                assert 'Steward' in html
            # When ttc fails, paths and escalations still render
            if failing_fn == 'get_time_centiles':
                assert 'one-pass' in html
                assert 'Steward' in html


_MG_TIMESERIES = {
    'labels': [f'{h:02d}:00' for h in range(24)],
    'reads': [0] * 24,
    'writes': [0] * 24,
}
_MG_OPERATIONS = {'labels': ['search', 'add_memory'], 'values': [100, 50]}
_MG_AGENTS = {'labels': ['claude-interactive', 'recon-consolidator'], 'values': [80, 70]}


def _patch_memory_graphs_integration(timeseries=_UNSET, operations=_UNSET, agents=_UNSET):
    """Return an ExitStack that patches all 3 memory-graphs data functions."""
    defaults = {
        'timeseries': timeseries if timeseries is not _UNSET else _MG_TIMESERIES,
        'operations': operations if operations is not _UNSET else _MG_OPERATIONS,
        'agents': agents if agents is not _UNSET else _MG_AGENTS,
    }
    stack = ExitStack()
    stack.enter_context(patch(
        'dashboard.app.get_memory_timeseries',
        new_callable=AsyncMock,
        return_value=defaults['timeseries'],
    ))
    stack.enter_context(patch(
        'dashboard.app.get_operations_breakdown',
        new_callable=AsyncMock,
        return_value=defaults['operations'],
    ))
    stack.enter_context(patch(
        'dashboard.app.get_agent_breakdown',
        new_callable=AsyncMock,
        return_value=defaults['agents'],
    ))
    return stack


class TestMemoryGraphsPartialIntegration:
    """Integration tests for GET /partials/memory-graphs."""

    def test_memory_graphs_with_data(self, client):
        with _patch_memory_graphs_integration():
            resp = client.get('/partials/memory-graphs')
            assert resp.status_code == 200
            assert 'text/html' in resp.headers['content-type']
            html = resp.text
            assert 'last 24' in html
            assert 'memoryTimeseriesChart' in html
            assert 'memoryOpsChart' in html
            assert 'memoryAgentChart' in html
            assert 'By operation' in html
            assert 'By agent' in html
            assert 'new Chart' in html

    def test_memory_graphs_empty(self, client):
        with _patch_memory_graphs_integration(
            timeseries={'labels': [], 'reads': [], 'writes': []},
            operations={'labels': [], 'values': []},
            agents={'labels': [], 'values': []},
        ):
            resp = client.get('/partials/memory-graphs')
            assert resp.status_code == 200
            html = resp.text
            assert 'memoryTimeseriesChart' in html
            assert 'memoryOpsChart' in html
            assert 'memoryAgentChart' in html
            # Verify that empty arrays are embedded in the serialized JSON, proving
            # the template received and rendered the empty payload (not just that the
            # canvas elements are unconditionally present regardless of data).
            assert '"reads": []' in html
            assert '"writes": []' in html

    def test_memory_graphs_backend_error_degrades_gracefully(self, client):
        with _patch_memory_graphs_integration(), patch(
            'dashboard.app.get_memory_timeseries',
            new_callable=AsyncMock,
            side_effect=RuntimeError('db unavailable'),
        ):
            resp = client.get('/partials/memory-graphs')
            assert resp.status_code == 200
            assert 'memoryTimeseriesChart' in resp.text

    @pytest.mark.parametrize('failing_fn', [
        'get_memory_timeseries',
        'get_operations_breakdown',
        'get_agent_breakdown',
    ])
    def test_memory_graphs_partial_failure(self, client, failing_fn):
        """One failing coroutine should not discard its siblings' data."""
        with _patch_memory_graphs_integration(), patch(
            f'dashboard.app.{failing_fn}',
            new_callable=AsyncMock,
            side_effect=RuntimeError('injected error'),
        ):
            resp = client.get('/partials/memory-graphs')
            assert resp.status_code == 200
            html = resp.text
            # All chart divs always present
            assert 'memoryTimeseriesChart' in html
            assert 'memoryOpsChart' in html
            assert 'memoryAgentChart' in html
            # When timeseries fails, ops and agents still have data
            if failing_fn == 'get_memory_timeseries':
                assert 'search' in html
                assert 'claude-interactive' in html
            # When ops fails, timeseries and agents still render (non-empty data)
            if failing_fn == 'get_operations_breakdown':
                assert 'claude-interactive' in html
            # When agents fails, timeseries and ops still render
            if failing_fn == 'get_agent_breakdown':
                assert 'search' in html


class TestHtmxErrorHandling:
    """Tests for global HTMX error handler script in base.html."""

    def test_script_has_response_error_listener(self, client):
        html = client.get('/').text
        assert 'htmx:responseError' in html

    def test_script_has_timeout_listener(self, client):
        html = client.get('/').text
        assert 'htmx:timeout' in html

    def test_script_has_send_error_listener(self, client):
        html = client.get('/').text
        assert 'htmx:sendError' in html

    def test_error_card_has_red_bg_class(self, client):
        html = client.get('/').text
        assert 'bg-red-900/30' in html

    def test_error_card_has_red_border_class(self, client):
        html = client.get('/').text
        assert 'border-red-800' in html

    def test_error_card_has_red_text_class(self, client):
        html = client.get('/').text
        assert 'text-red-300' in html

    def test_error_card_has_retrying_message(self, client):
        html = client.get('/').text
        assert 'retrying' in html


class TestHtmxTimeout:
    """Tests that polling sections have correct hx-request timeout attributes.

    Actual poll intervals and timeout values:
    - /partials/memory: poll 10s, timeout 8000ms
    - /partials/recon: poll 15s, timeout 12000ms
    - /partials/orchestrators: poll 10s, timeout 8000ms
    - /partials/performance: poll 30s, timeout 12000ms
    - /partials/memory-graphs: poll 60s, timeout 10000ms
    """

    @pytest.mark.parametrize('partial_url,timeout_ms', SECTION_TIMEOUTS)
    def test_section_has_correct_timeout(self, client, partial_url, timeout_ms):
        html = client.get('/').text
        section_html = _get_section_window(html, partial_url)
        assert (
            f'"timeout": {timeout_ms}' in section_html
            or f'"timeout":{timeout_ms}' in section_html
        ), f'timeout {timeout_ms} not found near {partial_url}'



class TestTailwindBuild:
    """Tests that Tailwind CSS is served locally (no CDN script tag)."""

    def test_no_cdn_tailwind(self, client):
        html = client.get('/').text
        assert 'cdn.tailwindcss.com' not in html

    def test_local_tailwind_css_linked(self, client):
        html = client.get('/').text
        assert 'href="/static/tailwind.css"' in html

    def test_tailwind_css_served(self, client):
        resp = client.get('/static/tailwind.css')
        assert resp.status_code == 200
        assert 'text/css' in resp.headers['content-type']

    def test_tailwind_css_has_utilities(self, client):
        css = client.get('/static/tailwind.css').text
        assert 'bg-gray-900' in css
        assert 'text-gray-100' in css

    def test_tailwind_css_has_template_utilities(self, client):
        css = client.get('/static/tailwind.css').text
        assert 'border-gray-700' in css, 'border-gray-700 missing from tailwind.css'
        assert 'animate-pulse' in css, 'animate-pulse missing from tailwind.css'
        assert 'rounded-lg' in css, 'rounded-lg missing from tailwind.css'

    def test_tailwind_css_minimum_size(self, client):
        resp = client.get('/static/tailwind.css')
        css = resp.text
        if '/* Auto-generated stub for testing' in css:
            pytest.skip('skipped: running against stub, not a real Tailwind build')
        assert len(resp.content) > 50_000, (
            f'tailwind.css is only {len(resp.content)} bytes — build may be incomplete'
        )

    def test_old_style_css_returns_404(self, client):
        resp = client.get('/static/style.css')
        assert resp.status_code == 404, 'style.css was renamed to input.css in task 276; it must not be served'


class TestFavicon:
    """Tests that the favicon SVG is linked and served."""

    def test_favicon_link_in_html(self, client):
        html = client.get('/').text
        assert '/static/favicon.svg' in html
        assert 'rel="icon"' in html

    def test_favicon_svg_served(self, client):
        resp = client.get('/static/favicon.svg')
        assert resp.status_code == 200
        assert 'image/svg+xml' in resp.headers['content-type']

    def test_favicon_svg_body_has_svg_element(self, client):
        body = client.get('/static/favicon.svg').text
        assert '<svg' in body, 'favicon.svg response body does not contain an SVG element'

    def test_favicon_svg_body_has_df_branding(self, client):
        body = client.get('/static/favicon.svg').text
        assert 'DF' in body, 'favicon.svg response body is missing the DF branding text'


class TestMainModule:
    """Tests for python -m dashboard entry point."""

    def test_main_calls_uvicorn_run(self):
        with patch('uvicorn.run') as mock_run:
            runpy.run_module('dashboard', run_name='__main__')
            mock_run.assert_called_once_with(
                'dashboard.app:app',
                host='127.0.0.1',
                port=8080,
                reload=True,
            )

    def test_main_respects_env_var_overrides(self):
        with (
            patch.dict(os.environ, {'DASHBOARD_HOST': '0.0.0.0', 'DASHBOARD_PORT': '9090'}),
            patch('uvicorn.run') as mock_run,
        ):
            runpy.run_module('dashboard', run_name='__main__')
            mock_run.assert_called_once_with(
                'dashboard.app:app',
                host='0.0.0.0',
                port=9090,
                reload=True,
            )


class TestAriaLivePollingsections:
    """Tests that all five auto-polling sections have aria-live='polite'."""

    @pytest.mark.parametrize('partial_url', PARTIAL_URLS)
    def test_section_has_aria_live_polite(self, client, partial_url):
        html = client.get('/').text
        section_html = _get_section_window(html, partial_url)
        assert 'aria-live="polite"' in section_html, (
            f'aria-live="polite" not found near {partial_url}'
        )

    def test_polling_sections_have_aria_live(self, client):
        html = client.get('/').text
        assert html.count('aria-live="polite"') >= 5


class TestNavBar:
    """Tests for the nav bar present in every page via base.html."""

    def test_nav_element_present(self, client):
        html = client.get('/').text
        assert '<nav' in html

    def test_dashboard_link_href(self, client):
        html = client.get('/').text
        assert 'href="/"' in html

    def test_costs_link_href(self, client):
        html = client.get('/').text
        assert 'href="/costs"' in html

    def test_dashboard_link_text(self, client):
        html = client.get('/').text
        assert 'Dashboard' in html

    def test_costs_link_text(self, client):
        html = client.get('/').text
        assert 'Costs' in html

    def test_costs_route_returns_200(self, client):
        response = client.get('/costs')
        assert response.status_code == 200

    def test_costs_page_extends_base(self, client):
        html = client.get('/costs').text
        assert '<nav' in html
        assert 'href="/"' in html

    # Active-state tests — use _get_nav_link for scoped assertions

    def test_dashboard_link_active_on_root(self, client):
        html = client.get('/').text
        tag = _get_nav_link(html, '/')
        assert 'border-blue-500' in tag

    def test_costs_link_not_active_on_root(self, client):
        html = client.get('/').text
        tag = _get_nav_link(html, '/costs')
        assert 'border-blue-500' not in tag

    def test_costs_link_active_on_costs_page(self, client):
        html = client.get('/costs').text
        tag = _get_nav_link(html, '/costs')
        assert 'border-blue-500' in tag

    def test_dashboard_link_not_active_on_costs_page(self, client):
        html = client.get('/costs').text
        tag = _get_nav_link(html, '/')
        assert 'border-blue-500' not in tag

    def test_active_link_has_white_text(self, client):
        html = client.get('/').text
        tag = _get_nav_link(html, '/')
        assert 'text-white' in tag

    def test_inactive_link_has_gray_text(self, client):
        html = client.get('/').text
        tag = _get_nav_link(html, '/costs')
        assert 'text-gray-400' in tag


class TestSafeGatherResult:
    """Tests for the _safe_gather_result helper."""

    def test_safe_gather_result_reraises_cancelled_error(self):
        """CancelledError (BaseException, not Exception) must propagate, not be swallowed."""
        with pytest.raises(asyncio.CancelledError):
            _safe_gather_result(asyncio.CancelledError(), 'default', 'test')

    def test_safe_gather_result_reraises_keyboard_interrupt(self):
        """KeyboardInterrupt (BaseException, not Exception) must propagate, not be swallowed."""
        with pytest.raises(KeyboardInterrupt):
            _safe_gather_result(KeyboardInterrupt(), 'default', 'test')
