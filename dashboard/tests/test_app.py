"""Integration tests for the full dashboard application."""

from __future__ import annotations

import os
import runpy
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch


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

    def test_memory_section_uses_morph(self, client):
        html = client.get('/').text
        assert 'hx-swap="morph:innerHTML"' in html
        assert 'hx-get="/partials/memory"' in html

    def test_recon_section_uses_morph(self, client):
        html = client.get('/').text
        assert 'hx-swap="morph:innerHTML"' in html
        assert 'hx-get="/partials/recon"' in html

    def test_orchestrators_section_uses_morph(self, client):
        html = client.get('/').text
        assert 'hx-swap="morph:innerHTML"' in html
        assert 'hx-get="/partials/orchestrators"' in html

    def test_performance_section_uses_morph(self, client):
        html = client.get('/').text
        assert 'hx-swap="morph:innerHTML"' in html
        assert 'hx-get="/partials/performance"' in html

    def test_memory_graphs_section_uses_morph(self, client):
        html = client.get('/').text
        assert 'hx-swap="morph:innerHTML"' in html
        assert 'hx-get="/partials/memory-graphs"' in html

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

    def test_get_root_htmx_memory(self, client):
        html = client.get('/').text
        assert 'hx-get="/partials/memory"' in html

    def test_get_root_htmx_recon(self, client):
        html = client.get('/').text
        assert 'hx-get="/partials/recon"' in html

    def test_get_root_htmx_orchestrators(self, client):
        html = client.get('/').text
        assert 'hx-get="/partials/orchestrators"' in html

    def test_get_root_htmx_performance(self, client):
        html = client.get('/').text
        assert 'hx-get="/partials/performance"' in html

    def test_get_root_htmx_memory_graphs(self, client):
        html = client.get('/').text
        assert 'hx-get="/partials/memory-graphs"' in html

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
    """Tests that polling sections have correct hx-request timeout attributes."""

    def test_memory_section_has_timeout_8000(self, client):
        html = client.get('/').text
        # memory section uses 10s poll interval → 8s timeout
        memory_idx = html.index('hx-get="/partials/memory"')
        section_html = html[memory_idx - 200:memory_idx + 500]
        assert '"timeout": 8000' in section_html or '"timeout":8000' in section_html

    def test_recon_section_has_timeout_12000(self, client):
        html = client.get('/').text
        # recon section uses 15s poll interval → 12s timeout
        recon_idx = html.index('hx-get="/partials/recon"')
        section_html = html[recon_idx - 200:recon_idx + 500]
        assert '"timeout": 12000' in section_html or '"timeout":12000' in section_html

    def test_orchestrators_section_has_timeout_8000(self, client):
        html = client.get('/').text
        # orchestrators section uses 10s poll interval → 8s timeout
        orch_idx = html.index('hx-get="/partials/orchestrators"')
        section_html = html[orch_idx - 200:orch_idx + 500]
        assert '"timeout": 8000' in section_html or '"timeout":8000' in section_html

    def test_performance_section_has_timeout_12000(self, client):
        html = client.get('/').text
        # performance section uses 30s poll interval → 12s timeout
        perf_idx = html.index('hx-get="/partials/performance"')
        section_html = html[perf_idx - 200:perf_idx + 500]
        assert '"timeout": 12000' in section_html or '"timeout":12000' in section_html

    def test_memory_graphs_section_has_timeout_10000(self, client):
        html = client.get('/').text
        # memory-graphs section uses 60s poll interval → 10s timeout
        mg_idx = html.index('hx-get="/partials/memory-graphs"')
        section_html = html[mg_idx - 200:mg_idx + 500]
        assert '"timeout": 10000' in section_html or '"timeout":10000' in section_html


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

    def test_memory_section_has_aria_live_polite(self, client):
        html = client.get('/').text
        memory_idx = html.index('hx-get="/partials/memory"')
        section_html = html[memory_idx - 100:memory_idx + 300]
        assert 'aria-live="polite"' in section_html

    def test_recon_section_has_aria_live_polite(self, client):
        html = client.get('/').text
        recon_idx = html.index('hx-get="/partials/recon"')
        section_html = html[recon_idx - 100:recon_idx + 300]
        assert 'aria-live="polite"' in section_html

    def test_orchestrators_section_has_aria_live_polite(self, client):
        html = client.get('/').text
        orch_idx = html.index('hx-get="/partials/orchestrators"')
        section_html = html[orch_idx - 100:orch_idx + 300]
        assert 'aria-live="polite"' in section_html

    def test_performance_section_has_aria_live_polite(self, client):
        html = client.get('/').text
        perf_idx = html.index('hx-get="/partials/performance"')
        section_html = html[perf_idx - 100:perf_idx + 300]
        assert 'aria-live="polite"' in section_html

    def test_memory_graphs_section_has_aria_live_polite(self, client):
        html = client.get('/').text
        mg_idx = html.index('hx-get="/partials/memory-graphs"')
        section_html = html[mg_idx - 100:mg_idx + 300]
        assert 'aria-live="polite"' in section_html

    def test_polling_sections_have_aria_live(self, client):
        html = client.get('/').text
        assert html.count('aria-live="polite"') == 5
