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
            'graphiti': {'connected': True, 'node_count': 42},
            'mem0': {'connected': True, 'memory_count': 128},
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


def _patch_recon_data(
    buffer_stats=None,
    burst_state=None,
    watermarks=None,
    verdict=None,
    runs=None,
):
    """Return an ExitStack that patches all 5 recon data functions."""
    defaults = {
        'buffer_stats': buffer_stats if buffer_stats is not None else {
            'buffered_count': 3, 'oldest_event_age_seconds': 600.0,
        },
        'burst_state': burst_state if burst_state is not None else [
            {
                'agent_id': 'agent-1',
                'state': 'bursting',
                'last_write_at': '2026-03-19T00:00:00+00:00',
                'burst_started_at': '2026-03-19T00:00:00+00:00',
            },
        ],
        'watermarks': watermarks if watermarks is not None else {
            'last_full_run_completed': '2026-03-19T10:00:00+00:00',
        },
        'verdict': verdict,
        'runs': runs if runs is not None else [
            {
                'id': 'run-001',
                'run_type': 'full',
                'trigger_reason': 'staleness_timer',
                'started_at': '2026-03-19T08:00:00+00:00',
                'completed_at': '2026-03-19T08:05:00+00:00',
                'events_processed': 7,
                'status': 'completed',
                'duration_seconds': 300.0,
            },
        ],
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
            watermarks={},
            runs=[],
        ):
            resp = client.get('/partials/recon')
            assert resp.status_code == 200
            html = resp.text
            assert 'No reconciliation runs' in html


_MOCK_ORCHESTRATOR = {
    'pid': 1234,
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


class TestSriIntegrity:
    """Tests that CDN script tags have SRI integrity hashes and crossorigin attributes."""

    def test_htmx_has_integrity(self, client):
        html = client.get('/').text
        htmx_idx = html.index('unpkg.com/htmx.org@2.0.4')
        tag_html = html[htmx_idx - 50:htmx_idx + 300]
        assert 'integrity="sha384-' in tag_html

    def test_idiomorph_has_integrity(self, client):
        html = client.get('/').text
        idio_idx = html.index('unpkg.com/idiomorph@0.3.0')
        tag_html = html[idio_idx - 50:idio_idx + 300]
        assert 'integrity="sha384-' in tag_html

    def test_alpine_has_integrity(self, client):
        html = client.get('/').text
        alpine_idx = html.index('alpinejs@3.14.8')
        tag_html = html[alpine_idx - 50:alpine_idx + 300]
        assert 'integrity="sha384-' in tag_html

    def test_all_cdn_scripts_have_crossorigin(self, client):
        html = client.get('/').text
        for url_fragment in [
            'unpkg.com/htmx.org@2.0.4',
            'unpkg.com/idiomorph@0.3.0',
            'alpinejs@3.14.8',
        ]:
            idx = html.index(url_fragment)
            tag_html = html[idx - 50:idx + 300]
            assert 'crossorigin="anonymous"' in tag_html, (
                f'Missing crossorigin=anonymous near {url_fragment}'
            )

    def test_no_cdn_script_without_integrity(self, client):
        import re
        html = client.get('/').text
        # Find all external script src tags (https://)
        script_tags = re.findall(r'<script[^>]+src="https://[^"]*"[^>]*>', html)
        # Filter out Tailwind CDN (dynamic JIT, not a static asset)
        checkable = [t for t in script_tags if 'tailwind' not in t]
        for tag in checkable:
            assert 'integrity=' in tag, f'Missing integrity on tag: {tag}'


class TestCdnVersionPinning:
    """Tests that Alpine.js CDN URL uses a pinned version, not a wildcard."""

    def test_alpine_no_wildcard_version(self, client):
        html = client.get('/').text
        assert '@3.x.x' not in html

    def test_alpine_pinned_to_3_14_8(self, client):
        html = client.get('/').text
        assert 'alpinejs@3.14.8' in html


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
