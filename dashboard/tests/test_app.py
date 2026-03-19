"""Integration tests for the full dashboard application."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch


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
