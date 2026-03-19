"""Integration tests for the full dashboard application."""

from __future__ import annotations


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
