"""Tests for the orchestrator panel route and template rendering."""

from __future__ import annotations

from unittest.mock import patch

# --- Mock data for route tests ---

MOCK_ORCHESTRATOR_RUNNING = {
    'pid': 1234,
    'prd': '/home/leo/src/dark-factory/prd/dashboard.md',
    'running': True,
    'started': 'Mar18',
    'tasks': [
        {'id': 1, 'title': 'Setup infra', 'status': 'done', 'priority': 'high', 'dependencies': [], 'metadata': {}},
        {'id': 2, 'title': 'Build auth', 'status': 'done', 'priority': 'high', 'dependencies': [1], 'metadata': {}},
        {'id': 3, 'title': 'Build API', 'status': 'in-progress', 'priority': 'medium', 'dependencies': [2], 'metadata': {}},
        {'id': 4, 'title': 'Review PR', 'status': 'blocked', 'priority': 'medium', 'dependencies': [3], 'metadata': {}},
        {'id': 5, 'title': 'Deploy', 'status': 'pending', 'priority': 'low', 'dependencies': [4], 'metadata': {}},
    ],
    'worktrees': {
        '1': {
            'phase': 'DONE',
            'plan_progress': {'done': 3, 'total': 3},
            'iteration_count': 2,
            'review_summary': '2/2 passed',
            'modules': ['infra/'],
        },
        '3': {
            'phase': 'EXECUTE',
            'plan_progress': {'done': 1, 'total': 4},
            'iteration_count': 5,
            'review_summary': '3/5 passed',
            'modules': ['auth/', 'api/'],
        },
    },
    'summary': {
        'total': 5,
        'done': 2,
        'in_progress': 1,
        'blocked': 1,
        'pending': 1,
    },
}


def _patch_orchestrator_data(return_value):
    """Patch discover_orchestrators at its app.py import location."""
    return patch('dashboard.app.discover_orchestrators', return_value=return_value)


class TestOrchestratorRouteBasics:
    """Tests for GET /partials/orchestrators with populated data."""

    def test_returns_200(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            resp = client.get('/partials/orchestrators')
        assert resp.status_code == 200

    def test_content_type_html(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            resp = client.get('/partials/orchestrators')
        assert 'text/html' in resp.headers['content-type']

    def test_header_title(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        assert 'Orchestrators' in html

    def test_count_badge(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        # Count badge should show "1" for single orchestrator
        assert '>1<' in html

    def test_card_shows_pid(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        assert '1234' in html

    def test_running_badge(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        assert 'bg-green-600' in html
        assert 'running' in html

    def test_prd_filename(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        assert 'dashboard.md' in html

    def test_progress_bar_container(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        assert 'h-3' in html
        assert 'rounded-full' in html
        assert 'bg-gray-700' in html
        assert 'overflow-hidden' in html

    def test_progress_bar_segments(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        # All four segment colors should be present (done=green, in_progress=blue, blocked=red, pending=yellow)
        assert 'bg-green-600' in html
        assert 'bg-blue-600' in html
        assert 'bg-yellow-600' in html
        assert 'bg-red-600' in html

    def test_stats_done(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        assert '2 done' in html

    def test_stats_in_progress(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        assert '1 in-progress' in html

    def test_stats_blocked(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        assert '1 blocked' in html

    def test_stats_pending(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        assert '1 pending' in html

    def test_alpine_toggle(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        assert 'x-data' in html
        assert 'x-show' in html
        assert '@click' in html

    def test_show_tasks_button(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        assert 'Show tasks' in html


class TestOrchestratorRouteEmpty:
    """Tests for GET /partials/orchestrators with no orchestrators."""

    def test_returns_200(self, client):
        with _patch_orchestrator_data([]):
            resp = client.get('/partials/orchestrators')
        assert resp.status_code == 200

    def test_no_orchestrators_message(self, client):
        with _patch_orchestrator_data([]):
            html = client.get('/partials/orchestrators').text
        assert 'No orchestrators detected' in html

    def test_no_progress_elements(self, client):
        with _patch_orchestrator_data([]):
            html = client.get('/partials/orchestrators').text
        assert 'bg-green-600' not in html
