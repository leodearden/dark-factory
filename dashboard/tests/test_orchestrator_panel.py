"""Tests for the orchestrator panel route and template rendering."""

from __future__ import annotations

from unittest.mock import patch

# --- Mock data for route tests ---

MOCK_ORCHESTRATOR_RUNNING = {
    'pids': [1234],
    'prd': '/home/leo/src/dark-factory/prd/dashboard.md',
    'label': '/home/leo/src/dark-factory/prd/dashboard.md',
    'project_root': '/home/leo/src/dark-factory',
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
        assert '$store.panels' in html

    def test_show_tasks_button(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        assert 'Show tasks' in html

    def test_table_wrapper_hidden_by_default(self, client):
        """Task table wrapper must have style='display:none' so it is hidden
        even before Alpine processes the element after an HTMX morph."""
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        assert 'style="display:none"' in html

    def test_table_wrapper_has_x_cloak(self, client):
        """Wrapper retains x-cloak for initial page load (non-HTMX) scenarios."""
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        assert 'x-cloak' in html

    def test_table_wrapper_x_show_uses_store(self, client):
        """x-show references $store.panels[key] so state persists across morphs."""
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        # The x-show must use $store.panels, not local x-data state
        assert "x-show=\"$store.panels[" in html

    def test_button_x_text_uses_store(self, client):
        """Button x-text expression uses $store.panels[key] for label toggling."""
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        assert "x-text=\"$store.panels[" in html

    def test_task_table_wrapper_has_x_cloak(self, client):
        """x-show div must have x-cloak so it is hidden before Alpine initializes
        after an innerHTML swap (MutationObserver detects new x-data elements)."""
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        assert 'x-cloak' in html

    def test_store_key_consistent_between_toggle_and_show(self, client):
        """The $store.panels key in @click toggle must match the x-show key.

        Extracts both keys from the rendered HTML and asserts they are equal.
        Guards against key drift between the button and the table wrapper.
        """
        import re
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        # Extract key from @click: $store.panels['<key>'] = !$store.panels['<key>']
        click_match = re.search(r"\$store\.panels\['([^']+)'\]\s*=\s*!", html)
        # Extract key from x-show: x-show="$store.panels['<key>']"
        show_match = re.search(r'x-show="\$store\.panels\[\'([^\']+)\'\]"', html)
        assert click_match is not None, '@click $store.panels key not found'
        assert show_match is not None, 'x-show $store.panels key not found'
        assert click_match.group(1) == show_match.group(1), (
            f'Key mismatch: @click uses {click_match.group(1)!r} '
            f'but x-show uses {show_match.group(1)!r}'
        )

    def test_card_shows_single_pid_label(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        assert 'PID 1234' in html


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


class TestTaskTable:
    """Tests for the task detail table rendered inside orchestrator cards."""

    def test_table_headers(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        for header in ('ID', 'Title', 'Status', 'Phase', 'Iterations', 'Reviews', 'Modules'):
            assert header in html

    def test_table_classes(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        assert 'w-full' in html
        assert 'text-sm' in html
        assert 'text-left' in html
        assert 'text-gray-300' in html

    def test_task_status_done_badge(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        # done tasks produce green badges (bg-green-600 already checked elsewhere,
        # but verify 'done' text appears in table context)
        assert 'done' in html

    def test_task_status_blocked_badge(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        assert 'bg-red-600' in html

    def test_phase_exec_label(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        # Task 3 has EXECUTE phase worktree, displayed as EXEC
        assert 'EXEC' in html

    def test_phase_done_label(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        # Task 1 has DONE phase worktree
        assert 'DONE' in html

    def test_iteration_count(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        # Task 3 worktree has iteration_count=5
        assert '5' in html

    def test_review_summary(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        # Task 3 worktree has review_summary='3/5 passed'
        assert '3/5 passed' in html

    def test_modules_displayed(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        # Task 3 worktree has modules=['auth/', 'api/']
        assert 'auth/' in html
        assert 'api/' in html

    def test_task_without_worktree_shows_dash(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        # Tasks 2, 4, 5 have no worktree entries — should show em-dash
        assert '\u2014' in html or '&mdash;' in html


MOCK_ORCHESTRATOR_COMPLETED = {
    'pids': [5678],
    'prd': '/other/memory.md',
    'label': '/other/memory.md',
    'project_root': '/other',
    'running': False,
    'started': 'Mar17',
    'tasks': [],
    'worktrees': {},
    'summary': {
        'total': 0,
        'done': 0,
        'in_progress': 0,
        'blocked': 0,
        'pending': 0,
    },
}


class TestOrchestratorRouteMultiple:
    """Tests for GET /partials/orchestrators with multiple orchestrators."""

    def test_count_badge_shows_two(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING, MOCK_ORCHESTRATOR_COMPLETED]):
            html = client.get('/partials/orchestrators').text
        assert '>2<' in html

    def test_both_pids(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING, MOCK_ORCHESTRATOR_COMPLETED]):
            html = client.get('/partials/orchestrators').text
        assert '1234' in html
        assert '5678' in html

    def test_completed_badge(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING, MOCK_ORCHESTRATOR_COMPLETED]):
            html = client.get('/partials/orchestrators').text
        assert 'completed' in html

    def test_zero_total_no_error(self, client):
        """Orchestrator with zero tasks doesn't cause division-by-zero."""
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING, MOCK_ORCHESTRATOR_COMPLETED]):
            resp = client.get('/partials/orchestrators')
        assert resp.status_code == 200

    def test_prd_filenames(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING, MOCK_ORCHESTRATOR_COMPLETED]):
            html = client.get('/partials/orchestrators').text
        assert 'dashboard.md' in html
        assert 'memory.md' in html


class TestOrchestratorBadgeAriaLabels:
    """Tests for ARIA labels on orchestrator running/completed status badges."""

    def test_running_badge_aria_label(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        assert 'aria-label="Orchestrator status: running"' in html

    def test_completed_badge_aria_label(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_COMPLETED]):
            html = client.get('/partials/orchestrators').text
        assert 'aria-label="Orchestrator status: completed"' in html

    def test_both_badges_aria_labels_when_multiple(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING, MOCK_ORCHESTRATOR_COMPLETED]):
            html = client.get('/partials/orchestrators').text
        assert 'aria-label="Orchestrator status: running"' in html
        assert 'aria-label="Orchestrator status: completed"' in html

    def test_progress_bar_aria_label(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        assert 'aria-label="Task progress' in html


MOCK_ORCHESTRATOR_MULTI_PID = {
    'pids': [1234, 5678],
    'prd': '/home/leo/src/dark-factory/prd/dashboard.md',
    'label': '/home/leo/src/dark-factory/prd/dashboard.md',
    'project_root': '/home/leo/src/dark-factory',
    'running': True,
    'started': 'Mar18',
    'tasks': [],
    'worktrees': {},
    'summary': {
        'total': 0,
        'done': 0,
        'in_progress': 0,
        'blocked': 0,
        'pending': 0,
    },
}


class TestOrchestratorMultiPid:
    """Tests for orchestrator cards with multiple PIDs sharing a single PRD."""

    def test_card_shows_multi_pid_label(self, client):
        """Multi-PID entry renders 'PIDs' (plural) with both PIDs present."""
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_MULTI_PID]):
            html = client.get('/partials/orchestrators').text
        assert 'PIDs' in html
        assert '1234' in html
        assert '5678' in html

    def test_count_badge_shows_one_for_grouped(self, client):
        """Count badge shows '1' for a single grouped entry with multiple PIDs."""
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_MULTI_PID]):
            html = client.get('/partials/orchestrators').text
        assert '>1<' in html

    def test_multi_pid_comma_separated(self, client):
        """Three PIDs render as comma-separated list '111, 222, 333'."""
        mock_entry = {
            **MOCK_ORCHESTRATOR_MULTI_PID,
            'pids': [111, 222, 333],
        }
        with _patch_orchestrator_data([mock_entry]):
            html = client.get('/partials/orchestrators').text
        assert '111' in html
        assert '222' in html
        assert '333' in html
        assert '111, 222, 333' in html


MOCK_ORCHESTRATOR_NO_PRD = {
    'pids': [9999],
    'prd': None,
    'label': '/home/leo/src/reify',
    'project_root': '/home/leo/src/reify',
    'running': True,
    'started': 'Mar30',
    'tasks': [
        {'id': 1, 'title': 'Init project', 'status': 'done', 'priority': 'high', 'dependencies': [], 'metadata': {}},
    ],
    'worktrees': {},
    'summary': {
        'total': 1,
        'done': 1,
        'in_progress': 0,
        'blocked': 0,
        'pending': 0,
    },
}


class TestOrchestratorNoPrd:
    """Tests for orchestrator cards where prd is None (config-only or bare run)."""

    def test_returns_200(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_NO_PRD]):
            resp = client.get('/partials/orchestrators')
        assert resp.status_code == 200

    def test_label_shows_project_dir(self, client):
        """When prd is None, label falls back to project root — last segment displayed."""
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_NO_PRD]):
            html = client.get('/partials/orchestrators').text
        assert 'reify' in html

    def test_count_badge(self, client):
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_NO_PRD]):
            html = client.get('/partials/orchestrators').text
        assert '>1<' in html

    def test_alpine_key_no_error(self, client):
        """Alpine.js store key derived from label works without errors."""
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_NO_PRD]):
            html = client.get('/partials/orchestrators').text
        assert '$store.panels' in html
