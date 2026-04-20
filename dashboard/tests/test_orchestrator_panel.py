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
        1: {
            'phase': 'DONE',
            'plan_progress': {'done': 3, 'total': 3},
            'iteration_count': 2,
            'review_summary': '2/2 passed',
            'modules': ['infra/'],
        },
        3: {
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
        """Card uses x-data, x-show on rows, x-model checkboxes, $store.panels."""
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        assert 'x-data' in html
        assert 'x-show' in html
        assert '$store.panels' in html

    def test_table_wrapper_has_x_cloak(self, client):
        """x-cloak appears on task rows so they are hidden before Alpine mounts."""
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        assert 'x-cloak' in html

    def test_table_wrapper_x_show_uses_store(self, client):
        """x-show on rows still references $store.panels[key] for persistence."""
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        # x-show is now on rows rather than the wrapper, but the store reference is the same
        assert "x-show=\"$store.panels[" in html

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


class TestOrchestratorFilterCheckboxes:
    """Tests for the three orthogonal filter checkboxes (Active / Pending / Done).

    The three checkboxes are independent: Active shows in-progress+blocked rows,
    Pending shows pending rows, Done shows done/deferred/cancelled rows.
    Default state: Active=on, Pending=off, Done=off (only active tasks visible).

    Note on default-visibility e2e testing: verifying that exactly 2 rows are
    *visually* shown on first load requires Alpine.js to initialise and apply
    x-cloak.  That is a browser/JS-level concern outside the scope of the
    current server-render test suite.  The x-init default (active: true) is
    asserted below; end-to-end row-count verification is deferred to a future
    Playwright smoke test.
    """

    def test_three_checkbox_inputs_rendered(self, client):
        """Three <input type="checkbox"> elements present inside the orchestrator card."""
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        count = html.count('type="checkbox"')
        assert count >= 3, f'Expected at least 3 checkboxes, found {count}'

    def test_done_label_with_count(self, client):
        """'Done/other' checkbox label shows count of done/other tasks (total minus active/pending)."""
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        # total=5, in_progress=1, blocked=1, pending=1 → done/other count = 5-1-1-1 = 2
        assert 'Done/other (2)' in html

    def test_active_label_with_count(self, client):
        """Label for 'Active' checkbox shows in-progress + blocked count."""
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        assert 'Active (2)' in html

    def test_pending_label_with_count(self, client):
        """Label for 'Pending' checkbox shows pending count."""
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        assert 'Pending (1)' in html

    def test_x_init_default_state(self, client):
        """x-init seeds active:true so in-progress/blocked rows start visible by default."""
        import re
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        # Use regex to tolerate whitespace variation around the colon (e.g. 'active:true' vs 'active: true')
        assert re.search(r'active\s*:\s*true', html), (
            "x-init does not contain active=true default (regex: r'active\\s*:\\s*true')"
        )


class TestOrchestratorTaskRowFiltering:
    """Tests for per-row x-show filtering on task table rows."""

    ORCH_KEY = '-home-leo-src-dark-factory-prd-dashboard-md'

    def test_all_tbody_rows_have_x_show(self, client):
        """Every data <tr> in <tbody> carries an x-show attribute."""
        import re
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        # Find tbody content
        tbody_match = re.search(r'<tbody>(.*?)</tbody>', html, re.DOTALL)
        assert tbody_match is not None, 'No <tbody> found'
        tbody = tbody_match.group(1)
        # Count <tr> tags and x-show occurrences in tbody
        tr_count = tbody.count('<tr ')
        xshow_count = tbody.count('x-show=')
        assert tr_count > 0, 'No <tr> rows found in tbody'
        assert xshow_count == tr_count, (
            f'Expected {tr_count} x-show attributes but found {xshow_count}'
        )

    def test_total_row_count_equals_task_count(self, client):
        """All 5 tasks are server-rendered; client-side x-show controls visibility."""
        import re
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        tbody_match = re.search(r'<tbody>(.*?)</tbody>', html, re.DOTALL)
        assert tbody_match is not None
        tbody = tbody_match.group(1)
        tr_count = tbody.count('<tr ')
        assert tr_count == 5, f'Expected 5 rows, found {tr_count}'

    def test_active_rows_reference_dot_active(self, client):
        """Rows for in-progress/blocked tasks reference .active in x-show."""
        import re
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        # Find x-show attributes referencing .active (for active bucket rows)
        active_rows = re.findall(r'x-show="([^"]*\.active[^"]*)"', html)
        assert len(active_rows) >= 2, (
            f'Expected at least 2 rows with .active in x-show, found {len(active_rows)}'
        )

    def test_active_rows_gate_on_active_only(self, client):
        """Active-bucket rows gate solely on .active (no .all dependency)."""
        import re
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        active_rows = re.findall(r'x-show="([^"]*\.active[^"]*)"', html)
        assert len(active_rows) >= 2
        for expr in active_rows:
            assert '.done' not in expr, (
                f'Active row x-show "{expr}" unexpectedly references .done'
            )

    def test_pending_row_references_dot_pending(self, client):
        """Row for pending task references .pending in x-show."""
        import re
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        pending_rows = re.findall(r'x-show="([^"]*\.pending[^"]*)"', html)
        assert len(pending_rows) >= 1, (
            f'Expected at least 1 row with .pending in x-show, found {len(pending_rows)}'
        )

    def test_pending_rows_gate_on_pending_only(self, client):
        """Pending-bucket rows gate solely on .pending (no .all or .done dependency)."""
        import re
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        pending_rows = re.findall(r'x-show="([^"]*\.pending[^"]*)"', html)
        assert len(pending_rows) >= 1
        for expr in pending_rows:
            assert '.done' not in expr, (
                f'Pending row x-show "{expr}" unexpectedly references .done'
            )

    def test_done_rows_gate_on_done_flag(self, client):
        """Done/other-bucket rows gate solely on .done (NOT .active or .pending)."""
        import re
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        tbody_match = re.search(r'<tbody>(.*?)</tbody>', html, re.DOTALL)
        assert tbody_match is not None
        tbody = tbody_match.group(1)
        all_xshow = re.findall(r'x-show="([^"]+)"', tbody)
        # "other" rows: reference .done but NOT .active AND NOT .pending
        done_rows = [
            expr for expr in all_xshow
            if '.done' in expr and '.active' not in expr and '.pending' not in expr
        ]
        # We have 2 done tasks (ids 1 and 2)
        assert len(done_rows) >= 2, (
            f'Expected at least 2 done/other rows gated only on .done, found {len(done_rows)}: {done_rows}'
        )

    def test_rows_have_x_cloak(self, client):
        """Task rows carry x-cloak to avoid flash before Alpine mounts."""
        import re
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        tbody_match = re.search(r'<tbody>(.*?)</tbody>', html, re.DOTALL)
        assert tbody_match is not None
        tbody = tbody_match.group(1)
        cloak_count = tbody.count('x-cloak')
        assert cloak_count >= 5, (
            f'Expected at least 5 x-cloak attributes on rows, found {cloak_count}'
        )

    def test_row_xshow_uses_optional_chaining(self, client):
        """Per-row x-show expressions use optional chaining (?.) on $store.panels[key].

        Guards against Alpine evaluating a child x-show before the parent x-init
        has populated the panel object (e.g. during HTMX morph), causing a crash
        instead of a transient hide.
        """
        import re
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text
        tbody_match = re.search(r'<tbody>(.*?)</tbody>', html, re.DOTALL)
        assert tbody_match is not None, 'No <tbody> found'
        tbody = tbody_match.group(1)
        exprs = re.findall(r'x-show="([^"]+)"', tbody)
        assert len(exprs) > 0, 'No x-show expressions found in tbody'
        for expr in exprs:
            assert '?.' in expr, (
                f'Per-row x-show expression does not use optional chaining: {expr!r}'
            )


class TestOrchestratorKeyConsistency:
    """Regression: x-init, checkboxes, and row x-show all reference the same orch_key."""

    def test_all_store_references_use_same_key(self, client):
        """Parse orch_key from the card id attribute, then verify x-init, all checkbox
        bindings, and all row x-show expressions reference exactly that key.

        Guards against future drift where orch_key is renamed in one template but
        not the other (the role previously played by test_store_key_consistent_between_toggle_and_show,
        now generalized to cover the full filter surface).
        """
        import re
        with _patch_orchestrator_data([MOCK_ORCHESTRATOR_RUNNING]):
            html = client.get('/partials/orchestrators').text

        # Derive expected orch_key from the card's id attribute: id="orch-<key>"
        id_match = re.search(r'id="orch-([^"]+)"', html)
        assert id_match is not None, 'Card id="orch-..." not found in HTML'
        orch_key = id_match.group(1)

        # x-init must reference the same key
        assert f"$store.panels['{orch_key}']" in html, (
            f"x-init does not reference key '{orch_key}'"
        )

        # All three x-model bindings must use the same key
        for attr in ['active', 'pending', 'done']:
            assert f"$store.panels['{orch_key}'].{attr}" in html, (
                f"No {attr} binding found for key '{orch_key}'"
            )

        # All x-show expressions on rows must also use the same key
        row_xshow_values = re.findall(r'x-show="([^"]+)"', html)
        assert len(row_xshow_values) > 0, 'No x-show attributes found'
        for expr in row_xshow_values:
            assert orch_key in expr, (
                f"x-show expression '{expr}' does not reference orch_key '{orch_key}'"
            )

