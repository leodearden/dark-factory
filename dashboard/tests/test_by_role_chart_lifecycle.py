"""Tests for DOM scoping, render guard, and chart lifecycle in the by_role partial."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

# ---------------------------------------------------------------------------
# Shared mock data and patch helper
# ---------------------------------------------------------------------------

_MOCK_BY_ROLE = {
    'dark_factory': {
        'implementer': {'claude-sonnet': 5.20, 'claude-opus': 1.50},
        'reviewer': {'claude-opus': 3.10},
    },
    'other_project': {
        'implementer': {'claude-haiku': 0.80},
    },
}


def _patch_by_role(return_value=_MOCK_BY_ROLE):
    return patch(
        'dashboard.app.get_cost_by_role',
        new_callable=AsyncMock,
        return_value=return_value,
    )


# ---------------------------------------------------------------------------
# Step-1: DOM scoping tests (TestByRoleDomScoping)
# ---------------------------------------------------------------------------


class TestByRoleDomScoping:
    """Tests that by_role.html scopes all DOM queries to the container div."""

    def test_outer_div_has_container_attribute(self, client):
        """Outer wrapper div must carry data-by-role-container attribute."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        assert 'data-by-role-container' in html

    def test_script_captures_container_via_current_script(self, client):
        """Script must capture container via document.currentScript.closest()."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        assert 'document.currentScript.closest' in html

    def test_canvas_queries_scoped_to_container(self, client):
        """Canvas look-ups must use container.querySelectorAll, not document.querySelectorAll."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        assert 'container.querySelectorAll' in html
        assert 'document.querySelectorAll' not in html

    def test_script_inside_container_div(self, client):
        """The <script> tag must appear *after* data-by-role-container in the HTML."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        container_pos = html.find('data-by-role-container')
        script_pos = html.find('<script')
        assert container_pos != -1, 'data-by-role-container not found in HTML'
        assert script_pos != -1, '<script> tag not found in HTML'
        assert container_pos < script_pos, (
            'data-by-role-container must appear before the <script> tag'
        )


# ---------------------------------------------------------------------------
# Step-3: Render guard tests (TestByRoleRenderGuard)
# ---------------------------------------------------------------------------


class TestByRoleSingleRenderPath:
    """Tests that by_role.html uses a single render path (no dual htmx listener + immediate call)."""

    def test_no_htmx_after_settle_listener(self, client):
        """htmx:afterSettle listener must NOT be registered (single code path)."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        assert 'htmx:afterSettle' not in html

    def test_direct_render_call(self, client):
        """renderCharts must be called directly (single code path)."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        # renderCharts() call as a statement (not inside addEventListener)
        assert 'renderCharts();' in html

    def test_uses_shared_chart_palette(self, client):
        """Must reference the shared CHART_PALETTE instead of a local palette array."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        assert 'CHART_PALETTE' in html
        assert 'var palette' not in html

    def test_no_rendered_guard_needed(self, client):
        """With a single render path, no 'rendered' boolean guard is needed."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        assert 'var rendered' not in html


# ---------------------------------------------------------------------------
# Step-5: Orphaned chart cleanup tests (TestByRoleChartCleanup)
# ---------------------------------------------------------------------------


class TestByRoleChartCleanup:
    """Tests that by_role.html tracks and destroys chart instances to prevent orphans."""

    def test_chart_instances_array_declared(self, client):
        """Script must declare a chartInstances tracking array."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        assert 'var chartInstances = []' in html

    def test_cleanup_charts_function_present(self, client):
        """Script must define a cleanupCharts function."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        assert 'function cleanupCharts' in html

    def test_cleanup_charts_called_in_render_charts(self, client):
        """cleanupCharts() must be called at the start of renderCharts."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        # cleanupCharts() call must appear before 'new Chart(' in renderCharts body
        cleanup_pos = html.find('cleanupCharts()')
        new_chart_pos = html.find('new Chart(')
        assert cleanup_pos != -1, 'cleanupCharts() call not found'
        assert new_chart_pos != -1, 'new Chart( not found'
        assert cleanup_pos < new_chart_pos, (
            'cleanupCharts() must be called before new Chart() in renderCharts'
        )

    def test_chart_instances_pushed_after_creation(self, client):
        """New Chart instances must be pushed into chartInstances array."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        assert 'chartInstances.push(' in html

    def test_get_or_destroy_chart_function_removed(self, client):
        """The old getOrDestroyChart standalone function must no longer appear."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        assert 'function getOrDestroyChart' not in html


# ---------------------------------------------------------------------------
# Step-7: document.currentScript null guard tests (TestByRoleCurrentScriptNullGuard)
# ---------------------------------------------------------------------------


class TestByRoleCurrentScriptNullGuard:
    """Tests that by_role.html guards against document.currentScript being null."""

    def test_container_assignment_checks_current_script_null(self, client):
        """Container assignment must null-check document.currentScript before .closest()."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        assert 'document.currentScript &&' in html

    def test_container_assignment_has_document_query_fallback(self, client):
        """Container assignment must fall back to document.querySelector when currentScript is null."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        assert "document.querySelector('[data-by-role-container]')" in html

    def test_early_exit_guard_after_container_assignment(self, client):
        """Script must have an early-exit guard immediately after container assignment."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        assert 'if (!container) return' in html


# ---------------------------------------------------------------------------
# Step-9: Render error recovery tests (TestByRoleRenderErrorRecovery)
# ---------------------------------------------------------------------------


class TestByRoleRenderChartStructure:
    """Tests that by_role.html renderCharts has correct structure."""

    def test_render_charts_function_present(self, client):
        """renderCharts function must be defined."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        assert 'function renderCharts' in html

    def test_foreach_iterates_project_ids(self, client):
        """The projectIds.forEach loop must iterate over project keys."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        assert 'projectIds.forEach(' in html

    def test_cleanup_before_new_charts(self, client):
        """cleanupCharts() must be called before creating new Chart instances."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        cleanup_pos = html.find('cleanupCharts()')
        new_chart_pos = html.find('new Chart(')
        assert cleanup_pos != -1, 'cleanupCharts() call not found'
        assert new_chart_pos != -1, 'new Chart( not found'
        assert cleanup_pos < new_chart_pos


# ---------------------------------------------------------------------------
# Step-11: Null-dereference guard tests (TestByRoleNullDereferenceGuard)
# ---------------------------------------------------------------------------


class TestByRoleNullDereferenceGuard:
    """Tests that by_role.html guards against null byRole and keeps cleanupCharts /
    Object.keys inside the try block so a null server value exits cleanly."""

    def test_byrole_null_guard_in_render_charts(self, client):
        """renderCharts must guard against null/non-object byRole before rendering."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        render_charts_pos = html.find('function renderCharts')
        assert render_charts_pos != -1, "'function renderCharts' not found in script"
        null_guard_pos = html.find('!byRole', render_charts_pos)
        assert null_guard_pos != -1, (
            "'!byRole' guard not found inside function renderCharts — "
            "renderCharts must guard against null/non-object byRole before rendering"
        )

    def test_cleanup_charts_after_null_guard(self, client):
        """cleanupCharts() call must appear after the null guard in renderCharts."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        render_charts_pos = html.find('function renderCharts')
        assert render_charts_pos != -1, "'function renderCharts' not found in script"
        null_guard_pos = html.find('!byRole', render_charts_pos)
        cleanup_call_pos = html.find('cleanupCharts()', render_charts_pos)
        assert null_guard_pos != -1, "'!byRole' guard not found"
        assert cleanup_call_pos != -1, "'cleanupCharts()' call not found"
        assert null_guard_pos < cleanup_call_pos, (
            "null guard must appear before cleanupCharts() call"
        )

    def test_object_keys_after_null_guard(self, client):
        """Object.keys(byRole) must appear after the null guard in renderCharts."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        render_charts_pos = html.find('function renderCharts')
        assert render_charts_pos != -1, "'function renderCharts' not found in script"
        null_guard_pos = html.find('!byRole', render_charts_pos)
        object_keys_pos = html.find('Object.keys(byRole)', render_charts_pos)
        assert null_guard_pos != -1, "'!byRole' guard not found"
        assert object_keys_pos != -1, "'Object.keys(byRole)' not found"
        assert null_guard_pos < object_keys_pos, (
            "null guard must appear before Object.keys(byRole)"
        )
