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


class TestByRoleRenderGuard:
    """Tests that by_role.html prevents double-render via a rendered boolean guard."""

    def test_rendered_guard_variable_declared(self, client):
        """Script must declare a 'rendered' boolean guard variable."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        assert 'var rendered = false' in html

    def test_render_charts_starts_with_guard_check(self, client):
        """renderCharts function body must start with the guard check."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        assert 'if (rendered) return' in html

    def test_htmx_after_settle_listener_present(self, client):
        """htmx:afterSettle listener must still be registered."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        assert "htmx:afterSettle" in html

    def test_rendered_set_to_true_inside_render_charts(self, client):
        """renderCharts must set rendered = true to latch after first execution."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        assert 'rendered = true' in html


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


class TestByRoleRenderErrorRecovery:
    """Tests that by_role.html recovers from render-time exceptions without permanent lockout."""

    def test_foreach_loop_wrapped_in_try_block(self, client):
        """The projectIds.forEach loop must be wrapped in a try block."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        try_pos = html.find('try {')
        foreach_pos = html.find('.forEach(')
        assert try_pos != -1, "'try {' not found in script"
        assert foreach_pos != -1, "'.forEach(' not found in script"
        assert try_pos < foreach_pos, (
            "'try {' must appear before '.forEach(' in the script"
        )

    def test_rendered_true_set_after_foreach_loop(self, client):
        """rendered = true must be set AFTER the forEach loop (inside the try block), not before it."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        foreach_pos = html.find('.forEach(')
        rendered_true_pos = html.find('rendered = true')
        assert foreach_pos != -1, "'.forEach(' not found in script"
        assert rendered_true_pos != -1, "'rendered = true' not found in script"
        assert rendered_true_pos > foreach_pos, (
            "'rendered = true' must appear AFTER '.forEach(' — it should only latch after successful loop completion"
        )

    def test_catch_block_resets_rendered_false(self, client):
        """A catch block must exist and reset rendered = false to allow retry."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        catch_pos = html.find('catch')
        rendered_false_pos = html.find('rendered = false')
        assert catch_pos != -1, "'catch' not found in script"
        assert rendered_false_pos != -1, "'rendered = false' not found in script"
        assert rendered_false_pos > catch_pos, (
            "'rendered = false' must appear inside the catch block (after 'catch')"
        )

    def test_catch_block_rethrows_error(self, client):
        """The catch block must re-throw the error so it is not silently swallowed."""
        with _patch_by_role():
            html = client.get('/costs/partials/by-role').text
        catch_pos = html.find('catch')
        throw_pos = html.find('throw', catch_pos)
        assert catch_pos != -1, "'catch' not found in script"
        assert throw_pos != -1, (
            "'throw' not found after 'catch' — catch block must re-throw the error"
        )
