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
