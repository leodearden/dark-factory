"""Unit tests for Jinja2 badge macros in macros/badges.html.

These tests render macros in isolation via a Jinja2 FileSystemLoader,
without requiring the full FastAPI stack.
"""

from __future__ import annotations

from pathlib import Path

import jinja2
import pytest

# Path to the templates directory (same root as app.py uses)
_TEMPLATES_DIR = (
    Path(__file__).parent.parent / 'src' / 'dashboard' / 'templates'
)


@pytest.fixture(scope='module')
def jinja_env() -> jinja2.Environment:
    """Jinja2 Environment pointing at the dashboard templates directory."""
    loader = jinja2.FileSystemLoader(str(_TEMPLATES_DIR))
    return jinja2.Environment(loader=loader, autoescape=True)


def render_status_badge(
    env: jinja2.Environment,
    status: str,
    aria_label: str | None = None,
    color: str | None = None,
) -> str:
    """Render the status_badge macro and return the HTML string."""
    source = (
        "{% from 'macros/badges.html' import status_badge %}"
        "{{ status_badge(status, aria_label, color) }}"
    )
    tmpl = env.from_string(source)
    return tmpl.render(status=status, aria_label=aria_label, color=color)


def render_severity_badge(
    env: jinja2.Environment,
    severity: str,
    aria_label: str | None = None,
) -> str:
    """Render the severity_badge macro and return the HTML string."""
    source = (
        "{% from 'macros/badges.html' import severity_badge %}"
        "{{ severity_badge(severity, aria_label) }}"
    )
    tmpl = env.from_string(source)
    return tmpl.render(severity=severity, aria_label=aria_label)


def render_indicator_dot(
    env: jinja2.Environment, connected: bool, label: str
) -> str:
    """Render the indicator_dot macro and return the HTML string."""
    source = (
        "{% from 'macros/badges.html' import indicator_dot %}"
        "{{ indicator_dot(connected, label) }}"
    )
    tmpl = env.from_string(source)
    return tmpl.render(connected=connected, label=label)


class TestStatusBadge:
    """Tests for the status_badge(status, aria_label=None, color=None) macro."""

    # --- Color mapping tests ---

    def test_done_green(self, jinja_env):
        html = render_status_badge(jinja_env, 'done')
        assert 'bg-green-600' in html

    def test_in_progress_blue(self, jinja_env):
        html = render_status_badge(jinja_env, 'in-progress')
        assert 'bg-blue-600' in html

    def test_pending_yellow(self, jinja_env):
        html = render_status_badge(jinja_env, 'pending')
        assert 'bg-yellow-600' in html

    def test_blocked_red(self, jinja_env):
        html = render_status_badge(jinja_env, 'blocked')
        assert 'bg-red-600' in html

    def test_cancelled_maps_to_gray(self, jinja_env):
        html = render_status_badge(jinja_env, 'cancelled')
        assert 'bg-gray-600' in html

    def test_deferred_maps_to_gray(self, jinja_env):
        html = render_status_badge(jinja_env, 'deferred')
        assert 'bg-gray-600' in html

    def test_running_blue(self, jinja_env):
        html = render_status_badge(jinja_env, 'running')
        assert 'bg-blue-600' in html

    def test_completed_green(self, jinja_env):
        html = render_status_badge(jinja_env, 'completed')
        assert 'bg-green-600' in html

    def test_failed_red(self, jinja_env):
        html = render_status_badge(jinja_env, 'failed')
        assert 'bg-red-600' in html

    def test_rolled_back_maps_to_orange(self, jinja_env):
        html = render_status_badge(jinja_env, 'rolled_back')
        assert 'bg-orange-600' in html

    def test_circuit_breaker_purple(self, jinja_env):
        html = render_status_badge(jinja_env, 'circuit_breaker')
        assert 'bg-purple-600' in html

    def test_bursting_orange(self, jinja_env):
        html = render_status_badge(jinja_env, 'bursting')
        assert 'bg-orange-600' in html

    def test_idle_maps_to_gray(self, jinja_env):
        html = render_status_badge(jinja_env, 'idle')
        assert 'bg-gray-600' in html

    # --- Fallback for unknown status ---

    def test_unknown_fallback_gray(self, jinja_env):
        html = render_status_badge(jinja_env, 'xyz')
        assert 'bg-gray-600' in html

    # --- Visible text tests ---

    def test_renders_status_text(self, jinja_env):
        html = render_status_badge(jinja_env, 'in-progress')
        assert 'in-progress' in html

    def test_unknown_status_text_appears_as_badge_content(self, jinja_env):
        html = render_status_badge(jinja_env, 'weird_state')
        assert 'weird_state' in html

    # --- aria-label tests ---

    def test_no_aria_label_by_default(self, jinja_env):
        html = render_status_badge(jinja_env, 'done')
        assert 'aria-label' not in html

    def test_aria_label_with_prefix(self, jinja_env):
        html = render_status_badge(jinja_env, 'done', aria_label='Run status')
        assert 'aria-label="Run status: done"' in html

    def test_aria_label_orchestrator_status(self, jinja_env):
        html = render_status_badge(jinja_env, 'running', aria_label='Orchestrator status')
        assert 'aria-label="Orchestrator status: running"' in html

    def test_aria_label_burst_state(self, jinja_env):
        html = render_status_badge(jinja_env, 'bursting', aria_label='Burst state')
        assert 'aria-label="Burst state: bursting"' in html

    # --- Color override tests ---

    def test_color_override(self, jinja_env):
        # running default is bg-blue-600, but override with bg-green-600
        html = render_status_badge(jinja_env, 'running', color='bg-green-600')
        assert 'bg-green-600' in html
        assert 'bg-blue-600' not in html

    def test_color_override_with_none_uses_map(self, jinja_env):
        # None override → falls back to color map
        html = render_status_badge(jinja_env, 'done', color=None)
        assert 'bg-green-600' in html

    # --- HTML structure tests ---

    def test_badge_span_classes(self, jinja_env):
        html = render_status_badge(jinja_env, 'done')
        assert 'inline-block' in html
        assert 'px-2' in html
        assert 'py-0.5' in html
        assert 'text-xs' in html
        assert 'font-medium' in html
        assert 'rounded' in html
        assert 'text-white' in html

    def test_renders_span_element(self, jinja_env):
        html = render_status_badge(jinja_env, 'done')
        assert '<span' in html
        assert '</span>' in html


class TestSeverityBadge:
    """Tests for the severity_badge(severity, aria_label=None) macro."""

    # --- Color mapping tests ---

    def test_ok_green(self, jinja_env):
        html = render_severity_badge(jinja_env, 'ok')
        assert 'bg-green-600' in html

    def test_minor_yellow(self, jinja_env):
        html = render_severity_badge(jinja_env, 'minor')
        assert 'bg-yellow-600' in html

    def test_moderate_orange(self, jinja_env):
        html = render_severity_badge(jinja_env, 'moderate')
        assert 'bg-orange-600' in html

    def test_serious_red(self, jinja_env):
        html = render_severity_badge(jinja_env, 'serious')
        assert 'bg-red-600' in html

    def test_unknown_fallback_gray(self, jinja_env):
        html = render_severity_badge(jinja_env, 'xyz')
        assert 'bg-gray-600' in html

    # --- Visible text tests ---

    def test_renders_severity_text(self, jinja_env):
        html = render_severity_badge(jinja_env, 'ok')
        assert 'ok' in html

    def test_unknown_severity_text_appears_as_content(self, jinja_env):
        html = render_severity_badge(jinja_env, 'critical')
        assert 'critical' in html

    # --- aria-label tests ---

    def test_no_aria_label_by_default(self, jinja_env):
        html = render_severity_badge(jinja_env, 'ok')
        assert 'aria-label' not in html

    def test_aria_label_with_prefix(self, jinja_env):
        html = render_severity_badge(jinja_env, 'ok', aria_label='Verdict severity')
        assert 'aria-label="Verdict severity: ok"' in html

    def test_aria_label_minor(self, jinja_env):
        html = render_severity_badge(jinja_env, 'minor', aria_label='Verdict severity')
        assert 'aria-label="Verdict severity: minor"' in html

    def test_aria_label_serious(self, jinja_env):
        html = render_severity_badge(jinja_env, 'serious', aria_label='Verdict severity')
        assert 'aria-label="Verdict severity: serious"' in html

    # --- HTML structure tests ---

    def test_renders_span_element(self, jinja_env):
        html = render_severity_badge(jinja_env, 'ok')
        assert '<span' in html
        assert '</span>' in html

    def test_has_badge_classes(self, jinja_env):
        html = render_severity_badge(jinja_env, 'ok')
        assert 'inline-block' in html
        assert 'text-white' in html
        assert 'rounded' in html


class TestIndicatorDot:
    """Tests for the indicator_dot(connected, label) macro."""

    # --- Color tests ---

    def test_connected_green(self, jinja_env):
        html = render_indicator_dot(jinja_env, connected=True, label='Graphiti')
        assert 'bg-green-500' in html

    def test_disconnected_red(self, jinja_env):
        html = render_indicator_dot(jinja_env, connected=False, label='Graphiti')
        assert 'bg-red-500' in html

    # --- role attribute tests ---

    def test_role_status(self, jinja_env):
        html = render_indicator_dot(jinja_env, connected=True, label='Mem0')
        assert 'role="status"' in html

    # --- aria-label tests ---

    def test_connected_aria_label(self, jinja_env):
        html = render_indicator_dot(jinja_env, connected=True, label='Graphiti')
        assert 'aria-label="Graphiti connected"' in html

    def test_disconnected_aria_label(self, jinja_env):
        html = render_indicator_dot(jinja_env, connected=False, label='Mem0')
        assert 'aria-label="Mem0 disconnected"' in html

    def test_label_appears_in_aria_label(self, jinja_env):
        html = render_indicator_dot(jinja_env, connected=True, label='Taskmaster')
        assert 'aria-label="Taskmaster connected"' in html

    # --- CSS class tests ---

    def test_dot_classes(self, jinja_env):
        html = render_indicator_dot(jinja_env, connected=True, label='Mem0')
        assert 'w-3' in html
        assert 'h-3' in html
        assert 'rounded-full' in html

    # --- Element type tests ---

    def test_renders_div(self, jinja_env):
        html = render_indicator_dot(jinja_env, connected=True, label='Graphiti')
        assert '<div' in html
        assert '</div>' in html
        assert '<span' not in html


class TestTemplateMacroImports:
    """Structural tests verifying template files import from macros/badges.html."""

    _tmpl_dir = (
        Path(__file__).parent.parent / 'src' / 'dashboard' / 'templates'
    )

    def test_tasks_imports_status_badge(self):
        content = (self._tmpl_dir / 'partials' / 'tasks.html').read_text()
        assert "from 'macros/badges.html' import status_badge" in content

    def test_orchestrators_imports_status_badge(self):
        content = (self._tmpl_dir / 'partials' / 'orchestrators.html').read_text()
        assert "from 'macros/badges.html' import status_badge" in content

    def test_recon_imports_status_badge(self):
        content = (self._tmpl_dir / 'partials' / 'recon.html').read_text()
        assert "from 'macros/badges.html' import" in content
        assert 'status_badge' in content

    def test_recon_imports_severity_badge(self):
        content = (self._tmpl_dir / 'partials' / 'recon.html').read_text()
        assert 'severity_badge' in content

    def test_memory_imports_indicator_dot(self):
        content = (self._tmpl_dir / 'partials' / 'memory.html').read_text()
        assert "from 'macros/badges.html' import indicator_dot" in content
