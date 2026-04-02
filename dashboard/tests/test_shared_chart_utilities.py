"""Regression tests for the shared Chart.js utilities (chart-utils.js).

Verifies:
  1. chart-utils.js defines CHART_PALETTE (10 colours) and getOrDestroyChart.
  2. base.html loads chart-utils.js after Chart.js CDN but before {% block content %}.
  3. No partial template re-defines either symbol inline.
  4. No partial registers an afterSettle/addEventListener handler (single IIFE path).
  5. The 5 palette-using partials reference CHART_PALETTE[] from the shared file.

All tests read template/static files directly via pathlib — no HTTP client needed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_DASHBOARD_ROOT = Path(__file__).resolve().parent.parent
_STATIC_DIR = _DASHBOARD_ROOT / 'src' / 'dashboard' / 'static'
_TEMPLATES_DIR = _DASHBOARD_ROOT / 'src' / 'dashboard' / 'templates'
_PARTIALS_DIR = _TEMPLATES_DIR / 'partials'

_CHART_UTILS_JS = _STATIC_DIR / 'chart-utils.js'
_BASE_HTML = _TEMPLATES_DIR / 'base.html'

# All 7 chart partials — regression coverage prevents re-introduction of inline copies.
_ALL_CHART_PARTIALS = [
    _PARTIALS_DIR / 'costs' / 'by_project.html',
    _PARTIALS_DIR / 'costs' / 'by_account.html',
    _PARTIALS_DIR / 'costs' / 'by_role.html',
    _PARTIALS_DIR / 'costs' / 'trend.html',
    _PARTIALS_DIR / 'memory_graphs.html',
    _PARTIALS_DIR / 'burndown' / 'charts.html',
    _PARTIALS_DIR / 'performance.html',
]

# 5 partials that cycle colours from the shared CHART_PALETTE array.
# burndown/charts.html (zoneColors) and performance.html (pathColors) use
# domain-specific colour maps — excluding them from this check is intentional.
_PALETTE_PARTIALS = [
    _PARTIALS_DIR / 'costs' / 'by_project.html',
    _PARTIALS_DIR / 'costs' / 'by_account.html',
    _PARTIALS_DIR / 'costs' / 'by_role.html',
    _PARTIALS_DIR / 'costs' / 'trend.html',
    _PARTIALS_DIR / 'memory_graphs.html',
]


# ===========================================================================
# Step-1 — TestChartUtilsJsContents
# ===========================================================================


class TestChartUtilsJsContents:
    """chart-utils.js defines both shared symbols correctly."""

    def test_chart_palette_defined(self):
        """chart-utils.js must declare var CHART_PALETTE."""
        src = _CHART_UTILS_JS.read_text()
        assert 'var CHART_PALETTE' in src

    def test_chart_palette_has_ten_colors(self):
        """CHART_PALETTE must contain exactly 10 hex colour literals."""
        src = _CHART_UTILS_JS.read_text()
        hex_colors = re.findall(r"'#[0-9a-fA-F]{6}'", src)
        assert len(hex_colors) == 10, (
            f'Expected 10 hex colours in CHART_PALETTE, found {len(hex_colors)}: {hex_colors}'
        )

    def test_get_or_destroy_chart_defined(self):
        """chart-utils.js must define function getOrDestroyChart."""
        src = _CHART_UTILS_JS.read_text()
        assert 'function getOrDestroyChart' in src

    def test_get_or_destroy_chart_returns_null_on_missing(self):
        """getOrDestroyChart must return null when element is absent from the DOM."""
        src = _CHART_UTILS_JS.read_text()
        assert 'return null' in src


# ===========================================================================
# Step-3 — TestBaseHtmlLoadsChartUtils
# ===========================================================================


class TestBaseHtmlLoadsChartUtils:
    """base.html loads chart-utils.js in the correct position."""

    def test_chart_utils_script_tag_present(self):
        """base.html must contain a script tag referencing chart-utils.js."""
        src = _BASE_HTML.read_text()
        assert 'chart-utils.js' in src

    def test_loaded_after_chartjs(self):
        """chart-utils.js must appear after the Chart.js CDN script in base.html."""
        src = _BASE_HTML.read_text()
        cdn_pos = src.find('cdn.jsdelivr.net/npm/chart.js')
        utils_pos = src.find('chart-utils.js')
        assert cdn_pos != -1, 'Chart.js CDN script tag not found in base.html'
        assert utils_pos != -1, 'chart-utils.js script tag not found in base.html'
        assert cdn_pos < utils_pos, (
            'chart-utils.js must appear after the Chart.js CDN script in base.html'
        )

    def test_loaded_before_content_block(self):
        """chart-utils.js must appear before {% block content %} in base.html."""
        src = _BASE_HTML.read_text()
        utils_pos = src.find('chart-utils.js')
        content_pos = src.find('{% block content %}')
        assert utils_pos != -1, 'chart-utils.js script tag not found in base.html'
        assert content_pos != -1, '{% block content %} not found in base.html'
        assert utils_pos < content_pos, (
            'chart-utils.js must be loaded before {% block content %} in base.html'
        )


# ===========================================================================
# Step-5 — TestNoInlineDefinitions
# ===========================================================================


class TestNoInlineDefinitions:
    """No chart partial re-defines CHART_PALETTE or getOrDestroyChart inline."""

    @pytest.mark.parametrize('partial', _ALL_CHART_PARTIALS, ids=lambda p: p.name)
    def test_no_inline_palette_definition(self, partial: Path):
        """Template must not declare its own CHART_PALETTE (would shadow the shared global)."""
        src = partial.read_text()
        assert 'var CHART_PALETTE' not in src, (
            f'{partial.name} defines var CHART_PALETTE inline — move it to chart-utils.js'
        )
        assert 'const CHART_PALETTE' not in src, (
            f'{partial.name} defines const CHART_PALETTE inline — move it to chart-utils.js'
        )

    @pytest.mark.parametrize('partial', _ALL_CHART_PARTIALS, ids=lambda p: p.name)
    def test_no_inline_get_or_destroy_definition(self, partial: Path):
        """Template must not define its own getOrDestroyChart (would duplicate the shared global)."""
        src = partial.read_text()
        assert 'function getOrDestroyChart' not in src, (
            f'{partial.name} defines function getOrDestroyChart inline — use the shared global'
        )


# ===========================================================================
# Step-7 — TestSingleRenderCodePath
# ===========================================================================


class TestSingleRenderCodePath:
    """Chart partials use a single immediate IIFE render path — no event listeners."""

    @pytest.mark.parametrize('partial', _ALL_CHART_PARTIALS, ids=lambda p: p.name)
    def test_no_after_settle_listener(self, partial: Path):
        """Template must not register an htmx:afterSettle listener for chart rendering."""
        src = partial.read_text()
        assert 'afterSettle' not in src, (
            f'{partial.name} registers an afterSettle listener — remove it and keep '
            f'the immediate IIFE render call only'
        )

    @pytest.mark.parametrize('partial', _ALL_CHART_PARTIALS, ids=lambda p: p.name)
    def test_no_add_event_listener_for_charts(self, partial: Path):
        """Template must not call addEventListener (charts render via IIFE, not events)."""
        src = partial.read_text()
        assert 'addEventListener' not in src, (
            f'{partial.name} calls addEventListener — chart partials must use a single '
            f'immediate IIFE render call, not event listener registration'
        )

    @pytest.mark.parametrize('partial', _ALL_CHART_PARTIALS, ids=lambda p: p.name)
    def test_immediate_render_call(self, partial: Path):
        """Template must contain an immediate top-level render call."""
        src = partial.read_text()
        has_render = (
            'renderChart();' in src
            or 'renderCharts();' in src
            or 'renderAll();' in src
        )
        assert has_render, (
            f'{partial.name} has no immediate render call '
            f'(expected renderChart();, renderCharts();, or renderAll();)'
        )


# ===========================================================================
# Step-9 — TestSharedPaletteUsageInPartials
# ===========================================================================


class TestSharedPaletteUsageInPartials:
    """Colour-cycling partials index into the shared CHART_PALETTE rather than a local array."""

    @pytest.mark.parametrize('partial', _PALETTE_PARTIALS, ids=lambda p: p.name)
    def test_references_shared_palette(self, partial: Path):
        """Template must reference CHART_PALETTE[…] to use colours from the shared palette."""
        src = partial.read_text()
        assert 'CHART_PALETTE[' in src, (
            f'{partial.name} does not reference CHART_PALETTE[] — replace any local '
            f'colour array with the shared global from chart-utils.js'
        )
