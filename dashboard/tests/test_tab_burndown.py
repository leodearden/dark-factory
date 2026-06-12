"""Static source-contract tests for the Burndown tab wiring.

Tests fetch JSX source via TestClient and assert structural contracts as text.
Follows the idiom in test_tab_curator.py / test_tab_overview.py.
"""

from __future__ import annotations

import re

import pytest
from starlette.testclient import TestClient


@pytest.fixture(scope='module')
def _client():
    from dashboard.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope='module')
def tabs_jsx_body(_client):
    return _client.get('/static/redux/tabs.jsx').text


@pytest.fixture(scope='module')
def app_jsx_body(_client):
    return _client.get('/static/redux/app.jsx').text


# ---------------------------------------------------------------------------
# app.jsx — passes displayWindow (not `window`) into BurnTab
# ---------------------------------------------------------------------------

class TestAppDisplayWindowProp:
    def test_burntab_receives_display_window(self, app_jsx_body):
        """app.jsx must forward the active display-window key into BurnTab."""
        assert re.search(r'<BurnTab[^>]*displayWindow=\{win\}', app_jsx_body)

    def test_burntab_prop_not_named_window(self, app_jsx_body):
        """The prop must not shadow the browser global `window` inside BurnTab."""
        assert not re.search(r'<BurnTab[^>]*\bwindow=', app_jsx_body)


# ---------------------------------------------------------------------------
# tabs.jsx BurnTab — signature and smoothing state
# ---------------------------------------------------------------------------

class TestBurnTabSignature:
    def test_destructures_display_window(self, tabs_jsx_body):
        """BurnTab must accept displayWindow in its parameter destructure."""
        assert re.search(r'function BurnTab\s*\(\s*\{[^}]*displayWindow', tabs_jsx_body)

    def test_does_not_destructure_window_param(self, tabs_jsx_body):
        """BurnTab must NOT have a param named `window` (would shadow the global)."""
        assert not re.search(r'function BurnTab\s*\(\s*\{[^}]*\bwindow\b', tabs_jsx_body)


class TestBurnTabSmoothingChip:
    def test_default_smoothing_for_window_referenced(self, tabs_jsx_body):
        """BurnTab must call defaultSmoothingForWindow to derive the default."""
        assert 'defaultSmoothingForWindow' in tabs_jsx_body

    def test_smoothing_options_referenced(self, tabs_jsx_body):
        """BurnTab must reference SMOOTHING_OPTIONS for the chip's option list."""
        assert 'SMOOTHING_OPTIONS' in tabs_jsx_body

    def test_chip_group_rendered_for_smoothing(self, tabs_jsx_body):
        """A ChipGroup (or equivalent) must appear in BurnTab for smoothing."""
        # Accept either ChipGroup or Segmented — both are reusable controls.
        assert 'ChipGroup' in tabs_jsx_body or re.search(
            r'Segmented[^;]*SMOOTHING_OPTIONS', tabs_jsx_body
        )

    def test_charts_destructure_includes_smoothing_exports(self, tabs_jsx_body):
        """tabs.jsx must destructure the smoothing helpers from window.DF_CHARTS."""
        assert 'deriveVelocitySeries' in tabs_jsx_body
        assert 'defaultSmoothingForWindow' in tabs_jsx_body
        assert 'SMOOTHING_OPTIONS' in tabs_jsx_body


# ---------------------------------------------------------------------------
# velocity sparks — added in step-7 (these fail until step-8 GREEN)
# ---------------------------------------------------------------------------

class TestVelocitySparkWiring:
    def test_net_velocity_tile_uses_derive(self, tabs_jsx_body):
        """Net velocity StatTile spark must use deriveVelocitySeries, not raw b.done."""
        assert re.search(
            r'label=["\']Net velocity["\'][^/]*/?>',
            tabs_jsx_body,
        ) or re.search(
            r'spark=\{deriveVelocitySeries\(',
            tabs_jsx_body,
        )

    def test_completion_trend_spark_uses_derive(self, tabs_jsx_body):
        """Per-project Velocity panel 'Completion trend' must use deriveVelocitySeries."""
        assert re.search(r'Completion trend', tabs_jsx_body)
        assert re.search(r'deriveVelocitySeries\(pb\.done', tabs_jsx_body)

    def test_backlog_trend_spark_uses_derive(self, tabs_jsx_body):
        """Per-project Velocity panel 'Backlog trend' must use deriveVelocitySeries."""
        assert re.search(r'Backlog trend', tabs_jsx_body)
        assert re.search(r'deriveVelocitySeries\(pb\.pending', tabs_jsx_body)

    def test_status_mix_area_stays_cumulative(self, tabs_jsx_body):
        """Status-mix StackedArea charts must still use raw values (not derived)."""
        # The SA stacks for the aggregate view use b.done directly.
        assert re.search(r"values:\s*b\.done", tabs_jsx_body)

    def test_completed_window_tile_stays_cumulative(self, tabs_jsx_body):
        """'Completed (window)' tile spark must remain on raw b.done."""
        assert re.search(
            r'label=["\']Completed \(window\)["\']',
            tabs_jsx_body,
        )
