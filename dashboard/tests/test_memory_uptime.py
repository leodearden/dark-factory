"""Wiring tests for the fused-memory uptime feature.

Follows the idiom from test_tab_overview.py / test_index_html.py:
fetch served .jsx via Starlette TestClient and assert structural substrings.
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
def shell_jsx_body(_client):
    return _client.get('/static/redux/shell.jsx').text


@pytest.fixture(scope='module')
def tabs_jsx_body(_client):
    return _client.get('/static/redux/tabs.jsx').text


@pytest.fixture(scope='module')
def tab_overview_jsx_body(_client):
    return _client.get('/static/redux/tab_overview.jsx').text


# ---------------------------------------------------------------------------
# shell.jsx — fmtUptime helper defined and exported (step-3 RED / step-4 GREEN)
# ---------------------------------------------------------------------------


class TestShellFmtUptimeDefined:
    """shell.jsx must define fmtUptime and export it via window.DF_SHELL."""

    def test_shell_jsx_served(self, _client):
        resp = _client.get('/static/redux/shell.jsx')
        assert resp.status_code == 200

    def test_fmtUptime_function_defined(self, shell_jsx_body):
        """fmtUptime function must be defined in shell.jsx."""
        assert 'function fmtUptime(' in shell_jsx_body

    def test_fmtUptime_exported_on_DF_SHELL(self, shell_jsx_body):
        """fmtUptime must appear inside the window.DF_SHELL export object."""
        # Find the window.DF_SHELL = { ... } assignment and check fmtUptime is in it
        assert re.search(r'window\.DF_SHELL\s*=\s*\{[^}]*fmtUptime', shell_jsx_body)


# ---------------------------------------------------------------------------
# tabs.jsx — MemoryTab uptime wiring (step-5 RED / step-6 GREEN)
# ---------------------------------------------------------------------------


class TestTabsMemoryTabUptimeWiring:
    """MemoryTab in tabs.jsx must reference uptime fields and fmtUptime."""

    def test_tabs_jsx_served(self, _client):
        resp = _client.get('/static/redux/tabs.jsx')
        assert resp.status_code == 200

    def test_positive_anchor_function(self, tabs_jsx_body):
        """tabs.jsx must still export MemoryTab — guards against rename."""
        assert 'function MemoryTab(' in tabs_jsx_body

    def test_references_uptime_seconds(self, tabs_jsx_body):
        """MemoryTab must reference MEMORY_STATUS.uptime_seconds."""
        assert 'MEMORY_STATUS.uptime_seconds' in tabs_jsx_body

    def test_calls_fmtUptime(self, tabs_jsx_body):
        """MemoryTab must call window.DF_SHELL.fmtUptime."""
        assert 'fmtUptime' in tabs_jsx_body

    def test_references_started_at(self, tabs_jsx_body):
        """MemoryTab must reference MEMORY_STATUS.started_at for hover/title."""
        assert 'MEMORY_STATUS.started_at' in tabs_jsx_body


# ---------------------------------------------------------------------------
# tab_overview.jsx — System-health fused-memory row (step-7 RED / step-8 GREEN)
# ---------------------------------------------------------------------------


class TestOverviewTabFusedMemoryRow:
    """System-health block in tab_overview.jsx must include a fused-memory row."""

    def test_tab_overview_jsx_served(self, _client):
        resp = _client.get('/static/redux/tab_overview.jsx')
        assert resp.status_code == 200

    def test_positive_anchor_function(self, tab_overview_jsx_body):
        """tab_overview.jsx must still export OverviewTab."""
        assert 'function OverviewTab(' in tab_overview_jsx_body

    def test_references_uptime_seconds(self, tab_overview_jsx_body):
        """System-health block must reference MEMORY_STATUS.uptime_seconds."""
        assert 'MEMORY_STATUS.uptime_seconds' in tab_overview_jsx_body

    def test_calls_fmtUptime(self, tab_overview_jsx_body):
        """System-health block must call fmtUptime."""
        assert 'fmtUptime' in tab_overview_jsx_body

    def test_renders_fused_memory_label(self, tab_overview_jsx_body):
        """System-health block must include a 'fused-memory' label."""
        assert 'fused-memory' in tab_overview_jsx_body
