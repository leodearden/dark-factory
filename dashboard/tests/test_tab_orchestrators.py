"""Wiring tests for the Orchestrators tab UI in tabs.jsx.

Tests parse JSX source as text and assert structural contracts.
Follows the idiom established in test_tab_curator.py / test_index_html.py.
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


class TestOrchTabCurrentFocusRemoved:
    """The 'Current focus' UI block must not appear in the Orchestrators tab."""

    def test_tabs_jsx_served(self, _client):
        resp = _client.get('/static/redux/tabs.jsx')
        assert resp.status_code == 200

    def test_orch_tab_positive_anchor_function(self, tabs_jsx_body):
        """File must still export OrchTab — guards against a renamed/empty file."""
        assert 'function OrchTab(' in tabs_jsx_body

    def test_orch_tab_positive_anchor_task_filter(self, tabs_jsx_body):
        """Task-filter segment must remain — it is NOT removed by this task."""
        assert 'aria-label="Task filter"' in tabs_jsx_body

    def test_current_focus_label_removed(self, tabs_jsx_body):
        """'Current focus' UI label must NOT appear in OrchTab."""
        assert 'Current focus' not in tabs_jsx_body

    def test_current_task_render_ref_removed(self, tabs_jsx_body):
        """The JSX expression {o.current_task} must NOT appear in tabs.jsx.

        Uses a regex to match the removed JSX expression specifically, avoiding
        false failures on related identifiers (e.g. o.current_task_count).

        NOTE: the backend field current_task and its last consumer
        (tab_overview.jsx, Overview tab) were fully removed by task 1571.
        This guard remains to prevent re-introducing a {o.current_task}
        consumer in tabs.jsx.
        """
        assert not re.search(r'\{\s*o\.current_task\s*\}', tabs_jsx_body)


class TestOrchTabLastUpdate:
    """OrchTab must render the per-orchestrator last-update timestamp."""

    def test_orch_tab_renders_last_update_via_timeago(self, tabs_jsx_body):
        """tabs.jsx must contain the render reference timeago(o.last_update)."""
        assert 'timeago(o.last_update)' in tabs_jsx_body

    def test_orch_tab_has_updated_label(self, tabs_jsx_body):
        """tabs.jsx must contain an 'Updated' label token in OrchTab."""
        assert 'Updated' in tabs_jsx_body
