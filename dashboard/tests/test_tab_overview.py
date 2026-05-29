"""Wiring tests for the Overview tab UI in tab_overview.jsx.

Tests parse JSX source as text and assert structural contracts.
Follows the idiom established in test_tab_curator.py / test_tab_orchestrators.py /
test_index_html.py.
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
def tab_overview_jsx_body(_client):
    return _client.get('/static/redux/tab_overview.jsx').text


class TestOverviewTabCurrentTaskRemoved:
    """The 'Current task' column must not appear in the Overview tab."""

    def test_tab_overview_jsx_served(self, _client):
        resp = _client.get('/static/redux/tab_overview.jsx')
        assert resp.status_code == 200

    def test_overview_tab_positive_anchor_function(self, tab_overview_jsx_body):
        """File must still export OverviewTab — guards against a renamed/empty file."""
        assert 'function OverviewTab(' in tab_overview_jsx_body

    def test_overview_tab_positive_anchor_table_title(self, tab_overview_jsx_body):
        """'Orchestrators · current work' table title must still be present."""
        assert 'Orchestrators · current work' in tab_overview_jsx_body

    def test_current_task_td_render_ref_removed(self, tab_overview_jsx_body):
        """The JSX expression {o.current_task} must NOT appear in tab_overview.jsx."""
        assert not re.search(r'\{\s*o\.current_task\s*\}', tab_overview_jsx_body)

    def test_current_task_th_removed(self, tab_overview_jsx_body):
        """The <th>Current task</th> column header must NOT appear in tab_overview.jsx."""
        assert '<th>Current task</th>' not in tab_overview_jsx_body
