"""Structural-contract tests for MergeTab in tabs.jsx.

Fetches the served .jsx asset as text and asserts structural contracts.
Follows the idiom established in test_tab_overview.py / test_tab_curator.py /
test_tab_orchestrators.py.
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


class TestMergeTabRecentMergesRowCount:
    """MergeTab's 'Recent merges' panel-head must display a row count."""

    def test_tabs_jsx_served(self, _client):
        resp = _client.get('/static/redux/tabs.jsx')
        assert resp.status_code == 200

    def test_merge_tab_positive_anchor(self, tabs_jsx_body):
        """File must still export MergeTab — guards against rename/empty."""
        assert 'function MergeTab(' in tabs_jsx_body

    def test_recent_merges_row_count_expression(self, tabs_jsx_body):
        """MergeTab's Recent merges panel-head must render d.recent.length
        matching the established Recent-runs convention ({N} matching)."""
        assert re.search(r'd\.recent\.length', tabs_jsx_body), (
            'Expected d.recent.length expression in tabs.jsx MergeTab — '
            'add a row-count meta span to the Recent merges panel-head'
        )
