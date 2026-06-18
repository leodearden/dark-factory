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

    def test_current_task_td_render_ref_removed(self, tab_overview_jsx_body):
        """The JSX expression {o.current_task} must NOT appear in tab_overview.jsx."""
        assert not re.search(r'\{\s*o\.current_task\s*\}', tab_overview_jsx_body)

    def test_current_task_th_removed(self, tab_overview_jsx_body):
        """The <th>Current task</th> column header must NOT appear in tab_overview.jsx."""
        assert '<th>Current task</th>' not in tab_overview_jsx_body


class TestHostLoadCardStaleness:
    """HostLoadCard must surface a stale/offline badge when /api/load fails.

    step-19 RED: All three tests fail today because HostLoadCard has no stale
    state, `if (!resp.ok) return;` silently drops the error, and the panel-head
    renders no badge.  After step-20 GREEN these all pass.
    """

    def test_host_load_card_stale_state_declared(self, tab_overview_jsx_body):
        """HostLoadCard must declare a useState hook for stale state.

        Looks for `setStale` as a setter from a useState destructuring, e.g.:
            const [stale, setStale] = useState(false)
        Fails today — no stale hook exists.
        """
        assert re.search(r'setStale\s*[,\)]', tab_overview_jsx_body), (
            'HostLoadCard must declare a stale state hook '
            '(e.g. const [stale, setStale] = useState(false)); '
            'currently no stale hook is declared in tab_overview.jsx'
        )

    def test_host_load_card_failure_sets_stale_true(self, tab_overview_jsx_body):
        """setStale(true) must be called on !resp.ok and/or in the catch block.

        Fails today — the `!resp.ok` branch only does `return`, and the catch
        swallows the error silently, leaving stale state never set.
        """
        assert re.search(r'setStale\s*\(\s*true\s*\)', tab_overview_jsx_body), (
            'HostLoadCard must call setStale(true) when the fetch fails '
            '(!resp.ok branch or catch block); '
            'currently the failure path only returns/swallows silently'
        )

    def test_host_load_card_panel_head_stale_badge(self, tab_overview_jsx_body):
        """HostLoadCard panel-head must render a stale/offline badge element.

        Accepts either a className containing 'stale' (e.g. className="badge stale")
        or an element with visible text '>stale<' / '>offline<'.
        Fails today — no such badge element exists in the panel-head.
        """
        jsx = tab_overview_jsx_body
        has_stale_class = bool(re.search(r'className=["\'][^"\']*stale', jsx))
        has_stale_text = bool(re.search(r'>stale<|>offline<', jsx))
        assert has_stale_class or has_stale_text, (
            'HostLoadCard panel-head must render a stale/offline badge '
            '(e.g. <span className="badge stale">stale</span> or equivalent); '
            'currently no stale badge is rendered in tab_overview.jsx'
        )
