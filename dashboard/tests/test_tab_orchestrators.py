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


def _extract_function_body(source: str, func_name: str) -> str:
    """Extract the source text of a top-level ``function <func_name>(`` up to
    (but not including) the next top-level ``function`` declaration, or to
    end-of-file if it's the last one.

    Used to scope structural assertions to a single tab's render function so
    an assertion can't be accidentally satisfied by an unrelated tab (e.g.
    ReconTab's "Recent runs" table also has a "Status" column header).
    """
    start_match = re.search(rf'^function {re.escape(func_name)}\(', source, re.MULTILINE)
    assert start_match is not None, f'function {func_name}( not found in tabs.jsx'
    start = start_match.start()
    next_match = re.search(r'^function \w+\(', source[start + 1 :], re.MULTILINE)
    end = start + 1 + next_match.start() if next_match else len(source)
    return source[start:end]


@pytest.fixture(scope='module')
def orch_tab_body(tabs_jsx_body):
    """The OrchTab function's own source text, scoped away from the other
    tab-render functions in the same file (see _extract_function_body)."""
    return _extract_function_body(tabs_jsx_body, 'OrchTab')


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


class TestOrchTabRuntimeColumns:
    """Lane/Phase/State columns + offline-aware rtCell/rtAge formatting for
    warm-lane runtime fields (task 2637).

    Post-task-2636, active-task rows already carry real loops/attempts/
    started/lane/phase/lane_state for online projects, but an offline
    project's rows render blank/`nullm` instead of a legible '—', and there
    are no Lane/Phase/State columns at all. These assertions pin the fix:
    the runtime cells route through window.DF_RUNTIME_FMT's rtCell/rtAge
    (null/undefined -> '—', honest 0 preserved) and the three new columns
    exist.
    """

    def test_destructures_runtime_fmt(self, tabs_jsx_body):
        """window.DF_RUNTIME_FMT (defined by runtime_format.js, loaded earlier
        in index.html) must be destructured at module top level — checked
        against the whole file since top-level destructures sit outside any
        single tab function."""
        assert re.search(r'\bDF_RUNTIME_FMT\b', tabs_jsx_body)

    def test_lane_column_header_present(self, orch_tab_body):
        assert re.search(r'>Lane<', orch_tab_body)

    def test_phase_column_header_present(self, orch_tab_body):
        assert re.search(r'>Phase<', orch_tab_body)

    def test_status_column_header_present(self, orch_tab_body):
        """The pre-existing task-status column header is renamed
        State->Status so the new lane_state column can use 'State' without a
        duplicate column label.

        Scoped to orch_tab_body (not the whole file): ReconTab's "Recent
        runs" table already has an unrelated ">Status<" header elsewhere in
        tabs.jsx, which would otherwise make this assertion a false pass
        before OrchTab itself is touched.
        """
        assert re.search(r'>Status<', orch_tab_body)

    def test_loops_cell_routes_through_rtcell(self, orch_tab_body):
        assert re.search(r'rtCell\(\s*t\.loops', orch_tab_body)

    def test_attempts_cell_routes_through_rtcell(self, orch_tab_body):
        assert re.search(r'rtCell\(\s*t\.attempts', orch_tab_body)

    def test_lane_cell_routes_through_rtcell(self, orch_tab_body):
        assert re.search(r'rtCell\(\s*t\.lane\b', orch_tab_body)

    def test_phase_cell_routes_through_rtcell(self, orch_tab_body):
        assert re.search(r'rtCell\(\s*t\.phase\b', orch_tab_body)

    def test_lane_state_cell_routes_through_rtcell(self, orch_tab_body):
        assert re.search(r'rtCell\(\s*t\.lane_state\b', orch_tab_body)

    def test_age_cell_routes_through_rtage(self, orch_tab_body):
        assert re.search(r'rtAge\(\s*t\.started\b', orch_tab_body)

    def test_unguarded_age_template_removed(self, orch_tab_body):
        """The old un-guarded `${t.started}m` template must be gone — proves
        an offline row's age now degrades to '—' via rtAge instead of
        rendering the literal string "nullm"."""
        assert not re.search(r'\$\{\s*t\.started\s*\}m', orch_tab_body)


class TestOrchTabEmptyState:
    """The empty-task-list cell must read as a sentence, not a stringified
    object (task 3313 — the regression itself is described in
    static/redux/orch_filter.js's header comment).

    These assertions pin only the wiring: the cell delegates to
    window.DF_ORCH_FILTER's orchEmptyLabel, whose eight-combination behaviour
    is exercised for real in dashboard/tests/js/orch_filter.test.mjs.

    The positive anchors in TestOrchTabCurrentFocusRemoved
    (`function OrchTab(`, `aria-label="Task filter"`) are what keep the
    negative assertions here from passing vacuously against a deleted or
    renamed file.
    """

    def test_stringified_filter_expression_removed(self, tabs_jsx_body):
        """The `filter === 'all'` comparison must be gone from tabs.jsx.

        This is the task's delivered_check verbatim (grep pattern
        `filter === 'all'`, expect absent, on this file), asserted in-suite so
        the gate cannot fail after a green local run.
        """
        assert "filter === 'all'" not in tabs_jsx_body

    def test_object_concatenation_removed(self, orch_tab_body):
        """The `filter + ' '` concatenation that produced the literal
        "[object Object] " must be gone, not merely rewritten around."""
        assert not re.search(r"filter\s*\+\s*'", orch_tab_body)

    def test_empty_cell_routes_through_orch_empty_label(self, orch_tab_body):
        """The empty cell must delegate to the tested pure helper rather than
        re-implementing the copy inline — inline copy would leave the
        eight-combination coverage in orch_filter.test.mjs asserting nothing
        about what actually renders."""
        assert re.search(r'orchEmptyLabel\(\s*filter\s*\)', orch_tab_body)

    def test_destructures_orch_filter_global(self, tabs_jsx_body):
        """orchEmptyLabel must be bound from window.DF_ORCH_FILTER (defined by
        orch_filter.js, loaded earlier in index.html) at module top level —
        checked against the whole file since top-level destructures sit outside
        any single tab function (same rationale as test_destructures_runtime_fmt).

        Pins the binding itself, not merely the token: a bare
        `\\bDF_ORCH_FILTER\\b` search would be satisfied by a comment mentioning
        the global or by a dead reference left behind after the destructure was
        deleted. The loose spacing lets the `|| { orchEmptyLabel: ... }`
        availability fallback and any reformatting through.
        """
        assert re.search(
            r'\{\s*orchEmptyLabel\s*\}\s*=\s*window\.DF_ORCH_FILTER', tabs_jsx_body
        )
