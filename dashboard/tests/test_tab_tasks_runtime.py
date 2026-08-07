"""Wiring tests for the Tasks-tab TaskDetail pane's runtime fields in
tab_tasks.jsx (task 2637, PRD Open-Q3).

Tests parse JSX source as text and assert structural contracts.
Follows the idiom established in test_tab_orchestrators.py / test_index_html.py.
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
def tab_tasks_jsx_body(_client):
    return _client.get('/static/redux/tab_tasks.jsx').text


def _extract_function_body(source: str, func_name: str) -> str:
    """Extract the source text of a top-level ``function <func_name>(`` up to
    (but not including) the next top-level ``function`` declaration, or to
    end-of-file if it's the last one.

    Used to scope structural assertions to a single component's render
    function so an assertion can't be accidentally satisfied by an unrelated
    component elsewhere in the same file (mirrors
    test_tab_orchestrators.py's identically-named helper).
    """
    start_match = re.search(rf'^function {re.escape(func_name)}\(', source, re.MULTILINE)
    assert start_match is not None, f'function {func_name}( not found in tab_tasks.jsx'
    start = start_match.start()
    next_match = re.search(r'^function \w+\(', source[start + 1 :], re.MULTILINE)
    end = start + 1 + next_match.start() if next_match else len(source)
    return source[start:end]


@pytest.fixture(scope='module')
def task_detail_body(tab_tasks_jsx_body):
    """TaskDetail's own source text, scoped away from the other component
    functions (TaskGraph, PrdBox, TasksTab, ...) in the same file."""
    return _extract_function_body(tab_tasks_jsx_body, 'TaskDetail')


@pytest.fixture(scope='module')
def task_graph_body(tab_tasks_jsx_body):
    """TaskGraph's own source text, scoped away from the other component
    functions in the same file."""
    return _extract_function_body(tab_tasks_jsx_body, 'TaskGraph')


@pytest.fixture(scope='module')
def fmt_age_body(tab_tasks_jsx_body):
    """fmtAge's own source text, scoped away from the other functions in the
    same file."""
    return _extract_function_body(tab_tasks_jsx_body, 'fmtAge')


@pytest.fixture(scope='module')
def tasks_tab_body(tab_tasks_jsx_body):
    """TasksTab's own source text.

    CAVEAT: TasksTab is the LAST top-level function in tab_tasks.jsx, so
    ``_extract_function_body`` returns everything from its declaration to EOF.
    That still excludes TaskDetail / TaskGraph / PrdBox (better than
    whole-file scoping), but do NOT use this fixture to prove a string is
    ABSENT from other components — it cannot.
    """
    return _extract_function_body(tab_tasks_jsx_body, 'TasksTab')


class TestTaskDetailRuntimeFields:
    """TaskDetail must mirror OrchTab's offline-'—' degradation for
    loops/attempts, and additionally surface lane/phase/lane_state (task
    2637, PRD Open-Q3 — droppable, but trivial given TaskDetail's existing
    k/v grid already renders loops/attempts)."""

    def test_tab_tasks_jsx_served(self, _client):
        resp = _client.get('/static/redux/tab_tasks.jsx')
        assert resp.status_code == 200

    def test_task_detail_positive_anchor_function(self, tab_tasks_jsx_body):
        """File must still export TaskDetail — guards against a renamed/empty file."""
        assert 'function TaskDetail(' in tab_tasks_jsx_body

    def test_destructures_runtime_fmt(self, tab_tasks_jsx_body):
        """window.DF_RUNTIME_FMT (defined by runtime_format.js, loaded earlier
        in index.html) must be destructured at module top level — checked
        against the whole file since top-level destructures sit outside any
        single component function."""
        assert re.search(r'\bDF_RUNTIME_FMT\b', tab_tasks_jsx_body)

    def test_loops_value_routes_through_rtcell(self, task_detail_body):
        assert re.search(r'rtCell\(\s*task\.loops', task_detail_body)

    def test_attempts_value_routes_through_rtcell(self, task_detail_body):
        assert re.search(r'rtCell\(\s*task\.attempts', task_detail_body)

    def test_lane_row_routes_through_rtcell(self, task_detail_body):
        assert re.search(r'rtCell\(\s*task\.lane\b', task_detail_body)

    def test_phase_row_routes_through_rtcell(self, task_detail_body):
        assert re.search(r'rtCell\(\s*task\.phase\b', task_detail_body)

    def test_lane_state_row_routes_through_rtcell(self, task_detail_body):
        assert re.search(r'rtCell\(\s*task\.lane_state\b', task_detail_body)

    def test_lane_kv_label_present(self, task_detail_body):
        assert re.search(r'className="k">lane<', task_detail_body)

    def test_phase_kv_label_present(self, task_detail_body):
        assert re.search(r'className="k">phase<', task_detail_body)

    def test_state_kv_label_present(self, task_detail_body):
        assert re.search(r'className="k">state<', task_detail_body)


class TestRuntimeFmtDestructureIncludesRtAge:
    """window.DF_RUNTIME_FMT must also expose rtAge at the top-level
    destructure (task 2699) — TaskDetail already relied on rtCell alone, but
    the graph-node meta and fmtAge() both need rtAge's null -> em-dash
    degradation for the `started` field."""

    def test_destructure_includes_rtage(self, tab_tasks_jsx_body):
        # Tolerates additional names after rtAge (task 3517 adds rtProbe /
        # rtProbeSummary) while still pinning that BOTH rtCell and rtAge are
        # destructured — the contract this test exists for.
        assert re.search(
            r'const\s*\{\s*rtCell\s*,\s*rtAge\s*[,}][^=]*=\s*window\.DF_RUNTIME_FMT',
            tab_tasks_jsx_body,
        )


class TestRuntimeFmtDestructureIncludesProbeHelpers:
    """window.DF_RUNTIME_FMT must also expose the probe-status helpers at the
    top-level destructure (task 3517). Without them the Tasks tab renders all
    three probe failure modes as identical blank cells, which is exactly the
    2026-07-30 misdiagnosis."""

    def test_destructure_includes_rtprobe(self, tab_tasks_jsx_body):
        assert re.search(
            r'const\s*\{[^}]*\brtProbe\b[^}]*\}\s*=\s*window\.DF_RUNTIME_FMT',
            tab_tasks_jsx_body,
        )

    def test_destructure_includes_rtprobesummary(self, tab_tasks_jsx_body):
        assert re.search(
            r'const\s*\{[^}]*\brtProbeSummary\b[^}]*\}\s*=\s*window\.DF_RUNTIME_FMT',
            tab_tasks_jsx_body,
        )


class TestTaskDetailSurfacesProbeStatus:
    """TaskDetail must say WHY loops/attempts/lane/phase are dashed, not just
    dash them (task 3517)."""

    def test_routes_runtime_status_through_rtprobe(self, task_detail_body):
        assert re.search(r'rtProbe\(\s*task\.runtime_status', task_detail_body)

    def test_runtime_kv_label_present(self, task_detail_body):
        assert re.search(r'className="k">runtime<', task_detail_body)

    def test_stable_testid_hook_present(self, task_detail_body):
        assert 'data-testid="task-runtime-probe"' in task_detail_body

    def test_badge_tone_comes_from_the_descriptor(self, task_detail_body):
        """The colour must follow the descriptor's `tone`, not be hard-coded —
        otherwise 'no runtime endpoint' (expected) would look as alarming as
        'orchestrator unreachable' (a real fault)."""
        assert re.search(r'badge\s*\$\{\s*\w+\.tone\s*\}|badge\s*\'\s*\+\s*\w+\.tone', task_detail_body)


class TestTasksTabRuntimeProbeBanner:
    """TasksTab must surface an aggregate probe banner, distinct from the
    existing fused-memory offline banner (task 3517)."""

    def test_calls_rtprobesummary(self, tasks_tab_body):
        assert 'rtProbeSummary(' in tasks_tab_body

    def test_banner_testid_present(self, tasks_tab_body):
        assert 'data-testid="tasks-runtime-probe-banner"' in tasks_tab_body

    def test_both_banners_exist_and_are_distinct(self, tab_tasks_jsx_body):
        """A starved-dashboard probe failure is NOT a fused-memory outage —
        the two banners answer different questions and must not be merged."""
        assert 'data-testid="tasks-offline-banner"' in tab_tasks_jsx_body
        assert 'data-testid="tasks-runtime-probe-banner"' in tab_tasks_jsx_body

    def test_probe_banner_is_not_gated_on_tasks_offline(self, tasks_tab_body):
        """The probe banner must render on its own condition — gating it on
        tasksOffline would hide a runtime-probe failure whenever fused-memory
        happens to be up, which is the common case."""
        match = re.search(
            r'\{\s*([A-Za-z0-9_.&!? ]*?)\s*&&\s*\(\s*\n?\s*<div[^>]*data-testid="tasks-runtime-probe-banner"',
            tasks_tab_body,
        )
        assert match is not None, 'probe banner should render under its own && guard'
        assert 'tasksOffline' not in match.group(1), (
            f'probe banner must not be gated on tasksOffline; guard was {match.group(1)!r}'
        )

    def test_banner_accent_derives_from_tone_not_selfinflicted(self, tasks_tab_body):
        """Colour must track WHAT is wrong, not whether we blamed ourselves.

        Keying the accent off `selfInflicted` painted a lone timed-out project
        — tone 'warn', quite possibly our own starved loop — in the same alarm
        colour as a confirmed orchestrator outage, and painted a purely
        expected `not_configured` set red. runtime_format.js owns the
        status -> tone map; this file only translates a tone to a CSS var.
        """
        assert 'PROBE_TONE_ACCENT_T[probeSummary.tone]' in tasks_tab_body
        assert "probeSummary.selfInflicted ? 'var(--warn)' : 'var(--bad)'" not in tasks_tab_body

    def test_summary_is_scoped_to_the_visible_projects(self, tasks_tab_body):
        """An operator filtered down to one healthy project is not looking at
        the rows the banner alarms about; a banner that narrowing the view
        cannot dismiss is just noise."""
        assert 'rtProbeSummary(allTasks)' not in tasks_tab_body, (
            'summary must be computed over the projectFilter-visible rows, not allTasks'
        )
        assert re.search(r'rtProbeSummary\(\s*allTasks\.filter\(', tasks_tab_body), (
            'expected rtProbeSummary over a filtered row set'
        )


class TestGraphNodeAgeRoutesThroughRtAge:
    """The task-graph node's in-progress runtime-age badge (task 2699) must
    route `t.started` through rtAge rather than interpolating it directly
    into a template literal — a raw `${t.started}m` renders the literal
    string "nullm" when started is null (offline snapshot)."""

    def test_started_not_raw_interpolated(self, task_graph_body):
        assert '${t.started}m' not in task_graph_body

    def test_started_routes_through_rtage(self, task_graph_body):
        assert re.search(r'rtAge\(\s*t\.started\s*\)', task_graph_body)


class TestFmtAgeRoutesThroughRtAge:
    """fmtAge()'s final "<n>m running" fallback (task 2699) must route
    `t.started` through rtAge and must not append " running" to a null
    (em-dash) age — that would render the nonsensical "— running"."""

    def test_started_not_raw_interpolated(self, fmt_age_body):
        assert '${t.started}m running' not in fmt_age_body

    def test_started_routes_through_rtage(self, fmt_age_body):
        assert re.search(r'rtAge\(\s*t\.started\s*\)', fmt_age_body)

    def test_null_age_does_not_get_running_suffix(self, fmt_age_body):
        assert '— running' not in fmt_age_body
