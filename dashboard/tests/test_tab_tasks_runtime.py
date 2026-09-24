"""Wiring tests for the Tasks-tab TaskDetail pane's runtime fields in
tab_tasks.jsx (task 2637, PRD Open-Q3).

Tests parse JSX source as text and assert structural contracts.
Follows the idiom established in test_tab_orchestrators.py / test_index_html.py.
"""

from __future__ import annotations

import re

import pytest
from _dashboard_helpers import extract_function_body, strip_js_comments


@pytest.fixture(scope='module')
def task_detail_body(tab_tasks_jsx_body):
    """TaskDetail's brace-delimited body, signature excluded.

    Scoped away from the other component functions (TaskGraph, PrdBox,
    TasksTab, ...) in the same file so an assertion cannot be satisfied by an
    unrelated component.
    """
    return extract_function_body(tab_tasks_jsx_body, 'TaskDetail')


@pytest.fixture(scope='module')
def task_graph_body(tab_tasks_jsx_body):
    r"""TaskGraph's brace-delimited body, signature excluded.

    Scoped away from the other component functions in the same file.  Note
    `function TaskGraphEdges(` is declared EARLIER in tab_tasks.jsx: only the
    extractor's trailing `\s*\(` keeps it from answering this request.
    """
    return extract_function_body(tab_tasks_jsx_body, 'TaskGraph')


@pytest.fixture(scope='module')
def fmt_age_body(tab_tasks_jsx_body):
    """fmtAge's brace-delimited body, signature excluded.

    Scoped away from the other functions in the same file.
    """
    return extract_function_body(tab_tasks_jsx_body, 'fmtAge')


@pytest.fixture(scope='module')
def tasks_tab_body(tab_tasks_jsx_body):
    """TasksTab's brace-delimited body, signature excluded.

    Scoped away from the other component functions (TaskGraph, PrdBox,
    TaskDetail, ...) in the same file so an assertion cannot be satisfied by
    an unrelated component.  The file-local extractor this used to call was
    promoted to ``_dashboard_helpers.extract_function_body``, whose brace walk
    runs over a masked copy of the source and stops at TasksTab's matching
    ``}`` — so the old caveat that this fixture ran to EOF no longer holds,
    and absence assertions over it are now genuinely scoped.
    """
    return extract_function_body(tab_tasks_jsx_body, 'TasksTab')


@pytest.fixture(scope='module')
def tasks_tab_code(tasks_tab_body):
    """TasksTab's body with comments blanked — what absence assertions run on.

    Same rationale as ``test_tab_tasks_status_counts.py``: an absence
    assertion run on raw source is broken by the very comment that explains
    the absence, and a presence assertion can be satisfied by prose naming the
    token instead of by code doing it.  That trap is latent rather than live
    here — today's rationale comment happens not to contain the forbidden
    tokens — which is exactly why it is worth closing now.
    """
    return strip_js_comments(tasks_tab_body)


def _balanced_slice(code: str, anchor: str, opener: str, closer: str) -> str:
    """The balanced *opener*..*closer* run that follows *anchor* in *code*.

    Brace/paren balance rather than a regex, so a nested literal cannot
    truncate the slice and leave an assertion over it passing vacuously.
    *code* must already be comment-stripped — a brace inside a comment would
    otherwise miscount.  Raises on a miss rather than returning '' for the
    same reason: an empty slice makes every `not in` assertion permanently
    green.
    """
    start = code.find(anchor)
    assert start != -1, f'{anchor!r} not found — the assertion below would be vacuous'
    open_at = code.index(opener, start)
    depth = 0
    for i in range(open_at, len(code)):
        if code[i] == opener:
            depth += 1
        elif code[i] == closer:
            depth -= 1
            if depth == 0:
                return code[open_at:i + 1]
    raise AssertionError(f'unbalanced {opener}{closer} after {anchor!r}')


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

    def test_both_banners_exist_and_are_distinct(self, tasks_tab_code):
        """A starved-dashboard probe failure is NOT a fused-memory outage —
        the two banners answer different questions and must not be merged.

        Re-pointed at the shape that exists now.  The fused-memory banner's
        test id is no longer written as a ``data-testid="..."`` literal at
        all: the Tasks-tab banners were refactored into a notice list, so the
        id lives in the ``bannerTestIds`` kind -> testid map and reaches the
        DOM via ``data-testid={bannerTestIds[notice.kind]}``.  The property
        being pinned is unchanged and still the whole point — the two banners
        must stay SEPARATELY ADDRESSABLE, because an operator triaging a
        fused-memory outage must still be able to see, and tell apart, a
        runtime-probe verdict.
        """
        testids = _balanced_slice(tasks_tab_code, 'const bannerTestIds', '{', '}')
        assert "'tasks-offline-banner'" in testids, (
            'the fused-memory banner must keep its own test id in bannerTestIds'
        )
        assert 'data-testid="tasks-runtime-probe-banner"' in tasks_tab_code, (
            'the probe banner must keep its own directly-rendered test id'
        )
        assert 'tasks-runtime-probe-banner' not in testids, (
            'the probe banner must not be addressed through the offline-banner '
            'map — see test_probe_banner_is_not_a_tasksbannernotices_kind'
        )

    def test_probe_banner_is_not_gated_on_tasks_offline(self, tasks_tab_code):
        """The probe banner must render on its own condition.

        Gating it on fused-memory's availability hides a runtime-probe failure
        at exactly the moment an operator is already triaging — and the two
        facts are independent: fused-memory being down says nothing about
        whether we could reach the orchestrators.

        RE-ARMED.  This originally asserted the guard did not mention
        ``tasksOffline``.  That binding was deleted when the banners became a
        notice list, which left the assertion unfalsifiable by ANY
        implementation — a guard that pinned a real property had gone vacuous
        without failing.  It now names the bindings that carry that fact
        today, ``bannerNotices`` and ``DF_T.TASKS_OFFLINE``, and keeps
        ``tasksOffline`` so a revival of the old name is caught too.
        """
        match = re.search(
            r'\{\s*([A-Za-z0-9_.&!? ]*?)\s*&&\s*\(\s*\n?\s*<div[^>]*data-testid="tasks-runtime-probe-banner"',
            tasks_tab_code,
        )
        assert match is not None, 'probe banner should render under its own && guard'
        guard = match.group(1)
        for forbidden in ('bannerNotices', 'DF_T.TASKS_OFFLINE', 'tasksOffline'):
            assert forbidden not in guard, (
                f'probe banner must not be gated on {forbidden}; guard was {guard!r}'
            )

    def test_probe_banner_is_not_a_tasksbannernotices_kind(self, tasks_tab_code):
        """The probe verdict must NOT be routed through ``tasksBannerNotices``.

        Not a style preference — a code fact.  ``tasks_offline_banner.js``
        short-circuits: when its ``offline`` input is true it returns a SINGLE
        global notice and never evaluates the other kinds.  A probe notice
        folded into that pipeline would therefore be silently SUPPRESSED
        during precisely the fused-memory outage the test above forbids gating
        on — the same failure, reintroduced through the back door and no
        longer visible to that test's guard-string check.

        That module also scopes itself to a distinction decided SERVER-SIDE in
        app.api_tasks, whereas the probe summary is by design a client-side
        derivation over the ACTIVE_TASKS rows.  So the probe banner stays an
        independent sibling of ``bannerNotices.map(...)``.
        """
        testids = _balanced_slice(tasks_tab_code, 'const bannerTestIds', '{', '}')
        assert 'tasks-runtime-probe-banner' not in testids, (
            'the probe banner must not become a bannerTestIds kind'
        )
        assert not re.search(r"kind:\s*'[a-z-]*probe[a-z-]*'", tasks_tab_code), (
            'no probe-flavoured notice kind may be introduced into the '
            'tasksBannerNotices vocabulary'
        )
        call_args = _balanced_slice(tasks_tab_code, 'tasksBannerNotices(', '(', ')')
        assert 'probe' not in call_args.lower(), (
            'the probe fact must not be fed into tasksBannerNotices; that '
            f'function short-circuits on offline. Call was {call_args!r}'
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

    def test_summary_denominator_is_not_scoped_by_project_filter(
        self, tasks_tab_body, tasks_tab_code,
    ):
        """The banner is a GLOBAL fact, deliberately not narrowable.

        This reverses an earlier decision to scope the summary to the
        projectFilter-visible rows, so the why matters. `selfInflicted` is a
        claim about the DASHBOARD's own health — it asserts the dashboard
        finished none of the probes it started — so its denominator has to be
        every probed project. Scoping the rows to `projectFilter` lets an
        operator who has narrowed to two timed-out projects MANUFACTURE a
        spurious all-at-once verdict, inventing the exact dashboard-blaming
        misdiagnosis this task exists to prevent.

        The listing being global too is the accepted cost. A banner naming a
        filtered-out project is strictly less harmful than a false "the
        orchestrators may be healthy — check the dashboard first" shown while
        an operator is triaging a real per-project outage.

        The two ABSENCE assertions run on the comment-stripped body: the
        rationale comment that explains why the scoped form must not return is
        the single most likely place for its text to appear, so asserting over
        raw source would make this test fail on the documentation of its own
        requirement.  The positive assertion stays on the raw body — it is
        satisfiable only by the call itself.
        """
        assert re.search(r'rtProbeSummary\(\s*allTasks\s*\)', tasks_tab_body), (
            'summary must be computed over the UNFILTERED row set (allTasks)'
        )
        assert 'rtProbeSummary(allTasks.filter(' not in tasks_tab_code, (
            'the projectFilter-scoped form must not come back — it lets a '
            'narrowed view manufacture a false selfInflicted verdict'
        )
        assert 'visibleProjectIds' not in tasks_tab_code, (
            'visibleProjectIds existed only to scope the summary; a lingering '
            'binding invites the scoped form to be silently reintroduced'
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
