"""Wiring tests for the Tasks-tab per-project header counts in tab_tasks.jsx
(task 3516).

The header used to render ONE merged "N active" pip over
{in-progress, blocked, merge-deferred}. Only the in-progress component is
bounded by max_concurrent_tasks, so the merged number routinely exceeded the
configured cap and read as a cap breach (2026-07-30: dark-factory showed
"43 active" against a cap of 24; reify "50 active" against 48).

The counting itself is pure and lives in task_status_counts.js, covered
executably by dashboard/tests/js/task_status_counts.test.mjs. What THAT suite
cannot see is the JSX wiring — whether tab_tasks.jsx actually calls it, and
whether the header still renders the merged pip alongside. That is what this
module asserts.

Deliberately a NEW module rather than an extension of test_tab_tasks_runtime.py,
whose docstring scopes it to TaskDetail's runtime fields (task 2637, PRD
Open-Q3): header counts do not belong there, and a sibling display-defect task
edits the same JSX file, so keeping this surface separate leaves only
tab_tasks.jsx itself as the shared merge surface.

Tests parse JSX source as text and assert structural contracts.
Follows the idiom established in test_tab_tasks_runtime.py /
test_tab_orchestrators.py / test_index_html.py.
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
    component elsewhere in the same 780-line file (mirrors
    test_tab_tasks_runtime.py's identically-named helper).
    """
    start_match = re.search(rf'^function {re.escape(func_name)}\(', source, re.MULTILINE)
    assert start_match is not None, f'function {func_name}( not found in tab_tasks.jsx'
    start = start_match.start()
    next_match = re.search(r'^function \w+\(', source[start + 1 :], re.MULTILINE)
    end = start + 1 + next_match.start() if next_match else len(source)
    return source[start:end]


@pytest.fixture(scope='module')
def tasks_tab_body(tab_tasks_jsx_body):
    """TasksTab's own source text, scoped away from the other component
    functions (TaskGraph, PrdBox, TaskDetail, ...) in the same file."""
    return _extract_function_body(tab_tasks_jsx_body, 'TasksTab')


def _strip_comments(source: str) -> str:
    """Drop ``/* … */`` blocks and ``// …`` line comments from JSX source.

    Every structural assertion below runs against the stripped text, because a
    comment can satisfy or falsify one either way: an absence assertion
    ("``counts.done`` must not appear") is broken by the very comment that
    explains why it must not appear, and a presence assertion can be met by
    prose mentioning the token instead of by code doing it. Stripping first
    keeps this module's contract — assert structure, never comment text.

    The ``(?<!:)`` guard leaves ``https://…`` inside a string literal alone.
    """
    without_blocks = re.sub(r'/\*.*?\*/', '', source, flags=re.DOTALL)
    return re.sub(r'(?<!:)//.*', '', without_blocks)


@pytest.fixture(scope='module')
def tasks_tab_code(tasks_tab_body):
    """TasksTab's source with comments stripped — what the assertions run on."""
    return _strip_comments(tasks_tab_body)


class TestHeaderCountsWiring:
    """The per-project header must source its counts from
    task_status_counts.js and render the three in-flight numbers separately."""

    def test_tab_tasks_jsx_served(self, _client):
        resp = _client.get('/static/redux/tab_tasks.jsx')
        assert resp.status_code == 200

    def test_destructures_task_status_counts_at_top_level(self, tab_tasks_jsx_body):
        """tab_tasks.jsx must destructure window.DF_TASK_STATUS_COUNTS.

        Whole-file scope on purpose: top-level destructures sit outside every
        component function, so scoping this to TasksTab would never match.
        Both names are required — projectStatusCounts does the counting,
        activityPips fixes the render order and the zero-suppression rule.
        """
        match = re.search(
            r'const\s*\{([^}]*)\}\s*=\s*window\.DF_TASK_STATUS_COUNTS\s*;',
            tab_tasks_jsx_body,
        )
        assert match is not None, (
            'tab_tasks.jsx does not destructure window.DF_TASK_STATUS_COUNTS at '
            'top level — the header counting module is registered in index.html '
            'but unused.'
        )
        names = {n.strip() for n in match.group(1).split(',') if n.strip()}
        assert 'projectStatusCounts' in names
        assert 'activityPips' in names

    def test_merged_active_pip_is_gone(self, tasks_tab_code):
        """The single merged "N active" pip must no longer be rendered.

        This is the defect itself: one number over
        {in-progress, blocked, merge-deferred}, compared by operators against
        a cap that only bounds the in-progress part of it.
        """
        assert 'counts.active' not in tasks_tab_code, (
            'TasksTab still references counts.active — the merged pip that '
            'reads as a max_concurrent_tasks breach must be split into '
            'separate running / blocked / merge-deferred counts.'
        )
        assert not re.search(r'\}\s*active\s*<', tasks_tab_code), (
            'TasksTab still renders a merged "... active" pip label.'
        )

    def test_header_counts_come_from_the_pure_module(self, tasks_tab_code):
        """TasksTab must call projectStatusCounts/activityPips rather than
        re-inlining status filters in the component.

        Re-inlined `t.status === ...` passes in the header are exactly what
        the node suite cannot cover, so they could drift back to a merged
        tally without a single test going red.
        """
        # Deliberately `\w+` and not the literal local names `projTasks` /
        # `counts`: what must hold is that the module is CALLED, not that a
        # particular local identifier survives. Pinning the identifier would
        # turn a zero-behaviour rename red with a message claiming the header
        # counts no longer come from the pure module.
        assert re.search(r'projectStatusCounts\(\s*\w+\s*\)', tasks_tab_code), (
            'TasksTab does not call projectStatusCounts(...) — the header '
            'counts must come from the executably-tested pure module.'
        )
        assert re.search(r'activityPips\(\s*\w+\s*\)', tasks_tab_code), (
            'TasksTab does not call activityPips(...) — pip order and '
            'zero-suppression must come from the pure module, not a JSX ternary.'
        )
        # The header's own counting passes must be gone. `filtered` (which
        # uses statusMatches/searchMatches, not a bare status comparison) is
        # unaffected; this pins that no hand-rolled status tally remains
        # beside the module call.
        inflight_filters = re.findall(
            r"t\.status\s*===\s*'(?:in-progress|blocked|merge-deferred|pending|done)'",
            tasks_tab_code,
        )
        assert inflight_filters == [], (
            'TasksTab still hand-rolls status filters for the header counts: '
            f'{inflight_filters} — these belong in projectStatusCounts.'
        )

    def test_renders_the_three_counts_separately(self, tasks_tab_code):
        """The header must render one pip per activityPips entry.

        Mapping over the returned entries (rather than three hardcoded pips)
        is what makes the zero-suppression rule — and the never-empty
        "0 running" fallback — actually reach the browser.
        """
        assert re.search(r'activityPips\(\s*\w+\s*\)\s*\.map\(', tasks_tab_code), (
            'TasksTab does not map over activityPips(...) to render one pip '
            'per in-flight status.'
        )
        assert re.search(r'key=\{\s*\w+\.key\s*\}', tasks_tab_code), (
            'The rendered activity pips are not keyed by their entry key.'
        )

    def test_every_activity_pip_key_has_a_dot_colour(self, tab_tasks_jsx_body):
        """The module/caller seam: activityPips owns the pip KEYS, the JSX owns
        the dot COLOUR per key. Nothing else asserts the two agree.

        That split is deliberate (the pure module must stay free of DOM and
        colour concerns), but it means a fourth pip added to activityPips — or
        a key renamed to match the counts spelling, e.g. 'merge-deferred' ->
        'mergeDeferred' — renders `background: undefined`: an invisible dot
        beside a number, with both suites still green. The node suite only
        sees the pure module; every other test here only sees the JSX.

        Source of truth for this key set is the `all` list in
        task_status_counts.js's activityPips, pinned there by
        dashboard/tests/js/task_status_counts.test.mjs.
        """
        match = re.search(
            r'const\s+PIP_DOT_COLOR_T\s*=\s*\{(.*?)\}\s*;',
            tab_tasks_jsx_body,
            re.DOTALL,
        )
        assert match is not None, (
            'tab_tasks.jsx no longer declares a PIP_DOT_COLOR_T map — the '
            'activity pips have no per-key dot colour.'
        )
        keys = set(re.findall(r"^\s*'?([\w-]+)'?\s*:", match.group(1), re.MULTILINE))
        assert keys == {'running', 'blocked', 'merge-deferred'}, (
            f'PIP_DOT_COLOR_T covers {sorted(keys)}, but activityPips emits '
            "keys {'running', 'blocked', 'merge-deferred'}. A key activityPips "
            'can emit with no entry here renders style={{background: undefined}} '
            '— an invisible dot next to a number. Keep the two in step (see the '
            "`all` list in task_status_counts.js's activityPips)."
        )


class TestUnchangedHeaderElements:
    """Everything else about the header is out of scope and must survive."""

    def test_pending_and_done_pips_still_rendered(self, tasks_tab_code):
        assert re.search(r'\{\s*counts\.pending\s*\}\s*pending', tasks_tab_code), (
            'the pending pip is no longer rendered'
        )
        assert re.search(r'\{\s*counts\.complete\s*\}\s*done', tasks_tab_code), (
            'the done pip is no longer rendered'
        )

    def test_shown_count_label_still_rendered(self, tasks_tab_code):
        assert re.search(
            r'\{\s*filtered\.length\s*\}\s*/\s*\{\s*counts\.total\s*\}\s*shown',
            tasks_tab_code,
        ), 'the "n/m shown" mono label is no longer rendered'

    def test_server_authoritative_done_count_preserved(self, tasks_tab_code):
        """DONE_COUNTS is the authoritative full count; the bounded fallback
        marks itself as a lower bound. Both must survive the rewiring — the
        pure module returns a raw `done` tally and must not have replaced this
        branch.

        Asserted on SHAPE, not on the literal page size: this test is about
        provenance (authoritative vs. lower bound), so changing the ACTIVE_TASKS
        page size from 50 must not turn it red.
        """
        assert 'DF_T.DONE_COUNTS' in tasks_tab_code, (
            'the server-authoritative DONE_COUNTS branch is gone'
        )
        assert re.search(r'_fallbackDone\s*>=\s*\d+', tasks_tab_code), (
            'the lower-bound page-size test for the done fallback is gone'
        )
        assert re.search(r"'\d+\+'", tasks_tab_code), (
            "the 'N+' lower-bound marker for the bounded done fallback is gone "
            '— without it the header shows a capped tally as if it were the '
            'authoritative count'
        )

    def test_bounded_done_tally_is_not_on_the_display_object(self, tasks_tab_code):
        """`counts` must not carry the module's raw `done` alongside `complete`.

        `statusCounts.done` counts only the done rows actually loaded (bounded
        per project); `counts.complete` is the authoritative server count, or an
        explicit lower-bound marker. Spreading the raw tally onto the display
        object parks two near-synonymous done keys of different trust levels
        side by side, and the next `{counts.done} done` edit silently renders
        the capped number. It stays reachable only via `_fallbackDone`, whose
        underscore says "not for display".
        """
        assert not re.search(r'\.\.\.\s*statusCounts', tasks_tab_code), (
            'TasksTab spreads statusCounts into the display object — pick the '
            'display keys explicitly so the bounded `done` tally never lands '
            'on `counts` beside the authoritative `complete`.'
        )
        assert 'counts.done' not in tasks_tab_code, (
            'TasksTab references counts.done — that is the BOUNDED tally, not '
            'the authoritative count; render counts.complete instead.'
        )


class TestActiveFilterStillMerged:
    """OUT-OF-SCOPE GUARD. Splitting the `active` FILTER toggle is a separate
    UX decision the task explicitly excludes. Without a positive guard, the
    next reader sees a split display next to an unsplit filter, reads it as an
    oversight, and "finishes the job"."""

    def test_status_matches_keeps_the_three_status_disjunction(self, tasks_tab_code):
        # statusMatches is declared INSIDE TasksTab, so it is not a top-level
        # `function` _extract_function_body can isolate. Scope to the
        # `filters.active && (...)` condition itself instead, which is both
        # narrower and exactly the thing that must not change.
        match = re.search(r'filters\.active\s*&&\s*\(([^)]*)\)', tasks_tab_code)
        assert match is not None, (
            "statusMatches no longer has a `filters.active && (...)` condition."
        )
        active_condition = match.group(1)
        for status in ('in-progress', 'blocked', 'merge-deferred'):
            assert f"'{status}'" in active_condition, (
                f'statusMatches no longer treats {status!r} as active — the '
                'filter toggle deliberately keeps its merged three-status '
                'meaning (task 3516 splits the DISPLAY only). The node suite '
                "pins that the split display's three counts still sum to this "
                'same population.'
            )

    def test_active_filter_button_still_present(self, tasks_tab_code):
        assert re.search(r"flipFilter\(\s*'active'\s*\)", tasks_tab_code), (
            'the `active` filter button is gone — the filter split is out of '
            'scope for task 3516.'
        )
