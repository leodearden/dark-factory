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
from _dashboard_helpers import extract_function_body, strip_js_comments


@pytest.fixture(scope='module')
def tasks_tab_body(tab_tasks_jsx_body):
    """TasksTab's brace-delimited body, signature excluded.

    Scoped away from the other component functions (TaskGraph, PrdBox,
    TaskDetail, ...) in the same 780-line file so an assertion cannot be
    satisfied by an unrelated component.
    """
    return extract_function_body(tab_tasks_jsx_body, 'TasksTab')


@pytest.fixture(scope='module')
def tasks_tab_code(tasks_tab_body):
    """TasksTab's body with comments blanked — what the assertions run on.

    Every structural assertion below runs against the stripped text, because a
    comment can satisfy or falsify one either way: an absence assertion
    ("``counts.done`` must not appear") is broken by the very comment that
    explains why it must not appear, and a presence assertion can be met by
    prose mentioning the token instead of by code doing it.
    """
    return strip_js_comments(tasks_tab_body)


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
        """The done pip shows a server-MEASURED count or an explicit unknown.

        Never a bounded tally shown as though it were the real number. The
        measured count is ``TASKS_SNAPSHOT[p].census`` now that ``DONE_COUNTS``
        is gone, and ``doneCount`` (task_done_count.js, executed by
        ``dashboard/tests/js/task_done_count.test.mjs``) is what turns an entry
        into the count or datum.js's placeholder. So the header must hand the
        entry to that guard and render its answer, with nothing else in the
        expression.

        The ``_fallbackDone`` / ``'50+'`` branch is gone rather than kept as a
        second route to the pip. It tallied the done rows in ``ACTIVE_TASKS``,
        and the default render fetches none, so it could only ever produce the
        confident "0 done" this guard exists to stop.
        """
        counts_literal = re.search(
            r'const\s+counts\s*=\s*\{(.*?)\};', tasks_tab_code, re.DOTALL,
        )
        assert counts_literal is not None, 'TasksTab no longer builds a `counts` object'
        complete = re.search(r'\bcomplete\s*:\s*([^,}]*)', counts_literal.group(1))
        assert complete is not None, 'the display object has no `complete` count'
        assert re.fullmatch(
            r'doneCount\(\s*DF_T\.TASKS_SNAPSHOT\s*\[\s*p\.id\s*\]\s*\)',
            complete.group(1).strip(),
        ), (
            'counts.complete must be exactly the guard reading this project\'s '
            f'TASKS_SNAPSHOT entry, got: {complete.group(1).strip()!r}'
        )
        assert 'DONE_COUNTS' not in tasks_tab_code, (
            'TasksTab still reads DONE_COUNTS, which /tasks no longer serves'
        )
        assert '_fallbackDone' not in tasks_tab_code, (
            'the bounded done fallback is back: it counts done rows the default '
            'render never fetches, so it can only render a fabricated zero'
        )
        assert not re.search(r"'\d+\+'", tasks_tab_code), (
            "an 'N+' marker is back, and it only ever decorated that fallback"
        )

    def test_bounded_done_tally_is_not_on_the_display_object(self, tasks_tab_code):
        """`counts` must not carry the module's raw `done` alongside `complete`.

        `statusCounts.done` counts only the done rows actually loaded, which the
        default render no longer fetches at all; `counts.complete` is the
        server-measured count, or datum.js's placeholder when there is none.
        Spreading the raw tally onto the display object parks two
        near-synonymous done keys of different trust levels side by side, and
        the next `{counts.done} done` edit silently renders a zero nobody
        measured.
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
        # statusMatches is declared INSIDE TasksTab, so the retired
        # top-level-slice extractor could not isolate it and this assertion was
        # scoped to the `filters.active && (...)` condition by an ad-hoc regex
        # instead. `extract_function_body` scopes to the nested declaration
        # directly (task 3549), which is narrower still and says what it means
        # — and raises loudly if statusMatches is ever renamed, where the regex
        # would have needed its own not-None guard to avoid going quiet.
        status_matches_body = extract_function_body(tasks_tab_code, 'statusMatches')
        for status in ('in-progress', 'blocked', 'merge-deferred'):
            assert f"'{status}'" in status_matches_body, (
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
