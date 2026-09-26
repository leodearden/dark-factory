"""Wiring tests for the Orchestrators tab UI in tabs.jsx.

Tests parse JSX source as text and assert structural contracts.
Follows the idiom established in test_tab_curator.py / test_index_html.py.
"""

from __future__ import annotations

import re

import pytest
from _dashboard_helpers import extract_function_body, strip_js_comments, walk_balanced


@pytest.fixture(scope='module')
def orch_tab_body(tabs_jsx_body):
    """OrchTab's brace-delimited body, signature excluded.

    Scoped away from the other tab-render functions in the same file so an
    assertion here cannot be accidentally satisfied by an unrelated tab (e.g.
    ReconTab's "Recent runs" table also has a "Status" column header).
    """
    return extract_function_body(tabs_jsx_body, 'OrchTab')


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


class TestOrchTabHealthState:
    """OrchTab must render *offline* and *degraded* as DISTINCT states.

    These pin WIRING only. The behavioural coverage of the split lives where
    the facts are shaped (``dashboard/tests/test_redux_api.py``), because this
    repo renders no JSX under test — what can be asserted here is that the tab
    reads each field and paints them differently. Nothing has PRODUCED either
    flag since task 5587 made discovery a read-free ``ps`` scan. Leaf γ2 (task
    5589) KEPT the pips rather than deleting them: the PRD Contract keeps
    ``offline``/``error`` on /orchestrators entries, so deleting the reader
    while the producer still projects the fields would split one decision
    across two leaves. Per-project COUNT health reaches OrchTab separately, as
    the census Datum's state and reason (TestOrchTabReadsTheCensus).

    The label and colour assertions are LINE-SCOPED (``[^\n]*``) and therefore
    assume each pip stays a single-line JSX expression, which is how the rest
    of OrchTab's summary fragment is written. A reformatter that wraps them
    must update these regexes rather than delete them. They also pin the exact
    operator-facing copy, so a pure wording edit reddens them with no change in
    behaviour — the same trade this file already takes with ``>Lane<`` and
    ``>Phase<``.

    The positive anchors in TestOrchTabCurrentFocusRemoved
    (``function OrchTab(``, ``aria-label="Task filter"``) are what keep the one
    negative assertion below from passing vacuously against a deleted or
    renamed file.
    """

    def test_offline_pip_branches_on_offline(self, orch_tab_body):
        """The tab reads o.offline — until this task it consumed neither field."""
        assert re.search(r'o\.offline\s*&&', orch_tab_body)

    def test_degraded_pip_branches_on_degraded(self, orch_tab_body):
        """The tab reads o.degraded, so a starved root is visible at all."""
        assert re.search(r'o\.degraded\s*&&', orch_tab_body)

    def test_offline_suppresses_the_degraded_pip(self, orch_tab_body):
        """A root that is proven down renders one pip, not two contradictory ones.

        The guard is the tab's one piece of defensive logic: shape_orchestrators
        bool()-coerces whatever raw entry it is handed, so the wire can in
        principle carry both flags set, and the operator must then read the
        stronger, PROVEN fact. Nothing else in this class covers it — dropping
        the ``!o.offline &&`` prefix leaves every other assertion here green,
        because the branch test above matches ``o.degraded &&`` either way and
        the label and colour regexes are line-scoped.
        """
        assert re.search(r'!\s*o\.offline\s*&&\s*o\.degraded', orch_tab_body)

    def test_degraded_label_is_distinct_from_offline(self, orch_tab_body):
        """The two states must not read as the same sentence to an operator.

        Proven-down and not-measured call for different actions, so they get
        different words, not merely different colours. The two literals below
        ARE the discrimination — comparing them to each other would compare two
        constants of this test and say nothing about the tab.
        """
        assert re.search(r'o\.offline[^\n]*>offline<', orch_tab_body)
        assert re.search(r'o\.degraded[^\n]*>state unknown<', orch_tab_body)

    def test_degraded_pip_does_not_render_in_the_offline_colour(self, orch_tab_body):
        """A degraded root must never be painted as proven-down.

        Negative and line-scoped, which survives reformatting of the colour
        expression in a way a positive colour regex would not.
        """
        assert not re.search(r'o\.degraded[^\n]*CP\.bad', orch_tab_body)

    def test_offline_pip_renders_in_the_bad_colour_and_degraded_in_warn(self, orch_tab_body):
        """The positive half: the palette entries are the ones intended."""
        assert re.search(r'o\.offline[^\n]*CP\.bad', orch_tab_body)
        assert re.search(r'o\.degraded[^\n]*CP\.warn', orch_tab_body)


@pytest.fixture(scope='module')
def orch_tab_code(tabs_jsx_body):
    """OrchTab's body with comments stripped, so prose satisfies no probe."""
    return extract_function_body(strip_js_comments(tabs_jsx_body), 'OrchTab')


def _map_calls(code: str, table: str) -> list[tuple[str, str]]:
    """Every ``<table>.map(p => ...)`` call: (parameter name, balanced call text)."""
    calls = []
    for match in re.finditer(rf'\b{table}\.map\(\s*(\w+)\s*=>', code):
        paren = code.index('(', match.start())
        calls.append((match.group(1), walk_balanced(code, paren, '(', ')')))
    return calls


def _the_map_rendering(code: str, table: str, marker: str) -> tuple[str, str]:
    calls = [(p, text) for p, text in _map_calls(code, table) if marker in text]
    assert len(calls) == 1, (
        f'OrchTab has {len(calls)} `{table}.map(...)` call(s) rendering `{marker}`; '
        'expected exactly one.'
    )
    return calls[0]


class TestOrchTabReadsTheCensus:
    """OrchTab renders every count from the served census (leaf γ2, task 5589).

    The reported defect: the Progress card read ``orchSummary(o)`` zeros with
    ``total || 1`` ("0/1") while the filter bar counted ACTIVE_TASKS rows
    ("Active · 33"). Every count now comes from ONE census Datum per project —
    ``projectCensus(DF, o.project)`` — through the shared Pip / StatTile /
    DatumReading with a named reading from task_snapshot.js, whose behaviour
    over the boundary sketches is executed in dashboard/tests/js/task_snapshot.test.mjs.
    These pins prove the wiring reaches it.
    """

    @pytest.fixture(scope='class')
    def tabs_code(self, tabs_jsx_body):
        return strip_js_comments(tabs_jsx_body)

    @pytest.mark.parametrize(
        'retired',
        [
            'DF_ORCH_SUMMARY',
            'orchSummary',
            'hasOrchSummary',
            'orchTotalDatum',
            'DF_TASK_DONE_COUNT',
            'doneCount',
            'ACTIVE_TASKS',
            'o.summary',
            '|| 1',
        ],
    )
    def test_the_interim_readers_are_gone(self, tabs_code, retired):
        assert retired not in tabs_code, (
            f'tabs.jsx still contains `{retired}`. OrchTab reads the census now; '
            'the interim guards and the ACTIVE_TASKS row count were the two '
            'populations behind "0/1" beside "Active · 33".'
        )

    def test_destructures_the_census_reader_without_fallback(self, tabs_code):
        assert re.search(r'=\s*window\.DF_TASK_SNAPSHOT\s*;', tabs_code), (
            'tabs.jsx does not destructure window.DF_TASK_SNAPSHOT at module scope.'
        )
        assert not re.search(r'window\.DF_TASK_SNAPSHOT\s*(\|\||&&|\?\?)', tabs_code)

    def test_binds_one_census_per_orchestrator(self, orch_tab_code):
        assert re.search(r'\bconst\s+census\s*=\s*projectCensus\(\s*DF\s*,\s*o\.project\s*\)', orch_tab_code), (
            'OrchTab does not bind `const census = projectCensus(DF, o.project)`.'
        )

    def test_every_pip_reads_the_census_through_a_view_reading(self, orch_tab_code):
        pips = re.findall(r'<Pip\b', orch_tab_code)
        assert len(pips) == 1, f'OrchTab renders {len(pips)} <Pip> sites; expected one, mapped over CENSUS_VIEWS'
        view, call = _the_map_rendering(orch_tab_code, 'CENSUS_VIEWS', '<Pip')
        assert 'datum={census}' in call
        assert f'format={{{view}.reading}}' in call

    def test_the_progress_header_is_terminal_of_total(self, orch_tab_code):
        assert re.search(r'<DatumReading\s+datum=\{census\}\s+format=\{terminalOfTotal\}', orch_tab_code), (
            'the Progress header does not render <DatumReading datum={census} format={terminalOfTotal} />.'
        )

    def test_the_progress_bar_maps_census_segments(self, orch_tab_code):
        assert re.search(r'censusSegments\(\s*census\s*\)\.map\(', orch_tab_code)

    def test_the_legend_maps_the_views_through_datum_reading(self, orch_tab_code):
        calls = [
            (view, call)
            for view, call in _map_calls(orch_tab_code, 'CENSUS_VIEWS')
            if '<DatumReading' in call and '<button' not in call
        ]
        assert len(calls) == 1, f'expected one legend map over CENSUS_VIEWS, found {len(calls)}'
        view, call = calls[0]
        assert re.search(rf'<DatumReading\s+datum=\{{census\}}\s+format=\{{{view}\.reading\}}', call)

    def test_the_filter_buttons_are_the_views_with_census_counts(self, orch_tab_code):
        view, call = _the_map_rendering(orch_tab_code, 'CENSUS_VIEWS', '<button')
        assert re.search(rf'flipFilter\(\s*o\.pid\s*,\s*{view}\.key\s*\)', call)
        assert re.search(rf'<DatumReading\s+datum=\{{census\}}\s+format=\{{{view}\.count\}}', call)
        assert f'{{{view}.label}}' in call

    def test_the_filter_persists_under_the_view_key(self, orch_tab_code):
        assert "'df.orch.views'" in orch_tab_code
        assert 'df.orch.filter' not in orch_tab_code, (
            'a browser holding the old {active,pending,complete} object would read '
            'as "nothing selected" under the new keys; the new storage key makes it '
            'fall back to the default instead.'
        )

    @pytest.mark.parametrize('retired', ['>Active', '>Pending ·', '>Complete ·', 'label="active"'])
    def test_the_active_label_is_retired(self, orch_tab_code, retired):
        assert retired not in orch_tab_code

    def test_the_task_tiles_are_one_mapped_member_tile(self, orch_tab_code):
        scope = re.search(r'\bconst\s+(\w+)\s*=\s*censusOver\(\s*DF\s*,', orch_tab_code)
        assert scope, 'OrchTab does not bind a scope census from censusOver(DF, ...)'
        tile, call = _the_map_rendering(orch_tab_code, 'CENSUS_TILES', '<ST')
        assert f'datum={{{scope.group(1)}}}' in call
        assert f'format={{{tile}.reading}}' in call
        assert 'history={censusHistory(' in call
        assert len(re.findall(r'<ST\b', orch_tab_code)) == 2, (
            'OrchTab keeps the Orchestrators tile plus ONE tile mapped over CENSUS_TILES'
        )

    def test_the_rows_come_from_the_snapshot_by_view(self, orch_tab_code):
        assert re.search(
            r'viewRows\(\s*projectRows\(\s*DF\s*,\s*o\.project\s*\)\s*,\s*'
            r'datumFor\(\s*ON_DEMAND_KEYS\.terminal\.key\(\s*o\.project\s*\)\s*\)',
            orch_tab_code,
        ), 'OrchTab rows do not come from viewRows(projectRows(DF, o.project), datumFor(ON_DEMAND_KEYS.terminal.key(o.project)), ...)'
        assert 'requestOnDemand' not in orch_tab_code, (
            'requesting the terminal window is leaf γ3\'s; γ2 only reads the datum.'
        )

    def test_the_placeholder_row_renders_the_reasoned_hole(self, orch_tab_code):
        assert re.search(r'title=\{\s*placeholder\.title\s*\}', orch_tab_code)
        assert re.search(r'\{\s*placeholder\.text\s*\}', orch_tab_code)
