"""Wiring tests for the Tasks tab's per-project header and view filter in
tab_tasks.jsx.

Task 3516 split the header's merged "N active" pip, because only its running
part is bounded by max_concurrent_tasks and the merged number read as a cap
breach. The census meets that goal directly now: the in-flight pip reads
"N running of M in-flight", the capped sub-view shown WITH its superset rather
than merged into it (PRD decision 3, plans/dashboard-one-datum-one-path-prd.md).

Every header count comes from ONE census Datum per project,
``projectCensus(DF_T, p.id)``, rendered through the shared ``<Pip>`` with a
named reading from task_snapshot.js, and the rows come from the same snapshot
by view (``viewRows``). The readings and the row selection execute in
dashboard/tests/js/task_snapshot.test.mjs. What that suite cannot see is the
JSX wiring, which these pins assert on comment-stripped source: this repo has
no DOM harness (see pins_recovery.js's CANONICAL header).

Two guards ride along because the census wiring created their hazard: every
TaskStatus member now reaches the graph, so each needs a node style; and the
module-scope destructures this wiring added must not collide with a classic
script's top-level declaration (esc-5590-1).
"""

from __future__ import annotations

import re

import pytest
from _dashboard_helpers import destructure_bindings, extract_function_body, strip_js_comments, walk_balanced

from shared.task_statuses import TaskStatus


@pytest.fixture(scope='module')
def tab_tasks_code(tab_tasks_jsx_body):
    return strip_js_comments(tab_tasks_jsx_body)


@pytest.fixture(scope='module')
def tasks_tab_code(tab_tasks_code):
    """TasksTab's body with comments blanked, so prose satisfies no probe."""
    return extract_function_body(tab_tasks_code, 'TasksTab')


@pytest.fixture(scope='module')
def snapshot_names(tab_tasks_code):
    """canonical -> local for tab_tasks.jsx's destructure of DF_TASK_SNAPSHOT.

    The probes follow these bindings rather than pin their spelling.
    """
    match = re.search(r'^const\s*\{([^{}]*)\}\s*=\s*window\.DF_TASK_SNAPSHOT\s*;', tab_tasks_code, re.M)
    assert match, 'tab_tasks.jsx does not destructure window.DF_TASK_SNAPSHOT at module scope'
    return dict(destructure_bindings(match.group(1)))


@pytest.fixture(scope='module')
def census(tasks_tab_code):
    bindings = re.findall(r'\bconst\s+(\w+)\s*=\s*projectCensus\(\s*DF_T\s*,\s*p\.id\s*\)', tasks_tab_code)
    assert len(bindings) == 1, (
        f'TasksTab binds projectCensus(DF_T, p.id) {len(bindings)} times; expected one binding '
        'that every header pip of the project reads.'
    )
    return bindings[0]


def _map_calls(code: str, table: str) -> list[tuple[str, str]]:
    """Every ``<table>.map(p => ...)`` call: (parameter name, balanced call text)."""
    calls = []
    for match in re.finditer(rf'\b{re.escape(table)}\.map\(\s*(\w+)\s*=>', code):
        paren = code.index('(', match.start())
        calls.append((match.group(1), walk_balanced(code, paren, '(', ')')))
    return calls


def _the_map_rendering(code: str, table: str, marker: str) -> tuple[str, str]:
    calls = [(p, text) for p, text in _map_calls(code, table) if marker in text]
    assert len(calls) == 1, (
        f'TasksTab has {len(calls)} `{table}.map(...)` call(s) rendering `{marker}`; expected exactly one.'
    )
    return calls[0]


def _view_rows_bindings(tasks_tab_code: str) -> list[tuple[str, str, str]]:
    """Every ``const X = viewRows(...)``: (name, argument text, suffix after the call)."""
    bindings = []
    for match in re.finditer(r'\bconst\s+(\w+)\s*=\s*viewRows\(', tasks_tab_code):
        call = walk_balanced(tasks_tab_code, match.end() - 1, '(', ')')
        after = tasks_tab_code[match.end() - 1 + len(call):]
        suffix = re.match(r'\s*(\.\w+)?', after).group(1) or ''
        bindings.append((match.group(1), call[1:-1], suffix))
    return bindings


class TestTheRetiredCountersAreGone:
    @pytest.mark.parametrize(
        'retired',
        [
            'DF_TASK_STATUS_COUNTS', 'DF_TASK_DONE_COUNT', 'projectStatusCounts', 'activityPips',
            'doneCount', 'PIP_DOT_COLOR_T', 'ACTIVE_TASKS', 'DONE_COUNTS', '_fallbackDone', 'statusMatches',
        ],
    )
    def test_no_client_bucketer_remains(self, tab_tasks_code, retired):
        assert not re.search(rf'\b{re.escape(retired)}\b', tab_tasks_code), (
            f'tab_tasks.jsx still contains `{retired}`. The header reads the served census '
            'and the rows come from the snapshot by view; no client module re-counts '
            'or re-buckets a status any more.'
        )


class TestHeaderReadsTheCensus:
    def test_destructures_the_census_reader_without_fallback(self, tab_tasks_code, snapshot_names):
        assert not re.search(r'window\.DF_TASK_SNAPSHOT\s*(\|\||&&|\?\?)', tab_tasks_code)
        for name in ('projectCensus', 'projectRows', 'viewRows', 'CENSUS_VIEWS', 'EVERY_VIEW'):
            assert name in snapshot_names, f'tab_tasks.jsx does not take {name} from window.DF_TASK_SNAPSHOT'

    def test_binds_one_census_per_project(self, tasks_tab_code, census):
        calls = re.findall(r'\bprojectCensus\(', tasks_tab_code)
        assert len(calls) == 1, (
            f'TasksTab calls projectCensus {len(calls)} times; every pip of one project reads '
            f'the single `{census}` binding.'
        )

    def test_the_header_is_one_pip_mapped_over_the_views(self, tasks_tab_code, snapshot_names, census):
        pips = re.findall(r'<Pip\b', tasks_tab_code)
        assert len(pips) == 1, f'TasksTab renders {len(pips)} <Pip> sites; expected one, mapped over CENSUS_VIEWS'
        view, call = _the_map_rendering(tasks_tab_code, snapshot_names['CENSUS_VIEWS'], '<Pip')
        assert f'datum={{{census}}}' in call, f'the header pip does not read the census binding: {call}'
        assert f'format={{{view}.reading}}' in call, f'the header pip does not use the view reading: {call}'
        assert re.search(rf'color=\{{\s*CP_T\[\s*{view}\.tone\s*\]\s*\}}', call), (
            f'the header pip is not coloured by its view tone through PALETTE: {call}'
        )

    def test_no_status_is_hand_bucketed(self, tasks_tab_code):
        assert not re.search(r'\.status\s*[!=]==', tasks_tab_code), (
            'TasksTab compares a status by hand. The header and the filter read the '
            'generated views; a hand-written status list drifts from them.'
        )


class TestRowsComeFromTheSnapshotByView:
    def test_the_rows_are_the_projects_snapshot_rows_selected_by_view(self, tasks_tab_code):
        rows = re.findall(r'\bconst\s+(\w+)\s*=\s*projectRows\(\s*DF_T\s*,\s*p\.id\s*\)', tasks_tab_code)
        assert len(rows) == 1, 'TasksTab does not bind the project rows from projectRows(DF_T, p.id)'
        bindings = _view_rows_bindings(tasks_tab_code)
        assert bindings, 'TasksTab selects no rows through viewRows(...)'
        for name, args, _ in bindings:
            assert re.match(rf'\s*{rows[0]}\s*,\s*\w+\s*,\s*\w+\s*$', args), (
                f'`{name}` is not viewRows({rows[0]}, <terminal>, <filter>): viewRows({args})'
            )

    def test_the_filter_bar_is_the_census_views(self, tasks_tab_code, snapshot_names):
        view, call = _the_map_rendering(tasks_tab_code, snapshot_names['CENSUS_VIEWS'], '<button')
        assert len(re.findall(r'<button\b', call)) == 1
        assert re.search(rf'flipFilter\(\s*{view}\.key\s*\)', call), f'the view button does not flip its view: {call}'
        assert f'{{{view}.label}}' in call, f'the view button does not show the view label: {call}'
        assert not re.search(r"flipFilter\(\s*['\"]", tasks_tab_code), (
            'a filter button still flips a hand-named status key'
        )

    def test_the_filter_does_not_persist_under_the_retired_key(self, tasks_tab_code):
        setter = re.search(r'\bconst\s+flipFilter\s*=\s*\(?\s*\w+\s*\)?\s*=>\s*(\w+)\(', tasks_tab_code)
        assert setter, 'TasksTab has no `const flipFilter = key => setX(...)`'
        state = re.search(
            rf'\bconst\s*\[\s*\w+\s*,\s*{setter.group(1)}\s*\]\s*=\s*tasksPersistedState\(\s*[\'"]([^\'"]+)[\'"]',
            tasks_tab_code,
        )
        assert state, f'the filter setter {setter.group(1)} is not a tasksPersistedState'
        assert state.group(1) != 'df.tasksFilters', (
            'a browser holding the old {active, pending, ...} object would read as "nothing '
            'selected" under the view keys; a fresh storage key makes it fall back to the default.'
        )

    def test_the_shown_label_counts_the_held_rows(self, tasks_tab_code, snapshot_names):
        """ "n/m shown": m is every row the tab holds for the project, not a census number.

        The numerator's provenance is test_tab_tasks_focus_header.py's.
        """
        every = snapshot_names['EVERY_VIEW']
        held = [name for name, args, suffix in _view_rows_bindings(tasks_tab_code)
                if re.search(rf',\s*{every}\s*$', args) and suffix == '.rows']
        assert len(held) == 1, f'TasksTab does not bind viewRows(..., {every}).rows once'
        assert re.search(rf'\.shownCount\s*\}}\s*/\s*\{{\s*{held[0]}\.length\s*\}}\s*shown', tasks_tab_code), (
            f'the "n/m shown" label does not read {held[0]}.length as its denominator'
        )

    def test_a_hole_and_a_partial_listing_render_their_reasons(self, tasks_tab_code):
        listed = [name for name, _, suffix in _view_rows_bindings(tasks_tab_code) if not suffix]
        assert len(listed) == 1, 'TasksTab does not bind the listed viewRows(...) result once'
        assert re.search(rf'title=\{{\s*{listed[0]}\.placeholder\.title\s*\}}', tasks_tab_code), (
            'the viewRows placeholder does not carry its reason as a title'
        )
        assert re.search(rf'\{{\s*{listed[0]}\.placeholder\.text\s*\}}', tasks_tab_code)
        assert re.search(rf'\b{listed[0]}\.notes\.map\(', tasks_tab_code), (
            'the viewRows notes (why listed rows are not the whole, current set) are not rendered'
        )

    def test_a_hole_with_nothing_listed_does_not_also_say_nothing_matches(self, tasks_tab_code):
        """OrchTab's ``!placeholder && filtered.length === 0`` precedent (esc-5590-2).

        Under a hole the placeholder is the whole answer; the graph's "no tasks
        match the current filter" would pass the outage off as an empty filter.
        Listed rows still draw: the gate closes only when nothing is listed.
        """
        listed = next(name for name, _, suffix in _view_rows_bindings(tasks_tab_code) if not suffix)
        filtered = re.search(rf'\bconst\s+(\w+)\s*=\s*{listed}\.rows\.filter\(', tasks_tab_code).group(1)
        gate = re.search(
            rf'\bconst\s+(\w+)\s*=\s*{listed}\.placeholder\s*&&\s*{filtered}\.length\s*===\s*0\s*;', tasks_tab_code,
        )
        body = gate and re.search(rf'\{{\s*!\s*{gate.group(1)}\s*&&\s*\(', tasks_tab_code)
        drawn = body and walk_balanced(tasks_tab_code, body.end() - 1, '(', ')')
        assert drawn and '<TaskGraph' in drawn and '<ProjectPrdGroups' in drawn, (
            f'TasksTab does not bind `const X = {listed}.placeholder && {filtered}.length === 0;` '
            'and gate both graph bodies behind `{!X && (...)}`'
        )


def test_every_status_member_has_a_node_style(_client):
    """Every TaskStatus member reaches the graph now, so each must be drawable.

    The status filter used to drop review and infra-hold rows; the views list
    them, and a node with no rule renders its status pip in no colour at all.
    """
    css = _client.get('/static/redux/styles.css').text
    for member in TaskStatus:
        assert re.search(
            rf'\.taskgraph\s+\.node\.s-{re.escape(member.value)}\s+\.status-pip\s*\{{[^}}]*\bbackground\s*:',
            css,
        ), f'styles.css has no `.taskgraph .node.s-{member.value} .status-pip` background rule'


_TOP_LEVEL_BINDING_RE = re.compile(r'^(?:const|let|var|function|class)\s+(\w+)', re.M)
_TOP_LEVEL_DESTRUCTURE_RE = re.compile(r'^(?:const|let|var)\s*\{([^{}]*)\}\s*=', re.M)
_CLASSIC_LEXICAL_RE = re.compile(r'^(?:const|let|class)\s+(\w+)', re.M)
_CLASSIC_DESTRUCTURE_RE = re.compile(r'^(?:const|let)\s*\{([^{}]*)\}\s*=', re.M)


def _destructured_locals(code: str, pattern: re.Pattern[str]) -> set[str]:
    return {local for body in pattern.findall(code) for _, local in destructure_bindings(body)}


def test_no_top_level_binding_collides_with_a_classic_script(_client, index_html_body, tab_tasks_code):
    """Babel turns tab_tasks.jsx's top-level bindings into globals; none may shadow a classic lexical one.

    A same-named top-level `const`/`let`/`class` in a classic script makes the
    whole of tab_tasks.jsx fail to load, so window.DF_TASKS stays undefined and
    the Tasks tab renders nothing (esc-5590-1). The mechanism is stated at
    test_tab_tasks_prose.py::_LOAD_SAFETY_MECHANISM. A classic `function` is
    safe to rebind, which is why tab_tasks.jsx binds function exports under
    their own names and renames every const export it takes.
    """
    jsx_names = set(_TOP_LEVEL_BINDING_RE.findall(tab_tasks_code))
    jsx_names |= _destructured_locals(tab_tasks_code, _TOP_LEVEL_DESTRUCTURE_RE)

    classic_scripts = re.findall(r'<script\s+src="/static/redux/([\w-]+\.js)\?v=\d+"', index_html_body)
    assert 'task_snapshot.js' in classic_scripts, 'the classic-script discovery found nothing to compare against'
    collisions = {}
    for script in classic_scripts:
        code = strip_js_comments(_client.get(f'/static/redux/{script}').text)
        lexical = set(_CLASSIC_LEXICAL_RE.findall(code)) | _destructured_locals(code, _CLASSIC_DESTRUCTURE_RE)
        for name in sorted(jsx_names & lexical):
            collisions[name] = script
    assert not collisions, (
        f'tab_tasks.jsx binds {collisions} at top level, which the named classic scripts '
        'declare as const/let/class. Rename the binding in the destructure (`{ X: X_T }`).'
    )
