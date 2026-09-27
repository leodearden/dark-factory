"""Wiring of the Tasks tab's on-demand terminal window and its PRD-box count.

PRD decisions 5 and 8 (plans/dashboard-one-datum-one-path-prd.md) and task
4416's option (a): terminal rows are never on the default render; the Tasks tab
requests a project's ``?terminal=<project>`` window when the terminal view is
selected or that project is grouped by PRD, and a PRD box reads '≥n/m terminal'
from a lower_bound Datum over the snapshot rows and that window.

This file pins only the WIRING, on comment-stripped served source. The rules
themselves execute under node: which projects request the window in
task_snapshot.test.mjs (terminalWindowProjects), what a requesting caller shows
in data_poll.test.mjs (onDemandDatum), and the PRD count in
prd_grouping.test.mjs (prdProgress, prdProgressReading).
"""

from __future__ import annotations

import re

import pytest
from _dashboard_helpers import (
    extract_function_body,
    find_function_params,
    strip_js_comments,
    walk_balanced,
)


@pytest.fixture(scope='module')
def tab_tasks_code(tab_tasks_jsx_body):
    return strip_js_comments(tab_tasks_jsx_body)


@pytest.fixture(scope='module')
def tasks_tab_code(tab_tasks_code):
    return extract_function_body(tab_tasks_code, 'TasksTab')


@pytest.fixture(scope='module')
def wanted(tasks_tab_code):
    """The name TasksTab binds the requesting projects to; the probes follow it."""
    calls = re.findall(r'\bconst\s+(\w+)\s*=\s*terminalWindowProjects\(', tasks_tab_code)
    assert len(calls) == 1, f'TasksTab binds terminalWindowProjects(...) {len(calls)} times; expected one'
    return calls[0]


def _binding(code: str, name: str) -> str:
    """The right-hand side of ``const <name> = ...;`` in *code*."""
    match = re.search(rf'\bconst\s+{re.escape(name)}\s*=\s*([^;]*);', code)
    assert match, f'`{name}` is not bound by a `const` in TasksTab'
    return match.group(1)


def _view_rows_args(tasks_tab_code: str) -> list[list[str]]:
    calls = []
    for match in re.finditer(r'\bviewRows\(', tasks_tab_code):
        call = walk_balanced(tasks_tab_code, match.end() - 1, '(', ')')
        calls.append([arg.strip() for arg in call[1:-1].split(',')])
    return calls


def test_the_wanted_set_is_terminal_window_projects_over_the_visible_projects(tasks_tab_code, wanted):
    args = re.search(rf'\bconst\s+{wanted}\s*=\s*terminalWindowProjects\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)', tasks_tab_code)
    assert args, f'{wanted} is not terminalWindowProjects(<project ids>, <filter>, <grouped ids>)'
    ids, view_filter, grouped = args.groups()
    assert re.fullmatch(r'projects\.map\(\s*(\w+)\s*=>\s*\1\.id\s*\)', _binding(tasks_tab_code, ids).strip()), (
        f'`{ids}` is not the visible projects\' ids'
    )
    assert re.fullmatch(
        rf"{ids}\.filter\(\s*(\w+)\s*=>\s*groupByPrdMap\[\s*\1\s*\]\s*===\s*'prd'\s*\)",
        _binding(tasks_tab_code, grouped).strip(),
    ), f'`{grouped}` is not the ids whose groupByPrdMap entry is \'prd\''
    assert any(call[2] == view_filter for call in _view_rows_args(tasks_tab_code)), (
        f'`{view_filter}` is not the view filter the listed rows are selected by'
    )


def test_the_window_is_requested_once_per_entry_into_the_wanted_set(tasks_tab_code, wanted):
    effects = [
        walk_balanced(tasks_tab_code, match.end() - 1, '(', ')')
        for match in re.finditer(r'\buE_T\s*\(', tasks_tab_code)
    ]
    requesting = [e for e in effects if re.search(r"DF_LOADER_T\.requestOnDemand\(\s*'terminal'\s*,", e)]
    assert len(requesting) == 1, f'{len(requesting)} effects request the terminal window; expected one'
    [effect] = requesting
    deps = re.search(r',\s*\[\s*(\w+)\s*\]\s*\)\Z', effect)
    assert deps, (
        'the terminal request must run in an effect keyed on ONE derived key of the wanted '
        f'set: once per entry into it, never on the 3 s poll and never only on mount. Found: {effect}'
    )
    assert re.fullmatch(rf'{wanted}\.join\([^)]*\)', _binding(tasks_tab_code, deps.group(1)).strip()), (
        f'the effect key `{deps.group(1)}` is not derived from `{wanted}`'
    )
    assert re.search(r'\.then\(\s*\(?\s*(\w+)\s*\)?\s*=>[^;]*\bset\w+\([^;]*\b\1\b', effect), (
        'the effect does not record the request outcome into state'
    )


def test_a_requesting_project_reads_its_own_outcome_and_the_rest_read_the_unrequested_window(
    tasks_tab_code, tab_tasks_code, wanted,
):
    assert re.search(
        rf"{wanted}\.includes\(\s*(\w+)\s*\)\s*\?\s*DF_LOADER_T\.onDemandDatum\(\s*'terminal'\s*,\s*\1\s*,[^:]*"
        r':\s*unrequestedTerminalRows\(\s*DF_T\[\s*DF_LOADER_T\.ON_DEMAND_KEYS\.terminal\.key\(\s*\1\s*\)\s*\]\s*\)',
        tasks_tab_code,
    ), (
        'a wanted project must read DF_LOADER_T.onDemandDatum(\'terminal\', id, <outcome>), and every '
        'other project unrequestedTerminalRows(DF_T[DF_LOADER_T.ON_DEMAND_KEYS.terminal.key(id)])'
    )
    assert 'TASKS_TERMINAL:' not in tab_tasks_code, (
        'the TASKS_TERMINAL: key is built in data.js only; a literal here can drift from it'
    )


def test_the_flat_view_the_grouped_view_and_the_prd_count_read_one_terminal_datum(tasks_tab_code):
    calls = _view_rows_args(tasks_tab_code)
    assert len(calls) == 2, f'TasksTab calls viewRows {len(calls)} times; expected the listed and the held rows'
    rows, terminal = {call[0] for call in calls}, {call[1] for call in calls}
    assert len(rows) == 1 and len(terminal) == 1, f'the two viewRows calls read different Datums: {calls}'
    [rows], [terminal] = rows, terminal
    assert re.search(rf'\bprdProgress\(\s*{rows}\s*,\s*{terminal}\s*\)', tasks_tab_code), (
        f'the PRD count is not prdProgress({rows}, {terminal}), the Datums the rows are listed from'
    )
    source = re.search(rf'\bconst\s+{terminal}\s*=\s*(\w+)\(\s*p\.id\s*\)', tasks_tab_code)
    assert source, f'`{terminal}` is not read per project through one helper'
    assert 'onDemandDatum(' in _binding(tasks_tab_code, source.group(1)), (
        f'`{source.group(1)}` does not route a wanted project to onDemandDatum'
    )


class TestThePrdBoxCount:
    @pytest.fixture(scope='class')
    def prd_box_body(self, tab_tasks_jsx_body):
        return extract_function_body(tab_tasks_jsx_body, 'PrdBox')

    @pytest.fixture(scope='class')
    def prd_box_code(self, prd_box_body):
        return strip_js_comments(prd_box_body)

    def test_the_count_is_the_progress_datum_read_as_terminal_of_total(self, prd_box_code):
        count = re.search(r'<span\s+className="prd-box-count"[^>]*>(.*?)</span>\s*</div>', prd_box_code, re.DOTALL)
        assert count, 'PrdBox renders no .prd-box-count span'
        assert re.fullmatch(
            r'\s*<DatumReading\s+datum=\{\s*\w+\s*\}\s+format=\{\s*prdProgressReading\([^)]*\)\s*\}\s*/>\s*terminal\s*',
            count.group(1),
        ), f'the PRD count is not <DatumReading datum={{…}} format={{prdProgressReading(…)}} /> terminal: {count.group(1)!r}'
        assert not re.search(r'\.views\.terminal\s*\}\s*/\s*\{', prd_box_code), (
            'the interim n/m tally over the client-held members is still rendered'
        )

    def test_the_undercount_tooltip_is_replaced_by_the_lower_bound(self, prd_box_body):
        """task 4416 option (a): the '≥' disclosure replaces the tooltip.

        Its premise, active_tasks.py's live-PRD exemption from the terminal cap, is gone.
        """
        assert 'countMayUndercount' not in prd_box_body
        assert 'active_tasks.py' not in prd_box_body, (
            'PrdBox still explains a count by active_tasks.py\'s retired exemption'
        )

    @pytest.mark.parametrize('focus_prop', ['focusMode', 'focusAnchorId'])
    def test_the_box_takes_no_focus_state(self, tab_tasks_jsx_body, focus_prop):
        masked, start, end = find_function_params(tab_tasks_jsx_body, 'PrdBox')
        params = tab_tasks_jsx_body[start:end]
        assert params.strip(), 'PrdBox declares no parameters; the absence below would be vacuous'
        assert not re.search(rf'\b{focus_prop}\b', params), f'PrdBox takes {focus_prop}: {params.strip()!r}'


def test_a_selected_terminal_task_opens_in_the_detail_pane(tasks_tab_code, wanted):
    found = re.search(r'\bconst\s+selectedTask\s*=\s*selectedId\s*\?\s*(\w+)\.find\(', tasks_tab_code)
    assert found, 'selectedTask is not looked up with `.find(` over one binding'
    assert re.search(rf'\[\s*\.\.\.allTasks\s*,\s*\.\.\.{wanted}\.flatMap\(', _binding(tasks_tab_code, found.group(1))), (
        f'`{found.group(1)}` does not add the wanted projects\' landed terminal rows to allTasks, so a '
        'selected done or cancelled node could never open in TaskDetail'
    )
