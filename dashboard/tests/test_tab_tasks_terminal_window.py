"""Wiring of the Tasks tab's on-demand terminal window.

PRD decision 5 (plans/dashboard-one-datum-one-path-prd.md) and task 4416's
option (a): terminal rows are never on the default render; the Tasks tab
requests a project's ``?terminal=<project>`` window only when the terminal view
is selected (plans/dashboard-prd-lanes-prd.md §4.4: the by-PRD lanes read
/prds, not the window).

This file pins only the WIRING, on comment-stripped served source. The rules
themselves execute under node: which projects request the window in
task_snapshot.test.mjs (terminalWindowProjects), and what a requesting caller
shows in data_on_demand.test.mjs (onDemandDatum).
"""

from __future__ import annotations

import re

import pytest
from _jsx_source_helpers import (
    extract_function_body,
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
    args = re.search(rf'\bconst\s+{wanted}\s*=\s*terminalWindowProjects\(\s*(\w+)\s*,\s*(\w+)\s*\)', tasks_tab_code)
    assert args, f'{wanted} is not terminalWindowProjects(<project ids>, <filter>)'
    ids, view_filter = args.groups()
    assert re.fullmatch(r'projects\.map\(\s*(\w+)\s*=>\s*\1\.id\s*\)', _binding(tasks_tab_code, ids).strip()), (
        f'`{ids}` is not the visible projects\' ids'
    )
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


def test_a_project_already_in_the_wanted_set_is_not_requested_again(tasks_tab_code, wanted):
    """Only the entrants are requested; which ids those are executes in task_snapshot.test.mjs."""
    [effect] = [
        e for e in (
            walk_balanced(tasks_tab_code, match.end() - 1, '(', ')')
            for match in re.finditer(r'\buE_T\s*\(', tasks_tab_code)
        )
        if "requestOnDemand('terminal'" in e
    ]
    entrants = re.search(rf'\bconst\s+(\w+)\s*=\s*terminalWindowEntrants\(\s*(\w+)\.current\s*,\s*{wanted}\s*\)', effect)
    assert entrants, (
        f'the effect does not compute terminalWindowEntrants(<ref>.current, {wanted}): every key change '
        're-requests the window of each project already wanted'
    )
    arrived, before = entrants.groups()
    assert re.search(rf'\bconst\s+{before}\s*=\s*uR_T\(', tasks_tab_code), f'`{before}` is not a ref TasksTab holds'
    assert re.search(rf'\b{before}\.current\s*=\s*{wanted}\s*;', effect), (
        f'the effect never records {wanted} as `{before}.current`, so the next run cannot tell who entered'
    )
    loop = re.search(r'\bfor\s*\(\s*const\s+(\w+)\s+of\s+(\w+)\s*\)', effect)
    assert loop and loop.group(2) == arrived, f'the request loop does not iterate `{arrived}`: {effect}'
    assert re.search(rf"\b{arrived}\.map\(\s*(\w+)\s*=>\s*\[\s*\1\s*,\s*null\s*\]", effect), (
        f'only the entrants\' outcomes are reset to null; resetting `{wanted}` blanks settled ones'
    )
    assert not re.search(r'\breturn\s*\(\s*\)\s*=>', effect), (
        'a cleanup that drops pending outcomes loses a still-wanted project\'s: the next run '
        'does not request it again'
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


def test_the_listed_and_the_held_rows_of_the_flat_view_read_one_terminal_datum(tasks_tab_code):
    calls = _view_rows_args(tasks_tab_code)
    assert len(calls) == 2, f'TasksTab calls viewRows {len(calls)} times; expected the listed and the held rows'
    rows, terminal = {call[0] for call in calls}, {call[1] for call in calls}
    assert len(rows) == 1 and len(terminal) == 1, f'the two viewRows calls read different Datums: {calls}'
    [terminal] = terminal
    source = re.search(rf'\bconst\s+{terminal}\s*=\s*(\w+)\(\s*p\.id\s*\)', tasks_tab_code)
    assert source, f'`{terminal}` is not read per project through one helper'
    assert 'onDemandDatum(' in _binding(tasks_tab_code, source.group(1)), (
        f'`{source.group(1)}` does not route a wanted project to onDemandDatum'
    )


def test_a_selected_terminal_task_opens_in_the_detail_pane(tasks_tab_code, wanted):
    found = re.search(r'\bconst\s+selectedTask\s*=\s*selectedId\s*\?\s*(\w+)\.find\(', tasks_tab_code)
    assert found, 'selectedTask is not looked up with `.find(` over one binding'
    assert re.search(rf'\[\s*\.\.\.allTasks\s*,\s*\.\.\.{wanted}\.flatMap\(', _binding(tasks_tab_code, found.group(1))), (
        f'`{found.group(1)}` does not add the wanted projects\' landed terminal rows to allTasks, so a '
        'selected done or cancelled node could never open in TaskDetail'
    )
