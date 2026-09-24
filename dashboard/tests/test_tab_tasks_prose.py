"""Wiring of the Task Detail pane's prose fetch in tab_tasks.jsx (task 5815).

The ACTIVE_TASKS rows no longer carry description/details, so TaskDetail
delegates them to a TaskProse child that fetches them per selection through
data.js's on-demand seam. This file pins only the WIRING, on comment-stripped
code: this repo has no DOM harness (see pins_recovery.js's CANONICAL header),
and the render DECISION is covered behaviourally by data_poll.test.mjs's
onDemandView tests.
"""

from __future__ import annotations

import re

import pytest
from _dashboard_helpers import extract_function_body, strip_js_comments, walk_balanced

_LOAD_SAFETY_MECHANISM = (
    'Babel-standalone 7.29.0 (the build index.html loads, default presets '
    "['react','env']) compiles a JSX file's top-level `const` into a global "
    '`var` with no wrapper function. A `var` sharing its name with a classic '
    "script's top-level `const` throws `SyntaxError: Identifier '...' has "
    'already been declared` at load, and data.js declares ON_DEMAND_KEYS, '
    'REFRESH_OUTCOMES and ON_DEMAND_VIEWS exactly that way. Then none of '
    'tab_tasks.jsx runs, window.DF_TASKS stays undefined and the Tasks tab '
    'blanks. No test here compiles JSX, so this pin is the only guard: bind the '
    'loader as ONE namespace alias and reach every export through it.'
)


@pytest.fixture(scope='module')
def tab_tasks_code(tab_tasks_jsx_body):
    return strip_js_comments(tab_tasks_jsx_body)


@pytest.fixture(scope='module')
def task_detail_code(tab_tasks_code):
    return extract_function_body(tab_tasks_code, 'TaskDetail')


@pytest.fixture(scope='module')
def task_prose_code(tab_tasks_code):
    return extract_function_body(tab_tasks_code, 'TaskProse')


def test_task_detail_delegates_the_prose_to_task_prose(task_detail_code):
    assert not re.search(r'\btask\.(description|details)\b', task_detail_code), (
        'TaskDetail still reads the prose off the list row, which no longer carries it'
    )
    assert re.search(r'<TaskProse\b[^>]*\buid=\{\s*task\.id\s*\}', task_detail_code), (
        'TaskDetail must render <TaskProse uid={task.id} /> for the selected task'
    )


def test_task_prose_requests_only_when_the_selection_changes(task_prose_code):
    effects = [
        walk_balanced(task_prose_code, match.end() - 1, '(', ')')
        for match in re.finditer(r'\buE_T\s*\(', task_prose_code)
    ]
    [request_effect] = [
        effect for effect in effects
        if re.search(r"DF_LOADER_T\.requestOnDemand\(\s*'taskProse'\s*,\s*uid\b", effect)
    ]
    assert re.search(r',\s*\[\s*uid\s*\]\s*\)\Z', request_effect), (
        'the prose request must run in an effect keyed on exactly [uid]: once per '
        f'selection, never on the 3 s poll. Found: {request_effect}'
    )


def test_task_prose_reads_the_value_through_the_rows_key_builder(tab_tasks_code, task_prose_code):
    assert re.search(
        r'DF_LOADER_T\.ON_DEMAND_KEYS\.taskProse\.key\(\s*uid\s*\)', task_prose_code,
    ), 'TaskProse must look the value up under the key the on-demand row builds'
    assert 'TASK_PROSE:' not in tab_tasks_code, (
        'the TASK_PROSE: key is built in data.js only; a literal here can drift from it'
    )


def test_task_prose_branches_on_the_shared_view_decision(task_prose_code):
    assert 'DF_LOADER_T.onDemandView(' in task_prose_code


def test_task_prose_renders_both_fields_as_markdown(task_prose_code):
    for field in ('description', 'details'):
        assert re.search(
            rf'<MarkdownText\b[^>]*\btext=\{{\s*\w+\.{field}\s*\}}', task_prose_code,
        ), f'TaskProse must render the fetched {field} through MarkdownText'


def test_the_data_loader_is_bound_once_as_a_namespace_alias(tab_tasks_code):
    assert len(re.findall(r'\bwindow\.DF_DATA_LOADER\b', tab_tasks_code)) == 1, (
        'window.DF_DATA_LOADER must be read exactly once, by the DF_LOADER_T alias. '
        + _LOAD_SAFETY_MECHANISM
    )
    assert re.search(
        r'^const\s+DF_LOADER_T\s*=\s*window\.DF_DATA_LOADER\s*;', tab_tasks_code, re.M,
    ), 'expected a top-level `const DF_LOADER_T = window.DF_DATA_LOADER;`. ' + _LOAD_SAFETY_MECHANISM
    assert not re.search(r'\}\s*=\s*window\.DF_DATA_LOADER\b', tab_tasks_code), (
        'the data loader must never be destructured. ' + _LOAD_SAFETY_MECHANISM
    )
