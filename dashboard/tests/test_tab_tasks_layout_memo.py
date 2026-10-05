"""Wiring tests for TaskGraph's layout memos in tab_tasks.jsx (task 5116).

TaskGraph's callers hand it a fresh ``tasks`` array on every render, so memos
keyed on that reference recomputed the whole Sugiyama layout every time. The
pure functions it now relies on (``layoutSignature``, ``taskGraphLayout``) are
covered executably by dashboard/tests/js/graph_layout.test.mjs. This module
asserts only that TaskGraph and its edge overlay, TaskGraphEdges, wire them: no
JSX render harness exists, so the wiring is reachable only as source structure.
"""

from __future__ import annotations

import re

import pytest
from _dashboard_helpers import (
    destructure_bindings,
    extract_function_body,
    strip_js_comments,
    walk_balanced,
)

_DEP_ARRAY_AT_CALL_END = re.compile(r'\[([^\[\]]*)\]\s*\)$')


@pytest.fixture(scope='module')
def task_graph_code(tab_tasks_jsx_body):
    """TaskGraph's body with comments blanked — what every assertion runs on."""
    return strip_js_comments(extract_function_body(tab_tasks_jsx_body, 'TaskGraph'))


@pytest.fixture(scope='module')
def task_graph_edges_code(tab_tasks_jsx_body):
    """TaskGraphEdges' body with comments blanked."""
    return strip_js_comments(extract_function_body(tab_tasks_jsx_body, 'TaskGraphEdges'))


@pytest.fixture(scope='module')
def layout_key(task_graph_code):
    """The name TaskGraph binds ``layoutSignature(tasks)`` to."""
    return _layout_key(task_graph_code, 'TaskGraph')


def _layout_key(code: str, component: str) -> str:
    """The name *component* binds ``layoutSignature(tasks)`` to."""
    match = re.search(r'const\s+(\w+)\s*=\s*layoutSignature\(\s*tasks\s*\)', code)
    assert match is not None, (
        f'{component} does not assign `const <key> = layoutSignature(tasks)` — '
        'without that content key its hooks can only be keyed on the `tasks` '
        'reference, which is new on every render, or on a second hand-rolled '
        'encoding of the same id/status/deps content.'
    )
    return match.group(1)


def _hook_calls(code: str, hook: str) -> list[str]:
    """Every full ``<hook>(...)`` call in *code*, parens balanced."""
    calls = []
    for match in re.finditer(rf'\b{hook}\(', code):
        call = walk_balanced(code, match.end() - 1, '(', ')')
        assert call, f'Unbalanced {hook}( call at offset {match.start()}.'
        calls.append(call)
    return calls


def _dep_array(call: str) -> str:
    """The inside of the trailing dependency array of a ``uM_T(...)`` call."""
    match = _DEP_ARRAY_AT_CALL_END.search(call)
    assert match is not None, f'uM_T call has no trailing dependency array: {call!r}'
    return match.group(1)


def _dep_names(call: str) -> set[str]:
    """The entries of the trailing dependency array of a ``uM_T(...)`` call."""
    return {name.strip() for name in _dep_array(call).split(',') if name.strip()}


class TestTaskGraphLayoutMemo:
    def test_tab_tasks_jsx_served(self, _client):
        assert _client.get('/static/redux/tab_tasks.jsx').status_code == 200

    def test_destructures_signature_and_layout_at_top_level(self, tab_tasks_jsx_body):
        match = re.search(
            r'const\s*\{([^{}]*)\}\s*=\s*window\.DF_GRAPH_LAYOUT\s*;',
            strip_js_comments(tab_tasks_jsx_body),
        )
        assert match is not None, 'tab_tasks.jsx has no top-level DF_GRAPH_LAYOUT destructure.'
        locals_bound = {local for _canonical, local in destructure_bindings(match.group(1))}
        missing = {'layoutSignature', 'taskGraphLayout'} - locals_bound
        assert not missing, (
            f'tab_tasks.jsx destructures {sorted(locals_bound)} from DF_GRAPH_LAYOUT '
            f'but not {sorted(missing)}.'
        )

    def test_layout_memo_is_keyed_on_the_signature_only(self, task_graph_code, layout_key):
        pattern = (
            r'uM_T\(\s*\(\)\s*=>\s*taskGraphLayout\(\s*tasks\s*\)\s*,\s*\[\s*'
            + re.escape(layout_key)
            + r'\s*\]\s*\)'
        )
        assert re.search(pattern, task_graph_code), (
            f'TaskGraph does not memoize `taskGraphLayout(tasks)` on exactly '
            f'[{layout_key}] — the layout must recompute only when its content '
            f'key changes.'
        )

    def test_neighborhood_memo_is_keyed_on_signature_and_selection(
        self, task_graph_code, layout_key
    ):
        match = re.search(
            r'\buM_T\(\s*\(\)\s*=>\s*computeNeighborhood\(\s*tasks\s*,\s*selectedId\s*\)',
            task_graph_code,
        )
        assert match is not None, (
            'TaskGraph has no `uM_T(() => computeNeighborhood(tasks, selectedId) ...)` memo.'
        )
        call = walk_balanced(task_graph_code, task_graph_code.index('(', match.start()), '(', ')')
        deps = _dep_names(call)
        assert deps == {layout_key, 'selectedId'}, (
            f'The neighborhood memo depends on {sorted(deps)}; it must depend on '
            f'exactly {sorted({layout_key, "selectedId"})}.'
        )

    def test_no_memo_is_keyed_on_the_tasks_reference(self, task_graph_code):
        calls = _hook_calls(task_graph_code, 'uM_T')
        assert calls, 'TaskGraph contains no uM_T( call — this test would pass vacuously.'
        keyed_on_tasks = [call for call in calls if re.search(r'\btasks\b', _dep_array(call))]
        assert not keyed_on_tasks, (
            'TaskGraph memoizes on the `tasks` reference, which callers rebuild '
            f'on every render, so the memo never hits: {keyed_on_tasks}'
        )

    @pytest.mark.parametrize('layout_fn', ['computeTiers', 'partitionComponents', 'orderRows'])
    def test_layout_functions_run_only_inside_task_graph_layout(
        self, task_graph_code, layout_fn
    ):
        assert not re.search(rf'\b{layout_fn}\s*\(', task_graph_code), (
            f'TaskGraph calls {layout_fn}( directly — it must run only inside '
            f'taskGraphLayout, so the signature-keyed memo is the only thing '
            f'that can trigger it.'
        )

    def test_nodes_render_current_task_objects_not_layout_output(self, task_graph_code):
        assert '.map(renderNode)' not in task_graph_code, (
            'TaskGraph applies renderNode straight to layout output. The layout '
            'holds ids, and the nodes must render the CURRENT task objects, or '
            'title/started/stranded freeze across polls.'
        )
        assert re.search(r'renderNode\(\s*\w+\.get\(', task_graph_code), (
            'TaskGraph never calls renderNode on a by-id lookup — the layout ids '
            'must be resolved against the current `tasks` to render.'
        )


class TestTaskGraphEdgesKey:
    def test_edge_effect_is_keyed_on_the_layout_signature_and_selection(
        self, task_graph_edges_code
    ):
        key = _layout_key(task_graph_edges_code, 'TaskGraphEdges')
        effects = _hook_calls(task_graph_edges_code, 'uLE_T')
        assert len(effects) == 1, f'TaskGraphEdges should hold one uLE_T( effect, found {len(effects)}.'
        deps = _dep_names(effects[0])
        missing = {key, 'selectedId'} - deps
        assert not missing, (
            f'The edge effect depends on {sorted(deps)} but not {sorted(missing)} — '
            'edges must redraw when the layout content or the selection changes.'
        )

    def test_no_edge_hook_is_keyed_on_the_tasks_reference(self, task_graph_edges_code):
        calls = _hook_calls(task_graph_edges_code, 'uM_T') + _hook_calls(
            task_graph_edges_code, 'uLE_T'
        )
        assert calls, 'TaskGraphEdges contains no uM_T(/uLE_T( call — this test would pass vacuously.'
        keyed_on_tasks = [call for call in calls if re.search(r'\btasks\b', _dep_array(call))]
        assert not keyed_on_tasks, (
            'TaskGraphEdges keys a hook on the `tasks` reference, which callers '
            f'rebuild on every render, so it reruns every time: {keyed_on_tasks}'
        )
