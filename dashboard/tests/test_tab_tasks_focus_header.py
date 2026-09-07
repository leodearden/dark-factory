"""Wiring tests for the Tasks-tab focus-mode group header in tab_tasks.jsx
(task 4137).

Focus state is GLOBAL — one ``focusMode``/``focusAnchorId`` for the whole tab —
but the narrowing was applied PER PROJECT GROUP, and the group header counted
the PRE-focus array. So in the all-projects view every non-anchor group
rendered an empty graph under an "N/N shown" header: the header counted
``filtered`` while the body was fed ``focusSubset(filtered, focusAnchorId)``,
two expressions that were supposed to agree and did not.

The fix is structural: ``focusGroupView`` (graph_layout.js) returns the
rendered array and its count together, so they cannot be derived from
different arrays. That function is pure and covered executably by
dashboard/tests/js/graph_layout.test.mjs. What THAT suite cannot see is
whether tab_tasks.jsx actually calls it — this repo has no React/jsdom
rendering harness, so the JSX wiring is only reachable as source structure.
That is what this module asserts.

Deliberately a NEW module rather than an extension of
test_tab_tasks_status_counts.py (scoped by its docstring to header status
counts, task 3516) or test_tab_tasks_runtime.py (TaskDetail runtime fields,
task 2637): keeping this surface separate leaves only tab_tasks.jsx itself as
the shared merge surface with a sibling JSX task.

Tests parse JSX source as text and assert structural contracts. Follows the
idiom established in test_tab_tasks_status_counts.py / test_tab_tasks_runtime.py
/ test_index_html.py.
"""

from __future__ import annotations

import re

import pytest
from _dashboard_helpers import extract_function_body, strip_js_comments

#: The filter-empty copy that must STAY in TaskGraph and must NOT be reused for
#: a focus-emptied group. Under focus that sentence is actively false: the
#: filter matched, focus is what removed the tasks.
FILTER_EMPTY_MESSAGE = 'no tasks match the current filter'


def _parameter_list(source: str, func_name: str) -> str:
    """Return the raw parameter-list text of a ``function <func_name>(`` decl.

    ``extract_function_body`` deliberately EXCLUDES the signature, so it cannot
    answer "which props does this component destructure". Both components
    asserted on here take a single destructured object with no nested parens,
    so a non-greedy up-to-``)`` match is exact.

    Raises rather than returning ``''`` on a miss, for the same reason
    ``extract_function_body`` does: an empty parameter list makes every
    ABSENCE assertion below pass vacuously — a permanent false GREEN.
    """
    match = re.search(rf'function\s+{re.escape(func_name)}\s*\(([^)]*)\)', source)
    assert match is not None, (
        f'Could not locate the `function {func_name}(` signature in '
        f'tab_tasks.jsx. Either the component was removed/renamed, or it was '
        f'rewritten as an arrow function — only a named `function` '
        f'declaration is matched. This cannot silently return an empty '
        f'parameter list: an absence assertion over one would pass vacuously.'
    )
    return match.group(1)


@pytest.fixture(scope='module')
def tasks_tab_body(tab_tasks_jsx_body):
    """TasksTab's brace-delimited body, signature excluded.

    Scoped away from the other component functions (TaskGraph, PrdBox,
    TaskDetail, ...) in the same file so an assertion cannot be satisfied by
    an unrelated component — in particular so the focus-empty message is
    checked in TasksTab and the filter-empty message stays checkable in
    TaskGraph independently.
    """
    return extract_function_body(tab_tasks_jsx_body, 'TasksTab')


@pytest.fixture(scope='module')
def tasks_tab_code(tasks_tab_body):
    """TasksTab's body with comments blanked — what the assertions run on.

    Every structural assertion below runs against the stripped text, because a
    comment can satisfy or falsify one either way: an absence assertion
    ("``filtered.length`` must not appear") is broken by the very comment that
    explains why it must not, and a presence assertion can be met by prose
    mentioning the token instead of by code doing it.
    """
    return strip_js_comments(tasks_tab_body)


@pytest.fixture(scope='module')
def project_task_graph_code(tab_tasks_jsx_body):
    """ProjectTaskGraph's body, comments blanked."""
    return strip_js_comments(extract_function_body(tab_tasks_jsx_body, 'ProjectTaskGraph'))


@pytest.fixture(scope='module')
def project_prd_groups_code(tab_tasks_jsx_body):
    """ProjectPrdGroups' body, comments blanked."""
    return strip_js_comments(extract_function_body(tab_tasks_jsx_body, 'ProjectPrdGroups'))


class TestFocusHeaderWiring:
    """The per-project group header must count the array the group actually
    renders, and an emptied group must say why it is empty."""

    def test_tab_tasks_jsx_served(self, _client):
        resp = _client.get('/static/redux/tab_tasks.jsx')
        assert resp.status_code == 200

    def test_destructures_focus_group_view_at_top_level(self, tab_tasks_jsx_body):
        """tab_tasks.jsx must destructure focusGroupView from DF_GRAPH_LAYOUT.

        Whole-file scope on purpose: top-level destructures sit outside every
        component function, so scoping this to TasksTab would never match.
        """
        match = re.search(
            r'const\s*\{([^}]*)\}\s*=\s*window\.DF_GRAPH_LAYOUT\s*;',
            tab_tasks_jsx_body,
        )
        assert match is not None, (
            'tab_tasks.jsx does not destructure window.DF_GRAPH_LAYOUT at top '
            'level — the layout module is registered in index.html but unused.'
        )
        names = {n.strip() for n in match.group(1).split(',') if n.strip()}
        assert 'focusGroupView' in names, (
            f'tab_tasks.jsx destructures {sorted(names)} from DF_GRAPH_LAYOUT '
            'but not focusGroupView — the group header cannot source its count '
            'from the same array the body renders without it.'
        )

    def test_tasks_tab_calls_focus_group_view(self, tasks_tab_code):
        """TasksTab must call the pure module rather than re-inlining the
        narrowing in the projects.map callback."""
        assert re.search(r'focusGroupView\s*\(', tasks_tab_code), (
            'TasksTab does not call focusGroupView(...) — the per-group shown '
            'count must come from the executably-tested pure module, which is '
            'what makes shownCount === shown.length structural.'
        )

    def test_header_no_longer_counts_the_pre_focus_array(self, tasks_tab_code):
        """This is the defect itself.

        ``{filtered.length}/{counts.total} shown`` counted the array BEFORE
        focus narrowing, while the body below it was fed the narrowed one.
        """
        assert not re.search(r'\{\s*filtered\.length\s*\}\s*/', tasks_tab_code), (
            'TasksTab still renders {filtered.length}/... in the group header '
            '— that is the PRE-focus array, a different array from the one the '
            'group body renders under focus. Count groupView.shownCount.'
        )

    def test_header_shown_label_reads_the_narrowed_count(self, tasks_tab_code):
        """The "shown" label must be adjacent to a shownCount reference.

        Pins the count's SOURCE, not just the absence of the old expression:
        without this, deleting the span entirely would pass the test above.
        """
        assert re.search(
            r'\.shownCount\s*\}\s*/\s*\{[^}]*\}\s*shown\b',
            tasks_tab_code,
        ), (
            'The group header\'s "N/M shown" label does not read a .shownCount '
            '— the displayed count must come from focusGroupView, whose '
            'shownCount is its shown array\'s length by construction.'
        )

    def test_narrowing_happens_at_exactly_one_site(
        self, project_task_graph_code, project_prd_groups_code
    ):
        """focusSubset must no longer be called inside either per-project
        component.

        Both used to call it with identical expressions (one per view), and the
        header counted a third array. Hoisting the narrowing to a single call
        site in projects.map is what makes it impossible to feed the header and
        the body different arrays again.
        """
        assert 'focusSubset(' not in project_task_graph_code, (
            'ProjectTaskGraph still calls focusSubset(...) — the narrowing must '
            'arrive as the graphTasks prop from the single focusGroupView call '
            'that also produces the header count.'
        )
        assert 'focusSubset(' not in project_prd_groups_code, (
            'ProjectPrdGroups still calls focusSubset(...) — the narrowing must '
            'arrive as the graphTasks prop from the single focusGroupView call '
            'that also produces the header count.'
        )

    @pytest.mark.parametrize('component', ['ProjectTaskGraph', 'ProjectPrdGroups'])
    def test_components_take_graph_tasks_not_focus_state(self, tab_tasks_jsx_body, component):
        """Both per-project components take the already-narrowed array.

        Asserted against each function's SIGNATURE text rather than the whole
        file, so an unrelated occurrence of the same identifier elsewhere
        cannot satisfy it. Keeping focusMode/focusAnchorId out of the
        parameter list is what prevents a second narrowing site growing back.
        """
        params = _parameter_list(tab_tasks_jsx_body, component)
        names = {n.strip() for n in params.replace('{', ',').replace('}', ',').split(',') if n.strip()}
        assert 'graphTasks' in names, (
            f'{component} does not destructure a graphTasks prop (params: '
            f'{sorted(names)}) — it must receive the array the header counted, '
            f'not re-derive it.'
        )
        assert 'focusMode' not in names, (
            f'{component} still destructures focusMode — focus state must not '
            f'reach the per-project components at all, or the narrowing can '
            f'diverge from the header count again.'
        )
        assert 'focusAnchorId' not in names, (
            f'{component} still destructures focusAnchorId — focus state must '
            f'not reach the per-project components at all.'
        )

    def test_emptied_by_focus_renders_a_focus_specific_empty_state(self, tasks_tab_code):
        """A group emptied BY FOCUS must say so, in its own words.

        The filter-empty copy is actively false there (the filter matched;
        focus removed the tasks), and on the grouped path there is no message
        at all today — groupTasksByPrd([]) yields zero boxes, so the body
        renders a silently blank .prd-groups div.
        """
        assert re.search(r'\.emptiedByFocus\b', tasks_tab_code), (
            'TasksTab never branches on focusGroupView\'s emptiedByFocus — a '
            'group emptied by focus renders blank (grouped view) or claims the '
            'filter emptied it (flat view).'
        )
        assert 'className="empty"' in tasks_tab_code, (
            'TasksTab renders no .empty element for the emptied-by-focus case.'
        )
        assert FILTER_EMPTY_MESSAGE not in tasks_tab_code, (
            f'TasksTab reuses the filter-empty copy {FILTER_EMPTY_MESSAGE!r} '
            f'for the focus-emptied group. Under focus that sentence is false: '
            f'the filter matched and focus is what removed the tasks. That copy '
            f'belongs to TaskGraph and must stay there.'
        )

    def test_filter_empty_copy_stays_in_task_graph(self, tab_tasks_jsx_body):
        """The other half of the assertion above: TaskGraph's own filter-empty
        message must be left untouched, so the two cases stay distinguishable
        rather than one replacing the other."""
        task_graph_code = strip_js_comments(extract_function_body(tab_tasks_jsx_body, 'TaskGraph'))
        assert FILTER_EMPTY_MESSAGE in task_graph_code, (
            f'TaskGraph no longer renders {FILTER_EMPTY_MESSAGE!r} — the '
            f'genuinely filter-empty case lost its message.'
        )

    def test_header_explains_an_emptied_group_even_when_collapsed(self, tasks_tab_code):
        """The emptiedByFocus signal must reach the HEADER, not only the body.

        shell.jsx::ProjectGroup renders `summary` unconditionally but renders
        `children` only when `open`, so a collapsed group shows the header
        alone. Without a header-side suffix, a collapsed emptied group reads as
        a bare "0/N shown" with nothing saying why.
        """
        summary_match = re.search(
            r'const\s+summary\s*=\s*\((.*?)\n\s*\)\s*;',
            tasks_tab_code,
            re.DOTALL,
        )
        assert summary_match is not None, (
            'Could not locate the `const summary = (...)` JSX block in '
            'TasksTab — the group header markup moved or was renamed.'
        )
        assert '.emptiedByFocus' in summary_match.group(1), (
            'The group header summary does not mention emptiedByFocus. A '
            'COLLAPSED group renders summary and no children (shell.jsx: '
            'ProjectGroup), so the body-side message is invisible there and the '
            'header must carry its own focus explanation.'
        )
