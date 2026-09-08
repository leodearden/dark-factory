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


def _parameter_list(source: str, func_name: str) -> str:
    """Return the raw parameter-list text of a ``function <func_name>(`` decl.

    ``extract_function_body`` deliberately EXCLUDES the signature, so it cannot
    answer "which props does this component destructure".

    Walks paren DEPTH rather than matching up to the first ``)``: a parameter
    default can carry its own parens (``onSelect = () => {}``), and a
    stop-at-the-first-``)`` match would silently return a TRUNCATED list. The
    ABSENCE assertions below would then run against text that never reached
    the name they forbid — vacuously GREEN for the same reason an empty list
    is, which is why this raises rather than returning ``''`` on a miss (the
    rule ``extract_function_body`` follows).

    Same paren-depth walk as test_charts_axis_labels.py's ``_extract_signature``.
    Its proper home is ``_dashboard_helpers`` beside ``extract_function_body``/
    ``strip_js_comments``, and hoisting it there is tracked by ticket
    tkt_0RSN5VVGAVK7BQ8K9GX4PM2YBZ (see test_esc_flow_diagram.py's header note
    on this helper family); task 4137 holds a lock on neither file, so it fixes
    the truncation here instead of widening its merge surface.
    """
    match = re.search(rf'\bfunction\s+{re.escape(func_name)}\s*\(', source)
    assert match is not None, (
        f'Could not locate the `function {func_name}(` signature in '
        f'tab_tasks.jsx. Either the component was removed/renamed, or it was '
        f'rewritten as an arrow function — only a named `function` '
        f'declaration is matched. This cannot silently return an empty '
        f'parameter list: an absence assertion over one would pass vacuously.'
    )
    depth, i = 1, match.end()
    while i < len(source) and depth > 0:
        if source[i] == '(':
            depth += 1
        elif source[i] == ')':
            depth -= 1
        i += 1
    assert depth == 0, (
        f'Unbalanced parameter list for `function {func_name}(` in '
        f'tab_tasks.jsx — the walk ran off the end of the file. Returning the '
        f'partial slice would make an absence assertion over it meaningless.'
    )
    return source[match.end():i - 1]


@pytest.fixture(scope='module')
def tasks_tab_body(tab_tasks_jsx_body):
    """TasksTab's brace-delimited body, signature excluded.

    Scoped away from the other component functions (TaskGraph, PrdBox,
    TaskDetail, ...) in the same file so an assertion cannot be satisfied by
    an unrelated component.
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
def tab_tasks_jsx_code(tab_tasks_jsx_body):
    """The WHOLE file with comments blanked.

    Deliberately unscoped, unlike ``tasks_tab_code``: the one assertion that
    uses it forbids a token ANYWHERE in the module, which is a claim no
    per-component slice can make.
    """
    return strip_js_comments(tab_tasks_jsx_body)


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

    def test_narrowing_happens_at_exactly_one_site(self, tab_tasks_jsx_code):
        """focusSubset must not be called ANYWHERE in tab_tasks.jsx.

        The flat and grouped views used to call it with identical expressions
        (one each) while the header counted a third array. Hoisting the
        narrowing to the single focusGroupView call in projects.map is what
        makes it impossible to feed the header and the body different arrays
        again — so this is whole-file rather than per-component: a second call
        inlined in the map callback, or in a newly reintroduced per-project
        wrapper, reopens exactly the same drift and no component-scoped
        assertion would see it.
        """
        assert 'focusSubset(' not in tab_tasks_jsx_code, (
            'tab_tasks.jsx calls focusSubset(...) again — narrowing must happen '
            'only in the single focusGroupView call that also produces the '
            'header count, or the header can once more display a number for an '
            'array the group body did not render.'
        )

    @pytest.mark.parametrize('component', ['TaskGraph', 'ProjectPrdGroups'])
    @pytest.mark.parametrize('focus_prop', ['focusMode', 'focusAnchorId'])
    def test_rendering_components_take_no_focus_state(
        self, tab_tasks_jsx_body, component, focus_prop
    ):
        """Neither component the group body renders may take focus state.

        These are the two the projects.map callback hands ``groupView.shown``
        to — TaskGraph for the flat view, ProjectPrdGroups for the grouped one.
        Keeping focusMode/focusAnchorId out of their parameter lists is what
        prevents a second narrowing site growing back and diverging from the
        header count again.

        The PROP NAME the narrowed array arrives under is deliberately NOT
        pinned: whether it is spelled ``tasks`` or ``graphTasks`` is a naming
        choice this file has no opinion about, and a rename must not turn it
        red. What it does insist on is a NON-EMPTY parameter list, without
        which the absence below would pass vacuously.

        Asserted against each function's SIGNATURE text rather than the whole
        file, so an unrelated occurrence of the same identifier elsewhere
        cannot satisfy it, and by regex rather than by splitting on commas: a
        defaulted parameter (``focusMode = false``) is not an exact set member
        but is every bit the focus state this forbids.
        """
        params = _parameter_list(tab_tasks_jsx_body, component)
        assert params.strip(), (
            f'{component} declares an empty parameter list. It is rendered with '
            f'an already-narrowed task array, so this is either a rewrite this '
            f'test no longer measures or a bad parse — and the absence '
            f'assertion below would pass vacuously over it.'
        )
        assert not re.search(rf'\b{focus_prop}\b', params), (
            f'{component} takes {focus_prop} (params: {params.strip()!r}). Focus '
            f'state must not reach the components that render a group body at '
            f'all: the narrowing belongs to the single focusGroupView call that '
            f'also produces the header count, and a second one can diverge from '
            f'it again.'
        )

    def test_emptied_by_focus_renders_a_focus_specific_empty_state(self, tasks_tab_code):
        """A group emptied BY FOCUS must render an empty state of its own.

        Structural only: that TasksTab branches on the flag and renders an
        .empty element. The WORDING is deliberately NOT asserted — it is UI
        prose, and a cosmetic rewrite of a message must not turn this red.

        Why the branch has to exist at all: the filter-empty sentence is
        actively false there (the filter matched; focus removed the tasks),
        and on the grouped path there is no message at all today —
        groupTasksByPrd([]) yields zero boxes, so the body renders a silently
        blank .prd-groups div.
        """
        assert re.search(
            r'\.emptiedByFocus\b[^;]{0,200}?className="empty"',
            tasks_tab_code,
            re.DOTALL,
        ), (
            'TasksTab has no .empty element inside an emptiedByFocus branch — a '
            'group emptied by focus renders blank (grouped view: '
            'groupTasksByPrd([]) yields zero boxes) or claims the filter '
            'emptied it (flat view). The two tokens are required in ONE '
            'expression on purpose: as independent whole-body existence checks '
            'this passed on the header suffix plus any unrelated .empty element '
            'in TasksTab, so deleting the body branch left it green.'
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
