"""Wiring tests for the stranded-task badge in tab_tasks.jsx / tabs.jsx (task 3543).

PRD ``plans/task-escalation-state-graph-prd.md`` task ι (spec S8/E12). The
dashboard's only per-task "something is working on this" signal was ``agent``,
which is set purely from ``TaskRuntimeEntry.has_worktree`` — a directory on
disk, truthy long after the agent that created it died. A task showing
``claude-task-42`` against a dead claimant therefore read as healthy. Steps 2/4
computed the honest verdict server-side (``row['stranded']``, via
``shared.task_claimant.is_stranded``); this module pins that the three task
surfaces actually RENDER it, and render it as a signal INDEPENDENT of ``agent``.

Three surfaces:

* ``tab_tasks.jsx`` ``TaskDetail`` — a badge beside the status badge.
* ``tab_tasks.jsx`` ``renderNode`` — a compact marker on the graph node,
  following the existing ``.train-badge`` precedent.
* ``tabs.jsx`` ``OrchTab`` — the same marker on the per-orchestrator task table.

Idiom: these are render-bound JSX contracts, and the node suite under
``dashboard/tests/js/`` cannot import a ``.jsx`` at all, so the assertions parse
JSX source as text (TestClient GET of ``/static/redux/*.jsx``). A local
``_extract_function_body`` scopes each assertion to one component — following
the established per-file precedent (independent copies live in
test_tab_orchestrators.py, test_tab_tasks_status_counts.py, test_tab_escalations.py
and test_charts_null_samples.py) — and every fixture carries the ``assert body``
anti-vacuity guard so a regex that matches nothing can never pass silently.
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
def tab_tasks_jsx(_client):
    return _client.get('/static/redux/tab_tasks.jsx').text


@pytest.fixture(scope='module')
def tabs_jsx(_client):
    return _client.get('/static/redux/tabs.jsx').text


def _extract_function_body(source: str, func_name: str, indent: str = '') -> str:
    """Source text of ``<indent>function <func_name>(`` up to the next
    declaration at the SAME indentation level (or end-of-file).

    *indent* selects the nesting level: ``''`` for a top-level component,
    ``'  '`` for a helper nested one level inside one (``renderNode`` lives
    inside ``TaskGraph``). Scoping matters — tab_tasks.jsx is ~800 lines and an
    unscoped regex could be satisfied by an unrelated component.
    """
    start_match = re.search(
        rf'^{indent}function {re.escape(func_name)}\(', source, re.MULTILINE
    )
    assert start_match is not None, f'{indent}function {func_name}( not found'
    start = start_match.start()
    rest = source[start:]
    # Anchor on an explicit newline rather than a MULTILINE '^': the latter also
    # matches at offset 0, which would slice the body down to nothing.
    next_match = re.search(rf'\n{indent}[A-Za-z]', rest[1:])
    end = start + 1 + next_match.start() + 1 if next_match else len(source)
    return source[start:end]


def _fragment_around(body: str, token: str, *, before: int = 80, after: int = 320) -> str:
    """A window of *body* around the first occurrence of *token*.

    Used to assert properties of the badge expression itself (its class family,
    its tooltip) without pinning the exact JSX spelling.
    """
    idx = body.find(token)
    assert idx != -1, f'{token!r} not found in the scoped body'
    return body[max(0, idx - before) : idx + after]


def _assert_not_derived_from_agent(body: str, field: str) -> None:
    """The strand marker must never be gated on the ``agent`` value.

    ``agent`` is worktree presence; ``stranded`` is claimant liveness. A leftover
    worktree with a dead claimant is precisely the case this task surfaces, so a
    marker gated on ``agent`` truthiness (either direction) would re-introduce
    the defect while looking like a fix.
    """
    assert not re.search(rf'{re.escape(field)}\.agent\s*&&[^\n]{{0,160}}stranded', body)
    assert not re.search(rf'stranded\s*&&[^\n]{{0,160}}{re.escape(field)}\.agent\b', body)
    assert not re.search(rf'!\s*{re.escape(field)}\.agent[^\n]{{0,120}}stranded', body)


# ---------------------------------------------------------------------------
# TaskDetail — the badge beside the status badge
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def task_detail_body(tab_tasks_jsx):
    body = _extract_function_body(tab_tasks_jsx, 'TaskDetail')
    assert body, 'TaskDetail body extracted empty — the regex matched nothing'
    return body


class TestTaskDetailStrandedBadge:
    def test_renders_a_badge_gated_on_the_stranded_field(self, task_detail_body):
        assert re.search(r'\btask\.stranded\b', task_detail_body), (
            'TaskDetail does not read task.stranded — the server-computed strand '
            'verdict is dropped at the render boundary'
        )
        frag = _fragment_around(task_detail_body, 'task.stranded')
        assert 'badge' in frag, 'the strand marker does not use the .badge class family'

    def test_badge_uses_the_bad_modifier(self, task_detail_body):
        """Same class family as the status badge; no new CSS."""
        frag = _fragment_around(task_detail_body, 'task.stranded')
        assert re.search(r'badge[^"\'`]*\bbad\b', frag), (
            'the strand badge should reuse the existing .badge.bad modifier'
        )

    def test_badge_tooltip_names_the_cause(self, task_detail_body):
        frag = _fragment_around(task_detail_body, 'task.stranded')
        assert 'title=' in frag, 'the strand badge carries no title= tooltip'
        assert re.search(r'claimant|heartbeat', frag), (
            'the tooltip must name the cause (no live claimant / stale heartbeat), '
            'not just say "stranded"'
        )

    def test_stranded_is_not_derived_from_agent(self, task_detail_body):
        _assert_not_derived_from_agent(task_detail_body, 'task')

    def test_agent_row_still_renders(self, task_detail_body):
        """The two signals coexist: the strand badge does not replace the agent row."""
        assert re.search(r'>\s*agent\s*<', task_detail_body)
        assert 'task.agent' in task_detail_body


# ---------------------------------------------------------------------------
# renderNode — the compact graph-node marker
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def render_node_body(tab_tasks_jsx):
    body = _extract_function_body(tab_tasks_jsx, 'renderNode', indent='  ')
    assert body, 'renderNode body extracted empty — the regex matched nothing'
    return body


class TestRenderNodeStrandedMarker:
    def test_node_carries_a_marker_gated_on_the_stranded_field(self, render_node_body):
        assert re.search(r'\bt\.stranded\b', render_node_body), (
            'graph nodes do not render the strand verdict'
        )

    def test_marker_follows_the_train_badge_precedent(self, render_node_body):
        """A conditional inline span in the node's .meta row, like .train-badge."""
        assert 'train-badge' in render_node_body, (
            'the .train-badge precedent this marker follows has moved — re-anchor'
        )
        assert re.search(r'\{\s*t\.stranded\s*&&', render_node_body), (
            'the marker must be a conditional render on t.stranded'
        )
        frag = _fragment_around(render_node_body, 't.stranded')
        assert 'title=' in frag, 'the node marker carries no title= tooltip'

    def test_marker_is_not_derived_from_agent(self, render_node_body):
        _assert_not_derived_from_agent(render_node_body, 't')


# ---------------------------------------------------------------------------
# OrchTab — the per-orchestrator task table
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def orch_tab_body(tabs_jsx):
    body = _extract_function_body(tabs_jsx, 'OrchTab')
    assert body, 'OrchTab body extracted empty — the regex matched nothing'
    return body


class TestOrchTabStrandedMarker:
    def test_task_row_carries_the_marker(self, orch_tab_body):
        assert re.search(r'\bt\.stranded\b', orch_tab_body), (
            "OrchTab's task table does not render the strand verdict"
        )
        frag = _fragment_around(orch_tab_body, 't.stranded')
        assert 'badge' in frag, 'the OrchTab strand marker does not use .badge'
        assert re.search(r'badge[^"\'`]*\bbad\b', frag)
        assert 'title=' in frag

    def test_marker_is_not_derived_from_agent(self, orch_tab_body):
        _assert_not_derived_from_agent(orch_tab_body, 't')

    def test_agent_cell_still_renders(self, orch_tab_body):
        assert 't.agent' in orch_tab_body
