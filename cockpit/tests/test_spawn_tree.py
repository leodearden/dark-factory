"""Tests for cockpit.panes.spawn_tree — pure spawn-forest helpers (Fleet Cockpit C9a, PRD §9).

Mirrors test_session_table.py's convention: every pure helper here is
deterministic and unit-testable directly, with no running Textual app.
Fail-soft is a hard constraint (PRD §2): a foreign status, an orphaned or
cyclic parent_session_id must degrade gracefully, never raise or hang.
"""

from __future__ import annotations

from orchestrator import session_registry as sr


def _make_record(**overrides):
    """Mirrors test_app.py's/test_session_table.py's _make_record convention."""
    fields: dict = {
        'session_slug': 'unblock-df-2085-4242',
        'status': sr.Status.RUNNING,
        'title': 'unblock:df#2085 slug',
        'role': 'unblock',
        'project': 'df',
        'task_id': '2085',
        'start_ts': '2026-07-07T00:00:00+00:00',
    }
    fields.update(overrides)
    return sr.SessionRecord(**fields)


def _find_node(forest, slug):
    """Depth-first search *forest* for the node whose slug == *slug*, or None."""
    for node in forest:
        if node.slug == slug:
            return node
        found = _find_node(node.children, slug)
        if found is not None:
            return found
    return None


class TestBuildSpawnForest:
    def test_renders_parent_child_structure(self):
        from cockpit.panes.spawn_tree import build_spawn_forest

        root = _make_record(session_slug='root-1', parent_session_id=None)
        child1 = _make_record(
            session_slug='child-1', parent_session_id='root-1', status=sr.Status.RUNNING
        )
        child2 = _make_record(
            session_slug='child-2', parent_session_id='root-1', status=sr.Status.RUNNING
        )

        forest = build_spawn_forest([root, child1, child2])

        assert len(forest) == 1
        root_node = forest[0]
        assert root_node.slug == 'root-1'
        assert {child.slug for child in root_node.children} == {'child-1', 'child-2'}

    def test_grandchildren_nest_correctly(self):
        from cockpit.panes.spawn_tree import build_spawn_forest

        root = _make_record(session_slug='root-1', parent_session_id=None)
        child1 = _make_record(session_slug='child-1', parent_session_id='root-1')
        grandchild = _make_record(session_slug='grandchild-1', parent_session_id='child-1')

        forest = build_spawn_forest([root, child1, grandchild])

        assert len(forest) == 1
        root_node = forest[0]
        assert len(root_node.children) == 1
        child_node = root_node.children[0]
        assert child_node.slug == 'child-1'
        assert len(child_node.children) == 1
        assert child_node.children[0].slug == 'grandchild-1'


class TestIsOutstanding:
    def test_non_terminal_statuses_are_outstanding(self):
        from cockpit.panes.spawn_tree import is_outstanding

        running = _make_record(status=sr.Status.RUNNING)
        awaiting = _make_record(status=sr.Status.AWAITING_INPUT)

        assert is_outstanding(running) is True
        assert is_outstanding(awaiting) is True

    def test_terminal_statuses_are_not_outstanding(self):
        from cockpit.panes.spawn_tree import is_outstanding

        exited = _make_record(status=sr.Status.EXITED)
        failed = _make_record(status=sr.Status.FAILED_TO_START)

        assert is_outstanding(exited) is False
        assert is_outstanding(failed) is False

    def test_unknown_status_degrades_to_false(self):
        from cockpit.panes.spawn_tree import is_outstanding

        foreign = _make_record(status='some-foreign-status')

        assert is_outstanding(foreign) is False


class TestOutstandingFlag:
    def test_non_terminal_child_is_outstanding(self):
        from cockpit.panes.spawn_tree import build_spawn_forest

        root = _make_record(session_slug='root-1', parent_session_id=None, status=sr.Status.IDLE)
        child = _make_record(
            session_slug='child-1', parent_session_id='root-1', status=sr.Status.RUNNING
        )

        forest = build_spawn_forest([root, child])

        assert _find_node(forest, 'child-1').outstanding is True

    def test_terminal_child_is_not_outstanding(self):
        from cockpit.panes.spawn_tree import build_spawn_forest

        root = _make_record(session_slug='root-1', parent_session_id=None)
        child = _make_record(
            session_slug='child-1', parent_session_id='root-1', status=sr.Status.EXITED
        )

        forest = build_spawn_forest([root, child])

        assert _find_node(forest, 'child-1').outstanding is False

    def test_root_is_never_outstanding_even_if_non_terminal(self):
        from cockpit.panes.spawn_tree import build_spawn_forest

        root = _make_record(
            session_slug='root-1', parent_session_id=None, status=sr.Status.RUNNING
        )

        forest = build_spawn_forest([root])

        assert forest[0].outstanding is False

    def test_foreign_child_status_is_fail_soft_not_outstanding(self):
        from cockpit.panes.spawn_tree import build_spawn_forest

        root = _make_record(session_slug='root-1', parent_session_id=None)
        child = _make_record(
            session_slug='child-1', parent_session_id='root-1', status='some-foreign-status'
        )

        forest = build_spawn_forest([root, child])

        assert _find_node(forest, 'child-1').outstanding is False
