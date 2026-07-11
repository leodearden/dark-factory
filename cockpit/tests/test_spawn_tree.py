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
