"""cockpit.panes.spawn_tree — pure spawn-forest helpers + the tree toggle (Fleet Cockpit C9a, PRD §9).

A v1.1, deferrable in-cockpit spawn-tree toggle: renders the
parent_session_id parent→child session tree, highlights outstanding
(non-terminal) children, and on Enter jumps to the highlighted child by
invoking focus() on that child's DisplayTarget via the focus backend.

Mirrors session_table.py's/decision_queue.py's own "pure helpers + widget in
one module" convention: build_spawn_forest is fast/deterministic to unit
test directly (no pilot, no event loop) -- only the SpawnTree widget and
SpawnTreeScreen require a running Textual app, and are covered by
test_app.py's pilot tests instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from orchestrator.session_registry import TERMINAL_STATUSES, SessionRecord, Status

from cockpit.panes.session_table import order_sessions


def is_outstanding(record: SessionRecord) -> bool:
    """Whether *record* counts as an "outstanding child" (Fleet Cockpit C9a).

    Reuses count_outstanding_children's exact "non-terminal" semantics
    (session_table.py) so the tree's highlight agrees with the session
    table's outstanding-children column. Fail-soft (PRD §2), mirroring
    state_glyph: a foreign/unrecognized status degrades to False rather
    than raising.
    """
    try:
        resolved = Status(record.status)
    except ValueError:
        return False
    return resolved not in TERMINAL_STATUSES


@dataclass(frozen=True)
class SpawnTreeNode:
    """One node in the rendered spawn forest -- a session plus its children.

    slug: this node's session_slug (SpawnTreeNode.record.session_slug,
        pulled out as its own field for cheap identity comparisons/lookups).
    record: the backing SessionRecord.
    outstanding: whether this node counts as an "outstanding child" (a
        non-terminal child -- see is_outstanding). Always False for a root
        -- a root is a human-launched session, not anyone's child.
    children: this node's direct children, already ordered (see
        build_spawn_forest).
    """

    slug: str
    record: SessionRecord
    outstanding: bool
    children: tuple[SpawnTreeNode, ...]


def build_spawn_forest(records: list[SessionRecord]) -> list[SpawnTreeNode]:
    """Build the parent→child forest of *records* (Fleet Cockpit C9a, PRD §9).

    Roots are records with parent_session_id is None OR whose named parent
    slug is not present in *records* (an orphan child surfaces as a root --
    totality: every record must appear exactly once, see the module's
    fail-soft design decisions). Children are grouped by parent_session_id;
    each sibling group (and the root list itself) is ordered via
    order_sessions, mirroring the session table's own deterministic
    blocked-first ordering. Empty input -> [].
    """
    by_slug = {record.session_slug: record for record in records}
    children_by_parent: dict[str, list[SessionRecord]] = {}
    roots: list[SessionRecord] = []
    for record in records:
        parent_slug = record.parent_session_id
        if parent_slug is not None and parent_slug in by_slug:
            children_by_parent.setdefault(parent_slug, []).append(record)
        else:
            roots.append(record)

    def _build_node(record: SessionRecord, *, is_root: bool) -> SpawnTreeNode:
        child_records = order_sessions(children_by_parent.get(record.session_slug, []))
        return SpawnTreeNode(
            slug=record.session_slug,
            record=record,
            outstanding=False if is_root else is_outstanding(record),
            children=tuple(_build_node(child, is_root=False) for child in child_records),
        )

    return [_build_node(record, is_root=True) for record in order_sessions(roots)]
