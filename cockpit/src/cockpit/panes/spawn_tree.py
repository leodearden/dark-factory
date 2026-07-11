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

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from orchestrator.session_registry import TERMINAL_STATUSES, SessionRecord, Status
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Tree
from textual.widgets.tree import TreeNode

from cockpit.panes.session_table import format_title, order_sessions, state_glyph


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
    slug is not present in *records* (an orphan child surfaces as a root).
    Children are grouped by parent_session_id; each sibling group (and the
    root list itself) is ordered via order_sessions, mirroring the session
    table's own deterministic blocked-first ordering. Empty input -> [].

    Total and fail-soft over a malformed parent graph (PRD §2): a global
    *visited* set guards the DFS so no slug is ever emitted twice and a
    self-parented record (parent_session_id == its own slug) or a cycle
    (A's parent is B, B's parent is A, ...) can never recurse forever --
    a would-be child already seen on this build is simply not re-descended
    into. Such a cycle has no member reachable from a genuine root (by
    construction, every node in it claims another cycle member as its
    parent), so after the real roots are built, any record STILL unvisited
    is promoted to an extra root and built under the same visited guard --
    guaranteeing every input record appears exactly once in the result,
    never silently dropped.
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

    visited: set[str] = set()

    def _build_node(record: SessionRecord, *, is_root: bool) -> SpawnTreeNode:
        visited.add(record.session_slug)
        candidates = children_by_parent.get(record.session_slug, [])
        child_records = order_sessions(
            [child for child in candidates if child.session_slug not in visited]
        )
        return SpawnTreeNode(
            slug=record.session_slug,
            record=record,
            outstanding=False if is_root else is_outstanding(record),
            children=tuple(_build_node(child, is_root=False) for child in child_records),
        )

    forest = [_build_node(record, is_root=True) for record in order_sessions(roots)]

    # Snapshot BEFORE the loop, but re-check `not in visited` INSIDE it too:
    # building one pure-cycle member (e.g. cycle-a) also visits the rest of
    # its cycle (cycle-b) as a descendant, so a later entry in this same
    # snapshot must be skipped rather than re-built (and re-emitted) as its
    # own extra root.
    orphaned_cycle_members = [record for record in records if record.session_slug not in visited]
    forest.extend(
        _build_node(record, is_root=True)
        for record in order_sessions(orphaned_cycle_members)
        if record.session_slug not in visited
    )

    return forest


_OUTSTANDING_MARKER = '●'
_OUTSTANDING_BLANK = ' '  # same display width as _OUTSTANDING_MARKER -- keeps columns aligned


def _populate(parent: TreeNode[str], node: SpawnTreeNode) -> None:
    """Recursively add *node* (and its subtree) under *parent* (Fleet Cockpit C9a).

    A node with children is added expanded (Tree.add(..., expand=True)) so
    the forest's structure is visible without the operator needing to
    expand each branch by hand; a childless node is added via add_leaf
    instead (there is nothing to expand). Either way `data` is the node's
    bare session_slug (str) -- never a Display/DisplayTarget -- so this
    stays import-clean of backends; resolving a slug to a focus target is
    the app's job (see cockpit.app.CockpitApp._focus_slug).

    An outstanding node's label (see SpawnTreeNode.outstanding /
    is_outstanding) is prefixed with _OUTSTANDING_MARKER; a non-outstanding
    node is prefixed with _OUTSTANDING_BLANK instead, an equal-width blank,
    so every row's glyph/title columns still line up regardless of which
    rows are marked.
    """
    marker = _OUTSTANDING_MARKER if node.outstanding else _OUTSTANDING_BLANK
    label = f'{marker}{state_glyph(node.record.status)} {format_title(node.record)}'
    if node.children:
        tree_node = parent.add(label, data=node.slug, expand=True)
        for child in node.children:
            _populate(tree_node, child)
    else:
        parent.add_leaf(label, data=node.slug)


class SpawnTree(Tree[str]):
    """Renders the parent->child spawn forest (Fleet Cockpit C9a, PRD §9).

    build(records) is the sole entry point: it fully replaces the tree's
    contents from a fresh snapshot of *records* every time it's called --
    there is no incremental update. show_root is False since the forest
    itself may hang multiple roots off Tree's own single hidden root node.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__('sessions', *args, **kwargs)
        self.show_root = False

    def build(self, records: Sequence[SessionRecord]) -> None:
        """Replace the tree's contents with the forest built from *records*."""
        self.clear()
        for root in build_spawn_forest(list(records)):
            _populate(self.root, root)


class SpawnTreeScreen(ModalScreen[None]):
    """The 't' spawn-tree toggle's modal (Fleet Cockpit C9a, PRD §9).

    A read-only, point-in-time snapshot view: *records* is captured once at
    construction (mirrors SpawnScreen's own constructed-with-data shape,
    see spawn_bar.py) and rendered via SpawnTree.build on mount -- this
    screen never re-scans the registry itself and never touches a backend
    directly. Selecting a node calls the injected *on_focus* callback with
    that node's slug (its `data`); resolving the slug to a Display target
    and routing it to a backend is the app's job (see
    cockpit.app.CockpitApp._focus_slug), preserving the same C4/C5b
    separation SpawnScreen keeps around spawn_session. Both Escape and 't'
    dismiss this screen directly via action_cancel -- see its docstring for
    why 't' can't simply rely on bubbling up to CockpitApp.action_toggle_tree
    the way an ordinary (non-modal) key press would.
    """

    DEFAULT_CSS = """
    SpawnTreeScreen {
        align: center middle;
    }

    SpawnTreeScreen > SpawnTree {
        width: 80%;
        height: 80%;
        border: round $accent;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding('t', 'cancel', 'Close tree', show=False),
        Binding('escape', 'cancel', 'Cancel', show=False),
    ]

    def __init__(
        self,
        records: Sequence[SessionRecord],
        on_focus: Callable[[str], None],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._records = list(records)
        # NOT named `_on_focus`: Widget already defines an internal
        # `_on_focus(self, event: events.Focus)` message-dispatch handler
        # (textual.widget.Widget._on_focus) that Textual's MessagePump looks
        # up by that exact name whenever this screen receives a real Focus
        # event -- an attribute of the same name here would silently shadow
        # it with an incompatible str -> None callable.
        self._focus_callback = on_focus

    def compose(self) -> ComposeResult:
        yield SpawnTree()

    def on_mount(self) -> None:
        self.query_one(SpawnTree).build(self._records)

    def on_tree_node_selected(self, event: Tree.NodeSelected[str]) -> None:
        """Route a selected node's slug to the injected *on_focus* callback.

        Tree posts NodeSelected for any selected node -- branch or leaf --
        and every node built by _populate carries its own session_slug as
        `data` regardless of whether it has children, so this fires
        uniformly across the whole forest. Fail-soft (PRD §2): `data` is
        only ever None for Tree's own hidden root (show_root=False, never
        selectable), so the guard is defensive rather than load-bearing.
        Resolving the slug to a Display target and routing it to a backend
        is the app's job (see cockpit.app.CockpitApp._focus_slug) -- this
        screen only ever forwards the bare slug.
        """
        if event.node.data is not None:
            self._focus_callback(event.node.data)

    def action_cancel(self) -> None:
        """Dismiss this screen -- bound to both Escape and 't' (Fleet Cockpit C9a).

        Mirrors SpawnScreen.action_cancel's own escape-to-dismiss shape.
        't' is bound here too (not just Escape) because Textual's
        ModalScreen truncates the non-priority key-binding chain at itself
        (Screen._modal_binding_chain stops at the first is_modal node in
        the DOM) -- so CockpitApp's own plain 't' -> toggle_tree Binding is
        never consulted while this screen sits on top of the stack; only
        bindings declared here (or on a focused descendant, e.g. the
        SpawnTree itself) are. Binding 't' directly to this same dismiss()
        is what actually makes the toggle's "press 't' again to close"
        contract work -- see CockpitApp.action_toggle_tree's own docstring
        for the app-level half of this.
        """
        self.dismiss()
