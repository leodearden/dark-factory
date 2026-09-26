"""Doc/table parity helpers for ARCHITECTURE.md section 3.1 (task 4535).

Test-support module — NOT production code; nothing under ``shared/src/``
changes for this. Pins ARCHITECTURE.md section 3.1's mermaid
``stateDiagram-v2`` diagram against the real transition authority,
``shared/src/shared/task_transitions.py::TRANSITIONS``: parses the diagram
out of the doc, reads the table, and reports the two as a set-difference so
a future edit to either side that the other doesn't mirror is a loud
failure instead of silent drift.

Consumed by ``shared/tests/test_architecture_doc_transition_parity.py``.
Importable bare (``from architecture_doc_transitions import ...``) because
``shared/tests/conftest.py`` inserts this ``tests/`` directory onto
``sys.path`` — the same mechanism ``silent_fallthrough_scan.py`` and
``capability_manifest_corpus.py`` already rely on.

Cross-file references below are cited as ``path/to/module.py::symbol``
(never a bare line number), per this repo's convention (CLAUDE.md).

Every failure mode here — a missing ``ARCHITECTURE.md``, a renamed section
heading, a section with zero or more than one mermaid fence — RAISES rather
than ``pytest.skip``s or returns an empty/default result. A tolerant reader
would let exactly the drift this guard exists to catch (a renamed heading,
a relocated diagram) pass through unnoticed, mirroring
``silent_fallthrough_scan.py::iter_first_party_files``'s sentinel
validation, done "to prevent a mis-resolved root from producing a
silently-empty scan".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

from shared import task_transitions
from shared.task_statuses import TaskStatus

# shared/tests/architecture_doc_transitions.py -> parents[0]=shared/tests,
# parents[1]=shared, parents[2]=repo root. Same idiom as
# shared/tests/conftest.py:23 (REPO_ROOT) and every sibling scanner module in
# this directory; correct inside a `.worktrees/<id>` checkout too.
REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = REPO_ROOT / 'ARCHITECTURE.md'

#: The section heading that anchors the lifecycle diagram. ARCHITECTURE.md
#: has THREE mermaid fences (process topology, this one, the escalation
#: ladder) — extraction must anchor to this literal heading text rather than
#: grabbing "the first fence in the doc", or it would silently read the
#: wrong diagram.
SECTION_HEADING = '### 3.1 Status vocabulary'

_NEXT_HEADING_RE = re.compile(r'^#{1,3} ', re.MULTILINE)
_MERMAID_FENCE_RE = re.compile(r'```mermaid\n(.*?)```', re.DOTALL)


def read_architecture_doc(root: Path = REPO_ROOT) -> str:
    """Return the full text of ``<root>/ARCHITECTURE.md``.

    Raises :class:`FileNotFoundError` — loudly, naming the resolved path —
    when the file is absent. Never ``pytest.skip``s and never falls back to
    an empty string: a caller with no doc to check against has no basis for
    asserting parity, and silently treating that as "nothing to check"
    would defeat the guard exactly when its precondition breaks.
    """
    doc_path = Path(root) / 'ARCHITECTURE.md'
    if not doc_path.is_file():
        raise FileNotFoundError(
            f'ARCHITECTURE.md not found at {doc_path!r} -- cannot check the '
            'task-status diagram/table parity guard (task 4535). Pass the real '
            'repo root, or restore the file.'
        )
    return doc_path.read_text(encoding='utf-8')


def extract_lifecycle_mermaid(text: str) -> str:
    """Return the body of section 3.1's mermaid fence (no backticks/prose).

    Slices *text* from :data:`SECTION_HEADING` to the next heading of the
    same or shallower depth (``#``, ``##``, or ``###``), then requires that
    slice to contain EXACTLY one ```` ```mermaid ```` fence. Raises
    :class:`RuntimeError` — naming what was expected and what was found —
    when the heading is absent, or the section holds zero or more than one
    mermaid fence. A tolerant "best effort" extractor here would let a
    renamed heading or a relocated diagram silently produce an empty or
    wrong result, which is precisely the drift this guard exists to catch.

    The terminator is deliberately NOT anchored to ``### `` alone: if
    section 3.1 ever lost its ``### 3.2`` sibling (renumbering, or 3.2's
    removal), a ``### ``-only terminator would run the slice past the
    enclosing ``## `` chapter boundary and into the next chapter's content.
    ARCHITECTURE.md's own section 6 shows this is a real shape to guard
    against: it carries a mermaid fence directly under ``## 6`` before its
    own first ``### 6.1`` subheading, so an over-long slice reaching that
    far would find "exactly one fence" and silently return the WRONG
    diagram instead of raising.
    """
    heading_idx = text.find(SECTION_HEADING)
    if heading_idx == -1:
        raise RuntimeError(
            f'heading {SECTION_HEADING!r} not found in ARCHITECTURE.md -- '
            'architecture_doc_transitions.py (task 4535) cannot locate the '
            'status-lifecycle diagram to check against '
            'shared/src/shared/task_transitions.py::TRANSITIONS. Was the section '
            'renamed or removed?'
        )
    section_start = heading_idx + len(SECTION_HEADING)
    next_heading_match = _NEXT_HEADING_RE.search(text, section_start)
    section_end = next_heading_match.start() if next_heading_match else len(text)
    section_text = text[section_start:section_end]

    fences = _MERMAID_FENCE_RE.findall(section_text)
    if len(fences) != 1:
        raise RuntimeError(
            f'expected exactly 1 mermaid fence under {SECTION_HEADING!r}, found '
            f'{len(fences)} (task 4535) -- architecture_doc_transitions.py cannot '
            'unambiguously locate the status-lifecycle diagram.'
        )
    return fences[0]


# ---------------------------------------------------------------------------
# Edge parser
# ---------------------------------------------------------------------------


def _build_node_aliases() -> dict[str, TaskStatus]:
    """Derive the mermaid-node-id -> TaskStatus map FROM TaskStatus itself.

    Mermaid ``stateDiagram-v2`` state ids cannot contain ``-``, so the
    diagram spells hyphenated statuses with underscores (``in_progress``,
    ``merge_deferred``, ``infra_hold``). The map is keyed ONLY on the
    underscore-folded id: ``_EDGE_RE``'s node-id group
    (``[A-Za-z_][A-Za-z0-9_]*``) cannot capture a hyphen, so a raw,
    un-folded ``status.value`` key would never be reachable through
    :func:`_resolve_node` and would be dead code.

    Deliberately NOT a hand-listed alias table: a hand-maintained copy of
    the three hyphenated names would itself rot the moment a fourth
    hyphenated status was added to :class:`TaskStatus`. Raises
    :class:`RuntimeError` if underscore-folding ever collides two distinct
    statuses onto the same id (none does today: 9 values fold to 9 distinct
    ids) — that would silently alias two real edges together, so it must be
    caught at construction, not left to surface as a wrong parse later.
    """
    aliases: dict[str, TaskStatus] = {}
    for status in TaskStatus:
        alias = status.value.replace('-', '_')
        existing = aliases.get(alias)
        if existing is not None and existing is not status:
            raise RuntimeError(
                f'mermaid node-id alias collision: {alias!r} would map to both '
                f'{existing!r} and {status!r} -- TaskStatus has grown a '
                'hyphen/underscore-ambiguous pair that architecture_doc_transitions.py '
                '(task 4535) cannot disambiguate.'
            )
        aliases[alias] = status
    return aliases


_NODE_ALIASES: dict[str, TaskStatus] = _build_node_aliases()

#: Sorted for a deterministic, readable error message (not a substitute for
#: _NODE_ALIASES, which is what parsing actually consults).
_VALID_STATUS_VALUES: list[str] = sorted(s.value for s in TaskStatus)

_PSEUDO_STATE = '[*]'
_STATE_DIAGRAM_HEADER = 'stateDiagram-v2'

#: A node id is either the mermaid pseudo-state marker or a bare identifier
#: (mermaid state ids: letters/digits/underscore, per the doc's own
#: `in_progress`-style spellings). Anchored full-line match (with an
#: optional trailing `: label`) so that anything this repo's real diagram
#: doesn't already use -- a pipe-label arrow (`A -->|x| B`), a `note` line,
#: a `state ... {` block -- fails to match and is rejected below, rather
#: than being partially parsed.
_EDGE_RE = re.compile(
    r'^(?P<frm>\[\*\]|[A-Za-z_][A-Za-z0-9_]*)'
    r'\s*-->\s*'
    r'(?P<to>\[\*\]|[A-Za-z_][A-Za-z0-9_]*)'
    r'(?:\s*:.*)?$'
)


def _resolve_node(node_id: str) -> TaskStatus:
    """Map a mermaid node id to its :class:`TaskStatus`, or raise loudly."""
    try:
        return _NODE_ALIASES[node_id]
    except KeyError:
        raise ValueError(
            f'unknown mermaid node id {node_id!r} in ARCHITECTURE.md section 3.1 -- '
            f'not a member of shared.task_statuses.TaskStatus '
            f'(valid: {_VALID_STATUS_VALUES})'
        ) from None


def parse_state_diagram_edges(body: str) -> frozenset[tuple[TaskStatus, TaskStatus]]:
    """Parse a mermaid ``stateDiagram-v2`` body into ``(from, to)`` status pairs.

    STRICT by design (this is the anti-rot core of the whole guard): every
    non-blank, non-``%%``-comment, non-header line must match the single
    supported ``<id> --> <id>[: label]`` edge form, or :class:`ValueError`
    is raised naming the offending line verbatim. A tolerant parser that
    skipped whatever it didn't understand would let a diagram edge drawn in
    an unsupported form (a pipe-labelled arrow, a ``note`` line, ...)
    silently vanish -- and the parity gate would stay green through exactly
    the drift it exists to catch.

    A trailing ``%%`` comment on an edge line is stripped before matching
    (not treated as part of the target). ``[*]`` pseudo-state entry/exit
    edges (``[*] --> pending``, ``done --> [*]``) are recognized and
    dropped -- they are diagram entry/terminal markers, not
    ``TRANSITIONS`` pairs, and cannot appear on either side of the table
    comparison.

    Endpoints are resolved through :data:`_NODE_ALIASES`, so both
    ``in_progress`` (the diagram spelling) and ``in-progress`` (the
    :class:`TaskStatus` value) resolve to the same member; an id outside
    that vocabulary raises via :func:`_resolve_node`.
    """
    edges: set[tuple[TaskStatus, TaskStatus]] = set()
    for raw_line in body.splitlines():
        line = raw_line.split('%%', 1)[0].strip()
        if not line or line == _STATE_DIAGRAM_HEADER:
            continue
        match = _EDGE_RE.match(line)
        if match is None:
            raise ValueError(
                f'unparseable line inside ARCHITECTURE.md section 3.1 mermaid diagram '
                f'(task 4535 strict-parser contract -- see '
                f'architecture_doc_transitions.py::parse_state_diagram_edges): {line!r}'
            )
        frm_id, to_id = match.group('frm'), match.group('to')
        if frm_id == _PSEUDO_STATE or to_id == _PSEUDO_STATE:
            continue
        edges.add((_resolve_node(frm_id), _resolve_node(to_id)))
    return frozenset(edges)


# ---------------------------------------------------------------------------
# Table side + comparator
# ---------------------------------------------------------------------------


def table_transition_edges() -> frozenset[tuple[TaskStatus, TaskStatus]]:
    """Return the union of every actor's edge set in ``TRANSITIONS``.

    Reads ``task_transitions.TRANSITIONS`` as a module ATTRIBUTE, inside
    this function body, so the read happens at CALL time. This is
    deliberate and load-bearing, not an incidental style choice: do NOT
    change this to ``from shared.task_transitions import TRANSITIONS`` at
    module scope, and do NOT cache the result in a module-level constant or
    ``functools.lru_cache``. Either change would bind the real table once
    and keep returning it forever, silently defeating
    ``monkeypatch.setattr(task_transitions, 'TRANSITIONS', ...)`` — which is
    exactly what the mutation / failure-direction proofs in
    ``test_architecture_doc_transition_parity.py`` depend on observing. A
    test pins this behaviour directly:
    ``TestTableTransitionEdges::test_reads_transitions_lazily_not_at_import_time``.

    The diagram carries no actor annotation, so the UNION over every
    actor's set (not any single actor's) is the only well-defined
    comparison target — see design decision in the task-4535 plan.
    """
    edges: set[tuple[TaskStatus, TaskStatus]] = set()
    for pairs in task_transitions.TRANSITIONS.values():
        edges.update(pairs)
    return frozenset(edges)


class ParityDiff(NamedTuple):
    """The two-directional result of comparing doc edges against table edges.

    Named fields (rather than a bare 2-tuple) so a caller cannot silently
    transpose the two halves the way positional unpacking would let it.
    """

    missing_from_doc: frozenset[tuple[TaskStatus, TaskStatus]]
    extra_in_doc: frozenset[tuple[TaskStatus, TaskStatus]]


def diff_transition_edges(
    doc_edges: frozenset[tuple[TaskStatus, TaskStatus]],
    table_edges: frozenset[tuple[TaskStatus, TaskStatus]],
) -> ParityDiff:
    """Compare the diagram's edges against the table's edges.

    Returns a :class:`ParityDiff` with ``missing_from_doc`` (edges the
    table has but the diagram doesn't draw) and ``extra_in_doc`` (edges the
    diagram draws but the table doesn't have) — plain set differences, in
    that order.
    """
    return ParityDiff(
        missing_from_doc=frozenset(table_edges - doc_edges),
        extra_in_doc=frozenset(doc_edges - table_edges),
    )


# ---------------------------------------------------------------------------
# Failure-message renderer
# ---------------------------------------------------------------------------

_ARCHITECTURE_DOC_REF = 'ARCHITECTURE.md section 3.1'
_TABLE_REF = 'TRANSITIONS (shared/src/shared/task_transitions.py)'


def _render_edge_lines(edges: frozenset[tuple[TaskStatus, TaskStatus]]) -> str:
    """Render *edges* sorted by ``(frm.value, to.value)`` for determinism."""
    ordered = sorted(edges, key=lambda edge: (edge[0].value, edge[1].value))
    return '\n'.join(f'  {frm.value} -> {to.value}' for frm, to in ordered)


def format_parity_failure(
    missing_from_doc: frozenset[tuple[TaskStatus, TaskStatus]],
    extra_in_doc: frozenset[tuple[TaskStatus, TaskStatus]],
) -> str:
    """Render a :class:`ParityDiff`'s two halves as a self-routing failure message.

    Each present half becomes its own labelled, sorted section (rendered
    with the real hyphenated ``TaskStatus`` VALUES, e.g. ``in-progress ->
    review`` — not the enum repr, not the diagram's underscore spelling —
    so the reader can grep ``task_transitions.py`` for it directly). A
    section whose set is empty is omitted entirely. Returns ``''`` when
    both halves are empty (nothing to report).
    """
    sections: list[str] = []
    if extra_in_doc:
        sections.append(
            f'drawn in {_ARCHITECTURE_DOC_REF} but NOT in {_TABLE_REF}:\n'
            + _render_edge_lines(extra_in_doc)
        )
    if missing_from_doc:
        sections.append(
            f'in {_TABLE_REF} but NOT drawn in {_ARCHITECTURE_DOC_REF}:\n'
            + _render_edge_lines(missing_from_doc)
        )
    if not sections:
        return ''
    sections.append(
        f'{_TABLE_REF} is the authority (task 4535) -- redraw {_ARCHITECTURE_DOC_REF} '
        'to match it.'
    )
    return '\n\n'.join(sections)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def check_architecture_transition_parity(
    root: Path = REPO_ROOT,
) -> frozenset[tuple[TaskStatus, TaskStatus]]:
    """Assert ARCHITECTURE.md section 3.1 draws exactly ``TRANSITIONS``' edges.

    Reads the doc, extracts and parses the diagram, reads the table (both
    at CALL time -- see :func:`table_transition_edges`), and diffs them.
    Raises :class:`AssertionError` (via :func:`format_parity_failure`) when
    either side of the diff is non-empty; otherwise returns the parsed doc
    edge set, so a caller can make further assertions (e.g. non-vacuousness)
    without re-parsing.
    """
    text = read_architecture_doc(root)
    body = extract_lifecycle_mermaid(text)
    doc_edges = parse_state_diagram_edges(body)
    table_edges = table_transition_edges()
    diff = diff_transition_edges(doc_edges, table_edges)
    if diff.missing_from_doc or diff.extra_in_doc:
        raise AssertionError(format_parity_failure(diff.missing_from_doc, diff.extra_in_doc))
    return doc_edges
