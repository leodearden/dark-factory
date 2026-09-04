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

_NEXT_HEADING_RE = re.compile(r'^### ', re.MULTILINE)
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

    Slices *text* from :data:`SECTION_HEADING` to the next ``### `` heading,
    then requires that slice to contain EXACTLY one ```` ```mermaid ```` fence.
    Raises :class:`RuntimeError` — naming what was expected and what was
    found — when the heading is absent, or the section holds zero or more
    than one mermaid fence. A tolerant "best effort" extractor here would
    let a renamed heading or a relocated diagram silently produce an empty
    or wrong result, which is precisely the drift this guard exists to
    catch.
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
    ``merge_deferred``, ``infra_hold``). Both the underscore-folded id and
    the literal value map to the same status, so this single table serves
    every status regardless of whether its value contains a hyphen.

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
        for alias in (status.value, status.value.replace('-', '_')):
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
