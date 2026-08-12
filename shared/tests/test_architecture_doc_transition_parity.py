"""Doc↔code parity for ARCHITECTURE.md §3.1's task-status state diagram.

The §3.1 ``stateDiagram-v2`` fence is a *rendering* of Table A — the
enumerated transition union in ``shared/src/shared/task_transitions.py``.  It
is the only prose artifact in that document with a live in-code oracle, so it
is the only one worth pinning mechanically: this module asserts set-equality
between the edges drawn in the diagram and the edges in the exported
``TRANSITIONS`` table.

ANTI-ROT CONTRACT (the reason this module exists — task 3544 / PRD leaf κ):
the diagram silently lost **19** of Table A's edges between the day it was
drawn and the day this test was written, because nothing checked it.  A future
edit that adds a pair to ``_UNION`` is *expected* to turn this test red until
ARCHITECTURE.md §3.1 is redrawn to match.  That red is the feature, not a
flake — redraw the diagram, do not weaken the assertion.

Deliberately asserts against the ``__all__``-exported ``TRANSITIONS`` mapping
rather than the private ``_UNION``, so a future actor-specific restriction
that narrows one actor's row (as ``RECONCILIATION`` already does) still
compares against the same union through the public surface.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from shared.task_statuses import TaskStatus
from shared.task_transitions import TRANSITIONS

# shared/tests/<this file> -> parents[2] is the dark-factory checkout root.
# Same idiom as shared/tests/conftest.py:21 (and its neighbours), which
# documents why it holds inside a task worktree too: verify runs
# `cd shared && uv run pytest tests/`, so rootdir is the subproject and the
# repo-root conftest is never loaded.
REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE_MD = REPO_ROOT / 'ARCHITECTURE.md'

# The diagram spells nodes with underscores because `in-progress` is not a
# legal mermaid state id; the TaskStatus vocabulary uses hyphens.
_MERMAID_FENCE = re.compile(r'^```mermaid$(?P<body>.*?)^```$', re.MULTILINE | re.DOTALL)
# `a --> b`, optionally `a --> b: label`.  Labels are decoration, not data.
_EDGE = re.compile(r'^\s*(?P<src>\S+)\s*-->\s*(?P<dst>[^\s:]+)\s*(?::.*)?$')
_PSEUDO_STATE = '[*]'


def _normalize(node: str) -> str:
    """Map a mermaid-legal state id onto its ``TaskStatus`` value spelling."""
    return node.replace('_', '-')


def _parse_state_diagram(text: str) -> set[tuple[str, str]]:
    """Return the ``(src, dst)`` edges of the one ``stateDiagram-v2`` fence.

    Selection is BY the ``stateDiagram-v2`` marker, never "the first mermaid
    fence" — ARCHITECTURE.md §6 carries a second mermaid fence (a
    ``flowchart LR`` of the escalation ladder) which must not be parsed as a
    state machine.  Edges touching the ``[*]`` start/end pseudo-state are
    dropped: they are not status transitions and have no Table A counterpart.
    """
    bodies = [
        m.group('body') for m in _MERMAID_FENCE.finditer(text) if 'stateDiagram-v2' in m.group('body')
    ]
    assert len(bodies) == 1, (
        f'expected exactly one ```mermaid fence containing `stateDiagram-v2` in '
        f'ARCHITECTURE.md, found {len(bodies)}; this parser (and the parity it '
        f'enforces) assumes a single state-diagram artifact'
    )
    edges: set[tuple[str, str]] = set()
    for line in bodies[0].splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('%%') or stripped == 'stateDiagram-v2':
            continue
        match = _EDGE.match(stripped)
        if match is None:
            continue
        src, dst = match.group('src'), match.group('dst')
        if src == _PSEUDO_STATE or dst == _PSEUDO_STATE:
            continue
        edges.add((_normalize(src), _normalize(dst)))
    return edges


def _fmt(edges: set[tuple[str, str]]) -> str:
    return '\n'.join(f'  {src} --> {dst}' for src, dst in sorted(edges)) or '  (none)'


@pytest.fixture(scope='module')
def diagram_edges() -> set[tuple[str, str]]:
    if not ARCHITECTURE_MD.is_file():
        pytest.skip(f'ARCHITECTURE.md not present in this checkout ({ARCHITECTURE_MD})')
    return _parse_state_diagram(ARCHITECTURE_MD.read_text(encoding='utf-8'))


# The full transition union, expressed through the exported table.  Every
# non-RECONCILIATION actor maps to `_UNION`; RECONCILIATION is a strict subset
# of it, so the union over all rows is exactly `_UNION`.
EXPECTED: frozenset[tuple[str, str]] = frozenset(
    (str(frm), str(to)) for pairs in TRANSITIONS.values() for frm, to in pairs
)
_STATUS_VALUES: frozenset[str] = frozenset(s.value for s in TaskStatus)


class TestArchitectureDiagramMatchesTransitionTable:
    """ARCHITECTURE.md §3.1's diagram == `shared.task_transitions.TRANSITIONS`."""

    def test_every_diagram_node_is_a_real_status(self, diagram_edges):
        """A typo'd node must fail loudly here, not drown in the edge set-diff."""
        nodes = {n for edge in diagram_edges for n in edge}
        unknown = sorted(nodes - _STATUS_VALUES)
        assert not unknown, (
            f'ARCHITECTURE.md §3.1 names state(s) that are not in the TaskStatus '
            f'vocabulary: {unknown}\n'
            f'legal values (underscores in mermaid map to hyphens): '
            f'{sorted(_STATUS_VALUES)}'
        )

    def test_diagram_has_no_illegal_edge(self, diagram_edges):
        """Every drawn edge must exist in Table A — no invented transitions."""
        extra = diagram_edges - EXPECTED
        assert not extra, (
            f'ARCHITECTURE.md §3.1 draws {len(extra)} edge(s) that are NOT in '
            f'shared/src/shared/task_transitions.py `_UNION`:\n{_fmt(extra)}\n'
            f'Either the edge is real (add it to the table, with its call-site '
            f'anchor) or the diagram is wrong (remove it).'
        )

    def test_diagram_omits_no_legal_edge(self, diagram_edges):
        """Table A is the authority; §3.1 must render all of it."""
        missing = set(EXPECTED) - diagram_edges
        assert not missing, (
            f'ARCHITECTURE.md §3.1 omits {len(missing)} legal transition(s) from '
            f'shared/src/shared/task_transitions.py `_UNION`:\n{_fmt(missing)}\n'
            f'Redraw the §3.1 stateDiagram-v2 fence to include them (mermaid state '
            f'ids use underscores: in_progress / merge_deferred / infra_hold).'
        )

    def test_deterministic_gate_edge_is_drawn(self, diagram_edges):
        """`pending -> blocked` is load-bearing for §3.8 and must be legible."""
        assert ('pending', 'blocked') in diagram_edges, (
            'ARCHITECTURE.md §3.1 does not draw `pending --> blocked`, the edge '
            "§3.8's deterministic-tasks contract runs on: a `metadata."
            'always_escalates` gate parks a never-dispatched `pending` task at '
            '`blocked` while its born-at-L2 record is open. The writers are '
            '`DeterministicRunner._block_with_infra_issue` and '
            '`_block_with_stop_instruction` (orchestrator/src/orchestrator/'
            'deterministic_runner.py) — neither goes through `_mark_blocked`.'
        )
