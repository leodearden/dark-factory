"""Doc↔code parity for ARCHITECTURE.md §3.1's task-status state diagram.

The §3.1 ``stateDiagram-v2`` fence is a *rendering* of Table A — the
enumerated transition union in ``shared/src/shared/task_transitions.py``.  It
is the only prose artifact in that document with a live in-code oracle, so it
is the only one worth pinning mechanically: this module asserts set-equality
between the edges drawn in the diagram and the edges in the exported
``TRANSITIONS`` table, and pins the one numeric claim the surrounding prose
makes about it (the stated edge count).

ANTI-ROT CONTRACT (the reason this module exists — task 3544 / PRD leaf κ):
the diagram silently lost **19** of Table A's edges between the day it was
drawn and the day this test was written, because nothing checked it.  A future
edit that adds a pair to ``_UNION`` is *expected* to turn this test red until
ARCHITECTURE.md §3.1 is redrawn to match.  That red is the feature, not a
flake — redraw the diagram, do not weaken the assertion.  For the same reason
a missing ``ARCHITECTURE.md`` is a hard failure, never a skip: a skip would
retire the guarantee silently, which is the failure mode this module exists to
prevent.

Deliberately asserts against the ``__all__``-exported ``TRANSITIONS`` mapping
rather than the private ``_UNION``, so a future actor-specific restriction
that narrows one actor's row (as ``RECONCILIATION`` already does) still
compares against the same union through the public surface.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest

from shared.task_statuses import TaskStatus
from shared.task_transitions import TRANSITIONS, ActorClass

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
_ARROW = '-->'
_PSEUDO_STATE = '[*]'
# The one numeric claim §3.1's prose makes about the diagram.  It is next to
# the artifact this module pins, so it is pinned by the same oracle rather
# than left as unguarded prose that rots the next time `_UNION` grows.
_EDGE_COUNT_CLAIM = re.compile(r'every one of its (?P<count>\d+) edges')


def _normalize(node: str) -> str:
    """Map a mermaid-legal state id onto its ``TaskStatus`` value spelling."""
    return node.replace('_', '-')


def _parse_state_diagram(text: str) -> set[tuple[str, str]]:
    """Return the ``(src, dst)`` edges of the one ``stateDiagram-v2`` fence.

    Selection is BY the ``stateDiagram-v2`` marker, never "the first mermaid
    fence" — ARCHITECTURE.md carries three mermaid fences (a ``flowchart TB``
    process topology in §1, this state diagram in §3.1, and a ``flowchart LR``
    escalation ladder in §6), and only the state diagram is a rendering of
    Table A.  Edges touching the ``[*]`` start/end pseudo-state are dropped:
    they are not status transitions and have no Table A counterpart.

    Any line inside the fence that contains an arrow but does not match
    ``_EDGE`` is a hard failure rather than a silent skip — a dropped edge
    would make the "no illegal edge" assertion pass vacuously.
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
            assert _ARROW not in stripped, (
                f'unparseable transition line in the §3.1 state diagram: {stripped!r}\n'
                f'It draws an arrow but does not match {_EDGE.pattern!r}, so this parser '
                f'would drop the edge and the parity assertions would pass vacuously on '
                f'it. Write the line as `src --> dst` (optionally `: label`), or widen '
                f'`_EDGE` deliberately.'
            )
            continue
        src, dst = match.group('src'), match.group('dst')
        if src == _PSEUDO_STATE or dst == _PSEUDO_STATE:
            continue
        edges.add((_normalize(src), _normalize(dst)))
    return edges


def _fmt(edges: set[tuple[str, str]]) -> str:
    return '\n'.join(f'  {src} --> {dst}' for src, dst in sorted(edges)) or '  (none)'


def _read_architecture() -> str:
    assert ARCHITECTURE_MD.is_file(), (
        f'ARCHITECTURE.md not found at {ARCHITECTURE_MD} — the `parents[2]` repo-root '
        f'idiom this module relies on no longer holds. Fix the path; do NOT convert '
        f'this to a skip: the doc is unconditionally present in every checkout, so a '
        f'skip here would retire the §3.1 anti-rot guarantee while the suite stayed '
        f'green.'
    )
    return ARCHITECTURE_MD.read_text(encoding='utf-8')


@pytest.fixture(scope='module')
def architecture_text() -> str:
    return _read_architecture()


@pytest.fixture(scope='module')
def diagram_edges(architecture_text: str) -> set[tuple[str, str]]:
    return _parse_state_diagram(architecture_text)


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

    def test_prose_edge_count_matches_the_table(self, architecture_text):
        """§3.1's prose states the edge count; pin it to the same oracle.

        The diagram itself is kept honest by the set-equality tests above, but
        the sentence next to it ("every one of its N edges is a `TRANSITIONS`
        pair") is prose and would otherwise go stale the moment `_UNION` grows.
        """
        claims = _EDGE_COUNT_CLAIM.findall(architecture_text)
        assert len(claims) == 1, (
            f'expected exactly one "every one of its N edges" claim in '
            f'ARCHITECTURE.md §3.1, found {len(claims)}: {claims}. If the sentence '
            f'was reworded, update `_EDGE_COUNT_CLAIM` — do not leave the count '
            f'unpinned.'
        )
        assert int(claims[0]) == len(EXPECTED), (
            f'ARCHITECTURE.md §3.1 claims the diagram has {claims[0]} edges, but '
            f'shared/src/shared/task_transitions.py `_UNION` has {len(EXPECTED)}. '
            f'Redraw the diagram and update the sentence together.'
        )

    def test_deterministic_gate_edge_is_in_the_table(self):
        """`pending -> blocked` is load-bearing for §3.8 and must stay legal.

        Asserted against the authority, not the drawing: the diagram side is
        already covered by the set-equality tests above, so the residual risk
        is the edge being dropped from `_UNION` *and* the diagram together,
        which only an assertion against `TRANSITIONS` catches.
        """
        assert (TaskStatus.PENDING, TaskStatus.BLOCKED) in TRANSITIONS[ActorClass.DETERMINISTIC], (
            '`pending -> blocked` is not a legal DETERMINISTIC transition, but it is '
            "the edge §3.8's deterministic-tasks contract runs on: a `metadata."
            'always_escalates` gate parks a never-dispatched `pending` task at '
            '`blocked` while its born-at-L2 record is open. The writers are '
            '`DeterministicRunner._file_infra_issue_and_block` and '
            '`_file_stop_instruction_and_block` (orchestrator/src/orchestrator/'
            'deterministic_runner.py) — neither goes through `_mark_blocked`.'
        )


def _fence(body: str) -> str:
    return '```mermaid\n' + textwrap.dedent(body).strip('\n') + '\n```\n'


class TestParseStateDiagram:
    """The parser's own contract, over synthetic documents.

    Exercising it only against the real ARCHITECTURE.md leaves the
    "no illegal edge" direction able to pass vacuously: an illegal edge drawn
    in a syntax `_EDGE` happens not to match would simply vanish. These pin
    the selection and rejection behaviour the parser documents.
    """

    _STATE_FENCE = _fence(
        """
        stateDiagram-v2
        %% a comment line
        [*] --> pending: normal submit
        pending --> in_progress: dispatch
        in_progress --> done
        done --> [*]
        """
    )

    def test_parses_labeled_and_bare_edges(self):
        assert _parse_state_diagram(self._STATE_FENCE) == {
            ('pending', 'in-progress'),
            ('in-progress', 'done'),
        }

    def test_drops_pseudo_state_edges(self):
        edges = _parse_state_diagram(self._STATE_FENCE)
        assert not [e for e in edges if _PSEUDO_STATE in e]

    def test_normalizes_underscores_to_status_spelling(self):
        doc = _fence(
            """
            stateDiagram-v2
            in_progress --> merge_deferred
            merge_deferred --> infra_hold
            """
        )
        assert _parse_state_diagram(doc) == {
            ('in-progress', 'merge-deferred'),
            ('merge-deferred', 'infra-hold'),
        }

    def test_ignores_flowchart_fences(self):
        doc = (
            _fence(
                """
                flowchart TB
                    A["a box"] --> B["another box"]
                """
            )
            + '\nsome prose\n\n'
            + self._STATE_FENCE
            + _fence(
                """
                flowchart LR
                    L0["L0"] --> L1["L1"]
                """
            )
        )
        assert _parse_state_diagram(doc) == {
            ('pending', 'in-progress'),
            ('in-progress', 'done'),
        }

    def test_rejects_two_state_diagram_fences(self):
        with pytest.raises(AssertionError, match='exactly one'):
            _parse_state_diagram(self._STATE_FENCE + '\n' + self._STATE_FENCE)

    def test_rejects_a_document_with_no_state_diagram(self):
        doc = _fence(
            """
            flowchart LR
                A --> B
            """
        )
        with pytest.raises(AssertionError, match='found 0'):
            _parse_state_diagram(doc)

    def test_unparseable_arrow_line_fails_loudly(self):
        """A drawn-but-unmatched edge must not vanish into a silent `continue`."""
        doc = _fence(
            """
            stateDiagram-v2
            pending --> in_progress
            blocked --> in_progress --> done
            """
        )
        with pytest.raises(AssertionError, match='unparseable transition line'):
            _parse_state_diagram(doc)
