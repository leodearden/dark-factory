"""shared.merge_state's SUBMIT vocabulary agrees with orchestrator's ``MergeOutcome.status``.

The rest of shared.merge_state's vocabulary is tested in
shared/tests/test_merge_state.py. This parity pin reads a sibling package's
source, so it lives here.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from shared.merge_state import SUBMIT_TERMINAL, MergeSubmitStatus
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MERGE_TYPES_SOURCE = (
    _REPO_ROOT / 'orchestrator' / 'src' / 'orchestrator' / 'merge_lane' / 'types.py'
)


def _merge_outcome_status_literal() -> frozenset[str] | None:
    """``MergeOutcome.status``'s ``Literal`` members, read from orchestrator SOURCE.

    ``escalation/src/escalation/server.py`` returns ``'status': outcome.status``
    VERBATIM, so that annotation IS the submit wire vocabulary's terminal half —
    which makes it the one legacy set that can be DERIVED rather than inlined.

    Derived, not copied, deliberately.  A hand-inlined frozenset here would be
    compared against the hand-written ``SUBMIT_TERMINAL`` in
    ``shared/src/shared/merge_state.py``: two copies of the same list agreeing
    with each other while the annotation they both claim to mirror widens
    underneath them.  That is the exact drift class this task exists to close,
    and it is how the four SKILL.md ``SUBMIT_TERMINAL`` spans came to be wrong in
    the first place.  Unlike the POLL half — where every partition is derived and
    self-reddening — nothing else in this chain would notice: ``MergeSubmitStatus``
    has no production consumer, because the server forwards ``outcome.status``
    rather than constructing a member.

    Read by ``ast`` rather than imported: ``orchestrator`` is not importable from
    the shared test environment, and reading the file needs no interpreter.  Returns ``None`` — for the caller to skip on — only when the
    file is ABSENT (``shared/`` vendored without its siblings).  Every other
    shape failure raises: a renamed class, a retyped annotation or a
    computed ``Literal`` must send a reader here, not pass quietly.
    """
    if not _MERGE_TYPES_SOURCE.is_file():
        return None

    tree = ast.parse(_MERGE_TYPES_SOURCE.read_text())
    annotation: ast.expr | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'MergeOutcome':
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                    and stmt.target.id == 'status'
                ):
                    annotation = stmt.annotation
            break

    assert annotation is not None, (
        f'{_MERGE_TYPES_SOURCE} defines no `MergeOutcome.status` annotation. '
        'The submit vocabulary is derived from it, so a rename leaves '
        'shared.merge_state.SUBMIT_TERMINAL pinned to nothing — re-point this '
        'reader at the new home.'
    )

    is_literal = isinstance(annotation, ast.Subscript) and (
        (isinstance(annotation.value, ast.Name) and annotation.value.id == 'Literal')
        or (isinstance(annotation.value, ast.Attribute) and annotation.value.attr == 'Literal')
    )
    assert is_literal, (
        f'{_MERGE_TYPES_SOURCE}::MergeOutcome.status is annotated '
        f'{ast.dump(annotation)}, not a `Literal[...]`.  SUBMIT_TERMINAL mirrors '
        'that Literal member-for-member; if the annotation is now computed, the '
        'mirror has to be re-derived some other way rather than silently skipped.'
    )
    assert isinstance(annotation, ast.Subscript)  # narrowing for the reader below

    elements = (
        annotation.slice.elts
        if isinstance(annotation.slice, ast.Tuple)
        else [annotation.slice]
    )
    members: list[str] = []
    for element in elements:
        assert isinstance(element, ast.Constant) and isinstance(element.value, str), (
            f'{_MERGE_TYPES_SOURCE}::MergeOutcome.status names a non-string '
            f'Literal member {ast.dump(element)}; the wire vocabulary is strings.'
        )
        members.append(element.value)

    assert members, (
        f'{_MERGE_TYPES_SOURCE}::MergeOutcome.status parsed to an EMPTY Literal. '
        'An empty expected set compares equal to nothing and would report '
        'SUBMIT_TERMINAL as entirely wrong.'
    )
    return frozenset(members)


# escalation/src/escalation/server.py::merge_request — the two NON-terminal
# response shapes: wait_secs=0 dispatched -> 'queued', coalesced -> 'attached'
# (a wait_secs>0 timeout also returns the 'queued' shape).
_LEGACY_SUBMIT_NON_TERMINAL = frozenset({'queued', 'attached'})


class TestMergeSubmitStatusEnum:
    def test_exact_vocabulary(self):
        # 14 = MergeOutcome.status's 12-member Literal + the 2 non-terminal
        # response shapes.  NOT the 6 the PRD's illustrative contract block
        # sketches — PRD Open Question 3 delegates membership to this task,
        # and the server returns outcome.status verbatim.
        terminal = _merge_outcome_status_literal()
        if terminal is None:
            pytest.skip(f'{_MERGE_TYPES_SOURCE} is absent in this tree')
        assert {s.value for s in MergeSubmitStatus} == terminal | _LEGACY_SUBMIT_NON_TERMINAL
        # The count is stated as well as derived: a member ADDED to the Literal
        # and to MergeSubmitStatus in the same edit satisfies the set equality
        # above, and this line is what then sends the author to the four SKILL.md
        # SUBMIT_TERMINAL spans that also need it.
        assert len(MergeSubmitStatus) == 14


class TestSubmitPartitions:
    def test_submit_terminal_matches_merge_outcome_status(self):
        # Derived from orchestrator/src/orchestrator/merge_types.py's source, so
        # a widening of MergeOutcome.status reddens HERE — and with it every
        # SKILL.md SUBMIT_TERMINAL span, via the drift guard in
        # scripts/tests/test_merge_state_vocabulary_consistency.py.
        expected = _merge_outcome_status_literal()
        if expected is None:
            pytest.skip(
                f'{_MERGE_TYPES_SOURCE} is absent — the submit vocabulary has '
                'nothing to be pinned against in this tree'
            )
        assert {s.value for s in SUBMIT_TERMINAL} == expected, (
            'shared.merge_state.SUBMIT_TERMINAL has drifted from '
            'orchestrator/src/orchestrator/merge_types.py::MergeOutcome.status.  '
            'The server returns `outcome.status` verbatim, so that Literal IS the '
            'submit wire vocabulary: add the new member to MergeSubmitStatus and '
            'fix every SKILL.md SUBMIT_TERMINAL span the guard then reddens.'
        )
