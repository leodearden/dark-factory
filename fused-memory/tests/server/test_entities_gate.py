"""Unit tests for the declared-referent write gate (task 3669, PRD leaf delta).

The gate is the POLICY half of the split leaf gamma drew deliberately:
``utils/referent_resolution`` REPORTS a conflict and never rejects, and
:func:`fused_memory.server.entities_gate.entities_gate` decides that a
conflict blocks the write.

Two arms live here; the third (CONFLICT) arrives in the sibling class below
once the gate can express it.  All of them are exercised against the PURE
function rather than through FastMCP, which is the whole reason the gate is a
module-level function instead of a ``create_mcp_server`` closure: it closes
over no server state, so it needs no server to test.

PRD resolved decision 3 — "reject on CONFLICT, never on ABSENCE" — is the
property the absence arm pins.  Rejecting on absence would lose memories from
every agent that does not retry (``/reflect`` at session end is the named
case), so the undeclared majority must flow through untouched.
"""

from __future__ import annotations

import json

import pytest

from fused_memory.server.entities_gate import entities_gate
from fused_memory.utils.referent_resolution import _DECLARED_REFERENT_HINT

GROUP = 'dark_factory'

#: Content whose prose names exactly one own-project referent, Task 3127.
_CITES_3127 = 'the fix for Task 3127 landed'
#: Content naming no referent of any kind — the scan comes back empty, which
#: gamma treats as UNINFORMATIVE rather than contradictory.
_CITES_NOTHING = 'the merge-lane hardening work'

#: The falsy NON-list shapes, as ``(id, value)`` pairs.  Their own arm below
#: pins the property that singles them out — that the gate short-circuits on
#: ``entities is None`` and never on ``not entities`` — and they are spliced
#: into ``_MALFORMED_SHAPES`` so the shared coverage sees them too, from ONE
#: spelling of the triple.
_FALSY_NON_LIST_SHAPES = (('empty-str', ''), ('empty-dict', {}), ('zero', 0))

#: Every shape gamma's TOTAL ``InputValidationError`` contract must reject,
#: hoisted to one list because THREE tests consume it: the two arms below, and
#: the tool-boundary arm in ``tests/server/test_entities_gate_ingestion.py``,
#: which imports it from here.  Kept as one list on purpose — a shape added to
#: a second copy and not the first would be silently unchecked for the property
#: the first pins, which is the lockstep duplication this batch keeps guarding
#: against.  The boundary import is what stops the pure-function claim and the
#: agent-visible behaviour from drifting apart: every shape here is reachable
#: through FastMCP only because both tools annotate ``entities`` as ``Any``.
_MALFORMED_SHAPES = [
    pytest.param('task 3127', id='a-str-not-a-list'),
    pytest.param([{'id': 'abc'}], id='non-digit-id'),
    pytest.param([{'id': 3127, 'projectId': 'reify'}], id='unrecognized-key'),
    pytest.param([{'id': True}], id='bool-id'),
    pytest.param([{'id': 3127, 'project_id': 'task'}], id='task-vocabulary-qualifier'),
    pytest.param([[1]], id='a-list-of-non-dicts'),
    pytest.param({'kind': 'task', 'id': 3127}, id='an-unwrapped-single-dict'),
    *(pytest.param(value, id=f'falsy-{label}') for label, value in _FALSY_NON_LIST_SHAPES),
]


class TestAbsenceIsNeverRejected:
    """The tri-state's two non-declaring arms, and the corroborating one.

    ``None`` short-circuits before any scan; ``[]`` is a REAL declaration
    ("considered, none apply") that still must not reject; and a declaration
    the scan cannot contradict is honoured.
    """

    def test_none_with_a_referent_in_the_prose_returns_none(self):
        assert entities_gate(None, content=_CITES_3127, group_id=GROUP) is None

    def test_none_with_no_referent_in_the_prose_returns_none(self):
        assert entities_gate(None, content=_CITES_NOTHING, group_id=GROUP) is None

    def test_the_empty_declaration_does_not_reject_even_when_the_scan_found_one(self):
        """``[]`` is tri-state arm 2, not a collapsed ``None``.

        The scan DID find Task 3127 and the declaration omits it — and that is
        still not a conflict.  Resolved decision 3 rejects on conflict, never
        on absence, and gamma's ``_conflicting_referents`` iterates the
        DECLARED entries, so an empty declaration has nothing to contradict.
        """
        assert entities_gate([], content=_CITES_3127, group_id=GROUP) is None

    def test_a_declaration_the_scan_cannot_speak_to_returns_none(self):
        """An empty per-kind scan is uninformative, never contradictory.

        The scanner's documented blind spots (bare-digit node names, title-only
        references, Greek-letter aliases) mean silence is silence.  Rejecting
        here would bounce an honest write whose prose simply named its task in
        a shape the scanner cannot see.
        """
        assert entities_gate(
            [{'kind': 'task', 'id': 3129}], content=_CITES_NOTHING, group_id=GROUP,
        ) is None

    def test_a_corroborated_declaration_returns_none(self):
        assert entities_gate(
            [{'kind': 'task', 'id': 3127}], content=_CITES_3127, group_id=GROUP,
        ) is None

    def test_agent_id_is_optional_on_the_accepted_path(self):
        """The parameter defaults, so a caller with no identity still gates."""
        assert entities_gate([], content=_CITES_3127, group_id=GROUP) is None


class TestMalformedDeclarationsAreRejected:
    """Gamma's ``InputValidationError`` contract is TOTAL, and lands here.

    Gamma states it wrote that contract so "delta's ``except
    InputValidationError`` gate catches all of them".  Each shape below leaves
    ``_declared_referents`` as exactly ``InputValidationError``, and the gate
    turns it into the flat house-shape block the sibling pre-service guards in
    ``server/tools.py`` already return.
    """

    @pytest.mark.parametrize('declared', _MALFORMED_SHAPES)
    def test_each_malformed_shape_blocks_the_write(self, declared):
        block = entities_gate(
            declared, content=_CITES_3127, group_id=GROUP, agent_id='claude-interactive',
        )

        assert isinstance(block, dict), f'expected a block, got {block!r}'
        assert block['error_type'] == 'ValidationError', f'{block!r}'
        assert isinstance(block['error'], str) and block['error'], f'{block!r}'
        assert block['agent_id'] == 'claude-interactive', f'{block!r}'
        assert block['content_excerpt'] == _CITES_3127[:200], f'{block!r}'

    @pytest.mark.parametrize('declared', _MALFORMED_SHAPES)
    def test_the_remediation_reaches_the_agent_inside_the_message(self, declared):
        """Gamma single-sources the accepted shape into the exception message.

        Imported from production rather than re-spelled here: a second copy of
        the hint text is exactly the lockstep duplication this batch keeps
        guarding against, and it would let the two drift silently.
        """
        block = entities_gate(declared, content=_CITES_3127, group_id=GROUP)

        assert isinstance(block, dict), f'expected a block, got {block!r}'
        assert _DECLARED_REFERENT_HINT in block['error'], f'{block!r}'

    def test_the_malformed_block_carries_no_separate_hint_key(self):
        """An exception has no room for a structured key, so gamma folds the
        remediation into the message.  A second copy under ``hint`` would be
        the same text maintained at two sites."""
        block = entities_gate([{'id': 'abc'}], content=_CITES_3127, group_id=GROUP)

        assert isinstance(block, dict), f'expected a block, got {block!r}'
        assert 'hint' not in block, f'{block!r}'

    def test_a_falsy_non_list_is_rejected_rather_than_read_as_absence(self):
        """The short-circuit is ``entities is None``, never ``not entities``.

        ``''``/``{}``/``0`` are wiring bugs, not declarations.  Treating them
        as absence would silently downgrade the write's referent source and
        lose the caller's intent invisibly — the silent degradation this
        repo's loud-over-silent norm forbids.

        These three are also spliced into ``_MALFORMED_SHAPES``, so the arms
        above and the tool boundary check them too.  This arm is kept separate
        anyway because it pins a DIFFERENT property — not "the gate rejects
        this shape" but "the gate did not read this shape as an absence" — and
        that is the claim whose failure message a reader needs when a future
        ``not entities`` shortcut appears.
        """
        for _label, falsy in _FALSY_NON_LIST_SHAPES:
            block = entities_gate(falsy, content=_CITES_3127, group_id=GROUP)
            assert isinstance(block, dict), f'{falsy!r} was read as absence: {block!r}'
            assert block['error_type'] == 'ValidationError', f'{falsy!r}: {block!r}'

    def test_the_malformed_block_is_json_safe(self):
        block = entities_gate([{'id': 'abc'}], content=_CITES_3127, group_id=GROUP)

        assert json.loads(json.dumps(block)) == block


class TestConflictingDeclarationsAreRejected:
    """The leaf's user-observable signal: a declaration its own prose refutes.

    Every case here reaches ``_conflicting_referents``, whose four documented
    choices this class pins from the OUTSIDE — the scoping rules are gamma's,
    and what delta adds is that a non-empty ``.conflicts`` blocks the write and
    names BOTH sides of the disagreement (INV-2, structured-facts-at-failure).
    """

    def test_the_prd_headline_row_blocks_and_names_both_sides(self):
        """Content says 3127, the caller declared 3129: the adjacent-number
        typo this PRD exists to catch."""
        block = entities_gate(
            [{'kind': 'task', 'id': 3129}],
            content=_CITES_3127,
            group_id=GROUP,
            agent_id='claude-interactive',
        )

        assert isinstance(block, dict), f'expected a block, got {block!r}'
        assert block['error'] == 'declared_referent_conflict', f'{block!r}'
        assert block['error_type'] == 'DeclaredReferentConflictRejected', f'{block!r}'
        assert block['agent_id'] == 'claude-interactive', f'{block!r}'
        assert block['content_excerpt'] == _CITES_3127[:200], f'{block!r}'
        assert block['hint'], f'a rejection with no remediation is a dead end: {block!r}'
        assert block['conflicts'] == ['Task 3129'], f'{block!r}'
        assert block['declared'] == ['Task 3129'], f'{block!r}'
        assert block['content_referents'] == ['Task 3127'], f'{block!r}'

    def test_the_conflict_block_is_json_safe(self):
        block = entities_gate(
            [{'kind': 'task', 'id': 3129}], content=_CITES_3127, group_id=GROUP,
        )

        assert isinstance(block, dict), f'expected a block, got {block!r}'
        assert json.loads(json.dumps(block)) == block

    def test_an_ambiguous_only_scan_still_names_the_content_side(self):
        """The correctness trap, and the reason the content side is
        ``refs + ambiguous`` rather than ``refs``.

        This content claims 2500 BOTH bare and foreign-qualified, so
        ``scan_content`` routes both spellings to ``LabelScan.ambiguous`` and
        leaves ``.refs`` EMPTY (the fixture is
        tests/test_canonical_labels.py's own ambiguity split, reused rather
        than reinvented).  ``_conflicting_referents`` tests membership against
        ``refs | ambiguous``, so the scan still "saw this kind" and the
        declaration still conflicts — but a block reporting ``refs`` alone
        would say the content cites NOTHING while rejecting the write for
        contradicting the content, which reads as a guard malfunction rather
        than a decision.
        """
        content = 'dark_factory:2500 blocks task 2500 here'
        block = entities_gate(
            [{'kind': 'task', 'id': 3129}], content=content, group_id='reify',
        )

        assert isinstance(block, dict), f'the ambiguous scan did not reject: {block!r}'
        assert block['error_type'] == 'DeclaredReferentConflictRejected', f'{block!r}'
        assert block['conflicts'] == ['Task 3129'], f'{block!r}'
        assert block['content_referents'], (
            f'the content side must never be empty beside a conflict: {block!r}'
        )
        assert block['content_referents'] == ['dark_factory:2500', 'Task 2500'], f'{block!r}'

    def test_a_declaration_that_resolves_an_ambiguity_does_not_conflict(self):
        """Naming an ambiguous referent SETTLES the contest the prose left
        open — gamma's choice 3, and the single most useful thing a
        declaration can do."""
        assert entities_gate(
            [{'kind': 'task', 'id': 2500}],
            content='dark_factory:2500 blocks task 2500 here',
            group_id='reify',
        ) is None

    def test_the_project_axis_conflicts_and_spells_the_foreign_node_name(self):
        """The cross-project collapse this PRD exists to detect.

        The prose names 'reify:3129' — a DIFFERENT project's task carrying the
        declared number — while the caller declared the OWN-project 'Task
        3129'.  An own-project declaration keeps the whole-kind scan bucket
        (gamma's choice 2 narrows only the FOREIGN side), so the foreign
        reference is evidence about it: not silence, but the number collapse
        that is invisible to any rule comparing bare numbers.

        The block must spell the content side 'reify:3129' rather than 'Task
        3129', or the reader cannot see what the disagreement IS.

        DIRECTION MATTERS, and the mirror image is the sibling test below.
        The plan's step-3 sketch asked for the INVERSE case — a foreign
        declaration against bare local prose — which is structurally identical
        to its own adjacent "a foreign declaration the prose never mentions
        does NOT conflict" bullet and measures as a NON-conflict (verified:
        declared reify:3127 against 'the fix for Task 3127 landed' yields
        conflicts=()).  Gamma's choice 2 is deliberate — the scan reaches a
        foreign task ONLY through an explicit qualifier, so its silence about
        that project is silence — and making that case reject would require
        the gate to grow policy of its own, which every design decision in
        this leaf forbids.  The property the plan wanted pinned (the project
        axis conflicts; foreign node names are spelled foreign) is pinned
        here and below, in the two directions that actually carry it.
        """
        block = entities_gate(
            [{'kind': 'task', 'id': 3129}],
            content='mirrors reify:3129',
            group_id=GROUP,
        )

        assert isinstance(block, dict), f'the project axis did not reject: {block!r}'
        assert block['declared'] == ['Task 3129'], f'{block!r}'
        assert block['conflicts'] == ['Task 3129'], f'{block!r}'
        assert block['content_referents'] == ['reify:3129'], f'{block!r}'

    def test_a_foreign_declaration_the_prose_contradicts_is_spelled_foreign(self):
        """The declared side spelled foreign, and choice 2's other half.

        The prose DOES name that project ('reify:3129'), so the foreign
        declaration's narrowed bucket is non-empty and its silence argument no
        longer applies — 'reify:132' is contradicted.  Pins that a foreign
        referent reaches the agent as 'reify:132' and never as 'Task 132',
        which is the spelling the graph uses and the one leaf zeta emits.
        """
        block = entities_gate(
            [{'kind': 'task', 'id': 132, 'project_id': 'reify'}],
            content='mirrors reify:3129',
            group_id=GROUP,
        )

        assert isinstance(block, dict), f'the foreign conflict did not reject: {block!r}'
        assert block['declared'] == ['reify:132'], f'{block!r}'
        assert block['conflicts'] == ['reify:132'], f'{block!r}'
        assert block['content_referents'] == ['reify:3129'], f'{block!r}'

    def test_a_foreign_declaration_the_prose_never_mentions_does_not_conflict(self):
        """Gamma's choice 2: the scan reaches a foreign task ONLY through an
        explicit qualifier, so its silence about that project is silence, not
        disagreement.  Read as contradiction, this rejected an honest write
        whenever ANY local task number happened to appear in the prose."""
        assert entities_gate(
            [{'kind': 'task', 'id': 132, 'project_id': 'reify'}],
            content='Fixed the bug in Task 3127.',
            group_id=GROUP,
        ) is None

    def test_a_multi_entry_partial_conflict_names_only_the_contradicted_entry(self):
        """The per-referent verdict, not a set-level disjointness one.

        3127 corroborates and 3129 is the adjacent-number typo; a
        "declared and scanned are disjoint" verdict would pass this write.
        ``declared`` still lists BOTH, so the agent can see what it sent
        beside what was refused.
        """
        block = entities_gate(
            [{'id': 3127}, {'id': 3129}], content=_CITES_3127, group_id=GROUP,
        )

        assert isinstance(block, dict), f'the partial conflict did not reject: {block!r}'
        assert block['conflicts'] == ['Task 3129'], f'{block!r}'
        assert block['declared'] == ['Task 3127', 'Task 3129'], f'{block!r}'

    def test_a_self_qualified_declaration_is_not_a_conflict(self):
        """Gamma reclassifies a declaration qualified with the LOCAL project to
        an own-project referent, exactly as ``scan_content`` does.  Without
        that, a caller who was right could never compare equal to the scanned
        form."""
        assert entities_gate(
            [{'id': 3127, 'project_id': 'dark_factory'}],
            content=_CITES_3127,
            group_id=GROUP,
        ) is None


class TestTheGateStaysPermissiveWhileTheProducerNarrows:
    """The gate acquires no project registry (task 5262 workstream B).

    ``MemoryService``'s three ``resolve_referents`` call sites now narrow with
    ``self._known_projects``, so a junk qualifier stops minting a referent on
    the write path. This call deliberately does NOT: the gate closes over no
    server state — the property the module docstring above opens with — so it
    holds no registry to narrow WITH, and acquiring one would only make it
    stop catching conflicts it catches today. Rejection is on CONFLICT and
    never on ABSENCE, so a narrowed scan here can subtract rejections and can
    never add one.

    A regression fence, not a new behaviour: green before workstream B and
    required to stay green after it.
    """

    def test_a_junk_qualified_conflict_is_still_rejected(self):
        """'evil_proj' is in no registry the factory holds, so a narrowed scan
        would drop it, empty the content side, and let a contradicted
        declaration through."""
        block = entities_gate(
            [{'kind': 'task', 'id': 3129}],
            content='mirrors evil_proj:132',
            group_id=GROUP,
            agent_id='claude-interactive',
        )

        assert isinstance(block, dict), f'the junk-qualified conflict was not caught: {block!r}'
        assert block['error_type'] == 'DeclaredReferentConflictRejected', f'{block!r}'
        assert block['conflicts'] == ['Task 3129'], f'{block!r}'
        assert block['content_referents'] == ['evil_proj:132'], f'{block!r}'
