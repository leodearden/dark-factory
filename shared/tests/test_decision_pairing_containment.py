"""THE containment measurement for semantic cross-pairing (task 3967).

This file converts an argument into an observation. The originating escalation
reasoned FROM THE DETECTOR'S CONTRACT that the write-time markup tripwire
cannot see a cross-paired ``add_design_decision`` call. That reasoning is
correct, and here it is measured instead: the real ``MarkupGuardMiddleware`` is
driven through the real in-process harness with a real cross-paired call, under
both policy tiers, and observed to admit it unmodified with no fact emitted.

## What is deliberately NOT asserted here

That the middleware is registered on NO production server. That is TRUE as of
2026-08-16 — the toolcall-markup PRD's leaf gamma (task **3690**) is still
pending, so containment at plan-tools is currently zero for BOTH damage classes
— and it is a real and damning finding. But as a test it would go RED the
moment 3690 lands, punishing exactly the work that closes the gap. It belongs
in ``docs/plan-decision-cross-pairing.md`` as a dated observation, and it is
recorded there.

What IS asserted here is durable under 3690 landing: the middleware decides by
running ``toolcall_markup.detect`` over each incoming string argument, so a
cross-paired call — two pieces of clean, well-formed prose — is admitted
however the middleware is wired up.

## Sentinel-literal hazard — DO NOT "helpfully" un-escape these

Every envelope literal in this file is spelled with the ``\\x3c`` escape for
the opening angle bracket, exactly as ``shared/tests/test_mcp_markup_middleware.py``
and ``shared/src/shared/toolcall_markup.py`` require. Writing one verbatim here
would force any agent editing this file to emit that literal inside its own
tool-call envelope. Leave it escaped.
"""
from __future__ import annotations

import contextlib

import pytest

# Bare cross-file import, resolved by conftest's tests/ sys.path insert — the
# `test_toolcall_markup_corpus.py` precedent. The harness is REUSED rather than
# cloned so there is exactly one mechanism for driving the middleware (INV-5),
# and so this measurement runs against the real policy enum and the real fact
# sink rather than a second, drifting copy.
from test_decision_pairing import load_corpus, replay
from test_mcp_markup_middleware import build_harness

from shared.decision_pairing import detect_mispairing
from shared.mcp_markup_middleware import RepairPolicy
from shared.toolcall_markup import detect

BOTH_POLICIES = pytest.mark.parametrize(
    'policy',
    [RepairPolicy.REJECT_WITH_REPAIR, RepairPolicy.FORWARD_REPAIR],
    ids=['reject_with_repair', 'forward_repair'],
)

# A cross-paired pair, modelled on the real 3727[0] shape: the decision-line
# states one choice and the rationale argues for an entirely different one.
# Both halves are clean, well-formed prose. NOTHING about either string is
# malformed — that is the whole point.
CROSS_PAIRED_DECISION = (
    'Return `Path | None` from `durable_archive_path`, never `bool`; callers '
    'spell existence as `is not None`.'
)
CROSS_PAIRED_RATIONALE = (
    'The emit-site guard has to tolerate a project_root that does not resolve '
    'under the archive root, because the offline lane seeds it from a '
    'different parent and the two only coincide on the main checkout.'
)


class TestDetectorBlindness:
    """The two damage classes are DISJOINT AT THE DETECTOR.

    This is why the envelope detector cannot simply be extended to cover this
    damage class, and why ``shared.decision_pairing`` is a separate module
    rather than a widening of ``shared.toolcall_markup``.
    """

    def test_the_envelope_detector_is_blind_to_every_envelope_clean_positive(self) -> None:
        """``detect`` returns None on exactly the strings the new predicate fires on.

        Restricted to corpus positives whose committed record declares them
        envelope-clean — the other positives happen to carry BOTH kinds of
        damage at once (task 3382's plan is the named example), which is a real
        and interesting shape but not the one this assertion is about.
        """
        checked = 0
        for record in load_corpus():
            if record['expect'] != 'mispaired' or not record['envelope_clean']:
                continue
            checked += 1
            where = f"{record['task_id']}[{record['index']}]"
            assert detect(record['text']) is None, f'{where}: decision'
            assert detect(record['rationale_text']) is None, f'{where}: rationale'
            assert replay(record) is not None, f'{where}: the new predicate must fire'
        assert checked, 'no envelope-clean positive in the corpus — the pin is vacuous'

    def test_the_hand_authored_cross_paired_pair_is_envelope_clean(self) -> None:
        """Guard for the harness measurements below.

        If the specimen ever acquired an envelope literal, the admission tests
        would start measuring the ENVELOPE path and silently stop measuring
        this one — passing for the wrong reason.
        """
        assert detect(CROSS_PAIRED_DECISION) is None
        assert detect(CROSS_PAIRED_RATIONALE) is None


class TestTripwireAdmitsCrossPairedCalls:
    """THE measurement: the C2 tripwire admits a cross-paired call, unmodified.

    Driven through the real ``MarkupGuardMiddleware`` via the shared in-process
    harness. ``REJECT_WITH_REPAIR`` is plan-tools' declared policy in the
    toolcall-markup PRD section 4, contract C2; ``FORWARD_REPAIR`` is asserted
    alongside it so the finding is a property of the DETECTOR rather than of
    one policy tier.
    """

    @BOTH_POLICIES
    async def test_the_call_is_admitted(self, policy) -> None:
        h = build_harness(policy)

        result = await h.call(
            'add_design_decision',
            {'decision': CROSS_PAIRED_DECISION, 'rationale': CROSS_PAIRED_RATIONALE},
        )

        assert result.data == 'decision_1', 'the call was not admitted'

    @BOTH_POLICIES
    async def test_both_arguments_reach_the_tool_verbatim(self, policy) -> None:
        """Not merely admitted — admitted UNMODIFIED.

        A middleware that admitted the call but rewrote an argument would be a
        different (and worse) finding, so the recorder is asserted rather than
        the result alone.
        """
        h = build_harness(policy)

        await h.call(
            'add_design_decision',
            {'decision': CROSS_PAIRED_DECISION, 'rationale': CROSS_PAIRED_RATIONALE},
        )

        assert h.recorder.args['decision'] == CROSS_PAIRED_DECISION
        assert h.recorder.args['rationale'] == CROSS_PAIRED_RATIONALE

    @BOTH_POLICIES
    async def test_no_fact_and_no_escalation_are_emitted(self, policy) -> None:
        """Nothing downstream ever learns this call happened.

        The fact stream is the only channel by which a cross-paired write could
        become visible to an operator. It stays empty, so the damage lands
        silently and is discoverable only by an author later noticing it.
        """
        h = build_harness(policy)

        await h.call(
            'add_design_decision',
            {'decision': CROSS_PAIRED_DECISION, 'rationale': CROSS_PAIRED_RATIONALE},
        )

        assert h.facts == [], 'a cross-paired call must not be expected to emit a fact'
        assert h.escalations == []

    @BOTH_POLICIES
    async def test_the_new_predicate_would_have_seen_it(self, policy) -> None:
        """The discrimination baseline: the specimen is not simply innocuous.

        Without this, every assertion above would pass just as well against a
        correctly-paired call, and the file would measure nothing. The
        middleware admits it; a detector that keys on the correction entry an
        author later appends is the only thing that ever sees this shape — and
        only AFTER a human noticed.
        """
        h = build_harness(policy)
        await h.call(
            'add_design_decision',
            {'decision': CROSS_PAIRED_DECISION, 'rationale': CROSS_PAIRED_RATIONALE},
        )
        assert h.facts == []

        # The cross-pairing is real damage, but at WRITE time nothing can see
        # it: both arguments are clean prose, so even this module's predicate
        # is silent until a correction entry exists.
        assert detect_mispairing(CROSS_PAIRED_DECISION, CROSS_PAIRED_RATIONALE) is None
        correction = (
            'CORRECTION of the preceding entry, whose rationale was mis-paired '
            'with its statement.'
        )
        assert detect_mispairing(correction, CROSS_PAIRED_RATIONALE) is not None

    @BOTH_POLICIES
    async def test_envelope_damage_on_the_same_tool_does_fire(self, policy) -> None:
        """THE CONTROL: the tripwire is live on this tool, it just cannot see THIS.

        Without this row, every assertion above would be equally satisfied by a
        middleware that simply was not guarding ``add_design_decision`` at all,
        and the whole file would measure nothing.

        Same tool, same harness, same policy — only the damage class differs.
        An envelope literal in ``rationale`` (the PRD's single largest real
        victim, 109 corrupted calls in section 2.3) DOES enter the fact stream
        and DOES stop the write.

        The observable asserted is the FACT, not the exception type. This
        specimen happens to be classified ``unrepairable``, which raises under
        BOTH tiers rather than only under ``REJECT_WITH_REPAIR`` — a detail of
        this one string, not of the policy, so pinning an exception type here
        would pin the wrong thing.
        """
        h = build_harness(policy)
        corrupted = (
            CROSS_PAIRED_RATIONALE
            + '\x3c/rationale>\n\x3cparameter name="step_type">impl\x3c/parameter>\n'
        )
        assert detect(corrupted) is not None, 'guard: the control specimen must be dirty'

        with contextlib.suppress(Exception):
            await h.call(
                'add_design_decision',
                {'decision': CROSS_PAIRED_DECISION, 'rationale': corrupted},
            )

        assert h.facts, 'the tripwire did not fire on envelope damage on this tool'
        assert h.recorder.calls == [], 'the corrupted write reached the tool anyway'


class TestWireEvidence:
    """THE MECHANISM FINDING: the harness hypothesis is REFUTED on this specimen.

    The originating escalation hypothesised that the cross-pairing was produced
    by the harness's tool-call argument parser over-consuming at an argument
    boundary. For task **3727** that is refuted, and this is what refutes it.

    All 7 of 3727's on-disk ``(decision, rationale)`` pairs are BYTE-IDENTICAL
    to an ``add_design_decision`` ``tool_use`` input recovered from its archived
    transcripts — INCLUDING the mis-paired entry ``[0]`` that entry ``[1]``
    corrects, which is the discriminating case. plan-tools therefore wrote
    exactly what it received: **the cross-pairing was already present in the
    arguments the model composed.** An argument-boundary over-consumption defect
    cannot produce a well-formed swap spanning two separate ``tool_use`` blocks
    — by its documented mechanism it truncates one argument and drops that
    call's siblings, leaving residue, and there is no residue here.

    ## SCOPE LIMIT — read this before citing the finding

    This is ONE clean specimen. Tasks **3567** and **4096** do NOT reproduce
    byte-identity on every field (consistent with task 3692's read-time envelope
    repair having rewritten them) and are INCONCLUSIVE; no wire record for
    either is committed, and neither may be cited as evidence in either
    direction. Nothing here licenses a general claim about every victim — only
    that for this specimen the fix does not belong at the harness boundary.

    The fixture is committed because ``data/`` is gitignored and
    retention-bounded, so the archive this was measured from will not outlive
    the finding.
    """

    def test_the_fixture_is_present_and_well_formed(self) -> None:
        doc = load_wire_evidence()
        assert doc['task_id'] == '3727'
        assert len(doc['on_disk']) == 7
        assert len(doc['wire']) == 7

    def test_all_seven_on_disk_pairs_are_byte_identical_to_a_wire_input(self) -> None:
        doc = load_wire_evidence()
        wire_pairs = {(w['input']['decision'], w['input']['rationale']) for w in doc['wire']}
        unmatched = [
            entry['index'] for entry in doc['on_disk']
            if (entry['decision'], entry['rationale']) not in wire_pairs
        ]
        assert not unmatched, (
            f'on-disk entries with no byte-identical wire input: {unmatched}. '
            'The mechanism finding does not hold as recorded — correct the '
            'FIXTURE to what you measured and restate the finding. Do NOT relax '
            'this assertion: that would launder a refuted claim into a green test.'
        )
        assert doc['measured_byte_identical_pairs'] == len(doc['on_disk'])

    def test_the_mispaired_entry_itself_arrived_byte_intact(self) -> None:
        """The discriminating case, stated on its own.

        A harness defect would have to have corrupted THIS entry specifically.
        It did not: entry [0] — the one entry [1] corrects — is on the wire
        exactly as it is on disk.
        """
        doc = load_wire_evidence()
        mispaired = doc['on_disk'][doc['mispaired_index']]
        wire_pairs = {(w['input']['decision'], w['input']['rationale']) for w in doc['wire']}
        assert (mispaired['decision'], mispaired['rationale']) in wire_pairs

    def test_the_correction_entry_is_detected_and_is_envelope_clean(self) -> None:
        """The correction is the ONLY machine-visible trace, and it carries no sentinel.

        So the envelope detector could never have flagged this specimen, and
        the write-time tripwire could never have contained it — which is the
        containment finding above, reproduced on real archived data rather than
        on a hand-authored pair.
        """
        doc = load_wire_evidence()
        correction = doc['on_disk'][doc['correction_index']]

        assert detect_mispairing(correction['decision'], correction['rationale']) is not None
        assert detect(correction['decision']) is None

    def test_no_inconclusive_specimen_is_committed_as_evidence(self) -> None:
        """3567 and 4096 must NOT be cited here, in either direction.

        They do not reproduce byte-identity on every field. Committing them
        alongside 3727 would let a reader take the general claim the scope
        limit explicitly withholds.
        """
        doc = load_wire_evidence()
        assert doc['task_id'] == '3727'
        assert '3567' in doc['note'] and '4096' in doc['note'], (
            'the fixture must NAME the inconclusive specimens, so their absence '
            'reads as a deliberate scope limit rather than an oversight'
        )
