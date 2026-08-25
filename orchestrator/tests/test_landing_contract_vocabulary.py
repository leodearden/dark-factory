"""The landing contract's closed, machine-checked vocabularies (task 4647).

The PRD "landed-not-done-recovery" Contract names a closed set of reason codes
and a closed set of attribution methods.  Before this task both lived as bare
string literals scattered across ``landing_evidence.py``, so nothing could tell
a typo from a new member and nothing forced a new member to be explicable.

Every assertion here iterates the ENUM rather than a hardcoded list.  That is
the G7 "contracts machine-checked" shape and it is the whole point of the file:
a list repeated in the test is just a second place to forget to update, and the
failure it would miss is user-visible — ``format_unattributed_landing_detail``
renders the literal ``'Unrecognized reason code: ...'`` into the L1 body a human
reads when a member has no explanation.
"""

from __future__ import annotations

import dataclasses
import enum

import pytest

from orchestrator import landing_evidence
from orchestrator.landing_evidence import (
    LandingEvidenceVerdict,
    LandingMethod,
    LandingReason,
    LandingVerdict,
)

#: The Contract's closed six, plus the three PRE-CONTRACT codes
#: ``validate_landing_evidence`` still emits until leaf epsilon repoints its
#: consumers.  ONE vocabulary, not two: a second enum for the legacy spellings
#: would be exactly the "two disagreeing authorities" shape G7 forbids, and
#: would let a legacy reason reach a formatter that cannot explain it.
EXPECTED_REASONS = {
    # The PRD Contract's closed six.
    'landed',
    'no_op_landing',
    'not_landed',
    'no_attribution',
    'degenerate_branch',
    'git_error',
    # Legacy, still emitted by validate_landing_evidence.
    'ok',
    'no_citation',
    'effect_absent',
}

#: The three production attribution paths, plus the hand-constructed default.
EXPECTED_METHODS = {'patch_id', 'merge_marker', 'citation', 'unspecified'}


class TestLandingReasonVocabulary:
    def test_is_a_str_enum(self) -> None:
        assert issubclass(LandingReason, enum.StrEnum)

    def test_members_are_exactly_the_contract_plus_the_legacy_codes(self) -> None:
        assert {m.name for m in LandingReason} == EXPECTED_REASONS

    def test_every_member_value_equals_its_name(self) -> None:
        """EXPLICIT string values, never ``enum.auto()``.

        ``auto()`` on a ``StrEnum`` happens to produce the lower-cased name, so
        this would pass either way — but the spelling is load-bearing beyond
        its value.  See :func:`test_reason_values_are_written_out_literally`.
        """
        for member in LandingReason:
            assert member.value == member.name, member

    def test_reason_values_are_written_out_literally_in_the_source(self) -> None:
        """The enum must be declared ``no_attribution = 'no_attribution'``.

        Not style.  This task's ``metadata.delivered_checks`` greps
        ``landing_evidence.py`` for the literal ``'no_attribution'`` WITH its
        single quotes, so an ``enum.auto()`` declaration would produce an
        identical runtime vocabulary and silently fail the mark-done gate.
        Pinned here so the gate's dependency is visible in the test suite
        rather than only in task metadata.
        """
        source = landing_evidence.__file__
        with open(source, encoding='utf-8') as handle:
            text = handle.read()
        assert 'enum.auto()' not in text, 'explicit string values, not auto()'
        for member in LandingReason:
            assert f"{member.name} = '{member.value}'" in text, member


class TestLandingMethodVocabulary:
    def test_is_a_str_enum(self) -> None:
        assert issubclass(LandingMethod, enum.StrEnum)

    def test_members_are_exactly_the_three_paths_plus_unspecified(self) -> None:
        assert {m.name for m in LandingMethod} == EXPECTED_METHODS

    def test_every_member_value_equals_its_name(self) -> None:
        for member in LandingMethod:
            assert member.value == member.name, member


class TestGenuineStrMembers:
    """Both vocabularies must compare and HASH as plain strings.

    ``_REASON_EXPLANATIONS.get(verdict.reason)`` is a dict lookup keyed by
    plain strings, and every incumbent assertion in the sibling suites compares
    ``verdict.reason`` to a bare literal.  A plain ``enum.Enum`` would break
    both silently — the lookup would miss and every equality would be False —
    so this is a behavioural pin, not a type-hygiene one.
    """

    @pytest.mark.parametrize(
        ('member', 'literal'),
        [
            (LandingReason.landed, 'landed'),
            (LandingReason.no_citation, 'no_citation'),
            (LandingReason.effect_absent, 'effect_absent'),
            (LandingMethod.patch_id, 'patch_id'),
        ],
    )
    def test_equality_holds_without_dot_value(self, member, literal: str) -> None:
        assert member == literal
        assert isinstance(member, str)

    def test_members_resolve_as_plain_string_dict_keys(self) -> None:
        assert {'landed': 1}[LandingReason.landed] == 1
        assert {LandingReason.landed: 1}['landed'] == 1
        assert {'patch_id': 2}[LandingMethod.patch_id] == 2


class TestLandingVerdictType:
    def test_is_a_frozen_dataclass_with_the_five_fields(self) -> None:
        assert dataclasses.is_dataclass(LandingVerdict)
        assert LandingVerdict.__dataclass_params__.frozen is True
        names = [f.name for f in dataclasses.fields(LandingVerdict)]
        assert names == ['accepted', 'evidence_sha', 'reason', 'probe', 'method']

    def test_the_legacy_name_is_an_alias_not_a_subclass(self) -> None:
        """ONE verdict type, not two.

        ``harness.py`` and ``merge_queue.py`` both import
        ``LandingEvidenceVerdict`` and use it as an annotation, and four
        sibling test files construct it by keyword.  A module-level alias keeps
        every one of them working with zero edits; a second dataclass — or a
        subclass — would be the lockstep-duplication shape G7 forbids, with two
        types that must be kept in step forever.
        """
        assert LandingEvidenceVerdict is LandingVerdict

    def test_four_kwarg_construction_still_works_and_defaults_the_method(self) -> None:
        """The backward-compat pin for the four sibling test files.

        test_harness_already_landed_gate_wiring.py (:316, :346, :1775) and
        test_merge_queue_coalesce.py (:1899, :1965, :2027, :2241) construct the
        verdict with exactly these four keywords.  ``method`` must therefore be
        LAST and defaulted, and a hand-constructed verdict must be readable AS
        hand-constructed — which is what ``unspecified`` means.
        """
        verdict = LandingVerdict(
            accepted=False, evidence_sha=None, reason='no_citation', probe={},
        )
        assert verdict.accepted is False
        assert verdict.evidence_sha is None
        assert verdict.reason == 'no_citation'
        assert verdict.probe == {}
        assert verdict.method is LandingMethod.unspecified

    def test_positional_construction_is_unchanged(self) -> None:
        verdict = LandingVerdict(True, 'a' * 40, LandingReason.landed, {'k': 1})
        assert verdict.method is LandingMethod.unspecified
        assert verdict.evidence_sha == 'a' * 40

    def test_is_frozen(self) -> None:
        verdict = LandingVerdict(
            accepted=False, evidence_sha=None, reason='no_citation', probe={},
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            verdict.accepted = True  # type: ignore[misc]


class TestEveryMemberIsExplicable:
    """The G7 contracts-machine-checked pin.

    ``format_unattributed_landing_detail`` falls back to the literal
    ``'Unrecognized reason code: {reason}'`` for anything it cannot explain.
    That string reaching an L1 body is a user-visible defect, so no member of
    the module's OWN vocabulary may ever produce it.  Iterating the enum — not
    a list repeated here — is what makes adding a member without an
    explanation impossible rather than merely discouraged.
    """

    def test_every_reason_has_an_explanation(self) -> None:
        missing = [m.name for m in LandingReason
                   if m not in landing_evidence._REASON_EXPLANATIONS]
        assert not missing, f'LandingReason members with no explanation: {missing}'

    def test_every_method_has_an_explanation(self) -> None:
        missing = [m.name for m in LandingMethod
                   if m not in landing_evidence._METHOD_EXPLANATIONS]
        assert not missing, f'LandingMethod members with no explanation: {missing}'

    def test_no_explanation_is_orphaned(self) -> None:
        """The other direction: an explanation for a code no member emits.

        A leftover key is how a vocabulary silently grows a tenth member that
        nothing validates — the same drift in the opposite direction.
        """
        reasons = {m.value for m in LandingReason}
        assert set(landing_evidence._REASON_EXPLANATIONS) <= reasons
        methods = {m.value for m in LandingMethod}
        assert set(landing_evidence._METHOD_EXPLANATIONS) <= methods

    def test_no_member_renders_the_unrecognized_fallback(self) -> None:
        """End to end, through the real renderer the operator actually reads."""
        for member in LandingReason:
            verdict = LandingVerdict(
                accepted=False, evidence_sha=None, reason=member, probe={},
            )
            summary, detail = landing_evidence.format_unattributed_landing_detail(
                '4647', 'task/4647', verdict,
            )
            assert 'Unrecognized reason code' not in detail, member
            assert member.value in summary, member
