"""Tests for escalation.pins — THE shared severity-aware pin classifier (task 3533).

PRD ``plans/task-escalation-state-graph-prd.md`` D3, spec
``docs/task-escalation-state-spec.md`` S6/E7. The classifier is the single
pin predicate (INV-5) every recovery/redispatch veto site consumes; this task
delivers the types + classifier + tests only — no veto site is rewired here.

Covers:
  step-3: pure type-surface contract — the StrEnum, the frozen report value
          object, and the STATICALLY-CHECKED structural Protocol seam, with no
          classification logic asserted yet.
  step-5: the SEVERITY rules (spec S6 clauses i and ii) — info never pins;
          missing/out-of-vocabulary severity fails safe to pinning, which
          deliberately OUTRANKS the dead-L0 rule.
  step-7: the LEVEL rule (L1/L2 always pin) and the DEAD-L0 FILING-INCARNATION
          rule (spec S6 clauses iii and iv) — liveness is judged against the
          incarnation that FILED the record.
  step-9: the STORE-UNAVAILABLE third state (never collapsed into "no
          records"), report aggregation, the two derived predicates, and
          purity.
  amend:  the two remaining fail-safes — a BORN_AT_L2 severity found at L0
          (contradictory state), and a live-claimant identity that is not in
          compose_claimant_run_id shape — plus classification of a REAL
          ``escalation.models.Escalation`` (the server-side record type).
"""

from __future__ import annotations

import dataclasses
import typing

import pytest

from escalation.models import BORN_AT_L2_SEVERITIES, KNOWN_SEVERITIES, Escalation
from escalation.pins import PinClass, PinRecord, PinReport, classify_pins

# ---------------------------------------------------------------------------
# step-3 — pure type-surface contract
# ---------------------------------------------------------------------------


class TestPinClass:
    """PinClass is a genuine-``str`` StrEnum with exactly three lowercase members."""

    def test_members_have_exact_lowercase_values(self) -> None:
        # Genuine str members — equality against a plain string holds without `.value`.
        assert PinClass.DEAD_L0 == 'dead_l0'
        assert PinClass.QUEUE_HANDOFF == 'queue_handoff'
        assert PinClass.NON_PINNING == 'non_pinning'

    def test_has_exactly_three_members(self) -> None:
        assert {m.value for m in PinClass} == {'dead_l0', 'queue_handoff', 'non_pinning'}


class TestPinReportTypeSurface:
    """PinReport is a frozen value object with the PRD's three positional buckets."""

    def test_is_frozen(self) -> None:
        report = PinReport((), (), ())
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.dead_l0 = ('esc-1-1',)  # type: ignore[misc]

    def test_constructs_positionally_with_prd_shape(self) -> None:
        """The PRD's literal ``PinReport(dead_l0, queue_handoff, non_pinning)`` shape."""
        report = PinReport((), (), ())
        assert report.dead_l0 == ()
        assert report.queue_handoff == ()
        assert report.non_pinning == ()

    def test_store_unavailable_and_task_id_default(self) -> None:
        report = PinReport((), (), ())
        assert report.store_unavailable is False
        assert report.task_id == ''

    def test_empty_available_report_does_not_pin_or_veto(self) -> None:
        report = PinReport((), (), ())
        assert report.pins is False
        assert report.vetoes_done_flip is False

    def test_derived_predicates_are_genuine_bools(self) -> None:
        """``pins`` / ``vetoes_done_flip`` return real bools, not truthy tuples."""
        report = PinReport((), (), ())
        assert isinstance(report.pins, bool)
        assert isinstance(report.vetoes_done_flip, bool)


class TestPinRecordProtocol:
    """PinRecord is a structural Protocol both record types satisfy.

    Conformance is pinned by a STATIC, pyright-checked assignment rather than
    runtime introspection: there is no runtime predicate for structural-Protocol
    conformance that is both public and meaningful (``isinstance`` needs
    ``@runtime_checkable``, which only re-checks attribute NAMES). The
    assignment below also checks attribute TYPES and read-only compatibility.
    """

    def test_escalation_satisfies_pin_record_statically(self) -> None:
        """The seam is checked by pyright too, not only at runtime — this
        assignment fails type-check if PinRecord and Escalation ever drift."""
        esc = Escalation(
            id='esc-42-1', task_id='42', agent_role='implementer',
            severity='blocking', category='infra_issue', summary='s',
        )
        record: PinRecord = esc
        assert record.id == 'esc-42-1'


class TestPinsModuleExports:
    def test_all_exports_the_public_surface(self) -> None:
        """Containment, not equality — a future export must not fail this for
        a purely cosmetic reason. The module-level imports above already pin
        that each name exists and is importable."""
        import escalation.pins as pins_mod

        assert {'PinClass', 'PinRecord', 'PinReport', 'classify_pins'} <= set(pins_mod.__all__)


# ---------------------------------------------------------------------------
# shared test doubles
# ---------------------------------------------------------------------------

#: A live incarnation's identity, in compose_claimant_run_id format.
LIVE_ID = 'run-B/sess-B/pid=2'
#: A DIFFERENT (prior) incarnation's identity.
OTHER_ID = 'run-A/sess-A/pid=1'


@dataclasses.dataclass(frozen=True)
class _Rec:
    """A minimal PinRecord double — keeps these tests independent of
    ``Escalation``'s (much wider) constructor."""

    id: str
    level: int
    severity: str
    filing_claimant_run_id: str | None


def _rec(
    *,
    id: str = 'esc-42-1',  # noqa: A002 — mirrors the PinRecord attribute name
    level: int = 0,
    severity: str | None = 'blocking',
    filing: str | None = None,
) -> _Rec:
    return _Rec(
        id=id,
        level=level,
        severity=typing.cast('str', severity),
        filing_claimant_run_id=filing,
    )


def _bucket_of(
    record: _Rec,
    *,
    live_claimant: bool = True,
    live_claimant_id: str | None = None,
) -> str:
    """Classify a single *record* and return the name of the bucket it lands in."""
    report = classify_pins(
        '42', [record], live_claimant=live_claimant, live_claimant_id=live_claimant_id,
    )
    landed = [
        name
        for name in ('dead_l0', 'queue_handoff', 'non_pinning')
        if record.id in getattr(report, name)
    ]
    assert len(landed) == 1, f'record must land in exactly one bucket, got {landed}'
    return landed[0]


# ---------------------------------------------------------------------------
# step-5 — severity rules (spec S6 clauses i and ii): precedence links 1 and 2
# ---------------------------------------------------------------------------


class TestInfoNeverPins:
    """Clause (i): an ``info`` record never pins, at any level, unconditionally."""

    @pytest.mark.parametrize('level', [0, 1, 2])
    @pytest.mark.parametrize('live_claimant', [True, False])
    @pytest.mark.parametrize('filing', [OTHER_ID, LIVE_ID, None])
    def test_info_is_non_pinning_at_every_level_and_liveness(
        self, level: int, live_claimant: bool, filing: str | None,
    ) -> None:
        bucket = _bucket_of(
            _rec(level=level, severity='info', filing=filing),
            live_claimant=live_claimant,
            live_claimant_id=LIVE_ID,
        )
        assert bucket == 'non_pinning'

    @pytest.mark.parametrize('raw', ['INFO', ' info ', 'Info', 'iNfO\n'])
    def test_severity_is_normalised_before_matching(self, raw: str) -> None:
        """Severity is stripped and lowercased before the vocabulary match."""
        assert _bucket_of(_rec(severity=raw)) == 'non_pinning'


class TestUnknownSeverityFailsSafeToPinning:
    """Clause (ii): missing / out-of-vocabulary severity fails safe to PINNING —
    "treated as a handoff, never to conversion" (spec S6)."""

    @pytest.mark.parametrize('severity', ['', None, '   ', 'bogus', 'BLOCKER'])
    @pytest.mark.parametrize('level', [0, 1, 2])
    def test_unknown_severity_is_queue_handoff_at_every_level(
        self, severity: str | None, level: int,
    ) -> None:
        assert severity is None or severity.strip().lower() not in KNOWN_SEVERITIES
        assert _bucket_of(_rec(level=level, severity=severity)) == 'queue_handoff'

    @pytest.mark.parametrize('severity', ['', None, '   ', 'bogus'])
    def test_unknown_severity_outranks_the_dead_l0_rule(self, severity: str | None) -> None:
        """THE precedence pin: an L0 with a provably-dead filer and an unknown
        severity STILL pins — spec S6's "never to conversion" clause is only
        meaningful if the fail-safe is evaluated ABOVE the L0 branch that
        produces the convertible DEAD_L0 class. Do not "fix" this ordering."""
        bucket = _bucket_of(
            _rec(level=0, severity=severity, filing=OTHER_ID),
            live_claimant=False,
        )
        assert bucket == 'queue_handoff', 'unknown severity must never convert to dead_l0'


class TestKnownNonInfoSeveritiesArePinCandidates:
    """Every known non-info severity is a pin candidate — never ``non_pinning``."""

    @pytest.mark.parametrize('severity', sorted(KNOWN_SEVERITIES - {'info'}))
    @pytest.mark.parametrize('level', [0, 1, 2])
    def test_known_non_info_severity_is_not_non_pinning(
        self, severity: str, level: int,
    ) -> None:
        bucket = _bucket_of(
            _rec(level=level, severity=severity), live_claimant=True, live_claimant_id=LIVE_ID,
        )
        assert bucket != 'non_pinning'

    def test_vocabulary_oracle_is_shared_with_models(self) -> None:
        """The classifier's "known severity" set is models.KNOWN_SEVERITIES —
        re-hardcoding it here would let a newly-added severity silently start
        failing the classifier's fail-safe branch."""
        assert 'info' in KNOWN_SEVERITIES
        assert {'info', 'blocking', 'critical', 'urgent'} <= KNOWN_SEVERITIES


# ---------------------------------------------------------------------------
# step-7 — level rule + dead-L0 filing-incarnation rule (spec S6 clauses
# iii and iv): precedence links 3 and 4
# ---------------------------------------------------------------------------

#: Severities for which the level/liveness outcome must be identical.
NON_INFO = sorted(KNOWN_SEVERITIES - {'info'})
#: The non-info severities an L0 can LEGALLY carry. critical/urgent are BORN at
#: L2 (models.BORN_AT_L2_SEVERITIES) and routed straight to a human, so an L0
#: carrying one is contradictory state that fails safe to pinning instead of
#: reaching the dead-L0 rule — see TestBornAtL2SeverityAtLevelZeroFailsSafe.
L0_NON_INFO = sorted(KNOWN_SEVERITIES - {'info'} - BORN_AT_L2_SEVERITIES)


class TestLevelRule:
    """Clause (iii): L1/L2 are queue-backed handoffs with supervised consumers —
    they pin regardless of liveness, so the filing-incarnation rule is L0-ONLY."""

    @pytest.mark.parametrize('severity', NON_INFO)
    @pytest.mark.parametrize('level', [1, 2])
    @pytest.mark.parametrize('live_claimant', [True, False])
    @pytest.mark.parametrize('filing', [LIVE_ID, OTHER_ID])
    def test_l1_and_l2_are_queue_handoff_under_every_liveness_combination(
        self, severity: str, level: int, live_claimant: bool, filing: str,
    ) -> None:
        bucket = _bucket_of(
            _rec(level=level, severity=severity, filing=filing),
            live_claimant=live_claimant,
            live_claimant_id=LIVE_ID,
        )
        assert bucket == 'queue_handoff', 'an L1/L2 is never dead_l0'

    @pytest.mark.parametrize('level', [3, 99, -1])
    def test_out_of_vocabulary_levels_fail_safe_to_queue_handoff(self, level: int) -> None:
        """A corrupt/out-of-range level pins. This is the exact shape that would
        wrongly become dead_l0 if the guard were written ``level >= 1`` instead
        of ``level != 0``."""
        bucket = _bucket_of(
            _rec(level=level, severity='blocking', filing=OTHER_ID),
            live_claimant=False,
        )
        assert bucket == 'queue_handoff'


class TestDeadL0FilingIncarnationRule:
    """Clause (iv): an L0 whose FILING incarnation is dead is DEAD_L0.

    The rule itself is documented on ``escalation.pins.classify_pins`` (spec
    ``docs/task-escalation-state-spec.md`` S6); these tests pin its outcomes.
    Parametrised over L0_NON_INFO, not NON_INFO: critical/urgent never reach
    this link (they are intercepted at L0 by link 3b)."""

    @pytest.mark.parametrize('severity', L0_NON_INFO)
    @pytest.mark.parametrize('filing', [OTHER_ID, None])
    def test_no_live_claimant_at_all_is_dead_l0(self, severity: str, filing: str | None) -> None:
        """With NO incarnation live, the filing one is necessarily dead —
        identity-independent. This is the 2026-08-02 strand shape: 7 of 9
        stranded tasks were pinned by their own dead-steward L0."""
        bucket = _bucket_of(
            _rec(level=0, severity=severity, filing=filing),
            live_claimant=False,
            live_claimant_id=None,
        )
        assert bucket == 'dead_l0'

    @pytest.mark.parametrize('severity', L0_NON_INFO)
    def test_newer_live_incarnation_does_not_keep_a_prior_incarnations_l0_alive(
        self, severity: str,
    ) -> None:
        """THE load-bearing case (spec S6): a claimant IS live, but it is not
        the one that filed this record — so the record's handoff has no
        consumer left and must NOT keep pinning the task."""
        bucket = _bucket_of(
            _rec(level=0, severity=severity, filing=OTHER_ID),
            live_claimant=True,
            live_claimant_id=LIVE_ID,
        )
        assert bucket == 'dead_l0'

    @pytest.mark.parametrize('severity', L0_NON_INFO)
    def test_live_filing_incarnation_is_a_genuine_handoff(self, severity: str) -> None:
        """An L0 whose filer IS the live incarnation is a real, live handoff —
        it pins (queue_handoff is the only pinning bucket)."""
        bucket = _bucket_of(
            _rec(level=0, severity=severity, filing=LIVE_ID),
            live_claimant=True,
            live_claimant_id=LIVE_ID,
        )
        assert bucket == 'queue_handoff'

    @pytest.mark.parametrize(
        ('filing', 'live_id'),
        [
            (None, LIVE_ID),      # record's filing identity unknown
            ('', LIVE_ID),        # ... blank
            ('   ', LIVE_ID),     # ... whitespace-only
            (OTHER_ID, None),     # caller could not resolve the live identity
            (OTHER_ID, ''),       # ... blank
            (OTHER_ID, '   '),    # ... whitespace-only
            (None, None),         # neither side known
        ],
    )
    def test_unknown_identity_on_either_side_fails_safe_to_pinning(
        self, filing: str | None, live_id: str | None,
    ) -> None:
        """The classifier may only convert an L0 when it can PROVE the filing
        incarnation is dead. It cannot here, so it pins.

        This is the branch that governs TODAY: until a producer stamps
        filing_claimant_run_id, every real record carries None — so the
        widening cannot by itself change any disposition."""
        bucket = _bucket_of(
            _rec(level=0, severity='blocking', filing=filing),
            live_claimant=True,
            live_claimant_id=live_id,
        )
        assert bucket == 'queue_handoff', 'must never convert on unprovable liveness'

    def test_identities_are_compared_as_exact_strings(self) -> None:
        """Identities differing only in the ``pid=`` suffix are DIFFERENT
        incarnations — shared.task_claimant ships a composer and deliberately
        no parser, so the classifier never decomposes the triple."""
        bucket = _bucket_of(
            _rec(level=0, severity='blocking', filing='run-A/sess-A/pid=1'),
            live_claimant=True,
            live_claimant_id='run-A/sess-A/pid=2',
        )
        assert bucket == 'dead_l0'


class TestBornAtL2SeverityAtLevelZeroFailsSafe:
    """A ``BORN_AT_L2_SEVERITIES`` record found at L0 is CONTRADICTORY state.

    critical/urgent escalations are created directly at L2 and routed straight
    to a human (models.py's consumer-per-level contract), which is why every
    in-repo producer stamps ``level=2`` alongside them. An L0 carrying one is
    therefore corrupt, and — in the same spirit as link 3's ``!= 0`` — it fails
    safe to PINNING rather than becoming a convertible ``dead_l0`` that
    silently stops pinning a human-routed record.
    """

    @pytest.mark.parametrize('severity', sorted(BORN_AT_L2_SEVERITIES))
    @pytest.mark.parametrize(
        ('live_claimant', 'live_id', 'filing'),
        [
            (False, None, OTHER_ID),   # no incarnation live at all
            (False, None, None),       # ... and no filing identity either
            (True, LIVE_ID, OTHER_ID),  # a DIFFERENT incarnation is live
        ],
    )
    def test_born_at_l2_severity_at_l0_is_queue_handoff(
        self, severity: str, live_claimant: bool, live_id: str | None, filing: str | None,
    ) -> None:
        bucket = _bucket_of(
            _rec(level=0, severity=severity, filing=filing),
            live_claimant=live_claimant,
            live_claimant_id=live_id,
        )
        assert bucket == 'queue_handoff', 'a human-routed record must never convert'

    @pytest.mark.parametrize('severity', sorted(BORN_AT_L2_SEVERITIES))
    @pytest.mark.parametrize('level', [1, 2])
    def test_born_at_l2_severity_at_its_own_levels_still_pins(
        self, severity: str, level: int,
    ) -> None:
        """The L0 interception changes nothing about the normal L1/L2 path."""
        assert _bucket_of(_rec(level=level, severity=severity), live_claimant=False) == (
            'queue_handoff'
        )


class TestLiveClaimantIdShapeGuard:
    """``live_claimant_id`` must be a full ``compose_claimant_run_id`` string.

    The only in-repo producer of a live identity
    (``TaskGroundTruth._resolve_live_claimant``) is heterogeneous: its DB
    source yields the composed identity, its plan.lock source yields a BARE
    ``session_id``, and its in-memory source yields ``None``. Comparing a bare
    session_id against a composed filing identity would always mismatch and so
    would convert a genuinely LIVE filer's L0 to ``dead_l0`` — the unsafe
    direction. A format mismatch is not PROOF that the filer is dead, so a
    non-composed identity on EITHER side is treated as unknown.
    """

    @pytest.mark.parametrize(
        'bare_live_id',
        ['sess-A', '3533-f3af2d2a', 'run-A/sess-A', 'pid=1'],
    )
    def test_non_composed_live_identity_is_treated_as_unknown(self, bare_live_id: str) -> None:
        bucket = _bucket_of(
            _rec(level=0, severity='blocking', filing=OTHER_ID),
            live_claimant=True,
            live_claimant_id=bare_live_id,
        )
        assert bucket == 'queue_handoff', 'a shape mismatch is not proof of a dead filer'

    @pytest.mark.parametrize('bare_filing', ['sess-A', 'run-A/sess-A'])
    def test_non_composed_filing_identity_is_treated_as_unknown(self, bare_filing: str) -> None:
        bucket = _bucket_of(
            _rec(level=0, severity='blocking', filing=bare_filing),
            live_claimant=True,
            live_claimant_id=LIVE_ID,
        )
        assert bucket == 'queue_handoff'

    def test_the_guard_is_a_shape_check_not_a_parse(self) -> None:
        """Two composed identities are still compared WHOLE — the guard only
        gates on the ``/pid=`` marker, it never decomposes the triple."""
        assert (
            _bucket_of(
                _rec(level=0, severity='blocking', filing='run-A/sess-A/pid=1'),
                live_claimant=True,
                live_claimant_id='run-A/sess-A/pid=2',
            )
            == 'dead_l0'
        )

    def test_the_no_live_claimant_branch_is_identity_independent(self) -> None:
        """``live_claimant=False`` decides before any identity is inspected, so
        a bare live id cannot suppress the 2026-08-02 strand conversion."""
        bucket = _bucket_of(
            _rec(level=0, severity='blocking', filing=OTHER_ID),
            live_claimant=False,
            live_claimant_id='sess-A',
        )
        assert bucket == 'dead_l0'


# ---------------------------------------------------------------------------
# step-9 — store-correctness third state, aggregation, derived predicates
# ---------------------------------------------------------------------------


class TestStoreUnavailableThirdState:
    """Spec S6 store-correctness contract (esc-3163 lesson): "store unavailable"
    is a DISTINGUISHABLE third result, never collapsed into "no records" — a
    false ``[]`` would route a genuinely-pinned strand into the plain revert
    branch."""

    def test_records_none_is_store_unavailable_and_pins(self) -> None:
        report = classify_pins('42', None, live_claimant=False)

        assert report.store_unavailable is True
        assert report.dead_l0 == ()
        assert report.queue_handoff == ()
        assert report.non_pinning == ()
        assert report.pins is True
        assert report.vetoes_done_flip is True

    def test_empty_records_is_store_available_and_does_not_pin(self) -> None:
        report = classify_pins('42', [], live_claimant=False)

        assert report.store_unavailable is False
        assert report.dead_l0 == ()
        assert report.queue_handoff == ()
        assert report.non_pinning == ()
        assert report.pins is False
        assert report.vetoes_done_flip is False

    def test_store_unavailable_is_not_equal_to_no_records(self) -> None:
        """The regression pin: a false ``[]`` cannot masquerade as a store
        failure (nor the reverse) — the collapse is structurally impossible,
        not merely discouraged."""
        unavailable = classify_pins('42', None, live_claimant=False)
        empty = classify_pins('42', [], live_claimant=False)
        assert unavailable != empty

    @pytest.mark.parametrize('live_claimant', [True, False])
    @pytest.mark.parametrize('live_id', [None, LIVE_ID, ''])
    def test_none_short_circuits_regardless_of_liveness_arguments(
        self, live_claimant: bool, live_id: str | None,
    ) -> None:
        report = classify_pins(
            '42', None, live_claimant=live_claimant, live_claimant_id=live_id,
        )
        assert report.store_unavailable is True

    def test_store_unavailable_report_echoes_task_id(self) -> None:
        assert classify_pins('77', None, live_claimant=False).task_id == '77'


class TestReportAggregation:
    """A mixed batch splits into the three buckets, preserving input order."""

    MIXED = [
        _rec(id='esc-42-1', level=0, severity='info'),
        _rec(id='esc-42-2', level=0, severity='blocking'),
        _rec(id='esc-42-3', level=1, severity='blocking'),
        _rec(id='esc-42-4', level=0, severity='bogus'),
    ]

    def _report(self) -> PinReport:
        return classify_pins('42', self.MIXED, live_claimant=False)

    def test_mixed_batch_splits_into_the_three_buckets(self) -> None:
        report = self._report()
        assert report.non_pinning == ('esc-42-1',)
        assert report.dead_l0 == ('esc-42-2',)
        assert report.queue_handoff == ('esc-42-3', 'esc-42-4')

    def test_queue_handoff_preserves_input_order(self) -> None:
        reordered = [self.MIXED[3], self.MIXED[2]]
        report = classify_pins('42', reordered, live_claimant=False)
        assert report.queue_handoff == ('esc-42-4', 'esc-42-3')

    def test_buckets_are_tuples_of_ids(self) -> None:
        report = self._report()
        for bucket in (report.dead_l0, report.queue_handoff, report.non_pinning):
            assert isinstance(bucket, tuple)
            assert all(isinstance(item, str) for item in bucket)

    def test_every_record_lands_in_exactly_one_bucket(self) -> None:
        """No drops, no duplicates."""
        report = self._report()
        landed = [*report.dead_l0, *report.queue_handoff, *report.non_pinning]
        assert len(landed) == len(self.MIXED)
        assert sorted(landed) == sorted(rec.id for rec in self.MIXED)

    def test_report_echoes_task_id_for_structured_emission(self) -> None:
        assert self._report().task_id == '42'

    def test_records_are_not_filtered_on_their_own_task_id(self) -> None:
        """The CALLER owns the read scope; the classifier classifies what it is
        given (an id-prefix filter here would silently drop rows)."""
        foreign = _rec(id='esc-999-1', level=1, severity='blocking')
        report = classify_pins('42', [foreign], live_claimant=False)
        assert report.queue_handoff == ('esc-999-1',)


class TestDerivedPredicates:
    """``pins`` is the recovery veto; ``vetoes_done_flip`` is the deliberately
    more conservative MARK_DONE veto (PRD D3)."""

    def _classify(self, records: list[_Rec]) -> PinReport:
        return classify_pins('42', records, live_claimant=False)

    def test_dead_l0_only_batch_does_not_pin(self) -> None:
        """PRD boundary row 7: a dead-L0 does not block conversion."""
        report = self._classify([_rec(id='esc-42-1', level=0, severity='blocking')])
        assert report.dead_l0 == ('esc-42-1',)
        assert report.pins is False

    def test_info_only_batch_does_not_pin(self) -> None:
        """PRD boundary row 8."""
        report = self._classify([_rec(id='esc-42-1', level=0, severity='info')])
        assert report.non_pinning == ('esc-42-1',)
        assert report.pins is False

    def test_queue_handoff_batch_pins(self) -> None:
        report = self._classify([_rec(id='esc-42-1', level=1, severity='blocking')])
        assert report.pins is True

    def test_dead_l0_still_vetoes_the_done_flip(self) -> None:
        """PRD D3: ANY non-info open record still vetoes MARK_DONE —
        phantom-done protection is the half of the veto that was always right."""
        report = self._classify([_rec(id='esc-42-1', level=0, severity='blocking')])
        assert report.vetoes_done_flip is True

    def test_info_only_batch_does_not_veto_the_done_flip(self) -> None:
        report = self._classify([_rec(id='esc-42-1', level=0, severity='info')])
        assert report.vetoes_done_flip is False

    def test_queue_handoff_vetoes_the_done_flip(self) -> None:
        report = self._classify([_rec(id='esc-42-1', level=2, severity='blocking')])
        assert report.vetoes_done_flip is True


class TestPurity:
    """The classifier mutates nothing and performs no I/O."""

    def test_input_sequence_and_records_are_not_mutated(self) -> None:
        records = [
            _rec(id='esc-42-1', level=0, severity='blocking', filing=OTHER_ID),
            _rec(id='esc-42-2', level=1, severity='info'),
        ]
        before = [dataclasses.replace(rec) for rec in records]

        classify_pins('42', records, live_claimant=True, live_claimant_id=LIVE_ID)

        assert records == before
        assert len(records) == 2

    def test_repeated_calls_are_deterministic(self) -> None:
        records = [_rec(id='esc-42-1', level=0, severity='blocking', filing=OTHER_ID)]
        first = classify_pins('42', records, live_claimant=False)
        second = classify_pins('42', records, live_claimant=False)
        assert first == second


# ---------------------------------------------------------------------------
# amend — the REAL server-side record type end to end
# ---------------------------------------------------------------------------


def _esc(
    *,
    id: str = 'esc-42-1',  # noqa: A002 — mirrors the PinRecord attribute name
    level: int = 0,
    severity: str = 'blocking',
    filing_claimant_run_id: str | None = None,
) -> Escalation:
    """A real ``escalation.models.Escalation`` — the server-side PinRecord."""
    return Escalation(
        id=id,
        task_id='42',
        agent_role='implementer',
        severity=severity,
        category='infra_issue',
        summary='s',
        level=level,
        filing_claimant_run_id=filing_claimant_run_id,
    )


class TestRealEscalationRecordsClassify:
    """``escalation.models.Escalation`` — not just the ``_Rec`` double — flows
    through the classifier, so the server-side consumer path is covered too."""

    def test_dead_filer_l0_escalation_is_dead_l0(self) -> None:
        report = classify_pins(
            '42', [_esc(level=0, filing_claimant_run_id=OTHER_ID)], live_claimant=False,
        )
        assert report.dead_l0 == ('esc-42-1',)
        assert report.pins is False
        assert report.vetoes_done_flip is True

    def test_l1_escalation_is_a_queue_handoff(self) -> None:
        report = classify_pins('42', [_esc(level=1)], live_claimant=False)
        assert report.queue_handoff == ('esc-42-1',)
        assert report.pins is True

    def test_info_escalation_is_non_pinning(self) -> None:
        report = classify_pins('42', [_esc(severity='info')], live_claimant=False)
        assert report.non_pinning == ('esc-42-1',)
        assert report.pins is False

    def test_escalation_rehydrated_with_null_severity_fails_safe_to_pinning(self) -> None:
        """The `or ''` in the classifier's severity normalisation is
        load-bearing: `Escalation.severity` is annotated `str`, but a record
        rehydrated from a payload carrying `"severity": null` really does hold
        None, and must fail safe to pinning rather than raising."""
        payload = _esc(level=0, filing_claimant_run_id=OTHER_ID).to_dict()
        payload['severity'] = None
        record = Escalation.from_dict(payload)
        assert record.severity is None

        report = classify_pins('42', [record], live_claimant=False)

        assert report.queue_handoff == ('esc-42-1',)
        assert report.dead_l0 == ()

    def test_legacy_escalation_without_filing_identity_classifies(self) -> None:
        """A legacy on-disk record has no ``filing_claimant_run_id`` key at
        all; it rehydrates to None and hits the unknown-identity fail-safe."""
        payload = _esc(level=0).to_dict()
        del payload['filing_claimant_run_id']
        record = Escalation.from_dict(payload)
        assert record.filing_claimant_run_id is None

        report = classify_pins(
            '42', [record], live_claimant=True, live_claimant_id=LIVE_ID,
        )

        assert report.queue_handoff == ('esc-42-1',)
