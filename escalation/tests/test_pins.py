"""Tests for escalation.pins — THE shared severity-aware pin classifier (task 3533).

PRD ``plans/task-escalation-state-graph-prd.md`` D3, spec
``docs/task-escalation-state-spec.md`` S6/E7. The classifier is the single
pin predicate (INV-5) every recovery/redispatch veto site consumes; this task
delivers the types + classifier + tests only — no veto site is rewired here.

Covers:
  step-3: pure type-surface contract — the StrEnum, the structural Protocol,
          and the frozen report value object, no classification logic yet.
  step-5: the SEVERITY rules (spec S6 clauses i and ii) — info never pins;
          missing/out-of-vocabulary severity fails safe to pinning, which
          deliberately OUTRANKS the dead-L0 rule.
"""

from __future__ import annotations

import dataclasses
import enum
import typing

import pytest

from escalation.models import KNOWN_SEVERITIES, Escalation
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

    def test_is_a_str_enum(self) -> None:
        assert issubclass(PinClass, enum.StrEnum)
        assert isinstance(PinClass.DEAD_L0, str)

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

    def test_derived_predicates_are_properties(self) -> None:
        assert isinstance(PinReport.pins, property)
        assert isinstance(PinReport.vetoes_done_flip, property)


class TestPinRecordProtocol:
    """PinRecord is a structural Protocol both record types satisfy."""

    def test_is_a_protocol(self) -> None:
        assert typing.get_origin(PinRecord) is None
        assert getattr(PinRecord, '_is_protocol', False) is True

    def test_escalation_structurally_satisfies_pin_record(self) -> None:
        """``escalation.models.Escalation`` carries every PinRecord attribute."""
        for name in ('id', 'level', 'severity', 'filing_claimant_run_id'):
            assert name in Escalation.__dataclass_fields__, (
                f'Escalation must carry {name!r} to satisfy PinRecord'
            )


class TestPinsModuleExports:
    def test_all_exports_exactly_the_public_surface(self) -> None:
        import escalation.pins as pins_mod

        assert pins_mod.__all__ == ['PinClass', 'PinRecord', 'PinReport', 'classify_pins']

    def test_classify_pins_is_callable(self) -> None:
        assert callable(classify_pins)


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
