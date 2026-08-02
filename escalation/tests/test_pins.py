"""Tests for escalation.pins — THE shared severity-aware pin classifier (task 3533).

PRD ``plans/task-escalation-state-graph-prd.md`` D3, spec
``docs/task-escalation-state-spec.md`` S6/E7. The classifier is the single
pin predicate (INV-5) every recovery/redispatch veto site consumes; this task
delivers the types + classifier + tests only — no veto site is rewired here.

Covers:
  step-3: pure type-surface contract — the StrEnum, the structural Protocol,
          and the frozen report value object, no classification logic yet.
"""

from __future__ import annotations

import dataclasses
import enum
import typing

import pytest

from escalation.models import Escalation
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
