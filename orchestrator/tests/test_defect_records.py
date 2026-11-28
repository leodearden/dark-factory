"""orchestrator.defect_records reads a task's stored defect records fail-open (task 7050, defect-tests δ2)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from _defect_ruling_fixtures import RULED_REF, SECOND_RULED_REF
from shared.defects import (
    DEFECT_TEST_RETIREMENTS_METADATA_KEY,
    DEFECTS_METADATA_KEY,
    Declare,
    DeclareClosure,
    DefectDeclaration,
    DefectStatus,
    Exemption,
    ExemptionReason,
    Retirement,
    RetirementCode,
    RetirementReason,
    Rule,
    RuledBy,
    Ruling,
    RulingSubject,
    apply_event,
    parse_ref,
)
from shared.governed_exceptions import Debt, TicketRef

from orchestrator.defect_records import (
    awaits_a_ruling,
    closures_awaiting_ruling,
    parse_record,
    readable_records,
    retirements_awaiting_ruling,
)

DECLARED_AT = datetime(2026, 11, 28, 12, 0, tzinfo=UTC)
EXEMPTION = Exemption(
    reason=ExemptionReason.NEEDS_EXTERNAL_SYSTEM, owner=Debt(TicketRef('tkt_0RW782Z6RY3X8W8Q3NX8J1EKD0')),
)
MALFORMED = {'ref': 'not a ref'}


def _open(ref: str) -> DefectDeclaration:
    return apply_event(None, Declare(declaration=DefectDeclaration(
        ref=ref, origin=parse_ref(ref).origin, summary='x is wrong', declared_at=DECLARED_AT,
        declared_by='reviewer_comprehensive', status=DefectStatus.OPEN,
    )))


def _pending_exemption(ref: str) -> DefectDeclaration:
    return apply_event(_open(ref), DeclareClosure(ref=ref, closure=EXEMPTION))


def _retirement(ref: str, *, ruled: bool = False) -> Retirement:
    ruling = Ruling(
        subject=RulingSubject.RETIREMENT, accepted=True, ruled_by=RuledBy.REVIEWER, verdict_cycle=1,
    )
    return Retirement(
        ref=ref, reason=RetirementReason(reason=RetirementCode.FEATURE_REMOVED),
        ruling=ruling if ruled else None,
    )


def _stored(*records: DefectDeclaration | Retirement | dict) -> list[object]:
    return [record if isinstance(record, dict) else record.model_dump(mode='json') for record in records]


class TestReadableRecords:
    def test_a_valid_entry_is_read_as_the_model(self):
        assert parse_record(DefectDeclaration, _stored(_open(RULED_REF))[0]) == _open(RULED_REF)

    def test_an_entry_that_does_not_validate_is_none(self):
        assert parse_record(DefectDeclaration, MALFORMED) is None

    def test_malformed_entries_are_skipped_and_the_rest_kept_in_order(self):
        raw = _stored(_open(RULED_REF), MALFORMED, _open(SECOND_RULED_REF))

        assert readable_records(DefectDeclaration, raw) == (_open(RULED_REF), _open(SECOND_RULED_REF))

    @pytest.mark.parametrize('raw', [None, {}, 'defects', 3])
    def test_a_value_that_is_not_a_list_holds_no_records(self, raw):
        assert readable_records(DefectDeclaration, raw) == ()


class TestAwaitingARuling:
    def test_a_pending_closure_awaits_its_ruling(self):
        metadata = {DEFECTS_METADATA_KEY: _stored(_open(SECOND_RULED_REF), _pending_exemption(RULED_REF))}

        assert closures_awaiting_ruling(metadata) == ((_pending_exemption(RULED_REF), EXEMPTION),)
        assert awaits_a_ruling(metadata)

    def test_a_ruled_closure_no_longer_awaits_one(self):
        ruled = apply_event(_pending_exemption(RULED_REF), Rule(ref=RULED_REF, ruling=Ruling(
            subject=RulingSubject.CLOSURE, accepted=True, ruled_by=RuledBy.REVIEWER, verdict_cycle=1,
        )))
        metadata = {DEFECTS_METADATA_KEY: _stored(ruled)}

        assert closures_awaiting_ruling(metadata) == ()
        assert not awaits_a_ruling(metadata)

    def test_an_unruled_retirement_awaits_its_ruling_and_a_ruled_one_does_not(self):
        metadata = {
            DEFECT_TEST_RETIREMENTS_METADATA_KEY: _stored(
                _retirement(RULED_REF, ruled=True), _retirement(SECOND_RULED_REF),
            ),
        }

        assert retirements_awaiting_ruling(metadata) == (_retirement(SECOND_RULED_REF),)
        assert awaits_a_ruling(metadata)

    @pytest.mark.parametrize('metadata', [
        {},
        {DEFECTS_METADATA_KEY: _stored(_open(RULED_REF), MALFORMED)},
        {DEFECT_TEST_RETIREMENTS_METADATA_KEY: _stored(_retirement(RULED_REF, ruled=True), MALFORMED)},
        {DEFECTS_METADATA_KEY: 'pending', DEFECT_TEST_RETIREMENTS_METADATA_KEY: {'ref': RULED_REF}},
    ])
    def test_nothing_awaits_a_ruling_without_a_readable_pending_record(self, metadata):
        assert not awaits_a_ruling(metadata)
