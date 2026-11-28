"""The defect-record events a review round produces, and their delivery (task 7050, defect-tests δ2)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from _defect_ruling_fixtures import RULED_REF, SECOND_RULED_REF
from shared.defects import (
    Declare,
    DefectOrigin,
    DefectStatus,
    IssueKind,
    Rule,
    RuledBy,
    Ruling,
    RulingSubject,
    Sighting,
    mint_review_ref,
)

from orchestrator.review_defects import review_defect_declarations, review_ruling_events
from orchestrator.review_payload import DefectRuling

TASK_NUMBER = 42
REVIEWER = 'reviewer_comprehensive'
DECLARED_AT = datetime(2026, 11, 28, 12, 0, tzinfo=UTC)
LOCATION = 'lib.py:5'
DESCRIPTION = 'farewell crashes on an empty name'


def _issue(
    kind: IssueKind | None = IssueKind.DEFECT,
    *,
    location: str = LOCATION,
    description: str = DESCRIPTION,
) -> dict[str, str]:
    """A blocking issue the way artifacts.py::aggregate_reviews emits it."""
    issue = {
        'severity': 'blocking',
        'location': location,
        'category': 'correctness',
        'description': description,
        'suggested_fix': 'Guard the empty name',
        'reviewer': REVIEWER,
    }
    if kind is not None:
        issue['kind'] = kind.value
    return issue


def _declarations(*issues: dict[str, str], known_refs: frozenset[str] = frozenset()):
    return review_defect_declarations(
        TASK_NUMBER, issues, known_refs=known_refs, declared_at=DECLARED_AT,
    )


class TestReviewDefectDeclarations:
    def test_a_defect_issue_is_declared_as_an_open_review_defect(self):
        (event,) = _declarations(_issue())

        assert isinstance(event, Declare)
        declaration = event.declaration
        assert declaration.ref == mint_review_ref(TASK_NUMBER, LOCATION, DESCRIPTION)
        assert declaration.origin is DefectOrigin.REVIEW
        assert declaration.status is DefectStatus.OPEN
        assert declaration.declared_at == DECLARED_AT
        assert declaration.declared_by == REVIEWER
        assert DESCRIPTION in declaration.summary

    @pytest.mark.parametrize(
        'kind', [kind for kind in IssueKind if kind is not IssueKind.DEFECT], ids=str,
    )
    def test_an_issue_of_another_kind_declares_nothing(self, kind):
        assert _declarations(_issue(kind)) == ()

    def test_an_issue_without_a_kind_is_declared_fail_closed(self):
        (event,) = _declarations(_issue(kind=None))

        assert isinstance(event, Declare)
        assert event.ref == mint_review_ref(TASK_NUMBER, LOCATION, DESCRIPTION)

    def test_a_known_ref_is_sighted_not_declared_again(self):
        ref = mint_review_ref(TASK_NUMBER, LOCATION, DESCRIPTION)

        assert _declarations(_issue(), known_refs=frozenset({ref})) == (Sighting(ref=ref),)

    def test_a_repeat_within_one_round_is_a_sighting_of_the_first_declaration(self):
        first = _issue()
        repeat = _issue(
            location=f'  {LOCATION.upper()} ', description=DESCRIPTION.upper().replace(' ', '\t'),
        )

        declare, sighting = _declarations(first, repeat)

        assert isinstance(declare, Declare)
        assert sighting == Sighting(ref=declare.ref)

    def test_a_blank_description_falls_back_to_the_location_for_the_summary(self):
        (event,) = _declarations(_issue(description='   '))

        assert isinstance(event, Declare)
        assert LOCATION in event.declaration.summary

    def test_a_blank_description_and_location_fall_back_to_a_summary_naming_the_reviewer(self):
        (event,) = _declarations(_issue(location='', description=''))

        assert isinstance(event, Declare)
        assert REVIEWER in event.declaration.summary


VERDICT_CYCLE = 3


def _reviewer_rule(ref: str, subject: RulingSubject, *, accepted: bool) -> Rule:
    return Rule(ref=ref, ruling=Ruling(
        subject=subject, accepted=accepted, ruled_by=RuledBy.REVIEWER, verdict_cycle=VERDICT_CYCLE,
    ))


def _rule_events(*rulings: DefectRuling) -> tuple[Rule, ...]:
    return review_ruling_events(rulings, verdict_cycle=VERDICT_CYCLE)


class TestReviewRulingEvents:
    def test_an_accepted_closure_ruling_is_one_reviewer_rule(self):
        assert _rule_events(DefectRuling(RULED_REF, RulingSubject.CLOSURE, True)) == (
            _reviewer_rule(RULED_REF, RulingSubject.CLOSURE, accepted=True),
        )

    def test_a_retirement_ruling_keeps_its_subject(self):
        assert _rule_events(DefectRuling(RULED_REF, RulingSubject.RETIREMENT, True)) == (
            _reviewer_rule(RULED_REF, RulingSubject.RETIREMENT, accepted=True),
        )

    def test_a_split_panel_rejects(self):
        events = _rule_events(
            DefectRuling(RULED_REF, RulingSubject.CLOSURE, True),
            DefectRuling(RULED_REF, RulingSubject.CLOSURE, False),
        )

        assert events == (_reviewer_rule(RULED_REF, RulingSubject.CLOSURE, accepted=False),)

    def test_a_unanimous_panel_accepts_once(self):
        events = _rule_events(
            DefectRuling(RULED_REF, RulingSubject.CLOSURE, True),
            DefectRuling(RULED_REF, RulingSubject.CLOSURE, True),
        )

        assert events == (_reviewer_rule(RULED_REF, RulingSubject.CLOSURE, accepted=True),)

    def test_one_ref_ruled_on_both_subjects_is_two_rules(self):
        events = _rule_events(
            DefectRuling(RULED_REF, RulingSubject.CLOSURE, True),
            DefectRuling(RULED_REF, RulingSubject.RETIREMENT, False),
        )

        assert events == (
            _reviewer_rule(RULED_REF, RulingSubject.CLOSURE, accepted=True),
            _reviewer_rule(RULED_REF, RulingSubject.RETIREMENT, accepted=False),
        )

    def test_rules_come_in_order_of_first_appearance(self):
        events = _rule_events(
            DefectRuling(SECOND_RULED_REF, RulingSubject.CLOSURE, True),
            DefectRuling(RULED_REF, RulingSubject.CLOSURE, True),
            DefectRuling(SECOND_RULED_REF, RulingSubject.CLOSURE, False),
        )

        assert [event.ref for event in events] == [SECOND_RULED_REF, RULED_REF]
        assert [event.ruling.accepted for event in events] == [False, True]

    def test_no_rulings_is_no_rules(self):
        assert _rule_events() == ()
