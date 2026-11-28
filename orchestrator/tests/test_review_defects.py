"""The defect-record events a review round produces, and their delivery (task 7050, defect-tests δ2)."""

from __future__ import annotations

import copy
import logging
from datetime import UTC, datetime

import pytest
from _defect_ruling_fixtures import RULED_REF, SECOND_RULED_REF
from _orch_helpers import mcp_tool_envelope
from shared.defects import (
    DEFECT_TEST_RETIREMENTS_METADATA_KEY,
    DEFECTS_METADATA_KEY,
    Declare,
    DefectOrigin,
    DefectStatus,
    IssueKind,
    Retirement,
    RetirementCode,
    RetirementReason,
    Rule,
    RuledBy,
    Ruling,
    RulingSubject,
    Sighting,
    mint_review_ref,
    parse_declare_event,
    parse_outcome_event,
)

from orchestrator.review_defects import (
    DefectWrite,
    declared_refs,
    review_defect_declarations,
    review_ruling_events,
    send_defect_events,
    with_live_defect_records,
)
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


PROJECT_ROOT = '/srv/project'
TASK_ID = '42'
SUCCESS_REPLY = mcp_tool_envelope({'success': True, 'id': TASK_ID, 'message': 'recorded'})
REFUSAL_REPLY = mcp_tool_envelope({
    'success': False, 'error': 'defect_transition_refused', 'task_id': TASK_ID,
    'detail': 'a declare event is refused from open',
})


class _RecordingDispatch:
    """A SchedulerFacade.dispatch_tool double: records each call, answers from *replies* in turn."""

    def __init__(self, *replies: dict | Exception) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._replies = list(replies)

    async def __call__(self, name: str, arguments: dict, *, timeout: float = 15) -> dict:
        self.calls.append((name, arguments))
        reply = self._replies.pop(0) if self._replies else SUCCESS_REPLY
        if isinstance(reply, Exception):
            raise reply
        return reply


def _declare_event(description: str = DESCRIPTION) -> Declare:
    (event,) = _declarations(_issue(description=description))
    assert isinstance(event, Declare)
    return event


async def _send(dispatch: _RecordingDispatch, *events: Declare | Sighting | Rule):
    return await send_defect_events(
        dispatch, project_root=PROJECT_ROOT, task_id=TASK_ID, events=events,
    )


@pytest.mark.asyncio
class TestSendDefectEvents:
    async def test_declarations_and_sightings_go_to_declare_defect(self):
        declare = _declare_event()
        sighting = Sighting(ref=declare.ref)
        dispatch = _RecordingDispatch()

        await _send(dispatch, declare, sighting)

        assert [name for name, _ in dispatch.calls] == ['declare_defect', 'declare_defect']
        for (_, arguments), event in zip(dispatch.calls, (declare, sighting), strict=True):
            assert arguments == {
                'project_root': PROJECT_ROOT, 'task_id': TASK_ID,
                'event': event.model_dump(mode='json'),
            }
            assert parse_declare_event(arguments['event']) == event

    async def test_a_rule_goes_to_record_defect_outcome_with_its_ref(self):
        rule = _reviewer_rule(RULED_REF, RulingSubject.CLOSURE, accepted=True)
        dispatch = _RecordingDispatch()

        await _send(dispatch, rule)

        ((name, arguments),) = dispatch.calls
        assert name == 'record_defect_outcome'
        assert arguments == {
            'project_root': PROJECT_ROOT, 'task_id': TASK_ID, 'ref': RULED_REF,
            'event': rule.model_dump(mode='json'),
        }
        assert parse_outcome_event(arguments['event']) == rule

    async def test_a_successful_reply_is_a_write_without_failure(self):
        declare = _declare_event()

        writes = await _send(_RecordingDispatch(SUCCESS_REPLY), declare)

        assert writes == (DefectWrite(tool='declare_defect', ref=declare.ref, failure=None),)

    async def test_a_refusal_is_a_logged_failure_and_later_events_are_still_sent(self, caplog):
        refused = _declare_event()
        later = _declare_event(description='another defect')
        dispatch = _RecordingDispatch(REFUSAL_REPLY, SUCCESS_REPLY)

        with caplog.at_level(logging.WARNING, logger='orchestrator.review_defects'):
            first, second = await _send(dispatch, refused, later)

        assert first.failure is not None
        assert 'defect_transition_refused' in first.failure
        assert second.failure is None
        assert len(dispatch.calls) == 2
        (record,) = [r for r in caplog.records if r.name == 'orchestrator.review_defects']
        assert record.levelno == logging.WARNING
        assert record.ref == refused.ref
        assert record.tool == 'declare_defect'

    async def test_a_dispatch_that_raises_is_a_failure_not_an_exception(self):
        declare = _declare_event()

        (write,) = await _send(_RecordingDispatch(TimeoutError('no reply')), declare)

        assert write.failure is not None
        assert 'no reply' in write.failure

    async def test_a_body_that_is_not_a_tool_reply_is_a_failure(self):
        (write,) = await _send(_RecordingDispatch({'unexpected': 'shape'}), _declare_event())

        assert write.failure is not None


def _defect_record(description: str) -> dict:
    return _declare_event(description).declaration.model_dump(mode='json')


def _retirement_record(ref: str) -> dict:
    return Retirement(
        ref=ref, reason=RetirementReason(reason=RetirementCode.FEATURE_REMOVED),
    ).model_dump(mode='json')


STALE_DEFECTS = [_defect_record('declared at dispatch')]
STALE_RETIREMENTS = [_retirement_record(SECOND_RULED_REF)]
LIVE_DEFECTS = [_defect_record('declared at dispatch'), _defect_record('declared mid-run')]
LIVE_RETIREMENTS = [_retirement_record(RULED_REF)]


def _dispatch_task(**metadata: object) -> dict:
    return {
        'id': TASK_ID,
        'title': 'the title dispatch saw',
        'description': 'the description dispatch saw',
        'metadata': {'files': ['lib.py'], **metadata},
    }


def _fresh_task(**metadata: object) -> dict:
    return {
        'id': TASK_ID,
        'title': 'a title edited since dispatch',
        'description': 'a description edited since dispatch',
        'metadata': {'files': ['elsewhere.py'], **metadata},
    }


class TestLiveDefectRecords:
    def test_the_two_defect_keys_come_from_the_fresh_read_and_the_rest_from_dispatch(self):
        dispatch = _dispatch_task(
            **{DEFECTS_METADATA_KEY: STALE_DEFECTS, DEFECT_TEST_RETIREMENTS_METADATA_KEY: STALE_RETIREMENTS},
        )
        fresh = _fresh_task(
            **{DEFECTS_METADATA_KEY: LIVE_DEFECTS, DEFECT_TEST_RETIREMENTS_METADATA_KEY: LIVE_RETIREMENTS},
        )

        live = with_live_defect_records(dispatch, fresh)

        assert live == {
            **dispatch,
            'metadata': {
                'files': ['lib.py'],
                DEFECTS_METADATA_KEY: LIVE_DEFECTS,
                DEFECT_TEST_RETIREMENTS_METADATA_KEY: LIVE_RETIREMENTS,
            },
        }

    def test_keys_the_fresh_read_lacks_are_dropped(self):
        dispatch = _dispatch_task(
            **{DEFECTS_METADATA_KEY: STALE_DEFECTS, DEFECT_TEST_RETIREMENTS_METADATA_KEY: STALE_RETIREMENTS},
        )

        live = with_live_defect_records(dispatch, _fresh_task())

        assert live == {**dispatch, 'metadata': {'files': ['lib.py']}}

    @pytest.mark.parametrize(
        'fresh',
        [None, 'not a task', ['not', 'a', 'task'], {'id': TASK_ID, 'metadata': 'not a dict'}],
        ids=['none', 'str', 'list', 'metadata-not-a-dict'],
    )
    def test_an_unreadable_fresh_read_leaves_the_dispatch_task(self, fresh):
        dispatch = _dispatch_task(**{DEFECTS_METADATA_KEY: STALE_DEFECTS})

        assert with_live_defect_records(dispatch, fresh) == dispatch

    def test_the_dispatch_task_is_not_mutated(self):
        dispatch = _dispatch_task(**{DEFECTS_METADATA_KEY: STALE_DEFECTS})
        before = copy.deepcopy(dispatch)

        live = with_live_defect_records(
            dispatch, _fresh_task(**{DEFECTS_METADATA_KEY: LIVE_DEFECTS}),
        )

        assert dispatch == before
        assert live['metadata'] is not dispatch['metadata']

    def test_declared_refs_are_the_refs_of_readable_entries(self):
        readable = _defect_record('a readable entry')
        metadata = {DEFECTS_METADATA_KEY: [readable, {'ref': 'not a record'}, 'not a dict']}

        assert declared_refs(metadata) == frozenset({readable['ref']})

    @pytest.mark.parametrize(
        'metadata',
        [{}, {DEFECTS_METADATA_KEY: None}, {DEFECTS_METADATA_KEY: {'ref': 'a dict, not a list'}}],
        ids=['absent', 'none', 'not-a-list'],
    )
    def test_no_readable_list_declares_no_refs(self, metadata):
        assert declared_refs(metadata) == frozenset()
