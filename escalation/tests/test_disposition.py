"""The route_info_l0 verdict table (plans/info-l0-disposition-router-prd.md
D7/D8/§Contract, boundary rows #11 and #12, decision halves).

route_info_l0 is the pure decision half of the info-L0 disposition router:
one info-severity L0 record plus its exit context in, one frozen disposition
variant out.  These tests pin the precedence chain end to end.
"""

from __future__ import annotations

import dataclasses
from typing import Any, get_args

import pytest
from _info_l0_records import (
    NOTE_DETAIL,
    done_step_tripwire,
    info_l0_note,
    info_scope_divergence,
)

from escalation.classify import DONE_STEP_COMMIT_ORPHAN_CLASS, INFO_L0_MECHANICAL_ROLES
from escalation.disposition import (
    Addressed,
    ConvertViaCurator,
    CuratorCandidate,
    Defer,
    DeferReason,
    Disposition,
    ExitKind,
    ObservationConsumed,
    PromoteReason,
    PromoteWithHint,
    ReviewerDisposition,
    StatusInfo,
    route_info_l0,
)
from escalation.models import RESOLUTION_CLASSES, Escalation


def _route(record: Escalation, **kw: Any) -> Disposition:
    kw.setdefault('task_status', 'done')
    kw.setdefault('exit_kind', ExitKind.DONE)
    kw.setdefault('mechanical_roles', INFO_L0_MECHANICAL_ROLES)
    return route_info_l0(record, **kw)


#: One instance of each Disposition variant, with a field to try mutating.
_ONE_OF_EACH_VARIANT: list[tuple[Disposition, str]] = [
    (Addressed(), 'resolution_class'),
    (ObservationConsumed(), 'resolution_class'),
    (StatusInfo('orchestrator-offline-lane'), 'class_key'),
    (ConvertViaCurator(CuratorCandidate('t', 'd', 'esc-77-3', '77')), 'payload'),
    (PromoteWithHint(PromoteReason.NO_CONVERTIBLE_CONTENT), 'reason'),
    (Defer(DeferReason.REQUEUED_EXIT), 'reason'),
]

_NON_REQUEUE_KINDS = [kind for kind in ExitKind if kind is not ExitKind.REQUEUED]

#: (id, record, reviewer_disposition) — one representative of each leg that a
#: requeue must pre-empt.
_REQUEUE_ROWS = [
    pytest.param(info_l0_note(), None, id='work-note'),
    pytest.param(info_l0_note(suggested_action='No action needed'), None, id='no-action-note'),
    pytest.param(
        info_l0_note(agent_role='orchestrator-starvation-watchdog'),
        None,
        id='starvation-watchdog',
    ),
    pytest.param(info_l0_note(), ReviewerDisposition.ADDRESSED, id='reviewer-addressed'),
]


class TestVocabulary:
    def test_exit_kinds(self):
        assert {m.value for m in ExitKind} == {
            'done', 'merge_deferred', 'blocked', 'cancelled', 'requeued', 'orphan', 'restart',
        }

    def test_reviewer_dispositions(self):
        assert {m.value for m in ReviewerDisposition} == {'addressed', 'no_action', 'work'}

    def test_promote_reasons(self):
        assert {m.value for m in PromoteReason} == {
            'no_convertible_content', 'ticket_failed', 'ticket_refused',
            'ticket_timeout', 'conversion_cap', 'router_error',
        }

    def test_defer_reasons(self):
        assert {m.value for m in DeferReason} == {'requeued_exit'}

    def test_closing_variants_carry_a_legal_resolution_class(self):
        assert Addressed.resolution_class == 'addressed'
        assert ObservationConsumed.resolution_class == 'observation-consumed'
        assert StatusInfo.resolution_class == 'status-info'
        for variant in (Addressed, ObservationConsumed, StatusInfo):
            assert variant.resolution_class in RESOLUTION_CLASSES


class TestDeferOnlyOnRequeue:
    @pytest.mark.parametrize(('record', 'reviewer'), _REQUEUE_ROWS)
    def test_requeued_exit_defers_every_leg(
        self, record: Escalation, reviewer: ReviewerDisposition | None,
    ):
        disposition = _route(
            record, exit_kind=ExitKind.REQUEUED, reviewer_disposition=reviewer,
        )
        assert disposition == Defer(DeferReason.REQUEUED_EXIT)

    @pytest.mark.parametrize('exit_kind', _NON_REQUEUE_KINDS)
    @pytest.mark.parametrize(('record', 'reviewer'), _REQUEUE_ROWS)
    def test_no_other_exit_kind_defers(
        self,
        record: Escalation,
        reviewer: ReviewerDisposition | None,
        exit_kind: ExitKind,
    ):
        disposition = _route(record, exit_kind=exit_kind, reviewer_disposition=reviewer)
        assert not isinstance(disposition, Defer)


class TestReviewerDispositionWins:
    def test_addressed(self):
        disposition = _route(info_l0_note(), reviewer_disposition=ReviewerDisposition.ADDRESSED)
        assert disposition == Addressed()

    def test_no_action_consumes_even_a_work_shaped_note(self):
        disposition = _route(info_l0_note(), reviewer_disposition=ReviewerDisposition.NO_ACTION)
        assert disposition == ObservationConsumed()

    def test_work_converts_even_when_the_author_declared_no_action(self):
        record = info_l0_note(suggested_action='No action needed')
        disposition = _route(record, reviewer_disposition=ReviewerDisposition.WORK)
        assert isinstance(disposition, ConvertViaCurator)
        assert disposition.payload.escalation_id == record.id

    def test_work_without_content_promotes(self):
        record = info_l0_note(summary='', detail='  ')
        disposition = _route(record, reviewer_disposition=ReviewerDisposition.WORK)
        assert disposition == PromoteWithHint(PromoteReason.NO_CONVERTIBLE_CONTENT)

    def test_addressed_beats_the_mechanical_leg(self):
        record = info_l0_note(agent_role='orchestrator-starvation-watchdog')
        disposition = _route(record, reviewer_disposition=ReviewerDisposition.ADDRESSED)
        assert disposition == Addressed()


class TestMechanicalLeg:
    """Boundary row #11: mechanical notices close per class."""

    @pytest.mark.parametrize('role', sorted(INFO_L0_MECHANICAL_ROLES))
    def test_registered_role_is_status_info_keyed_by_role(self, role: str):
        assert _route(info_l0_note(agent_role=role)) == StatusInfo(class_key=role)

    def test_starvation_storm_collapses_to_one_class(self):
        records = [
            info_l0_note(id=f'esc-77-{seq}', agent_role='orchestrator-starvation-watchdog')
            for seq in range(489)
        ]
        dispositions = [
            _route(record, exit_kind=ExitKind.ORPHAN, task_status=None) for record in records
        ]
        assert all(d == StatusInfo('orchestrator-starvation-watchdog') for d in dispositions)
        assert len(set(dispositions)) == 1

    def test_done_step_tripwire_is_its_own_class(self):
        assert _route(done_step_tripwire()) == StatusInfo(DONE_STEP_COMMIT_ORPHAN_CLASS)

    def test_mechanical_leg_precedes_the_observation_leg(self):
        record = info_l0_note(
            agent_role='orchestrator-merge-skew-tripwire', suggested_action='No action needed',
        )
        assert _route(record) == StatusInfo('orchestrator-merge-skew-tripwire')


class TestFailLoud:
    """Boundary row #12: an unrecognised mechanical record reaches the curator,
    never a silent status-info close (D8)."""

    def test_unknown_mechanical_role_converts(self):
        record = info_l0_note(agent_role='orchestrator-some-new-monitor')
        disposition = _route(record, exit_kind=ExitKind.ORPHAN, task_status=None)
        assert isinstance(disposition, ConvertViaCurator)
        assert disposition.payload.escalation_id == record.id

    def test_empty_registry_converts_a_known_role(self):
        record = info_l0_note(agent_role='orchestrator-starvation-watchdog')
        disposition = _route(record, mechanical_roles=frozenset())
        assert isinstance(disposition, ConvertViaCurator)

    def test_info_scope_divergence_shape_converts(self):
        """Not mechanical: see
        test_classify.py::TestInfoL0MechanicalClass::test_scope_divergence_shape_is_not_mechanical."""
        assert isinstance(_route(info_scope_divergence()), ConvertViaCurator)


class TestObservationLeg:
    @pytest.mark.parametrize('exit_kind', _NON_REQUEUE_KINDS)
    @pytest.mark.parametrize(
        'suggested_action',
        [
            'No action needed — recorded so a reviewer sees it',
            'No action required.',
            '  no further action needed',
            'None required. If a reviewer prefers a helper, extract one.',
            'None needed',
        ],
    )
    def test_author_no_action_declaration_is_consumed(
        self, suggested_action: str, exit_kind: ExitKind,
    ):
        record = info_l0_note(suggested_action=suggested_action)
        assert _route(record, exit_kind=exit_kind) == ObservationConsumed()

    @pytest.mark.parametrize(
        ('suggested_action', 'detail'),
        [
            pytest.param(
                'File a follow-up; no action needed on this branch',
                NOTE_DETAIL,
                id='not-at-start',
            ),
            pytest.param('Nonetheless add a test', NOTE_DETAIL, id='word-boundary'),
            pytest.param(
                '', 'Add a regression test for helper X before the next release.', id='blank',
            ),
        ],
    )
    def test_anything_else_goes_to_the_curator(self, suggested_action: str, detail: str):
        record = info_l0_note(suggested_action=suggested_action, detail=detail)
        assert isinstance(_route(record), ConvertViaCurator)


class TestConvertPayload:
    def _payload(self, record: Escalation, **kw: Any) -> CuratorCandidate:
        disposition = _route(record, **kw)
        assert isinstance(disposition, ConvertViaCurator)
        return disposition.payload

    def test_candidate_identity_fields(self):
        record = info_l0_note()
        payload = self._payload(
            record, exit_kind=ExitKind.MERGE_DEFERRED, task_status='merge-deferred',
        )
        assert payload.title == record.summary.strip()
        assert payload.escalation_id == 'esc-77-3'
        assert payload.spawned_from == '77'

    def test_description_stands_alone(self):
        record = info_l0_note()
        payload = self._payload(
            record, exit_kind=ExitKind.MERGE_DEFERRED, task_status='merge-deferred',
        )
        for fragment in (
            record.id,
            record.agent_role,
            record.summary,
            record.suggested_action,
            record.detail,
            'merge-deferred',
            ExitKind.MERGE_DEFERRED.value,
        ):
            assert fragment in payload.description

    def test_unknown_subject_status(self):
        payload = self._payload(info_l0_note(), exit_kind=ExitKind.ORPHAN, task_status=None)
        assert 'unknown' in payload.description

    def test_blank_summary_titles_from_the_first_non_blank_detail_line(self):
        record = info_l0_note(summary='', detail='\n   \n  First real line  \nSecond line')
        assert self._payload(record).title == 'First real line'


class TestTerminus:
    def test_nothing_convertible_promotes(self):
        record = info_l0_note(summary='', detail='   ', suggested_action='Investigate')
        assert _route(record) == PromoteWithHint(PromoteReason.NO_CONVERTIBLE_CONTENT)


class TestPrecondition:
    @pytest.mark.parametrize(
        ('severity', 'level'),
        [('blocking', 0), ('info', 1), ('critical', 2)],
    )
    def test_non_info_or_non_l0_raises_naming_the_record(self, severity: str, level: int):
        record = info_l0_note(severity=severity, level=level)
        with pytest.raises(ValueError, match=record.id):
            _route(record)

    def test_severity_is_normalised_like_the_pin_chain(self):
        assert isinstance(_route(info_l0_note(severity=' Info ')), ConvertViaCurator)


class TestPurity:
    @pytest.mark.parametrize(
        ('record', 'kw'),
        [
            pytest.param(info_l0_note(), {'exit_kind': ExitKind.REQUEUED}, id='defer'),
            pytest.param(
                info_l0_note(),
                {'reviewer_disposition': ReviewerDisposition.ADDRESSED},
                id='reviewer',
            ),
            pytest.param(info_l0_note(agent_role='orchestrator-offline-lane'), {}, id='mechanical'),
            pytest.param(
                info_l0_note(suggested_action='No action required.'), {}, id='observation',
            ),
            pytest.param(info_l0_note(), {}, id='convert'),
            pytest.param(info_l0_note(summary='', detail=''), {}, id='promote'),
        ],
    )
    def test_record_is_not_mutated(self, record: Escalation, kw: dict[str, Any]):
        before = record.to_dict()
        _route(record, **kw)
        assert record.to_dict() == before

    @pytest.mark.parametrize(
        ('value', 'attr'),
        [*_ONE_OF_EACH_VARIANT, (CuratorCandidate('t', 'd', 'esc-77-3', '77'), 'title')],
    )
    def test_dispositions_are_frozen(self, value: object, attr: str):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(value, attr, 'changed')


class TestUnionIsClosed:
    def test_frozen_table_covers_every_disposition_variant(self):
        assert {type(d) for d, _ in _ONE_OF_EACH_VARIANT} == set(get_args(Disposition))
