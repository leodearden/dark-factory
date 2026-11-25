"""Every payload the filer writes (plans/client-feed-prd.md §14.3, §14.2 decision 10), as pure functions of an inbox line and the policy."""

from datetime import UTC, datetime

import pytest
from _defects import quarantined, reported, restated
from shared.defects import (
    Declare,
    DefectDeclaration,
    DefectStatus,
    Exemption,
    ExemptionReason,
    RuledBy,
    Withdrawal,
    WithdrawalCode,
    WithdrawalReason,
    apply_event,
    parse_declare_event,
    parse_outcome_event,
)
from shared.governed_exceptions import Debt, TaskRef

from client_feed.contract import DefectReported, QuarantineReason
from client_feed.filer_tasks import (
    ClientDefectFacts,
    FilerFallback,
    client_declaration,
    facts_of,
    fix_submission,
    gate_submission,
    is_filer_withdrawal,
    operator_acceptance_event,
    requires_gate,
    withdrawal_event,
)
from client_feed.filer_tools import FILER_IDENTITY, RefusalKind
from client_feed.policy import TaskPriority

PROJECT = 'acme-widgets'
REF = 'portal:acme-widgets:2:3'
AT = datetime(2026, 11, 20, 9, 30, tzinfo=UTC)
OBSERVED = 'the total omits shipping'
EXPECTED = 'the total includes shipping'
STEPS = ('add an item', 'open the basket')
REASONS = (QuarantineReason.SUSPECTED_INJECTION, QuarantineReason.NOT_RESTATABLE)
FALLBACK = FilerFallback(kind=RefusalKind.TOOL_ERROR, code='is_error')


def _restated_line() -> DefectReported:
    return reported(2, 3, at=AT, criterion_id='c-1', outcome=restated(OBSERVED, EXPECTED, STEPS))


def _quarantined_line() -> DefectReported:
    return reported(2, 3, at=AT, criterion_id='c-1', outcome=quarantined(*REASONS))


def test_a_restated_fix_carries_the_restatement_under_its_provenance() -> None:
    fix = fix_submission(_restated_line(), ref=REF, priority=TaskPriority.MEDIUM, fallback=None)

    assert fix.title == 'Client defect portal:acme-widgets:2:3 against criterion c-1'
    assert fix.task_kind == 'normal'
    assert fix.priority == 'medium'
    assert "restated by the portal from the client's report" in fix.description
    assert all(text in fix.description for text in ('c-1', 'acceptance/criteria.yaml', OBSERVED, EXPECTED, *STEPS))
    assert fix.metadata['source'] == FILER_IDENTITY
    assert fix.metadata['defects'] == [
        client_declaration(
            REF,
            summary='Client report 3 on block 2 against criterion c-1',
            declared_at=AT,
            declared_by=FILER_IDENTITY,
        )
    ]
    assert fix.metadata['x_client_defect'] == {
        'block': 2,
        'report': 3,
        'criterion_id': 'c-1',
        'outcome_kind': 'restated',
        'restater_prompt_version': 'restate-v1',
    }


def test_a_quarantined_fix_names_its_reasons_and_carries_no_restatement() -> None:
    fix = fix_submission(_quarantined_line(), ref=REF, priority=TaskPriority.HIGH, fallback=None)

    assert all(reason.value in fix.description for reason in REASONS)
    assert 'Leo writes this description at the gate' in fix.description
    assert 'restated by the portal' not in fix.description
    assert fix.metadata['x_client_defect']['outcome_kind'] == 'quarantined'


@pytest.mark.parametrize('line', [_restated_line(), _quarantined_line()], ids=['restated', 'quarantined'])
def test_a_fallback_fix_names_the_fallback_and_carries_no_client_text(line: DefectReported) -> None:
    fix = fix_submission(line, ref=REF, priority=TaskPriority.HIGH, fallback=FALLBACK)

    assert 'TOOL_ERROR' in fix.description and 'is_error' in fix.description
    assert 'Leo writes this description at the gate' in fix.description
    assert all(text not in str(fix) for text in (OBSERVED, EXPECTED, *STEPS))
    assert fix.metadata['x_client_defect']['filer_fallback'] == {'kind': 'TOOL_ERROR', 'code': 'is_error'}


@pytest.mark.parametrize(
    ('line', 'fallback'),
    [(_restated_line(), None), (_quarantined_line(), None), (_restated_line(), FALLBACK)],
    ids=['restated', 'quarantined', 'fallback'],
)
def test_the_gate_is_a_deterministic_human_gate_naming_the_fix(
    line: DefectReported, fallback: FilerFallback | None
) -> None:
    facts = facts_of(line, fallback)

    gate = gate_submission(
        line, ref=REF, fix_task_id='41', facts=facts, priority=TaskPriority.HIGH, project_id=PROJECT
    )

    assert gate.title == 'Human gate: client defect portal:acme-widgets:2:3'
    assert gate.task_kind == 'deterministic'
    assert gate.priority == 'high'
    assert gate.metadata == {
        'always_escalates': True,
        'source': FILER_IDENTITY,
        'x_client_defect_gate_for': REF,
    }
    assert all(text in gate.description for text in (REF, '41', facts.outcome_kind, PROJECT, 'block 2', 'report 3'))
    assert 'edit task 41 if needed, record x_client_defect_review, then resume' in gate.description
    assert all(text not in gate.description for text in (OBSERVED, EXPECTED, *STEPS))


def test_the_gate_names_quarantine_reasons_and_the_fallback() -> None:
    quarantined_gate = gate_submission(
        _quarantined_line(),
        ref=REF,
        fix_task_id='41',
        facts=facts_of(_quarantined_line(), None),
        priority=TaskPriority.HIGH,
        project_id=PROJECT,
    )
    fallback_gate = gate_submission(
        _restated_line(),
        ref=REF,
        fix_task_id='41',
        facts=facts_of(_restated_line(), FALLBACK),
        priority=TaskPriority.HIGH,
        project_id=PROJECT,
    )

    assert all(reason.value in quarantined_gate.description for reason in REASONS)
    assert 'TOOL_ERROR' in fallback_gate.description and 'is_error' in fallback_gate.description


@pytest.mark.parametrize(
    ('line', 'fallback', 'auto_dispatch', 'gated'),
    [
        (_restated_line(), None, True, False),
        (_restated_line(), None, False, True),
        (_quarantined_line(), None, True, True),
        (_restated_line(), FALLBACK, True, True),
        (_quarantined_line(), FALLBACK, False, True),
    ],
)
def test_only_an_auto_dispatched_clean_restatement_goes_ungated(
    line: DefectReported, fallback: FilerFallback | None, auto_dispatch: bool, gated: bool
) -> None:
    assert requires_gate(facts_of(line, fallback), auto_dispatch=auto_dispatch) is gated


@pytest.mark.parametrize('fallback', [None, FALLBACK], ids=['clean', 'fallback'])
def test_the_facts_read_back_from_a_fix_are_the_facts_it_was_built_from(fallback: FilerFallback | None) -> None:
    fix = fix_submission(_restated_line(), ref=REF, priority=TaskPriority.HIGH, fallback=fallback)

    assert ClientDefectFacts.model_validate(fix.metadata['x_client_defect']) == facts_of(_restated_line(), fallback)


def _declared() -> DefectDeclaration:
    declaration = client_declaration(REF, summary='s', declared_at=AT, declared_by=FILER_IDENTITY)
    return Declare.model_validate({'declaration': declaration}).declaration


def test_the_client_declaration_is_a_fresh_client_defect() -> None:
    declaration = _declared()

    assert (declaration.origin, declaration.status, declaration.sightings) == ('client', DefectStatus.OPEN, 1)


def test_the_filers_withdrawal_and_acceptance_close_the_defect_as_withdrawn() -> None:
    pending = apply_event(_declared(), parse_declare_event(withdrawal_event(REF)))
    withdrawn = apply_event(pending, parse_outcome_event(operator_acceptance_event(REF)))

    assert pending.status is DefectStatus.PENDING
    assert pending.closure == Withdrawal(reason=WithdrawalReason(reason=WithdrawalCode.NOT_A_DEFECT))
    assert withdrawn.status is DefectStatus.WITHDRAWN
    assert withdrawn.ruling is not None
    assert (withdrawn.ruling.accepted, withdrawn.ruling.ruled_by) == (True, RuledBy.OPERATOR)
    assert is_filer_withdrawal(pending.model_dump(mode='json')) is True


def test_another_pending_closure_is_not_the_filers_withdrawal() -> None:
    exempted = _declared().model_copy(
        update={
            'status': DefectStatus.PENDING,
            'closure': Exemption(reason=ExemptionReason.NEEDS_EXTERNAL_SYSTEM, owner=Debt(TaskRef(1))),
        }
    )
    duplicate = _declared().model_copy(
        update={
            'status': DefectStatus.PENDING,
            'closure': Withdrawal(
                reason=WithdrawalReason(reason=WithdrawalCode.DUPLICATE_OF, ref='portal:acme-widgets:2:1')
            ),
        }
    )

    assert is_filer_withdrawal(exempted.model_dump(mode='json')) is False
    assert is_filer_withdrawal(duplicate.model_dump(mode='json')) is False
