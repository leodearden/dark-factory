"""Tests for shared.defects, the defect-tests vocabulary (plans/defect-tests-prd.md).

Every test drives a public seam. The oracles (expected hashes, I2 destinations)
are written here from the PRD, never derived from the module's own tables.
"""

from __future__ import annotations

import copy
import hashlib
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from shared.defects import (
    DEFECT_TRANSITIONS,
    Declare,
    DeclareClosure,
    DeclareRetirement,
    DefectDeclaration,
    DefectEventKind,
    DefectOrigin,
    DefectProof,
    DefectStatus,
    DuplicateDefectRef,
    Exemption,
    ExemptionReason,
    IllegalDefectTransition,
    IssueKind,
    IssueKindFlag,
    KeepCheck,
    KeepOutcome,
    MalformedDefectRef,
    MissingSymbol,
    NodeId,
    PlanCheck,
    PlanCheckOutcome,
    Proof,
    ProofOutcome,
    Retirement,
    RetirementCode,
    RetirementReason,
    RetirementState,
    Rule,
    RuledBy,
    Ruling,
    RulingSubject,
    SideOutcome,
    SideResult,
    Sighting,
    UnavailableReason,
    Withdrawal,
    WithdrawalCode,
    WithdrawalReason,
    apply_event,
    apply_retirement_event,
    issue_kind_of,
    mint_explore_ref,
    mint_finding_ref,
    mint_followup_ref,
    mint_human_ref,
    mint_portal_ref,
    mint_redmain_ref,
    mint_review_ref,
    parse_declare_event,
    parse_outcome_event,
    parse_ref,
    require_unique_refs,
)
from shared.finding_key import finding_key
from shared.governed_exceptions import Debt, TaskRef, TicketRef, owner_record
from shared.task_metadata import RetryLedger, parse_metadata


def _norm(text: str) -> str:
    return ' '.join(text.split()).lower()


def _issue_hash(location: str, description: str) -> str:
    basis = f'{_norm(location)}\x1f{_norm(description)}'
    return hashlib.sha256(basis.encode()).hexdigest()[:12]


def _hash8(summary: str) -> str:
    return hashlib.sha256(summary.encode()).hexdigest()[:8]


_SIGNATURE = RetryLedger.compute_merge_outcome_signature('cat', 'hint')
_FINDING_KEY = finding_key('area', 'x.py', 'tag')
_FILED_AT = datetime(2026, 10, 18, 12, 34, 56, 123456, tzinfo=UTC)
_HEX12 = 'abcdef012345'


class TestMint:
    def test_review_ref_is_task_and_issue_hash(self) -> None:
        expected = f'review:7043:{_issue_hash("foo.py::bar", "bad thing")}'
        assert mint_review_ref(7043, 'foo.py::bar', 'bad thing') == expected

    def test_review_ref_normalises_whitespace_and_case(self) -> None:
        assert mint_review_ref(7043, '  Foo.py::bar ', 'Bad   THING\n') == mint_review_ref(
            7043, 'foo.py::bar', 'bad thing'
        )

    def test_review_ref_keeps_location_and_description_apart(self) -> None:
        assert mint_review_ref(7043, 'a', 'bc') != mint_review_ref(7043, 'ab', 'c')

    def test_finding_ref_wraps_the_real_finding_key(self) -> None:
        assert mint_finding_ref(_FINDING_KEY) == f'finding:{_FINDING_KEY}'

    def test_redmain_ref_takes_the_real_main_health_signature(self) -> None:
        ref = mint_redmain_ref(_SIGNATURE)
        assert ref == f'redmain:{_SIGNATURE}'
        assert parse_ref(ref).component('signature') == _SIGNATURE

    def test_followup_ref(self) -> None:
        assert mint_followup_ref(7043, 'abc-1.2_x') == 'followup:7043:abc-1.2_x'

    def test_human_ref_is_basic_utc_seconds_and_hash8(self) -> None:
        expected = f'human:20261018T123456Z:{_hash8("Some summary")}'
        assert mint_human_ref(_FILED_AT, 'Some summary') == expected

    def test_human_ref_converts_an_aware_offset_to_utc(self) -> None:
        plus_two = datetime(2026, 10, 18, 14, 34, 56, tzinfo=timezone(timedelta(hours=2)))
        expected = f'human:20261018T123456Z:{_hash8("Some summary")}'
        assert mint_human_ref(plus_two, 'Some summary') == expected

    def test_human_ref_rejects_a_naive_datetime(self) -> None:
        with pytest.raises(MalformedDefectRef) as excinfo:
            mint_human_ref(datetime(2026, 10, 18, 12, 34, 56), 'Some summary')
        assert excinfo.value.origin is DefectOrigin.HUMAN

    def test_portal_ref(self) -> None:
        assert mint_portal_ref('acme', 'block-3', 2) == 'portal:acme:block-3:2'

    def test_explore_ref(self) -> None:
        assert mint_explore_ref('run-1', 4) == 'explore:run-1:4'


_MINT_REJECTIONS = [
    pytest.param(lambda: mint_review_ref(0, 'l', 'd'), DefectOrigin.REVIEW, id='review-task-0'),
    pytest.param(lambda: mint_review_ref(-1, 'l', 'd'), DefectOrigin.REVIEW, id='review-task--1'),
    pytest.param(
        lambda: mint_review_ref(True, 'l', 'd'), DefectOrigin.REVIEW, id='review-task-bool'
    ),
    pytest.param(
        lambda: mint_followup_ref(0, 'abc'), DefectOrigin.AGENT_FOLLOWUP, id='followup-task-0'
    ),
    pytest.param(
        lambda: mint_followup_ref(-1, 'abc'), DefectOrigin.AGENT_FOLLOWUP, id='followup-task--1'
    ),
    pytest.param(
        lambda: mint_followup_ref(True, 'abc'),
        DefectOrigin.AGENT_FOLLOWUP,
        id='followup-task-bool',
    ),
    pytest.param(
        lambda: mint_followup_ref(7043, 'a:b'),
        DefectOrigin.AGENT_FOLLOWUP,
        id='followup-token-colon',
    ),
    pytest.param(
        lambda: mint_followup_ref(7043, 'a b'),
        DefectOrigin.AGENT_FOLLOWUP,
        id='followup-token-space',
    ),
    pytest.param(lambda: mint_portal_ref('acme', 'b', 0), DefectOrigin.CLIENT, id='portal-n-0'),
    pytest.param(lambda: mint_portal_ref('acme', 'b', -1), DefectOrigin.CLIENT, id='portal-n--1'),
    pytest.param(
        lambda: mint_portal_ref('acme', 'b', True), DefectOrigin.CLIENT, id='portal-n-bool'
    ),
    pytest.param(
        lambda: mint_portal_ref('ac:me', 'b', 1), DefectOrigin.CLIENT, id='portal-token-colon'
    ),
    pytest.param(
        lambda: mint_portal_ref('acme', 'b 1', 1), DefectOrigin.CLIENT, id='portal-token-space'
    ),
    pytest.param(lambda: mint_explore_ref('r', 0), DefectOrigin.EXPLORATORY, id='explore-n-0'),
    pytest.param(lambda: mint_explore_ref('r', -1), DefectOrigin.EXPLORATORY, id='explore-n--1'),
    pytest.param(
        lambda: mint_explore_ref('r', True), DefectOrigin.EXPLORATORY, id='explore-n-bool'
    ),
    pytest.param(
        lambda: mint_explore_ref('r:1', 1), DefectOrigin.EXPLORATORY, id='explore-token-colon'
    ),
    pytest.param(
        lambda: mint_explore_ref('r 1', 1), DefectOrigin.EXPLORATORY, id='explore-token-space'
    ),
    pytest.param(
        lambda: mint_finding_ref('not-a-key'),
        DefectOrigin.REVIEW_INSTRUMENT,
        id='finding-not-fk',
    ),
    pytest.param(
        lambda: mint_redmain_ref(_SIGNATURE.upper()),
        DefectOrigin.RED_MAIN,
        id='redmain-uppercase',
    ),
    pytest.param(
        lambda: mint_redmain_ref(_SIGNATURE[:-1]), DefectOrigin.RED_MAIN, id='redmain-short'
    ),
    pytest.param(
        lambda: mint_redmain_ref(_SIGNATURE + '0'), DefectOrigin.RED_MAIN, id='redmain-long'
    ),
]


@pytest.mark.parametrize(('mint', 'origin'), _MINT_REJECTIONS)
def test_mint_rejection_names_the_origin(mint, origin: DefectOrigin) -> None:
    with pytest.raises(MalformedDefectRef) as excinfo:
        mint()
    assert excinfo.value.origin is origin
    assert origin.value in str(excinfo.value)


_ROUND_TRIP = {
    DefectOrigin.REVIEW: (
        lambda: mint_review_ref(7043, 'foo.py::bar', 'bad thing'),
        {'task': '7043', 'issue_hash': _issue_hash('foo.py::bar', 'bad thing')},
    ),
    DefectOrigin.REVIEW_INSTRUMENT: (lambda: mint_finding_ref(_FINDING_KEY), {'key': _FINDING_KEY}),
    DefectOrigin.RED_MAIN: (lambda: mint_redmain_ref(_SIGNATURE), {'signature': _SIGNATURE}),
    DefectOrigin.AGENT_FOLLOWUP: (
        lambda: mint_followup_ref(7043, 'abc-1.2_x'),
        {'spawned_from': '7043', 'suggestion_hash': 'abc-1.2_x'},
    ),
    DefectOrigin.HUMAN: (
        lambda: mint_human_ref(_FILED_AT, 'Some summary'),
        {'filed_at': '20261018T123456Z', 'hash8': _hash8('Some summary')},
    ),
    DefectOrigin.CLIENT: (
        lambda: mint_portal_ref('acme', 'block-3', 2),
        {'project': 'acme', 'block': 'block-3', 'n': '2'},
    ),
    DefectOrigin.EXPLORATORY: (
        lambda: mint_explore_ref('run-1', 4),
        {'run': 'run-1', 'n': '4'},
    ),
}


def test_round_trip_table_covers_every_origin() -> None:
    assert set(_ROUND_TRIP) == set(DefectOrigin)


@pytest.mark.parametrize('origin', list(_ROUND_TRIP), ids=str)
def test_minted_ref_parses_back_to_its_components(origin: DefectOrigin) -> None:
    mint, components = _ROUND_TRIP[origin]
    minted = mint()
    parsed = parse_ref(minted)
    assert parsed.origin is origin
    assert parsed.ref == minted
    assert dict(parsed.components) == components
    for name, value in components.items():
        assert parsed.component(name) == value


def test_component_accessor_names_the_valid_components() -> None:
    parsed = parse_ref(mint_explore_ref('run-1', 4))
    with pytest.raises(KeyError) as excinfo:
        parsed.component('task')
    assert 'run' in str(excinfo.value)


_PARSE_REJECTIONS = [
    pytest.param('', None, id='empty'),
    pytest.param('bogus:1', None, id='unknown-prefix'),
    pytest.param('review:7043', DefectOrigin.REVIEW, id='too-few-components'),
    pytest.param(f'review:7043:{_HEX12}:x', DefectOrigin.REVIEW, id='too-many-components'),
    pytest.param(f'review:07043:{_HEX12}', DefectOrigin.REVIEW, id='leading-zero'),
    pytest.param('review:7043:ABCDEF012345', DefectOrigin.REVIEW, id='uppercase-hex'),
    pytest.param('human:2026-10-18T12:34:56Z:abcd1234', DefectOrigin.HUMAN, id='colon-timestamp'),
    pytest.param('human:20261399T999999Z:abcd1234', DefectOrigin.HUMAN, id='impossible-time'),
    pytest.param(f'redmain:{_SIGNATURE}\n', DefectOrigin.RED_MAIN, id='trailing-newline'),
    pytest.param(
        f'finding:{_FINDING_KEY}\n',
        DefectOrigin.REVIEW_INSTRUMENT,
        id='finding-trailing-newline',
    ),
    pytest.param('portal:acme:b:0', DefectOrigin.CLIENT, id='portal-n-zero'),
    pytest.param(123, None, id='non-str'),
]


@pytest.mark.parametrize(('value', 'origin'), _PARSE_REJECTIONS)
def test_parse_rejection_names_value_and_origin(
    value: object, origin: DefectOrigin | None
) -> None:
    with pytest.raises(MalformedDefectRef) as excinfo:
        parse_ref(value)
    assert excinfo.value.value == value
    assert excinfo.value.origin is origin
    assert repr(value) in str(excinfo.value)
    if origin is not None:
        assert origin.value in str(excinfo.value)


def test_unknown_prefix_lists_the_known_prefixes() -> None:
    with pytest.raises(MalformedDefectRef) as excinfo:
        parse_ref('bogus:1')
    for prefix in ('review', 'finding', 'redmain', 'followup', 'human', 'portal', 'explore'):
        assert prefix in str(excinfo.value)


# --- Leaf records --------------------------------------------------------------

_REF = mint_human_ref(_FILED_AT, 'Some summary')
_OTHER_REF = mint_explore_ref('run-1', 4)
_BASE_SHA = 'a' * 40
_TIP_SHA = 'b' * 64
_MEASURED_AT = datetime(2026, 10, 18, 13, 0, tzinfo=UTC)
_LATER = _MEASURED_AT + timedelta(hours=1)
_TICKET = 'tkt_0RSM2ECX'


def _assert_round_trips(record: BaseModel) -> None:
    dumped = record.model_dump(mode='json')
    model = type(record)
    assert model.model_validate(dumped) == record
    assert model(**dumped) == record


def _node(name: str = 'test_reproduces') -> NodeId:
    return NodeId(module_root='shared', path='tests/test_x.py', name=name)


def _proof_fields(outcome: ProofOutcome = ProofOutcome.PROVEN, **overrides: Any) -> dict[str, Any]:
    node = _node()
    unavailable = UnavailableReason.TIMEOUT if outcome is ProofOutcome.PROOF_UNAVAILABLE else None
    fields: dict[str, Any] = {
        'ref': _REF,
        'outcome': outcome,
        'base_sha': _BASE_SHA,
        'tip_sha': _TIP_SHA,
        'measured_at': _MEASURED_AT,
        'node_ids': (node,),
        'base': (
            SideResult(
                node_id=node,
                outcome=SideOutcome.ERROR_ATTRIBUTED,
                exc_type='ImportError',
                missing_symbol=MissingSymbol(module='shared.defects', symbol='issue_kind_of'),
            ),
        ),
        'tip': (SideResult(node_id=node, outcome=SideOutcome.PASSED),),
        'unavailable_reason': unavailable,
        'runner_rc': 1,
        'duration_s': 2.5,
    }
    return {**fields, **overrides}


def _proof(outcome: ProofOutcome = ProofOutcome.PROVEN, **overrides: Any) -> DefectProof:
    return DefectProof(**_proof_fields(outcome, **overrides))


def _ruling(
    accepted: bool = True,
    subject: RulingSubject = RulingSubject.CLOSURE,
    ruled_by: RuledBy = RuledBy.REVIEWER,
) -> Ruling:
    cycle = 2 if ruled_by is RuledBy.REVIEWER else None
    return Ruling(subject=subject, accepted=accepted, ruled_by=ruled_by, verdict_cycle=cycle)


def _exemption() -> Exemption:
    return Exemption(reason=ExemptionReason.NEEDS_EXTERNAL_SYSTEM, owner=Debt(TaskRef(5602)))


def _withdrawal(code: WithdrawalCode = WithdrawalCode.NOT_A_DEFECT, ref: str | None = None) -> Withdrawal:
    return Withdrawal(reason=WithdrawalReason(reason=code, ref=ref))


def _retirement(ruling: Ruling | None = None) -> Retirement:
    return Retirement(
        ref=_REF,
        reason=RetirementReason(reason=RetirementCode.FEATURE_REMOVED),
        ruling=ruling,
    )


def _plan_check() -> PlanCheck:
    return PlanCheck(outcome=PlanCheckOutcome.MISSING, defect_refs=(_REF, _OTHER_REF))


def _keep_check(**overrides: Any) -> KeepCheck:
    fields: dict[str, Any] = {
        'outcome': KeepOutcome.REMOVED_RETIRED,
        'removed_refs': (_REF,),
        'retired_refs': (_REF,),
        'incomplete_files': ('tests/test_broken.py',),
        'parent_sha': _BASE_SHA,
        'candidate_sha': _TIP_SHA,
    }
    return KeepCheck(**{**fields, **overrides})


class TestNodeId:
    @pytest.mark.parametrize('field', ['module_root', 'path', 'name'])
    def test_every_part_is_non_empty(self, field: str) -> None:
        fields = {'module_root': 'shared', 'path': 'tests/test_x.py', 'name': 'test_a'}
        with pytest.raises(ValidationError):
            NodeId(**{**fields, field: ''})

    def test_path_may_not_hold_the_node_separator(self) -> None:
        with pytest.raises(ValidationError):
            NodeId(module_root='shared', path='tests/test_x.py::test_a', name='test_a')

    def test_is_frozen(self) -> None:
        node = _node()
        with pytest.raises(ValidationError):
            node.name = 'other'

    def test_round_trips(self) -> None:
        _assert_round_trips(_node())


class TestMissingSymbol:
    def test_needs_at_least_one_part(self) -> None:
        with pytest.raises(ValidationError):
            MissingSymbol(module=None, symbol=None, owner_type=None)

    def test_any_single_part_is_enough(self) -> None:
        _assert_round_trips(MissingSymbol(owner_type='module'))


class TestSideResult:
    def test_error_attributed_needs_a_missing_symbol(self) -> None:
        with pytest.raises(ValidationError):
            SideResult(node_id=_node(), outcome=SideOutcome.ERROR_ATTRIBUTED, exc_type='ImportError')

    @pytest.mark.parametrize(
        'outcome',
        [SideOutcome.PASSED, SideOutcome.FAILED, SideOutcome.SKIPPED, SideOutcome.NOT_COLLECTED],
        ids=str,
    )
    def test_non_error_outcomes_carry_no_missing_symbol(self, outcome: SideOutcome) -> None:
        with pytest.raises(ValidationError):
            SideResult(node_id=_node(), outcome=outcome, missing_symbol=MissingSymbol(module='m'))

    def test_error_unattributed_with_exc_type_only_is_accepted(self) -> None:
        result = SideResult(
            node_id=_node(), outcome=SideOutcome.ERROR_UNATTRIBUTED, exc_type='RuntimeError'
        )
        _assert_round_trips(result)


class TestDefectProof:
    def test_ref_outside_the_grammar_is_rejected_and_named(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _proof(ref='bogus:1')
        assert 'bogus:1' in str(excinfo.value)

    def test_unavailable_needs_a_reason(self) -> None:
        with pytest.raises(ValidationError):
            _proof(ProofOutcome.PROOF_UNAVAILABLE, unavailable_reason=None)

    @pytest.mark.parametrize(
        'outcome', [o for o in ProofOutcome if o is not ProofOutcome.PROOF_UNAVAILABLE], ids=str
    )
    def test_only_unavailable_carries_a_reason(self, outcome: ProofOutcome) -> None:
        with pytest.raises(ValidationError):
            _proof(outcome, unavailable_reason=UnavailableReason.CRASH)

    def test_proven_needs_node_ids(self) -> None:
        with pytest.raises(ValidationError):
            _proof(node_ids=())

    @pytest.mark.parametrize(
        'sha',
        ['A' * 40, 'a' * 39, 'a' * 41, 'g' * 40, 'a' * 63, 'a' * 40 + '\n'],
        ids=['uppercase', 'short', 'long', 'non-hex', 'between-widths', 'trailing-newline'],
    )
    def test_shas_are_git_object_ids(self, sha: str) -> None:
        with pytest.raises(ValidationError):
            _proof(base_sha=sha)
        with pytest.raises(ValidationError):
            _proof(tip_sha=sha)

    def test_measured_at_must_be_aware(self) -> None:
        with pytest.raises(ValidationError):
            _proof(measured_at=datetime(2026, 10, 18, 13, 0))

    def test_duration_is_not_negative(self) -> None:
        with pytest.raises(ValidationError):
            _proof(duration_s=-0.1)

    @pytest.mark.parametrize('outcome', list(ProofOutcome), ids=str)
    def test_a_coherent_proof_round_trips(self, outcome: ProofOutcome) -> None:
        _assert_round_trips(_proof(outcome))


class TestRuling:
    def test_operator_has_no_verdict_cycle(self) -> None:
        with pytest.raises(ValidationError):
            Ruling(
                subject=RulingSubject.CLOSURE,
                accepted=True,
                ruled_by=RuledBy.OPERATOR,
                verdict_cycle=1,
            )

    def test_reviewer_needs_a_verdict_cycle(self) -> None:
        with pytest.raises(ValidationError):
            Ruling(
                subject=RulingSubject.CLOSURE,
                accepted=True,
                ruled_by=RuledBy.REVIEWER,
                verdict_cycle=None,
            )

    def test_cycle_is_not_negative(self) -> None:
        with pytest.raises(ValidationError):
            Ruling(
                subject=RulingSubject.CLOSURE,
                accepted=True,
                ruled_by=RuledBy.REVIEWER,
                verdict_cycle=-1,
            )

    @pytest.mark.parametrize('ruled_by', list(RuledBy), ids=str)
    def test_round_trips(self, ruled_by: RuledBy) -> None:
        _assert_round_trips(_ruling(ruled_by=ruled_by))


class TestExemption:
    @pytest.mark.parametrize(
        ('owner', 'expected'),
        [
            pytest.param(Debt(TaskRef(5602)), Debt(TaskRef(5602)), id='debt'),
            pytest.param({'type': 'task', 'id': 5602}, Debt(TaskRef(5602)), id='task-record'),
            pytest.param({'type': 'ticket', 'id': _TICKET}, Debt(TicketRef(_TICKET)), id='ticket'),
        ],
    )
    def test_owner_reads_as_a_debt(self, owner: object, expected: Debt) -> None:
        exemption = Exemption.model_validate(
            {'reason': ExemptionReason.NO_RUNNABLE_HARNESS, 'owner': owner}
        )
        assert exemption.owner == expected

    def test_owner_wire_is_the_governed_exceptions_owner_record(self) -> None:
        exemption = Exemption(
            reason=ExemptionReason.NO_RUNNABLE_HARNESS, owner=Debt(TicketRef(_TICKET))
        )
        assert exemption.model_dump(mode='json')['owner'] == owner_record(TicketRef(_TICKET))
        _assert_round_trips(exemption)

    @pytest.mark.parametrize(
        'owner',
        [{'type': 'task', 'id': 0}, {'kind': 'task', 'id': 1}, 'task 5602'],
        ids=['task-zero', 'kind-key', 'string'],
    )
    def test_malformed_owner_is_a_validation_error(self, owner: object) -> None:
        with pytest.raises(ValidationError):
            Exemption.model_validate({'reason': 'no_runnable_harness', 'owner': owner})

    def test_unknown_reason_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Exemption.model_validate({'reason': 'flaky', 'owner': {'type': 'task', 'id': 1}})


class TestWithdrawalReason:
    def test_duplicate_of_needs_a_ref(self) -> None:
        with pytest.raises(ValidationError):
            WithdrawalReason(reason=WithdrawalCode.DUPLICATE_OF)

    def test_not_a_defect_takes_no_ref(self) -> None:
        with pytest.raises(ValidationError):
            WithdrawalReason(reason=WithdrawalCode.NOT_A_DEFECT, ref=_OTHER_REF)

    def test_ref_is_grammar_checked(self) -> None:
        with pytest.raises(ValidationError):
            WithdrawalReason(reason=WithdrawalCode.DUPLICATE_OF, ref='bogus:1')

    def test_wire_is_structured(self) -> None:
        reason = WithdrawalReason(reason=WithdrawalCode.DUPLICATE_OF, ref=_OTHER_REF)
        assert reason.model_dump(mode='json') == {'reason': 'duplicate_of', 'ref': _OTHER_REF}
        _assert_round_trips(reason)

    def test_flattened_string_form_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WithdrawalReason.model_validate({'reason': f'duplicate_of:{_OTHER_REF}'})

    def test_withdrawal_round_trips(self) -> None:
        _assert_round_trips(_withdrawal(WithdrawalCode.DUPLICATE_OF, _OTHER_REF))


class TestRetirement:
    def test_superseded_by_needs_a_ref(self) -> None:
        with pytest.raises(ValidationError):
            RetirementReason(reason=RetirementCode.SUPERSEDED_BY)

    def test_feature_removed_takes_no_ref(self) -> None:
        with pytest.raises(ValidationError):
            RetirementReason(reason=RetirementCode.FEATURE_REMOVED, ref=_OTHER_REF)

    def test_reason_ref_is_grammar_checked(self) -> None:
        with pytest.raises(ValidationError):
            RetirementReason(reason=RetirementCode.SUPERSEDED_BY, ref='bogus:1')

    def test_reason_wire_is_structured(self) -> None:
        reason = RetirementReason(reason=RetirementCode.SUPERSEDED_BY, ref=_OTHER_REF)
        assert reason.model_dump(mode='json') == {'reason': 'superseded_by', 'ref': _OTHER_REF}

    def test_ruling_must_be_about_a_retirement(self) -> None:
        with pytest.raises(ValidationError):
            _retirement(_ruling(subject=RulingSubject.CLOSURE))

    def test_cannot_be_superseded_by_itself(self) -> None:
        with pytest.raises(ValidationError):
            Retirement(
                ref=_REF,
                reason=RetirementReason(reason=RetirementCode.SUPERSEDED_BY, ref=_REF),
            )

    @pytest.mark.parametrize(
        ('ruling', 'state'),
        [
            pytest.param(None, RetirementState.PENDING, id='pending'),
            pytest.param(
                _ruling(True, RulingSubject.RETIREMENT), RetirementState.ACCEPTED, id='accepted'
            ),
            pytest.param(
                _ruling(False, RulingSubject.RETIREMENT), RetirementState.REJECTED, id='rejected'
            ),
        ],
    )
    def test_state_derives_from_the_ruling(
        self, ruling: Ruling | None, state: RetirementState
    ) -> None:
        retirement = _retirement(ruling)
        assert retirement.state is state
        _assert_round_trips(retirement)


class TestPlanCheck:
    def test_refs_are_grammar_checked(self) -> None:
        with pytest.raises(ValidationError):
            PlanCheck(outcome=PlanCheckOutcome.UNKNOWN_REF, defect_refs=(_REF, 'bogus:1'))

    def test_round_trips(self) -> None:
        _assert_round_trips(_plan_check())


class TestKeepCheck:
    @pytest.mark.parametrize('field', ['removed_refs', 'retired_refs'])
    def test_refs_are_grammar_checked(self, field: str) -> None:
        with pytest.raises(ValidationError):
            _keep_check(**{field: ('bogus:1',)})

    @pytest.mark.parametrize('field', ['parent_sha', 'candidate_sha'])
    def test_shas_are_git_object_ids(self, field: str) -> None:
        with pytest.raises(ValidationError):
            _keep_check(**{field: 'A' * 40})

    def test_round_trips(self) -> None:
        _assert_round_trips(_keep_check())


_LEAF_SAMPLES = [
    pytest.param(_node, id='NodeId'),
    pytest.param(lambda: MissingSymbol(module='m'), id='MissingSymbol'),
    pytest.param(
        lambda: SideResult(node_id=_node(), outcome=SideOutcome.PASSED), id='SideResult'
    ),
    pytest.param(_proof, id='DefectProof'),
    pytest.param(_ruling, id='Ruling'),
    pytest.param(_exemption, id='Exemption'),
    pytest.param(
        lambda: WithdrawalReason(reason=WithdrawalCode.TEST_WAS_WRONG), id='WithdrawalReason'
    ),
    pytest.param(_withdrawal, id='Withdrawal'),
    pytest.param(
        lambda: RetirementReason(reason=RetirementCode.TEST_RELOCATED), id='RetirementReason'
    ),
    pytest.param(_retirement, id='Retirement'),
    pytest.param(_plan_check, id='PlanCheck'),
    pytest.param(_keep_check, id='KeepCheck'),
]


@pytest.mark.parametrize('build', _LEAF_SAMPLES)
def test_leaf_records_forbid_extra_fields(build) -> None:
    sample = build()
    dumped = {**sample.model_dump(mode='json'), 'bogus': 1}
    with pytest.raises(ValidationError):
        type(sample).model_validate(dumped)


# --- DefectDeclaration and events ---------------------------------------------

_DECLARED_AT = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)


def _declaration(**overrides: Any) -> DefectDeclaration:
    fields: dict[str, Any] = {
        'ref': _REF,
        'origin': DefectOrigin.HUMAN,
        'summary': 'Some summary',
        'declared_at': _DECLARED_AT,
        'declared_by': 'leo',
        'status': DefectStatus.OPEN,
        'sightings': 1,
        'proof': None,
        'closure': None,
        'ruling': None,
    }
    return DefectDeclaration(**{**fields, **overrides})


class TestDefectDeclaration:
    def test_fresh_entry_round_trips(self) -> None:
        _assert_round_trips(_declaration())

    def test_origin_must_match_the_ref(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _declaration(origin=DefectOrigin.REVIEW)
        message = excinfo.value.errors()[0]['msg']
        assert 'review' in message
        assert 'human' in message

    @pytest.mark.parametrize(
        'overrides',
        [
            pytest.param({'sightings': 0}, id='sightings-0'),
            pytest.param({'summary': ''}, id='empty-summary'),
            pytest.param({'declared_by': ''}, id='empty-declared-by'),
            pytest.param({'declared_at': datetime(2026, 10, 18, 12, 0)}, id='naive-declared-at'),
        ],
    )
    def test_field_floors(self, overrides: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            _declaration(**overrides)


_ACCEPTED = _ruling(True)
_REJECTED = _ruling(False)
_SELF_DUPLICATE = _withdrawal(WithdrawalCode.DUPLICATE_OF, _REF)

_COHERENCE = [
    pytest.param(DefectStatus.PENDING, None, _exemption(), None, True, id='pending-ok'),
    pytest.param(DefectStatus.PENDING, None, None, None, False, id='pending-no-closure'),
    pytest.param(
        DefectStatus.PENDING, None, _exemption(), _ACCEPTED, False, id='pending-accepted-ruling'
    ),
    pytest.param(
        DefectStatus.PENDING, None, _exemption(), _REJECTED, False, id='pending-rejected-ruling'
    ),
    pytest.param(DefectStatus.EXEMPTED, None, _exemption(), _ACCEPTED, True, id='exempted-ok'),
    pytest.param(
        DefectStatus.EXEMPTED, None, _withdrawal(), _ACCEPTED, False, id='exempted-withdrawal'
    ),
    pytest.param(
        DefectStatus.EXEMPTED, None, _exemption(), _REJECTED, False, id='exempted-rejected'
    ),
    pytest.param(DefectStatus.WITHDRAWN, None, _withdrawal(), _ACCEPTED, True, id='withdrawn-ok'),
    pytest.param(
        DefectStatus.WITHDRAWN, None, _exemption(), _ACCEPTED, False, id='withdrawn-exemption'
    ),
    pytest.param(
        DefectStatus.WITHDRAWN, None, _withdrawal(), _REJECTED, False, id='withdrawn-rejected'
    ),
    pytest.param(
        DefectStatus.OPEN, None, _exemption(), _REJECTED, True, id='open-retained-rejection'
    ),
    pytest.param(DefectStatus.OPEN, None, _exemption(), None, False, id='open-unruled-closure'),
    pytest.param(DefectStatus.OPEN, None, _exemption(), _ACCEPTED, False, id='open-accepted'),
    pytest.param(
        DefectStatus.OPEN,
        _proof(ProofOutcome.PASSES_AT_BASE),
        None,
        None,
        True,
        id='open-latest-fix-finding',
    ),
    pytest.param(DefectStatus.PROVEN, _proof(), None, None, True, id='proven-ok'),
    pytest.param(
        DefectStatus.PROVEN,
        _proof(),
        _withdrawal(),
        _REJECTED,
        True,
        id='proven-retained-rejection',
    ),
    pytest.param(DefectStatus.PROVEN, None, None, None, False, id='proven-no-proof'),
    pytest.param(
        DefectStatus.PROVEN, _proof(ref=_OTHER_REF), None, None, False, id='proof-other-ref'
    ),
    pytest.param(
        DefectStatus.OPEN,
        None,
        _exemption(),
        _ruling(False, RulingSubject.RETIREMENT),
        False,
        id='retirement-subject-ruling',
    ),
    pytest.param(DefectStatus.OPEN, None, None, _REJECTED, False, id='ruling-without-closure'),
    pytest.param(
        DefectStatus.PENDING, None, _SELF_DUPLICATE, None, False, id='duplicate-of-itself'
    ),
]


@pytest.mark.parametrize(('status', 'proof', 'closure', 'ruling', 'ok'), _COHERENCE)
def test_declaration_coherence(
    status: DefectStatus,
    proof: DefectProof | None,
    closure: Exemption | Withdrawal | None,
    ruling: Ruling | None,
    ok: bool,
) -> None:
    fields = {'status': status, 'proof': proof, 'closure': closure, 'ruling': ruling}
    if ok:
        _assert_round_trips(_declaration(**fields))
    else:
        with pytest.raises(ValidationError):
            _declaration(**fields)


def _declare() -> Declare:
    return Declare(declaration=_declaration())


def _declare_closure() -> DeclareClosure:
    return DeclareClosure(ref=_REF, closure=_exemption())


def _declare_retirement() -> DeclareRetirement:
    return DeclareRetirement(retirement=_retirement())


def _sighting() -> Sighting:
    return Sighting(ref=_REF)


def _proof_event(outcome: ProofOutcome = ProofOutcome.PROVEN) -> Proof:
    return Proof(proof=_proof(outcome, measured_at=_LATER))


def _rule(ruling: Ruling = _ACCEPTED) -> Rule:
    return Rule(ref=_REF, ruling=ruling)


_DECLARE_EVENTS = [
    pytest.param(_declare, Declare, id='declare'),
    pytest.param(_declare_closure, DeclareClosure, id='declare_closure'),
    pytest.param(_declare_retirement, DeclareRetirement, id='declare_retirement'),
    pytest.param(_sighting, Sighting, id='sighting'),
]
_OUTCOME_EVENTS = [
    pytest.param(_proof_event, Proof, id='proof'),
    pytest.param(_rule, Rule, id='rule'),
]


class TestEvents:
    @pytest.mark.parametrize(('build', 'event_type'), _DECLARE_EVENTS)
    def test_declare_event_parses_from_the_wire(self, build, event_type: type) -> None:
        event = build()
        parsed = parse_declare_event(event.model_dump(mode='json'))
        assert type(parsed) is event_type
        assert parsed == event

    @pytest.mark.parametrize(('build', 'event_type'), _OUTCOME_EVENTS)
    def test_outcome_event_parses_from_the_wire(self, build, event_type: type) -> None:
        event = build()
        parsed = parse_outcome_event(event.model_dump(mode='json'))
        assert type(parsed) is event_type
        assert parsed == event

    def test_declare_closure_wire_carries_the_closure_kind(self) -> None:
        parsed = parse_declare_event(
            {
                'event': 'declare_closure',
                'ref': _REF,
                'closure': {
                    'kind': 'exemption',
                    'reason': 'needs_external_system',
                    'owner': {'type': 'ticket', 'id': _TICKET},
                },
            }
        )
        assert isinstance(parsed, DeclareClosure)
        assert parsed.closure == Exemption(
            reason=ExemptionReason.NEEDS_EXTERNAL_SYSTEM, owner=Debt(TicketRef(_TICKET))
        )

    def test_declare_parser_refuses_an_outcome_event(self) -> None:
        with pytest.raises(ValidationError):
            parse_declare_event(_proof_event().model_dump(mode='json'))

    def test_declare_parser_refuses_an_unknown_event(self) -> None:
        with pytest.raises(ValidationError):
            parse_declare_event({'event': 'retract', 'ref': _REF})

    def test_outcome_parser_refuses_a_sighting(self) -> None:
        with pytest.raises(ValidationError):
            parse_outcome_event(_sighting().model_dump(mode='json'))

    @pytest.mark.parametrize(('build', 'event_type'), _DECLARE_EVENTS + _OUTCOME_EVENTS)
    def test_every_event_answers_its_ref(self, build, event_type: type) -> None:
        assert build().ref == _REF

    @pytest.mark.parametrize(
        'overrides',
        [
            pytest.param(
                {'status': DefectStatus.PROVEN, 'proof': _proof()}, id='not-open'
            ),
            pytest.param({'sightings': 2}, id='repeat-sighting'),
            pytest.param({'proof': _proof(ProofOutcome.PASSES_AT_BASE)}, id='with-proof'),
            pytest.param({'closure': _exemption(), 'ruling': _REJECTED}, id='with-closure'),
            pytest.param(
                {'status': DefectStatus.PENDING, 'closure': _exemption()}, id='pending'
            ),
        ],
    )
    def test_declare_takes_only_a_fresh_declaration(self, overrides: dict[str, Any]) -> None:
        declaration = _declaration(**overrides)
        with pytest.raises(ValidationError):
            Declare(declaration=declaration)

    def test_declare_retirement_takes_only_an_unruled_retirement(self) -> None:
        with pytest.raises(ValidationError):
            DeclareRetirement(retirement=_retirement(_ruling(True, RulingSubject.RETIREMENT)))

    def test_declare_closure_refuses_a_duplicate_of_itself(self) -> None:
        with pytest.raises(ValidationError):
            DeclareClosure(ref=_REF, closure=_SELF_DUPLICATE)

    def test_events_are_frozen(self) -> None:
        sighting = _sighting()
        with pytest.raises(ValidationError):
            sighting.ref = _OTHER_REF


# --- I2: apply_event --------------------------------------------------------------

_O, _PE, _PR, _EX, _WD = (
    DefectStatus.OPEN,
    DefectStatus.PENDING,
    DefectStatus.PROVEN,
    DefectStatus.EXEMPTED,
    DefectStatus.WITHDRAWN,
)

_FROM_STATES = {
    'ABSENT': (lambda: None, None),
    'OPEN': (_declaration, _O),
    'OPEN_AFTER_REJECTION': (lambda: _declaration(closure=_exemption(), ruling=_REJECTED), _O),
    'PENDING_EXEMPTION': (lambda: _declaration(status=_PE, closure=_exemption()), _PE),
    'PENDING_WITHDRAWAL': (lambda: _declaration(status=_PE, closure=_withdrawal()), _PE),
    'PROVEN': (lambda: _declaration(status=_PR, proof=_proof()), _PR),
    'EXEMPTED': (
        lambda: _declaration(status=_EX, closure=_exemption(), ruling=_ACCEPTED),
        _EX,
    ),
    'WITHDRAWN': (
        lambda: _declaration(status=_WD, closure=_withdrawal(), ruling=_ACCEPTED),
        _WD,
    ),
}

_EVENTS = {
    'declare': (_declare, DefectEventKind.DECLARE),
    'declare_closure_exemption': (_declare_closure, DefectEventKind.DECLARE_CLOSURE),
    'declare_closure_withdrawal': (
        lambda: DeclareClosure(ref=_REF, closure=_withdrawal()),
        DefectEventKind.DECLARE_CLOSURE,
    ),
    'sighting': (_sighting, DefectEventKind.SIGHTING),
    **{
        f'proof_{outcome}': ((lambda o=outcome: _proof_event(o)), DefectEventKind.PROOF)
        for outcome in ProofOutcome
    },
    'rule_accepted': (lambda: _rule(_ACCEPTED), DefectEventKind.RULE),
    'rule_rejected': (lambda: _rule(_REJECTED), DefectEventKind.RULE),
    'rule_accepted_retirement': (
        lambda: _rule(_ruling(True, RulingSubject.RETIREMENT)),
        DefectEventKind.RULE,
    ),
    'declare_retirement': (_declare_retirement, DefectEventKind.DECLARE_RETIREMENT),
}

# PRD I2, written out by hand. Every pair not listed here is refused.
_OPEN_DESTINATIONS = {
    'declare_closure_exemption': _PE,
    'declare_closure_withdrawal': _PE,
    'sighting': _O,
    'proof_PROVEN': _PR,
    'proof_PASSES_AT_BASE': _O,
    'proof_FAILS_AT_TIP': _O,
    'proof_NOT_COLLECTED': _O,
    'proof_MARKER_NOT_STATIC': _O,
    'proof_PROOF_UNAVAILABLE': _O,
    'proof_PROOF_UNSUPPORTED_RUNNER': _O,
}
_I2_ORACLE = {
    ('ABSENT', 'declare'): _O,
    **{('OPEN', event): status for event, status in _OPEN_DESTINATIONS.items()},
    **{('OPEN_AFTER_REJECTION', event): status for event, status in _OPEN_DESTINATIONS.items()},
    ('PENDING_EXEMPTION', 'rule_accepted'): _EX,
    ('PENDING_EXEMPTION', 'rule_rejected'): _O,
    ('PENDING_EXEMPTION', 'sighting'): _PE,
    ('PENDING_WITHDRAWAL', 'rule_accepted'): _WD,
    ('PENDING_WITHDRAWAL', 'rule_rejected'): _O,
    ('PENDING_WITHDRAWAL', 'sighting'): _PE,
    ('PROVEN', 'proof_PROVEN'): _PR,
    ('PROVEN', 'proof_PASSES_AT_BASE'): _O,
    ('PROVEN', 'proof_FAILS_AT_TIP'): _O,
    ('PROVEN', 'proof_NOT_COLLECTED'): _O,
    ('PROVEN', 'proof_MARKER_NOT_STATIC'): _O,
    ('PROVEN', 'proof_PROOF_UNAVAILABLE'): _PR,
    ('PROVEN', 'proof_PROOF_UNSUPPORTED_RUNNER'): _PR,
    ('PROVEN', 'sighting'): _PR,
    ('EXEMPTED', 'sighting'): _EX,
    ('WITHDRAWN', 'sighting'): _WD,
}


def test_oracle_names_only_real_states_and_events() -> None:
    assert {state for state, _ in _I2_ORACLE} <= set(_FROM_STATES)
    assert {event for _, event in _I2_ORACLE} <= set(_EVENTS)


@pytest.mark.parametrize('state_name', list(_FROM_STATES))
@pytest.mark.parametrize('event_name', list(_EVENTS))
def test_i2_sweep(state_name: str, event_name: str) -> None:
    build_entry, from_status = _FROM_STATES[state_name]
    build_event, kind = _EVENTS[event_name]
    entry = build_entry()
    event = build_event()
    expected = _I2_ORACLE.get((state_name, event_name))
    if expected is not None:
        result = apply_event(entry, event)
        assert result.status is expected
        assert DefectDeclaration.model_validate(result.model_dump(mode='json')) == result
        return
    with pytest.raises(IllegalDefectTransition) as excinfo:
        apply_event(entry, event)
    refusal = excinfo.value
    assert refusal.ref == _REF
    assert refusal.from_status == from_status
    assert refusal.event is kind
    message = str(refusal)
    assert _REF in message
    assert (from_status.value if from_status is not None else 'absent') in message
    assert kind.value in message


# I2's "a later proof": an entry's proof is replaced only by one measured after it.
_PROOF_HOLDERS = {
    'PROVEN': lambda: _declaration(status=_PR, proof=_proof()),
    'OPEN_AFTER_FIX_FINDING': lambda: _declaration(proof=_proof(ProofOutcome.PASSES_AT_BASE)),
}
_NOT_LATER = [
    pytest.param(_MEASURED_AT, id='same-instant'),
    pytest.param(
        _MEASURED_AT.astimezone(timezone(timedelta(hours=2))), id='same-instant-other-offset'
    ),
    pytest.param(_MEASURED_AT - timedelta(seconds=1), id='earlier'),
]


@pytest.mark.parametrize('state_name', list(_PROOF_HOLDERS))
@pytest.mark.parametrize('outcome', list(ProofOutcome), ids=str)
@pytest.mark.parametrize('measured_at', _NOT_LATER)
def test_a_proof_not_measured_after_the_latest_is_refused(
    state_name: str, outcome: ProofOutcome, measured_at: datetime
) -> None:
    entry = _PROOF_HOLDERS[state_name]()
    with pytest.raises(IllegalDefectTransition) as excinfo:
        apply_event(entry, Proof(proof=_proof(outcome, measured_at=measured_at)))
    refusal = excinfo.value
    assert (refusal.ref, refusal.from_status, refusal.event) == (
        _REF,
        entry.status,
        DefectEventKind.PROOF,
    )
    assert measured_at.isoformat() in str(refusal)
    assert _MEASURED_AT.isoformat() in str(refusal)


def _assign_item(container, key, value) -> None:
    container[key] = value


class TestApplyEventEffects:
    def test_sighting_counts_and_changes_nothing_else(self) -> None:
        before = _declaration(status=_PR, proof=_proof())
        after = apply_event(before, _sighting())
        assert after.sightings == before.sightings + 1
        assert after.model_dump(exclude={'sightings'}) == before.model_dump(exclude={'sightings'})

    @pytest.mark.parametrize(
        ('entry', 'outcome', 'status'),
        [
            pytest.param(
                _declaration(status=_PR, proof=_proof()),
                ProofOutcome.PROOF_UNAVAILABLE,
                _PR,
                id='proven-keeps-status-records-unavailable',
            ),
            pytest.param(
                _declaration(proof=_proof(ProofOutcome.PROOF_UNAVAILABLE)),
                ProofOutcome.FAILS_AT_TIP,
                _O,
                id='open-records-fix-finding',
            ),
        ],
    )
    def test_proof_replaces_the_latest_proof(
        self, entry: DefectDeclaration, outcome: ProofOutcome, status: DefectStatus
    ) -> None:
        event = _proof_event(outcome)
        result = apply_event(entry, event)
        assert result.status is status
        assert result.proof == event.proof

    def test_closure_after_rejection_replaces_closure_and_clears_ruling(self) -> None:
        entry = _declaration(closure=_exemption(), ruling=_REJECTED)
        new_closure = _withdrawal(WithdrawalCode.TEST_WAS_WRONG)
        result = apply_event(entry, DeclareClosure(ref=_REF, closure=new_closure))
        assert result.status is _PE
        assert result.closure == new_closure
        assert result.ruling is None

    def test_accepted_rule_keeps_the_closure_and_stores_the_ruling(self) -> None:
        result = apply_event(_declaration(status=_PE, closure=_exemption()), _rule(_ACCEPTED))
        assert result.status is _EX
        assert result.closure == _exemption()
        assert result.ruling == _ACCEPTED

    def test_rejected_rule_reopens_and_keeps_the_rejected_pair(self) -> None:
        result = apply_event(_declaration(status=_PE, closure=_withdrawal()), _rule(_REJECTED))
        assert result.status is _O
        assert result.closure == _withdrawal()
        assert result.ruling == _REJECTED

    def test_declare_on_absent_returns_the_declaration(self) -> None:
        event = _declare()
        assert apply_event(None, event) == event.declaration

    def test_event_for_another_ref_is_refused_naming_both(self) -> None:
        with pytest.raises(IllegalDefectTransition) as excinfo:
            apply_event(_declaration(), Sighting(ref=_OTHER_REF))
        assert _REF in str(excinfo.value)
        assert _OTHER_REF in str(excinfo.value)

    def test_input_entry_is_unchanged(self) -> None:
        entry = _declaration(status=_PE, closure=_exemption())
        before = entry.model_dump()
        apply_event(entry, _rule(_ACCEPTED))
        assert entry.model_dump() == before

    def test_transition_table_is_read_only(self) -> None:
        key = next(iter(DEFECT_TRANSITIONS))
        with pytest.raises(TypeError):
            _assign_item(DEFECT_TRANSITIONS, key, DefectStatus.OPEN)


# --- Retirement lifecycle: apply_retirement_event ------------------------------------

_SUPERSEDED = RetirementReason(reason=RetirementCode.SUPERSEDED_BY, ref=_OTHER_REF)

_RETIREMENT_FROM_STATES = {
    'ABSENT': (lambda: None, None),
    'PENDING': (_retirement, RetirementState.PENDING),
    'ACCEPTED': (
        lambda: _retirement(_ruling(True, RulingSubject.RETIREMENT)),
        RetirementState.ACCEPTED,
    ),
    'REJECTED': (
        lambda: _retirement(_ruling(False, RulingSubject.RETIREMENT)),
        RetirementState.REJECTED,
    ),
}

_RETIREMENT_EVENTS = {
    'declare_retirement': (
        lambda: DeclareRetirement(retirement=Retirement(ref=_REF, reason=_SUPERSEDED)),
        DefectEventKind.DECLARE_RETIREMENT,
    ),
    'rule_accepted': (
        lambda: _rule(_ruling(True, RulingSubject.RETIREMENT)),
        DefectEventKind.RULE,
    ),
    'rule_rejected': (
        lambda: _rule(_ruling(False, RulingSubject.RETIREMENT)),
        DefectEventKind.RULE,
    ),
    'rule_accepted_closure': (lambda: _rule(_ACCEPTED), DefectEventKind.RULE),
    'declare': (_declare, DefectEventKind.DECLARE),
    'declare_closure': (_declare_closure, DefectEventKind.DECLARE_CLOSURE),
    'sighting': (_sighting, DefectEventKind.SIGHTING),
    'proof_PROVEN': (_proof_event, DefectEventKind.PROOF),
}

# PRD I2 / decision 13, written out by hand. Every pair not listed here is refused.
_RETIREMENT_ORACLE = {
    ('ABSENT', 'declare_retirement'): RetirementState.PENDING,
    ('PENDING', 'rule_accepted'): RetirementState.ACCEPTED,
    ('PENDING', 'rule_rejected'): RetirementState.REJECTED,
    ('REJECTED', 'declare_retirement'): RetirementState.PENDING,
}


def test_retirement_oracle_names_only_real_states_and_events() -> None:
    assert {state for state, _ in _RETIREMENT_ORACLE} <= set(_RETIREMENT_FROM_STATES)
    assert {event for _, event in _RETIREMENT_ORACLE} <= set(_RETIREMENT_EVENTS)


@pytest.mark.parametrize('state_name', list(_RETIREMENT_FROM_STATES))
@pytest.mark.parametrize('event_name', list(_RETIREMENT_EVENTS))
def test_retirement_sweep(state_name: str, event_name: str) -> None:
    build_record, from_state = _RETIREMENT_FROM_STATES[state_name]
    build_event, kind = _RETIREMENT_EVENTS[event_name]
    record = build_record()
    event = build_event()
    expected = _RETIREMENT_ORACLE.get((state_name, event_name))
    if expected is not None:
        result = apply_retirement_event(record, event)
        assert result.state is expected
        assert Retirement.model_validate(result.model_dump(mode='json')) == result
        return
    with pytest.raises(IllegalDefectTransition) as excinfo:
        apply_retirement_event(record, event)
    refusal = excinfo.value
    assert refusal.ref == _REF
    assert refusal.from_status == from_state
    assert refusal.event is kind


class TestApplyRetirementEventEffects:
    def test_redeclaring_a_rejected_retirement_carries_the_new_reason_unruled(self) -> None:
        rejected = _retirement(_ruling(False, RulingSubject.RETIREMENT))
        event = DeclareRetirement(retirement=Retirement(ref=_REF, reason=_SUPERSEDED))
        result = apply_retirement_event(rejected, event)
        assert result.reason == _SUPERSEDED
        assert result.ruling is None

    def test_rule_stores_the_ruling(self) -> None:
        ruling = _ruling(True, RulingSubject.RETIREMENT, RuledBy.OPERATOR)
        result = apply_retirement_event(_retirement(), _rule(ruling))
        assert result.ruling == ruling
        assert result.reason == _retirement().reason

    def test_rule_for_another_ref_is_refused_naming_both(self) -> None:
        event = Rule(ref=_OTHER_REF, ruling=_ruling(True, RulingSubject.RETIREMENT))
        with pytest.raises(IllegalDefectTransition) as excinfo:
            apply_retirement_event(_retirement(), event)
        assert _REF in str(excinfo.value)
        assert _OTHER_REF in str(excinfo.value)


# --- issue_kind_of (decision 16) ---------------------------------------------------


class TestIssueKindOf:
    @pytest.mark.parametrize('kind', list(IssueKind), ids=str)
    def test_a_known_kind_reads_unflagged(self, kind: IssueKind) -> None:
        reading = issue_kind_of({'kind': kind.value, 'severity': 'blocking'})
        assert reading.kind is kind
        assert reading.flag is None
        assert reading.flagged is False

    @pytest.mark.parametrize(
        'issue',
        [{'severity': 'blocking'}, {'kind': None, 'severity': 'blocking'}],
        ids=['absent', 'none'],
    )
    def test_a_missing_kind_fails_closed_to_defect(self, issue: dict[str, object]) -> None:
        reading = issue_kind_of(issue)
        assert reading.kind is IssueKind.DEFECT
        assert reading.flag is IssueKindFlag.MISSING
        assert reading.flagged is True

    @pytest.mark.parametrize(
        'value', ['bug', 'Defect', 1, ['defect']], ids=['bug', 'wrong-case', 'int', 'list']
    )
    def test_an_unrecognised_kind_fails_closed_to_defect(self, value: object) -> None:
        reading = issue_kind_of({'kind': value, 'severity': 'blocking'})
        assert reading.kind is IssueKind.DEFECT
        assert reading.flag is IssueKindFlag.UNRECOGNISED
        assert reading.flagged is True

    def test_the_issue_is_not_mutated(self) -> None:
        issue: dict[str, object] = {'kind': ['defect'], 'severity': 'blocking'}
        before = copy.deepcopy(issue)
        issue_kind_of(issue)
        assert issue == before


# --- Registration, through parse_metadata -----------------------------------------


def _proven_entry() -> DefectDeclaration:
    return _declaration(
        ref=_OTHER_REF,
        origin=DefectOrigin.EXPLORATORY,
        status=_PR,
        proof=_proof(ref=_OTHER_REF),
    )


def _accepted_retirement() -> Retirement:
    return _retirement(_ruling(True, RulingSubject.RETIREMENT))


_REGISTERED_KEYS = [
    pytest.param('defects', _declaration, id='defects'),
    pytest.param('defect_test_retirements', _accepted_retirement, id='defect_test_retirements'),
]


class TestMetadataRegistration:
    def test_both_keys_parse_to_typed_slices(self) -> None:
        fresh, proven, retirement = _declaration(), _proven_entry(), _accepted_retirement()
        model, warnings = parse_metadata(
            {
                'defects': [fresh.model_dump(mode='json'), proven.model_dump(mode='json')],
                'defect_test_retirements': [retirement.model_dump(mode='json')],
            },
            direction='write',
            enforce=True,
        )
        assert warnings == []
        assert model.model_extra is not None
        assert model.model_extra['defects'] == [fresh, proven]
        assert model.model_extra['defect_test_retirements'] == [retirement]

    @pytest.mark.parametrize(('key', 'build'), _REGISTERED_KEYS)
    def test_a_single_mapping_is_the_wrong_cardinality(self, key: str, build) -> None:
        blob = {key: build().model_dump(mode='json')}
        _model, warnings = parse_metadata(blob, direction='read')
        assert [(w.field, w.code) for w in warnings] == [(key, 'wrong_cardinality')]
        with pytest.raises(TypeError):
            parse_metadata(blob, direction='write', enforce=True)

    def test_an_entry_outside_the_grammar_is_an_invalid_submodel(self) -> None:
        raw = [{**_declaration().model_dump(mode='json'), 'ref': 'bogus:1'}]
        model, warnings = parse_metadata({'defects': raw}, direction='read')
        assert [(w.field, w.code) for w in warnings] == [('defects', 'invalid_submodel')]
        assert model.model_dump()['defects'] == raw
        with pytest.raises(ValidationError):
            parse_metadata({'defects': raw}, direction='write', enforce=True)

    def test_a_malformed_owner_never_escapes_parse_metadata(self) -> None:
        pending = _declaration(status=_PE, closure=_exemption()).model_dump(mode='json')
        pending['closure']['owner'] = {'type': 'task', 'id': 0}
        _model, warnings = parse_metadata({'defects': [pending]}, direction='read')
        assert [(w.field, w.code) for w in warnings] == [('defects', 'invalid_submodel')]


# --- I1: one record per ref within a task ------------------------------------------


class TestRequireUniqueRefs:
    def test_distinct_refs_pass(self) -> None:
        require_unique_refs([_declaration(), _proven_entry()])

    def test_no_records_pass(self) -> None:
        require_unique_refs([])

    def test_a_repeated_ref_is_refused_however_the_records_differ(self) -> None:
        with pytest.raises(DuplicateDefectRef) as excinfo:
            require_unique_refs([_declaration(), _declaration(summary='Another summary')])
        assert excinfo.value.refs == (_REF,)
        assert _REF in str(excinfo.value)

    def test_each_repeated_ref_is_named_once_in_first_seen_order(self) -> None:
        once = _declaration(ref=mint_explore_ref('run-1', 9), origin=DefectOrigin.EXPLORATORY)
        records = [_proven_entry(), _declaration(), once, _declaration(), _proven_entry()]
        with pytest.raises(DuplicateDefectRef) as excinfo:
            require_unique_refs(records)
        assert excinfo.value.refs == (_OTHER_REF, _REF)

    def test_retirements_are_unique_by_ref_too(self) -> None:
        with pytest.raises(DuplicateDefectRef) as excinfo:
            require_unique_refs([_retirement(), _accepted_retirement()])
        assert excinfo.value.refs == (_REF,)
