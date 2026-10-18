"""Tests for shared.defects, the defect-tests vocabulary (plans/defect-tests-prd.md).

Every test drives a public seam. The oracles (expected hashes, I2 destinations)
are written here from the PRD, never derived from the module's own tables.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from shared.defects import (
    DefectOrigin,
    DefectProof,
    Exemption,
    ExemptionReason,
    KeepCheck,
    KeepOutcome,
    MalformedDefectRef,
    MissingSymbol,
    NodeId,
    PlanCheck,
    PlanCheckOutcome,
    ProofOutcome,
    Retirement,
    RetirementCode,
    RetirementReason,
    RetirementState,
    RuledBy,
    Ruling,
    RulingSubject,
    SideOutcome,
    SideResult,
    UnavailableReason,
    Withdrawal,
    WithdrawalCode,
    WithdrawalReason,
    mint_explore_ref,
    mint_finding_ref,
    mint_followup_ref,
    mint_human_ref,
    mint_portal_ref,
    mint_redmain_ref,
    mint_review_ref,
    parse_ref,
)
from shared.finding_key import finding_key
from shared.governed_exceptions import Debt, TaskRef, TicketRef, owner_record
from shared.task_metadata import RetryLedger


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
