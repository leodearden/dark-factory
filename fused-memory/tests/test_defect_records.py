"""Pure tests of fused_memory.backends.defect_records: the write rules for a task's defect keys.

No IO. Records are built with the shared.defects models and stored as their
``model_dump(mode='json')``, the shape a task's metadata blob holds. The I2
table and the retirement lifecycle are shared.defects' (tested there); these
tests pin that the write path routes every event through them, surfaces their
refusals as structured data, and floors what a creating write may carry.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, get_args

import pytest

from fused_memory.backends.defect_records import (
    DefectCreationRefused,
    DefectEventInvalid,
    DefectRecords,
    DefectRefMismatch,
    DefectRefsNotUnique,
    DefectTransitionRefused,
    DefectWriteRefused,
    StoredDefectRecordsMalformed,
    parse_declare_wire,
    parse_outcome_wire,
    refs_required_live,
    require_creation_floor,
)
from fused_memory.backends.task_backend_errors import TaskmasterError
from shared.defects import (
    DEFECT_TEST_RETIREMENTS_METADATA_KEY,
    DEFECTS_METADATA_KEY,
    Declare,
    DeclareClosure,
    DeclareEvent,
    DeclareRetirement,
    DefectDeclaration,
    DefectOrigin,
    DefectProof,
    DefectStatus,
    DefectTrigger,
    Exemption,
    ExemptionReason,
    NodeId,
    OutcomeEvent,
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
    Withdrawal,
    WithdrawalCode,
    WithdrawalReason,
    mint_explore_ref,
    mint_human_ref,
)
from shared.governed_exceptions import Debt, TaskRef

_REF = mint_human_ref(datetime(2026, 10, 18, 12, 34, 56, tzinfo=UTC), 'Some summary')
_OTHER_REF = mint_explore_ref('run-1', 4)
_DECLARED_AT = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)
_MEASURED_AT = datetime(2026, 10, 18, 13, 0, tzinfo=UTC)


def _declaration(ref: str = _REF, **overrides: Any) -> DefectDeclaration:
    origin = DefectOrigin.HUMAN if ref == _REF else DefectOrigin.EXPLORATORY
    fields: dict[str, Any] = {
        'ref': ref,
        'origin': origin,
        'summary': 'Some summary',
        'declared_at': _DECLARED_AT,
        'declared_by': 'leo',
        'status': DefectStatus.OPEN,
    }
    return DefectDeclaration(**{**fields, **overrides})


def _proof(ref: str = _REF, measured_at: datetime = _MEASURED_AT) -> DefectProof:
    node = NodeId(module_root='fused-memory', path='tests/test_x.py', name='test_reproduces')
    return DefectProof(
        ref=ref,
        outcome=ProofOutcome.PROVEN,
        base_sha='a' * 40,
        tip_sha='b' * 40,
        measured_at=measured_at,
        node_ids=(node,),
        base=(SideResult(node_id=node, outcome=SideOutcome.FAILED),),
        tip=(SideResult(node_id=node, outcome=SideOutcome.PASSED),),
        duration_s=1.0,
    )


def _ruling(subject: RulingSubject, accepted: bool = True) -> Ruling:
    return Ruling(subject=subject, accepted=accepted, ruled_by=RuledBy.OPERATOR)


def _exemption() -> Exemption:
    return Exemption(reason=ExemptionReason.NEEDS_EXTERNAL_SYSTEM, owner=Debt(TaskRef(5602)))


def _withdrawal(code: WithdrawalCode = WithdrawalCode.NOT_A_DEFECT, ref: str | None = None) -> Withdrawal:
    return Withdrawal(reason=WithdrawalReason(reason=code, ref=ref))


def _retirement(
    ruling: Ruling | None = None,
    code: RetirementCode = RetirementCode.FEATURE_REMOVED,
    superseded_by: str | None = None,
) -> Retirement:
    reason = RetirementReason(reason=code, ref=superseded_by)
    return Retirement(ref=_REF, reason=reason, ruling=ruling)


def _json(record: Any) -> dict[str, Any]:
    return record.model_dump(mode='json')


def _stored(*declarations: DefectDeclaration, **extra: Any) -> dict[str, Any]:
    return {DEFECTS_METADATA_KEY: [_json(d) for d in declarations], **extra}


def _event_tags(union: Any) -> list[str]:
    members = get_args(get_args(union)[0])
    return [member.model_fields['event'].default.value for member in members]


# --- (a) the fold appends, and reports only the keys it addressed -------------


def test_empty_metadata_reads_as_no_records() -> None:
    records = DefectRecords.from_metadata({})
    assert records.defects == ()
    assert records.retirements == ()


def test_declare_appends_an_open_entry_and_changes_only_the_defects_key() -> None:
    declaration = _declaration()
    stored = {'files': ['x.py'], 'x_note': 'kept'}

    change = DefectRecords.from_metadata(stored).apply([Declare(declaration=declaration)])

    assert change.records.defects == (declaration,)
    assert change.records.defects[0].status is DefectStatus.OPEN
    assert change.changed_metadata() == {DEFECTS_METADATA_KEY: [_json(declaration)]}


def test_refusals_are_taskmaster_errors() -> None:
    assert issubclass(DefectWriteRefused, TaskmasterError)


# --- (b) sightings, and transitions I2 refuses ---------------------------------


def test_sighting_on_an_existing_ref_increments_sightings() -> None:
    change = DefectRecords.from_metadata(_stored(_declaration())).apply([Sighting(ref=_REF)])
    assert change.records.defects[0].sightings == 2


def test_sighting_on_an_absent_ref_is_refused_with_the_transition_fields() -> None:
    with pytest.raises(DefectTransitionRefused) as excinfo:
        DefectRecords.from_metadata({}).apply([Sighting(ref=_REF)])
    wire = excinfo.value.to_error_dict()
    assert wire['success'] is False
    assert wire['error'] == 'defect_transition_refused'
    assert wire['ref'] == _REF
    assert wire['from_status'] is None
    assert wire['event'] == 'sighting'
    assert wire['trigger'] == DefectTrigger.SIGHTING.value
    assert _REF in wire['detail']


def test_declaring_an_existing_ref_is_refused_from_open() -> None:
    with pytest.raises(DefectTransitionRefused) as excinfo:
        DefectRecords.from_metadata(_stored(_declaration())).apply(
            [Declare(declaration=_declaration())]
        )
    wire = excinfo.value.to_error_dict()
    assert wire['from_status'] == DefectStatus.OPEN.value
    assert wire['event'] == 'declare'
    assert wire['trigger'] == DefectTrigger.DECLARE.value


def test_to_error_dict_carries_the_task_id_it_is_given() -> None:
    with pytest.raises(DefectTransitionRefused) as excinfo:
        DefectRecords.from_metadata({}).apply([Sighting(ref=_REF)])
    assert excinfo.value.to_error_dict(task_id='42')['task_id'] == '42'
    assert excinfo.value.to_error_dict()['task_id'] is None


# --- (c) closures and their rulings --------------------------------------------


def test_declare_closure_makes_the_entry_pending() -> None:
    change = DefectRecords.from_metadata(_stored(_declaration())).apply(
        [DeclareClosure(ref=_REF, closure=_exemption())]
    )
    assert change.records.defects[0].status is DefectStatus.PENDING


@pytest.mark.parametrize(
    ('closure', 'accepted', 'expected'),
    [
        pytest.param(_exemption(), True, DefectStatus.EXEMPTED, id='exemption-accepted'),
        pytest.param(_withdrawal(), True, DefectStatus.WITHDRAWN, id='withdrawal-accepted'),
        pytest.param(_exemption(), False, DefectStatus.OPEN, id='rejected'),
    ],
)
def test_a_closure_ruling_settles_the_entry(
    closure: Exemption | Withdrawal, accepted: bool, expected: DefectStatus
) -> None:
    change = DefectRecords.from_metadata(_stored(_declaration())).apply(
        [
            DeclareClosure(ref=_REF, closure=closure),
            Rule(ref=_REF, ruling=_ruling(RulingSubject.CLOSURE, accepted)),
        ]
    )
    assert change.records.defects[0].status is expected


# --- (d) proofs ----------------------------------------------------------------


def test_a_proven_proof_proves_an_open_entry() -> None:
    change = DefectRecords.from_metadata(_stored(_declaration())).apply([Proof(proof=_proof())])
    entry = change.records.defects[0]
    assert entry.status is DefectStatus.PROVEN
    assert entry.proof == _proof()


@pytest.mark.parametrize(
    'measured_at',
    [
        pytest.param(_MEASURED_AT, id='same-instant'),
        pytest.param(_MEASURED_AT - timedelta(minutes=1), id='earlier'),
    ],
)
def test_a_proof_not_later_than_the_stored_one_is_refused(measured_at: datetime) -> None:
    proven = _declaration(status=DefectStatus.PROVEN, proof=_proof())
    with pytest.raises(DefectTransitionRefused) as excinfo:
        DefectRecords.from_metadata(_stored(proven)).apply(
            [Proof(proof=_proof(measured_at=measured_at))]
        )
    assert excinfo.value.to_error_dict()['from_status'] == DefectStatus.PROVEN.value


# --- (e) retirements ------------------------------------------------------------


def test_declare_retirement_lands_in_the_retirements_key_only() -> None:
    declaration = _declaration()
    retirement = _retirement()

    change = DefectRecords.from_metadata(_stored(declaration)).apply(
        [DeclareRetirement(retirement=retirement)]
    )

    assert change.records.retirements == (retirement,)
    assert change.records.retirements[0].state is RetirementState.PENDING
    assert change.records.defects == (declaration,)
    assert change.changed_metadata() == {DEFECT_TEST_RETIREMENTS_METADATA_KEY: [_json(retirement)]}


def test_an_accepted_retirement_ruling_accepts_the_retirement() -> None:
    stored = {DEFECT_TEST_RETIREMENTS_METADATA_KEY: [_json(_retirement())]}
    change = DefectRecords.from_metadata(stored).apply(
        [Rule(ref=_REF, ruling=_ruling(RulingSubject.RETIREMENT))]
    )
    assert change.records.retirements[0].state is RetirementState.ACCEPTED
    assert DEFECTS_METADATA_KEY not in change.changed_metadata()


def test_a_retirement_ruling_with_no_retirement_record_is_refused() -> None:
    with pytest.raises(DefectTransitionRefused) as excinfo:
        DefectRecords.from_metadata(_stored(_declaration())).apply(
            [Rule(ref=_REF, ruling=_ruling(RulingSubject.RETIREMENT))]
        )
    assert excinfo.value.to_error_dict()['ref'] == _REF


# --- (f) several events, all or nothing ----------------------------------------


def test_events_apply_in_order() -> None:
    change = DefectRecords.from_metadata({}).apply(
        [Declare(declaration=_declaration()), Sighting(ref=_REF), Sighting(ref=_REF)]
    )
    assert change.records.defects[0].sightings == 3


def test_an_illegal_second_event_refuses_the_whole_batch() -> None:
    records = DefectRecords.from_metadata({})
    with pytest.raises(DefectTransitionRefused) as excinfo:
        records.apply([Declare(declaration=_declaration()), Sighting(ref=_OTHER_REF)])
    assert excinfo.value.to_error_dict()['ref'] == _OTHER_REF
    assert records.defects == ()


# --- (g) malformed stored records ----------------------------------------------


@pytest.mark.parametrize(
    ('stored', 'key', 'index'),
    [
        pytest.param(
            {DEFECTS_METADATA_KEY: _json(_declaration())}, DEFECTS_METADATA_KEY, None, id='not-a-list'
        ),
        pytest.param({DEFECTS_METADATA_KEY: None}, DEFECTS_METADATA_KEY, None, id='null'),
        pytest.param(
            {DEFECTS_METADATA_KEY: [_json(_declaration()), {'ref': _OTHER_REF}]},
            DEFECTS_METADATA_KEY,
            1,
            id='bad-entry',
        ),
        pytest.param(
            {DEFECT_TEST_RETIREMENTS_METADATA_KEY: [{'ref': 'bogus:1'}]},
            DEFECT_TEST_RETIREMENTS_METADATA_KEY,
            0,
            id='bad-retirement',
        ),
    ],
)
def test_malformed_stored_records_are_refused_naming_key_and_index(
    stored: dict[str, Any], key: str, index: int | None
) -> None:
    with pytest.raises(StoredDefectRecordsMalformed) as excinfo:
        DefectRecords.from_metadata(stored)
    wire = excinfo.value.to_error_dict()
    assert wire['error'] == 'stored_defect_records_malformed'
    assert wire['key'] == key
    assert wire['index'] == index


def test_a_stored_list_repeating_a_ref_is_refused_naming_the_ref() -> None:
    with pytest.raises(DefectRefsNotUnique) as excinfo:
        DefectRecords.from_metadata(_stored(_declaration(), _declaration()))
    wire = excinfo.value.to_error_dict()
    assert wire['error'] == 'defect_refs_not_unique'
    assert wire['key'] == DEFECTS_METADATA_KEY
    assert wire['refs'] == [_REF]


# --- (h) refs another task must already declare ---------------------------------


@pytest.mark.parametrize(
    ('event', 'expected'),
    [
        pytest.param(
            DeclareClosure(ref=_REF, closure=_withdrawal(WithdrawalCode.DUPLICATE_OF, _OTHER_REF)),
            ((_OTHER_REF, 'duplicate_of'),),
            id='duplicate-of',
        ),
        pytest.param(
            DeclareRetirement(
                retirement=_retirement(code=RetirementCode.SUPERSEDED_BY, superseded_by=_OTHER_REF)
            ),
            ((_OTHER_REF, 'superseded_by'),),
            id='superseded-by',
        ),
        pytest.param(DeclareClosure(ref=_REF, closure=_withdrawal()), (), id='not-a-defect'),
        pytest.param(DeclareClosure(ref=_REF, closure=_exemption()), (), id='exemption'),
        pytest.param(DeclareRetirement(retirement=_retirement()), (), id='feature-removed'),
        pytest.param(Declare(declaration=_declaration()), (), id='declare'),
        pytest.param(Sighting(ref=_REF), (), id='sighting'),
        pytest.param(Proof(proof=_proof()), (), id='proof'),
        pytest.param(Rule(ref=_REF, ruling=_ruling(RulingSubject.CLOSURE)), (), id='rule'),
    ],
)
def test_refs_required_live(event: Any, expected: tuple[tuple[str, str], ...]) -> None:
    assert refs_required_live([event]) == expected


# --- (i) the creation floor ---------------------------------------------------


_ALLOWED_DEFECTS = {
    'status': [DefectStatus.OPEN.value],
    'origin': [origin.value for origin in DefectOrigin],
}
_ALLOWED_RETIREMENTS = {
    'reason': [code.value for code in RetirementCode],
    'ruling': [None],
}
_ALLOWED = {
    DEFECTS_METADATA_KEY: _ALLOWED_DEFECTS,
    DEFECT_TEST_RETIREMENTS_METADATA_KEY: _ALLOWED_RETIREMENTS,
}


@pytest.mark.parametrize(
    'metadata',
    [
        pytest.param(None, id='none'),
        pytest.param({}, id='empty'),
        pytest.param({'files': ['x.py']}, id='no-defect-keys'),
        pytest.param(
            {DEFECTS_METADATA_KEY: [_json(_declaration()), _json(_declaration(_OTHER_REF))]},
            id='fresh-entries',
        ),
        pytest.param({DEFECT_TEST_RETIREMENTS_METADATA_KEY: [_json(_retirement())]}, id='unruled'),
    ],
)
def test_the_creation_floor_admits_fresh_records(metadata: dict[str, Any] | None) -> None:
    require_creation_floor(metadata)


def _with(entry: dict[str, Any], **changes: Any) -> dict[str, Any]:
    return {**entry, **changes}


_FRESH = _json(_declaration())

_CREATION_REFUSALS = [
    pytest.param(
        DEFECTS_METADATA_KEY,
        [_json(_declaration(_OTHER_REF)), _json(_declaration(status=DefectStatus.PROVEN, proof=_proof()))],
        1,
        id='proven',
    ),
    pytest.param(DEFECTS_METADATA_KEY, [_json(_declaration(sightings=2))], 0, id='sightings-2'),
    pytest.param(
        DEFECTS_METADATA_KEY,
        [_json(_declaration(status=DefectStatus.PENDING, closure=_exemption()))],
        0,
        id='closure',
    ),
    pytest.param(
        DEFECTS_METADATA_KEY,
        [
            _json(
                _declaration(closure=_exemption(), ruling=_ruling(RulingSubject.CLOSURE, False))
            )
        ],
        0,
        id='ruling',
    ),
    pytest.param(DEFECTS_METADATA_KEY, [_with(_FRESH, origin='bogus')], 0, id='unknown-origin'),
    pytest.param(DEFECTS_METADATA_KEY, [_with(_FRESH, ref='bogus:1')], 0, id='malformed-ref'),
    pytest.param(DEFECTS_METADATA_KEY, [_with(_FRESH, status='open-ish')], 0, id='unknown-status'),
    pytest.param(DEFECTS_METADATA_KEY, _FRESH, None, id='not-a-list'),
    pytest.param(DEFECTS_METADATA_KEY, None, None, id='null'),
    pytest.param(DEFECTS_METADATA_KEY, [_FRESH, _FRESH], None, id='repeated-ref'),
    pytest.param(
        DEFECT_TEST_RETIREMENTS_METADATA_KEY,
        [_json(_retirement(ruling=_ruling(RulingSubject.RETIREMENT)))],
        0,
        id='ruled-retirement',
    ),
    pytest.param(
        DEFECT_TEST_RETIREMENTS_METADATA_KEY,
        [_with(_json(_retirement()), reason={'reason': 'bogus'})],
        0,
        id='unknown-retirement-reason',
    ),
    pytest.param(DEFECT_TEST_RETIREMENTS_METADATA_KEY, 'x', None, id='retirements-not-a-list'),
    pytest.param(
        DEFECT_TEST_RETIREMENTS_METADATA_KEY,
        [_json(_retirement()), _json(_retirement())],
        None,
        id='repeated-retirement-ref',
    ),
]


@pytest.mark.parametrize(('key', 'value', 'index'), _CREATION_REFUSALS)
def test_the_creation_floor_refuses_naming_key_index_and_allowed_values(
    key: str, value: Any, index: int | None
) -> None:
    with pytest.raises(DefectCreationRefused) as excinfo:
        require_creation_floor({'files': ['x.py'], key: value})
    wire = excinfo.value.to_error_dict()
    assert wire['success'] is False
    assert wire['error'] == 'defect_creation_refused'
    assert wire['key'] == key
    assert wire['index'] == index
    assert isinstance(wire['detail'], str) and wire['detail']
    assert wire['allowed'] == _ALLOWED[key]


def test_a_repeated_ref_at_creation_is_named() -> None:
    with pytest.raises(DefectCreationRefused) as excinfo:
        require_creation_floor({DEFECTS_METADATA_KEY: [_FRESH, _FRESH]})
    assert _REF in excinfo.value.to_error_dict()['detail']


# --- (j) wire parsing -----------------------------------------------------------


def test_parse_declare_wire_returns_the_event() -> None:
    event = Declare(declaration=_declaration())
    assert parse_declare_wire(_json(event)) == event


def test_parse_outcome_wire_returns_the_event_for_its_ref() -> None:
    event = Proof(proof=_proof())
    assert parse_outcome_wire(_json(event), ref=_REF) == event


@pytest.mark.parametrize(
    'event',
    [
        pytest.param(Proof(proof=_proof()), id='proof'),
        pytest.param(Rule(ref=_REF, ruling=_ruling(RulingSubject.CLOSURE)), id='rule'),
    ],
)
def test_parse_declare_wire_refuses_an_outcome_event_naming_the_declare_tags(event: Any) -> None:
    with pytest.raises(DefectEventInvalid) as excinfo:
        parse_declare_wire(_json(event))
    wire = excinfo.value.to_error_dict()
    assert wire['error'] == 'defect_event_invalid'
    assert wire['expected_events'] == _event_tags(DeclareEvent)
    for tag in _event_tags(DeclareEvent):
        assert tag in wire['detail']


def test_parse_outcome_wire_refuses_a_declare_event_naming_the_outcome_tags() -> None:
    with pytest.raises(DefectEventInvalid) as excinfo:
        parse_outcome_wire(_json(Declare(declaration=_declaration())), ref=_REF)
    wire = excinfo.value.to_error_dict()
    assert wire['error'] == 'defect_event_invalid'
    assert wire['expected_events'] == _event_tags(OutcomeEvent)
    for tag in _event_tags(OutcomeEvent):
        assert tag in wire['detail']


def test_parse_outcome_wire_refuses_an_event_for_another_ref() -> None:
    with pytest.raises(DefectRefMismatch) as excinfo:
        parse_outcome_wire(_json(Proof(proof=_proof())), ref=_OTHER_REF)
    assert isinstance(excinfo.value, DefectEventInvalid)
    wire = excinfo.value.to_error_dict()
    assert wire['error'] == 'defect_ref_mismatch'
    assert wire['ref'] == _OTHER_REF
    assert wire['event_ref'] == _REF
