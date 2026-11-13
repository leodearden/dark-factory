"""shared.cancel_dispositions: the cancel-with-children contract.

Contract: plans/cancel-with-children-prd.md §4.2. Everything goes through the public names.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping, MutableMapping
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from test_pure_stdlib_leaves import run_python_child

from shared.cancel_dispositions import (
    CALLER_ACTIONS,
    CANCEL_CHILDREN_SKILL,
    CHILD_CANCELLING_ACTIONS,
    DECISION_TABLE,
    ENTRY_ONLY_ACTIONS,
    GET_CANCEL_CHILDREN_TOOL,
    MISSING_FILES_CAP,
    PAYLOAD_ID_CAP,
    CancelAction,
    CancelChildrenInvalid,
    CancelChildrenInvalidReason,
    CancelChildrenRefusalCode,
    CancelChildrenView,
    CancelDispositionEntry,
    ChildDisposition,
    ChildDispositionVariant,
    ChildKey,
    ChildRelation,
    ChildRow,
    CoCancelledEntry,
    DecisionRow,
    DispositionSource,
    ExternalChild,
    FoldDisposition,
    FoldEntry,
    InvalidEntries,
    KeepDisposition,
    KeepEntry,
    MootDisposition,
    MootEntry,
    RepointDisposition,
    RepointEntry,
    capped,
    check_targets,
    children_to_json,
    co_cancelled_entry,
    entries_to_json,
    entry_for_parent,
    parse_children,
    parse_entries,
    parse_superseded_by,
    qualified_task_ref,
    render_cancel_children_rules,
    render_decision_table_markdown,
    rows_to_json,
    sibling_lookup_values,
    stamp_entry,
    with_entry,
)
from shared.task_metadata import parse_metadata
from shared.task_statuses import TaskStatus

PARENT = 5869
PROJECT = 'dark_factory'

# --- the caller disposition shape (§4.2) -------------------------------------


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [
        pytest.param({'action': 'keep'}, KeepDisposition(), id='keep'),
        pytest.param(
            {'action': 'keep', 'reason': 'stands on main'},
            KeepDisposition(reason='stands on main'),
            id='keep-with-reason',
        ),
        pytest.param(
            {'action': 'repoint', 'onto': [8032]}, RepointDisposition(onto=(8032,)), id='repoint'
        ),
        pytest.param({'action': 'fold', 'into': 8032}, FoldDisposition(into=8032), id='fold'),
        pytest.param({'action': 'moot'}, MootDisposition(), id='moot'),
    ],
)
def test_each_documented_example_parses_to_its_variant(raw, expected):
    parsed = ChildDisposition.model_validate(raw).root

    assert type(parsed) is type(expected)
    assert parsed == expected


def test_repoint_onto_is_held_as_a_tuple():
    parsed = ChildDisposition.model_validate({'action': 'repoint', 'onto': [8032]}).root

    assert isinstance(parsed, RepointDisposition)
    assert parsed.onto == (8032,)


@pytest.mark.parametrize(
    ('disposition', 'field', 'value'),
    [
        pytest.param(KeepDisposition(), 'reason', 'changed', id='keep'),
        pytest.param(RepointDisposition(onto=(1,)), 'onto', (2,), id='repoint'),
        pytest.param(FoldDisposition(into=1), 'into', 2, id='fold'),
        pytest.param(MootDisposition(), 'reason', 'changed', id='moot'),
    ],
)
def test_every_variant_is_frozen(disposition, field, value):
    with pytest.raises(ValidationError):
        setattr(disposition, field, value)


def test_the_action_sets_partition_the_action_enum():
    assert {
        CancelAction.KEEP,
        CancelAction.REPOINT,
        CancelAction.FOLD,
        CancelAction.MOOT,
    } == CALLER_ACTIONS
    assert {CancelAction.CO_CANCELLED} == ENTRY_ONLY_ACTIONS
    assert {CancelAction.FOLD, CancelAction.MOOT} == CHILD_CANCELLING_ACTIONS
    assert set(CancelAction) == CALLER_ACTIONS | ENTRY_ONLY_ACTIONS
    assert not CALLER_ACTIONS & ENTRY_ONLY_ACTIONS
    assert CHILD_CANCELLING_ACTIONS <= CALLER_ACTIONS


@pytest.mark.parametrize(
    'raw',
    [
        pytest.param({'action': 'co_cancelled'}, id='entry-only-action'),
        pytest.param({'action': 'keep', 'onto': [1]}, id='extra-field'),
        pytest.param({'action': 'repoint', 'onto': []}, id='empty-onto'),
        pytest.param({'action': 'repoint', 'onto': [3, 3]}, id='duplicate-onto'),
        pytest.param({'action': 'fold', 'into': True}, id='bool-is-not-an-id'),
        pytest.param({'action': 'moot', 'reason': 'x' * 501}, id='reason-too-long'),
    ],
)
def test_child_disposition_rejects_malformed_input(raw):
    with pytest.raises(ValidationError):
        ChildDisposition.model_validate(raw)


# --- the children map and superseded_by parsers (§4.2) ------------------------

Reason = CancelChildrenInvalidReason


def _parse(raw: object) -> Mapping[ChildKey, ChildDispositionVariant]:
    return parse_children(raw, parent_id=PARENT, parent_project_id=PROJECT)


def test_parse_children_keys_local_ids_as_ints_and_external_dependents_as_qualified_refs():
    parsed = _parse(
        {'6905': {'action': 'keep'}, 'reify:44': {'action': 'repoint', 'onto': [8032]}}
    )

    assert dict(parsed) == {
        6905: KeepDisposition(),
        ExternalChild(project_id='reify', task_id=44): RepointDisposition(onto=(8032,)),
    }


def test_parse_children_returns_an_immutable_mapping():
    parsed = _parse({'6905': {'action': 'keep'}})

    assert isinstance(parsed, Mapping)
    assert not isinstance(parsed, MutableMapping)


def test_an_external_child_key_is_the_qualified_task_ref():
    child = ExternalChild(project_id='reify', task_id=44)

    assert child.key == 'reify:44' == qualified_task_ref('reify', 44)


def test_a_qualified_task_ref_spells_the_project_id_canonically():
    assert qualified_task_ref('Dark-Factory', 12) == 'dark_factory:12'


@pytest.mark.parametrize(
    ('project_id', 'task_id'),
    [
        pytest.param('Reify', 44, id='non-canonical-project'),
        pytest.param('', 44, id='empty-project'),
        pytest.param('re:ify', 44, id='project-with-a-colon'),
        pytest.param('reify', 0, id='task-id-zero'),
        pytest.param('reify', True, id='bool-task-id'),
    ],
)
def test_an_external_child_is_refused_unless_its_key_parses_back_to_it(project_id, task_id):
    with pytest.raises(ValueError, match='canonical project id'):
        ExternalChild(project_id=project_id, task_id=task_id)


def test_children_round_trip_through_their_wire_form():
    parsed = _parse(
        {
            '6905': {'action': 'keep', 'reason': 'stands on main'},
            '7007': {'action': 'fold', 'into': 8032},
            '7008': {'action': 'moot'},
            'reify:44': {'action': 'repoint', 'onto': [8032, 8033]},
        }
    )

    wire = children_to_json(parsed)

    assert json.loads(json.dumps(wire)) == wire
    assert _parse(wire) == parsed


@pytest.mark.parametrize(
    ('raw', 'reason', 'field', 'value'),
    [
        pytest.param(
            {'7007': {'action': 'repoint', 'onto': [PARENT]}},
            Reason.TARGET_IS_PARENT,
            'children.7007.onto',
            PARENT,
            id='repoint-onto-the-parent',
        ),
        pytest.param(
            {'7007': {'action': 'fold', 'into': 7007}},
            Reason.TARGET_IS_CHILD,
            'children.7007.into',
            7007,
            id='fold-into-itself',
        ),
        pytest.param(
            {'7007': {'action': 'repoint', 'onto': [7007]}},
            Reason.TARGET_IS_CHILD,
            'children.7007.onto',
            7007,
            id='repoint-onto-itself',
        ),
        pytest.param(
            {'7007': {'action': 'fold', 'into': PARENT}},
            Reason.TARGET_IS_PARENT,
            'children.7007.into',
            PARENT,
            id='fold-into-the-parent',
        ),
        pytest.param(
            {str(PARENT): {'action': 'keep'}},
            Reason.CHILD_IS_PARENT,
            f'children.{PARENT}',
            str(PARENT),
            id='parent-as-its-own-child',
        ),
        pytest.param(
            {'abc': {'action': 'keep'}},
            Reason.INVALID_CHILD_KEY,
            'children.abc',
            'abc',
            id='key-not-an-id',
        ),
        pytest.param(
            {'0': {'action': 'keep'}},
            Reason.INVALID_CHILD_KEY,
            'children.0',
            '0',
            id='key-zero',
        ),
        pytest.param(
            {f'{PROJECT}:12': {'action': 'keep'}},
            Reason.INVALID_CHILD_KEY,
            f'children.{PROJECT}:12',
            f'{PROJECT}:12',
            id='own-project-qualified',
        ),
        pytest.param(
            {'Dark-Factory:12': {'action': 'repoint', 'onto': [8032]}},
            Reason.INVALID_CHILD_KEY,
            'children.Dark-Factory:12',
            'Dark-Factory:12',
            id='own-project-qualified-in-another-spelling',
        ),
        pytest.param(
            {'Reify:44': {'action': 'repoint', 'onto': [8032]}},
            Reason.INVALID_CHILD_KEY,
            'children.Reify:44',
            'Reify:44',
            id='external-project-not-canonical',
        ),
        pytest.param(
            {7007: {'action': 'keep'}},
            Reason.INVALID_CHILD_KEY,
            'children.7007',
            7007,
            id='key-not-a-string',
        ),
        pytest.param(
            {'reify:44': {'action': 'moot'}},
            Reason.EXTERNAL_NOT_REPOINT,
            'children.reify:44.action',
            'moot',
            id='external-moot',
        ),
        pytest.param(
            {'1': {'action': 'co_cancelled'}},
            Reason.ENTRY_ONLY_ACTION,
            'children.1.action',
            'co_cancelled',
            id='entry-only-action',
        ),
        pytest.param(
            {'1': {'action': 'nope'}},
            Reason.UNKNOWN_ACTION,
            'children.1.action',
            'nope',
            id='unknown-action',
        ),
        pytest.param(
            {'1': {'action': 'keep', 'into': 2}},
            Reason.FIELD_NOT_ALLOWED,
            'children.1.into',
            2,
            id='field-not-allowed',
        ),
        pytest.param(
            {'1': {'action': 'fold'}},
            Reason.MISSING_FIELD,
            'children.1.into',
            None,
            id='missing-into',
        ),
        pytest.param(
            {'1': {'reason': 'no action'}},
            Reason.MISSING_FIELD,
            'children.1.action',
            None,
            id='missing-action',
        ),
        pytest.param(
            {'7007': {'action': 'repoint', 'onto': [8032, 0]}},
            Reason.INVALID_VALUE,
            'children.7007.onto.1',
            0,
            id='one-onto-element-not-an-id',
        ),
        pytest.param(
            {'1': {'action': 'moot', 'reason': 'x' * 501}},
            Reason.TEXT_TOO_LONG,
            'children.1.reason',
            'x' * 501,
            id='reason-too-long',
        ),
        pytest.param(
            {'1': {'action': 'repoint', 'onto': [3, 3]}},
            Reason.DUPLICATE_TARGET,
            'children.1.onto',
            [3],
            id='duplicate-onto',
        ),
        pytest.param(
            {'1': 'keep'},
            Reason.INVALID_VALUE,
            'children.1',
            'keep',
            id='disposition-not-a-mapping',
        ),
        pytest.param(
            [{'action': 'keep'}],
            Reason.INVALID_VALUE,
            'children',
            [{'action': 'keep'}],
            id='children-not-a-mapping',
        ),
    ],
)
def test_parse_children_names_the_reason_field_and_value_of_a_refusal(raw, reason, field, value):
    with pytest.raises(CancelChildrenInvalid) as caught:
        _parse(raw)

    assert (caught.value.reason, caught.value.field, caught.value.value) == (reason, field, value)


def test_parse_superseded_by_returns_the_ids_in_order():
    assert parse_superseded_by([8032, 8033], parent_id=PARENT) == (8032, 8033)


@pytest.mark.parametrize(
    ('raw', 'reason', 'field', 'value'),
    [
        pytest.param([], Reason.INVALID_VALUE, 'superseded_by', [], id='empty'),
        pytest.param('8032', Reason.INVALID_VALUE, 'superseded_by', '8032', id='not-a-list'),
        pytest.param(
            [8032, True], Reason.INVALID_VALUE, 'superseded_by.1', True, id='bool-is-not-an-id'
        ),
        pytest.param(
            [8032, 0], Reason.INVALID_VALUE, 'superseded_by.1', 0, id='one-element-not-an-id'
        ),
        pytest.param(
            [8032, 8032], Reason.DUPLICATE_TARGET, 'superseded_by', [8032], id='duplicate'
        ),
        pytest.param(
            [*range(1, 200), 5],
            Reason.DUPLICATE_TARGET,
            'superseded_by',
            [5],
            id='duplicate-in-a-long-list-names-only-the-repeat',
        ),
        pytest.param(
            [8032, PARENT], Reason.TARGET_IS_PARENT, 'superseded_by', PARENT, id='names-the-parent'
        ),
    ],
)
def test_parse_superseded_by_names_the_reason_field_and_value_of_a_refusal(
    raw, reason, field, value
):
    with pytest.raises(CancelChildrenInvalid) as caught:
        parse_superseded_by(raw, parent_id=PARENT)

    assert (caught.value.reason, caught.value.field, caught.value.value) == (reason, field, value)


def test_a_duplicate_refusal_names_at_most_the_payload_cap_of_repeated_ids():
    with pytest.raises(CancelChildrenInvalid) as caught:
        parse_superseded_by([*range(1, 80), *range(1, 80)], parent_id=PARENT)

    assert caught.value.value == list(range(1, PAYLOAD_ID_CAP + 1))


def test_the_refusal_codes_are_the_house_slugs():
    assert {code.value for code in CancelChildrenRefusalCode} == {
        'cancel_children_undispositioned',
        'cancel_children_invalid',
        'cancel_children_unknown',
        'cancel_children_nested',
        'cancel_children_execution_failed',
        'cancel_children_parent_terminal',
        'cancel_children_unverified',
    }


@pytest.mark.parametrize(
    ('code', 'error_type'),
    [
        (CancelChildrenRefusalCode.UNDISPOSITIONED, 'CancelChildrenUndispositioned'),
        (CancelChildrenRefusalCode.EXECUTION_FAILED, 'CancelChildrenExecutionFailed'),
        (CancelChildrenRefusalCode.PARENT_TERMINAL, 'CancelChildrenParentTerminal'),
    ],
)
def test_a_refusal_code_error_type_is_its_slug_in_camel_case(code, error_type):
    assert code.error_type == error_type


def test_the_invalid_refusal_is_the_house_gate_shape():
    refusal = CancelChildrenInvalid(Reason.TARGET_CANCELLED, 'children.6905.onto', 8032)

    payload = refusal.to_error_dict(task_id='5869')

    assert payload == {
        'success': False,
        'error': 'cancel_children_invalid',
        'error_type': 'CancelChildrenInvalid',
        'task_id': '5869',
        'reason_code': 'target_cancelled',
        'field': 'children.6905.onto',
        'value': 8032,
        'hint': payload['hint'],
    }
    assert payload['hint']


def test_the_invalid_refusal_repr_s_a_value_json_cannot_carry():
    payload = CancelChildrenInvalid(Reason.INVALID_VALUE, 'children', {1, 2}).to_error_dict(
        task_id='5869'
    )

    assert payload['value'] == repr({1, 2})
    assert json.dumps(payload)


# --- the referential and cycle check over injected facts (§4.2) ---------------

SUPERSEDER = 8032
OTHER = 8033
DONE_SUPERSEDER = 8040
CANCELLED = 8050
MISSING = 8099

_STATUSES = {
    SUPERSEDER: TaskStatus.PENDING,
    OTHER: TaskStatus.IN_PROGRESS,
    DONE_SUPERSEDER: TaskStatus.DONE,
    CANCELLED: TaskStatus.CANCELLED,
    MISSING: None,
    7007: TaskStatus.PENDING,
    7008: TaskStatus.PENDING,
    7009: TaskStatus.PENDING,
}


def _check(
    children: Mapping[ChildKey, ChildDispositionVariant],
    superseded_by: tuple[int, ...] = (),
    depends_on: Mapping[int, list[int]] | None = None,
    co_cancelled: tuple[int, ...] = (),
) -> None:
    check_targets(
        children,
        superseded_by,
        parent_id=PARENT,
        co_cancelled=co_cancelled,
        target_status=_STATUSES,
        depends_on=depends_on or {},
    )


def _refusal_of(*args, **kwargs) -> tuple[CancelChildrenInvalidReason, str, object]:
    with pytest.raises(CancelChildrenInvalid) as caught:
        _check(*args, **kwargs)
    return caught.value.reason, caught.value.field, caught.value.value


def test_a_map_naming_live_and_done_targets_passes():
    children: dict[ChildKey, ChildDispositionVariant] = {
        6905: RepointDisposition(onto=(SUPERSEDER,)),
        7007: FoldDisposition(into=DONE_SUPERSEDER),
        7008: KeepDisposition(),
        7009: MootDisposition(),
    }

    assert _check(children, superseded_by=(DONE_SUPERSEDER,)) is None


def test_repoint_onto_a_cancelled_task_is_refused():
    refusal = _refusal_of({6905: RepointDisposition(onto=(SUPERSEDER, CANCELLED))})

    assert refusal == (Reason.TARGET_CANCELLED, 'children.6905.onto', CANCELLED)


def test_fold_into_a_missing_task_is_refused():
    refusal = _refusal_of({7007: FoldDisposition(into=MISSING)})

    assert refusal == (Reason.TARGET_NOT_FOUND, 'children.7007.into', MISSING)


def test_a_target_absent_from_the_facts_reads_as_missing():
    refusal = _refusal_of({7007: FoldDisposition(into=12345)})

    assert refusal == (Reason.TARGET_NOT_FOUND, 'children.7007.into', 12345)


def test_superseded_by_naming_a_cancelled_task_is_refused():
    refusal = _refusal_of({}, superseded_by=(SUPERSEDER, CANCELLED))

    assert refusal == (Reason.TARGET_CANCELLED, 'superseded_by', CANCELLED)


def test_superseded_by_naming_a_missing_task_is_refused():
    refusal = _refusal_of({}, superseded_by=(SUPERSEDER, MISSING))

    assert refusal == (Reason.TARGET_NOT_FOUND, 'superseded_by', MISSING)


def test_a_repoint_that_would_close_a_dependency_cycle_is_refused():
    refusal = _refusal_of(
        {7007: RepointDisposition(onto=(SUPERSEDER,))},
        depends_on={SUPERSEDER: [OTHER], OTHER: [7007]},
    )

    assert refusal == (Reason.REPOINT_CYCLE, 'children.7007.onto', SUPERSEDER)


def test_an_external_child_cannot_close_a_local_cycle():
    child = ExternalChild(project_id='reify', task_id=7007)

    _check(
        {child: RepointDisposition(onto=(SUPERSEDER,))},
        depends_on={SUPERSEDER: [OTHER], OTHER: [7007]},
    )


def test_a_cyclic_edge_map_that_never_reaches_the_child_terminates_and_passes():
    _check(
        {7007: RepointDisposition(onto=(SUPERSEDER,))},
        depends_on={SUPERSEDER: [SUPERSEDER, OTHER], OTHER: [SUPERSEDER]},
    )


def test_a_mutual_repoint_within_the_map_is_refused():
    refusal = _refusal_of(
        {7007: RepointDisposition(onto=(7008,)), 7008: RepointDisposition(onto=(7007,))},
        depends_on={7007: [PARENT], 7008: [PARENT]},
    )

    assert refusal == (Reason.REPOINT_CYCLE, 'children.7007.onto', 7008)


def test_a_three_child_repoint_chain_within_the_map_is_refused():
    refusal = _refusal_of(
        {
            7007: RepointDisposition(onto=(7008,)),
            7008: RepointDisposition(onto=(7009,)),
            7009: RepointDisposition(onto=(7007,)),
        },
        depends_on={7007: [PARENT], 7008: [PARENT], 7009: [PARENT]},
    )

    assert refusal == (Reason.REPOINT_CYCLE, 'children.7007.onto', 7008)


def test_a_cycle_joining_a_map_repoint_to_a_stored_edge_is_refused():
    refusal = _refusal_of(
        {7007: RepointDisposition(onto=(SUPERSEDER,)), 7008: RepointDisposition(onto=(7007,))},
        depends_on={SUPERSEDER: [7008], 7007: [PARENT], 7008: [PARENT]},
    )

    assert refusal == (Reason.REPOINT_CYCLE, 'children.7007.onto', SUPERSEDER)


def test_a_repointed_childs_dropped_parent_edge_closes_no_cycle():
    _check(
        {7007: RepointDisposition(onto=(SUPERSEDER,)), 7008: RepointDisposition(onto=(OTHER,))},
        depends_on={PARENT: [7007], SUPERSEDER: [7008], 7008: [PARENT]},
    )


def test_a_path_through_the_cancelled_parent_closes_no_cycle():
    _check(
        {7007: RepointDisposition(onto=(SUPERSEDER,)), 7008: KeepDisposition()},
        depends_on={PARENT: [7007], SUPERSEDER: [7008], 7008: [PARENT]},
    )


@pytest.mark.parametrize(
    'cancelling',
    [
        pytest.param(MootDisposition(), id='moot'),
        pytest.param(FoldDisposition(into=OTHER), id='fold'),
    ],
)
def test_a_path_through_a_sibling_the_map_cancels_closes_no_cycle(
    cancelling: ChildDispositionVariant,
):
    _check(
        {7007: RepointDisposition(onto=(SUPERSEDER,)), 7008: cancelling},
        depends_on={SUPERSEDER: [7008], 7008: [7007]},
    )


def test_repoint_onto_a_sibling_the_map_cancels_is_refused():
    refusal = _refusal_of({7007: RepointDisposition(onto=(7008,)), 7008: MootDisposition()})

    assert refusal == (Reason.TARGET_CANCELLED, 'children.7007.onto', 7008)


def test_fold_into_a_sibling_the_map_cancels_is_refused():
    refusal = _refusal_of(
        {7007: FoldDisposition(into=7008), 7008: FoldDisposition(into=SUPERSEDER)}
    )

    assert refusal == (Reason.TARGET_CANCELLED, 'children.7007.into', 7008)


def test_superseded_by_naming_a_child_the_map_cancels_is_refused():
    refusal = _refusal_of({7008: MootDisposition()}, superseded_by=(7008,))

    assert refusal == (Reason.TARGET_CANCELLED, 'superseded_by', 7008)


@pytest.mark.parametrize(
    ('children', 'superseded_by', 'field'),
    [
        pytest.param(
            {7007: RepointDisposition(onto=(7008,))}, (), 'children.7007.onto', id='repoint-onto'
        ),
        pytest.param({7007: FoldDisposition(into=7008)}, (), 'children.7007.into', id='fold-into'),
        pytest.param({}, (7008,), 'superseded_by', id='superseded-by'),
    ],
)
def test_a_target_the_same_act_co_cancels_is_refused(
    children: Mapping[ChildKey, ChildDispositionVariant],
    superseded_by: tuple[int, ...],
    field: str,
):
    refusal = _refusal_of(children, superseded_by=superseded_by, co_cancelled=(7008,))

    assert refusal == (Reason.TARGET_CANCELLED, field, 7008)


def test_a_path_through_a_co_cancelled_task_closes_no_cycle():
    _check(
        {7007: RepointDisposition(onto=(SUPERSEDER,))},
        depends_on={SUPERSEDER: [7008], 7008: [7007]},
        co_cancelled=(7008,),
    )


@pytest.mark.parametrize(
    'disposition',
    [
        pytest.param(KeepDisposition(), id='keep'),
        pytest.param(RepointDisposition(onto=(SUPERSEDER,)), id='repoint'),
        pytest.param(FoldDisposition(into=SUPERSEDER), id='fold'),
        pytest.param(MootDisposition(), id='moot'),
    ],
)
def test_a_child_the_same_act_co_cancels_cannot_also_be_dispositioned(
    disposition: ChildDispositionVariant,
):
    refusal = _refusal_of({7007: disposition}, co_cancelled=(7007,))

    assert refusal == (Reason.CO_CANCELLED_CHILD, 'children.7007', 7007)


def test_a_co_cancelled_child_is_refused_before_any_cycle_its_repoint_would_close():
    refusal = _refusal_of(
        {7007: RepointDisposition(onto=(SUPERSEDER,))},
        depends_on={SUPERSEDER: [7007]},
        co_cancelled=(7007,),
    )

    assert refusal == (Reason.CO_CANCELLED_CHILD, 'children.7007', 7007)


def test_of_two_co_cancelled_children_the_lowest_id_is_reported():
    refusal = _refusal_of(
        {7008: KeepDisposition(), 7007: MootDisposition()}, co_cancelled=(7008, 7007)
    )

    assert refusal[1] == 'children.7007'


@pytest.mark.parametrize(
    'children',
    [
        pytest.param(
            {7007: FoldDisposition(into=7008), 7008: KeepDisposition()}, id='fold-into-kept'
        ),
        pytest.param(
            {7007: RepointDisposition(onto=(7009,)), 7009: RepointDisposition(onto=(SUPERSEDER,))},
            id='repoint-onto-repointed',
        ),
    ],
)
def test_a_kept_or_repointed_sibling_remains_a_legal_target(
    children: Mapping[ChildKey, ChildDispositionVariant],
):
    _check(children)


def test_of_two_failing_children_the_lowest_local_id_is_reported():
    children = {
        7008: RepointDisposition(onto=(CANCELLED,)),
        ExternalChild(project_id='reify', task_id=1): RepointDisposition(onto=(MISSING,)),
        7007: FoldDisposition(into=MISSING),
    }

    assert _refusal_of(children)[1] == 'children.7007.into'


def test_external_children_are_checked_after_local_ones_in_key_order():
    children = {
        ExternalChild(project_id='zeta', task_id=1): RepointDisposition(onto=(MISSING,)),
        ExternalChild(project_id='reify', task_id=2): RepointDisposition(onto=(CANCELLED,)),
    }

    assert _refusal_of(children)[1] == 'children.reify:2.onto'


# --- the stamped child record (§4.8) -----------------------------------------

NOW = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)
P = qualified_task_ref(PROJECT, PARENT)
_STAMP = {'stamped_at': NOW, 'stamped_by': 'claude-x', 'source': DispositionSource.CANCELLER}


@pytest.mark.parametrize(
    ('disposition', 'entry_type'),
    [
        pytest.param(KeepDisposition(reason='stands on main'), KeepEntry, id='keep'),
        pytest.param(RepointDisposition(onto=(8032,)), RepointEntry, id='repoint'),
        pytest.param(FoldDisposition(into=8032), FoldEntry, id='fold'),
        pytest.param(MootDisposition(), MootEntry, id='moot'),
    ],
)
def test_stamp_entry_carries_the_disposition_and_the_stamp(disposition, entry_type):
    entry = stamp_entry(disposition, parent=P, **_STAMP)

    assert type(entry) is entry_type
    assert entry.model_dump(exclude={'parent', 'stamped_at', 'stamped_by', 'source'}) == (
        disposition.model_dump()
    )
    assert (entry.parent, entry.stamped_at, entry.stamped_by, entry.source) == (
        P,
        NOW,
        'claude-x',
        DispositionSource.CANCELLER,
    )


@pytest.mark.parametrize(
    'not_a_variant',
    [
        pytest.param(ChildDisposition(KeepDisposition()), id='the-root-model-wrapper'),
        pytest.param({'action': 'keep'}, id='the-wire-form'),
    ],
)
def test_stamp_entry_refuses_anything_but_a_disposition_variant(not_a_variant):
    with pytest.raises(AssertionError):
        stamp_entry(not_a_variant, parent=P, **_STAMP)


def test_a_repoint_entry_keeps_its_targets():
    entry = stamp_entry(RepointDisposition(onto=(8032,)), parent=P, **_STAMP)

    assert isinstance(entry, RepointEntry)
    assert (entry.action, entry.onto) == (CancelAction.REPOINT, (8032,))


def test_a_co_cancelled_entry_names_no_target():
    entry = co_cancelled_entry(parent=P, **_STAMP)

    assert isinstance(entry, CoCancelledEntry)
    assert entry.action == CancelAction.CO_CANCELLED
    assert not {'onto', 'into'} & set(entry.model_dump())


def _entry_json(**overrides: object) -> dict[str, object]:
    return {
        'parent': P,
        'action': 'keep',
        'stamped_at': NOW.isoformat(),
        'stamped_by': None,
        'source': 'canceller',
        **overrides,
    }


def test_a_well_formed_entry_validates():
    entry = CancelDispositionEntry.model_validate(_entry_json()).root

    assert entry == KeepEntry(parent=P, stamped_at=NOW, source=DispositionSource.CANCELLER)


@pytest.mark.parametrize(
    'overrides',
    [
        pytest.param({'parent': str(PARENT)}, id='bare-parent-id'),
        pytest.param({'parent': f'Dark-Factory:{PARENT}'}, id='non-canonical-parent-project'),
        pytest.param({'stamped_at': '2026-11-03T12:00:00'}, id='naive-stamped-at'),
        pytest.param({'source': 'other'}, id='unknown-source'),
        pytest.param({'onto': [8032]}, id='onto-on-keep'),
    ],
)
def test_cancel_disposition_entry_rejects_malformed_input(overrides):
    with pytest.raises(ValidationError):
        CancelDispositionEntry.model_validate(_entry_json(**overrides))


A = qualified_task_ref(PROJECT, 100)
B = qualified_task_ref('reify', 200)
C = qualified_task_ref(PROJECT, 300)


def _keep_for(parent: str, reason: str | None = None) -> KeepEntry:
    return KeepEntry(parent=parent, reason=reason, stamped_at=NOW, source=DispositionSource.BACKSTOP)


def test_with_entry_replaces_the_entry_for_the_same_parent_in_place():
    entries = [_keep_for(A, 'first'), _keep_for(B)]
    original = list(entries)
    replacement = stamp_entry(RepointDisposition(onto=(8032,)), parent=A, **_STAMP)

    updated = with_entry(entries, replacement)

    assert updated == (replacement, _keep_for(B))
    assert entries == original


def test_with_entry_appends_an_entry_for_a_new_parent():
    entries = (_keep_for(A), _keep_for(B))

    assert with_entry(entries, _keep_for(C)) == (_keep_for(A), _keep_for(B), _keep_for(C))


def test_with_entry_leaves_its_parent_exactly_one_entry():
    entries = (_keep_for(A, 'first'), _keep_for(B), _keep_for(A, 'second'), _keep_for(C))

    assert with_entry(entries, _keep_for(A, 'new')) == (
        _keep_for(A, 'new'),
        _keep_for(B),
        _keep_for(C),
    )


def test_entry_for_parent_finds_the_parent_s_entry_or_none():
    entries = (_keep_for(A, 'a'), _keep_for(B, 'b'))

    assert entry_for_parent(entries, A) == _keep_for(A, 'a')
    assert entry_for_parent(entries, C) is None


@pytest.mark.parametrize(
    'parent',
    [
        pytest.param(f'Dark-Factory:{PARENT}', id='non-canonical-project'),
        pytest.param(str(PARENT), id='bare-id'),
    ],
)
def test_entry_for_parent_refuses_a_parent_qualified_task_ref_would_not_spell(parent: str):
    entries = (_keep_for(qualified_task_ref(PROJECT, PARENT)),)

    with pytest.raises(ValidationError):
        entry_for_parent(entries, parent)


_ENTRIES = (
    _keep_for(A, 'stands on main'),
    stamp_entry(RepointDisposition(onto=(8032, 8033)), parent=B, **_STAMP),
    stamp_entry(FoldDisposition(into=8032), parent=C, **_STAMP),
    co_cancelled_entry(parent=qualified_task_ref(PROJECT, 400), **_STAMP),
)


def test_entries_round_trip_through_their_json_form():
    wire = entries_to_json(_ENTRIES)

    assert json.loads(json.dumps(wire)) == wire
    assert parse_entries(wire) == _ENTRIES


def test_parse_entries_reads_an_absent_slice_as_no_entries():
    assert parse_entries(None) == ()


def test_parse_entries_accepts_the_typed_entries_parse_metadata_hands_back():
    typed = [CancelDispositionEntry(_ENTRIES[0]), _ENTRIES[1]]

    assert parse_entries(typed) == _ENTRIES[:2]


@pytest.mark.parametrize(
    ('raw', 'named'),
    [
        pytest.param([_entry_json(parent='5869')], 'parent', id='bare-parent'),
        pytest.param([_entry_json(action='repoint')], 'onto', id='repoint-without-onto'),
        pytest.param({'parent': P}, 'list', id='not-a-list'),
        pytest.param(
            [_entry_json(), _entry_json(action='moot')], f'more than one entry for parent {P}',
            id='two-entries-for-one-parent',
        ),
    ],
)
def test_parse_entries_reports_a_malformed_slice(raw, named):
    parsed = parse_entries(raw)

    assert isinstance(parsed, InvalidEntries)
    assert named in parsed.detail


# --- the read view (§4.3) and the payload cap (§4.4) --------------------------


def _row(**overrides: object) -> ChildRow:
    fields: dict[str, object] = {
        'id': 6905,
        'project_id': PROJECT,
        'title': 'repoint me',
        'status': TaskStatus.PENDING,
        'relation': ChildRelation.DEPENDENT,
        'declared_files_total': 3,
        'declared_files_missing_from_main': ('a.py',),
        'declared_files_missing_from_main_total': 2,
        'depends_on_superseder': False,
        'named_superseder': False,
        'has_live_children': False,
        'escalation_id': 'esc-6905-1',
        'existing_disposition': stamp_entry(
            RepointDisposition(onto=(SUPERSEDER,)), parent=P, **_STAMP
        ),
        **overrides,
    }
    return ChildRow.model_validate(fields)


def test_a_row_carrying_every_documented_fact_validates():
    row = _row()

    assert (row.status, row.relation) == (TaskStatus.PENDING, ChildRelation.DEPENDENT)
    assert isinstance(row.existing_disposition, RepointEntry)


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        pytest.param('title', 'changed', id='title'),
        pytest.param('status', TaskStatus.DONE, id='status'),
    ],
)
def test_a_row_is_frozen(field: str, value: object):
    with pytest.raises(ValidationError):
        setattr(_row(), field, value)


@pytest.mark.parametrize(
    'overrides',
    [
        pytest.param(
            {
                'declared_files_total': 30,
                'declared_files_missing_from_main': tuple(f'f{n}.py' for n in range(21)),
                'declared_files_missing_from_main_total': 21,
            },
            id='more-missing-files-than-the-cap',
        ),
        pytest.param(
            {
                'declared_files_missing_from_main': ('a.py', 'b.py'),
                'declared_files_missing_from_main_total': 1,
            },
            id='missing-total-below-the-listed-count',
        ),
        pytest.param(
            {'declared_files_total': 1, 'declared_files_missing_from_main_total': 2},
            id='missing-total-above-the-declared-total',
        ),
        pytest.param({'relation': 'sibling'}, id='unknown-relation'),
        pytest.param({'project_id': 'Dark-Factory'}, id='non-canonical-project'),
        pytest.param({'project_id': ''}, id='empty-project'),
    ],
)
def test_a_row_with_inconsistent_facts_is_rejected(overrides):
    with pytest.raises(ValidationError):
        _row(**overrides)


def test_the_missing_files_cap_is_twenty():
    assert MISSING_FILES_CAP == 20
    files = tuple(f'f{n}.py' for n in range(MISSING_FILES_CAP))

    row = _row(
        declared_files_total=40,
        declared_files_missing_from_main=files,
        declared_files_missing_from_main_total=35,
    )

    assert row.declared_files_missing_from_main == files


def test_a_row_key_is_the_children_map_key():
    local = _row()
    external = _row(id=44, project_id='reify', relation=ChildRelation.EXTERNAL_DEPENDENT)

    assert (local.key, external.key) == ('6905', 'reify:44')


def test_rows_round_trip_through_their_json_form():
    row = _row()

    wire = rows_to_json([row])

    assert json.loads(json.dumps(wire)) == wire
    assert ChildRow.model_validate(wire[0]) == row


def _view(**overrides: object) -> CancelChildrenView:
    fields: dict[str, object] = {
        'task_id': PARENT,
        'project_id': PROJECT,
        'status': TaskStatus.PENDING,
        'children': (_row(), _row(id=44, project_id='reify', relation='external_dependent')),
        'superseded_by': (SUPERSEDER,),
        'external_dependents_checked': True,
        'enforce': False,
        **overrides,
    }
    return CancelChildrenView.model_validate(fields)


def test_a_view_carries_the_parent_its_rows_and_both_flags():
    view = _view()

    assert [row.key for row in view.children] == ['6905', 'reify:44']
    assert (view.external_dependents_checked, view.enforce) == (True, False)


@pytest.mark.parametrize('flag', ['external_dependents_checked', 'enforce'])
def test_a_view_has_no_default_for_either_flag(flag):
    fields = _view().model_dump()
    del fields[flag]

    with pytest.raises(ValidationError):
        CancelChildrenView.model_validate(fields)


def test_a_view_listing_one_child_twice_is_rejected():
    with pytest.raises(ValidationError):
        _view(children=(_row(), _row(title='again')))


@pytest.mark.parametrize(
    'superseded_by',
    [
        pytest.param((SUPERSEDER, SUPERSEDER), id='a-superseder-twice'),
        pytest.param((SUPERSEDER, PARENT), id='the-task-itself'),
    ],
)
def test_a_view_rejects_a_superseded_by_the_parser_refuses(superseded_by: tuple[int, ...]):
    with pytest.raises(ValidationError):
        _view(superseded_by=superseded_by)


@pytest.mark.parametrize(
    'row',
    [
        pytest.param(_row(project_id='reify'), id='local-row-in-another-project'),
        pytest.param(
            _row(id=44, relation=ChildRelation.EXTERNAL_DEPENDENT), id='external-row-in-this-project'
        ),
    ],
)
def test_a_view_rejects_a_row_whose_project_contradicts_its_relation(row: ChildRow):
    with pytest.raises(ValidationError):
        _view(children=(row,))


def test_a_view_rejects_a_non_canonical_project_id():
    with pytest.raises(ValidationError):
        _view(project_id='Dark-Factory', children=())


def test_capped_lists_at_most_the_payload_cap_and_the_total():
    assert PAYLOAD_ID_CAP == 50
    assert capped('undispositioned', list(range(75))) == {
        'undispositioned': list(range(50)),
        'undispositioned_total': 75,
    }


def test_capped_returns_a_short_list_whole():
    assert capped('unknown', [3, 1]) == {'unknown': [3, 1], 'unknown_total': 2}


# --- the decision table and its renders (§5, §6 row 25) -----------------------


def test_the_decision_table_is_seven_rows_in_order():
    assert isinstance(DECISION_TABLE, tuple)
    assert all(isinstance(row, DecisionRow) for row in DECISION_TABLE)
    assert [row.order for row in DECISION_TABLE] == list(range(7))


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        pytest.param('order', 9, id='order'),
        pytest.param('question', 'changed', id='question'),
    ],
)
def test_a_decision_row_is_frozen(field: str, value: object):
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(DECISION_TABLE[0], field, value)


def test_the_decision_table_reaches_every_caller_action_and_no_entry_only_one():
    reached = {action for row in DECISION_TABLE for action in row.dispositions}

    assert reached == CALLER_ACTIONS
    assert not reached & ENTRY_ONLY_ACTIONS


@pytest.mark.parametrize('row', DECISION_TABLE, ids=lambda row: f'row-{row.order}')
def test_a_row_s_action_prose_names_each_of_its_dispositions_in_bold(row: DecisionRow):
    assert all(f'**{action.value}**' in row.action_text for action in row.dispositions)


def test_the_first_and_last_rows_dispose_of_nothing():
    assert DECISION_TABLE[0].dispositions == ()
    assert DECISION_TABLE[-1].dispositions == ()


def _table_lines() -> list[str]:
    return [line for line in render_decision_table_markdown().splitlines() if line.startswith('|')]


def test_the_table_render_is_deterministic_and_one_line_per_row():
    assert render_decision_table_markdown() == render_decision_table_markdown()
    assert len(_table_lines()) == len(DECISION_TABLE) + 2


@pytest.mark.parametrize('row', DECISION_TABLE, ids=lambda row: f'row-{row.order}')
def test_each_row_renders_its_question_and_examples_on_its_own_line(row):
    line = _table_lines()[row.order + 2]

    assert row.question in line
    assert all(example in line for example in row.report_examples)


def test_the_rules_render_is_one_subsection_of_a_stage2_region():
    rules = render_cancel_children_rules()

    assert rules.startswith('### ')
    assert not [line for line in rules.splitlines() if line.startswith('## ')]


def test_the_rules_name_the_read_tool_the_skill_and_every_caller_action():
    rules = render_cancel_children_rules()

    assert GET_CANCEL_CHILDREN_TOOL in rules
    assert CANCEL_CHILDREN_SKILL in rules
    assert all(f'`{action.value}`' in rules for action in CALLER_ACTIONS)


@pytest.mark.parametrize(
    'action_with_fields',
    [
        pytest.param('- `keep` (`reason`?): ', id='keep'),
        pytest.param('- `repoint` (`onto`, `reason`?): ', id='repoint'),
        pytest.param('- `fold` (`into`, `reason`?): ', id='fold'),
        pytest.param('- `moot` (`reason`?): ', id='moot'),
    ],
)
def test_the_rules_list_each_action_with_the_fields_its_schema_declares(action_with_fields):
    assert action_with_fields in render_cancel_children_rules()


def test_the_rules_embed_the_table_render_verbatim():
    assert render_decision_table_markdown() in render_cancel_children_rules()


def test_the_rules_forbid_folding_or_mooting_an_in_progress_child():
    child_cancelling = ' or '.join(
        f'`{action.value}`' for action in sorted(CHILD_CANCELLING_ACTIONS)
    )
    sentences = render_cancel_children_rules().split('. ')

    assert [
        sentence
        for sentence in sentences
        if 'may not' in sentence
        and child_cancelling in sentence
        and f'`{TaskStatus.IN_PROGRESS.value}`' in sentence
    ]


def test_the_rules_key_children_only_by_fields_a_listed_row_carries():
    rules = render_cancel_children_rules()
    named = re.findall(r"row's `(\w+)`", rules)
    row_fields = set(rows_to_json([_row()])[0])

    assert named
    assert set(named) <= row_fields
    assert f'`<project_id>:<id>` for a `{ChildRelation.EXTERNAL_DEPENDENT.value}` row' in rules


# --- the metadata vocabulary (§4.8, §6 row 26) --------------------------------

_TASK_METADATA_ONLY_CHILD = """
import json, sys
from shared.task_metadata import parse_metadata
metadata, warnings = parse_metadata(json.loads(sys.argv[1]), direction='read')
entries = metadata.model_extra['cancel_dispositions']
print(json.dumps([
    [warning.code for warning in warnings],
    [type(entry).__name__ for entry in entries],
    [type(entry.root).__name__ for entry in entries],
]))
"""

_CANCEL_DISPOSITIONS_FIRST_CHILD = """
import json, sys
import shared.cancel_dispositions
loaded_metadata = 'shared.task_metadata' in sys.modules
from shared.task_metadata import is_known_metadata_key
print(json.dumps([loaded_metadata, is_known_metadata_key('cancel_dispositions')]))
"""


def test_a_process_importing_only_task_metadata_types_the_cancel_dispositions_slice():
    blob = json.dumps({'cancel_dispositions': entries_to_json(_ENTRIES[:2])})

    seen = run_python_child(_TASK_METADATA_ONLY_CHILD, blob)

    assert seen == [
        [],
        ['CancelDispositionEntry', 'CancelDispositionEntry'],
        ['KeepEntry', 'RepointEntry'],
    ]


def test_cancel_dispositions_loads_without_task_metadata_and_the_key_is_known_after():
    assert run_python_child(_CANCEL_DISPOSITIONS_FIRST_CHILD) == [False, True]


def test_parsed_entries_from_metadata_read_back_through_parse_entries():
    metadata, _ = parse_metadata(
        {'cancel_dispositions': entries_to_json(_ENTRIES)}, direction='read'
    )

    assert metadata.model_extra is not None
    assert parse_entries(metadata.model_extra['cancel_dispositions']) == _ENTRIES


def test_a_single_entry_not_in_a_list_is_the_wrong_cardinality():
    raw = {'cancel_dispositions': entries_to_json(_ENTRIES[:1])[0]}

    _, warnings = parse_metadata(raw, direction='read')

    assert [(w.field, w.code) for w in warnings] == [('cancel_dispositions', 'wrong_cardinality')]
    with pytest.raises(TypeError):
        parse_metadata(raw, direction='write', enforce=True)


def test_an_entry_naming_a_bare_parent_id_is_an_invalid_submodel_kept_raw():
    raw_entries = [_entry_json(parent=str(PARENT))]

    metadata, warnings = parse_metadata({'cancel_dispositions': raw_entries}, direction='read')

    assert [(w.field, w.code) for w in warnings] == [('cancel_dispositions', 'invalid_submodel')]
    assert metadata.model_dump()['cancel_dispositions'] == raw_entries


# --- the type-strict spawned_from lookup forms (§4.3) -------------------------


def test_a_decimal_parent_id_is_looked_up_in_both_stored_forms():
    assert sibling_lookup_values('5869') == ('5869', 5869)


def test_a_non_decimal_parent_id_is_looked_up_as_text_only():
    assert sibling_lookup_values('task-x') == ('task-x',)


def test_an_id_wider_than_a_sqlite_integer_still_yields_both_forms():
    too_wide = str(2**63)

    assert sibling_lookup_values(too_wide) == (too_wide, 2**63)
