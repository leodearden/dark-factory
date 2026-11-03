"""shared.cancel_dispositions: the cancel-with-children contract.

Contract: plans/cancel-with-children-prd.md §4.2. Everything goes through the public names.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
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
    with pytest.raises(TypeError):
        parsed[6906] = KeepDisposition()  # type: ignore[index]


def test_an_external_child_key_is_the_qualified_task_ref():
    child = ExternalChild(project_id='reify', task_id=44)

    assert child.key == 'reify:44' == qualified_task_ref('reify', 44)


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
            [3, 3],
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
    ('raw', 'reason'),
    [
        pytest.param([], Reason.INVALID_VALUE, id='empty'),
        pytest.param('8032', Reason.INVALID_VALUE, id='not-a-list'),
        pytest.param([True], Reason.INVALID_VALUE, id='bool-is-not-an-id'),
        pytest.param([8032, 8032], Reason.DUPLICATE_TARGET, id='duplicate'),
        pytest.param([PARENT], Reason.TARGET_IS_PARENT, id='names-the-parent'),
    ],
)
def test_parse_superseded_by_refuses_on_the_superseded_by_field(raw, reason):
    with pytest.raises(CancelChildrenInvalid) as caught:
        parse_superseded_by(raw, parent_id=PARENT)

    assert (caught.value.reason, caught.value.field) == (reason, 'superseded_by')


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
}


def _check(
    children: Mapping[ChildKey, ChildDispositionVariant],
    superseded_by: tuple[int, ...] = (),
    depends_on: Mapping[int, list[int]] | None = None,
) -> None:
    check_targets(
        children, superseded_by, target_status=_STATUSES, depends_on=depends_on or {}
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


def test_entry_for_parent_finds_the_parent_s_entry_or_none():
    entries = (_keep_for(A, 'a'), _keep_for(B, 'b'))

    assert entry_for_parent(entries, A) == _keep_for(A, 'a')
    assert entry_for_parent(entries, C) is None


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


def test_a_row_carrying_every_documented_fact_validates_and_is_frozen():
    row = _row()

    assert (row.status, row.relation) == (TaskStatus.PENDING, ChildRelation.DEPENDENT)
    assert isinstance(row.existing_disposition, RepointEntry)
    with pytest.raises(ValidationError):
        row.title = 'changed'  # type: ignore[misc]


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


def test_capped_lists_at_most_the_payload_cap_and_the_total():
    assert PAYLOAD_ID_CAP == 50
    assert capped('undispositioned', list(range(75))) == {
        'undispositioned': list(range(50)),
        'undispositioned_total': 75,
    }


def test_capped_returns_a_short_list_whole():
    assert capped('unknown', [3, 1]) == {'unknown': [3, 1], 'unknown_total': 2}


# --- the decision table and its renders (§5, §6 row 25) -----------------------


def test_the_decision_table_is_seven_frozen_rows_in_order():
    assert isinstance(DECISION_TABLE, tuple)
    assert all(isinstance(row, DecisionRow) for row in DECISION_TABLE)
    assert [row.order for row in DECISION_TABLE] == list(range(7))
    with pytest.raises(dataclasses.FrozenInstanceError):
        DECISION_TABLE[0].order = 9  # type: ignore[misc]


def test_the_decision_table_reaches_every_caller_action_and_no_entry_only_one():
    reached = {action for row in DECISION_TABLE for action in row.dispositions}

    assert reached == CALLER_ACTIONS
    assert not reached & ENTRY_ONLY_ACTIONS


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
    assert all(example in line for example in row.example_ids)


def test_the_rules_render_is_one_subsection_of_a_stage2_region():
    rules = render_cancel_children_rules()

    assert rules.startswith('### ')
    assert not [line for line in rules.splitlines() if line.startswith('## ')]


def test_the_rules_name_the_read_tool_the_skill_and_every_caller_action():
    rules = render_cancel_children_rules()

    assert GET_CANCEL_CHILDREN_TOOL in rules
    assert CANCEL_CHILDREN_SKILL in rules
    assert all(f'`{action.value}`' in rules for action in CALLER_ACTIONS)


def test_the_rules_name_each_action_s_fields_from_its_schema():
    rules = render_cancel_children_rules()

    assert '`onto`' in rules
    assert '`into`' in rules


def test_the_rules_embed_the_table_render_verbatim():
    assert render_decision_table_markdown() in render_cancel_children_rules()


def test_the_rules_forbid_folding_or_mooting_an_in_progress_child():
    assert f'`{TaskStatus.IN_PROGRESS.value}`' in render_cancel_children_rules()


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
