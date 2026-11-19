"""Tests for shared.config_fields: tiers and spellings derived from declared field markers."""

import copy
from collections.abc import Mapping, MutableMapping
from typing import Annotated

import pytest
from _config_fixtures import ROW_KNOBS, Role, Row, Sub
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from shared.config_fields import (
    ConfigDeclarationError,
    ConflictingSpellings,
    DeprecatedSpelling,
    LegacySpelling,
    Reload,
    RestartOnlyRows,
    TableLegacy,
    UnplaceableSpelling,
    declared_fields,
    legacy_spellings,
    pivot_legacy_spellings,
    reloadable_paths,
)


class SubPlusOne(BaseModel):
    alpha: int = 1
    beta: str = 'b'
    added: bool = False


class BeforeAdding(BaseModel):
    sub: Annotated[Sub, Reload.GREEN] = Field(default_factory=Sub)
    flat: int = 0


class AfterAdding(BaseModel):
    sub: Annotated[SubPlusOne, Reload.GREEN] = Field(default_factory=SubPlusOne)
    flat: int = 0


class Inner(BaseModel):
    leaf: int = 0


class Outer(BaseModel):
    inner: Inner = Field(default_factory=Inner)


class Nested(BaseModel):
    outer: Annotated[Outer, Reload.GREEN] = Field(default_factory=Outer)


class Optionals(BaseModel):
    opt_green: Annotated[Sub | None, Reload.GREEN] = None
    opt: Sub | None = None


class OneGreenLeaf(BaseModel):
    tuned: Annotated[int, Reload.GREEN] = 0
    fixed: int = 0


class LeafMarked(BaseModel):
    section: OneGreenLeaf = Field(default_factory=OneGreenLeaf)


class Tabled(BaseModel):
    roles: Annotated[
        dict[Role, Row], Reload.GREEN, RestartOnlyRows(frozenset({'reviewer'}))
    ] = Field(default_factory=dict)


class Atomic(BaseModel):
    by_name: Annotated[dict[str, Sub], Reload.GREEN] = Field(default_factory=dict)
    in_order: Annotated[list[Sub], Reload.GREEN] = Field(default_factory=list)


def test_a_field_added_to_a_green_submodel_is_reloadable_and_an_unmarked_root_is_not():
    assert 'sub.added' in reloadable_paths(AfterAdding)
    assert 'sub.added' not in reloadable_paths(BeforeAdding)
    assert 'flat' not in reloadable_paths(AfterAdding)
    assert 'flat' not in reloadable_paths(BeforeAdding)
    assert reloadable_paths(BeforeAdding) == frozenset({'sub.alpha', 'sub.beta'})


def test_green_is_inherited_at_full_depth():
    assert reloadable_paths(Nested) == frozenset({'outer.inner.leaf'})


def test_a_green_optional_submodel_contributes_its_leaves_and_an_unmarked_one_none():
    assert reloadable_paths(Optionals) == frozenset({'opt_green.alpha', 'opt_green.beta'})


def test_optional_submodels_are_descended_whether_marked_or_not():
    paths = {declared.path for declared in declared_fields(Optionals)}
    assert {'opt.alpha', 'opt.beta', 'opt_green.alpha', 'opt_green.beta'} <= paths


def test_a_green_leaf_inside_an_unmarked_submodel_is_the_only_reloadable_path():
    assert reloadable_paths(LeafMarked) == frozenset({'section.tuned'})


def test_a_green_table_expands_per_row_and_leaves_out_restart_only_rows():
    expected = {
        f'roles.{role.value}.{knob}'
        for role in Role
        if role is not Role.REVIEWER
        for knob in ROW_KNOBS
    }
    assert reloadable_paths(Tabled) == frozenset(expected)


def test_table_rows_come_from_the_key_enum_in_declaration_order():
    row_paths = [
        declared.path
        for declared in declared_fields(Tabled)
        if declared.path.endswith('.model')
    ]
    assert row_paths == [f'roles.{role.value}.model' for role in Role]


def test_str_keyed_dicts_and_lists_of_models_are_atomic_leaves():
    assert reloadable_paths(Atomic) == frozenset({'by_name', 'in_order'})


class RestartRowsOnUnmarkedTable(BaseModel):
    roles: Annotated[dict[Role, Row], RestartOnlyRows(frozenset({'reviewer'}))] = Field(
        default_factory=dict
    )


class RestartRowsOnScalar(BaseModel):
    slots: Annotated[int, Reload.GREEN, RestartOnlyRows(frozenset({'reviewer'}))] = 1


class RestartRowsNamingAStranger(BaseModel):
    roles: Annotated[
        dict[Role, Row], Reload.GREEN, RestartOnlyRows(frozenset({'nope'}))
    ] = Field(default_factory=dict)


class TableLegacyOnScalar(BaseModel):
    slots: Annotated[int, TableLegacy.knob_major('models', 'model')] = 1


class MarkerInsideUnionArm(BaseModel):
    opt: Annotated[Sub, Reload.GREEN] | None = None


class MarkerInsideListArgument(BaseModel):
    items: list[Annotated[Sub, Reload.GREEN]] = Field(default_factory=list)


@pytest.mark.parametrize(
    ('model_cls', 'named'),
    [
        (RestartRowsOnUnmarkedTable, ('roles', 'reviewer')),
        (RestartRowsOnScalar, ('slots', 'reviewer')),
        (RestartRowsNamingAStranger, ('roles', 'nope', 'architect', 'reviewer', 'unblock_auto')),
        (TableLegacyOnScalar, ('slots', 'models')),
        (MarkerInsideUnionArm, ('opt', 'Reload.GREEN')),
        (MarkerInsideListArgument, ('items', 'Reload.GREEN')),
    ],
    ids=[
        'restart-rows-on-unmarked-table',
        'restart-rows-on-non-table',
        'restart-rows-naming-unknown-row',
        'table-legacy-on-non-table',
        'marker-inside-union-arm',
        'marker-inside-list-argument',
    ],
)
def test_a_misplaced_marker_is_a_declaration_error_naming_path_and_value(model_cls, named):
    with pytest.raises(ConfigDeclarationError) as raised:
        reloadable_paths(model_cls)
    for text in named:
        assert text in str(raised.value)


def test_a_declaration_error_is_a_type_error():
    assert issubclass(ConfigDeclarationError, TypeError)


class Admission(BaseModel):
    slots: Annotated[int, LegacySpelling('old_slots')] = 4


class ScalarSpelled(BaseModel):
    admission: Admission = Field(default_factory=Admission)


class KnobMajorTable(BaseModel):
    roles: Annotated[dict[Role, Row], TableLegacy.knob_major('models', 'model')] = Field(
        default_factory=dict
    )


ROW_MAP = TableLegacy(
    rows={
        'unblock_auto.model': ('unblock_auto', 'model'),
        'watcher_model': ('reviewer', 'model'),
    }
)


class RowMapTable(BaseModel):
    roles: Annotated[dict[Role, Row], ROW_MAP] = Field(default_factory=dict)


class UnionedTable(BaseModel):
    roles: Annotated[
        dict[Role, Row],
        TableLegacy.knob_major('models', 'model'),
        TableLegacy.knob_major('budgets', 'budget_usd'),
        ROW_MAP,
    ] = Field(default_factory=dict)


def test_a_scalar_spelling_maps_its_root_relative_old_path_to_the_field_path():
    assert dict(legacy_spellings(ScalarSpelled)) == {'old_slots': 'admission.slots'}


def test_knob_major_maps_every_row_of_the_key_enum():
    assert dict(legacy_spellings(KnobMajorTable)) == {
        f'models.{role.value}': f'roles.{role.value}.model' for role in Role
    }


def test_a_row_map_maps_each_old_path_to_its_row_and_knob():
    assert dict(legacy_spellings(RowMapTable)) == {
        'unblock_auto.model': 'roles.unblock_auto.model',
        'watcher_model': 'roles.reviewer.model',
    }


def test_several_table_markers_on_one_field_union():
    expected = {f'models.{role.value}': f'roles.{role.value}.model' for role in Role}
    expected |= {f'budgets.{role.value}': f'roles.{role.value}.budget_usd' for role in Role}
    expected |= {
        'unblock_auto.model': 'roles.unblock_auto.model',
        'watcher_model': 'roles.reviewer.model',
    }
    assert dict(legacy_spellings(UnionedTable)) == expected


def test_the_spelling_map_is_read_only():
    spellings = legacy_spellings(ScalarSpelled)
    assert isinstance(spellings, Mapping)
    assert not isinstance(spellings, MutableMapping)


class DoublyClaimed(BaseModel):
    first: Annotated[int, LegacySpelling('legacy_knob')] = 0
    second: Annotated[int, LegacySpelling('legacy_knob')] = 0


class Shadowing(BaseModel):
    flat: int = 0
    slots: Annotated[int, LegacySpelling('flat')] = 1


class RowMapNamingAStranger(BaseModel):
    roles: Annotated[
        dict[Role, Row], TableLegacy(rows={'stranger_model': ('nope', 'model')})
    ] = Field(default_factory=dict)


class KnobMajorNamingAStranger(BaseModel):
    roles: Annotated[dict[Role, Row], TableLegacy.knob_major('temps', 'temperature')] = Field(
        default_factory=dict
    )


class SpelledRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    model: Annotated[str, LegacySpelling('old_model')] = 'sonnet'


class SpellingInsideARow(BaseModel):
    roles: dict[Role, SpelledRow] = Field(default_factory=dict)


@pytest.mark.parametrize(
    ('model_cls', 'named'),
    [
        (DoublyClaimed, ('legacy_knob', 'first', 'second')),
        (Shadowing, ('flat', 'slots')),
        (RowMapNamingAStranger, ('roles', 'nope', 'architect', 'reviewer', 'unblock_auto')),
        (KnobMajorNamingAStranger, ('roles', 'temperature', 'model', 'budget_usd', 'max_turns')),
        (SpellingInsideARow, ('roles.', 'old_model')),
    ],
    ids=[
        'old-path-declared-twice',
        'old-path-shadows-a-declared-path',
        'table-legacy-row-not-in-key-enum',
        'table-legacy-knob-not-a-row-field',
        'legacy-spelling-inside-a-table-row',
    ],
)
def test_an_inconsistent_spelling_is_a_declaration_error_naming_the_values(model_cls, named):
    with pytest.raises(ConfigDeclarationError) as raised:
        legacy_spellings(model_cls)
    for text in named:
        assert text in str(raised.value)


SLOTS_MOVED = DeprecatedSpelling('old_slots', 'admission.slots')


@pytest.mark.parametrize(
    'data',
    [{'old_slots': 6}, {'old_slots': 6, 'admission': {'slots': 6}}],
    ids=['old-only', 'old-and-equal-new'],
)
def test_a_scalar_spelling_pivots_to_its_new_path_with_one_deprecation(data):
    assert pivot_legacy_spellings(ScalarSpelled, data) == (
        {'admission': {'slots': 6}},
        (SLOTS_MOVED,),
    )


def test_a_scalar_spelling_disagreeing_with_its_new_path_conflicts():
    with pytest.raises(ConflictingSpellings) as raised:
        pivot_legacy_spellings(ScalarSpelled, {'old_slots': 6, 'admission': {'slots': 7}})
    assert (raised.value.old_path, raised.value.new_path) == ('old_slots', 'admission.slots')
    assert raised.value.new_value_path == 'admission.slots'
    for text in ('old_slots', 'admission.slots', '6', '7'):
        assert text in str(raised.value)


REVIEWER_MODEL_SPELLINGS = (
    DeprecatedSpelling('models.reviewer', 'roles.reviewer.model'),
    DeprecatedSpelling('watcher_model', 'roles.reviewer.model'),
)


def test_two_agreeing_old_spellings_of_one_new_path_give_two_deprecations():
    data = {'models': {'reviewer': 'opus'}, 'watcher_model': 'opus'}
    assert pivot_legacy_spellings(UnionedTable, data) == (
        {'roles': {'reviewer': {'model': 'opus'}}},
        REVIEWER_MODEL_SPELLINGS,
    )


def test_two_disagreeing_old_spellings_of_one_new_path_conflict_naming_both_old_paths():
    data = {'models': {'reviewer': 'opus'}, 'watcher_model': 'sonnet'}
    with pytest.raises(ConflictingSpellings) as raised:
        pivot_legacy_spellings(UnionedTable, data)
    conflict = raised.value
    assert (conflict.old_path, conflict.new_value_path, conflict.new_path) == (
        'watcher_model',
        'models.reviewer',
        'roles.reviewer.model',
    )
    message = str(conflict)
    for text in ("models.reviewer='opus'", "watcher_model='sonnet'", 'roles.reviewer.model'):
        assert text in message
    assert 'roles.reviewer.model=' not in message


class UnblockAuto(BaseModel):
    enabled: bool = False


class PivotTable(BaseModel):
    unblock_auto: UnblockAuto = Field(default_factory=UnblockAuto)
    roles: Annotated[
        dict[Role, Row],
        TableLegacy.knob_major('models', 'model'),
        TableLegacy(rows={'unblock_auto.model': ('unblock_auto', 'model')}),
    ] = Field(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def _pivot_legacy_spellings(cls, data: object) -> object:
        if isinstance(data, Mapping):
            pivoted, _ = pivot_legacy_spellings(cls, data)
            return pivoted
        return data


def test_a_knob_major_block_pivots_into_the_table_and_its_emptied_block_goes():
    assert pivot_legacy_spellings(PivotTable, {'models': {'architect': 'opus'}}) == (
        {'roles': {'architect': {'model': 'opus'}}},
        (DeprecatedSpelling('models.architect', 'roles.architect.model'),),
    )


def test_a_row_map_entry_pivots_into_its_row_and_keeps_the_declared_siblings():
    assert pivot_legacy_spellings(
        PivotTable, {'unblock_auto': {'model': 'opus', 'enabled': True}}
    ) == (
        {'unblock_auto': {'enabled': True}, 'roles': {'unblock_auto': {'model': 'opus'}}},
        (DeprecatedSpelling('unblock_auto.model', 'roles.unblock_auto.model'),),
    )


def test_a_declared_container_emptied_by_the_pivot_is_kept():
    pivoted, _ = pivot_legacy_spellings(PivotTable, {'unblock_auto': {'model': 'opus'}})
    assert pivoted['unblock_auto'] == {}


def test_a_table_spelling_disagreeing_with_its_row_conflicts_naming_both_paths():
    data = {'models': {'architect': 'opus'}, 'roles': {'architect': {'model': 'sonnet'}}}
    with pytest.raises(ConflictingSpellings) as raised:
        pivot_legacy_spellings(PivotTable, data)
    assert (raised.value.old_path, raised.value.new_path) == (
        'models.architect',
        'roles.architect.model',
    )


@pytest.mark.parametrize(
    ('model_cls', 'data'),
    [
        (
            PivotTable,
            {'models': {'architect': 'opus'}, 'unblock_auto': {'model': 'opus', 'enabled': True}},
        ),
        (ScalarSpelled, {'old_slots': 6, 'admission': {'slots': 6, 'other': [1, 2]}}),
    ],
    ids=['table', 'scalar'],
)
def test_the_pivot_never_mutates_its_input(model_cls, data):
    before = copy.deepcopy(data)
    pivot_legacy_spellings(model_cls, data)
    assert data == before


def test_an_undeclared_key_stays_exactly_where_it_was():
    pivoted, deprecated = pivot_legacy_spellings(
        PivotTable,
        {'models': {'architect': 'opus', 'stranger': 'x'}, 'mystery': {'depth': 1}},
    )
    assert pivoted == {
        'models': {'stranger': 'x'},
        'mystery': {'depth': 1},
        'roles': {'architect': {'model': 'opus'}},
    }
    assert deprecated == (DeprecatedSpelling('models.architect', 'roles.architect.model'),)


@pytest.mark.parametrize(
    ('model_cls', 'data'),
    [
        (ScalarSpelled, {'old_slots': 6, 'admission': 5}),
        (PivotTable, {'models': 'opus'}),
    ],
    ids=['new-path-intermediate-is-a-scalar', 'old-path-intermediate-is-a-scalar'],
)
def test_a_key_that_cannot_be_placed_stays_at_its_old_path_unreported(model_cls, data):
    assert pivot_legacy_spellings(model_cls, data) == (data, ())


@pytest.mark.parametrize(
    ('model_cls', 'data', 'named'),
    [
        (
            ScalarSpelled,
            {'old_slots': 9, 'admission': Admission()},
            ('old_slots', 'admission.slots', 'admission', 'Admission'),
        ),
        (
            PivotTable,
            {'models': {'architect': 'opus'}, 'roles': {Role.ARCHITECT: Row()}},
            ('models.architect', 'roles.architect.model', 'roles.architect', 'Row'),
        ),
    ],
    ids=['submodel-instance', 'table-row-instance'],
)
def test_an_old_spelling_blocked_by_a_model_instance_is_refused_not_dropped(
    model_cls, data, named
):
    with pytest.raises(UnplaceableSpelling) as raised:
        pivot_legacy_spellings(model_cls, data)
    assert (raised.value.old_path, raised.value.new_path, raised.value.holder_path) == named[:3]
    for text in named:
        assert text in str(raised.value)


def test_an_unplaceable_spelling_is_a_value_error_so_a_root_validator_surfaces_it():
    assert issubclass(UnplaceableSpelling, ValueError)
    with pytest.raises(ValidationError) as raised:
        PivotTable.model_validate({'models': {'architect': 'opus'}, 'roles': {Role.ARCHITECT: Row()}})
    assert 'models.architect' in str(raised.value)


def test_a_root_before_validator_loads_an_old_spelling_at_its_new_path():
    loaded = PivotTable.model_validate({'models': {'architect': 'opus'}})
    assert loaded.roles[Role.ARCHITECT].model == 'opus'


def test_a_conflict_inside_the_root_validator_is_a_validation_error_naming_both_paths():
    data = {'models': {'architect': 'opus'}, 'roles': {'architect': {'model': 'sonnet'}}}
    with pytest.raises(ValidationError) as raised:
        PivotTable.model_validate(data)
    assert 'models.architect' in str(raised.value)
    assert 'roles.architect.model' in str(raised.value)
