"""Tests for shared.config_fields: tiers and spellings derived from declared field markers."""

from enum import StrEnum
from typing import Annotated

import pytest
from pydantic import BaseModel, ConfigDict, Field

from shared.config_fields import (
    ConfigDeclarationError,
    LegacySpelling,
    Reload,
    RestartOnlyRows,
    TableLegacy,
    declared_fields,
    legacy_spellings,
    reloadable_paths,
)


class Role(StrEnum):
    ARCHITECT = 'architect'
    REVIEWER = 'reviewer'
    UNBLOCK_AUTO = 'unblock_auto'


class Row(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    model: str = 'sonnet'
    budget_usd: float = 1.0
    max_turns: int = 10


ROW_KNOBS = ('model', 'budget_usd', 'max_turns')


class Sub(BaseModel):
    alpha: int = 1
    beta: str = 'b'


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


@pytest.mark.parametrize(
    ('model_cls', 'named'),
    [
        (RestartRowsOnUnmarkedTable, ('roles', 'reviewer')),
        (RestartRowsOnScalar, ('slots', 'reviewer')),
        (RestartRowsNamingAStranger, ('roles', 'nope', 'architect', 'reviewer', 'unblock_auto')),
        (TableLegacyOnScalar, ('slots', 'models')),
        (MarkerInsideUnionArm, ('opt', 'Reload.GREEN')),
    ],
    ids=[
        'restart-rows-on-unmarked-table',
        'restart-rows-on-non-table',
        'restart-rows-naming-unknown-row',
        'table-legacy-on-non-table',
        'marker-inside-union-arm',
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


def _assign_item(container, key, value) -> None:
    container[key] = value


def test_the_spelling_map_is_read_only():
    with pytest.raises(TypeError):
        _assign_item(legacy_spellings(ScalarSpelled), 'another', 'admission.slots')


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
