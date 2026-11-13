"""The declared structure of a pydantic config class: reload tier and legacy spellings.

A field states its own tier and its old spellings as ``Annotated`` markers, and
every derivation here walks the declared annotations only, never a runtime value.
Contract: plans/review-all-s16-orchestrator-config-shape-defaults-and-routing-prd.md, C3.
"""

from __future__ import annotations

import functools
import types
import typing
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from types import MappingProxyType
from typing import Annotated, TypeGuard, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo


class Reload(Enum):
    GREEN = 'green'


@dataclass(frozen=True)
class LegacySpelling:
    path: str


@dataclass(frozen=True)
class TableLegacy:
    rows: Mapping[str, tuple[str, str]] = field(default_factory=dict)
    knob_major_blocks: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, 'rows', MappingProxyType(dict(self.rows)))

    def __hash__(self) -> int:
        return hash((tuple(sorted(self.rows.items())), self.knob_major_blocks))

    @classmethod
    def knob_major(cls, old_block: str, knob: str) -> TableLegacy:
        return cls(knob_major_blocks=((old_block, knob),))


@dataclass(frozen=True)
class RestartOnlyRows:
    rows: frozenset[str]


_MARKER_TYPES = (LegacySpelling, TableLegacy, RestartOnlyRows)


class ConfigDeclarationError(TypeError):
    """A config class places or spells its field markers inconsistently."""


@dataclass(frozen=True)
class Leaf:
    pass


@dataclass(frozen=True)
class Submodel:
    model_cls: type[BaseModel]
    optional: bool


@dataclass(frozen=True)
class Table:
    key_enum: type[StrEnum]
    row_cls: type[BaseModel]


Shape = Leaf | Submodel | Table


@dataclass(frozen=True)
class DeclaredField:
    segments: tuple[str, ...]
    info: FieldInfo
    shape: Shape
    reloadable: bool
    in_table_row: bool

    @property
    def path(self) -> str:
        return '.'.join(self.segments)


@functools.cache
def declared_fields(model_cls: type[BaseModel]) -> tuple[DeclaredField, ...]:
    return tuple(_walk(model_cls, (), green_above=False, row_restart_only=False, in_row=False))


@functools.cache
def reloadable_paths(model_cls: type[BaseModel]) -> frozenset[str]:
    return frozenset(
        declared.path
        for declared in declared_fields(model_cls)
        if isinstance(declared.shape, Leaf) and declared.reloadable
    )


@functools.cache
def legacy_spellings(model_cls: type[BaseModel]) -> Mapping[str, str]:
    spellings: dict[str, str] = {}
    for declared in declared_fields(model_cls):
        for old_path, new_path in _declared_moves(declared):
            if old_path in spellings:
                raise ConfigDeclarationError(
                    f'{old_path!r} is declared as the old spelling of both '
                    f'{spellings[old_path]} and {new_path}'
                )
            spellings[old_path] = new_path
    for old_path, new_path in spellings.items():
        if old_path in _declared_paths(model_cls):
            raise ConfigDeclarationError(
                f'{old_path!r} is declared as the old spelling of {new_path}, '
                f'but {old_path} is still a declared path of {model_cls.__name__}'
            )
    return MappingProxyType(spellings)


@functools.cache
def _declared_paths(model_cls: type[BaseModel]) -> frozenset[str]:
    paths = {declared.path for declared in declared_fields(model_cls)}
    for declared in declared_fields(model_cls):
        if isinstance(declared.shape, Table):
            paths |= {f'{declared.path}.{member.value}' for member in declared.shape.key_enum}
    return frozenset(paths)


def _declared_moves(declared: DeclaredField) -> Iterator[tuple[str, str]]:
    for marker in declared.info.metadata:
        if not isinstance(marker, (LegacySpelling, TableLegacy)):
            continue
        if declared.in_table_row:
            raise ConfigDeclarationError(
                f'{declared.path}: {marker!r} sits inside a table-row class, which is '
                f'mounted once per row; declare the move as TableLegacy on the table field'
            )
        if isinstance(marker, LegacySpelling):
            yield marker.path, declared.path
        elif isinstance(declared.shape, Table):
            for old_path, (row, knob) in _table_moves(declared.path, declared.shape, marker):
                yield old_path, f'{declared.path}.{row}.{knob}'


def _table_moves(
    path: str, table: Table, marker: TableLegacy
) -> list[tuple[str, tuple[str, str]]]:
    moves = [
        (f'{old_block}.{member.value}', (member.value, knob))
        for old_block, knob in marker.knob_major_blocks
        for member in table.key_enum
    ]
    moves += marker.rows.items()
    _require_rows(path, table, frozenset(row for _, (row, _) in moves), 'TableLegacy')
    knobs = list(table.row_cls.model_fields)
    unknown = sorted({knob for _, (_, knob) in moves} - set(knobs))
    if unknown:
        raise ConfigDeclarationError(
            f'{path}: TableLegacy names {unknown}, which are not fields of '
            f'{table.row_cls.__name__}; fields: {knobs}'
        )
    return moves


def _walk(
    model_cls: type[BaseModel],
    prefix: tuple[str, ...],
    *,
    green_above: bool,
    row_restart_only: bool,
    in_row: bool,
) -> Iterator[DeclaredField]:
    for name, info in model_cls.model_fields.items():
        segments = (*prefix, name)
        path = '.'.join(segments)
        _reject_nested_markers(path, info.annotation)
        shape = _shape(info.annotation)
        green = green_above or any(marker is Reload.GREEN for marker in info.metadata)
        restart_rows = _checked_restart_rows(path, info, shape, green)
        yield DeclaredField(segments, info, shape, green and not row_restart_only, in_row)
        if isinstance(shape, Submodel):
            yield from _walk(
                shape.model_cls,
                segments,
                green_above=green,
                row_restart_only=row_restart_only,
                in_row=in_row,
            )
        elif isinstance(shape, Table):
            for member in shape.key_enum:
                yield from _walk(
                    shape.row_cls,
                    (*segments, member.value),
                    green_above=green,
                    row_restart_only=row_restart_only or member.value in restart_rows,
                    in_row=True,
                )


def _shape(annotation: object) -> Shape:
    if _is_model_class(annotation):
        return Submodel(annotation, optional=False)
    if get_origin(annotation) in (typing.Union, types.UnionType):
        arms = get_args(annotation)
        models = [arm for arm in arms if arm is not type(None)]
        if len(arms) == 2 and len(models) == 1 and _is_model_class(models[0]):
            return Submodel(models[0], optional=True)
        return Leaf()
    origin = get_origin(annotation)
    if isinstance(origin, type) and issubclass(origin, Mapping):
        args = get_args(annotation)
        if len(args) == 2 and _is_str_enum_class(args[0]) and _is_model_class(args[1]):
            return Table(args[0], args[1])
    return Leaf()


def _is_model_class(candidate: object) -> TypeGuard[type[BaseModel]]:
    return isinstance(candidate, type) and issubclass(candidate, BaseModel)


def _is_str_enum_class(candidate: object) -> TypeGuard[type[StrEnum]]:
    return isinstance(candidate, type) and issubclass(candidate, StrEnum)


def _is_marker(candidate: object) -> bool:
    return candidate is Reload.GREEN or isinstance(candidate, _MARKER_TYPES)


def _reject_nested_markers(path: str, annotation: object) -> None:
    for arg in get_args(annotation):
        if get_origin(arg) is Annotated:
            misplaced = [marker for marker in arg.__metadata__ if _is_marker(marker)]
            if misplaced:
                raise ConfigDeclarationError(
                    f'{path}: {_describe(misplaced[0])} is nested inside the annotation '
                    f'{annotation!r}; a marker annotates the field itself, e.g. '
                    f'Annotated[X | None, Reload.GREEN], or it has no effect'
                )
        _reject_nested_markers(path, arg)


def _checked_restart_rows(
    path: str, info: FieldInfo, shape: Shape, green: bool
) -> frozenset[str]:
    for marker in info.metadata:
        if isinstance(marker, (RestartOnlyRows, TableLegacy)) and not isinstance(shape, Table):
            raise ConfigDeclarationError(
                f'{path}: {_describe(marker)} belongs on a Mapping[StrEnum, BaseModel] '
                f'table field, and this field is not one'
            )
        if isinstance(marker, RestartOnlyRows) and not green:
            raise ConfigDeclarationError(
                f'{path}: {_describe(marker)} is only meaningful on a Reload.GREEN '
                f'table; every row of this table is restart-only already'
            )
    restart_rows = frozenset(
        row for marker in info.metadata if isinstance(marker, RestartOnlyRows) for row in marker.rows
    )
    if isinstance(shape, Table):
        _require_rows(path, shape, restart_rows, 'RestartOnlyRows')
    return restart_rows


def _require_rows(path: str, table: Table, rows: frozenset[str], named_by: str) -> None:
    allowed = [member.value for member in table.key_enum]
    unknown = sorted(rows - set(allowed))
    if unknown:
        raise ConfigDeclarationError(
            f'{path}: {named_by} names {unknown}, which are not rows of '
            f'{table.key_enum.__name__}; allowed: {allowed}'
        )


def _describe(marker: object) -> str:
    return 'Reload.GREEN' if marker is Reload.GREEN else repr(marker)
