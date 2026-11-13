"""Diff and in-place reload between two instances of one pydantic config class.

The leaf set is the class's declared fields (``shared.config_fields``), so both sides
of a diff carry the same paths whatever their runtime values. Pure apart from the
in-place write: no I/O and no logging.
Contract: plans/review-all-s16-orchestrator-config-shape-defaults-and-routing-prd.md, C4.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final, TypedDict, TypeVar

from pydantic import BaseModel

from shared.config_fields import (
    DeclaredField,
    DeprecatedSpelling,
    Leaf,
    Table,
    declared_fields,
    reloadable_paths,
)

M = TypeVar('M', bound=BaseModel)

_ABSENT_RENDERED: Final = '<absent>'


class _Absent:
    __slots__ = ()

    def __repr__(self) -> str:
        return _ABSENT_RENDERED

    def __reduce__(self) -> str:
        return 'ABSENT'


ABSENT: Final = _Absent()


@dataclass(frozen=True)
class Change:
    old: object
    new: object


@dataclass(frozen=True)
class ConfigDiff:
    applied_candidates: Mapping[str, Change]
    restart_required: Mapping[str, Change]
    unchanged: int


class ReloadReport(TypedDict):
    reloaded: bool
    applied: dict[str, dict[str, object]]
    restart_required: dict[str, dict[str, object]]
    unchanged: int
    error: str | None
    deprecated: list[dict[str, str]]


def iter_leaves(
    model_cls: type[BaseModel], model: BaseModel | None
) -> Iterator[tuple[str, object]]:
    for declared in _leaf_fields(model_cls):
        yield declared.path, _read(model, declared.segments)


def diff_config(live: BaseModel, fresh: BaseModel, reloadable: frozenset[str]) -> ConfigDiff:
    diff, _ = _planned_reload(live, fresh, reloadable)
    return diff


def apply_reload(
    live: M,
    fresh: M,
    *,
    reloadable: frozenset[str] | None = None,
    deprecated: Sequence[DeprecatedSpelling] = (),
) -> ReloadReport:
    if reloadable is None:
        reloadable = reloadable_paths(type(live))
    diff, write_points = _planned_reload(live, fresh, reloadable)
    undo: list[_Undo] = []
    try:
        for point in write_points:
            _put(_slots(live, point), _fresh_value(fresh, point), undo)
        type(live).model_validate(live.model_dump())
    except BaseException as exc:
        for entry in reversed(undo):
            _assign(entry.slot, entry.previous, entry.key_order)
        if not isinstance(exc, ValueError):
            raise
        return _report(False, {}, diff, deprecated, error=f'hybrid-invariant: {exc}')
    return _report(True, diff.applied_candidates, diff, deprecated, error=None)


@functools.cache
def _leaf_fields(model_cls: type[BaseModel]) -> tuple[DeclaredField, ...]:
    return tuple(
        declared for declared in declared_fields(model_cls) if isinstance(declared.shape, Leaf)
    )


def _planned_reload(
    live: BaseModel, fresh: BaseModel, reloadable: frozenset[str]
) -> tuple[ConfigDiff, list[tuple[str, ...]]]:
    applied: dict[str, Change] = {}
    restart_required: dict[str, Change] = {}
    write_points: list[tuple[str, ...]] = []
    for point, changes in _change_groups(live, fresh).items():
        if changes.keys() <= reloadable:
            applied.update(changes)
            write_points.append(point)
        else:
            restart_required.update(changes)
    unchanged = len(_leaf_fields(type(live))) - len(applied) - len(restart_required)
    diff = ConfigDiff(MappingProxyType(applied), MappingProxyType(restart_required), unchanged)
    return diff, write_points


def _change_groups(
    live: BaseModel, fresh: BaseModel
) -> dict[tuple[str, ...], dict[str, Change]]:
    if type(live) is not type(fresh):
        raise TypeError(
            f'a config diff needs two instances of one class, got '
            f'{type(live).__qualname__} and {type(fresh).__qualname__}'
        )
    groups: dict[tuple[str, ...], dict[str, Change]] = {}
    for declared in _leaf_fields(type(live)):
        old = _read(live, declared.segments)
        new = _read(fresh, declared.segments)
        if old != new:
            point = _structural_point(live, fresh, declared.segments) or declared.segments
            groups.setdefault(point, {})[declared.path] = Change(old, new)
    return groups


def _structural_point(
    live: BaseModel, fresh: BaseModel, segments: tuple[str, ...]
) -> tuple[str, ...] | None:
    live_node: object = live
    fresh_node: object = fresh
    for depth, name in enumerate(segments[:-1], start=1):
        live_node = _child(live_node, name)
        fresh_node = _child(fresh_node, name)
        if _is_hole(live_node) or _is_hole(fresh_node):
            return segments[:depth]
    return None


def _read(model: BaseModel | None, segments: tuple[str, ...]) -> object:
    node: object = model
    for name in segments:
        node = _child(node, name)
    return node


def _child(node: object, name: str) -> object:
    if isinstance(node, BaseModel):
        return node.__dict__.get(name, ABSENT)
    if isinstance(node, Mapping):
        return node.get(name, ABSENT)
    return ABSENT


def _is_hole(node: object) -> bool:
    return node is None or node is ABSENT


@dataclass(frozen=True)
class _Slot:
    container: BaseModel | MutableMapping[str, object]
    key: str
    row_order: tuple[StrEnum, ...] = ()


@dataclass(frozen=True)
class _Undo:
    slot: _Slot
    previous: object
    key_order: tuple[object, ...]


def _slots(live: BaseModel, point: tuple[str, ...]) -> list[_Slot]:
    key_enums = _table_key_enums(type(live))
    slots: list[_Slot] = []
    node: object = live
    for depth, name in enumerate(point):
        if isinstance(node, BaseModel):
            slots.append(_Slot(node, name))
        elif isinstance(node, MutableMapping):
            key_enum = key_enums[point[:depth]]
            slots.append(_Slot(node, key_enum(name), tuple(key_enum)))
        else:
            raise TypeError(
                f'{".".join(point[:depth])} holds a {type(node).__qualname__}, '
                f'which a reload cannot write into'
            )
        node = _child(node, name)
    return slots


@functools.cache
def _table_key_enums(model_cls: type[BaseModel]) -> Mapping[tuple[str, ...], type[StrEnum]]:
    return MappingProxyType({
        declared.segments: declared.shape.key_enum
        for declared in declared_fields(model_cls)
        if isinstance(declared.shape, Table)
    })


def _fresh_value(fresh: BaseModel, point: tuple[str, ...]) -> object:
    value = _read(fresh, point)
    return value.model_copy(deep=True) if isinstance(value, BaseModel) else value


def _put(slots: list[_Slot], value: object, undo: list[_Undo]) -> None:
    slot = slots[-1]
    container = slot.container
    if isinstance(container, BaseModel) and container.model_config.get('frozen'):
        if len(slots) == 1:
            raise TypeError(f'cannot reload into the frozen root {type(container).__qualname__}')
        _put(slots[:-1], container.model_copy(update={slot.key: value}), undo)
        return
    key_order = () if isinstance(container, BaseModel) else tuple(container)
    undo.append(_Undo(slot, _child(container, slot.key), key_order))
    _assign(slot, value, slot.row_order)


def _assign(slot: _Slot, value: object, key_order: Sequence[object]) -> None:
    container = slot.container
    if isinstance(container, BaseModel):
        object.__setattr__(container, slot.key, value)
        return
    if value is ABSENT:
        del container[slot.key]
    else:
        container[slot.key] = value
    _reorder(container, key_order)


def _reorder(table: MutableMapping[str, object], key_order: Sequence[object]) -> None:
    rank = {key: index for index, key in enumerate(key_order)}
    ordered = sorted(table, key=lambda key: rank.get(key, len(rank)))
    if ordered == list(table):
        return
    rows = {key: table[key] for key in ordered}
    table.clear()
    table.update(rows)


def _report(
    reloaded: bool,
    applied: Mapping[str, Change],
    diff: ConfigDiff,
    deprecated: Sequence[DeprecatedSpelling],
    *,
    error: str | None,
) -> ReloadReport:
    return ReloadReport(
        reloaded=reloaded,
        applied=_rendered(applied),
        restart_required=_rendered(diff.restart_required),
        unchanged=diff.unchanged,
        error=error,
        deprecated=[
            {'old_path': spelling.old_path, 'new_path': spelling.new_path}
            for spelling in deprecated
        ],
    )


def _rendered(changes: Mapping[str, Change]) -> dict[str, dict[str, object]]:
    return {
        path: {'old': _rendered_value(change.old), 'new': _rendered_value(change.new)}
        for path, change in changes.items()
    }


def _rendered_value(value: object) -> object:
    return _ABSENT_RENDERED if value is ABSENT else value
