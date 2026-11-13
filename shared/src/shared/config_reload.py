"""Diff and in-place reload between two instances of one pydantic config class.

The leaf set is the class's declared fields (``shared.config_fields``), so both sides
of a diff carry the same paths whatever their runtime values. Pure apart from the
in-place write: no I/O and no logging.
Contract: plans/review-all-s16-orchestrator-config-shape-defaults-and-routing-prd.md, C4.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel

from shared.config_fields import DeclaredField, Leaf, declared_fields


class _Absent:
    __slots__ = ()

    def __repr__(self) -> str:
        return '<absent>'

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


def iter_leaves(
    model_cls: type[BaseModel], model: BaseModel | None
) -> Iterator[tuple[str, object]]:
    for declared in _leaf_fields(model_cls):
        yield declared.path, _read(model, declared.segments)


def diff_config(live: BaseModel, fresh: BaseModel, reloadable: frozenset[str]) -> ConfigDiff:
    applied: dict[str, Change] = {}
    restart_required: dict[str, Change] = {}
    for changes in _change_groups(live, fresh).values():
        bucket = applied if changes.keys() <= reloadable else restart_required
        bucket.update(changes)
    unchanged = len(_leaf_fields(type(live))) - len(applied) - len(restart_required)
    return ConfigDiff(MappingProxyType(applied), MappingProxyType(restart_required), unchanged)


@functools.cache
def _leaf_fields(model_cls: type[BaseModel]) -> tuple[DeclaredField, ...]:
    return tuple(
        declared for declared in declared_fields(model_cls) if isinstance(declared.shape, Leaf)
    )


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
