#!/usr/bin/env python3
"""Measure one module into units, the input to judging a proposed split of it.

A unit is a method of the target class, named ``Class.method``, or a top-level
def, class or assignment of the module, named by what it binds; every binding
of one name (a property's getter and setter, an overload) is one unit. A unit
spans from its first decorator, or the comment block directly above it, to its
last line. It records how many of those lines are prose, the attributes it
writes and reads on the class's receiver (``self``, ``cls`` or the class name),
and the units it refs: a receiver attribute naming a method, called or not, or
a bare name bound to a top-level unit and not shadowed locally, counting
everything nested inside the unit. Every line of the file is measured exactly
once, by a unit or by a named residual bucket; a map that does not reconcile
with the file total is a MetricsError, never a result (INV-11).
"""
from __future__ import annotations

import ast
import dataclasses
import enum
from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import TypeGuard

import _checkout as _checkout
import source_measures
from source_measures import MetricsError

__all__ = ['ConcernMap', 'MetricsError', 'Unit', 'UnitKind', 'measure_module']

RESIDUAL_BUCKETS: tuple[str, ...] = ('module_preamble', 'class_non_method', 'module_other', 'inter_unit_gaps')

_FUNCTION_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)
_DEFS = (*_FUNCTION_DEFS, ast.ClassDef)


class UnitKind(enum.Enum):
    METHOD = 'method'
    MODULE = 'module'


@dataclasses.dataclass(frozen=True)
class Unit:
    name: str
    kind: UnitKind
    lines: int
    prose_lines: int
    writes: frozenset[str]
    reads: frozenset[str]
    refs: frozenset[str]


@dataclasses.dataclass(frozen=True)
class ConcernMap:
    path: str
    class_name: str
    total_lines: int
    units: tuple[Unit, ...]
    residual: Mapping[str, int]


@dataclasses.dataclass(frozen=True)
class _Piece:
    name: str
    kind: UnitKind
    node: ast.stmt
    first: int
    last: int


@dataclasses.dataclass(frozen=True)
class _Scope:
    class_name: str
    members: frozenset[str]
    module_names: Mapping[str, str]


@dataclasses.dataclass(frozen=True)
class _Access:
    writes: frozenset[str] = frozenset()
    reads: frozenset[str] = frozenset()
    refs: frozenset[str] = frozenset()

    def __or__(self, other: _Access) -> _Access:
        return _Access(
            self.writes | other.writes, self.reads | other.reads, self.refs | other.refs
        )


_NO_ACCESS = _Access()


def measure_module(source: str, *, path: str, class_name: str) -> ConcernMap:
    tree = source_measures.parse_source(source, path=path)
    target = _target_class(tree, path=path, class_name=class_name)
    prose = source_measures.prose_line_numbers_in_tree(source, tree, path=path)
    lines = source.splitlines()
    pieces = tuple(_pieces(tree, target, lines))
    scope = _Scope(class_name, _member_names(target), _module_names(tree, target))
    units = _units(pieces, prose, scope)
    residual = _residual(tree, target, pieces, lines)
    _reconcile(path, len(lines), units, residual)
    return ConcernMap(path, class_name, len(lines), units, MappingProxyType(residual))


# ---------------------------------------------------------------------------
# Spans: which statements are units, and which lines each one owns.


def _target_class(tree: ast.Module, *, path: str, class_name: str) -> ast.ClassDef:
    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef) and stmt.name == class_name:
            return stmt
    raise MetricsError(f'{path}: no top-level class {class_name!r} to measure')


def _last_line(stmt: ast.stmt) -> int:
    return stmt.end_lineno if stmt.end_lineno is not None else stmt.lineno


def _first_line(stmt: ast.stmt, *, floor: int, lines: list[str]) -> int:
    decorators = getattr(stmt, 'decorator_list', [])
    first = min([stmt.lineno, *(decorator.lineno for decorator in decorators)])
    while first - 1 > floor and lines[first - 2].lstrip().startswith('#'):
        first -= 1
    return first


def _spans(body: Iterable[ast.stmt], *, floor: int, lines: list[str]) -> Iterator[tuple[ast.stmt, int, int]]:
    for stmt in body:
        yield stmt, _first_line(stmt, floor=floor, lines=lines), _last_line(stmt)
        floor = _last_line(stmt)


def _assignment_targets(stmt: ast.stmt) -> list[ast.expr]:
    if isinstance(stmt, ast.Assign):
        return stmt.targets
    if isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
        return [stmt.target]
    return []


def _bound_names(stmt: ast.stmt) -> list[str]:
    if isinstance(stmt, _DEFS):
        return [stmt.name]
    return [
        node.id
        for target in _assignment_targets(stmt)
        for node in ast.walk(target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    ]


def _pieces(tree: ast.Module, target: ast.ClassDef, lines: list[str]) -> Iterator[_Piece]:
    for stmt, first, last in _spans(tree.body, floor=0, lines=lines):
        if stmt is target:
            yield from _method_pieces(target, lines)
        elif names := _bound_names(stmt):
            yield _Piece(', '.join(names), UnitKind.MODULE, stmt, first, last)


def _method_pieces(target: ast.ClassDef, lines: list[str]) -> Iterator[_Piece]:
    for stmt, first, last in _spans(target.body, floor=target.lineno, lines=lines):
        if isinstance(stmt, _FUNCTION_DEFS):
            yield _Piece(f'{target.name}.{stmt.name}', UnitKind.METHOD, stmt, first, last)


def _member_names(target: ast.ClassDef) -> frozenset[str]:
    return frozenset(stmt.name for stmt in target.body if isinstance(stmt, _FUNCTION_DEFS))


def _module_names(tree: ast.Module, target: ast.ClassDef) -> Mapping[str, str]:
    names: dict[str, str] = {}
    for stmt in tree.body:
        bound = _bound_names(stmt) if stmt is not target else []
        for name in bound:
            names.setdefault(name, ', '.join(bound))
    return MappingProxyType(names)


# ---------------------------------------------------------------------------
# Access: what each unit writes, reads and refs.


def _units(pieces: tuple[_Piece, ...], prose: frozenset[int], scope: _Scope) -> tuple[Unit, ...]:
    grouped: dict[str, list[_Piece]] = {}
    for piece in pieces:
        grouped.setdefault(piece.name, []).append(piece)
    return tuple(_unit(group, prose, scope) for group in grouped.values())


def _unit(group: list[_Piece], prose: frozenset[int], scope: _Scope) -> Unit:
    span = [line for piece in group for line in range(piece.first, piece.last + 1)]
    access = _NO_ACCESS
    for piece in group:
        access |= _access(piece, scope)
    return Unit(
        name=group[0].name,
        kind=group[0].kind,
        lines=len(span),
        prose_lines=sum(1 for line in span if line in prose),
        writes=access.writes,
        reads=access.reads,
        refs=access.refs,
    )


def _receivers(piece: _Piece, class_name: str) -> frozenset[str]:
    receiver = _receiver_param(piece)
    return frozenset({class_name} if receiver is None else {class_name, receiver})


def _receiver_param(piece: _Piece) -> str | None:
    node = piece.node
    if piece.kind is not UnitKind.METHOD or not isinstance(node, _FUNCTION_DEFS):
        return None
    if any(_is_name(decorator, 'staticmethod') for decorator in node.decorator_list):
        return None
    params = [*node.args.posonlyargs, *node.args.args]
    return params[0].arg if params else None


def _is_name(node: ast.expr, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _local_names(unit_node: ast.stmt) -> frozenset[str]:
    bound: set[str] = set()
    declared_outer: set[str] = set()
    for child in ast.walk(unit_node):
        if isinstance(child, (ast.Global, ast.Nonlocal)):
            declared_outer.update(child.names)
        elif child is not unit_node:
            bound.update(_names_bound_by(child))
    return frozenset(bound - declared_outer)


def _names_bound_by(node: ast.AST) -> Iterator[str]:
    if isinstance(node, ast.arg):
        yield node.arg
    elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
        yield node.id
    elif isinstance(node, _DEFS):
        yield node.name
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        yield from ((alias.asname or alias.name).split('.')[0] for alias in node.names)
    elif isinstance(node, ast.ExceptHandler) and node.name:
        yield node.name


def _access(piece: _Piece, scope: _Scope) -> _Access:
    receivers = _receivers(piece, scope.class_name)
    local = _local_names(piece.node)
    access = _NO_ACCESS
    for child in ast.walk(piece.node):
        if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
            if child.value.id in receivers:
                access |= _attribute_access(child, scope)
        elif _is_module_ref(child, local, scope):
            access |= _Access(refs=frozenset({scope.module_names[child.id]}))
    return access


def _is_module_ref(node: ast.AST, local: frozenset[str], scope: _Scope) -> TypeGuard[ast.Name]:
    return (
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id not in local
        and node.id in scope.module_names
    )


def _attribute_access(node: ast.Attribute, scope: _Scope) -> _Access:
    if not isinstance(node.ctx, ast.Load):
        return _Access(writes=frozenset({node.attr}))
    if node.attr in scope.members:
        return _Access(refs=frozenset({f'{scope.class_name}.{node.attr}'}))
    return _Access(reads=frozenset({node.attr}))


# ---------------------------------------------------------------------------
# Residual and reconciliation: every line not in a unit is in a named bucket.


def _residual_bucket(stmt: ast.stmt, tree: ast.Module, target: ast.ClassDef) -> str:
    if stmt is target:
        return 'class_non_method'
    if stmt is source_measures.docstring_of(tree) or isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return 'module_preamble'
    return 'module_other'


def _residual(
    tree: ast.Module, target: ast.ClassDef, pieces: tuple[_Piece, ...], lines: list[str]
) -> dict[str, int]:
    in_units = {line for piece in pieces for line in range(piece.first, piece.last + 1)}
    bucket_of: dict[int, str] = {}
    for stmt, first, last in _spans(tree.body, floor=0, lines=lines):
        for line in range(first, last + 1):
            bucket_of[line] = _residual_bucket(stmt, tree, target)
    residual: dict[str, int] = dict.fromkeys(RESIDUAL_BUCKETS, 0)
    for line in range(1, len(lines) + 1):
        if line not in in_units:
            residual[bucket_of.get(line, 'inter_unit_gaps')] += 1
    return residual


def _reconcile(path: str, total: int, units: tuple[Unit, ...], residual: Mapping[str, int]) -> None:
    measured = sum(unit.lines for unit in units) + sum(residual.values())
    if measured != total:
        raise MetricsError(
            f'{path}: units and residual buckets measure {measured} lines but the '
            f'file has {total} -- a line was measured twice or never'
        )
