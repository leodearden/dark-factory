#!/usr/bin/env python3
"""Measure one module into units, and evaluate a proposed partition of them.

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
with the file total is a MetricsError, never a result (INV-11). A partition
assigns every unit to exactly one cluster; its evaluation reports what the
split would cost in heuristic-13 terms: refs crossing clusters, cluster
cycles, and attributes written from more than one cluster.

Exit codes
----------
0  measured -- the map, and the partition report under ``--partition``, is
   printed to stdout.
2  instrument failure (``source_measures.MetricsError``, or its subclass
   ``PartitionError``) -- the source cannot be read or parsed, the class is
   absent, the map does not reconcile, the partition file is unreadable or
   not a JSON object of cluster name to unit names, or the partition does not
   assign every unit exactly once. Never 1: this is a measurement, not a gate,
   so no measured result is a failure.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import enum
import json
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TypeGuard

import _checkout
import source_measures
from source_measures import MetricsError

__all__ = [
    'ClusterSize',
    'ConcernMap',
    'Edge',
    'MetricsError',
    'PartitionError',
    'PartitionReport',
    'Unit',
    'UnitKind',
    'evaluate_partition',
    'main',
    'measure_module',
]

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


# ---------------------------------------------------------------------------
# Partition evaluation: the measurable heuristic-13 symptoms of a proposed split.


class PartitionError(MetricsError):
    """A partition that does not assign every measured unit to exactly one cluster."""


@dataclasses.dataclass(frozen=True)
class ClusterSize:
    units: int
    lines: int
    prose_lines: int


@dataclasses.dataclass(frozen=True)
class Edge:
    count: int
    sample: tuple[str, str]


@dataclasses.dataclass(frozen=True)
class PartitionReport:
    clusters: Mapping[str, ClusterSize]
    edges: Mapping[tuple[str, str], Edge]
    cyclic_groups: tuple[frozenset[str], ...]
    shared_writes: Mapping[str, frozenset[str]]
    module_unit_users: Mapping[str, frozenset[str]]
    cluster_of: Mapping[str, str]


def evaluate_partition(cmap: ConcernMap, partition: Mapping[str, Sequence[str]]) -> PartitionReport:
    cluster_of = _cluster_of(cmap, partition)
    edges = _edges(cmap.units, cluster_of)
    return PartitionReport(
        clusters=_cluster_sizes(cmap.units, partition),
        edges=edges,
        cyclic_groups=_cyclic_groups(partition, edges),
        shared_writes=_shared_writes(cmap.units, cluster_of),
        module_unit_users=_module_unit_users(cmap.units, cluster_of),
        cluster_of=MappingProxyType(cluster_of),
    )


def _cluster_of(cmap: ConcernMap, partition: Mapping[str, Sequence[str]]) -> dict[str, str]:
    measured = {unit.name for unit in cmap.units}
    listed = Counter(name for members in partition.values() for name in members)
    problems = [
        f'{label}: {", ".join(sorted(names))}'
        for label, names in (
            ('unassigned', measured - listed.keys()),
            ('assigned more than once', {name for name, count in listed.items() if count > 1}),
            ('not a measured unit', listed.keys() - measured),
        )
        if names
    ]
    if problems:
        raise PartitionError(
            f'{cmap.path}: the partition must assign every unit of {cmap.class_name!r} '
            f'and the module exactly once -- {"; ".join(problems)}'
        )
    return {name: cluster for cluster, members in partition.items() for name in members}


def _cluster_sizes(
    units: tuple[Unit, ...], partition: Mapping[str, Sequence[str]]
) -> Mapping[str, ClusterSize]:
    by_name = {unit.name: unit for unit in units}
    return MappingProxyType({
        cluster: ClusterSize(
            units=len(members),
            lines=sum(by_name[name].lines for name in members),
            prose_lines=sum(by_name[name].prose_lines for name in members),
        )
        for cluster, members in partition.items()
    })


def _edges(units: tuple[Unit, ...], cluster_of: Mapping[str, str]) -> Mapping[tuple[str, str], Edge]:
    pairs: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for unit in units:
        source = cluster_of[unit.name]
        for ref in unit.refs:
            if cluster_of[ref] != source:
                pairs.setdefault((source, cluster_of[ref]), []).append((unit.name, ref))
    return MappingProxyType({
        key: Edge(count=len(found), sample=min(found)) for key, found in sorted(pairs.items())
    })


def _cyclic_groups(
    clusters: Iterable[str], edges: Mapping[tuple[str, str], Edge]
) -> tuple[frozenset[str], ...]:
    successors: dict[str, set[str]] = {cluster: set() for cluster in clusters}
    for source, target in edges:
        successors[source].add(target)
    reach = {cluster: _reachable(cluster, successors) for cluster in successors}
    components = {
        frozenset({cluster, *(other for other in reach[cluster] if cluster in reach[other])})
        for cluster in successors
    }
    return tuple(sorted((group for group in components if len(group) > 1), key=sorted))


def _reachable(start: str, successors: Mapping[str, set[str]]) -> frozenset[str]:
    seen: set[str] = set()
    stack = [start]
    while stack:
        fresh = successors[stack.pop()] - seen
        seen |= fresh
        stack.extend(fresh)
    return frozenset(seen)


def _shared_writes(
    units: tuple[Unit, ...], cluster_of: Mapping[str, str]
) -> Mapping[str, frozenset[str]]:
    writers: dict[str, set[str]] = {}
    for unit in units:
        for attribute in unit.writes:
            writers.setdefault(attribute, set()).add(cluster_of[unit.name])
    return MappingProxyType({
        attribute: frozenset(clusters)
        for attribute, clusters in sorted(writers.items())
        if len(clusters) > 1
    })


def _module_unit_users(
    units: tuple[Unit, ...], cluster_of: Mapping[str, str]
) -> Mapping[str, frozenset[str]]:
    users: dict[str, set[str]] = {
        unit.name: set() for unit in units if unit.kind is UnitKind.MODULE
    }
    for unit in units:
        for ref in unit.refs & users.keys():
            users[ref].add(cluster_of[unit.name])
    return MappingProxyType({name: frozenset(clusters) for name, clusters in users.items()})


# ---------------------------------------------------------------------------
# Renderings: sorted and deterministic, so a committed one diffs cleanly.


def _edge_key(key: tuple[str, str]) -> str:
    return f'{key[0]} -> {key[1]}'


def _unit_json(unit: Unit) -> dict[str, object]:
    return {
        'name': unit.name,
        'kind': unit.kind.value,
        'lines': unit.lines,
        'prose_lines': unit.prose_lines,
        'writes': sorted(unit.writes),
        'reads': sorted(unit.reads),
        'refs': sorted(unit.refs),
    }


def _report_json(report: PartitionReport) -> dict[str, object]:
    return {
        'clusters': {name: dataclasses.asdict(size) for name, size in report.clusters.items()},
        'edges': {
            _edge_key(key): {'count': edge.count, 'sample': list(edge.sample)}
            for key, edge in report.edges.items()
        },
        'cyclic_groups': [sorted(group) for group in report.cyclic_groups],
        'shared_writes': {name: sorted(owners) for name, owners in report.shared_writes.items()},
        'module_unit_users': {
            name: sorted(users) for name, users in report.module_unit_users.items()
        },
        'cluster_of': dict(report.cluster_of),
    }


def _render_json(cmap: ConcernMap, report: PartitionReport | None) -> str:
    rendered: dict[str, object] = {
        'path': cmap.path,
        'class_name': cmap.class_name,
        'total_lines': cmap.total_lines,
        'residual': dict(cmap.residual),
        'units': [_unit_json(unit) for unit in cmap.units],
    }
    if report is not None:
        rendered['partition'] = _report_json(report)
    return json.dumps(rendered, indent=2)


def _table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    body = [list(row) for row in rows]
    numeric = [
        bool(body) and all(isinstance(row[column], int) for row in body)
        for column in range(len(header))
    ]
    separator = ['---:' if right else '---' for right in numeric]
    return '\n'.join(
        '| ' + ' | '.join(str(cell) for cell in row) + ' |'
        for row in (header, separator, *body)
    )


def _names(names: Iterable[str]) -> str:
    return ', '.join(sorted(names))


def _bullets(items: Sequence[str]) -> str:
    return '\n'.join(f'- {item}' for item in items) if items else 'none'


def _markdown_header(cmap: ConcernMap) -> str:
    in_units = sum(unit.lines for unit in cmap.units)
    buckets = ', '.join(f'{name} {count}' for name, count in cmap.residual.items())
    return (
        f'# Concern map: `{cmap.path}`, class `{cmap.class_name}`\n\n'
        f'{cmap.total_lines} lines: {in_units} in {len(cmap.units)} units, and the '
        f'residual buckets {buckets}.'
    )


def _markdown_units(cmap: ConcernMap, report: PartitionReport | None) -> str:
    cluster_column = [] if report is None else ['cluster']
    rows = [
        [
            unit.name, unit.kind.value, unit.lines, unit.prose_lines,
            _names(unit.writes), _names(unit.reads), _names(unit.refs),
            *([] if report is None else [report.cluster_of[unit.name]]),
        ]
        for unit in cmap.units
    ]
    header = ['unit', 'kind', 'lines', 'prose', 'writes', 'reads', 'refs', *cluster_column]
    return '## Units\n\n' + _table(header, rows)


def _markdown_partition(report: PartitionReport) -> list[str]:
    clusters = _table(
        ['cluster', 'units', 'lines', 'prose'],
        ([name, size.units, size.lines, size.prose_lines] for name, size in report.clusters.items()),
    )
    edges = _table(
        ['from', 'to', 'refs', 'sample'],
        ([*key, edge.count, ' -> '.join(edge.sample)] for key, edge in report.edges.items()),
    ) if report.edges else 'none'
    users = _table(
        ['module unit', 'clusters'],
        ([name, _names(found) or 'none'] for name, found in report.module_unit_users.items()),
    )
    return [
        '## Clusters\n\n' + clusters,
        '## Cross-cluster edges\n\n' + edges,
        '## Cyclic cluster groups\n\n' + _bullets([_names(group) for group in report.cyclic_groups]),
        '## Shared writes\n\n' + _bullets(
            [f'`{name}`: {_names(owners)}' for name, owners in report.shared_writes.items()]
        ),
        '## Module-unit users\n\n' + users,
    ]


def _render_markdown(cmap: ConcernMap, report: PartitionReport | None) -> str:
    sections = [_markdown_header(cmap), _markdown_units(cmap, report)]
    if report is not None:
        sections.extend(_markdown_partition(report))
    return '\n\n'.join(sections)


# ---------------------------------------------------------------------------
# Command line, in the house shape of scripts/merge_lane_metrics.py.


def _is_partition(content: object) -> TypeGuard[dict[str, list[str]]]:
    return isinstance(content, dict) and all(
        isinstance(cluster, str)
        and isinstance(members, list)
        and all(isinstance(member, str) for member in members)
        for cluster, members in content.items()
    )


def _load_partition(path: str) -> dict[str, list[str]]:
    try:
        content = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetricsError(
            f'{path}: could not be read as a partition -- {exc.__class__.__name__}: {exc}'
        ) from exc
    if not _is_partition(content):
        raise MetricsError(
            f'{path}: a partition is a JSON object mapping each cluster name to a '
            'list of unit names'
        )
    return content


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure one module into units (a class's methods and the module's "
            'top-level bindings), and evaluate a proposed partition of them.'
        ),
        epilog=(
            'Exit codes: 0 measured; 2 instrument failure (an unreadable or '
            'unparseable source, an absent class, a map that does not reconcile, '
            'an unreadable or malformed partition file, or a partition that does '
            'not assign every unit exactly once). Never 1: a measurement is not a gate.'
        ),
    )
    parser.add_argument('path', help='The module to measure, relative to --root.')
    parser.add_argument(
        '--class', dest='class_name', required=True, help='The class whose methods are units.'
    )
    parser.add_argument(
        '--partition',
        help='A JSON object mapping each cluster name to its unit names; adds its report.',
    )
    parser.add_argument('--format', choices=('markdown', 'json'), default='markdown')
    parser.add_argument(
        '--root', default=str(_checkout.ROOT), help='The checkout root (default: this one).'
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. See the module docstring for the exit-code contract."""
    args = _build_parser().parse_args(argv)
    try:
        source = source_measures.read_source(Path(args.root), args.path)
        cmap = measure_module(source, path=args.path, class_name=args.class_name)
        report = (
            None if args.partition is None
            else evaluate_partition(cmap, _load_partition(args.partition))
        )
    except MetricsError as exc:
        print(f'concern_map: {exc}', file=sys.stderr)
        return 2
    render = _render_json if args.format == 'json' else _render_markdown
    print(render(cmap, report))
    return 0


if __name__ == '__main__':
    sys.exit(main())
