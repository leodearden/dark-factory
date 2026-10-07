#!/usr/bin/env python3
"""The whole-repo quality metrics snapshot: plans/quality-metrics-snapshot-prd.md §Contract, schema 1.

A report, never a gate: it emits measures and the quality doc's two heuristic-14
marks, and no verdict, ranking, average or threshold of its own.

    --run-id RUN_ID --out PATH [--root DIR] [--diff PREVIOUS.json]   measure HEAD, write PATH
    --current SNAPSHOT.json --diff PREVIOUS.json                      diff two snapshots
    --summary SNAPSHOT.json                                           per-member table

Exit 0 when done; exit 2 on an instrument failure, with the cause named on
stderr. The PRD holds the rationale for every choice made here.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import source_measures

from shared import safe_io

SCHEMA_VERSION = 1
INSTRUMENT = 'quality-metrics-snapshot'

#: heuristic 14's two marks (docs/code-quality.md, heuristic 14) -- the doc's figures, not this script's thresholds.
H14_SOFT_CEILING_LINES = 1500
H14_ALARM_LINES = 2000

NO_PREVIOUS_LINE = 'no previous snapshot given; since = none'

_DEFAULT_ROOT = Path(__file__).resolve().parents[1]

_SRC = str(source_measures.FileKind.SRC)
_TESTS = str(source_measures.FileKind.TESTS)

_TOP_LEVEL_KEYS = (
    'schema_version', 'instrument', 'run_id', 'as_of_sha', 'since',
    'evidence', 'cost', 'params', 'files', 'functions', 'import_graph',
)
_EVIDENCE_KEYS = ('members', 'domain_files', 'measured_files', 'unreadable', 'complete')
_COMMON_FIELDS = frozenset({
    'member', 'kind', 'blob', 'lines', 'prose_lines', 'prose_ratio',
    'cognitive_total', 'cognitive_max', 'cognitive_max_function', 'functions',
})
_KIND_FIELDS: Mapping[str, frozenset[str]] = {
    _SRC: frozenset({
        'module', 'package_init', 'function_local_imports', 'reexport_names',
        'reach_back_imports', 'fan_out', 'fan_in_src', 'fan_in_tests',
    }),
    _TESTS: frozenset({'private_patch_targets', 'private_reads'}),
}
_INTEGER_FIELDS = frozenset({
    'lines', 'prose_lines', 'cognitive_total', 'cognitive_max', 'functions',
    'function_local_imports', 'reach_back_imports', 'fan_out', 'fan_in_src', 'fan_in_tests',
    'private_reads',
})
_IMPORT_GRAPH_KEYS = ('edges', 'reach_back', 'deferred', 'cycles')


# ---------------------------------------------------------------------------
# The import graph: pure functions over module names and parsed statements.


@dataclasses.dataclass(frozen=True)
class _ImportStatement:
    node: ast.Import | ast.ImportFrom
    line: int
    runtime: bool  # False inside an `if TYPE_CHECKING:` body
    deferred: bool  # inside a function body


@dataclasses.dataclass(frozen=True)
class _FileImports:
    """One readable domain file's import statements; ``module`` is None for a tests file."""

    module: str | None
    is_package: bool
    statements: tuple[_ImportStatement, ...]


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == 'TYPE_CHECKING') or (
        isinstance(test, ast.Attribute) and test.attr == 'TYPE_CHECKING'
    )


def _nested_bodies(node: ast.stmt) -> Iterator[list[ast.stmt]]:
    """The statement lists directly inside *node*, in source order."""
    for _field, value in ast.iter_fields(node):
        if not isinstance(value, list) or not value:
            continue
        if isinstance(value[0], ast.stmt):
            yield value
        elif isinstance(value[0], ast.ExceptHandler | ast.match_case):
            for clause in value:
                yield clause.body


def _statements_in(
    body: Sequence[ast.stmt], *, runtime: bool, deferred: bool
) -> Iterator[_ImportStatement]:
    for node in body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            yield _ImportStatement(node, node.lineno, runtime=runtime, deferred=deferred)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            yield from _statements_in(node.body, runtime=runtime, deferred=True)
        elif isinstance(node, ast.If) and _is_type_checking(node.test):
            yield from _statements_in(node.body, runtime=False, deferred=deferred)
            yield from _statements_in(node.orelse, runtime=runtime, deferred=deferred)
        else:
            for nested in _nested_bodies(node):
                yield from _statements_in(nested, runtime=runtime, deferred=deferred)


def _import_statements(tree: ast.Module) -> tuple[_ImportStatement, ...]:
    """Every import statement in *tree*, in source order, with its context."""
    return tuple(_statements_in(tree.body, runtime=True, deferred=False))


def _from_base(importer: str, is_package: bool, level: int, module: str | None) -> str | None:
    """The absolute module a relative import names; None above the top package."""
    package = importer.split('.') if is_package else importer.split('.')[:-1]
    keep = len(package) - (level - 1)
    if keep <= 0:
        return None
    base = '.'.join(package[:keep])
    return f'{base}.{module}' if module else base


def _from_module(node: ast.ImportFrom, source: _FileImports) -> str | None:
    """The absolute module *node* imports from; a tests file's relative imports never resolve."""
    if node.level == 0:
        return node.module
    if source.module is None:
        return None
    return _from_base(source.module, source.is_package, node.level, node.module)


def _from_targets(node: ast.ImportFrom, resolved: str | None, known: frozenset[str]) -> list[str]:
    if resolved is None:
        return []
    found = []
    for alias in node.names:
        submodule = f'{resolved}.{alias.name}'
        if alias.name != '*' and submodule in known:
            found.append(submodule)
        elif resolved in known:
            found.append(resolved)
    return found


def _targets(
    statement: _ImportStatement, source: _FileImports, known: frozenset[str]
) -> tuple[str, ...]:
    """The first-party src modules *statement* imports, its own module excluded."""
    node = statement.node
    if isinstance(node, ast.Import):
        found = [alias.name for alias in node.names if alias.name in known]
    else:
        found = _from_targets(node, _from_module(node, source), known)
    return tuple(dict.fromkeys(target for target in found if target != source.module))


def _ancestor_packages(module: str, known: frozenset[str]) -> frozenset[str]:
    parts = module.split('.')
    prefixes = ('.'.join(parts[:count]) for count in range(1, len(parts)))
    return frozenset(prefix for prefix in prefixes if prefix in known)


def _reach_back(
    statement: _ImportStatement, module: str, source: _FileImports, known: frozenset[str]
) -> dict[str, Any] | None:
    """A from-import of names (not submodules) out of an ancestor package, or None."""
    node = statement.node
    if not isinstance(node, ast.ImportFrom):
        return None
    resolved = _from_module(node, source)
    if resolved is None or resolved not in _ancestor_packages(module, known):
        return None
    names = sorted({alias.name for alias in node.names if f'{resolved}.{alias.name}' not in known})
    if not names:
        return None
    return {'from': module, 'to': resolved, 'names': names, 'line': statement.line}


def _imported_names(statement: _ImportStatement, source: _FileImports) -> list[str]:
    """The dotted names *statement* imports, as resolved as they can be."""
    node = statement.node
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    resolved = _from_module(node, source)
    if resolved is not None:
        return [f'{resolved}.{alias.name}' for alias in node.names]
    prefix = '.' * node.level + (f'{node.module}.' if node.module else '')
    return [f'{prefix}{alias.name}' for alias in node.names]


def _deferred_entry(
    statement: _ImportStatement, module: str, source: _FileImports
) -> dict[str, Any]:
    return {
        'from': module,
        'line': statement.line,
        'imports': sorted(_imported_names(statement, source)),
    }


def _postorder(successors: Mapping[str, Sequence[str]]) -> list[str]:
    """Every node in DFS finishing order, iteratively: the real graph is deep."""
    seen: set[str] = set()
    order: list[str] = []
    for start in sorted(successors):
        if start in seen:
            continue
        seen.add(start)
        work = [(start, iter(successors[start]))]
        while work:
            node, children = work[-1]
            child = next((child for child in children if child not in seen), None)
            if child is None:
                work.pop()
                order.append(node)
            else:
                seen.add(child)
                work.append((child, iter(successors[child])))
    return order


def _reachable(adjacency: Mapping[str, Sequence[str]], start: str, excluded: set[str]) -> set[str]:
    found = {start}
    frontier = [start]
    while frontier:
        for neighbour in adjacency[frontier.pop()]:
            if neighbour not in found and neighbour not in excluded:
                found.add(neighbour)
                frontier.append(neighbour)
    return found


def _cycles(edges: Iterable[tuple[str, str]]) -> tuple[tuple[str, ...], ...]:
    """Strongly connected components of size > 1 (Kosaraju), each sorted, sorted."""
    successors: dict[str, list[str]] = {}
    predecessors: dict[str, list[str]] = {}
    for source, target in sorted(set(edges)):
        for node in (source, target):
            successors.setdefault(node, [])
            predecessors.setdefault(node, [])
        successors[source].append(target)
        predecessors[target].append(source)
    assigned: set[str] = set()
    components: list[tuple[str, ...]] = []
    for node in reversed(_postorder(successors)):
        if node in assigned:
            continue
        component = _reachable(predecessors, node, assigned)
        assigned |= component
        if len(component) > 1:
            components.append(tuple(sorted(component)))
    return tuple(sorted(components))


def _module_edges(
    module: str, source: _FileImports, known: frozenset[str], *, runtime_only: bool
) -> set[tuple[str, str]]:
    return {
        (module, target)
        for statement in source.statements
        if not runtime_only or (statement.runtime and not statement.deferred)
        for target in _targets(statement, source, known)
    }


@dataclasses.dataclass(frozen=True)
class _ImportGraph:
    edges: tuple[tuple[str, str], ...]
    reach_back: tuple[Mapping[str, Any], ...]
    deferred: tuple[Mapping[str, Any], ...]
    cycles: tuple[tuple[str, ...], ...]
    field_counts: Mapping[str, Mapping[str, int]]  # src record field -> module -> count

    def file_fields(self, module: str) -> dict[str, int]:
        """The graph fields of src *module*'s file record."""
        return {field: counts.get(module, 0) for field, counts in self.field_counts.items()}

    def to_json(self) -> dict[str, list[Any]]:
        return {
            'edges': [list(edge) for edge in self.edges],
            'reach_back': [dict(entry) for entry in self.reach_back],
            'deferred': [dict(entry) for entry in self.deferred],
            'cycles': [list(cycle) for cycle in self.cycles],
        }


def _tests_importers(sources: Iterable[_FileImports], known: frozenset[str]) -> dict[str, int]:
    """How many distinct tests files import each src module (absolute imports only)."""
    importers: Counter[str] = Counter()
    for source in sources:
        if source.module is None:
            importers.update(
                {target for statement in source.statements for target in _targets(statement, source, known)}
            )
    return dict(importers)


def _import_graph(sources: Sequence[_FileImports], known: frozenset[str]) -> _ImportGraph:
    """The graph over src modules; tests files only count toward fan_in_tests."""
    edges: set[tuple[str, str]] = set()
    runtime_edges: set[tuple[str, str]] = set()
    reach_back: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    for source in sources:
        module = source.module
        if module is None:
            continue
        edges |= _module_edges(module, source, known, runtime_only=False)
        runtime_edges |= _module_edges(module, source, known, runtime_only=True)
        for statement in source.statements:
            if (entry := _reach_back(statement, module, source, known)) is not None:
                reach_back.append(entry)
            if statement.deferred:
                deferred.append(_deferred_entry(statement, module, source))
    return _ImportGraph(
        edges=tuple(sorted(edges)),
        reach_back=tuple(sorted(reach_back, key=lambda e: (e['from'], e['line'], e['to']))),
        deferred=tuple(sorted(deferred, key=lambda e: (e['from'], e['line']))),
        cycles=_cycles(runtime_edges),
        field_counts={
            'reach_back_imports': Counter(entry['from'] for entry in reach_back),
            'fan_out': Counter(source for source, _target in edges),
            'fan_in_src': Counter(target for _source, target in edges),
            'fan_in_tests': _tests_importers(sources, known),
        },
    )


# ---------------------------------------------------------------------------
# Measuring the domain.


@dataclasses.dataclass(frozen=True)
class Measurement:
    """Every readable domain file's record, the src functions, and the import graph."""

    files: Mapping[str, Mapping[str, Any]]
    functions: Mapping[str, int]
    import_graph: Mapping[str, Any]
    unreadable: tuple[str, ...]


def _cognitive_max(per_function: Mapping[str, int]) -> tuple[int, str | None]:
    """The highest function score, and the smallest qualname holding it."""
    if not per_function:
        return 0, None
    top = max(per_function.values())
    return top, min(name for name, score in per_function.items() if score == top)


def _is_package_init(path: str) -> bool:
    return path.endswith('/__init__.py')


def _src_fields(domain_file: source_measures.DomainFile, tree: ast.Module) -> dict[str, Any]:
    return {
        'module': domain_file.import_name,
        'package_init': _is_package_init(domain_file.path),
        'function_local_imports': source_measures.function_local_imports_in_tree(tree),
        'reexport_names': sorted(source_measures.reexport_names_in_tree(tree)),
    }


def _below_top_level_is_private(dotted: str) -> bool:
    """Any segment after the top-level package is single-underscore."""
    return any(
        segment.startswith('_') and not segment.startswith('__')
        for segment in dotted.split('.')[1:]
    )


def _tests_fields(tree: ast.Module, known: frozenset[str]) -> dict[str, Any]:
    patched = (
        f'{target.module}.{target.leaf}'
        for target in source_measures.patch_targets_in_tree(tree, known)
    )
    return {
        'private_patch_targets': sorted(
            {dotted for dotted in patched if _below_top_level_is_private(dotted)}
        ),
        'private_reads': source_measures.private_reads_in_tree(tree),
    }


@dataclasses.dataclass(frozen=True)
class _FileMeasure:
    """One file's record before the graph fields, its function scores, and its imports."""

    record: Mapping[str, Any]
    per_function: Mapping[str, int]
    imports: _FileImports


def _measure_file(
    root: Path, member: str, domain_file: source_measures.DomainFile, known: frozenset[str]
) -> _FileMeasure:
    """One file's measures, from one read and one parse; *known* is every src module name."""
    path = domain_file.path
    source = source_measures.read_source(root, path)
    tree = source_measures.parse_source(source, path=path)
    size = source_measures.file_size_measures_in_tree(source, tree, path=path)
    cognitive = source_measures.file_cognitive_measures(root / path)
    per_function = {name: int(score) for name, score in cognitive.per_function.items()}
    cognitive_max, cognitive_max_function = _cognitive_max(per_function)
    record: dict[str, Any] = {
        'member': member,
        'kind': str(domain_file.kind),
        'blob': domain_file.blob,
        'lines': size.lines,
        'prose_lines': size.prose_lines,
        'prose_ratio': None if size.lines == 0 else round(size.prose_lines / size.lines, 4),
        'cognitive_total': cognitive.total,
        'cognitive_max': cognitive_max,
        'cognitive_max_function': cognitive_max_function,
        'functions': len(per_function),
    }
    if domain_file.kind is source_measures.FileKind.SRC:
        record.update(_src_fields(domain_file, tree))
    else:
        record.update(_tests_fields(tree, known))
    imports = _FileImports(
        module=domain_file.import_name,
        is_package=_is_package_init(path),
        statements=_import_statements(tree),
    )
    return _FileMeasure(record=record, per_function=per_function, imports=imports)


def _file_records(
    measured: Mapping[str, _FileMeasure], graph: _ImportGraph
) -> dict[str, Mapping[str, Any]]:
    """Every record in path order, a src record completed with its graph fields."""
    return {
        path: (
            measure.record
            if measure.imports.module is None
            else {**measure.record, **graph.file_fields(measure.imports.module)}
        )
        for path, measure in sorted(measured.items())
    }


def _src_functions(measured: Mapping[str, _FileMeasure]) -> dict[str, int]:
    return dict(sorted(
        (f'{path}::{name}', score)
        for path, measure in measured.items()
        if measure.imports.module is not None
        for name, score in measure.per_function.items()
    ))


def _require_unique_module_names(domain: Sequence[source_measures.DomainMember]) -> None:
    """The import graph is keyed by module name, so one name must mean one file."""
    paths_by_name: dict[str, list[str]] = {}
    for member in domain:
        for domain_file in member.files:
            if domain_file.import_name is not None:
                paths_by_name.setdefault(domain_file.import_name, []).append(domain_file.path)
    collisions = {name: sorted(paths) for name, paths in paths_by_name.items() if len(paths) > 1}
    if collisions:
        raise source_measures.MetricsError(
            'src files that import as one module name would make the import graph '
            'ambiguous: '
            + '; '.join(f'{name!r}: {", ".join(paths)}' for name, paths in sorted(collisions.items()))
        )


def measure_domain(
    root: Path, domain: Sequence[source_measures.DomainMember]
) -> Measurement:
    """Measure every file of *domain* under *root*, each read and parsed once.

    A file whose read, parse, tokenize or complexipy step fails is named in
    ``unreadable`` (its reason on stderr) and has no record.
    """
    _require_unique_module_names(domain)
    known = frozenset(
        domain_file.import_name
        for member in domain
        for domain_file in member.files
        if domain_file.import_name is not None
    )
    measured: dict[str, _FileMeasure] = {}
    unreadable: list[str] = []
    for member in domain:
        for domain_file in member.files:
            try:
                measured[domain_file.path] = _measure_file(
                    root, member.name, domain_file, known
                )
            except source_measures.MetricsError as exc:
                unreadable.append(domain_file.path)
                print(f'unreadable: {domain_file.path}: {exc}', file=sys.stderr)
    graph = _import_graph([measure.imports for measure in measured.values()], known)
    return Measurement(
        files=_file_records(measured, graph),
        functions=_src_functions(measured),
        import_graph=graph.to_json(),
        unreadable=tuple(sorted(unreadable)),
    )


def _evidence(
    domain: Sequence[source_measures.DomainMember], measured: Measurement
) -> dict[str, Any]:
    return {
        'members': [
            {'name': member.name, 'pseudo': member.pseudo, 'domain_files': len(member.files)}
            for member in domain
        ],
        'domain_files': sum(len(member.files) for member in domain),
        'measured_files': len(measured.files),
        'unreadable': list(measured.unreadable),
        'complete': not measured.unreadable,
    }


def _require_clean_domain(root: Path, as_of: str) -> None:
    dirty = source_measures.uncommitted_domain_paths(root)
    if dirty:
        raise source_measures.MetricsError(
            f'the domain has uncommitted changes, so HEAD ({as_of}) is not what the '
            f'work tree holds: {", ".join(dirty)}; commit, or measure a clean checkout'
        )


def _require_unmoved_head(root: Path, as_of: str) -> None:
    now = source_measures.head_commit(root)
    if now != as_of:
        raise source_measures.MetricsError(
            f'HEAD moved from {as_of} to {now} during the measurement (a merge '
            'landed?); re-run, or measure a pinned worktree'
        )


def take_snapshot(root: Path, *, run_id: str, since: str) -> dict[str, Any]:
    """Measure *root*'s HEAD as a schema-1 snapshot; refuse a dirty domain or a moved HEAD."""
    started = time.monotonic()
    version = source_measures.require_complexipy()
    as_of = source_measures.head_commit(root)
    domain = source_measures.workspace_domain(root)
    _require_clean_domain(root, as_of)
    measured = measure_domain(root, domain)
    _require_unmoved_head(root, as_of)
    _require_clean_domain(root, as_of)
    snapshot = {
        'schema_version': SCHEMA_VERSION,
        'instrument': INSTRUMENT,
        'run_id': run_id,
        'as_of_sha': as_of,
        'since': since,
        'evidence': _evidence(domain, measured),
        'cost': {'wall_clock_s': round(time.monotonic() - started, 1)},
        'params': {
            'complexipy_version': version,
            'h14_soft_ceiling_lines': H14_SOFT_CEILING_LINES,
            'h14_alarm_lines': H14_ALARM_LINES,
        },
        'files': {path: dict(record) for path, record in measured.files.items()},
        'functions': dict(measured.functions),
        'import_graph': dict(measured.import_graph),
    }
    return validate_snapshot(snapshot, origin='this measurement')


# ---------------------------------------------------------------------------
# The snapshot file: rendering, loading, and the one shape check.


def _render_mapping(name: str, mapping: Mapping[str, Any]) -> str:
    """One line per entry, in sorted key order."""
    if not mapping:
        return f'  {json.dumps(name)}: {{}}'
    rows = ',\n'.join(
        f'    {json.dumps(key)}: {json.dumps(mapping[key], sort_keys=True)}'
        for key in sorted(mapping)
    )
    return f'  {json.dumps(name)}: {{\n{rows}\n  }}'


def _render_list(name: str, items: Sequence[Any]) -> str:
    """One line per element, in the list's own order."""
    if not items:
        return f'    {json.dumps(name)}: []'
    rows = ',\n'.join(f'      {json.dumps(item, sort_keys=True)}' for item in items)
    return f'    {json.dumps(name)}: [\n{rows}\n    ]'


def _render_import_graph(graph: Mapping[str, Sequence[Any]]) -> str:
    lists = ',\n'.join(_render_list(key, graph[key]) for key in _IMPORT_GRAPH_KEYS)
    return f'  "import_graph": {{\n{lists}\n  }}'


def render_snapshot(snapshot: Mapping[str, Any]) -> str:
    """The snapshot file's exact text; rendering a parsed rendering reproduces it."""
    entries = []
    for key, value in snapshot.items():
        if key in ('files', 'functions'):
            entries.append(_render_mapping(key, value))
        elif key == 'import_graph':
            entries.append(_render_import_graph(value))
        else:
            block = json.dumps(value, indent=2).replace('\n', '\n  ')
            entries.append(f'  {json.dumps(key)}: {block}')
    return '{\n' + ',\n'.join(entries) + '\n}\n'


def load_snapshot(path: Path) -> dict[str, Any]:
    """Read and validate the snapshot file at *path*."""
    try:
        text = path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        raise source_measures.MetricsError(
            f'{path}: could not be read -- {exc.__class__.__name__}: {exc}'
        ) from exc
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise source_measures.MetricsError(f'{path}: is not JSON -- {exc}') from exc
    return validate_snapshot(parsed, origin=str(path))


def _invalid(origin: str, where: str, found: object, expected: object) -> source_measures.MetricsError:
    return source_measures.MetricsError(
        f'{origin}: {where} is {found!r}; schema {SCHEMA_VERSION} expects {expected}'
    )


def _object_at(snapshot: Mapping[str, Any], key: str, origin: str) -> dict[str, Any]:
    value = snapshot[key]
    if not isinstance(value, dict):
        raise _invalid(origin, key, type(value).__name__, 'a JSON object')
    return value


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_record(path: str, record: object, origin: str) -> None:
    where = f'files[{path!r}]'
    if not isinstance(record, dict):
        raise _invalid(origin, where, type(record).__name__, 'a JSON object')
    kind = record.get('kind')
    if not isinstance(kind, str) or kind not in _KIND_FIELDS:
        raise _invalid(origin, f'{where}.kind', kind, sorted(_KIND_FIELDS))
    expected = _COMMON_FIELDS | _KIND_FIELDS[kind]
    if set(record) != expected:
        raise _invalid(origin, f'{where} keys', sorted(record), sorted(expected))
    for field in sorted(_INTEGER_FIELDS & expected):
        if not _is_int(record[field]):
            raise _invalid(origin, f'{where}.{field}', record[field], 'an integer')


def validate_snapshot(snapshot: object, *, origin: str) -> dict[str, Any]:
    """*snapshot*, once it is shown to have the schema-1 shape; else MetricsError naming *origin*."""
    if not isinstance(snapshot, dict):
        raise _invalid(origin, 'the top level', type(snapshot).__name__, 'a JSON object')
    if snapshot.get('instrument') != INSTRUMENT:
        raise _invalid(origin, 'instrument', snapshot.get('instrument'), repr(INSTRUMENT))
    version = snapshot.get('schema_version')
    if not _is_int(version) or version != SCHEMA_VERSION:
        raise _invalid(origin, 'schema_version', version, SCHEMA_VERSION)
    if tuple(snapshot) != _TOP_LEVEL_KEYS:
        raise _invalid(origin, 'the top-level key order', list(snapshot), list(_TOP_LEVEL_KEYS))
    evidence = _object_at(snapshot, 'evidence', origin)
    if tuple(evidence) != _EVIDENCE_KEYS:
        raise _invalid(origin, 'evidence keys', list(evidence), list(_EVIDENCE_KEYS))
    for path, record in _object_at(snapshot, 'files', origin).items():
        _validate_record(path, record, origin)
    graph = _object_at(snapshot, 'import_graph', origin)
    if set(graph) != set(_IMPORT_GRAPH_KEYS):
        raise _invalid(origin, 'import_graph keys', sorted(graph), sorted(_IMPORT_GRAPH_KEYS))
    return snapshot


# ---------------------------------------------------------------------------
# --diff: what moved between two validated snapshots, in the contract's order.

#: The quality doc's two readings of a complexity pair (docs/code-quality.md §What to measure).
_PAIR_MOVED = (
    '; max down, total flat or down: complexity moved, per docs/code-quality.md §What to measure'
)
_PAIR_ADDED = '; total up: complexity added'
_CROSSING_LABEL = 'measure per heuristic 14, not a target'


@dataclasses.dataclass(frozen=True)
class _Pairing:
    """How the two snapshots' domain paths correspond."""

    added: tuple[str, ...]
    removed: tuple[str, ...]
    renamed: tuple[tuple[str, str], ...]
    compared: tuple[tuple[str, str], ...]  # (previous, current), measured on both sides


def _domain_paths(snapshot: Mapping[str, Any]) -> set[str]:
    return set(snapshot['files']) | set(snapshot['evidence']['unreadable'])


def _renames(
    gone: Iterable[str], arrived: Iterable[str], previous: Mapping[str, Any], current: Mapping[str, Any]
) -> tuple[tuple[str, str], ...]:
    """Measured paths that left and arrived with one blob, paired in sorted order."""
    gone_by_blob: dict[str, list[str]] = {}
    for path in sorted(gone):
        gone_by_blob.setdefault(previous['files'][path]['blob'], []).append(path)
    arrived_by_blob: dict[str, list[str]] = {}
    for path in sorted(arrived):
        arrived_by_blob.setdefault(current['files'][path]['blob'], []).append(path)
    pairs = [
        pair
        for blob in gone_by_blob.keys() & arrived_by_blob.keys()
        for pair in zip(gone_by_blob[blob], arrived_by_blob[blob], strict=False)
    ]
    return tuple(sorted(pairs))


def _pairing(current: Mapping[str, Any], previous: Mapping[str, Any]) -> _Pairing:
    current_domain, previous_domain = _domain_paths(current), _domain_paths(previous)
    renamed = _renames(
        (path for path in previous['files'] if path not in current_domain),
        (path for path in current['files'] if path not in previous_domain),
        previous,
        current,
    )
    both = [(path, path) for path in set(previous['files']) & set(current['files'])]
    return _Pairing(
        added=tuple(sorted(current_domain - previous_domain - {new for _old, new in renamed})),
        removed=tuple(sorted(previous_domain - current_domain - {old for old, _new in renamed})),
        renamed=renamed,
        compared=tuple(sorted([*both, *renamed], key=lambda pair: pair[1])),
    )


_RecordPair = tuple[str, Mapping[str, Any] | None, Mapping[str, Any] | None]


def _record_pairs(
    pairing: _Pairing, current: Mapping[str, Any], previous: Mapping[str, Any]
) -> list[_RecordPair]:
    """(path, previous record, current record) in path order; None for an absent side.

    A path unreadable on either side has no pair: its measures are unknown.
    """
    now, before = current['files'], previous['files']
    pairs: list[_RecordPair] = [(new, before[old], now[new]) for old, new in pairing.compared]
    pairs += [(path, None, now[path]) for path in pairing.added if path in now]
    pairs += [(path, before[path], None) for path in pairing.removed if path in before]
    return sorted(pairs, key=lambda pair: pair[0])


def _header(label: str, snapshot: Mapping[str, Any]) -> str:
    complete = json.dumps(snapshot['evidence']['complete'])
    return f'{label}: {snapshot["run_id"]} as_of {snapshot["as_of_sha"]} complete={complete}'


def _unreadable_entries(current: Mapping[str, Any], previous: Mapping[str, Any]) -> list[str]:
    return [
        *(f'current: {path}' for path in current['evidence']['unreadable']),
        *(f'previous: {path}' for path in previous['evidence']['unreadable']),
    ]


def _pair_entry(path: str, before: Mapping[str, Any], after: Mapping[str, Any]) -> str | None:
    old_max, old_total = before['cognitive_max'], before['cognitive_total']
    new_max, new_total = after['cognitive_max'], after['cognitive_total']
    if (old_max, old_total) == (new_max, new_total):
        return None
    if new_max < old_max and new_total <= old_total:
        reading = _PAIR_MOVED
    elif new_total > old_total:
        reading = _PAIR_ADDED
    else:
        reading = ''
    return f'{path}: max {old_max} -> {new_max}, total {old_total} -> {new_total}{reading}'


def _pair_entries(pairs: Sequence[_RecordPair]) -> list[str]:
    entries = (
        _pair_entry(path, before, after)
        for path, before, after in pairs
        if before is not None and after is not None
    )
    return [entry for entry in entries if entry is not None]


def _crossing_entries(pairs: Sequence[_RecordPair]) -> list[str]:
    entries = []
    for path, before, after in pairs:
        old = before['lines'] if before is not None else 0
        new = after['lines'] if after is not None else 0
        for mark, prefix in ((H14_SOFT_CEILING_LINES, ''), (H14_ALARM_LINES, 'ALARM ')):
            if (old >= mark) != (new >= mark):
                direction = 'up' if new >= mark else 'down'
                entries.append(
                    f'{path}: {old} -> {new} lines, {prefix}crossed {mark} {direction}; '
                    f'{_CROSSING_LABEL}'
                )
    return entries


#: (label, import_graph list, comparison key without line numbers, rendering of a key).
_GRAPH_KINDS: tuple[tuple[str, str, Any, Any], ...] = (
    ('edge', 'edges', lambda edge: tuple(edge), lambda key: f'{key[0]} -> {key[1]}'),
    (
        'reach-back',
        'reach_back',
        lambda entry: (entry['from'], entry['to'], tuple(entry['names'])),
        lambda key: f'{key[0]} -> {key[1]} ({", ".join(key[2])})',
    ),
    (
        'deferred',
        'deferred',
        lambda entry: (entry['from'], tuple(entry['imports'])),
        lambda key: f'{key[0]}: {", ".join(key[1])}',
    ),
    ('cycle', 'cycles', lambda cycle: tuple(cycle), lambda key: ', '.join(key)),
)


def _graph_change_entries(current: Mapping[str, Any], previous: Mapping[str, Any]) -> list[str]:
    entries = []
    for label, name, key_of, render in _GRAPH_KINDS:
        before = Counter(key_of(item) for item in previous['import_graph'][name])
        after = Counter(key_of(item) for item in current['import_graph'][name])
        entries += [f'{label} added: {render(key)}' for key in sorted((after - before).elements())]
        entries += [f'{label} removed: {render(key)}' for key in sorted((before - after).elements())]
    return entries


def _change_parts(old: Iterable[str], new: Iterable[str]) -> list[str]:
    old_set, new_set = set(old), set(new)
    return [*(f'+{name}' for name in sorted(new_set - old_set)), *(f'-{name}' for name in sorted(old_set - new_set))]


def _reexport_entries(pairs: Sequence[_RecordPair]) -> list[str]:
    entries = []
    for path, before, after in pairs:
        if before is None or after is None or 'reexport_names' not in before:
            continue
        parts = _change_parts(before['reexport_names'], after['reexport_names'])
        if parts:
            entries.append(f're-export names: {path}: {", ".join(parts)}')
    return entries


_NO_COUPLING: Mapping[str, Any] = {'private_patch_targets': [], 'private_reads': 0}


def _test_file_entries(pairs: Sequence[_RecordPair]) -> list[str]:
    entries = []
    for path, before, after in pairs:
        if _TESTS not in ((before or {}).get('kind'), (after or {}).get('kind')):
            continue
        old, new = before or _NO_COUPLING, after or _NO_COUPLING
        parts = []
        if targets := _change_parts(old['private_patch_targets'], new['private_patch_targets']):
            parts.append(', '.join(targets))
        if old['private_reads'] != new['private_reads']:
            parts.append(f'private reads {old["private_reads"]} -> {new["private_reads"]}')
        if parts:
            entries.append(f'{path}: {"; ".join(parts)}')
    return entries


def _diff_section(header: str, entries: Sequence[str]) -> list[str]:
    """A header, then its entries indented; an empty section says so."""
    return [header, *(f'  {entry}' for entry in entries or ['(none)'])]


def diff_lines(current: Mapping[str, Any], previous: Mapping[str, Any]) -> list[str]:
    """What moved from *previous* to *current*: measures and their changes, never a verdict."""
    pairing = _pairing(current, previous)
    pairs = _record_pairs(pairing, current, previous)
    return [
        _header('current', current),
        _header('previous', previous),
        *_diff_section('unreadable (measures unknown, never zero):', _unreadable_entries(current, previous)),
        *_diff_section('files added:', pairing.added),
        *_diff_section('files removed:', pairing.removed),
        *_diff_section('files renamed (same blob):', [f'{old} -> {new}' for old, new in pairing.renamed]),
        *_diff_section('complexity pair (max per function, module total):', _pair_entries(pairs)),
        *_diff_section('heuristic-14 crossings (measure, not fix):', _crossing_entries(pairs)),
        *_diff_section(
            'import graph:',
            [*_graph_change_entries(current, previous), *_reexport_entries(pairs)],
        ),
        *_diff_section('test files (private patch targets, private reads):', _test_file_entries(pairs)),
    ]


# ---------------------------------------------------------------------------
# The command line.


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Measure every workspace member\'s tracked Python at HEAD as a schema-1 '
            'snapshot (plans/quality-metrics-snapshot-prd.md), diff two snapshots, '
            'or print one as a per-member table. A report, never a gate.'
        ),
        epilog=(
            'Exit codes: 0 done; 2 instrument failure (git or complexipy missing, '
            'a dirty domain, a moved HEAD, an unreadable snapshot) or a misused '
            'command line. There is no exit 1: a report has nothing to fail.'
        ),
    )
    parser.add_argument('--run-id', help='The measuring run\'s id, recorded as run_id.')
    parser.add_argument('--out', help='Where to write the snapshot.')
    parser.add_argument(
        '--root', help=f'The checkout to measure (default: {_DEFAULT_ROOT}).'
    )
    parser.add_argument('--diff', help='The previous snapshot to diff against.')
    parser.add_argument('--current', help='Diff this snapshot instead of measuring.')
    parser.add_argument('--summary', help='Print this snapshot as a per-member table.')
    return parser


def _require_one_mode(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    measuring = (args.run_id, args.out, args.root) != (None, None, None)
    if args.summary is not None:
        if measuring or args.diff is not None or args.current is not None:
            parser.error('--summary takes no other argument')
    elif args.current is not None:
        if measuring or args.diff is None:
            parser.error('--current needs --diff, and takes no --run-id, --out or --root')
    elif args.run_id is None or args.out is None:
        parser.error(
            'measuring needs both --run-id and --out; '
            'otherwise give --current with --diff, or --summary'
        )


def _write(out: Path, text: str) -> None:
    try:
        safe_io.atomic_write_text(out, text, mkdir=True)
    except OSError as exc:
        raise source_measures.MetricsError(
            f'{out}: could not be written -- {exc.__class__.__name__}: {exc}'
        ) from exc


def _measure(args: argparse.Namespace) -> int:
    root = Path(args.root) if args.root is not None else _DEFAULT_ROOT
    previous = load_snapshot(Path(args.diff)) if args.diff is not None else None
    since = previous['as_of_sha'] if previous is not None else 'none'
    snapshot = take_snapshot(root, run_id=args.run_id, since=since)
    out = Path(args.out)
    _write(out, render_snapshot(snapshot))
    evidence = snapshot['evidence']
    print(
        f'wrote {out}: {evidence["measured_files"]}/{evidence["domain_files"]} domain '
        f'files measured, complete={json.dumps(evidence["complete"])}'
    )
    if previous is None:
        print(NO_PREVIOUS_LINE)
    else:
        print('\n'.join(diff_lines(snapshot, previous)))
    return 0


def _compare(args: argparse.Namespace) -> int:
    current = load_snapshot(Path(args.current))
    previous = load_snapshot(Path(args.diff))
    print('\n'.join(diff_lines(current, previous)))
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; see the module docstring for the exit codes."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _require_one_mode(parser, args)
    try:
        if args.current is not None:
            return _compare(args)
        return _measure(args)
    except source_measures.MetricsError as exc:
        print(f'quality_metrics_snapshot: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
