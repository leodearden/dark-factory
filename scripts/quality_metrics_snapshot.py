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
import dataclasses
import json
import sys
import time
from collections.abc import Mapping, Sequence
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
    _SRC: frozenset({'module', 'package_init', 'function_local_imports', 'reexport_names'}),
    _TESTS: frozenset(),
}
_INTEGER_FIELDS = frozenset({
    'lines', 'prose_lines', 'cognitive_total', 'cognitive_max', 'functions',
    'function_local_imports',
})
_IMPORT_GRAPH_KEYS = ('edges', 'reach_back', 'deferred', 'cycles')


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


def _src_fields(domain_file: source_measures.DomainFile, tree: Any) -> dict[str, Any]:
    return {
        'module': domain_file.import_name,
        'package_init': domain_file.path.endswith('/__init__.py'),
        'function_local_imports': source_measures.function_local_imports_in_tree(tree),
        'reexport_names': sorted(source_measures.reexport_names_in_tree(tree)),
    }


def _measure_file(
    root: Path, member: str, domain_file: source_measures.DomainFile
) -> tuple[dict[str, Any], Mapping[str, int]]:
    """One file's record and its per-function scores, from one read and one parse."""
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
    return record, per_function


def _empty_import_graph() -> dict[str, list[Any]]:
    return {key: [] for key in _IMPORT_GRAPH_KEYS}


def measure_domain(
    root: Path, domain: Sequence[source_measures.DomainMember]
) -> Measurement:
    """Measure every file of *domain* under *root*, each read and parsed once.

    A file whose read, parse, tokenize or complexipy step fails is named in
    ``unreadable`` (its reason on stderr) and has no record.
    """
    files: dict[str, Mapping[str, Any]] = {}
    functions: dict[str, int] = {}
    unreadable: list[str] = []
    for member in domain:
        for domain_file in member.files:
            try:
                record, per_function = _measure_file(root, member.name, domain_file)
            except source_measures.MetricsError as exc:
                unreadable.append(domain_file.path)
                print(f'unreadable: {domain_file.path}: {exc}', file=sys.stderr)
                continue
            files[domain_file.path] = record
            if domain_file.kind is source_measures.FileKind.SRC:
                functions.update(
                    (f'{domain_file.path}::{name}', score) for name, score in per_function.items()
                )
    return Measurement(
        files=dict(sorted(files.items())),
        functions=dict(sorted(functions.items())),
        import_graph=_empty_import_graph(),
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


def take_snapshot(root: Path, *, run_id: str, since: str) -> dict[str, Any]:
    """Measure *root*'s HEAD as a schema-1 snapshot."""
    started = time.monotonic()
    version = source_measures.require_complexipy()
    as_of = source_measures.head_commit(root)
    domain = source_measures.workspace_domain(root)
    measured = measure_domain(root, domain)
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


def _section(snapshot: Mapping[str, Any], key: str, origin: str) -> dict[str, Any]:
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
    evidence = _section(snapshot, 'evidence', origin)
    if tuple(evidence) != _EVIDENCE_KEYS:
        raise _invalid(origin, 'evidence keys', list(evidence), list(_EVIDENCE_KEYS))
    for path, record in _section(snapshot, 'files', origin).items():
        _validate_record(path, record, origin)
    graph = _section(snapshot, 'import_graph', origin)
    if set(graph) != set(_IMPORT_GRAPH_KEYS):
        raise _invalid(origin, 'import_graph keys', sorted(graph), sorted(_IMPORT_GRAPH_KEYS))
    return snapshot


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
    snapshot = take_snapshot(root, run_id=args.run_id, since='none')
    out = Path(args.out)
    _write(out, render_snapshot(snapshot))
    evidence = snapshot['evidence']
    print(
        f'wrote {out}: {evidence["measured_files"]}/{evidence["domain_files"]} domain '
        f'files measured, complete={json.dumps(evidence["complete"])}'
    )
    print(NO_PREVIOUS_LINE)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; see the module docstring for the exit codes."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _require_one_mode(parser, args)
    try:
        return _measure(args)
    except source_measures.MetricsError as exc:
        print(f'quality_metrics_snapshot: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
