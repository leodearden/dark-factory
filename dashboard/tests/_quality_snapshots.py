"""Quality-metrics snapshot files for the dashboard tests: minimal documents the instrument accepts.

Builders only; tests hold the expectations. ``snapshot_doc`` returns a schema-1, -2 or -3 snapshot
in the shape scripts/quality_metrics_snapshot.py::validate_snapshot checks, so a test reading one
through the REAL instrument proves the fixture as well as the reader.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SOFT_CEILING_LINES = 1500
ALARM_LINES = 2000


@dataclass(frozen=True)
class FileFigures:
    """One file record's chosen figures; every other measure is a fixed small value."""

    kind: str  # 'src' or 'tests'
    lines: int
    private_patch_targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemberFiles:
    pseudo: bool
    files: Mapping[str, FileFigures]


DEFAULT_MEMBERS: Mapping[str, MemberFiles] = {
    'alpha': MemberFiles(False, {
        'alpha/src/alpha/big.py': FileFigures('src', 2100),
        'alpha/src/alpha/mod.py': FileFigures('src', 40),
        'alpha/tests/test_mod.py': FileFigures('tests', 1600, ('alpha.mod._x', 'alpha.mod._y')),
    }),
    'beta': MemberFiles(False, {
        'beta/src/beta/b.py': FileFigures('src', 1700),
        'beta/tests/test_b.py': FileFigures('tests', 12, ('beta.b._z',)),
    }),
    'scripts': MemberFiles(True, {'scripts/x.py': FileFigures('src', 10)}),
}

#: The schema-3 import graph: two cycles, one hidden and one typing cycle, three deferred imports
#: of which two close a cycle.
DEFAULT_GRAPH: Mapping[str, list[Any]] = {
    'edges': [['alpha.a', 'alpha.b'], ['alpha.b', 'alpha.a'], ['beta.c', 'beta.d'], ['beta.d', 'beta.c']],
    'reach_back': [],
    'deferred': [
        {'from': 'alpha.e', 'line': 3, 'imports': ['alpha.f'], 'closes_cycle': True},
        {'from': 'alpha.g', 'line': 4, 'imports': ['alpha.a'], 'closes_cycle': False},
        {'from': 'beta.h', 'line': 5, 'imports': ['beta.i'], 'closes_cycle': True},
    ],
    'cycles': [['alpha.a', 'alpha.b'], ['beta.c', 'beta.d']],
    'hidden_cycles': [['alpha.e', 'alpha.f']],
    'typing_cycles': [['beta.j', 'beta.k']],
}


def default_params(schema: int) -> dict[str, Any]:
    """The params a schema-*schema* snapshot records; schema 3 adds python_version and instrument."""
    params: dict[str, Any] = {
        'complexipy_version': '6.2.0',
        'h14_soft_ceiling_lines': SOFT_CEILING_LINES,
        'h14_alarm_lines': ALARM_LINES,
    }
    if schema >= 3:
        params['python_version'] = '3.13.9'
        params['instrument'] = {'quality_metrics_snapshot': 'a' * 40, 'source_measures': 'b' * 40}
    return params


def _record(member: str, figures: FileFigures, schema: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        'member': member, 'kind': figures.kind, 'blob': '0' * 40, 'lines': figures.lines,
        'prose_lines': 0, 'prose_ratio': None, 'cognitive_total': 1, 'cognitive_max': 1,
        'cognitive_max_function': None, 'functions': 1,
    }
    if figures.kind == 'tests':
        return {**record, 'private_patch_targets': list(figures.private_patch_targets), 'private_reads': 0}
    record.update({
        'module': member, 'package_init': False, 'function_local_imports': 0, 'reexport_names': [],
        'reach_back_imports': 0, 'fan_out': 0, 'fan_in_src': 0, 'fan_in_tests': 0,
    })
    if schema >= 2:
        record['cycle_closing_imports'] = 0
    return record


def _graph(schema: int) -> dict[str, Any]:
    if schema >= 2:
        return {key: list(value) for key, value in DEFAULT_GRAPH.items()}
    return {
        'edges': list(DEFAULT_GRAPH['edges']),
        'reach_back': [],
        'deferred': [
            {key: value for key, value in entry.items() if key != 'closes_cycle'}
            for entry in DEFAULT_GRAPH['deferred']
        ],
        'cycles': list(DEFAULT_GRAPH['cycles']),
    }


def snapshot_doc(
    schema: int,
    *,
    as_of_sha: str,
    run_id: str,
    params: Mapping[str, Any] | None = None,
    members: Mapping[str, MemberFiles] = DEFAULT_MEMBERS,
) -> dict[str, Any]:
    """A minimal schema-*schema* snapshot the instrument's validate_snapshot accepts."""
    files = {
        path: _record(name, figures, schema)
        for name, member in members.items()
        for path, figures in member.files.items()
    }
    return {
        'schema_version': schema,
        'instrument': 'quality-metrics-snapshot',
        'run_id': run_id,
        'as_of_sha': as_of_sha,
        'since': 'none',
        'evidence': {
            'members': [
                {'name': name, 'pseudo': member.pseudo, 'domain_files': len(member.files)}
                for name, member in members.items()
            ],
            'domain_files': len(files),
            'measured_files': len(files),
            'unreadable': [],
            'complete': True,
        },
        'cost': {'wall_clock_s': 1.0},
        'params': dict(params) if params is not None else default_params(schema),
        'files': files,
        'functions': {},
        'import_graph': _graph(schema),
    }


def write_snapshot(path: Path, doc: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1), encoding='utf-8')
    return path
