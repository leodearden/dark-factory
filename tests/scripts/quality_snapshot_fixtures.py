"""The test harness of ``scripts/quality_metrics_snapshot.py``: fixture repos measured and diffed through its real entry point.

Each fixture is a small REAL git repository whose root pyproject.toml names
two workspace members, committed and then measured (or measured, changed and
measured again with ``--diff``) through ``quality_metrics_snapshot.main``.
``as_schema_2`` and ``as_schema_1`` give what an older-schema instrument would
have recorded for the same tree.

IMPORT ME, DO NOT COPY ME. Importable from ``tests/scripts/test_*.py`` because
``tests/scripts/conftest.py`` puts this directory on ``sys.path``.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pytest
import quality_metrics_snapshot as snapshot
from git_listing import git

WORKSPACE_PYPROJECT = '[tool.uv.workspace]\nmembers = ["alpha", "beta"]\n'

#: The row-5 'before' source: complexipy 6.2 scores f at 2, and the file at 2.
MOD_BEFORE = (
    '"""The row-5 fixture."""\n'
    '# f holds two sequential ifs.\n'
    'def f(a, b):\n'
    '    if a:\n'
    '        x = 1\n'
    '    if b:\n'
    '        x = 2\n'
    '    return 0\n'
)

TRIVIAL_TEST = 'def test_x():\n    assert True\n'

BASE_FILES: dict[str, str] = {
    'alpha/src/alpha/__init__.py': '',
    'alpha/src/alpha/mod.py': MOD_BEFORE,
    'alpha/tests/test_mod.py': TRIVIAL_TEST,
    'beta/src/beta/b.py': 'B = 1\n',
    'beta/tests/test_b.py': TRIVIAL_TEST,
    'scripts/x.py': 'X = 1\n',
    'scripts/tests/test_x.py': TRIVIAL_TEST,
    'tests/test_y.py': TRIVIAL_TEST,
    'hooks/h.py': 'H = 1\n',
}

#: A package reach-back, a runtime cycle a <-> b, a TYPE_CHECKING-only c <-> d,
#: a function-local e -> a, a tests importer and a scripts pseudo-member.
GRAPH_FILES: dict[str, str] = {
    'alpha/src/pkg/__init__.py': 'THING = 1\n',
    'alpha/src/pkg/a.py': 'from pkg import THING\n',
    'alpha/src/pkg/b.py': 'from pkg import c\n',
    'alpha/src/pkg/c.py': 'X = 1\n',
    'alpha/src/a.py': 'import b\n',
    'alpha/src/b.py': 'import a\n',
    'alpha/src/c.py': 'from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import d\n',
    'alpha/src/d.py': 'from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import c\n',
    'alpha/src/e.py': 'import os\n\n\ndef f():\n    import a\n    return a\n',
    'beta/src/beta/b.py': 'import pkg.c\n',
    'alpha/tests/test_pkg.py': 'from pkg import a\nimport pkg.c\n',
    'scripts/x.py': 'import legibility.y\n',
    'scripts/legibility/y.py': 'Y = 1\n',
}

_COMMITTER = ('-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid')


def write_files(root: Path, files: dict[str, str]) -> None:
    for relpath, text in files.items():
        target = root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')


def committed_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    """A committed repo under *tmp_path*; a 'pyproject.toml' in *files* overrides the default."""
    root = tmp_path / 'repo'
    root.mkdir()
    git(root, 'init', '-q')
    write_files(root, {'pyproject.toml': WORKSPACE_PYPROJECT, **files})
    git(root, 'add', '-A')
    git(root, *_COMMITTER, 'commit', '--no-verify', '-q', '-m', 'base')
    return root


def commit(
    root: Path,
    message: str,
    *,
    write: dict[str, str] | None = None,
    remove: Iterable[str] = (),
    moves: Iterable[tuple[str, str]] = (),
) -> None:
    write_files(root, write or {})
    for relpath in remove:
        git(root, 'rm', '-q', relpath)
    for source, destination in moves:
        git(root, 'mv', source, destination)
    git(root, 'add', '-A')
    git(root, *_COMMITTER, 'commit', '--no-verify', '--allow-empty', '-q', '-m', message)


def run_snapshot(root: Path, out: Path, *extra: str) -> int:
    return snapshot.main(['--run-id', 'run-1', '--out', str(out), '--root', str(root), *extra])


def measured_snapshot(tmp_path: Path, files: dict[str, str]) -> dict[str, Any]:
    root = committed_repo(tmp_path, files)
    out = tmp_path / 'out' / 'snapshot.json'
    assert run_snapshot(root, out) == 0
    return json.loads(out.read_text(encoding='utf-8'))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def diff_chain(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    base_files: dict[str, str],
    *changes: Any,
) -> list[tuple[dict[str, Any], list[str]]]:
    """Measure the base commit, then each change in turn with --diff against the one before.

    Each entry is (snapshot, the diff lines its measuring run printed after the
    'wrote' line); the --current form is checked to print the same lines.
    """
    root = committed_repo(tmp_path, base_files)
    previous = tmp_path / 's1.json'
    assert snapshot.main(['--run-id', 'run-1', '--out', str(previous), '--root', str(root)]) == 0
    results: list[tuple[dict[str, Any], list[str]]] = [(read_json(previous), [])]
    for number, change in enumerate(changes, start=2):
        change(root)
        current = tmp_path / f's{number}.json'
        capsys.readouterr()
        argv = ['--run-id', f'run-{number}', '--out', str(current), '--root', str(root)]
        assert snapshot.main([*argv, '--diff', str(previous)]) == 0
        printed = capsys.readouterr().out.splitlines()
        assert printed[0].startswith('wrote '), printed
        assert snapshot.main(['--current', str(current), '--diff', str(previous)]) == 0
        assert capsys.readouterr().out.splitlines() == printed[1:]
        results.append((read_json(current), printed[1:]))
        previous = current
    return results


def diff_of(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], base_files: dict[str, str], change: Any
) -> list[str]:
    return diff_chain(tmp_path, capsys, base_files, change)[1][1]


def diff_section(diff: list[str], header: str) -> list[str]:
    start = diff.index(header) + 1
    end = next(
        (index for index in range(start, len(diff)) if not diff[index].startswith('  ')),
        len(diff),
    )
    return diff[start:end]


def rewriting(relpath: str, text: str) -> Any:
    return lambda root: commit(root, f'rewrite {relpath}', write={relpath: text})


SnapshotEdit = Callable[[dict[str, Any]], object]


def as_schema_2(taken: dict[str, Any]) -> dict[str, Any]:
    """What a pre-8463 measurement of the same tree recorded: no python_version or instrument params."""
    old = copy.deepcopy(taken)
    old['schema_version'] = 2
    del old['params']['python_version'], old['params']['instrument']
    return old


def as_schema_1(taken: dict[str, Any]) -> dict[str, Any]:
    """What a pre-6612 measurement of the same tree recorded, e.g. plans/quality-metrics/review-all-dark_factory-20261008.json."""
    old = as_schema_2(taken)
    old['schema_version'] = 1
    graph = old['import_graph']
    del graph['hidden_cycles'], graph['typing_cycles']
    for entry in graph['deferred']:
        del entry['closes_cycle']
    for record in old['files'].values():
        record.pop('cycle_closing_imports', None)
    return old
