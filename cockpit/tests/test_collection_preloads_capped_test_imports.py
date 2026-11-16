"""Collecting a timeout-capped test module loads the dependencies its tests import.

A test under an explicit pytest-timeout cap must not be charged the per-process
cold import of a third-party dependency. That import belongs to collection,
which pytest-timeout does not time (task 8762: merge verify of task 8719 timed
out the first test on each of 16 xdist workers inside a function-local cold
``cockpit.app`` import, which compiles textual).

Every test module that carries a timeout marker gets a row; none is listed by
hand. The row replays all of the module's function-local imports, not only the
capped tests' own, because pytest-timeout also times the fixtures and helpers a
test runs. It replays them in a fresh interpreter, right after executing the
module the way collection does, because in-process any earlier test on the same
xdist worker would already have loaded them. Only modules loaded from
site-packages count: those are the third-party distributions a fresh verify
venv holds without bytecode. The standard library and the workspace's editable
packages are small enough to import inside a cap.

Not covered: the first ``run_test`` lazily imports textual internals inside the
cap. Compiling those cold is task 8790's to fix.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent

_SITE_PACKAGES = frozenset(Path(sysconfig.get_paths()[key]).resolve() for key in ('purelib', 'platlib'))

_PROBE = """\
import importlib.util
import json
import sys

path, statements = sys.argv[1], json.loads(sys.argv[2])
spec = importlib.util.spec_from_file_location('collected_test_module', path)
collected = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collected)
newly_loaded = {}
for statement in statements:
    before = set(sys.modules)
    exec(statement, {})
    newly_loaded[statement] = {
        name: getattr(sys.modules[name], '__file__', None) for name in sorted(set(sys.modules) - before)
    }
print(json.dumps(newly_loaded))
"""


def _is_timeout_mark(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == 'timeout'
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == 'mark'
    )


def _parse(test_file: Path) -> ast.Module:
    return ast.parse(test_file.read_text(), filename=str(test_file))


def _timeout_capped_test_files() -> list[Path]:
    return sorted(
        path
        for path in TESTS_DIR.rglob('test_*.py')
        if any(_is_timeout_mark(node) for node in ast.walk(_parse(path)))
    )


def _function_local_import_statements(tree: ast.Module) -> list[str]:
    return sorted(
        {
            ast.unparse(node)
            for function in ast.walk(tree)
            if isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef)
            for node in ast.walk(function)
            if isinstance(node, ast.Import | ast.ImportFrom)
        }
    )


def _replay_after_collection_in_fresh_interpreter(
    test_file: Path, statements: list[str], cwd: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-c', _PROBE, str(test_file), json.dumps(statements)],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _from_site_packages(module_file: str | None) -> bool:
    if module_file is None:
        return False
    resolved = Path(module_file).resolve()
    return any(resolved.is_relative_to(site_packages) for site_packages in _SITE_PACKAGES)


@pytest.mark.parametrize(
    'test_file',
    [
        pytest.param(path, id=path.relative_to(TESTS_DIR).with_suffix('').as_posix())
        for path in _timeout_capped_test_files()
    ],
)
def test_collecting_the_module_loads_what_its_capped_tests_import(test_file: Path, tmp_path: Path) -> None:
    statements = _function_local_import_statements(_parse(test_file))

    result = _replay_after_collection_in_fresh_interpreter(test_file, statements, cwd=tmp_path)

    assert result.returncode == 0, f'replaying the imports of {test_file.name} failed:\n{result.stderr}'
    newly_loaded: dict[str, dict[str, str | None]] = json.loads(result.stdout)
    charged_inside_the_cap = {
        statement: charged
        for statement, module_files in newly_loaded.items()
        if (charged := sorted(name for name, file in module_files.items() if _from_site_packages(file)))
    }
    assert charged_inside_the_cap == {}, (
        f'collecting {test_file.name} leaves these function-local imports a cold third-party import '
        'to pay inside a timeout cap; import them at module scope'
    )
