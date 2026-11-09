"""Collecting a timeout-capped test module loads the module its tests exercise.

A timeout-capped test must not be charged the per-process cold import of the
module it exercises. That import belongs to collection, which pytest-timeout
does not time (task 8762: merge verify of task 8719 timed out the first test on
each of 16 xdist workers inside a function-local cold ``cockpit.app`` import).

The probe runs in a fresh interpreter because, in-process, any earlier test on
the same xdist worker would already have loaded the module.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent

_PROBE = """\
import importlib.util
import json
import sys

path, module = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location('collected_test_module', path)
collected = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collected)
print(json.dumps(module in sys.modules))
"""


def _exec_test_module_in_fresh_interpreter(test_file: str, module: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-c', _PROBE, str(TESTS_DIR / test_file), module],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ('test_file', 'module'),
    [
        pytest.param('test_app.py', 'cockpit.app', id='test_app'),
        pytest.param('test_spawn_tree.py', 'cockpit.panes.spawn_tree', id='test_spawn_tree'),
    ],
)
def test_collecting_the_module_loads_what_its_capped_tests_import(test_file: str, module: str, tmp_path: Path) -> None:
    result = _exec_test_module_in_fresh_interpreter(test_file, module, cwd=tmp_path)

    assert result.returncode == 0, f'exec of {test_file} failed:\n{result.stderr}'
    assert json.loads(result.stdout) is True, (
        f'collecting {test_file} leaves {module} unloaded, so its first capped test pays the cold import'
    )
