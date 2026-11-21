"""The timing-dependence lint's wiring across every package that lint scans.

fused-memory/scripts/_timing_dependence.py accepts a real sleep, an elapsed-vs-literal
assert or a literal wait deadline only under ``@pytest.mark.timing(reason=...)``.  An
unregistered marker only warns, because ``--strict-markers`` is inert in addopts, so
the repo root and each scanned package register it, as they do ``reads_git_history``.
Each scanned package's tests/conftest.py also binds the sanctioned ``fake_clock``
fixture (shared/src/shared/pytest_fake_clock.py::fake_clock) by import.

The package list is DERIVED from the gate's own lint_command, so which packages the
rule scans has one home.
"""

from __future__ import annotations

import ast
import functools
import re
import shlex
import tomllib
from pathlib import Path, PurePosixPath

import pytest
import yaml
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = Path(__file__).resolve().parents[2]
_GATE_CONFIG = REPO_ROOT / 'dark-factory-orchestrator.yaml'
_CHECKER_SUFFIX = 'check_bare_magicmock_config.py'
_TIMING_MARKER_ENTRY = (
    '"timing(reason): a deadline, grace period or timeout is the behaviour under test,'
    ' named by reason=; the timing-dependence lint (fused-memory/scripts/_timing_dependence.py)'
    ' accepts a real sleep, an elapsed-vs-literal assert or a literal wait deadline only under'
    ' it. Not deselected by default."'
)


_FAKE_CLOCK_MODULE = 'shared.pytest_fake_clock'
_FAKE_CLOCK_BINDING = (
    f'from {_FAKE_CLOCK_MODULE} import fake_clock as fake_clock'
)


def _checker_arguments(command: str) -> list[str]:
    tokens = shlex.split(command)
    start = next((i + 1 for i, token in enumerate(tokens) if token.endswith(_CHECKER_SUFFIX)), None)
    if start is None:
        return []
    arguments: list[str] = []
    for token in tokens[start:]:
        if token == '&&':
            break
        arguments.append(token)
    return arguments


def _package_of(scanned_dir: str, command: str) -> str:
    parts = PurePosixPath(scanned_dir).parts
    if len(parts) != 2 or parts[1] != 'tests':
        pytest.fail(f'{scanned_dir!r} in lint_command {command!r} is not a <package>/tests directory')
    return parts[0]


@functools.cache
def _scanned_packages() -> tuple[str, ...]:
    command = yaml.safe_load(_GATE_CONFIG.read_text(encoding='utf-8'))['lint_command']
    packages = tuple(_package_of(arg, command) for arg in _checker_arguments(command))
    if not packages:
        pytest.fail(f'found no {_CHECKER_SUFFIX} test directories in lint_command {command!r}')
    return packages


def _registered_marker_names(pyproject: Path) -> set[str]:
    config = tomllib.loads(pyproject.read_text(encoding='utf-8'))
    markers = config.get('tool', {}).get('pytest', {}).get('ini_options', {}).get('markers', [])
    return {re.split(r'[(:]', marker, maxsplit=1)[0].strip() for marker in markers}


def test_every_scanned_package_registers_the_timing_marker():
    pyprojects = [REPO_ROOT / 'pyproject.toml'] + [
        REPO_ROOT / package / 'pyproject.toml' for package in _scanned_packages()
    ]
    missing = [
        str(pyproject.relative_to(REPO_ROOT))
        for pyproject in pyprojects
        if 'timing' not in _registered_marker_names(pyproject)
    ]
    assert not missing, (
        'these pyproject.toml files do not register the `timing` marker the timing-dependence'
        f' lint accepts: {missing}. Add this entry to each one\'s [tool.pytest.ini_options]'
        f' markers list: {_TIMING_MARKER_ENTRY}'
    )


def _binds_fake_clock(conftest: Path) -> bool:
    module = ast.parse(conftest.read_text(encoding='utf-8'), filename=str(conftest))
    return any(
        isinstance(statement, ast.ImportFrom)
        and statement.module == _FAKE_CLOCK_MODULE
        and any(alias.name == 'fake_clock' and alias.asname in (None, 'fake_clock') for alias in statement.names)
        for statement in module.body
    )


def test_every_scanned_package_conftest_binds_fake_clock():
    conftests = [REPO_ROOT / package / 'tests' / 'conftest.py' for package in _scanned_packages()]
    missing = [
        str(conftest.relative_to(REPO_ROOT))
        for conftest in conftests
        if not conftest.is_file() or not _binds_fake_clock(conftest)
    ]
    assert not missing, (
        f'these conftests do not bind the fake_clock fixture at module level: {missing}.'
        f' Add this line to each, after the sys.path setup that makes shared importable:'
        f' {_FAKE_CLOCK_BINDING}'
    )
