"""The two-module workspace a verify-result-cache test verifies, its modules and config, and the pytest launcher its test legs need."""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Any

import pytest
from _git_fixtures import RepoSeed

from orchestrator.config import (
    ModuleConfig,
    ModuleVerifyCache,
    OrchestratorConfig,
    VerifyCacheConfig,
)
from orchestrator.merge_lane.gates import PostMergePyrightResult
from orchestrator.verify_cache.modes import CacheMode

#: Every leg of the fixture appends ``<module>:<leg>`` to the file this names.
MARKERS_ENV = 'DF_VERIFY_CACHE_TEST_MARKERS'


def _marking_test(prefix: str) -> str:
    return (
        f'import os\n\n\ndef test_{prefix}():\n'
        f"    with open(os.environ['{MARKERS_ENV}'], 'a') as markers:\n"
        f"        markers.write('{prefix}:test\\n')\n"
    )


WORKSPACE_SEED = RepoSeed(
    files=(
        ('.gitignore', '.task/\n.pytest_cache/\n__pycache__/\n'),
        ('pyproject.toml', '[tool.uv.workspace]\nmembers = ["alpha", "beta"]\n'),
        ('alpha/pyproject.toml', '[project]\nname = "alpha"\nversion = "0"\n'),
        ('alpha/src/alpha/__init__.py', 'A = 1\n'),
        ('alpha/tests/test_alpha.py', _marking_test('alpha')),
        ('beta/pyproject.toml', '[project]\nname = "beta"\nversion = "0"\ndependencies = ["alpha>=0"]\n'),
        ('beta/src/beta/__init__.py', 'B = 1\n'),
        ('beta/tests/test_beta.py', _marking_test('beta')),
    ),
    message='verify cache decision fixture',
)

ON_MODES = VerifyCacheConfig(task_mode=CacheMode.ON, merge_mode=CacheMode.ON)
LOG_MODES = VerifyCacheConfig(task_mode=CacheMode.LOG, merge_mode=CacheMode.LOG)
TEST_CODE = 'c' * 40


def cache_module(prefix: str, markers: Path, *, verify_cache: ModuleVerifyCache = 'default') -> ModuleConfig:
    return ModuleConfig(
        prefix=prefix,
        test_command=f'pytest {prefix}/tests -q',
        lint_command=f'echo {prefix}:lint >> {markers}',
        type_check_command=f'echo {prefix}:type >> {markers}',
        verify_cache=verify_cache,
    )


def cache_config(
    repo: Path,
    modes: VerifyCacheConfig = ON_MODES,
    *,
    test_command: str = 'pytest',
    lint_command: str = 'ruff check',
    type_check_command: str = 'pyright',
) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=repo, merge_verify_workspace=True, merge_verify_breadth='full',
        verify_cold_preprovision_command='', verify_cache=modes,
        test_command=test_command, lint_command=lint_command, type_check_command=type_check_command,
    )


async def clean_unscoped_gate(*_args: Any, **_kwargs: Any) -> PostMergePyrightResult:
    return PostMergePyrightResult()


def put_pytest_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare ``pytest`` on PATH running this interpreter's pytest: the launcher
    the verify path injects ``--junitxml`` into (``python -m pytest`` gets none)."""
    bin_dir = tmp_path / 'pytest-bin'
    bin_dir.mkdir()
    shim = bin_dir / 'pytest'
    shim.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -m pytest "$@"\n')
    shim.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
