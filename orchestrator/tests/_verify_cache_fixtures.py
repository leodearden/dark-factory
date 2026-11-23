"""The two-module workspace a verify-result-cache test verifies: its modules and config,
real legs that mark each run, the verifies that run them, and what their results say."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pytest
from _git_fixtures import RepoSeed
from _orch_helpers import git_env_with_ceiling

from orchestrator import verify_plan
from orchestrator.config import (
    ModuleConfig,
    ModuleVerifyCache,
    OrchestratorConfig,
    VerifyCacheConfig,
)
from orchestrator.deployed_sha import DeployedSha
from orchestrator.verify import VerifyResult, run_scoped_verification
from orchestrator.verify_cache.modes import CacheMode, VerifyRole
from orchestrator.verify_cache.recorder import VerifyEvidenceRef, record_greens
from orchestrator.verify_cache.registry import Entry, Registry
from orchestrator.verify_cache.replay import CacheAccess
from orchestrator.verify_evidence import Leg, UnitOutcome, VerifyUnit, module_leg_name
from orchestrator.verify_plan import VerifyPlan

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


def put_pytest_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare ``pytest`` on PATH running this interpreter's pytest: the launcher
    the verify path injects ``--junitxml`` into (``python -m pytest`` gets none)."""
    bin_dir = tmp_path / 'pytest-bin'
    bin_dir.mkdir()
    shim = bin_dir / 'pytest'
    shim.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -m pytest "$@"\n')
    shim.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')


def use_real_legs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, markers: Path) -> None:
    """Real legs: a bare ``pytest`` whose junit the verify reads, and every leg leaves a marker in *markers*."""
    put_pytest_on_path(tmp_path, monkeypatch)
    monkeypatch.setenv(MARKERS_ENV, str(markers))


def ran_legs(markers: Path) -> set[str]:
    """The ``<module>:<leg>`` of every leg that executed since *markers* was last cleared."""
    if not markers.exists():
        return set()
    return {line.split()[0] for line in markers.read_text().splitlines() if line.strip()}


def clear_markers(markers: Path) -> None:
    markers.unlink(missing_ok=True)


def module_legs(prefix: str) -> set[str]:
    """The marker each leg of module *prefix* leaves when it runs."""
    return {f'{prefix}:{leg}' for leg in Leg}


def module_units(prefix: str) -> set[str]:
    """The unit id of each leg of module *prefix*."""
    return {module_leg_name(prefix, leg) for leg in Leg}


def workspace_git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-c', 'user.email=t@t', '-c', 'user.name=t', *args], cwd=repo,
        env=git_env_with_ceiling(repo), check=True, capture_output=True, text=True,
    ).stdout.strip()


def commit_edit(repo: Path, path: str, text: str) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    workspace_git(repo, 'add', path)
    workspace_git(repo, 'commit', '-q', '-m', f'edit {path}')


def full_suite_plan(modules: Sequence[ModuleConfig], role: VerifyRole = 'merge') -> VerifyPlan:
    plan_role: Literal['merge', 'task'] = 'task' if role == 'task' else 'merge'
    return VerifyPlan(runs=tuple(
        run for mc in modules for run in verify_plan._derive_full_suite_runs(mc, plan_role)
    ))


async def merge_gate(
    repo: Path, config: OrchestratorConfig, modules: Sequence[ModuleConfig], access: CacheAccess | None,
) -> VerifyResult:
    return await run_scoped_verification(
        repo, config, list(modules), role='merge', is_merge_verify=True, force_workspace=True,
        verify_cache=access,
    )


async def task_verify(
    repo: Path, config: OrchestratorConfig, modules: Sequence[ModuleConfig], access: CacheAccess,
    task_files: list[str],
) -> VerifyResult:
    """A task verify as the workflow runs one: its junit cost report is what admits its test leg."""
    return await run_scoped_verification(
        repo, config, list(modules), task_files, role='task', verify_cache=access,
        attempt_id=1, task_id='5653', archive_root=repo.parent / 'verify-archive',
    )


async def record_run(registry: Registry, result: VerifyResult, *, event_id: int = 41) -> dict[str, Entry]:
    """Record *result*'s admitted greens as the dispatcher would, citing verify event *event_id*."""
    assert result.passed, result.summary
    assert result.cache_record is not None
    outcome = await record_greens(
        registry, result.cache_record, flake_suppressed=False,
        evidence=VerifyEvidenceRef(event_id, 'r', 'sha-first', True),
    )
    return {entry.unit_id: entry for entry in outcome.recorded}


def units_by_id(result: VerifyResult) -> dict[str, VerifyUnit]:
    assert result.units is not None
    return {unit.unit: unit for unit in result.units}


def outcomes_of(result: VerifyResult, unit_ids: set[str]) -> set[UnitOutcome]:
    units = units_by_id(result)
    return {units[unit_id].outcome for unit_id in unit_ids}


#: Two deployed shas that name no commit of any checkout.
CODE_D1 = 'd' * 40
CODE_D2 = 'e' * 40


def deployed_at(sha: str = CODE_D1, *, indeterminate: bool = False, dirty_wire_path: bool = False) -> DeployedSha:
    return DeployedSha(sha=sha, indeterminate=indeterminate, dirty_wire_path=dirty_wire_path)
