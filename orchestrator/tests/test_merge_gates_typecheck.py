"""The merge lane's type-check gates, driven through the lane's production
``VerifyPort`` and the ``MergeLane`` facade, over real commands that leave a
trace."""
from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import pytest
from _merge_lane_fakes import lane_scene_config

from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig
from orchestrator.merge_lane.gates import PostMergePyrightResult
from orchestrator.merge_lane.worker import PRODUCTION_VERIFIER

GIT_CONFIG = GitConfig(
    main_branch='main',
    branch_prefix='task/',
    remote='origin',
    worktree_dir='.worktrees',
    push_after_advance=False,
)

_HANG = 'sleep 60'


def _config(root: Path, **overrides: Any) -> OrchestratorConfig:
    """A lane-scene config that never runs the operational cold preprovision.

    The autouse ``_isolate_orch_config`` loads the operational yaml, whose
    ``verify_cold_preprovision_command`` would run on every merge verify.
    """
    return lane_scene_config(
        root, GIT_CONFIG, verify_cold_preprovision_command='', **overrides,
    )


def _impatient(root: Path, **extra: Any) -> OrchestratorConfig:
    """``_config`` with every per-command verify budget at one second."""
    return _config(
        root,
        verify_command_timeout_secs=1.0,
        verify_cold_command_timeout_secs=1.0,
        merge_verify_cold_command_timeout_secs=1.0,
        **extra,
    )


def _module(
    prefix: str,
    *,
    type_cmd: str,
    test_cmd: str | None = None,
    lint_cmd: str | None = None,
) -> ModuleConfig:
    return ModuleConfig(
        prefix=prefix,
        test_command=test_cmd,
        lint_command=lint_cmd,
        type_check_command=type_cmd,
    )


def _at(trace: Path, name: str) -> str:
    return shlex.quote(str(trace / name))


async def _typecheck(
    worktree: Path,
    config: OrchestratorConfig,
    modules: list[ModuleConfig],
    *,
    block_on_timeout: bool,
) -> PostMergePyrightResult:
    return await PRODUCTION_VERIFIER.run_unscoped_typechecks(
        worktree, config, modules, block_on_timeout=block_on_timeout,
    )


@pytest.fixture
def trace(tmp_path: Path) -> Path:
    """Where the commands leave their marks, outside the worktree."""
    path = tmp_path / 'trace'
    path.mkdir()
    return path


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    path = tmp_path / 'wt'
    path.mkdir()
    return path


@pytest.mark.asyncio
async def test_only_the_type_check_leg_runs(
    tmp_path: Path, trace: Path, worktree: Path,
) -> None:
    module = _module(
        'pkg',
        type_cmd=f'touch {_at(trace, "type")}',
        test_cmd=f'touch {_at(trace, "test")}',
        lint_cmd=f'touch {_at(trace, "lint")}',
    )

    result = await _typecheck(
        worktree, _config(tmp_path), [module], block_on_timeout=True,
    )

    assert sorted(p.name for p in trace.iterdir()) == ['type']
    assert result.failing_subprojects == []
    assert result.timed_out_subprojects == []


@pytest.mark.asyncio
async def test_the_type_check_runs_in_the_merge_role(
    tmp_path: Path, trace: Path, worktree: Path,
) -> None:
    module = _module(
        'pkg', type_cmd=f'printf %s "$DF_VERIFY_ROLE" > {_at(trace, "role")}',
    )

    await _typecheck(worktree, _config(tmp_path), [module], block_on_timeout=True)

    assert (trace / 'role').read_text() == 'merge'


@pytest.mark.asyncio
async def test_the_type_check_gets_the_merge_verify_budget(
    tmp_path: Path, worktree: Path,
) -> None:
    config = _config(
        tmp_path,
        verify_command_timeout_secs=1.0,
        verify_cold_command_timeout_secs=1.0,
        merge_verify_cold_command_timeout_secs=60.0,
    )

    result = await _typecheck(
        worktree, config, [_module('pkg', type_cmd='sleep 3')], block_on_timeout=True,
    )

    assert result.timed_out_subprojects == []
    assert result.failing_subprojects == []


@pytest.mark.asyncio
async def test_a_timed_out_type_check_is_not_retried(
    tmp_path: Path, trace: Path, worktree: Path,
) -> None:
    module = _module('pkg', type_cmd=f'echo run >> {_at(trace, "runs")}; {_HANG}')

    result = await _typecheck(
        worktree, _impatient(tmp_path, verify_timeout_retries=2), [module],
        block_on_timeout=False,
    )

    assert (trace / 'runs').read_text().splitlines() == ['run']
    assert result.timed_out_subprojects == ['pkg']


@pytest.mark.asyncio
async def test_a_timed_out_subproject_does_not_hide_a_later_failure(
    tmp_path: Path, worktree: Path,
) -> None:
    modules = [_module('slow', type_cmd=_HANG), _module('red', type_cmd='exit 1')]

    result = await _typecheck(
        worktree, _impatient(tmp_path), modules, block_on_timeout=False,
    )

    assert result.timed_out_subprojects == ['slow']
    assert result.failing_subprojects == ['red']


@pytest.mark.asyncio
async def test_a_silent_failing_type_check_still_carries_a_detail(
    tmp_path: Path, worktree: Path,
) -> None:
    result = await _typecheck(
        worktree, _config(tmp_path), [_module('pkg', type_cmd='exit 3')],
        block_on_timeout=True,
    )

    assert result.failing_subprojects == ['pkg']
    assert result.detail.strip() != ''
