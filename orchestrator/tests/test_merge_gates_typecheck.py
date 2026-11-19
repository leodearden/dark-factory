"""The merge lane's type-check gates, driven through the lane's production
``VerifyPort`` and the ``MergeLane`` facade, over real commands that leave a
trace."""
from __future__ import annotations

import asyncio
import logging
import shlex
from pathlib import Path
from typing import Any

import pytest
from _git_fixtures import seed_repo
from _merge_lane_fakes import FakeVerifier, lane_scene_config, make_lane, merge_through_lane

from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane import (
    POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX,
    MergeOutcome,
    MergeRequest,
    QueuedBranch,
)
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
_IMPATIENT_SECS = 5.0
"""Cuts ``_HANG`` off, yet gives a loaded host time to spawn a shell."""


def _config(root: Path, **overrides: Any) -> OrchestratorConfig:
    """A lane-scene config that never runs the operational cold preprovision.

    The autouse ``_isolate_orch_config`` loads the operational yaml, whose
    ``verify_cold_preprovision_command`` would run on every merge verify.
    """
    return lane_scene_config(
        root, GIT_CONFIG, verify_cold_preprovision_command='', **overrides,
    )


def _impatient(root: Path, **extra: Any) -> OrchestratorConfig:
    """``_config`` with every per-command verify budget at ``_IMPATIENT_SECS``."""
    return _config(
        root,
        verify_command_timeout_secs=_IMPATIENT_SECS,
        verify_cold_command_timeout_secs=_IMPATIENT_SECS,
        merge_verify_cold_command_timeout_secs=_IMPATIENT_SECS,
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


BROKEN_CMD = 'test ! -e .BROKEN_UNION'
"""A type check that fails exactly when the landed tree carries the marker."""


@pytest.fixture
def git_ops(tmp_path: Path) -> GitOps:
    return GitOps(GIT_CONFIG, seed_repo(tmp_path / 'repo'))


def _request(
    task_id: str,
    worktree: Path,
    config: OrchestratorConfig,
    modules: list[ModuleConfig],
) -> MergeRequest:
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(task_id, config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=modules,
        config=config,
        result=asyncio.get_running_loop().create_future(),
    )


async def _branch_with(git_ops: GitOps, task_id: str, filename: str) -> Path:
    """A task worktree whose branch commits one file named *filename*."""
    worktree = (await git_ops.create_worktree(task_id)).path
    (worktree / filename).write_text(f'{filename}\n')
    await git_ops.commit(worktree, f'Add {filename}')
    return worktree


async def _merge(git_ops: GitOps, request: MergeRequest) -> MergeOutcome:
    """Merge *request* through a lane whose scoped verify passes.

    The post-advance type check is not a port call, so it still runs the
    request's real commands against the landed tree.
    """
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(git_ops, queue, verifier=FakeVerifier())
    return await merge_through_lane(lane, queue, request, timeout=60)


@pytest.mark.asyncio
async def test_the_blocked_reason_names_every_failing_subproject_and_the_landed_sha(
    git_ops: GitOps,
) -> None:
    task_id = 'tc6414-broken'
    worktree = await _branch_with(git_ops, task_id, '.BROKEN_UNION')
    config = _config(git_ops.project_root)
    modules = [
        _module('pkg-a', type_cmd=BROKEN_CMD),
        _module('pkg-b', type_cmd=BROKEN_CMD),
    ]

    outcome = await _merge(git_ops, _request(task_id, worktree, config, modules))
    landed = await git_ops.get_main_sha()

    assert outcome.status == 'blocked'
    assert outcome.reason.startswith(POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX)
    assert f'failed for pkg-a, pkg-b on {landed[:12]}. ' in outcome.reason


@pytest.mark.asyncio
async def test_a_timed_out_post_advance_check_lands_and_names_the_sha_and_task(
    git_ops: GitOps, caplog: pytest.LogCaptureFixture,
) -> None:
    task_id = 'tc6414-hang'
    worktree = await _branch_with(git_ops, task_id, 'mod.txt')
    config = _impatient(git_ops.project_root)
    request = _request(task_id, worktree, config, [_module('pkg', type_cmd=_HANG)])

    with caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane'):
        outcome = await _merge(git_ops, request)
    landed = await git_ops.get_main_sha()

    messages = [r.getMessage() for r in caplog.records if r.name == 'orchestrator.merge_lane']
    assert outcome.status == 'done'
    assert any(
        'failing open' in m and f'on {landed[:12]}' in m and f'task_id={task_id}' in m
        for m in messages
    ), messages
