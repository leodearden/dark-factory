"""An interrupted lane acquisition gives back what it took (task 7721).

A cancel, a deadline or an exception that lands while ``GitOps.acquire_spec_lane``
readies a ``_spec-`` lane must leave the lane FREE, and one that lands during a
cold ``_merge-*`` worktree's ``git worktree add`` must leave no tree behind.
Real git throughout: interruptions are injected through the seed script, a
vanished lane directory and a PATH ``git`` shim, never by patching GitOps.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
from collections.abc import Awaitable, Callable, Coroutine
from pathlib import Path
from typing import Any

import pytest
from _seed_script_stubs import (
    BLOCKING_TARGET_MODE_SEED_SCRIPT,
    TARGET_MODE_SEED_SCRIPT,
    SeedBlock,
    arm_seed_block,
    commit_seed_script,
    make_seed_test_repo,
    wait_until_exists,
    warm_pool_git_config,
)

from orchestrator.git_ops import GitOps, WorktreeMissing, _run
from orchestrator.warm_lane_pool import LaneState, WarmLanePool

Interrupt = Callable[[Coroutine[Any, Any, object], SeedBlock], Awaitable[None]]


async def _head(repo: Path) -> str:
    _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    return sha.strip()


def _spec_pool(git_ops: GitOps) -> WarmLanePool:
    pool = git_ops.spec_warm_lane_pool
    assert pool is not None
    return pool


async def _cancel_once_started(
    call: Coroutine[Any, Any, object], block: SeedBlock,
) -> None:
    task = asyncio.ensure_future(call)
    await block.wait_started()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def _deadline_once_started(
    call: Coroutine[Any, Any, object], block: SeedBlock,
) -> None:
    deadline = asyncio.timeout(None)

    async def under_deadline() -> None:
        async with deadline:
            await call

    task = asyncio.ensure_future(under_deadline())
    await block.wait_started()
    deadline.reschedule(asyncio.get_running_loop().time())
    with pytest.raises(TimeoutError):
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'interrupt', [_cancel_once_started, _deadline_once_started],
    ids=['cancel', 'deadline'],
)
async def test_an_interrupted_seed_frees_the_lane_and_the_next_acquire_reseeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: Interrupt,
) -> None:
    repo = make_seed_test_repo(tmp_path)
    git_ops = GitOps(warm_pool_git_config(), repo, merge_spec_warm_lane_pool_size=1)
    generation = git_ops.warm_lane_base_target_path / 'generation'
    generation.write_text('1')
    await commit_seed_script(repo, BLOCKING_TARGET_MODE_SEED_SCRIPT)
    head = await _head(repo)
    lane = git_ops.worktree_base / '_spec-0'
    block = arm_seed_block(monkeypatch, tmp_path / 'seed-block')

    await interrupt(git_ops.acquire_spec_lane(head), block)

    assert (lane / 'target' / 'partial.bin').exists(), (
        'the interruption did not land inside the seed'
    )
    assert _spec_pool(git_ops).state(lane) == LaneState.FREE, (
        'an interrupted acquire_spec_lane left its _spec- lane ASSIGNED'
    )

    block.disarm()
    generation.write_text('2')
    assert await git_ops.acquire_spec_lane(head) == (lane, True)
    assert not (lane / 'target' / 'partial.bin').exists(), (
        "the next acquire's --fresh-checkout re-seed kept the half-cloned target/"
    )
    assert (lane / 'target' / 'generation').read_text() == '2'


@pytest.mark.asyncio
async def test_an_exception_while_readying_the_lane_propagates_and_frees_it(
    tmp_path: Path,
) -> None:
    repo = make_seed_test_repo(tmp_path)
    git_ops = GitOps(warm_pool_git_config(), repo, merge_spec_warm_lane_pool_size=1)
    await commit_seed_script(repo, TARGET_MODE_SEED_SCRIPT)
    head = await _head(repo)
    lane, warm = await git_ops.acquire_spec_lane(head)
    assert warm is True
    await git_ops.release_spec_lane(lane, warm=True)
    shutil.rmtree(lane)

    with pytest.raises(WorktreeMissing):
        await git_ops.acquire_spec_lane(head)

    assert _spec_pool(git_ops).state(lane) == LaneState.FREE, (
        'an exception escaping the reset-in-place left the _spec- lane ASSIGNED'
    )


def _install_hanging_worktree_add(monkeypatch: pytest.MonkeyPatch, where: Path) -> Path:
    """Put a ``git`` shim first on PATH whose ``worktree add`` succeeds, then hangs."""
    real_git = shutil.which('git')
    assert real_git is not None
    bin_dir = where / 'bin'
    bin_dir.mkdir(parents=True)
    started = where / 'worktree-add-started'
    shim = bin_dir / 'git'
    shim.write_text(
        '#!/bin/sh\n'
        'case "$*" in\n'
        '  *"worktree add"*)\n'
        f'    {shlex.quote(real_git)} "$@"; rc=$?\n'
        f'    : > {shlex.quote(str(started))}\n'
        '    sleep 60\n'
        '    exit "$rc" ;;\n'
        'esac\n'
        f'exec {shlex.quote(real_git)} "$@"\n'
    )
    shim.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    return started


@pytest.mark.asyncio
async def test_a_cancel_during_the_cold_worktree_add_leaves_no_merge_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = make_seed_test_repo(tmp_path, warm_base=False)
    git_ops = GitOps(warm_pool_git_config(), repo)
    head = await _head(repo)
    started = _install_hanging_worktree_add(monkeypatch, tmp_path / 'shim')

    task = asyncio.ensure_future(git_ops.create_ephemeral_verify_worktree(head))
    await wait_until_exists(started)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    leftovers = sorted(p.name for p in git_ops.worktree_base.glob('_merge-*'))
    assert leftovers == [], f'a cancelled cold worktree add left {leftovers}'
    _, listing, _ = await _run(['git', 'worktree', 'list', '--porcelain'], cwd=repo)
    assert '_merge-' not in listing
