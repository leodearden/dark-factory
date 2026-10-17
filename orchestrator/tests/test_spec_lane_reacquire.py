"""A re-acquired ``_spec-`` lane must seed warm from the CURRENT base (task 5380).

``acquire_spec_lane``'s reset branch keeps the lane's ``target/`` through its
``git clean``.  Driven by a stub of reify's seed-mode contract, these tests
fail for the production reason: ``--reset-in-place`` refuses a non-empty
``target/`` (the clobber guard), so every acquire after the first fell back to
a cold build.  That went unnoticed because the fallback logged at DEBUG, so a
fallback that gives back an already-held lane must be visible at WARNING.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from _seed_script_stubs import (
    TARGET_MODE_SEED_SCRIPT,
    commit_seed_script,
    make_seed_test_repo,
    warm_pool_git_config,
)

from orchestrator.git_ops import GitOps, _run
from orchestrator.warm_lane_pool import LaneState

FAILING_SEED_SCRIPT = '#!/usr/bin/env bash\nexit 1\n'


async def _head(repo: Path) -> str:
    _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    return sha.strip()


async def _commit_change(repo: Path) -> str:
    (repo / 'change.txt').write_text('advance\n')
    await _run(['git', 'add', 'change.txt'], cwd=repo)
    await _run(['git', 'commit', '-m', 'advance main'], cwd=repo)
    return await _head(repo)


def _warnings_naming(caplog: pytest.LogCaptureFixture, needle: str) -> list[str]:
    return [
        record.getMessage() for record in caplog.records
        if record.levelno >= logging.WARNING and needle in record.getMessage()
    ]


@pytest.mark.asyncio
async def test_reacquired_spec_lane_is_warm_and_reseeded_from_current_base(
    tmp_path: Path,
) -> None:
    repo = make_seed_test_repo(tmp_path)
    git_ops = GitOps(warm_pool_git_config(), repo, merge_spec_warm_lane_pool_size=1)
    generation = git_ops.warm_lane_base_target_path / 'generation'
    generation.write_text('1')
    await commit_seed_script(repo, TARGET_MODE_SEED_SCRIPT)

    lane, warm = await git_ops.acquire_spec_lane(await _head(repo))
    assert warm is True
    assert (lane / 'target' / 'generation').read_text() == '1'
    await git_ops.release_spec_lane(lane, warm=True)

    generation.write_text('2')
    lane2, warm2 = await git_ops.acquire_spec_lane(await _commit_change(repo))

    assert warm2 is True, (
        're-acquired _spec- lane fell back COLD: the seed refused the target/ '
        "the reset branch's git clean retained (reify's --reset-in-place "
        'clobber guard)'
    )
    assert lane2 == lane
    assert (lane2 / 'target' / 'generation').read_text() == '2', (
        'inv.8: a re-acquired lane must be re-seeded from the CURRENT warm base'
    )
    assert git_ops.spec_warm_lane_pool is not None
    assert git_ops.spec_warm_lane_pool.state(lane2) == LaneState.ASSIGNED


@pytest.mark.asyncio
async def test_seed_failure_cold_fallback_warns_naming_the_lane(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger='orchestrator.git_ops')
    repo = make_seed_test_repo(tmp_path)
    await commit_seed_script(repo, FAILING_SEED_SCRIPT)
    git_ops = GitOps(warm_pool_git_config(), repo, merge_spec_warm_lane_pool_size=1)

    _, warm = await git_ops.acquire_spec_lane(await _head(repo))

    assert warm is False
    assert git_ops.spec_warm_lane_pool is not None
    lane = git_ops.worktree_base / '_spec-0'
    assert git_ops.spec_warm_lane_pool.state(lane) == LaneState.FREE
    warnings = _warnings_naming(caplog, '_spec-0')
    assert any('rc=1' in warning for warning in warnings), (
        'a seed that exited 1 gave back an acquired lane and went cold without '
        f'a WARNING naming the lane and the exit code: {warnings!r}'
    )


@pytest.mark.asyncio
async def test_reset_failure_cold_fallback_warns_naming_the_lane(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger='orchestrator.git_ops')
    repo = make_seed_test_repo(tmp_path)
    await commit_seed_script(repo, TARGET_MODE_SEED_SCRIPT)
    git_ops = GitOps(warm_pool_git_config(), repo, merge_spec_warm_lane_pool_size=1)
    lane, warm = await git_ops.acquire_spec_lane(await _head(repo))
    assert warm is True
    await git_ops.release_spec_lane(lane, warm=True)
    (lane / '.git').write_text('gitdir: /nonexistent\n')
    caplog.clear()

    cold, warm2 = await git_ops.acquire_spec_lane(await _head(repo))

    assert warm2 is False
    assert cold != lane
    assert git_ops.spec_warm_lane_pool is not None
    assert git_ops.spec_warm_lane_pool.state(lane) == LaneState.FREE
    assert _warnings_naming(caplog, '_spec-0'), (
        'a failed reset gave back an acquired lane and went cold without a '
        'WARNING naming the lane'
    )
