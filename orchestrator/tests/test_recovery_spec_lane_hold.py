"""Startup recovery leaves a merge-worker-held ``_spec-`` lane assigned (task 7745, folded into 8268).

A ``_spec-`` lane is owned in memory by the merge worker's spec pool. That pool
has no LaneLifecycle, so every slot starts FREE in a new process. A spec lane
carries no plan, task identity or agent session.

Since task 8268 the merge loop is launched only after
``scheduler.finish_startup()``, so in production no spec lane is held while 2c
(``_recover_crashed_tasks``) runs. Startup recovery ALSO leaves every
``_spec-`` lane to its owner, so the property holds by ownership as well as by
order (docs/code-quality.md heuristic 10, redundantly).
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _seed_script_stubs import (
    TARGET_MODE_SEED_SCRIPT,
    commit_seed_script,
    make_seed_test_repo,
    warm_pool_git_config,
)

from orchestrator.harness import Harness


async def _run_startup_recovery(harness: Harness, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger='orchestrator.harness'):
        await harness._recover_crashed_tasks()


def _info_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]


@pytest.mark.asyncio
async def test_startup_recovery_leaves_a_merge_worker_held_spec_lane_assigned(
    tmp_path: Path, mock_orch_config: MagicMock, caplog: pytest.LogCaptureFixture,
) -> None:
    repo = make_seed_test_repo(tmp_path)
    await commit_seed_script(repo, TARGET_MODE_SEED_SCRIPT)
    mock_orch_config.git = warm_pool_git_config().model_copy(update={'warm_lane_pool': False})
    mock_orch_config.project_root = repo
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        harness = Harness(mock_orch_config)
    harness.scheduler = MagicMock()
    harness.scheduler.get_task = AsyncMock(return_value={})
    harness.scheduler.get_status = AsyncMock(return_value=None)
    harness.event_store = MagicMock()
    head = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()
    git_ops = harness.git_ops
    spec_pool = git_ops.spec_warm_lane_pool
    assert spec_pool is not None

    lane, warm = await git_ops.acquire_spec_lane(head)
    assert warm is True
    assert git_ops.pool_storage_present()

    await _run_startup_recovery(harness, caplog)

    assert spec_pool.is_assigned(lane)
    assert lane.exists()
    infos = _info_messages(caplog)
    assert any('_merge-verify' in m for m in infos), infos
    assert any(lane.name in m for m in infos), infos

    lane2, warm2 = await git_ops.acquire_spec_lane(head)
    assert lane2 != lane
    assert warm2 is False
    assert spec_pool.is_assigned(lane)
