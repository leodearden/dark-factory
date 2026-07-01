"""Tests for the offline deep-test lane singleton worker (task 1953, β2).

Covers ``orchestrator.offline_lane.OfflineLaneWorker`` in isolation: the
``on_post_merge`` enqueue-and-return trigger seam, the always-from-head
``_run_once`` snapshot, the coalescing ``run()`` loop (trigger-driven +
poll-backstop, fail-open), the lockfile singleton, and the default
``run-offline-deep.sh`` suite-runner seam.

See also ``test_harness_offline_lane.py`` for the harness-side
launch/stop/registration wiring, and
``test_harness_offline_lane_trigger.py`` (task 1951, β1) for the
``_offline_lane_notifiee`` fan-out this worker registers into.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from orchestrator.config import GitConfig
from orchestrator.offline_lane import OfflineLaneWorker

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_git_ops(*, head: str = 'headsha', worktree_path: Path | None = None) -> MagicMock:
    """MagicMock git_ops with the async methods OfflineLaneWorker calls."""
    git_ops = MagicMock()
    git_ops.get_main_sha = AsyncMock(return_value=head)
    git_ops.reset_persistent_offline_deep_worktree = AsyncMock(
        return_value=worktree_path or Path('/tmp/_offline-deep')
    )
    return git_ops


def _make_config(tmp_path: Path, **git_overrides) -> MagicMock:
    """MagicMock OrchestratorConfig with a real GitConfig at .git."""
    config = MagicMock()
    config.project_root = tmp_path
    config.git = GitConfig(**git_overrides)
    return config


def _make_worker(
    tmp_path: Path,
    *,
    git_ops: MagicMock | None = None,
    config: MagicMock | None = None,
    suite_runner=None,
) -> OfflineLaneWorker:
    """Build an OfflineLaneWorker wired with mock deps for isolated testing."""
    return OfflineLaneWorker(
        git_ops if git_ops is not None else _make_git_ops(),
        config if config is not None else _make_config(tmp_path),
        lock_path=tmp_path / 'offline_lane.lock',
        suite_runner=suite_runner,
    )


# ---------------------------------------------------------------------------
# GitConfig knobs (step-1/2)
# ---------------------------------------------------------------------------


def test_git_config_offline_lane_knobs():
    """GitConfig exposes the three offline-lane knobs with correct defaults.

    Step 1 (RED): the fields do not yet exist — must fail before impl.
    """
    cfg_default = GitConfig()
    assert cfg_default.offline_lane_enabled is False, (
        'offline_lane_enabled must default to False (feature off)'
    )
    assert cfg_default.offline_lane_test_threads == 6, (
        'offline_lane_test_threads must default to 6 (§11.2 small fixed N)'
    )
    assert cfg_default.offline_lane_poll_interval_secs == 120.0, (
        'offline_lane_poll_interval_secs must default to 120.0'
    )

    cfg_set = GitConfig(
        offline_lane_enabled=True,
        offline_lane_test_threads=4,
        offline_lane_poll_interval_secs=30.0,
    )
    assert cfg_set.offline_lane_enabled is True
    assert cfg_set.offline_lane_test_threads == 4
    assert cfg_set.offline_lane_poll_interval_secs == 30.0


def test_git_config_offline_lane_knobs_validation():
    """offline_lane_test_threads is ge=1; offline_lane_poll_interval_secs is gt=0."""
    with pytest.raises(ValidationError):
        GitConfig(offline_lane_test_threads=0)
    with pytest.raises(ValidationError):
        GitConfig(offline_lane_poll_interval_secs=0.0)


# ---------------------------------------------------------------------------
# on_post_merge — enqueue-and-return trigger seam (step-3/4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_post_merge_sets_dirty_and_wakes_without_running(tmp_path: Path):
    """on_post_merge flips dirty + wakes the loop and returns — never runs inline.

    Per the β1 hot-path contract (harness.py:5040-5046), the SHAs are
    advisory only: get_main_sha and suite_runner must NOT be awaited by
    on_post_merge itself.

    Step 3 (RED): on_post_merge is a NotImplementedError stub — must fail
    before impl.
    """
    git_ops = _make_git_ops()
    suite_runner = AsyncMock(return_value=(0, ''))
    worker = _make_worker(tmp_path, git_ops=git_ops, suite_runner=suite_runner)

    await worker.on_post_merge('task-1', 'base-sha', 'head-sha')

    assert worker._dirty is True
    assert worker._wake.is_set()
    git_ops.get_main_sha.assert_not_awaited()
    suite_runner.assert_not_awaited()


# ---------------------------------------------------------------------------
# _run_once — always-from-head snapshot (step-5/6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_once_snapshots_head_at_run_start_and_invokes_seam(tmp_path: Path):
    """_run_once clears dirty BEFORE snapshotting head, then resets + runs the seam.

    Step 5 (RED): _run_once is a NotImplementedError stub — must fail
    before impl.
    """
    worker_ref: dict[str, OfflineLaneWorker] = {}

    async def _fake_get_main_sha() -> str:
        # Clear-before-snapshot: by the time get_main_sha is awaited, dirty
        # must already be cleared (no lost-update window).
        assert worker_ref['worker']._dirty is False, (
            '_run_once must clear _dirty BEFORE snapshotting head'
        )
        return 'HEAD1'

    wt_path = tmp_path / '_offline-deep'
    git_ops = MagicMock()
    git_ops.get_main_sha = AsyncMock(side_effect=_fake_get_main_sha)
    git_ops.reset_persistent_offline_deep_worktree = AsyncMock(return_value=wt_path)
    suite_runner = AsyncMock(return_value=(0, ''))
    config = _make_config(tmp_path, offline_lane_test_threads=6)
    worker = _make_worker(tmp_path, git_ops=git_ops, config=config, suite_runner=suite_runner)
    worker_ref['worker'] = worker
    worker._dirty = True

    await worker._run_once()

    git_ops.get_main_sha.assert_awaited_once()
    git_ops.reset_persistent_offline_deep_worktree.assert_awaited_once_with('HEAD1')
    suite_runner.assert_awaited_once_with(wt_path, 'HEAD1', 6)
    assert worker._last_run_head == 'HEAD1', (
        '_last_run_head must equal the snapshot head, never the advisory trigger SHA'
    )
