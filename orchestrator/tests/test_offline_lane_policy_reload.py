"""A reload of the offline lane's persistence knobs reaches the worker's next pass (task 6862, δ3).

The worker is driven over a REAL ``OrchestratorConfig`` so ``apply_reload`` can mutate it in
place, exactly as ``mcp__escalation__reload_config`` does on a running orchestrator.  Fixtures
are module-local (a conftest edit would force the full-package verify).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _lane_episode_fakes import Clock, FakeLedgerClient

from orchestrator.config import (
    RELOADABLE_FIELDS,
    GitConfig,
    LaneCommand,
    OrchestratorConfig,
    apply_reload,
    diff_config,
)
from orchestrator.offline_lane import OfflineLaneWorker
from orchestrator.offline_lane_confirm import Confirmed, ConfirmMode, ConfirmTarget

POLICY_KNOBS = (
    'git.offline_lane_persist_passes',
    'git.offline_lane_recheck_secs',
    'git.offline_lane_red_unlanded_hours',
)
RETIRED_KNOB = 'git.offline_lane_red_' + 'advances_before_blocker'
COMMAND = LaneCommand(name='q', command='pytest -m integration', fix_task_priority='low')
HELD = Confirmed(ConfirmMode.NARROWED, ('tests/t.py::test_x',), (), ())


class ConfirmingHeld:
    """A confirm seam that reproduces the one failing test on every red pass."""

    async def __call__(self, target: ConfirmTarget, detect, runner) -> Confirmed:
        return HELD


def lane_git(**overrides) -> GitConfig:
    return GitConfig(
        offline_lane_legacy_numeric_enabled=False,
        offline_lane_commands=[COMMAND],
        **overrides,
    )


@pytest.fixture
def isolated_config_source(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')


class RedLane:
    """A worker whose every pass is red, over a real config it shares with the test."""

    def __init__(self, tmp_path: Path, live: OrchestratorConfig) -> None:
        heads = iter(['HEAD1', 'HEAD2', 'HEAD3'])
        git_ops = MagicMock()
        git_ops.get_main_sha = AsyncMock(side_effect=lambda: next(heads))
        git_ops.reset_persistent_offline_deep_worktree = AsyncMock(return_value=tmp_path / '_offline-deep')
        git_ops.is_ancestor_checked = AsyncMock(return_value=True)
        self.client = FakeLedgerClient([])
        self.clock = Clock()
        self.worker = OfflineLaneWorker(
            git_ops,
            live,
            lock_path=tmp_path / 'offline_lane.lock',
            command_runner=AsyncMock(return_value=(1, '')),
            confirm=ConfirmingHeld(),
            task_client=self.client,
            now=self.clock,
        )

    async def red_pass_after(self, minutes: float) -> None:
        self.clock.advance(minutes=minutes)
        await self.worker._run_once()


def live_config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=tmp_path, git=lane_git(offline_lane_persist_passes=2))


def fresh_config(tmp_path: Path, **git_overrides) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=tmp_path, git=lane_git(**git_overrides))


@pytest.mark.asyncio
async def test_a_reload_raising_persist_passes_holds_the_next_spaced_red_and_files_on_the_third(
    tmp_path: Path, isolated_config_source: None,
):
    live = live_config(tmp_path)
    lane = RedLane(tmp_path, live)

    await lane.red_pass_after(0)
    assert lane.client.submitted == [], 'a first reproduced pass is held at 1 of 2'

    report = apply_reload(live, fresh_config(tmp_path, offline_lane_persist_passes=3))
    assert report['applied'] == {'git.offline_lane_persist_passes': {'old': 2, 'new': 3}}

    await lane.red_pass_after(30)
    assert lane.client.submitted == [], 'the reload made the second spaced pass 2 of 3, still held'

    await lane.red_pass_after(30)
    assert len(lane.client.submitted) == 1, 'the third spaced pass files exactly one task'


@pytest.mark.asyncio
async def test_without_a_reload_the_second_spaced_red_files(tmp_path: Path, isolated_config_source: None):
    lane = RedLane(tmp_path, live_config(tmp_path))

    await lane.red_pass_after(0)
    assert lane.client.submitted == []
    await lane.red_pass_after(30)

    assert len(lane.client.submitted) == 1, 'the control: at the default 2 the second spaced pass files'


@pytest.mark.asyncio
async def test_a_reload_widening_recheck_secs_stops_the_next_pass_counting_at_thirty_minutes(
    tmp_path: Path, isolated_config_source: None,
):
    live = live_config(tmp_path)
    lane = RedLane(tmp_path, live)

    await lane.red_pass_after(0)
    apply_reload(live, fresh_config(tmp_path, offline_lane_persist_passes=2, offline_lane_recheck_secs=7200.0))

    await lane.red_pass_after(30)
    assert lane.client.submitted == [], 'a pass 30 minutes on is inside the reloaded 2h spacing, so uncounted'

    await lane.red_pass_after(120)
    assert len(lane.client.submitted) == 1, 'a pass past the reloaded spacing is the second counted pass'


@pytest.mark.asyncio
async def test_a_reload_shrinking_red_unlanded_hours_expires_the_held_episode(
    tmp_path: Path, isolated_config_source: None,
):
    live = live_config(tmp_path)
    lane = RedLane(tmp_path, live)

    await lane.red_pass_after(0)
    apply_reload(live, fresh_config(tmp_path, offline_lane_persist_passes=2, offline_lane_red_unlanded_hours=1.0))

    await lane.red_pass_after(120)
    assert lane.client.submitted == [], 'the held episode outlived the reloaded 1h expiry, so it restarts at 1 of 2'

    await lane.red_pass_after(30)
    assert len(lane.client.submitted) == 1, 'the restarted episode files on its own second spaced pass'


@pytest.mark.parametrize('path', POLICY_KNOBS)
def test_the_policy_knobs_are_reloadable_and_the_retired_knob_is_not(path: str):
    assert path in RELOADABLE_FIELDS
    assert RETIRED_KNOB not in RELOADABLE_FIELDS


@pytest.mark.parametrize(
    ('leaf', 'new_value'),
    [
        ('offline_lane_persist_passes', 3),
        ('offline_lane_recheck_secs', 900.0),
        ('offline_lane_red_unlanded_hours', 48.0),
    ],
)
def test_diff_config_reports_each_policy_knob_as_applied_not_restart_required(
    tmp_path: Path, isolated_config_source: None, leaf: str, new_value: float,
):
    diff = diff_config(live_config(tmp_path), fresh_config(tmp_path, **{leaf: new_value}))

    path = f'git.{leaf}'
    assert path in diff.applied_candidates
    assert path not in diff.restart_required
