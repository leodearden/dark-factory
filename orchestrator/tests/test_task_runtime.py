"""Tests for orchestrator.task_runtime — the free-function core behind
Harness.task_runtime_snapshot() (task 2634, PRD
plans/dashboard-task-runtime-endpoint-prd.md task alpha).

Self-contained, mirroring test_lane_lifecycle_gitops.py: conftest.py
provides no shared `git_repo` fixture, so this module owns its own fixture.
"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.lane_lifecycle import LaneState
from orchestrator.task_runtime import (
    TaskRuntimeState,
    _derive_phase,
    _map_lane_state,
    build_task_runtime_snapshot,
)

# ---------------------------------------------------------------------------
# Repo fixture (mirrors test_lane_lifecycle_gitops.py)
# ---------------------------------------------------------------------------


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Temporary git repo with an initial commit + a resolvable warm base.

    Mirrors test_lane_lifecycle_gitops.py's git_repo fixture so the same
    fixture serves both the non-pooled and pooled (TestPooled) scenarios
    below.
    """
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    default_base = repo / '.worktrees' / '_merge-verify' / 'target'
    default_base.mkdir(parents=True, exist_ok=True)
    (default_base / '.keep').write_text('warm base sentinel\n')
    (repo / '.worktrees' / '.pool-root').touch()
    return repo


def _make_task_artifacts(worktree_base: Path, name: str, task_id: str) -> TaskArtifacts:
    """Create <worktree_base>/<name> plus its .task-meta/<name> artifacts root."""
    worktree = worktree_base / name
    worktree.mkdir(parents=True, exist_ok=True)
    meta_root = TaskArtifacts.meta_root_for(worktree_base, name)
    ta = TaskArtifacts(worktree, meta_root)
    ta.init(task_id, f'Task {task_id}', 'desc')
    return ta


# ---------------------------------------------------------------------------
# Pure helpers — _derive_phase / _map_lane_state / TaskRuntimeState shape
# ---------------------------------------------------------------------------


class TestPureHelpers:
    def test_derive_phase_empty_plan_is_plan(self):
        assert _derive_phase({}) == 'PLAN'

    def test_derive_phase_empty_steps_list_is_plan(self):
        assert _derive_phase({'steps': []}) == 'PLAN'

    def test_derive_phase_all_done_is_done(self):
        plan = {'steps': [{'status': 'done'}, {'status': 'done'}]}
        assert _derive_phase(plan) == 'DONE'

    def test_derive_phase_mixed_is_execute(self):
        plan = {'steps': [{'status': 'done'}, {'status': 'pending'}]}
        assert _derive_phase(plan) == 'EXECUTE'

    def test_map_lane_state_assigned(self):
        assert _map_lane_state(LaneState.ASSIGNED) == 'assigned'

    def test_map_lane_state_in_use(self):
        assert _map_lane_state(LaneState.IN_USE) == 'assigned'

    def test_map_lane_state_quarantined(self):
        assert _map_lane_state(LaneState.QUARANTINED) == 'quarantined'

    def test_map_lane_state_released(self):
        assert _map_lane_state(LaneState.RELEASED) == 'released'

    def test_map_lane_state_seed_is_none(self):
        assert _map_lane_state(LaneState.SEED) is None

    def test_map_lane_state_registered_is_none(self):
        assert _map_lane_state(LaneState.REGISTERED) is None


class TestTaskRuntimeStateDataclass:
    def test_is_a_dataclass_with_expected_fields(self):
        assert dataclasses.is_dataclass(TaskRuntimeState)
        field_names = {f.name for f in dataclasses.fields(TaskRuntimeState)}
        assert field_names == {
            'task_id', 'has_worktree', 'loops', 'attempts', 'started',
            'lane', 'phase', 'lane_state', 'error',
        }

    def test_error_defaults_to_none(self):
        state = TaskRuntimeState(
            task_id=1,
            has_worktree=True,
            loops=0,
            attempts=0,
            started=None,
            lane=None,
            phase='PLAN',
            lane_state=None,
        )
        assert state.error is None


# ---------------------------------------------------------------------------
# Non-pooled layout (scenario B2) — per-task worktree dirs, no lane concept.
# ---------------------------------------------------------------------------


class TestNonPooled:
    def test_single_task_worktree(self, git_repo: Path):
        git_ops = GitOps(GitConfig(), git_repo)
        assert not git_ops.pool_in_use()
        worktree_base = git_ops.worktree_base

        ta = _make_task_artifacts(worktree_base, '42', '42')
        ta.write_plan({'steps': [
            {'id': 's1', 'status': 'done'},
            {'id': 's2', 'status': 'pending'},
        ]})
        ta.append_iteration_log({'note': 'iter-1'})
        ta.append_iteration_log({'note': 'iter-2'})
        ta.write_review('reviewer-a', {'verdict': 'PASS', 'issues': []})
        expected_started = ta.read_created_at()

        # Dirs that don't yield a task id must be ignored by enumeration.
        (worktree_base / '_lane-9').mkdir(parents=True, exist_ok=True)
        (worktree_base / 'random-dir').mkdir(parents=True, exist_ok=True)

        result = build_task_runtime_snapshot(git_ops=git_ops)

        assert len(result) == 1
        entry = result[0]
        assert entry.task_id == 42
        assert entry.loops == 2
        assert entry.attempts == 1
        assert entry.lane is None
        assert entry.lane_state is None
        assert entry.phase == 'EXECUTE'
        assert entry.has_worktree is True
        assert entry.started == expected_started
        assert entry.error is None
