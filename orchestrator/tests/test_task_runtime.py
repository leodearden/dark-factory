"""Tests for orchestrator.task_runtime — the free-function core behind
Harness.task_runtime_snapshot() (task 2634, PRD
plans/dashboard-task-runtime-endpoint-prd.md task alpha).

Self-contained, mirroring test_lane_lifecycle_gitops.py: conftest.py
provides no shared `git_repo` fixture, so this module owns its own fixture.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig
from orchestrator.event_store import EventStore, EventType
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


def _warm_config(**overrides) -> GitConfig:
    """Build a GitConfig with the warm-lane pool enabled (mirrors
    test_lane_lifecycle_gitops.py's _warm_config)."""
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        warm_lane_pool=True,
        **overrides,
    )


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


# ---------------------------------------------------------------------------
# Pooled layout (scenarios B1/B3/B4) — durable .lane-state records drive
# task<->lane, task<->lane_state; artifacts still read via the shared
# .task-meta/<lane> root.
# ---------------------------------------------------------------------------


class TestPooled:
    def test_pooled_layout(self, git_repo: Path):
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        assert git_ops.pool_in_use()
        worktree_base = git_ops.worktree_base
        lifecycle = git_ops._lane_lifecycle

        # _lane-3 -> task 42, ASSIGNED (B1): 3 iteration lines + 1 review
        # PASS + a plan.json.
        lifecycle.note_assigned('_lane-3', task_id='42')
        ta42 = _make_task_artifacts(worktree_base, '_lane-3', '42')
        ta42.write_plan({'steps': [{'id': 's1', 'status': 'done'}]})
        ta42.append_iteration_log({'note': 'iter-1'})
        ta42.append_iteration_log({'note': 'iter-2'})
        ta42.append_iteration_log({'note': 'iter-3'})
        ta42.write_review('reviewer-a', {'verdict': 'PASS', 'issues': []})

        # _lane-5 -> task 43, ASSIGNED then QUARANTINED (B3).
        lifecycle.note_assigned('_lane-5', task_id='43')
        lifecycle.transition('_lane-5', LaneState.QUARANTINED)

        # _lane-7 -> task 44, ASSIGNED with an EMPTY iterations.jsonl (B4:
        # honest zero, not a read failure).
        lifecycle.note_assigned('_lane-7', task_id='44')
        _make_task_artifacts(worktree_base, '_lane-7', '44')
        empty_log = TaskArtifacts.meta_root_for(worktree_base, '_lane-7') / 'iterations.jsonl'
        empty_log.write_text('')

        # A task-less lane record (RELEASED clears task_id) must produce NO
        # entry at all.
        lifecycle.note_assigned('_lane-9', task_id='99')
        lifecycle.transition('_lane-9', LaneState.RELEASED)

        result = build_task_runtime_snapshot(git_ops=git_ops)

        by_task = {r.task_id: r for r in result}
        assert set(by_task) == {42, 43, 44}, (
            f'expected exactly tasks {{42, 43, 44}} (task-less lane record '
            f'excluded); got {sorted(by_task)!r}'
        )

        entry_42 = by_task[42]
        assert entry_42.loops == 3
        assert entry_42.attempts == 1
        assert entry_42.lane == '_lane-3'
        assert entry_42.lane_state == 'assigned'

        entry_43 = by_task[43]
        assert entry_43.lane_state == 'quarantined'

        entry_44 = by_task[44]
        assert entry_44.loops == 0
        assert entry_44.error is None


# ---------------------------------------------------------------------------
# INV-2 structured-facts-at-failure: a per-task artifact READ FAILURE is
# marked (error set, loops/attempts/phase/started all None) — never coerced
# to a fabricated 0/'PLAN'. Honest-empty artifacts stay 0/'PLAN'/error=None.
# ---------------------------------------------------------------------------


class TestReadFailureMarked:
    def test_corrupt_plan_json_marks_error_not_zero(self, git_repo: Path):
        git_ops = GitOps(GitConfig(), git_repo)
        worktree_base = git_ops.worktree_base

        _make_task_artifacts(worktree_base, '50', '50')
        meta_root = TaskArtifacts.meta_root_for(worktree_base, '50')
        (meta_root / 'plan.json').write_text('{not valid json')

        result = build_task_runtime_snapshot(git_ops=git_ops)

        assert len(result) == 1
        entry = result[0]
        assert entry.task_id == 50
        assert entry.error is not None and entry.error != '', (
            f'expected a non-empty error diagnostic; got {entry.error!r}'
        )
        assert entry.loops is None
        assert entry.attempts is None
        assert entry.phase is None
        assert entry.started is None
        # lane/lane_state/has_worktree are sourced from the enumeration
        # itself (dir/record), not the failed artifact read — they must
        # remain populated even though the artifact read failed.
        assert entry.has_worktree is True
        assert entry.lane is None
        assert entry.lane_state is None

    def test_empty_iteration_log_is_honest_zero_not_a_failure(self, git_repo: Path):
        """Contrast case: a merely empty iterations.jsonl is honest-empty,
        not a read failure — loops stays 0 and error stays None."""
        git_ops = GitOps(GitConfig(), git_repo)
        worktree_base = git_ops.worktree_base

        _make_task_artifacts(worktree_base, '51', '51')
        meta_root = TaskArtifacts.meta_root_for(worktree_base, '51')
        (meta_root / 'iterations.jsonl').write_text('')

        result = build_task_runtime_snapshot(git_ops=git_ops)

        assert len(result) == 1
        entry = result[0]
        assert entry.task_id == 51
        assert entry.loops == 0
        assert entry.error is None


# ---------------------------------------------------------------------------
# `started` fallback: metadata.json created_at wins when present; an
# event_store's earliest task_started event is consulted only when
# created_at is absent; with no event_store and no created_at, started
# stays None.
# ---------------------------------------------------------------------------


def _bare_task_dir(worktree_base: Path, name: str, task_id: str) -> Path:
    """Create <worktree_base>/<name> plus a metadata.json with NO
    'created_at' key (distinct from _make_task_artifacts, which always
    stamps one via TaskArtifacts.init)."""
    worktree = worktree_base / name
    worktree.mkdir(parents=True, exist_ok=True)
    meta_root = TaskArtifacts.meta_root_for(worktree_base, name)
    meta_root.mkdir(parents=True, exist_ok=True)
    (meta_root / 'metadata.json').write_text(json.dumps({'task_id': task_id}))
    return worktree


class TestStartedEventStoreFallback:
    def test_started_falls_back_to_earliest_task_started_event(
        self, git_repo: Path, tmp_path: Path,
    ):
        git_ops = GitOps(GitConfig(), git_repo)
        worktree_base = git_ops.worktree_base
        _bare_task_dir(worktree_base, '60', '60')

        event_store = EventStore(tmp_path / 'events.db', run_id='run-1')
        event_store.emit(EventType.task_started, task_id='60')
        expected = event_store.fetch_events_by_type(EventType.task_started)[0]['timestamp']

        result = build_task_runtime_snapshot(git_ops=git_ops, event_store=event_store)

        assert len(result) == 1
        entry = result[0]
        assert entry.task_id == 60
        assert entry.started == expected

    def test_started_is_none_without_event_store_and_no_created_at(self, git_repo: Path):
        git_ops = GitOps(GitConfig(), git_repo)
        worktree_base = git_ops.worktree_base
        _bare_task_dir(worktree_base, '61', '61')

        result = build_task_runtime_snapshot(git_ops=git_ops, event_store=None)

        assert len(result) == 1
        assert result[0].started is None

    def test_created_at_wins_over_event_store(self, git_repo: Path, tmp_path: Path):
        git_ops = GitOps(GitConfig(), git_repo)
        worktree_base = git_ops.worktree_base
        ta = _make_task_artifacts(worktree_base, '62', '62')
        expected_created_at = ta.read_created_at()
        assert expected_created_at is not None

        event_store = EventStore(tmp_path / 'events2.db', run_id='run-1')
        event_store.emit(EventType.task_started, task_id='62')

        result = build_task_runtime_snapshot(git_ops=git_ops, event_store=event_store)

        assert len(result) == 1
        assert result[0].started == expected_created_at
