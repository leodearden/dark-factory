"""On-disk task builders shared by the task-runtime snapshot suites.

test_task_runtime.py, test_harness_task_runtime.py and
test_task_runtime_snapshot.py all lay out real worktrees and warm-lane
configs before reading them back through
orchestrator.task_runtime.build_task_runtime_snapshot.
"""

from __future__ import annotations

from pathlib import Path

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig


def make_task_artifacts(worktree_base: Path, name: str, task_id: str) -> TaskArtifacts:
    """Create <worktree_base>/<name> plus both of its .task-meta roots, as a dispatch does."""
    worktree = worktree_base / name
    worktree.mkdir(parents=True, exist_ok=True)
    ta = TaskArtifacts.for_task(worktree, task_id)
    ta.init(task_id, f'Task {task_id}', 'desc')
    return ta


def warm_config(**overrides) -> GitConfig:
    """A GitConfig with the warm-lane pool enabled and no push after advance."""
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        warm_lane_pool=True,
        **overrides,
    )
