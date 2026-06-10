"""Tests for the β train former — selection, stackability, and formation.

Covers:
  step-7:  _line_ranges_stackable (pure predicate)
  step-9:  _select_train_members (pure selection)
  step-11: _train_candidates (async candidate discovery)
  step-13: _maybe_form_train guards (disabled, lone-task, already-member)
  step-15: _maybe_form_train formation, overlap-rejection, and cap
  step-17: _maybe_defer_as_train_member routing helper
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


# ---------------------------------------------------------------------------
# Fixture helpers (mirrors test_workflow_train_completion._make)
# ---------------------------------------------------------------------------

@dataclass
class _Fixture:
    wf: TaskWorkflow
    scheduler: MagicMock
    git_ops: MagicMock


def _make(
    *,
    task_id: str = '200',
    metadata: dict | None = None,
    former_enabled: bool = False,
    max_members: int = 3,
    get_tasks_return: list[dict] | None = None,
) -> _Fixture:
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {
        'id': task_id,
        'title': 'Former test task',
        'description': 'd',
        'metadata': metadata or {},
        'status': 'in-progress',
    }
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = Path('/tmp/non-existent-for-test')
    config.max_consecutive_infra_resumes = 3
    config.max_consecutive_merge_thrash = 3
    config.merge_train_former_enabled = former_enabled
    config.merge_train_max_members = max_members
    config.main_branch = 'main'

    scheduler = MagicMock()
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.set_task_status = AsyncMock()
    scheduler.get_status = AsyncMock(return_value='in-progress')
    scheduler.get_task = AsyncMock(return_value=assignment.task)
    scheduler.mark_done = AsyncMock()
    scheduler.clear_requeue_count = MagicMock()
    scheduler.tasks_by_train = AsyncMock(return_value=[])
    scheduler.get_statuses = AsyncMock(return_value=({}, None))

    if get_tasks_return is not None:
        scheduler.get_tasks = AsyncMock(return_value=get_tasks_return)
    else:
        scheduler.get_tasks = AsyncMock(return_value=[])

    git_ops = MagicMock()
    git_ops.config.branch_prefix = 'task/'
    git_ops.config.main_branch = 'main'
    git_ops.resolve_branch_sha = AsyncMock(return_value='abc123')
    git_ops.get_changed_line_ranges = AsyncMock(return_value={})

    esc_queue = MagicMock()
    esc_queue.has_open_l1 = MagicMock(return_value=False)
    esc_queue.make_id = MagicMock(return_value=f'esc-{task_id}-1')
    esc_queue.submit = MagicMock()
    esc_queue.get_by_task = MagicMock(return_value=[])

    merge_queue: asyncio.Queue = asyncio.Queue()

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
        escalation_queue=esc_queue,  # type: ignore[arg-type]
        merge_queue=merge_queue,
    )

    wf.artifacts = MagicMock()
    wf.artifacts.read_iteration_log = MagicMock(return_value=([], []))
    wf.worktree = Path(f'/tmp/wt-{task_id}')
    wf.event_store = None

    return _Fixture(wf=wf, scheduler=scheduler, git_ops=git_ops)


# ---------------------------------------------------------------------------
# step-7: _line_ranges_stackable — pure predicate tests
# ---------------------------------------------------------------------------


class TestLineRangesStackable:
    """Unit tests for the _line_ranges_stackable pure predicate in workflow.py."""

    def _fn(self):
        from orchestrator.workflow import _line_ranges_stackable
        return _line_ranges_stackable

    def test_disjoint_files_are_stackable(self):
        """Two tasks touching completely different files are always stackable."""
        fn = self._fn()
        a = {'src/foo.rs': [(1, 10)]}
        b = {'src/bar.rs': [(1, 10)]}
        assert fn(a, b) is True

    def test_same_file_non_overlapping_ranges_are_stackable(self):
        """Core stackable invariant: same file, different non-overlapping lines."""
        fn = self._fn()
        # Task A modifies lines 10-20 of foo.rs
        a = {'src/foo.rs': [(10, 20)]}
        # Task B modifies lines 30-40 of foo.rs — different lines, stackable
        b = {'src/foo.rs': [(30, 40)]}
        assert fn(a, b) is True

    def test_same_file_overlapping_ranges_not_stackable(self):
        """Same file, ranges overlap → not stackable."""
        fn = self._fn()
        a = {'src/foo.rs': [(10, 25)]}
        b = {'src/foo.rs': [(20, 35)]}
        assert fn(a, b) is False

    def test_same_file_identical_ranges_not_stackable(self):
        """Same file, identical ranges (extreme overlap) → not stackable."""
        fn = self._fn()
        a = {'src/foo.rs': [(10, 20)]}
        b = {'src/foo.rs': [(10, 20)]}
        assert fn(a, b) is False

    def test_one_side_empty_is_stackable(self):
        """A task touching no files is trivially stackable with anything."""
        fn = self._fn()
        a = {'src/foo.rs': [(10, 20)]}
        b: dict = {}
        assert fn(a, b) is True
        assert fn(b, a) is True

    def test_both_empty_is_stackable(self):
        fn = self._fn()
        assert fn({}, {}) is True

    def test_adjacent_ranges_are_stackable(self):
        """Ranges that abut (end of A + 1 == start of B) are non-overlapping."""
        fn = self._fn()
        a = {'src/foo.rs': [(10, 20)]}
        b = {'src/foo.rs': [(21, 30)]}
        assert fn(a, b) is True

    def test_shared_file_with_other_disjoint_file_not_stackable_when_overlap(self):
        """Overlapping shared file makes pair not stackable even if another file is disjoint."""
        fn = self._fn()
        a = {'src/foo.rs': [(10, 20)], 'src/bar.rs': [(1, 5)]}
        b = {'src/foo.rs': [(15, 25)], 'src/baz.rs': [(100, 110)]}
        assert fn(a, b) is False
