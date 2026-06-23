"""Tests for Scheduler deterministic-task additions (β).

Step-1: RED — Scheduler.is_deterministic staticmethod
Step-3: RED — _get_modules no-lock invariant (I4/B12) + eligibility unchanged (B1)
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Minimal task-dict helpers
# ---------------------------------------------------------------------------

def _det_task(task_id: str = '42', files: list | None = None) -> dict:
    """Return a task dict with metadata.task_kind='deterministic'."""
    md: dict = {'task_kind': 'deterministic'}
    if files is not None:
        md['files'] = files
    return {'id': task_id, 'title': 'Gate task', 'metadata': md}


def _normal_task(task_id: str = '43', files: list | None = None) -> dict:
    """Return a task dict with metadata.task_kind='normal'."""
    md: dict = {'task_kind': 'normal'}
    if files is not None:
        md['files'] = files
    return {'id': task_id, 'title': 'Normal task', 'metadata': md}


# ---------------------------------------------------------------------------
# Step-1: is_deterministic staticmethod
# (RED until step-2 adds the method)
# ---------------------------------------------------------------------------

class TestIsDeterministic:
    """Unit tests for Scheduler.is_deterministic."""

    def test_returns_true_for_deterministic_task_kind(self):
        """Returns True when metadata.task_kind == 'deterministic'."""
        from orchestrator.scheduler import Scheduler
        task = _det_task()
        assert Scheduler.is_deterministic(task) is True

    def test_returns_false_for_normal_task_kind(self):
        """Returns False when metadata.task_kind == 'normal'."""
        from orchestrator.scheduler import Scheduler
        task = _normal_task()
        assert Scheduler.is_deterministic(task) is False

    def test_returns_false_for_missing_task_kind(self):
        """Returns False when task_kind key is absent from metadata."""
        from orchestrator.scheduler import Scheduler
        task = {'id': '1', 'metadata': {'files': []}}
        assert Scheduler.is_deterministic(task) is False

    def test_returns_false_for_missing_metadata(self):
        """Returns False when metadata key is absent entirely."""
        from orchestrator.scheduler import Scheduler
        task = {'id': '1', 'title': 'No metadata'}
        assert Scheduler.is_deterministic(task) is False

    def test_returns_false_for_empty_metadata(self):
        """Returns False when metadata is an empty dict."""
        from orchestrator.scheduler import Scheduler
        task = {'id': '1', 'metadata': {}}
        assert Scheduler.is_deterministic(task) is False

    def test_returns_false_for_metadata_none(self):
        """Returns False when metadata is explicitly None."""
        from orchestrator.scheduler import Scheduler
        task = {'id': '1', 'metadata': None}
        assert Scheduler.is_deterministic(task) is False


# ---------------------------------------------------------------------------
# Step-3: _get_modules no-lock invariant (I4/B12) + eligibility unchanged (B1)
# (RED until step-4 adds the short-circuit in _get_modules)
# ---------------------------------------------------------------------------

@pytest.fixture
def real_scheduler(tmp_path: Path):
    """Return a minimal Scheduler with a real OrchestratorConfig."""
    from orchestrator.config import OrchestratorConfig
    from orchestrator.scheduler import Scheduler
    config = OrchestratorConfig(project_root=tmp_path)
    return Scheduler(config)


class TestGetModulesNoLock:
    """I4/B12: deterministic tasks get [] from _get_modules (no module lock)."""

    def test_det_task_no_files_returns_empty(self, real_scheduler):
        """Deterministic task with no metadata.files → [] (no synthetic fallback)."""
        task = _det_task(task_id='42')
        modules = real_scheduler._get_modules(task)
        # Fails until step-4 short-circuit: currently returns ['task-42']
        assert modules == []

    def test_det_task_with_files_still_returns_empty(self, real_scheduler):
        """Deterministic task WITH files still gets [] (gate has no file locks)."""
        task = _det_task(task_id='42', files=['orchestrator/harness.py'])
        modules = real_scheduler._get_modules(task)
        # Files on a deterministic task are irrelevant — always []
        assert modules == []

    def test_normal_task_no_files_returns_synthetic_fallback(self, real_scheduler):
        """Normal task without files still gets synthetic ['task-<id>'] fallback."""
        task = _normal_task(task_id='43')
        modules = real_scheduler._get_modules(task)
        assert modules == ['task-43']

    def test_normal_task_with_files_returns_file_modules(self, real_scheduler):
        """Normal task with files gets its file-derived module locks (unchanged)."""
        task = _normal_task(task_id='44', files=['orchestrator/harness.py'])
        modules = real_scheduler._get_modules(task)
        # files_to_modules derives at least one module from the file path
        assert len(modules) >= 1
        assert all(isinstance(m, str) for m in modules)


class TestAcquireEmptyLockSet:
    """B12: try_acquire with [] holds no module locks — does not block other tasks."""

    def test_after_acquire_empty_held_set(self, real_scheduler):
        """After try_acquire(task_id, []), lock_table holds empty set for task_id."""
        task = _det_task(task_id='42')
        modules = real_scheduler._get_modules(task)
        assert modules == []  # pre-condition (depends on step-4)
        result = real_scheduler.lock_table.try_acquire('42', modules)
        assert result is True  # empty acquire always succeeds
        # The held set for task_id is empty — no module locks taken (I4/B12)
        held = real_scheduler.lock_table._held.get('42', set())
        assert held == set()


class TestDepsSatisfiedDeterministicTask:
    """B1: deterministic task eligibility — deps-unsatisfied gate untouched."""

    def test_det_task_with_pending_dep_not_satisfied(self, real_scheduler):
        """Deterministic gate with one pending dep → _deps_satisfied returns False."""
        # Dep '10' has status 'pending' → not in TERMINAL_STATUSES → False
        task = _det_task(task_id='42')
        task['dependencies'] = [10]
        status_map = {'10': 'pending'}
        result = real_scheduler._deps_satisfied(task, status_map)
        assert result is False

    def test_det_task_with_done_dep_satisfied(self, real_scheduler):
        """Deterministic gate with done dep → _deps_satisfied returns True."""
        task = _det_task(task_id='42')
        task['dependencies'] = [10]
        status_map = {'10': 'done'}
        result = real_scheduler._deps_satisfied(task, status_map)
        assert result is True
