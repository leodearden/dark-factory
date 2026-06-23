"""Tests for Scheduler deterministic-task additions (β).

Step-1: RED — Scheduler.is_deterministic staticmethod
Step-3: RED — _get_modules no-lock invariant (I4/B12) + eligibility unchanged (B1)
"""

from __future__ import annotations

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
