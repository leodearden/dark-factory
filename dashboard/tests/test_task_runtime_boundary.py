"""B+H integration gate — consumer leg (B5-B7).

Task 2638 (epsilon), PRD ``plans/dashboard-task-runtime-endpoint-prd.md``.
Feeds a DECODED producer-shaped wire payload —
``TaskRuntimeSnapshot.model_validate`` of the exact ``get_task_runtime_state``
envelope shape — into ``_shape_one_project``, roping the SAME shared contract
(``shared.task_runtime_state``) the producer side
(``orchestrator/tests/test_task_runtime_snapshot.py``) emits onto the wire.
Existing consumer unit tests (``test_active_tasks.py``) construct
``TaskRuntimeEntry``/``TaskRuntimeSnapshot`` by hand and never decode a real
producer emission — so a producer/consumer contract drift would not be
caught there.

No product code (alpha/beta/gamma/delta) is modified — all four are already
merged and green; this is a pure characterization/integration suite (every
test passes on arrival).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from shared.task_runtime_state import TaskRuntimeSnapshot

import dashboard.data.active_tasks as active_tasks_mod
import dashboard.data.orchestrator as dash_orch
from dashboard.config import DashboardConfig
from dashboard.data.active_tasks import _shape_one_project

# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------


def _shape_task(task_id: int, title: str, status: str) -> dict:
    """Minimal dashboard-shaped task dict (mirrors test_active_tasks.py's _shape_task)."""
    return {
        'id': task_id,
        'title': title,
        'description': '',
        'details': '',
        'status': status,
        'priority': None,
        'dependencies': [],
        'metadata': {},
        'updated_at': None,
    }


def _register_fetch_tasks(monkeypatch, tasks: list[dict]) -> None:
    """Monkeypatch fetch_tasks to return a fixed dashboard-shaped task list."""

    async def _fake_fetch_tasks(client, config, project_root):
        return list(tasks)

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_tasks', _fake_fetch_tasks)


def _producer_wire_entry(
    task_id: int,
    *,
    has_worktree: bool = True,
    loops: int | None = 0,
    attempts: int | None = 0,
    started: str | None = None,
    lane: str | None = None,
    phase: str | None = None,
    lane_state: str | None = None,
    error: str | None = None,
) -> dict:
    """One entry in the get_task_runtime_state wire envelope's 'tasks' list."""
    return {
        'task_id': task_id, 'has_worktree': has_worktree, 'loops': loops,
        'attempts': attempts, 'started': started, 'lane': lane, 'phase': phase,
        'lane_state': lane_state, 'error': error,
    }


def _producer_wire_dict(*, offline: bool = False, tasks: list[dict] | None = None) -> dict:
    """The exact get_task_runtime_state JSON envelope shape."""
    return {'offline': offline, 'tasks': tasks or []}


def _decode(wire: dict) -> TaskRuntimeSnapshot:
    """Decode a producer-shaped wire dict through the SAME shared contract
    the consumer join (``fetch_task_runtime``) decodes in production."""
    return TaskRuntimeSnapshot.model_validate(wire)


def _project_root(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    root.mkdir()
    return root
