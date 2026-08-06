"""B+H integration gate — consumer leg (B5-B6).

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


def _producer_wire_dict(
    *,
    offline: bool = False,
    offline_reason: str | None = None,
    tasks: list[dict] | None = None,
) -> dict:
    """The exact get_task_runtime_state JSON envelope shape.

    ``offline_reason`` is dashboard-synthesized (task 3517) — the producer
    itself always emits ``None`` — but it travels through this SAME envelope
    and must decode through the SAME shared contract, so it belongs here.
    """
    return {'offline': offline, 'offline_reason': offline_reason, 'tasks': tasks or []}


def _decode(wire: dict) -> TaskRuntimeSnapshot:
    """Decode a producer-shaped wire dict through the SAME shared contract
    the consumer join (``fetch_task_runtime``) decodes in production."""
    return TaskRuntimeSnapshot.model_validate(wire)


def _project_root(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    root.mkdir()
    return root


# ---------------------------------------------------------------------------
# B5 — Consumer join
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_b5_producer_wire_entry_populates_row_via_join(dummy_client, monkeypatch, tmp_path):
    """A decoded producer wire entry's loops/attempts/lane/phase/lane_state
    join onto the row: proves the join consumes a REAL decoded producer
    emission (not a hand-built TaskRuntimeEntry) end-to-end.
    """
    fixed = datetime(2026, 7, 16, 12, 0, 0, tzinfo=UTC)
    entry_started = (fixed - timedelta(minutes=7)).isoformat()
    wire = _producer_wire_dict(tasks=[
        _producer_wire_entry(
            42, loops=3, attempts=1, started=entry_started,
            lane='_lane-3', phase='EXECUTE', lane_state='assigned',
        ),
    ])
    snapshot = _decode(wire)

    _register_fetch_tasks(monkeypatch, [_shape_task(42, 'warm task', 'in-progress')])
    root = _project_root(tmp_path, 'warmlane')
    config = DashboardConfig(project_root=root)

    active, offline, _ = await _shape_one_project(
        dummy_client, config, root, runtime=snapshot, now=fixed,
    )

    assert offline is False
    assert len(active) == 1
    row = active[0]
    assert row['loops'] == 3
    assert row['attempts'] == 1
    assert row['lane'] == '_lane-3'
    assert row['phase'] == 'EXECUTE'
    assert row['lane_state'] == 'assigned'
    assert row['agent'] == 'claude-task-42'
    assert row['started'] == 7
    assert row['runtime_offline'] is False
    assert row['runtime_status'] == 'ok'


# ---------------------------------------------------------------------------
# B6 — Consumer offline degradation (+ honest-empty contrast)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_b6_offline_snapshot_yields_none_not_zero(dummy_client, monkeypatch, tmp_path):
    """runtime=TaskRuntimeSnapshot(offline=True) degrades every runtime-sourced
    field to None (never a fabricated 0) with runtime_offline=True.
    """
    _register_fetch_tasks(monkeypatch, [_shape_task(5, 'stuck task', 'in-progress')])
    root = _project_root(tmp_path, 'downlane')
    config = DashboardConfig(project_root=root)

    active, _, _ = await _shape_one_project(
        dummy_client, config, root, runtime=TaskRuntimeSnapshot(offline=True),
    )

    assert len(active) == 1
    row = active[0]
    assert row['runtime_offline'] is True
    for key in ('loops', 'attempts', 'started', 'lane', 'phase', 'lane_state'):
        assert row[key] is None, f'expected {key} is None when offline, got {row[key]!r}'
    # No reason on the snapshot -> the honest sentinel, not a guessed diagnosis.
    assert row['runtime_status'] == 'unknown'


@pytest.mark.asyncio
async def test_b6_wire_offline_reason_reaches_the_row(dummy_client, monkeypatch, tmp_path):
    """A wire envelope carrying offline_reason decodes through the SAME shared
    contract and lands on the row as runtime_status (task 3517).

    Ropes the new discriminator end-to-end — decode -> _probe_status ->
    _runtime_fields -> row — rather than only against hand-built models.
    """
    snapshot = _decode(_producer_wire_dict(
        offline=True, offline_reason='deadline_exceeded',
    ))
    assert snapshot.offline_reason == 'deadline_exceeded'

    _register_fetch_tasks(monkeypatch, [_shape_task(5, 'stuck task', 'in-progress')])
    root = _project_root(tmp_path, 'starvedlane')
    config = DashboardConfig(project_root=root)

    active, _, _ = await _shape_one_project(
        dummy_client, config, root, runtime=snapshot,
    )

    assert len(active) == 1
    row = active[0]
    assert row['runtime_status'] == 'deadline_exceeded'
    assert row['runtime_offline'] is True
    for key in ('loops', 'attempts', 'started', 'lane', 'phase', 'lane_state'):
        assert row[key] is None, f'expected {key} is None when offline, got {row[key]!r}'


@pytest.mark.asyncio
async def test_b6_online_but_task_absent_gets_honest_zero_contrast(dummy_client, monkeypatch, tmp_path):
    """Contrast case: an ONLINE snapshot that simply has no entry for the
    task (``TaskRuntimeSnapshot(tasks=[])``) gets honest 0s, not None --
    offline and honest-empty must stay distinguishable on the data contract
    that feeds rtCell (``dashboard/tests/js/runtime_format.test.mjs``:
    ``rtCell(None) -> '—'``, ``rtCell(0) -> 0`` -- the render leg,
    already surfaced under pytest via test_graph_layout_js.py's node --test
    wrapper).
    """
    _register_fetch_tasks(monkeypatch, [_shape_task(5, 'stuck task', 'in-progress')])
    root = _project_root(tmp_path, 'downlane-online')
    config = DashboardConfig(project_root=root)

    active, _, _ = await _shape_one_project(
        dummy_client, config, root, runtime=TaskRuntimeSnapshot(tasks=[]),
    )

    assert len(active) == 1
    row = active[0]
    assert row['runtime_offline'] is False
    assert row['runtime_status'] == 'ok'
    assert row['loops'] == 0
    assert row['attempts'] == 0
    assert row['started'] == 0
