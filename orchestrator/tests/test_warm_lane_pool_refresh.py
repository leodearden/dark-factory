"""WarmLanePool.refresh_assignment, the REUSE-dispatch activity stamp (task 7863).

A lane record's ``updated_at`` is the age clock warm-lane-gc.sh's record gate
reads. ``refresh_assignment`` restamps it through the pool, the single durable
writer, for a task that already holds the lane, so an actively re-dispatched
task keeps its GC protection. ``restore_assignment`` (crash recovery, run on
every fleet restart) must never restamp, or abandoned lanes would stay young
forever. A refused refresh is drift: logged, counted toward the pool's L2
callback, and never raised. Self-contained: no conftest edit.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from orchestrator.lane_lifecycle import LaneLifecycle, LaneRecord
from orchestrator.lane_lifecycle import LaneState as DurableLaneState
from orchestrator.warm_lane_pool import WarmLanePool


def _backdate(lifecycle: LaneLifecycle, lane: Path, *, days: int) -> str:
    """Rewrite *lane*'s on-disk updated_at to *days* ago; return that stamp."""
    path = lifecycle.state_dir / f'{lane.name}.json'
    record = LaneRecord.from_json(path.read_text(encoding='utf-8'))
    stamp = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    path.write_text(replace(record, updated_at=stamp).to_json(), encoding='utf-8')
    return stamp


def _assigned_lane(base: Path, lifecycle: LaneLifecycle, task_id: str) -> Path:
    lane = base / '_lane-0'
    lifecycle.transition(lane, DurableLaneState.SEED, seeded_from_sha='abc')
    lifecycle.transition(lane, DurableLaneState.REGISTERED, branch=f'task/{task_id}')
    lifecycle.transition(lane, DurableLaneState.ASSIGNED, task_id=task_id)
    return lane


def _record_path(lifecycle: LaneLifecycle, lane: Path) -> Path:
    return lifecycle.state_dir / f'{lane.name}.json'


def _wired_pool(
    base: Path, lifecycle: LaneLifecycle, *, drift_l2_threshold: int = 3,
) -> WarmLanePool:
    pool = WarmLanePool(worktree_base=base, size=2, drift_l2_threshold=drift_l2_threshold)
    pool.set_lane_lifecycle(lifecycle)
    return pool


def test_refresh_assignment_restamps_a_held_record(tmp_path: Path):
    lifecycle = LaneLifecycle(worktree_base=tmp_path)
    lane = _assigned_lane(tmp_path, lifecycle, '42')
    old = _backdate(lifecycle, lane, days=30)
    pool = _wired_pool(tmp_path, lifecycle)
    pool.restore_assignment('42', lane)
    cache_before = (pool.state(lane), pool.assignment_for('42'))

    pool.refresh_assignment('42', lane)

    record = lifecycle.read(lane)
    assert record is not None
    refreshed_at = datetime.fromisoformat(record.updated_at)
    assert refreshed_at > datetime.fromisoformat(old)
    assert abs(datetime.now(UTC) - refreshed_at) < timedelta(minutes=5)
    assert (record.state, record.task_id) == (DurableLaneState.ASSIGNED, '42')
    assert (pool.state(lane), pool.assignment_for('42')) == cache_before


def test_restore_assignment_never_restamps(tmp_path: Path):
    lifecycle = LaneLifecycle(worktree_base=tmp_path)
    lane = _assigned_lane(tmp_path, lifecycle, '42')
    _backdate(lifecycle, lane, days=30)
    before_bytes = _record_path(lifecycle, lane).read_bytes()
    pool = _wired_pool(tmp_path, lifecycle)

    pool.restore_assignment('42', lane)

    assert _record_path(lifecycle, lane).read_bytes() == before_bytes


def test_refresh_assignment_without_lifecycle_is_a_noop(tmp_path: Path):
    pool = WarmLanePool(worktree_base=tmp_path, size=2)

    pool.refresh_assignment('42', tmp_path / '_lane-0')

    assert not (tmp_path / '.lane-state').exists()


def test_refused_refresh_is_loud_drift_and_never_raises(tmp_path: Path):
    lifecycle = LaneLifecycle(worktree_base=tmp_path)
    lane = _assigned_lane(tmp_path, lifecycle, '99')
    before_bytes = _record_path(lifecycle, lane).read_bytes()
    pool = _wired_pool(tmp_path, lifecycle, drift_l2_threshold=1)
    drift_calls: list[int] = []
    pool.set_on_lane_record_drift(drift_calls.append)

    pool.refresh_assignment('42', lane)

    assert _record_path(lifecycle, lane).read_bytes() == before_bytes
    assert drift_calls == [1]
