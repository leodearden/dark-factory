"""Over-cap warm-lane slots: the lanes a pool SHRINK leaves behind (task 5565).

A restart with a smaller ``max_concurrent_tasks + git.spare_warm_lanes`` builds
a pool of N slots, but lanes ``_lane-N`` and up stay on disk, some still
ASSIGNED to live tasks.  The pool recognises such a lane by NAME within its
namespace, admits it only through ``restore_assignment`` (so it is never FREE
and never handed to new work) and drains it on ``release``.

Pure state-machine tests: no git.  A real ``LaneLifecycle`` is wired only
where the durable mirror is asserted.
"""

from pathlib import Path

import pytest

from orchestrator.lane_lifecycle import LaneLifecycle, LaneRecord
from orchestrator.lane_lifecycle import LaneState as DurableLaneState
from orchestrator.warm_lane_pool import (
    LaneState,
    OverCapLane,
    WarmLanePool,
    WarmLanePoolShrink,
)


def _make_pool(tmp_path: Path, size: int, name_prefix: str = '_lane-') -> WarmLanePool:
    base = tmp_path / 'worktrees'
    base.mkdir(parents=True, exist_ok=True)
    return WarmLanePool(worktree_base=base, size=size, name_prefix=name_prefix)


def _reported_over_cap(pool: WarmLanePool, path: Path) -> bool:
    return bool(pool.over_cap_report([path], {}).lanes)


def _wire_lifecycle(pool: WarmLanePool, tmp_path: Path) -> LaneLifecycle:
    lifecycle = LaneLifecycle(worktree_base=tmp_path / 'worktrees')
    pool.set_lane_lifecycle(lifecycle)
    return lifecycle


class TestCapacity:
    def test_capacity_is_configured_size_and_fixed_after_admission(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=2)
        base = tmp_path / 'worktrees'
        assert pool.capacity == 2

        pool.restore_assignment('43', base / '_lane-5')

        assert pool.capacity == 2


class TestLaneRecognition:
    def test_configured_slot_is_lane_and_not_over_cap(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=2)
        base = tmp_path / 'worktrees'
        lane = base / '_lane-0'

        assert pool.is_lane(lane)
        assert not _reported_over_cap(pool, lane)
        assert pool.canonical_lane(lane) == lane

    def test_over_cap_slot_is_lane_and_over_cap(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=2)
        base = tmp_path / 'worktrees'
        lane = base / '_lane-5'

        assert pool.is_lane(lane)
        assert _reported_over_cap(pool, lane)
        assert pool.canonical_lane(lane) == base / '_lane-5'

    @pytest.mark.parametrize(
        'relative', ['_lane-x', '_lane-05', '_lane-', 'task-7', '_lane--3', '_lane-1a'],
    )
    def test_non_slot_names_are_not_lanes(self, tmp_path: Path, relative: str):
        pool = _make_pool(tmp_path, size=2)
        path = tmp_path / 'worktrees' / relative

        assert not pool.is_lane(path)
        assert not _reported_over_cap(pool, path)
        assert pool.canonical_lane(path) is None

    def test_same_name_under_another_base_is_not_a_lane(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=2)
        path = tmp_path / 'elsewhere' / '_lane-5'

        assert not pool.is_lane(path)
        assert not _reported_over_cap(pool, path)
        assert pool.canonical_lane(path) is None

    def test_prefix_scopes_the_namespace(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=2, name_prefix='_spec-')
        base = tmp_path / 'worktrees'

        assert pool.is_lane(base / '_spec-4')
        assert _reported_over_cap(pool, base / '_spec-4')
        assert not pool.is_lane(base / '_lane-4')


class TestOverCapAdmission:
    def test_restore_admits_over_cap_lane_as_assigned(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=2)
        lifecycle = _wire_lifecycle(pool, tmp_path)
        base = tmp_path / 'worktrees'
        lane = base / '_lane-5'

        pool.restore_assignment('43', lane)

        assert pool.state(lane) is LaneState.ASSIGNED
        assert pool.assignment_for('43') == lane
        assert pool.size == 3
        assert pool.lane_paths() == [base / '_lane-0', base / '_lane-1']
        record = lifecycle.read(lane)
        assert record is not None
        assert record.state is DurableLaneState.ASSIGNED
        assert record.task_id == '43'


class TestOverCapDrain:
    @pytest.mark.asyncio
    async def test_release_drains_admitted_over_cap_lane(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=2)
        lifecycle = _wire_lifecycle(pool, tmp_path)
        base = tmp_path / 'worktrees'
        lane = base / '_lane-5'
        pool.restore_assignment('43', lane)

        await pool.release(lane)

        assert pool.state(lane) is None
        assert pool.assignment_for('43') is None
        assert pool.size == 2
        record = lifecycle.read(lane)
        assert record is not None
        assert record.state is DurableLaneState.RELEASED

    @pytest.mark.asyncio
    async def test_drained_over_cap_lane_is_never_handed_out(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=2)
        base = tmp_path / 'worktrees'
        lane = base / '_lane-5'
        pool.restore_assignment('43', lane)
        await pool.release(lane)

        acquired = await pool.acquire_for('99')
        second = await pool.try_acquire()
        third = await pool.try_acquire()

        assert acquired == (base / '_lane-0', False)
        assert second == base / '_lane-1'
        assert third is None

    @pytest.mark.asyncio
    async def test_release_of_non_member_over_cap_lane_writes_released_mirror(
        self, tmp_path: Path,
    ):
        pool = _make_pool(tmp_path, size=2)
        lifecycle = _wire_lifecycle(pool, tmp_path)
        base = tmp_path / 'worktrees'
        lane = base / '_lane-5'
        lifecycle.note_assigned(lane, task_id='43')

        await pool.release(lane)

        record = lifecycle.read(lane)
        assert record is not None
        assert record.state is DurableLaneState.RELEASED
        assert pool.size == 2
        assert pool.state(lane) is None


class TestOverCapNeverStolen:
    @pytest.mark.asyncio
    async def test_reclaim_victim_skips_over_cap_lane(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=2)
        base = tmp_path / 'worktrees'
        over_cap = base / '_lane-5'
        pool.restore_assignment('43', over_cap)
        await pool.acquire_for('1')
        await pool.acquire_for('2')

        stolen = await pool.reclaim_victim(
            '7', candidates={'43'}, is_dispatched=lambda _branch: False,
        )

        assert stolen is None
        assert pool.assignment_for('43') == over_cap


class TestOverCapReport:
    def test_report_lists_resident_over_cap_lanes_with_records(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=2)
        base = tmp_path / 'worktrees'
        resident = [base / '_lane-0', base / '_lane-3', base / '_lane-2', base / 'task-7']
        records = {
            '_lane-2': LaneRecord(state=DurableLaneState.ASSIGNED, task_id='42'),
            '_lane-3': LaneRecord(state=DurableLaneState.RELEASED),
        }

        report = pool.over_cap_report(resident, records)

        assert isinstance(report, WarmLanePoolShrink)
        assert report.capacity == 2
        assert report.lanes == (
            OverCapLane(
                name='_lane-2', record_state=DurableLaneState.ASSIGNED, record_task_id='42',
            ),
            OverCapLane(
                name='_lane-3', record_state=DurableLaneState.RELEASED, record_task_id=None,
            ),
        )
        rendered = report.render()
        for fragment in ('capacity=2', '_lane-2', 'assigned', '42', '_lane-3', 'released'):
            assert fragment in rendered

    def test_resident_over_cap_lane_without_record_has_no_state(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=2)
        base = tmp_path / 'worktrees'

        report = pool.over_cap_report([base / '_lane-4'], {})

        assert report.lanes == (
            OverCapLane(name='_lane-4', record_state=None, record_task_id=None),
        )

    def test_undrained_keeps_lanes_that_may_hold_an_assignment(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=2)
        base = tmp_path / 'worktrees'
        records = {
            '_lane-2': LaneRecord(state=DurableLaneState.ASSIGNED, task_id='42'),
            '_lane-3': LaneRecord(state=DurableLaneState.RELEASED),
            '_lane-5': LaneRecord(state=DurableLaneState.IN_USE, task_id='45'),
            '_lane-6': LaneRecord(state=DurableLaneState.QUARANTINED),
        }
        resident = [base / f'_lane-{k}' for k in range(2, 7)]

        report = pool.over_cap_report(resident, records)

        undrained, drained = report.undrained(), report.drained()
        assert [lane.name for lane in undrained.lanes] == ['_lane-2', '_lane-4', '_lane-5']
        assert [lane.name for lane in drained.lanes] == ['_lane-3', '_lane-6']
        assert undrained.capacity == drained.capacity == 2

    def test_admitted_over_cap_lane_is_reported_in_index_order(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=2)
        base = tmp_path / 'worktrees'
        pool.restore_assignment('43', base / '_lane-10')

        report = pool.over_cap_report([base / '_lane-10', base / '_lane-9'], {})

        assert [lane.name for lane in report.lanes] == ['_lane-9', '_lane-10']

    def test_no_resident_over_cap_lane_gives_empty_report(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=2)
        base = tmp_path / 'worktrees'

        report = pool.over_cap_report([base / '_lane-0', base / '_lane-1'], {})

        assert report.capacity == 2
        assert report.lanes == ()
