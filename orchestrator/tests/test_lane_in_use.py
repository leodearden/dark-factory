"""Leaf δ (task 3076): ``LaneState.IN_USE`` is actually written.

Before δ, IN_USE was a declared state with no writer. Now a lane reads IN_USE
while dark-factory has a live child rooted in it and steps back to ASSIGNED
when the last one exits. Which children count, and why it is a span at
dark-factory's own spawn points rather than a /proc scan, is decided in
``plans/warm-lane-infra-repatriation-prd.md`` §11 q2.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest
from escalation.queue import EscalationQueue

from orchestrator.lane_lifecycle import LaneLifecycle, LaneState

LANE_NAME = '_lane-0'


def _record_path(worktree_base: Path, lane_name: str = LANE_NAME) -> Path:
    return worktree_base / '.lane-state' / f'{lane_name}.json'


def _on_disk(worktree_base: Path, lane_name: str = LANE_NAME) -> dict:
    return json.loads(_record_path(worktree_base, lane_name).read_text())


def _legal_ladder(task_id: str) -> list[tuple[LaneState, dict[str, str]]]:
    return [
        (LaneState.SEED, {'seeded_from_sha': 'abc123'}),
        (LaneState.REGISTERED, {'branch': f'task/{task_id}'}),
        (LaneState.ASSIGNED, {'task_id': task_id, 'title': f'demo-{task_id}'}),
        (LaneState.IN_USE, {}),
        (LaneState.RELEASED, {}),
    ]


def _drive_to(
    lifecycle: LaneLifecycle, lane: Path, target: LaneState, task_id: str = '42',
) -> None:
    """Bring *lane*'s record to *target* through public, legal transitions only."""
    if target is LaneState.QUARANTINED:
        _drive_to(lifecycle, lane, LaneState.ASSIGNED, task_id)
        lifecycle.transition(lane, LaneState.QUARANTINED)
        return
    for state, fields in _legal_ladder(task_id):
        lifecycle.transition(lane, state, **fields)
        if state is target:
            return


@pytest.fixture
def lifecycle(tmp_path: Path) -> LaneLifecycle:
    return LaneLifecycle(worktree_base=tmp_path)


@pytest.fixture
def lane(tmp_path: Path) -> Path:
    return tmp_path / LANE_NAME


class TestNoteInUseAndIdle:
    def test_assigned_lane_flips_to_in_use_and_back_keeping_the_assignment(
        self, tmp_path: Path, lifecycle: LaneLifecycle, lane: Path,
    ):
        _drive_to(lifecycle, lane, LaneState.ASSIGNED)
        assigned = _on_disk(tmp_path)

        assert lifecycle.note_in_use(lane) is True
        in_use = _on_disk(tmp_path)
        assert in_use['state'] == 'in_use'
        assert (in_use['task_id'], in_use['title'], in_use['branch']) == (
            assigned['task_id'], assigned['title'], assigned['branch'],
        ) == ('42', 'demo-42', 'task/42')

        assert lifecycle.note_idle(lane) is True
        idle = _on_disk(tmp_path)
        assert idle['state'] == 'assigned'
        assert idle['task_id'] == '42'

    @pytest.mark.parametrize(
        'state',
        [LaneState.RELEASED, LaneState.REGISTERED, LaneState.SEED, LaneState.QUARANTINED],
    )
    def test_note_in_use_writes_nothing_on_a_lane_with_no_live_assignment(
        self, tmp_path: Path, lifecycle: LaneLifecycle, lane: Path, state: LaneState,
    ):
        _drive_to(lifecycle, lane, state)
        before = _record_path(tmp_path).read_bytes()

        assert lifecycle.note_in_use(lane) is False

        assert _record_path(tmp_path).read_bytes() == before

    def test_note_in_use_on_an_absent_record_creates_nothing(
        self, tmp_path: Path, lifecycle: LaneLifecycle, lane: Path,
    ):
        assert lifecycle.note_in_use(lane) is False

        assert not _record_path(tmp_path).exists()

    def test_note_in_use_on_a_corrupt_record_writes_and_escalates_nothing(
        self, tmp_path: Path, lane: Path,
    ):
        queue = EscalationQueue(tmp_path / 'escalations')
        lifecycle = LaneLifecycle(worktree_base=tmp_path, escalation_queue=queue)
        record_path = _record_path(tmp_path)
        record_path.parent.mkdir(parents=True)
        record_path.write_text('{not valid json')
        before = record_path.read_bytes()

        assert lifecycle.note_in_use(lane) is False

        assert record_path.read_bytes() == before
        assert queue.get_pending() == []

    @pytest.mark.parametrize(
        'state',
        [
            LaneState.ASSIGNED, LaneState.RELEASED, LaneState.REGISTERED,
            LaneState.SEED, LaneState.QUARANTINED,
        ],
    )
    def test_note_idle_writes_nothing_unless_the_lane_reads_in_use(
        self, tmp_path: Path, lifecycle: LaneLifecycle, lane: Path, state: LaneState,
    ):
        _drive_to(lifecycle, lane, state)
        before = _record_path(tmp_path).read_bytes()

        assert lifecycle.note_idle(lane) is False

        assert _record_path(tmp_path).read_bytes() == before

    def test_note_idle_on_an_absent_record_creates_nothing(
        self, tmp_path: Path, lifecycle: LaneLifecycle, lane: Path,
    ):
        assert lifecycle.note_idle(lane) is False

        assert not _record_path(tmp_path).exists()

    def test_note_in_use_on_an_in_use_lane_is_true_without_a_write(
        self, tmp_path: Path, lifecycle: LaneLifecycle, lane: Path,
    ):
        _drive_to(lifecycle, lane, LaneState.IN_USE)
        before = _record_path(tmp_path).read_bytes()

        assert lifecycle.note_in_use(lane) is True

        assert _record_path(tmp_path).read_bytes() == before


@pytest.mark.skipif(os.geteuid() == 0, reason='root ignores directory write bits')
class TestNeverRaises:
    """A liveness marker must never abort the child-spawning caller.

    The fault is a real one: the ``.lane-state`` dir is made read-only, so
    the atomic write itself raises ``PermissionError``.
    """

    def _assert_swallowed(
        self, tmp_path: Path, caplog, call, expected_state: str,
    ) -> None:
        state_dir = tmp_path / '.lane-state'
        state_dir.chmod(0o500)
        try:
            with caplog.at_level(logging.WARNING, logger='orchestrator.lane_lifecycle'):
                assert call() is False
        finally:
            state_dir.chmod(0o700)
        assert _on_disk(tmp_path)['state'] == expected_state
        assert any(
            r.levelno == logging.WARNING and LANE_NAME in r.getMessage()
            for r in caplog.records
        ), 'the swallowed write failure must be logged naming the lane'

    def test_note_in_use_write_failure_returns_false_and_warns(
        self, tmp_path: Path, lifecycle: LaneLifecycle, lane: Path, caplog,
    ):
        _drive_to(lifecycle, lane, LaneState.ASSIGNED)

        self._assert_swallowed(
            tmp_path, caplog, lambda: lifecycle.note_in_use(lane), 'assigned',
        )

    def test_note_idle_write_failure_returns_false_and_warns(
        self, tmp_path: Path, lifecycle: LaneLifecycle, lane: Path, caplog,
    ):
        _drive_to(lifecycle, lane, LaneState.IN_USE)

        self._assert_swallowed(
            tmp_path, caplog, lambda: lifecycle.note_idle(lane), 'in_use',
        )


class TestResetStaleInUse:
    """A crash mid-span leaves IN_USE fossils; the sweep steps them back."""

    STALE = {'_lane-0': '42', '_lane-1': '43'}
    UNTOUCHED = ('_lane-2', '_lane-3', '_lane-4', '_lane-5')

    @pytest.fixture
    def queue(self, tmp_path: Path) -> EscalationQueue:
        return EscalationQueue(tmp_path / 'escalations')

    @pytest.fixture
    def swept(self, tmp_path: Path, queue: EscalationQueue) -> LaneLifecycle:
        lifecycle = LaneLifecycle(worktree_base=tmp_path, escalation_queue=queue)
        for lane_name, task_id in self.STALE.items():
            _drive_to(lifecycle, tmp_path / lane_name, LaneState.IN_USE, task_id)
        _drive_to(lifecycle, tmp_path / '_lane-2', LaneState.ASSIGNED, '44')
        _drive_to(lifecycle, tmp_path / '_lane-3', LaneState.RELEASED, '45')
        _drive_to(lifecycle, tmp_path / '_lane-4', LaneState.QUARANTINED, '46')
        _record_path(tmp_path, '_lane-5').write_text('{not valid json')
        return lifecycle

    def _snapshot(self, tmp_path: Path, lane_names) -> dict[str, bytes]:
        return {name: _record_path(tmp_path, name).read_bytes() for name in lane_names}

    def test_resets_exactly_the_in_use_lanes_keeping_their_assignments(
        self, tmp_path: Path, swept: LaneLifecycle, queue: EscalationQueue, caplog,
    ):
        before = {name: _on_disk(tmp_path, name) for name in self.STALE}
        untouched = self._snapshot(tmp_path, self.UNTOUCHED)

        with caplog.at_level(logging.INFO, logger='orchestrator.lane_lifecycle'):
            reset = swept.reset_stale_in_use()

        assert reset == ['_lane-0', '_lane-1']
        for name, task_id in self.STALE.items():
            after = _on_disk(tmp_path, name)
            assert after['state'] == 'assigned'
            assert (after['task_id'], after['title'], after['branch']) == (
                before[name]['task_id'], before[name]['title'], before[name]['branch'],
            ) == (task_id, f'demo-{task_id}', f'task/{task_id}')
            assert [
                r for r in caplog.records
                if r.levelno == logging.INFO
                and name in r.getMessage() and task_id in r.getMessage()
            ], f'expected an INFO line naming the reset of {name} (task {task_id})'
        assert self._snapshot(tmp_path, self.UNTOUCHED) == untouched
        assert queue.get_pending() == []

    def test_a_lane_in_the_skip_set_keeps_its_live_in_use_record(
        self, tmp_path: Path, swept: LaneLifecycle,
    ):
        reset = swept.reset_stale_in_use(skip=['_lane-1'])

        assert reset == ['_lane-0']
        assert _on_disk(tmp_path, '_lane-1')['state'] == 'in_use'

    def test_a_second_sweep_is_a_no_op(self, tmp_path: Path, swept: LaneLifecycle):
        swept.reset_stale_in_use()
        settled = self._snapshot(tmp_path, [*self.STALE, *self.UNTOUCHED])

        assert swept.reset_stale_in_use() == []

        assert self._snapshot(tmp_path, [*self.STALE, *self.UNTOUCHED]) == settled
