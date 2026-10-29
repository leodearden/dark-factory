"""What a warm-lane acquire does about a task branch already checked out somewhere (task 5565).

Pure tests: the decision table and the typed refusal carry no git and no I/O.
"""

from pathlib import Path

import pytest

from orchestrator.lane_branch_holder import (
    BranchCheckedOutElsewhere,
    BranchHolderAction,
    decide_branch_holder,
)
from orchestrator.lane_lifecycle import LaneRecord
from orchestrator.lane_lifecycle import LaneState as DurableLaneState

_LANE = Path('/pool/_lane-3')


class TestDecideBranchHolder:
    def test_holder_outside_the_pool_is_refused_as_foreign(self):
        action = decide_branch_holder(task_id='43', lane=None, lane_held=False, record=None)

        assert action is BranchHolderAction.REFUSE_FOREIGN_WORKTREE

    def test_lane_the_pool_holds_is_refused_even_when_its_record_names_this_task(self):
        record = LaneRecord(state=DurableLaneState.ASSIGNED, task_id='43')

        action = decide_branch_holder(task_id='43', lane=_LANE, lane_held=True, record=record)

        assert action is BranchHolderAction.REFUSE_HELD_LANE

    @pytest.mark.parametrize('state', [DurableLaneState.ASSIGNED, DurableLaneState.IN_USE])
    def test_unheld_lane_durably_owned_by_this_task_is_reused(self, state: DurableLaneState):
        record = LaneRecord(state=state, task_id='43')

        action = decide_branch_holder(task_id='43', lane=_LANE, lane_held=False, record=record)

        assert action is BranchHolderAction.REUSE_OWN_LANE

    @pytest.mark.parametrize(
        'record',
        [
            pytest.param(LaneRecord(state=DurableLaneState.ASSIGNED, task_id='99'), id='other-task'),
            pytest.param(LaneRecord(state=DurableLaneState.RELEASED), id='released'),
            pytest.param(LaneRecord(state=DurableLaneState.QUARANTINED), id='quarantined'),
            pytest.param(None, id='no-record'),
        ],
    )
    def test_unheld_lane_not_owned_by_this_task_is_detached(self, record: LaneRecord | None):
        action = decide_branch_holder(task_id='43', lane=_LANE, lane_held=False, record=record)

        assert action is BranchHolderAction.DETACH_OWNERLESS_LANE


class TestBranchCheckedOutElsewhere:
    def test_foreign_refusal_carries_branch_holder_and_action(self):
        exc = BranchCheckedOutElsewhere(
            branch='task/77',
            holder=Path('/x/side'),
            action=BranchHolderAction.REFUSE_FOREIGN_WORKTREE,
        )

        assert isinstance(exc, RuntimeError)
        assert exc.branch == 'task/77'
        assert exc.holder == Path('/x/side')
        assert exc.action is BranchHolderAction.REFUSE_FOREIGN_WORKTREE
        message = str(exc)
        assert 'task/77' in message
        assert '/x/side' in message
        assert 'no lane was touched' in message
        assert 'git -C /x/side checkout --detach' in message

    def test_held_lane_refusal_says_no_lane_was_touched(self):
        exc = BranchCheckedOutElsewhere(
            branch='task/99',
            holder=Path('/pool/_lane-1'),
            action=BranchHolderAction.REFUSE_HELD_LANE,
        )

        message = str(exc)
        assert 'task/99' in message
        assert '/pool/_lane-1' in message
        assert 'no lane was touched' in message

    def test_failed_detach_says_the_lane_stays_pinned_to_the_task(self):
        exc = BranchCheckedOutElsewhere(
            branch='task/77',
            holder=Path('/pool/_lane-3'),
            action=BranchHolderAction.DETACH_OWNERLESS_LANE,
        )

        message = str(exc)
        assert 'task/77' in message
        assert '/pool/_lane-3' in message
        assert 'detach' in message
        assert 'failed' in message
        assert 'stays pinned' in message

    def test_reuse_is_not_a_refusal(self):
        with pytest.raises(ValueError, match='reuse_own_lane'):
            BranchCheckedOutElsewhere(
                branch='task/43',
                holder=_LANE,
                action=BranchHolderAction.REUSE_OWN_LANE,
            )
