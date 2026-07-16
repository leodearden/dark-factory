"""Tests for orchestrator.task_runtime — the free-function core behind
Harness.task_runtime_snapshot() (task 2634, PRD
plans/dashboard-task-runtime-endpoint-prd.md task alpha).
"""

from __future__ import annotations

import dataclasses

from orchestrator.lane_lifecycle import LaneState
from orchestrator.task_runtime import TaskRuntimeState, _derive_phase, _map_lane_state

# ---------------------------------------------------------------------------
# Pure helpers — _derive_phase / _map_lane_state / TaskRuntimeState shape
# ---------------------------------------------------------------------------


class TestPureHelpers:
    def test_derive_phase_empty_plan_is_plan(self):
        assert _derive_phase({}) == 'PLAN'

    def test_derive_phase_empty_steps_list_is_plan(self):
        assert _derive_phase({'steps': []}) == 'PLAN'

    def test_derive_phase_all_done_is_done(self):
        plan = {'steps': [{'status': 'done'}, {'status': 'done'}]}
        assert _derive_phase(plan) == 'DONE'

    def test_derive_phase_mixed_is_execute(self):
        plan = {'steps': [{'status': 'done'}, {'status': 'pending'}]}
        assert _derive_phase(plan) == 'EXECUTE'

    def test_map_lane_state_assigned(self):
        assert _map_lane_state(LaneState.ASSIGNED) == 'assigned'

    def test_map_lane_state_in_use(self):
        assert _map_lane_state(LaneState.IN_USE) == 'assigned'

    def test_map_lane_state_quarantined(self):
        assert _map_lane_state(LaneState.QUARANTINED) == 'quarantined'

    def test_map_lane_state_released(self):
        assert _map_lane_state(LaneState.RELEASED) == 'released'

    def test_map_lane_state_seed_is_none(self):
        assert _map_lane_state(LaneState.SEED) is None

    def test_map_lane_state_registered_is_none(self):
        assert _map_lane_state(LaneState.REGISTERED) is None


class TestTaskRuntimeStateDataclass:
    def test_is_a_dataclass_with_expected_fields(self):
        assert dataclasses.is_dataclass(TaskRuntimeState)
        field_names = {f.name for f in dataclasses.fields(TaskRuntimeState)}
        assert field_names == {
            'task_id', 'has_worktree', 'loops', 'attempts', 'started',
            'lane', 'phase', 'lane_state', 'error',
        }

    def test_error_defaults_to_none(self):
        state = TaskRuntimeState(
            task_id=1,
            has_worktree=True,
            loops=0,
            attempts=0,
            started=None,
            lane=None,
            phase='PLAN',
            lane_state=None,
        )
        assert state.error is None
