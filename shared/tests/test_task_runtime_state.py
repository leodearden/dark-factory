"""Tests for shared.task_runtime_state — the TaskRuntimeEntry/TaskRuntimeSnapshot
wire-contract model.

PRD ``plans/dashboard-task-runtime-endpoint-prd.md`` task beta (Open-Q2).
This is the ONE machine-checked contract (INV-1 contracts-machine-checked)
shared by the escalation ``get_task_runtime_state`` tool (beta) and the
dashboard join (gamma) — INV-5 no-lockstep-duplication.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.task_runtime_state import TaskRuntimeEntry, TaskRuntimeSnapshot


class TestTaskRuntimeEntry:
    """Construction + validation of the per-task wire entry."""

    def test_success_entry_round_trips(self):
        """A full SUCCESS entry's fields are preserved as-given."""
        entry = TaskRuntimeEntry(
            task_id=42,
            has_worktree=True,
            loops=3,
            attempts=1,
            started='2026-07-16T00:00:00+00:00',
            lane='_lane-3',
            phase='EXECUTE',
            lane_state='assigned',
            error=None,
        )
        assert entry.task_id == 42
        assert entry.has_worktree is True
        assert entry.loops == 3
        assert entry.attempts == 1
        assert entry.started == '2026-07-16T00:00:00+00:00'
        assert entry.lane == '_lane-3'
        assert entry.phase == 'EXECUTE'
        assert entry.lane_state == 'assigned'
        assert entry.error is None

    def test_honest_failure_preserves_none_not_coerced(self):
        """A read-failure entry keeps loops/attempts/phase/started as None
        (never fabricated to 0/'PLAN') and carries the diagnostic in error."""
        entry = TaskRuntimeEntry(
            task_id=7,
            has_worktree=True,
            loops=None,
            attempts=None,
            started=None,
            lane='_lane-1',
            phase=None,
            lane_state='assigned',
            error='ValueError: bad plan.json',
        )
        assert entry.loops is None
        assert entry.attempts is None
        assert entry.phase is None
        assert entry.started is None
        assert entry.error == 'ValueError: bad plan.json'

    def test_non_pooled_entry_has_no_lane(self):
        """A non-pooled task reports lane=None, lane_state=None (distinct
        from an honest-empty pooled task)."""
        entry = TaskRuntimeEntry(
            task_id=8,
            has_worktree=True,
            loops=2,
            attempts=0,
            started='2026-07-16T00:00:00+00:00',
            lane=None,
            phase='PLAN',
            lane_state=None,
            error=None,
        )
        assert entry.lane is None
        assert entry.lane_state is None

    def test_bogus_phase_raises_validation_error(self):
        """phase is a machine-checked enum — an out-of-vocab value raises,
        it is not silently accepted as a free-form string."""
        with pytest.raises(ValidationError):
            TaskRuntimeEntry(
                task_id=1,
                has_worktree=False,
                loops=0,
                attempts=0,
                started=None,
                lane=None,
                phase='BOGUS',  # type: ignore[arg-type]
                lane_state=None,
                error=None,
            )

    def test_bogus_lane_state_raises_validation_error(self):
        """lane_state is a machine-checked enum — an out-of-vocab value
        raises."""
        with pytest.raises(ValidationError):
            TaskRuntimeEntry(
                task_id=1,
                has_worktree=False,
                loops=0,
                attempts=0,
                started=None,
                lane='_lane-1',
                phase=None,
                lane_state='weird',  # type: ignore[arg-type]
                error=None,
            )


class TestTaskRuntimeSnapshot:
    """Defaults + round-trip serialization of the snapshot envelope."""

    def test_defaults(self):
        snapshot = TaskRuntimeSnapshot()
        assert snapshot.offline is False
        assert snapshot.tasks == []
        # offline_reason is the same species as offline: always None as
        # emitted by the server, synthesized client-side by the dashboard.
        assert snapshot.offline_reason is None

    @pytest.mark.parametrize('reason', ['unreachable', 'deadline_exceeded'])
    def test_offline_reason_accepts_vocabulary(self, reason):
        """Both members of the discriminator vocabulary are preserved as-given.

        ``deadline_exceeded`` (the probe's own budget fired — under a starved
        dashboard event loop that is the DASHBOARD's fault) and
        ``unreachable`` (connect refused / HTTP error / malformed payload —
        the orchestrator's fault) must stay separable on the wire, which is
        the whole point of task 3517.
        """
        snapshot = TaskRuntimeSnapshot(offline=True, offline_reason=reason)
        assert snapshot.offline is True
        assert snapshot.offline_reason == reason

    def test_bogus_offline_reason_raises_validation_error(self):
        """offline_reason is a machine-checked enum — an out-of-vocab value
        raises (INV-1: the model surfaces drift instead of silently widening),
        mirroring the lane_state guard above."""
        with pytest.raises(ValidationError):
            TaskRuntimeSnapshot(
                offline=True,
                offline_reason='wat',  # type: ignore[arg-type]
            )

    def test_offline_without_reason_is_constructible(self):
        """offline=True with no reason stays legal — it is out-of-contract for
        a dashboard-synthesized snapshot, and consumers must render it as an
        honest 'unknown' rather than fabricate a diagnosis."""
        snapshot = TaskRuntimeSnapshot(offline=True)
        assert snapshot.offline is True
        assert snapshot.offline_reason is None

    def test_model_dump_round_trip(self):
        entry = TaskRuntimeEntry(
            task_id=42,
            has_worktree=True,
            loops=3,
            attempts=1,
            started='2026-07-16T00:00:00+00:00',
            lane='_lane-3',
            phase='EXECUTE',
            lane_state='assigned',
            error=None,
        )
        snapshot = TaskRuntimeSnapshot(offline=False, tasks=[entry])
        dumped = snapshot.model_dump(mode='json')
        assert dumped == {
            'offline': False,
            'offline_reason': None,
            'tasks': [
                {
                    'task_id': 42,
                    'has_worktree': True,
                    'loops': 3,
                    'attempts': 1,
                    'started': '2026-07-16T00:00:00+00:00',
                    'lane': '_lane-3',
                    'phase': 'EXECUTE',
                    'lane_state': 'assigned',
                    'error': None,
                }
            ],
        }

    def test_model_dump_round_trip_carries_offline_reason(self):
        """A non-None reason survives the JSON dump — the discriminator has to
        cross the wire, not just live in memory."""
        snapshot = TaskRuntimeSnapshot(
            offline=True, offline_reason='deadline_exceeded'
        )
        dumped = snapshot.model_dump(mode='json')
        assert dumped == {
            'offline': True,
            'offline_reason': 'deadline_exceeded',
            'tasks': [],
        }
        assert TaskRuntimeSnapshot(**dumped) == snapshot
