"""Tests for LaneLifecycle (W11 alpha) — LaneState, LEGAL_TRANSITIONS, durable
record I/O, illegal-transition born-at-L2 escalation, quarantine helper.

Step test-contract: RED — module orchestrator.lane_lifecycle is absent; import fails.
Step test-legal-roundtrip: RED — LaneLifecycle/transition/read absent.
Step test-illegal-escalate: RED — illegal branch not implemented (files nothing).
Step test-quarantine: RED — quarantine method absent.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from escalation.queue import EscalationQueue

from orchestrator.lane_lifecycle import (
    ESCALATION_SENTINEL_ROLE,
    LEGAL_TRANSITIONS,
    IllegalLaneTransition,
    LaneLifecycle,
    LaneState,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _lifecycle(tmp_path: Path, **kwargs) -> LaneLifecycle:
    return LaneLifecycle(worktree_base=tmp_path, **kwargs)

# ---------------------------------------------------------------------------
# Static contract
# ---------------------------------------------------------------------------


class TestStaticContract:
    def test_lane_state_has_exactly_six_members(self):
        assert {member.name for member in LaneState} == {
            'SEED', 'REGISTERED', 'ASSIGNED', 'IN_USE', 'RELEASED', 'QUARANTINED',
        }

    def test_lane_state_values_are_lowercase_strings(self):
        for member in LaneState:
            assert member.value == member.name.lower()

    def test_legal_transitions_contains_key_contractual_edges(self):
        # A focused spot-check, not the full table — duplicating the exact
        # table/comprehension LEGAL_TRANSITIONS is built from would just be a
        # change-detector. Exhaustive behavioral coverage of the sequence
        # (including the RELEASED -> ASSIGNED reuse edge) lives in
        # TestLegalRoundtrip, TestIllegalEscalate, and TestQuarantine.
        key_edges = {
            (None, LaneState.SEED),
            (LaneState.ASSIGNED, LaneState.IN_USE),
            (LaneState.IN_USE, LaneState.RELEASED),
            (LaneState.RELEASED, LaneState.ASSIGNED),
        }
        assert key_edges <= LEGAL_TRANSITIONS
        # "any -> QUARANTINED" holds from a real state and the None origin.
        assert (LaneState.ASSIGNED, LaneState.QUARANTINED) in LEGAL_TRANSITIONS
        assert (None, LaneState.QUARANTINED) in LEGAL_TRANSITIONS

    def test_legal_transitions_excludes_illegal_edge(self):
        assert (LaneState.RELEASED, LaneState.IN_USE) not in LEGAL_TRANSITIONS

    def test_illegal_lane_transition_is_an_exception(self):
        assert issubclass(IllegalLaneTransition, Exception)

    def test_escalation_sentinel_role(self):
        assert ESCALATION_SENTINEL_ROLE == 'harness-lane-lifecycle'
        assert ESCALATION_SENTINEL_ROLE.startswith('harness-')


# ---------------------------------------------------------------------------
# Legal-transition durable-record round-trip (user-observable signal, half 1)
# ---------------------------------------------------------------------------


class TestLegalRoundtrip:
    def test_legal_sequence_persists_and_reads_back(self, tmp_path: Path):
        lifecycle = _lifecycle(tmp_path)
        lane = tmp_path / '_lane-0'

        record = lifecycle.transition(lane, LaneState.SEED, seeded_from_sha='abc123')
        assert record.state == LaneState.SEED
        assert record.seeded_from_sha == 'abc123'
        assert lifecycle.read(lane) == record

        record = lifecycle.transition(lane, LaneState.REGISTERED, branch='task/foo')
        assert record.state == LaneState.REGISTERED
        assert record.branch == 'task/foo'
        assert record.seeded_from_sha == 'abc123'  # carried forward
        assert lifecycle.read(lane) == record

        record = lifecycle.transition(
            lane, LaneState.ASSIGNED, task_id='2254', title='demo',
        )
        assert record.state == LaneState.ASSIGNED
        assert record.task_id == '2254'
        assert record.title == 'demo'
        assert lifecycle.read(lane) == record

        record = lifecycle.transition(lane, LaneState.IN_USE)
        assert record.state == LaneState.IN_USE
        assert record.task_id == '2254'  # carried forward, unchanged
        assert record.title == 'demo'
        assert lifecycle.read(lane) == record

        record = lifecycle.transition(lane, LaneState.RELEASED)
        assert record.state == LaneState.RELEASED
        assert record.task_id is None  # cleared on the RELEASED edge
        assert record.title is None
        assert record.branch == 'task/foo'  # branch is NOT cleared
        assert lifecycle.read(lane) == record

    def test_record_file_path_and_json_shape(self, tmp_path: Path):
        lifecycle = _lifecycle(tmp_path)
        lane = tmp_path / '_lane-0'
        lifecycle.transition(lane, LaneState.SEED, seeded_from_sha='abc123')

        record_path = tmp_path / '.lane-state' / f'{lane.name}.json'
        assert record_path.is_file()
        data = json.loads(record_path.read_text())
        assert data['state'] == 'seed'
        assert data['seeded_from_sha'] == 'abc123'
        # updated_at must be a parseable ISO timestamp.
        datetime.fromisoformat(data['updated_at'])

    def test_no_leftover_tmp_files(self, tmp_path: Path):
        lifecycle = _lifecycle(tmp_path)
        lane = tmp_path / '_lane-0'
        lifecycle.transition(lane, LaneState.SEED, seeded_from_sha='abc123')
        lifecycle.transition(lane, LaneState.REGISTERED, branch='task/foo')

        leftovers = list((tmp_path / '.lane-state').glob('*.tmp'))
        assert leftovers == []

    def test_read_returns_none_when_no_record(self, tmp_path: Path):
        lifecycle = _lifecycle(tmp_path)
        lane = tmp_path / '_lane-0'
        assert lifecycle.read(lane) is None

    def test_released_lane_reassigned_carries_branch_and_seed_forward(
        self, tmp_path: Path,
    ):
        """RELEASED -> ASSIGNED warm-lane reuse: task_id/title come from the
        new assignment, while branch/seeded_from_sha (never cleared by the
        RELEASED edge) carry forward from before the release.
        """
        lifecycle = _lifecycle(tmp_path)
        lane = _released_lane(tmp_path, lifecycle)

        record = lifecycle.transition(
            lane, LaneState.ASSIGNED, task_id='9999', title='reused-demo',
        )
        assert record.state == LaneState.ASSIGNED
        assert record.task_id == '9999'
        assert record.title == 'reused-demo'
        assert record.branch == 'task/foo'
        assert record.seeded_from_sha == 'abc123'
        assert lifecycle.read(lane) == record


# ---------------------------------------------------------------------------
# Illegal transition -> born-at-L2 escalation (user-observable signal, half 2)
# ---------------------------------------------------------------------------


def _released_lane(tmp_path: Path, lifecycle: LaneLifecycle) -> Path:
    """Drive a lane through the legal sequence to RELEASED and return its path."""
    lane = tmp_path / '_lane-0'
    lifecycle.transition(lane, LaneState.SEED, seeded_from_sha='abc123')
    lifecycle.transition(lane, LaneState.REGISTERED, branch='task/foo')
    lifecycle.transition(lane, LaneState.ASSIGNED, task_id='2254', title='demo')
    lifecycle.transition(lane, LaneState.IN_USE)
    lifecycle.transition(lane, LaneState.RELEASED)
    return lane


class TestIllegalEscalate:
    def test_illegal_transition_files_born_at_l2_escalation(self, tmp_path: Path):
        queue = EscalationQueue(tmp_path / 'escalations')
        lifecycle = _lifecycle(tmp_path, escalation_queue=queue)
        lane = _released_lane(tmp_path, lifecycle)

        record_path = tmp_path / '.lane-state' / f'{lane.name}.json'
        before_bytes = record_path.read_bytes()

        with pytest.raises(IllegalLaneTransition):
            lifecycle.transition(lane, LaneState.IN_USE)

        sentinel_task_id = f'lane-lifecycle-{lane.name}'
        escalations = queue.get_by_task(sentinel_task_id)
        assert len(escalations) == 1
        esc = escalations[0]
        assert esc.severity in {'critical', 'urgent'}
        assert esc.agent_role == ESCALATION_SENTINEL_ROLE
        assert esc.agent_role.startswith('harness-')
        assert esc.level == 2
        assert esc.category == 'risk_identified'

        assert record_path.read_bytes() == before_bytes

    def test_illegal_transition_without_queue_still_raises_and_files_nothing(
        self, tmp_path: Path,
    ):
        lifecycle = _lifecycle(tmp_path)  # escalation_queue=None
        lane = _released_lane(tmp_path, lifecycle)

        with pytest.raises(IllegalLaneTransition):
            lifecycle.transition(lane, LaneState.IN_USE)


# ---------------------------------------------------------------------------
# Unknown **fields validation
# ---------------------------------------------------------------------------


class TestUnknownFieldValidation:
    def test_transition_rejects_unknown_field_name(self, tmp_path: Path):
        lifecycle = _lifecycle(tmp_path)
        lane = tmp_path / '_lane-0'

        with pytest.raises(TypeError):
            lifecycle.transition(lane, LaneState.SEED, seeded_sha='abc123')  # typo

        # The rejected call must not have written a record.
        assert lifecycle.read(lane) is None


# ---------------------------------------------------------------------------
# Corrupt on-disk record handling
# ---------------------------------------------------------------------------


def _seed_corrupt_record(tmp_path: Path, lane: Path) -> Path:
    """Write unparseable bytes directly to *lane*'s record path."""
    record_path = tmp_path / '.lane-state' / f'{lane.name}.json'
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text('{not valid json')
    return record_path


class TestCorruptRecord:
    def test_corrupt_record_blocks_transition_and_escalates(self, tmp_path: Path):
        queue = EscalationQueue(tmp_path / 'escalations')
        lifecycle = _lifecycle(tmp_path, escalation_queue=queue)
        lane = tmp_path / '_lane-0'
        record_path = _seed_corrupt_record(tmp_path, lane)
        before_bytes = record_path.read_bytes()

        with pytest.raises(IllegalLaneTransition):
            lifecycle.transition(lane, LaneState.SEED, seeded_from_sha='xyz')

        # Never silent-heal: a corrupt record is not treated as a fresh,
        # None-origin lane, and the corrupt bytes are left untouched.
        assert record_path.read_bytes() == before_bytes

        sentinel_task_id = f'lane-lifecycle-{lane.name}'
        escalations = queue.get_by_task(sentinel_task_id)
        assert len(escalations) == 1
        esc = escalations[0]
        assert esc.agent_role == ESCALATION_SENTINEL_ROLE
        assert esc.level == 2
        assert esc.category == 'risk_identified'

    def test_corrupt_record_still_allows_quarantine(self, tmp_path: Path):
        # any -> QUARANTINED (including a corrupt "any") stays reachable so a
        # corrupt lane can still be recovered via quarantine.
        lifecycle = _lifecycle(tmp_path)
        lane = tmp_path / '_lane-0'
        _seed_corrupt_record(tmp_path, lane)

        record = lifecycle.transition(lane, LaneState.QUARANTINED)

        assert record.state == LaneState.QUARANTINED
        assert lifecycle.read(lane) == record


# ---------------------------------------------------------------------------
# Quarantine helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestQuarantine:
    async def test_quarantine_delegates_and_persists_quarantined_state(
        self, tmp_path: Path,
    ):
        dest = tmp_path / 'worktrees-orphaned' / '_lane-0-20260706T000000Z'
        fake_quarantine_worktree = AsyncMock(return_value=dest)
        lifecycle = _lifecycle(tmp_path, quarantine_worktree=fake_quarantine_worktree)
        lane = tmp_path / '_lane-0'
        # Non-terminal state, proving any -> QUARANTINED (not just RELEASED ->).
        lifecycle.transition(lane, LaneState.SEED, seeded_from_sha='abc123')
        lifecycle.transition(lane, LaneState.REGISTERED, branch='task/foo')
        lifecycle.transition(lane, LaneState.ASSIGNED, task_id='2254', title='demo')

        result = await lifecycle.quarantine(lane, branch='task/foo', reason='divergence')

        fake_quarantine_worktree.assert_awaited_once_with(lane, 'task/foo', 'divergence')
        assert result == dest
        record = lifecycle.read(lane)
        assert record is not None
        assert record.state == LaneState.QUARANTINED

    async def test_quarantine_without_callable_wired_raises(self, tmp_path: Path):
        lifecycle = _lifecycle(tmp_path)  # quarantine_worktree=None
        lane = tmp_path / '_lane-0'
        lifecycle.transition(lane, LaneState.SEED, seeded_from_sha='abc123')

        with pytest.raises(RuntimeError):
            await lifecycle.quarantine(lane, branch='task/foo', reason='divergence')
