"""Tests for LaneLifecycle (W11 alpha) — LaneState, LEGAL_TRANSITIONS, durable
record I/O, illegal-transition born-at-L2 escalation, quarantine helper.

Step test-contract: RED — module orchestrator.lane_lifecycle is absent; import fails.
Step test-legal-roundtrip: RED — LaneLifecycle/transition/read absent.
Step test-illegal-escalate: RED — illegal branch not implemented (files nothing).
Step test-quarantine: RED — quarantine method absent.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from escalation.queue import EscalationQueue

from orchestrator.lane_lifecycle import (
    ESCALATION_SENTINEL_ROLE,
    LEGAL_TRANSITIONS,
    AcquireRoute,
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

    def test_quarantined_has_exactly_one_sanctioned_recycle_edge(self):
        # The ONE sanctioned exit from QUARANTINED: recycle a quarantined slot
        # back into service (QUARANTINED -> RELEASED) once its bad worktree was
        # relocated to quarantine_base, so a genuine fresh dispatch can bring
        # the durable record back to ASSIGNED via the existing RELEASED edge.
        assert (LaneState.QUARANTINED, LaneState.RELEASED) in LEGAL_TRANSITIONS
        # Every OTHER QUARANTINED -> X stays illegal: a divergent quarantine
        # record must never be silently adopted straight back into service.
        for illegal_target in (
            LaneState.ASSIGNED,
            LaneState.IN_USE,
            LaneState.SEED,
            LaneState.REGISTERED,
        ):
            assert (
                LaneState.QUARANTINED, illegal_target,
            ) not in LEGAL_TRANSITIONS

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
# all_records — enumerate every lane's durable record (task 2634)
# ---------------------------------------------------------------------------


class TestAllRecords:
    def test_returns_records_keyed_by_lane_name(self, tmp_path: Path):
        lifecycle = _lifecycle(tmp_path)
        lane0 = tmp_path / '_lane-0'
        lane1 = tmp_path / '_lane-1'
        lifecycle.note_assigned(lane0, task_id='42', branch='task/42')
        lifecycle.note_assigned(lane1, task_id='43', branch='task/43')
        # A third lane whose record file is unparseable — must be skipped,
        # not raise.
        _seed_corrupt_record(tmp_path, tmp_path / '_lane-2')

        records = lifecycle.all_records()

        assert set(records) == {'_lane-0', '_lane-1'}
        assert records['_lane-0'].task_id == '42'
        assert records['_lane-0'].state == LaneState.ASSIGNED
        assert records['_lane-1'].task_id == '43'
        assert records['_lane-1'].state == LaneState.ASSIGNED

    def test_empty_state_dir_yields_empty_dict(self, tmp_path: Path):
        lifecycle = _lifecycle(tmp_path)
        lifecycle.state_dir.mkdir(parents=True)

        assert lifecycle.all_records() == {}

    def test_absent_state_dir_yields_empty_dict(self, tmp_path: Path):
        lifecycle = _lifecycle(tmp_path)

        assert not lifecycle.state_dir.exists()
        assert lifecycle.all_records() == {}


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


# ---------------------------------------------------------------------------
# .pool-root sentinel fold (W11 gamma) — POOL_ROOT_SENTINEL,
# pool_storage_present() / mark_pool_storage_present() move here from
# git_ops.py; GitOps becomes a thin delegator (see test_lane_lifecycle_gitops.py).
# ---------------------------------------------------------------------------


class TestPoolStorageSentinel:
    def test_present_false_before_marking(self, tmp_path: Path):
        lifecycle = _lifecycle(tmp_path)
        assert lifecycle.pool_storage_present() is False

    def test_mark_then_present_is_true(self, tmp_path: Path):
        lifecycle = _lifecycle(tmp_path)
        lifecycle.mark_pool_storage_present()
        sentinel = tmp_path / '.pool-root'
        assert sentinel.is_file()
        assert lifecycle.pool_storage_present() is True

    def test_mark_creates_missing_worktree_base(self, tmp_path: Path):
        worktree_base = tmp_path / 'not-yet-created'
        lifecycle = _lifecycle(worktree_base)
        assert not worktree_base.exists()

        lifecycle.mark_pool_storage_present()

        assert worktree_base.exists()
        assert lifecycle.pool_storage_present() is True


# ---------------------------------------------------------------------------
# AcquireRoute vocabulary + ACQUIRE_ROUTE_TRANSITIONS table (W11 eta): the
# named route table git_ops.py's _acquire_warm_lane_impl threads through its
# 7 branches — see test_lane_lifecycle_gitops.py for the GitOps-side writer
# and route-classification tests.
# ---------------------------------------------------------------------------


class TestAcquireRouteTable:
    # NOTE: table/vocabulary shape invariants (every route has an edge, every
    # edge is a legal (from, to) tuple, every edge targets ASSIGNED) are
    # enforced by `_validate_acquire_route_transitions()`, which runs at
    # `orchestrator.lane_lifecycle` import time and raises AssertionError on
    # collection if violated — see that module for the single source of
    # truth. Re-asserting them here would be pure duplication (this suite
    # could never even collect if they failed). What IS worth pinning here is
    # the literal route-name vocabulary itself, which the import-time
    # validator does not check (it only checks table/enum concordance,
    # whatever the names are).
    def test_acquire_route_has_exactly_seven_members(self):
        assert {member.name for member in AcquireRoute} == {
            'REUSE', 'REUSE_REPAIR', 'CREATE_ONCE_FRESH', 'CREATE_ONCE_REATTACH',
            'DISK_BACKSTOP_REUSE', 'RESET_IN_PLACE_REATTACH', 'RECYCLE',
        }


# ---------------------------------------------------------------------------
# note_assigned — idempotent bring-to-ASSIGNED (W11 delta): the reader-side /
# cache-mirror analog of GitOps._note_assigned_via_route, used by
# WarmLanePool.restore_assignment/note_assignment to keep the durable record
# coherent with the in-memory cache (PRD dec.3, I1).
# ---------------------------------------------------------------------------


class TestNoteAssigned:
    def test_released_lane_transitions_to_assigned(self, tmp_path: Path):
        lifecycle = _lifecycle(tmp_path)
        lane = _released_lane(tmp_path, lifecycle)  # branch='task/foo', RELEASED

        record = lifecycle.note_assigned(lane, task_id='42', branch='task/42')

        assert record.state == LaneState.ASSIGNED
        assert record.task_id == '42'
        assert record.branch == 'task/42'
        assert lifecycle.read(lane) == record

    def test_registered_lane_transitions_to_assigned(self, tmp_path: Path):
        lifecycle = _lifecycle(tmp_path)
        lane = tmp_path / '_lane-0'
        lifecycle.transition(lane, LaneState.SEED, seeded_from_sha='abc123')
        lifecycle.transition(lane, LaneState.REGISTERED, branch='task/42')

        record = lifecycle.note_assigned(lane, task_id='42', branch='task/42')

        assert record.state == LaneState.ASSIGNED
        assert record.task_id == '42'
        assert record.branch == 'task/42'
        assert lifecycle.read(lane) == record

    def test_absent_record_seeds_up_to_assigned(self, tmp_path: Path):
        lifecycle = _lifecycle(tmp_path)
        lane = tmp_path / '_lane-0'
        assert lifecycle.read(lane) is None

        record = lifecycle.note_assigned(lane, task_id='42', branch='task/42')

        assert record.state == LaneState.ASSIGNED
        assert record.task_id == '42'
        assert record.branch == 'task/42'
        assert lifecycle.read(lane) == record

    def test_already_assigned_same_task_is_idempotent_noop(self, tmp_path: Path):
        lifecycle = _lifecycle(tmp_path)
        lane = tmp_path / '_lane-0'
        lifecycle.transition(lane, LaneState.SEED, seeded_from_sha='abc123')
        lifecycle.transition(lane, LaneState.REGISTERED, branch='task/42')
        lifecycle.transition(lane, LaneState.ASSIGNED, task_id='42', title='demo')
        record_path = tmp_path / '.lane-state' / f'{lane.name}.json'
        before_bytes = record_path.read_bytes()

        record = lifecycle.note_assigned(lane, task_id='42', branch='task/42')

        assert record.state == LaneState.ASSIGNED
        assert record.task_id == '42'
        # No I/O at all on the idempotent path — byte-identical on disk.
        assert record_path.read_bytes() == before_bytes

    def test_already_assigned_different_task_raises_and_leaves_record_unchanged(
        self, tmp_path: Path,
    ):
        lifecycle = _lifecycle(tmp_path)
        lane = tmp_path / '_lane-0'
        lifecycle.transition(lane, LaneState.SEED, seeded_from_sha='abc123')
        lifecycle.transition(lane, LaneState.REGISTERED, branch='task/42')
        lifecycle.transition(lane, LaneState.ASSIGNED, task_id='42', title='demo')
        record_path = tmp_path / '.lane-state' / f'{lane.name}.json'
        before_bytes = record_path.read_bytes()

        with pytest.raises(IllegalLaneTransition):
            lifecycle.note_assigned(lane, task_id='99', branch='task/99')

        # Never silent-steal: the record is untouched on conflict.
        assert record_path.read_bytes() == before_bytes


# ---------------------------------------------------------------------------
# note_assigned recycles a durably-QUARANTINED slot (task 3029): a genuine
# fresh dispatch onto a slot whose bad worktree was relocated to
# quarantine_base must bring the durable record back to ASSIGNED via the
# sanctioned QUARANTINED -> RELEASED -> ASSIGNED recycle, LOUDLY (never
# silent-heal) — rather than raising IllegalLaneTransition and drifting the
# record (the live esc-__lane_record_drift__-1 incident).
# ---------------------------------------------------------------------------


class TestNoteAssignedRecyclesQuarantined:
    def _quarantined_lane_with_stale_pin(
        self, tmp_path: Path, lifecycle: LaneLifecycle,
    ) -> Path:
        """Drive a lane to a QUARANTINED record still carrying a stale
        task_id/title/branch from its prior (now-quarantined) assignment.
        """
        lane = tmp_path / '_lane-0'
        lifecycle.transition(lane, LaneState.SEED, seeded_from_sha='abc123')
        lifecycle.transition(lane, LaneState.REGISTERED, branch='task/old')
        lifecycle.transition(
            lane, LaneState.ASSIGNED, task_id='old', title='old-demo',
        )
        lifecycle.transition(lane, LaneState.QUARANTINED)
        return lane

    def test_quarantined_lane_recycled_to_assigned_without_raising(
        self, tmp_path: Path, caplog,
    ):
        lifecycle = _lifecycle(tmp_path)
        lane = self._quarantined_lane_with_stale_pin(tmp_path, lifecycle)
        seeded = lifecycle.read(lane)
        assert seeded is not None
        assert seeded.state == LaneState.QUARANTINED

        # Must NOT raise IllegalLaneTransition (the drift bug): it recycles.
        with caplog.at_level(logging.WARNING, logger='orchestrator.lane_lifecycle'):
            record = lifecycle.note_assigned(lane, task_id='42', branch='task/42')

        # Recycled to a clean ASSIGNED record for the NEW task.
        assert record.state == LaneState.ASSIGNED
        assert record.task_id == '42'
        assert record.branch == 'task/42'
        # The stale prior pin does NOT leak: task_id/title were cleared by the
        # RELEASED hop before the fresh ASSIGNED (task_id above proves the id;
        # title proves the RELEASED clear happened).
        assert record.title is None
        # Durable record == returned record (record <-> cache consistent, no
        # drift).
        assert lifecycle.read(lane) == record

        # LOUD, not silent (never-silent-heal): exactly one WARNING on the
        # lane_lifecycle logger names the lane and the recycle.
        recycle_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == 'orchestrator.lane_lifecycle'
            and lane.name in r.getMessage()
            and 'recycl' in r.getMessage().lower()
        ]
        assert len(recycle_warnings) == 1


class TestDelegatesToSharedAtomicWriter:
    """``LaneLifecycle._write`` delegates to ``shared.safe_io.atomic_write_text``.

    Task 3223 consolidated the repo's tmp+rename writers into ``shared.safe_io``.
    These pin that this site's semantics survived the move — in particular the
    LOCALE-dependent encoding, which the inlined ``os.fdopen(fd, 'w')`` produced
    and which this task deliberately preserves rather than silently upgrading
    to utf-8.
    """

    def test_delegates_with_preserved_semantics(self, tmp_path, monkeypatch):
        """One delegated call carrying mkdir=True, 0600, locale encoding, no fsync."""
        import locale

        import shared.safe_io as _safe_io

        calls = []
        monkeypatch.setattr(
            _safe_io,
            'atomic_write_text',
            lambda path, text, **kwargs: calls.append((path, text, kwargs)),
        )

        lifecycle = _lifecycle(tmp_path)
        lifecycle.transition(tmp_path / 'lane-1', LaneState.SEED)

        assert len(calls) == 1, f'expected exactly one delegated call, got {calls}'
        _path, text, kwargs = calls[0]
        assert json.loads(text)['state'] == LaneState.SEED.value
        assert kwargs.get('mkdir') is True, 'this site created its parent dir'
        assert kwargs.get('mode') == 0o600, 'mkstemp created 0600; must not widen'
        assert not kwargs.get('fsync'), 'this site never fsynced'
        assert kwargs.get('encoding') == locale.getpreferredencoding(False), (
            'locale-dependent encoding must be PRESERVED, not upgraded to utf-8'
        )

    def test_transition_creates_missing_state_dir_and_round_trips(self, tmp_path):
        """End-to-end: a first transition creates state_dir and the record reloads."""
        lifecycle = _lifecycle(tmp_path)
        assert not lifecycle.state_dir.exists()

        lifecycle.transition(tmp_path / 'lane-1', LaneState.SEED)

        record = lifecycle.read('lane-1')
        assert record is not None
        assert record.state is LaneState.SEED
        assert lifecycle._record_path('lane-1').stat().st_mode & 0o777 == 0o600
