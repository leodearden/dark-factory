"""The quarantine provenance record (task 4845).

A quarantine parks a lane under ``quarantine_base``. These tests pin what it
leaves behind, using the filesystem only, with no git: the event payload, a
point-in-time snapshot of the task's artifacts inside the parking, and a
``_quarantined_from`` back-pointer in the task-root plan.json that survives a
re-plan.
"""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from orchestrator.artifacts import PlanConfirmation, PlanState, TaskArtifacts
from orchestrator.quarantine_provenance import (
    NOT_RELOCATED,
    QUARANTINED_FROM_KEY,
    QuarantineOutcome,
    quarantined_from_of,
    record_quarantine_provenance,
    snapshot_into_parking,
    stamp_quarantined_from,
)

_STAMP_KEYS = {'branch', 'tip_sha', 'unique_commit_count', 'quarantined_at', 'reason', 'parking'}
_SNAPSHOT_NAMES: tuple[str, ...] = ('plan.json', 'metadata.json', 'review_state.json', 'iterations.jsonl')
_PLAN = {
    'task_id': '42',
    'title': 'T',
    'steps': [
        {'id': 'step-1', 'type': 'test', 'description': 'red', 'status': 'done', 'commit': 'abc'},
        {'id': 'step-2', 'type': 'impl', 'description': 'green', 'status': 'pending'},
    ],
}


@pytest.fixture()
def lane(tmp_path: Path) -> Path:
    worktree = tmp_path / 'worktrees' / '_lane-0'
    worktree.mkdir(parents=True)
    return worktree


@pytest.fixture()
def artifacts(lane: Path) -> TaskArtifacts:
    ta = TaskArtifacts.for_task(lane, '42')
    ta.init('42', 'T', 'd')
    ta.write_plan(json.loads(json.dumps(_PLAN)))
    ta.set_review_counters(amendment_rounds_total=1, review_cycles_total=2)
    ta.append_iteration_log({'iteration': 1, 'summary': 'did a thing'})
    ta.ensure_lane_symlinks()
    return ta


def _stamp(reason: str = 'unit-test', **overrides: object) -> dict[str, object]:
    stamp: dict[str, object] = {
        'branch': 'task/42-20261112T000000Z',
        'tip_sha': 'abc123',
        'unique_commit_count': 2,
        'quarantined_at': datetime.now(UTC).isoformat(),
        'reason': reason,
        'parking': '/quarantine/42-20261112T000000Z',
    }
    stamp.update(overrides)
    return stamp


def _confirm(ta: TaskArtifacts) -> None:
    plan = ta.read_plan_model()
    assert plan is not None
    ta.write_plan_model(plan.confirmed(at=datetime.now(UTC).isoformat(), by=PlanConfirmation.ARCHITECT))


class TestEventPayload:
    def test_a_relocated_outcome_carries_every_field(self, tmp_path: Path):
        parking = tmp_path / 'q'
        outcome = QuarantineOutcome(dest=parking, branch='task/42-X', tip_sha='abc', unique_commit_count=3)

        assert outcome.relocated
        assert outcome.event_data('orphan-reaper') == {
            'reason': 'orphan-reaper',
            'dest': str(parking),
            'branch': 'task/42-X',
            'tip_sha': 'abc',
            'unique_commit_count': 3,
        }

    def test_not_relocated_carries_the_reason_and_nulls(self):
        assert not NOT_RELOCATED.relocated
        assert NOT_RELOCATED.event_data('recovery-identity-mismatch') == {
            'reason': 'recovery-identity-mismatch',
            'dest': None,
            'branch': None,
            'tip_sha': None,
            'unique_commit_count': None,
        }


class TestSnapshotIntoParking:
    def test_each_artifact_becomes_a_regular_file_holding_the_source_bytes(self, artifacts: TaskArtifacts, lane):
        sources = {name: artifacts.artifact_path(name).read_bytes() for name in _SNAPSHOT_NAMES}
        assert (lane / '.task' / 'plan.json').is_symlink()

        copied = snapshot_into_parking(artifacts, lane)

        assert set(copied) == set(_SNAPSHOT_NAMES)
        for name in _SNAPSHOT_NAMES:
            parked = lane / '.task' / name
            assert not parked.is_symlink(), name
            assert parked.is_file(), name
            assert parked.read_bytes() == sources[name], name

    def test_the_live_plan_is_left_byte_identical(self, artifacts: TaskArtifacts, lane):
        live = artifacts.artifact_path('plan.json')
        before = live.read_bytes()

        snapshot_into_parking(artifacts, lane)

        assert not live.is_symlink()
        assert live.read_bytes() == before

    def test_the_parked_copy_is_point_in_time(self, artifacts: TaskArtifacts, lane):
        snapshot_into_parking(artifacts, lane)
        parked = (lane / '.task' / 'plan.json').read_bytes()

        artifacts.write_plan({**json.loads(json.dumps(_PLAN)), 'title': 'rewritten'})

        assert (lane / '.task' / 'plan.json').read_bytes() == parked
        assert artifacts.read_plan()['title'] == 'rewritten'

    def test_a_missing_source_is_skipped(self, artifacts: TaskArtifacts, lane):
        artifacts.artifact_path('review_state.json').unlink()

        copied = snapshot_into_parking(artifacts, lane)

        assert 'review_state.json' not in copied
        assert not (lane / '.task' / 'review_state.json').exists()
        assert set(copied) == set(_SNAPSHOT_NAMES) - {'review_state.json'}

    def test_a_symlinked_task_dir_is_replaced_and_its_target_left_alone(self, artifacts: TaskArtifacts, tmp_path):
        parking = tmp_path / 'parking'
        parking.mkdir()
        elsewhere = tmp_path / 'elsewhere'
        elsewhere.mkdir()
        (elsewhere / 'keep.txt').write_text('untouched')
        os.symlink(elsewhere, parking / '.task')

        snapshot_into_parking(artifacts, parking)

        assert not (parking / '.task').is_symlink()
        assert (parking / '.task').is_dir()
        assert (parking / '.task' / 'plan.json').is_file()
        assert sorted(p.name for p in elsewhere.iterdir()) == ['keep.txt']
        assert (elsewhere / 'keep.txt').read_text() == 'untouched'


class TestStampQuarantinedFrom:
    def test_each_call_appends_one_entry_in_order(self, artifacts: TaskArtifacts):
        first, second = _stamp('first'), _stamp('second', unique_commit_count=None, branch=None, tip_sha=None)

        assert stamp_quarantined_from(artifacts, first)
        assert stamp_quarantined_from(artifacts, second)

        entries = artifacts.read_plan()[QUARANTINED_FROM_KEY]
        assert entries == [first, second]
        assert all(set(entry) == _STAMP_KEYS for entry in entries)

    def test_the_steps_are_untouched(self, artifacts: TaskArtifacts):
        steps = artifacts.read_plan()['steps']

        stamp_quarantined_from(artifacts, _stamp())

        assert artifacts.read_plan()['steps'] == steps

    def test_a_confirmation_survives_the_stamp(self, artifacts: TaskArtifacts):
        _confirm(artifacts)
        before = artifacts.read_plan_model()
        assert before is not None
        assert before.state is PlanState.CONFIRMED

        stamp_quarantined_from(artifacts, _stamp())

        after = artifacts.read_plan_model()
        assert after is not None
        assert after.state is PlanState.CONFIRMED
        assert after.lifecycle == before.lifecycle

    def test_no_plan_means_no_stamp_and_no_fabricated_plan(self, lane: Path):
        ta = TaskArtifacts.for_task(lane, '7')
        ta.init('7', 'T', 'd')
        plan_path = ta.artifact_path('plan.json')

        assert stamp_quarantined_from(ta, _stamp()) is False
        assert not plan_path.exists()

    def test_a_non_list_value_is_restarted_as_a_list(self, artifacts: TaskArtifacts):
        artifacts.write_plan({**json.loads(json.dumps(_PLAN)), QUARANTINED_FROM_KEY: 'garbage'})
        stamp = _stamp()

        stamp_quarantined_from(artifacts, stamp)

        assert artifacts.read_plan()[QUARANTINED_FROM_KEY] == [stamp]


class TestRecordQuarantineProvenance:
    def test_snapshots_then_stamps(self, artifacts: TaskArtifacts, lane, tmp_path):
        parking = tmp_path / 'quarantine' / '42-20261112T000000Z'
        parking.parent.mkdir()
        os.rename(lane, parking)
        outcome = QuarantineOutcome(dest=parking, branch='task/42-X', tip_sha='abc', unique_commit_count=3)

        record_quarantine_provenance(lane, '42', outcome, 'recovery-record-divergence')

        parked_plan = parking / '.task' / 'plan.json'
        assert not parked_plan.is_symlink()
        assert QUARANTINED_FROM_KEY not in json.loads(parked_plan.read_text())
        (entry,) = artifacts.read_plan()[QUARANTINED_FROM_KEY]
        assert set(entry) == _STAMP_KEYS
        assert entry['branch'] == 'task/42-X'
        assert entry['tip_sha'] == 'abc'
        assert entry['unique_commit_count'] == 3
        assert entry['reason'] == 'recovery-record-divergence'
        assert entry['parking'] == str(parking)
        assert datetime.fromisoformat(entry['quarantined_at']).tzinfo is not None

    def test_not_relocated_writes_nothing(self, artifacts: TaskArtifacts, lane):
        before = artifacts.artifact_path('plan.json').read_bytes()

        record_quarantine_provenance(lane, '42', NOT_RELOCATED, 'orphan-reaper')

        assert artifacts.artifact_path('plan.json').read_bytes() == before
        assert (lane / '.task' / 'plan.json').is_symlink()

    def test_an_invalid_task_id_returns_normally(self, lane, tmp_path):
        outcome = QuarantineOutcome(dest=tmp_path / 'q')

        record_quarantine_provenance(lane, 'bad id!', outcome, 'orphan-reaper')

    def test_an_unwritable_parking_returns_normally(self, artifacts: TaskArtifacts, lane, tmp_path):
        blocker = tmp_path / 'a-file'
        blocker.write_text('not a directory')
        outcome = QuarantineOutcome(dest=blocker / 'parking')

        record_quarantine_provenance(lane, '42', outcome, 'orphan-reaper')


class TestQuarantinedFromOf:
    def test_well_formed_entries_survive_in_order(self):
        first, second = _stamp('first'), _stamp('second', branch=None, tip_sha=None, unique_commit_count=None)

        assert quarantined_from_of({QUARANTINED_FROM_KEY: [first, second]}) == {
            QUARANTINED_FROM_KEY: [first, second],
        }

    def test_unknown_keys_are_dropped(self):
        stamp = _stamp()

        projected = quarantined_from_of({QUARANTINED_FROM_KEY: [{**stamp, 'unreviewed': 'prose'}]})

        assert projected == {QUARANTINED_FROM_KEY: [stamp]}

    @pytest.mark.parametrize('bad', [
        pytest.param('not-a-dict', id='str'),
        pytest.param(['a', 'list'], id='list'),
        pytest.param(_stamp(unique_commit_count='lots'), id='count-str'),
        pytest.param(_stamp(unique_commit_count=True), id='count-bool'),
        pytest.param(_stamp(branch=7), id='branch-int'),
        pytest.param(_stamp(parking=None), id='parking-none'),
        pytest.param({k: v for k, v in _stamp().items() if k != 'reason'}, id='missing-key'),
    ])
    def test_malformed_entries_are_dropped(self, bad: object):
        good = _stamp()

        assert quarantined_from_of({QUARANTINED_FROM_KEY: [bad, good]}) == {QUARANTINED_FROM_KEY: [good]}

    @pytest.mark.parametrize('plan', [
        pytest.param({}, id='absent'),
        pytest.param({QUARANTINED_FROM_KEY: 'garbage'}, id='not-a-list'),
        pytest.param({QUARANTINED_FROM_KEY: []}, id='empty'),
        pytest.param({QUARANTINED_FROM_KEY: ['x', {'branch': 'y'}]}, id='nothing-survives'),
    ])
    def test_nothing_to_carry_is_an_empty_dict(self, plan: dict):
        assert quarantined_from_of(plan) == {}
