"""Tests for the digest-time warm-lane map↔record divergence cross-check (task 2987).

This file imports ``orchestrator.lane_lifecycle``, a merge-lane ratchet cluster
module, so it drives the cross-check through public names only: no private
attribute reads and no patching.
"""

from __future__ import annotations

import dataclasses
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from escalation.queue import EscalationQueue

from orchestrator.event_store import EventStore, EventType
from orchestrator.lane_lifecycle import LaneRecord, LaneState
from orchestrator.lane_record_divergence import (
    LANE_RECORD_DIVERGENCE,
    PERSISTENT_DIVERGENCE_DIGESTS,
    PERSISTENT_DIVERGENCE_MIN_AGE,
    STREAKS_FILENAME,
    DivergenceStreak,
    LaneDivergence,
    advance_lane_record_divergence,
    cross_check_lane_records,
    file_persistent_divergence_alarm,
)

_T0 = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)


def _lane(n: int) -> Path:
    return Path(f'/wt/_lane-{n}')


def _check(assignments, records, previous_streaks=None, *, now=_T0):
    return cross_check_lane_records(assignments, records, previous_streaks, now=now)


class TestCrossCheckLaneRecords:
    def test_coherent_views_have_no_divergence(self):
        assignments = {'T': _lane(0), 'U': _lane(1)}
        records = {
            '_lane-0': LaneRecord(state=LaneState.ASSIGNED, task_id='T'),
            '_lane-1': LaneRecord(state=LaneState.IN_USE, task_id='U'),
        }

        assert _check(assignments, records) == ()

    def test_map_entry_with_released_record_reports_the_real_record_state(self):
        assignments = {'T': _lane(0)}
        records = {'_lane-0': LaneRecord(state=LaneState.RELEASED, task_id=None)}

        (divergence,) = _check(assignments, records)

        assert divergence.lane == '_lane-0'
        assert divergence.map_task_ids == ('T',)
        assert divergence.record_state is LaneState.RELEASED
        assert divergence.record_task_id is None

    def test_record_held_for_another_task_carries_both_task_ids(self):
        assignments = {'T': _lane(0)}
        records = {'_lane-0': LaneRecord(state=LaneState.ASSIGNED, task_id='U')}

        (divergence,) = _check(assignments, records)

        assert divergence.lane == '_lane-0'
        assert divergence.map_task_ids == ('T',)
        assert divergence.record_state is LaneState.ASSIGNED
        assert divergence.record_task_id == 'U'

    def test_lane_shared_by_two_map_tasks_diverges_even_when_the_record_names_one(self):
        assignments = {'U': _lane(0), 'T': _lane(0)}
        records = {'_lane-0': LaneRecord(state=LaneState.ASSIGNED, task_id='T')}

        (divergence,) = _check(assignments, records)

        assert divergence.lane == '_lane-0'
        assert divergence.map_task_ids == ('T', 'U')
        assert divergence.record_task_id == 'T'
        assert 'T, U' in divergence.render()

    def test_map_entry_without_any_record(self):
        (divergence,) = _check({'T': _lane(0)}, {})

        assert divergence.lane == '_lane-0'
        assert divergence.map_task_ids == ('T',)
        assert divergence.record_state is None
        assert divergence.record_task_id is None

    @pytest.mark.parametrize('state', [LaneState.ASSIGNED, LaneState.IN_USE])
    def test_holding_record_without_map_entry(self, state):
        records = {'_lane-0': LaneRecord(state=state, task_id='T')}

        (divergence,) = _check({}, records)

        assert divergence.lane == '_lane-0'
        assert divergence.map_task_ids == ()
        assert divergence.record_state is state
        assert divergence.record_task_id == 'T'

    def test_non_holding_records_without_map_entry_are_coherent(self):
        records = {
            '_lane-0': LaneRecord(state=LaneState.RELEASED),
            '_lane-1': LaneRecord(state=LaneState.QUARANTINED),
            '_lane-2': LaneRecord(state=LaneState.REGISTERED),
            '_lane-3': LaneRecord(state=LaneState.SEED),
        }

        assert _check({}, records) == ()

    def test_task_moved_between_lanes_diverges_on_both_lanes(self):
        assignments = {'T': _lane(1)}
        records = {'_lane-0': LaneRecord(state=LaneState.ASSIGNED, task_id='T')}

        divergences = _check(assignments, records)

        assert [d.lane for d in divergences] == ['_lane-0', '_lane-1']
        stale_record, unrecorded_map = divergences
        assert (stale_record.map_task_ids, stale_record.record_task_id) == ((), 'T')
        assert (unrecorded_map.map_task_ids, unrecorded_map.record_state) == (('T',), None)

    def test_result_is_a_tuple_sorted_by_lane_of_frozen_values(self):
        assignments = {'A': _lane(2), 'B': _lane(0), 'C': _lane(1)}

        divergences = _check(assignments, {})

        assert isinstance(divergences, tuple)
        assert [d.lane for d in divergences] == ['_lane-0', '_lane-1', '_lane-2']
        pytest.raises(
            dataclasses.FrozenInstanceError, setattr, divergences[0], 'lane', '_lane-9',
        )

    def test_streaks_continue_from_previous_and_start_now(self):
        assignments = {'T': _lane(0), 'U': _lane(1)}
        later = _T0 + timedelta(minutes=5)

        divergences = _check(
            assignments, {}, {'_lane-0': DivergenceStreak.starting(_T0)}, now=later,
        )

        by_lane = {d.lane: d.streak for d in divergences}
        assert by_lane == {
            '_lane-0': DivergenceStreak(consecutive_digests=2, first_seen=_T0, last_seen=later),
            '_lane-1': DivergenceStreak.starting(later),
        }

    def test_render_names_the_lane_and_both_views(self):
        divergence = LaneDivergence(
            lane='_lane-4', map_task_ids=('T7',), record_state=LaneState.ASSIGNED,
            record_task_id='U8',
            streak=DivergenceStreak(consecutive_digests=3, first_seen=_T0, last_seen=_T0),
        )

        text = divergence.render()

        for fragment in ('_lane-4', 'T7', LaneState.ASSIGNED.value, 'U8', '3', '2026-11-09T12:00'):
            assert fragment in text

    def test_render_marks_missing_views(self):
        unassigned = LaneDivergence(
            lane='_lane-4', map_task_ids=(), record_state=LaneState.IN_USE,
            record_task_id='U8', streak=DivergenceStreak.starting(_T0),
        ).render()
        unrecorded = LaneDivergence(
            lane='_lane-5', map_task_ids=('T7',), record_state=None, record_task_id=None,
            streak=DivergenceStreak.starting(_T0),
        ).render()

        assert 'unassigned' in unassigned
        assert 'no readable record' in unrecorded
        assert '_lane-5' in unrecorded and 'T7' in unrecorded

    def test_contract_fixed_anchor(self):
        assert LANE_RECORD_DIVERGENCE == 'lane_record_divergence'


class TestDivergenceStreakPersistence:
    def _streak(self, consecutive: int, age: timedelta) -> DivergenceStreak:
        return DivergenceStreak(
            consecutive_digests=consecutive, first_seen=_T0, last_seen=_T0 + age,
        )

    def test_a_first_sighting_is_not_persistent(self):
        assert DivergenceStreak.starting(_T0).is_persistent is False

    def test_enough_digests_and_old_enough_is_persistent(self):
        streak = self._streak(PERSISTENT_DIVERGENCE_DIGESTS, PERSISTENT_DIVERGENCE_MIN_AGE)

        assert streak.is_persistent is True

    def test_enough_digests_inside_the_minimum_age_is_not_persistent(self):
        young = PERSISTENT_DIVERGENCE_MIN_AGE - timedelta(seconds=1)

        assert self._streak(PERSISTENT_DIVERGENCE_DIGESTS, young).is_persistent is False

    def test_old_but_too_few_digests_is_not_persistent(self):
        old = PERSISTENT_DIVERGENCE_MIN_AGE * 10

        assert self._streak(PERSISTENT_DIVERGENCE_DIGESTS - 1, old).is_persistent is False


_UNRECORDED_MAP_ENTRY = ({'T': _lane(0)}, {})


class TestAdvanceLaneRecordDivergence:
    @pytest.fixture
    def streaks_path(self, tmp_path):
        return tmp_path / 'digests' / STREAKS_FILENAME

    def _advance(self, streaks_path, assignments=None, records=None, *, now=_T0):
        if assignments is None:
            assignments, records = _UNRECORDED_MAP_ENTRY
        return advance_lane_record_divergence(
            assignments, records or {}, streaks_path=streaks_path, now=now,
        )

    def test_first_divergence_starts_a_streak_on_disk(self, streaks_path):
        (divergence,) = self._advance(streaks_path)

        assert divergence.streak == DivergenceStreak.starting(_T0)
        assert streaks_path.is_file()

    def test_streak_survives_through_the_file(self, streaks_path):
        later = _T0 + PERSISTENT_DIVERGENCE_MIN_AGE
        self._advance(streaks_path)

        (divergence,) = self._advance(streaks_path, now=later)

        assert divergence.streak == DivergenceStreak(
            consecutive_digests=2, first_seen=_T0, last_seen=later,
        )
        assert divergence.streak.is_persistent is True

    def test_back_to_back_digests_inside_the_minimum_age_are_not_persistent(
        self, streaks_path,
    ):
        self._advance(streaks_path)

        (divergence,) = self._advance(streaks_path, now=_T0 + timedelta(minutes=2))

        assert divergence.streak.consecutive_digests == PERSISTENT_DIVERGENCE_DIGESTS
        assert divergence.streak.is_persistent is False

    def test_coherence_drops_the_streak(self, streaks_path):
        self._advance(streaks_path)
        coherent_records = {'_lane-0': LaneRecord(state=LaneState.ASSIGNED, task_id='T')}

        assert self._advance(streaks_path, {'T': _lane(0)}, coherent_records) == ()
        later = _T0 + timedelta(hours=2)
        (divergence,) = self._advance(streaks_path, now=later)

        assert divergence.streak == DivergenceStreak.starting(later)

    def test_streak_is_keyed_by_lane_not_by_divergence_shape(self, streaks_path):
        assignments = {'T': _lane(0)}
        self._advance(
            streaks_path, assignments, {'_lane-0': LaneRecord(state=LaneState.RELEASED)},
        )

        (divergence,) = self._advance(
            streaks_path,
            assignments,
            {'_lane-0': LaneRecord(state=LaneState.ASSIGNED, task_id='U')},
        )

        assert divergence.streak.consecutive_digests == 2

    @pytest.mark.parametrize('content', [
        'not json',
        '["_lane-0"]',
        '{"_lane-0": "x"}',
        '{"_lane-0": 2}',
        '{"_lane-0": {"consecutive_digests": 2, "first_seen": "2026-11-09T12:00:00",'
        ' "last_seen": "2026-11-09T12:00:00"}}',
    ])
    def test_corrupt_streaks_file_counts_as_no_prior_streak(
        self, streaks_path, content, caplog,
    ):
        streaks_path.parent.mkdir(parents=True)
        streaks_path.write_text(content, encoding='utf-8')

        with caplog.at_level('WARNING'):
            (first,) = self._advance(streaks_path)
        (second,) = self._advance(streaks_path)

        assert first.streak.consecutive_digests == 1
        assert _warned_about(caplog, streaks_path)
        assert second.streak.consecutive_digests == 2

    def test_unreadable_streaks_path_counts_afresh(self, streaks_path, caplog):
        streaks_path.mkdir(parents=True)

        with caplog.at_level('WARNING'):
            (divergence,) = self._advance(streaks_path)

        assert divergence.streak.consecutive_digests == 1
        assert _warned_about(caplog, streaks_path)

    @pytest.mark.skipif(os.geteuid() == 0, reason='root writes into a 0o500 directory')
    def test_unsaveable_streaks_still_return_the_divergences(self, streaks_path, caplog):
        streaks_path.parent.mkdir()
        streaks_path.parent.chmod(0o500)
        try:
            with caplog.at_level('WARNING'):
                (divergence,) = self._advance(streaks_path)
        finally:
            streaks_path.parent.chmod(0o700)

        assert divergence.streak.consecutive_digests == 1
        assert not streaks_path.exists()
        assert _warned_about(caplog, streaks_path)


def _warned_about(caplog, path: Path) -> bool:
    return any(
        r.levelname == 'WARNING' and str(path) in r.getMessage() for r in caplog.records
    )


def _divergence(lane: str, streak: DivergenceStreak) -> LaneDivergence:
    return LaneDivergence(
        lane=lane, map_task_ids=(), record_state=LaneState.ASSIGNED,
        record_task_id='101', streak=streak,
    )


_PERSISTENT = _divergence('_lane-0', DivergenceStreak(
    consecutive_digests=PERSISTENT_DIVERGENCE_DIGESTS,
    first_seen=_T0,
    last_seen=_T0 + PERSISTENT_DIVERGENCE_MIN_AGE,
))
_FRESH = _divergence('_lane-1', DivergenceStreak.starting(_T0))


class TestFilePersistentDivergenceAlarm:
    @pytest.fixture
    def queue(self, tmp_path):
        return EscalationQueue(tmp_path / 'esc')

    @pytest.fixture
    def event_store(self):
        return MagicMock(spec=EventStore)

    def _file(self, queue, divergences, event_store=None):
        return file_persistent_divergence_alarm(
            queue, divergences, event_store=event_store, filing_claimant_run_id=None,
        )

    def test_no_alarm_before_any_lane_persists(self, queue, event_store):
        assert self._file(queue, [_FRESH], event_store) is None
        assert queue.get_pending() == []
        event_store.emit.assert_not_called()

    def test_persistent_divergence_files_one_born_at_l2(self, queue, event_store):
        escalation = self._file(queue, [_PERSISTENT, _FRESH], event_store)

        assert escalation is not None
        (pending,) = queue.get_pending()
        assert pending.id == escalation.id
        assert pending.level == 2
        assert pending.root_cause == LANE_RECORD_DIVERGENCE
        assert pending.category == 'infra_issue'
        assert pending.severity == 'urgent'
        assert pending.agent_role.startswith('orchestrator-')
        assert _PERSISTENT.render() in pending.detail
        assert '_lane-0' in pending.summary
        assert '_lane-1' not in pending.summary
        event_store.emit.assert_called_once()
        (event_type,) = event_store.emit.call_args.args
        assert event_type is EventType.escalation_created
        assert event_store.emit.call_args.kwargs['data']['escalation_id'] == escalation.id

    def test_pending_alarm_dedups(self, queue):
        self._file(queue, [_PERSISTENT])

        assert self._file(queue, [_PERSISTENT]) is None
        assert len(queue.get_pending()) == 1

    def test_realarms_after_a_human_resolves_while_divergence_persists(self, queue):
        first = self._file(queue, [_PERSISTENT])
        assert first is not None
        queue.resolve(first.id, 'investigated')

        second = self._file(queue, [_PERSISTENT])

        assert second is not None and second.id != first.id
        assert [e.id for e in queue.get_pending()] == [second.id]
