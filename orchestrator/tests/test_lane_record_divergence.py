"""Tests for the digest-time warm-lane map↔record divergence cross-check (task 2987).

This file imports ``orchestrator.lane_lifecycle``, a merge-lane ratchet cluster
module, so it drives the cross-check through public names only: no private
attribute reads and no patching.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from orchestrator.lane_lifecycle import LaneRecord, LaneState
from orchestrator.lane_record_divergence import (
    LANE_RECORD_DIVERGENCE,
    PERSISTENT_DIVERGENCE_DIGESTS,
    STREAKS_FILENAME,
    LaneDivergence,
    advance_lane_record_divergence,
    cross_check_lane_records,
)


def _lane(n: int) -> Path:
    return Path(f'/wt/_lane-{n}')


class TestCrossCheckLaneRecords:
    def test_coherent_views_have_no_divergence(self):
        assignments = {'T': _lane(0), 'U': _lane(1)}
        records = {
            '_lane-0': LaneRecord(state=LaneState.ASSIGNED, task_id='T'),
            '_lane-1': LaneRecord(state=LaneState.IN_USE, task_id='U'),
        }

        assert cross_check_lane_records(assignments, records) == ()

    def test_map_entry_with_released_record_reports_the_real_record_state(self):
        assignments = {'T': _lane(0)}
        records = {'_lane-0': LaneRecord(state=LaneState.RELEASED, task_id=None)}

        (divergence,) = cross_check_lane_records(assignments, records)

        assert divergence.lane == '_lane-0'
        assert divergence.map_task_id == 'T'
        assert divergence.record_state is LaneState.RELEASED
        assert divergence.record_task_id is None

    def test_record_held_for_another_task_carries_both_task_ids(self):
        assignments = {'T': _lane(0)}
        records = {'_lane-0': LaneRecord(state=LaneState.ASSIGNED, task_id='U')}

        (divergence,) = cross_check_lane_records(assignments, records)

        assert divergence.lane == '_lane-0'
        assert divergence.map_task_id == 'T'
        assert divergence.record_state is LaneState.ASSIGNED
        assert divergence.record_task_id == 'U'

    def test_map_entry_without_any_record(self):
        (divergence,) = cross_check_lane_records({'T': _lane(0)}, {})

        assert divergence.lane == '_lane-0'
        assert divergence.map_task_id == 'T'
        assert divergence.record_state is None
        assert divergence.record_task_id is None

    @pytest.mark.parametrize('state', [LaneState.ASSIGNED, LaneState.IN_USE])
    def test_holding_record_without_map_entry(self, state):
        records = {'_lane-0': LaneRecord(state=state, task_id='T')}

        (divergence,) = cross_check_lane_records({}, records)

        assert divergence.lane == '_lane-0'
        assert divergence.map_task_id is None
        assert divergence.record_state is state
        assert divergence.record_task_id == 'T'

    def test_non_holding_records_without_map_entry_are_coherent(self):
        records = {
            '_lane-0': LaneRecord(state=LaneState.RELEASED),
            '_lane-1': LaneRecord(state=LaneState.QUARANTINED),
            '_lane-2': LaneRecord(state=LaneState.REGISTERED),
            '_lane-3': LaneRecord(state=LaneState.SEED),
        }

        assert cross_check_lane_records({}, records) == ()

    def test_task_moved_between_lanes_diverges_on_both_lanes(self):
        assignments = {'T': _lane(1)}
        records = {'_lane-0': LaneRecord(state=LaneState.ASSIGNED, task_id='T')}

        divergences = cross_check_lane_records(assignments, records)

        assert [d.lane for d in divergences] == ['_lane-0', '_lane-1']
        stale_record, unrecorded_map = divergences
        assert (stale_record.map_task_id, stale_record.record_task_id) == (None, 'T')
        assert (unrecorded_map.map_task_id, unrecorded_map.record_state) == ('T', None)

    def test_result_is_a_tuple_sorted_by_lane_of_frozen_values(self):
        assignments = {'A': _lane(2), 'B': _lane(0), 'C': _lane(1)}

        divergences = cross_check_lane_records(assignments, {})

        assert isinstance(divergences, tuple)
        assert [d.lane for d in divergences] == ['_lane-0', '_lane-1', '_lane-2']
        pytest.raises(
            dataclasses.FrozenInstanceError, setattr, divergences[0], 'lane', '_lane-9',
        )

    def test_streaks_continue_from_previous_and_start_at_one(self):
        assignments = {'T': _lane(0), 'U': _lane(1)}

        divergences = cross_check_lane_records(
            assignments, {}, previous_streaks={'_lane-0': 1},
        )

        by_lane = {d.lane: d.consecutive_digests for d in divergences}
        assert by_lane == {'_lane-0': 2, '_lane-1': 1}

    def test_render_names_the_lane_and_both_views(self):
        divergence = LaneDivergence(
            lane='_lane-4', map_task_id='T7', record_state=LaneState.ASSIGNED,
            record_task_id='U8', consecutive_digests=3,
        )

        text = divergence.render()

        for fragment in ('_lane-4', 'T7', LaneState.ASSIGNED.value, 'U8', '3'):
            assert fragment in text

    def test_render_marks_missing_views(self):
        unassigned = LaneDivergence(
            lane='_lane-4', map_task_id=None, record_state=LaneState.IN_USE,
            record_task_id='U8',
        ).render()
        unrecorded = LaneDivergence(
            lane='_lane-5', map_task_id='T7', record_state=None, record_task_id=None,
        ).render()

        assert 'unassigned' in unassigned
        assert 'no record' in unrecorded
        assert '_lane-5' in unrecorded and 'T7' in unrecorded

    def test_persistence_threshold(self):
        def divergence(consecutive: int) -> LaneDivergence:
            return LaneDivergence(
                lane='_lane-0', map_task_id='T', record_state=None,
                record_task_id=None, consecutive_digests=consecutive,
            )

        assert LANE_RECORD_DIVERGENCE == 'lane_record_divergence'
        assert PERSISTENT_DIVERGENCE_DIGESTS == 2
        assert divergence(1).is_persistent is False
        assert divergence(PERSISTENT_DIVERGENCE_DIGESTS).is_persistent is True


_UNRECORDED_MAP_ENTRY = ({'T': _lane(0)}, {})


class TestAdvanceLaneRecordDivergence:
    @pytest.fixture
    def streaks_path(self, tmp_path):
        return tmp_path / 'digests' / STREAKS_FILENAME

    def test_first_divergence_starts_a_streak_on_disk(self, streaks_path):
        (divergence,) = advance_lane_record_divergence(
            *_UNRECORDED_MAP_ENTRY, streaks_path=streaks_path,
        )

        assert divergence.consecutive_digests == 1
        assert streaks_path.is_file()

    def test_streak_survives_through_the_file(self, streaks_path):
        advance_lane_record_divergence(*_UNRECORDED_MAP_ENTRY, streaks_path=streaks_path)

        (divergence,) = advance_lane_record_divergence(
            *_UNRECORDED_MAP_ENTRY, streaks_path=streaks_path,
        )

        assert divergence.consecutive_digests == 2

    def test_coherence_drops_the_streak(self, streaks_path):
        advance_lane_record_divergence(*_UNRECORDED_MAP_ENTRY, streaks_path=streaks_path)
        coherent_records = {'_lane-0': LaneRecord(state=LaneState.ASSIGNED, task_id='T')}

        assert advance_lane_record_divergence(
            {'T': _lane(0)}, coherent_records, streaks_path=streaks_path,
        ) == ()
        (divergence,) = advance_lane_record_divergence(
            *_UNRECORDED_MAP_ENTRY, streaks_path=streaks_path,
        )

        assert divergence.consecutive_digests == 1

    def test_streak_is_keyed_by_lane_not_by_divergence_shape(self, streaks_path):
        assignments = {'T': _lane(0)}
        advance_lane_record_divergence(
            assignments,
            {'_lane-0': LaneRecord(state=LaneState.RELEASED)},
            streaks_path=streaks_path,
        )

        (divergence,) = advance_lane_record_divergence(
            assignments,
            {'_lane-0': LaneRecord(state=LaneState.ASSIGNED, task_id='U')},
            streaks_path=streaks_path,
        )

        assert divergence.consecutive_digests == 2

    @pytest.mark.parametrize('content', ['not json', '["_lane-0"]', '{"_lane-0": "x"}'])
    def test_corrupt_streaks_file_counts_as_no_prior_streak(
        self, streaks_path, content, caplog,
    ):
        streaks_path.parent.mkdir(parents=True)
        streaks_path.write_text(content, encoding='utf-8')

        with caplog.at_level('WARNING'):
            (first,) = advance_lane_record_divergence(
                *_UNRECORDED_MAP_ENTRY, streaks_path=streaks_path,
            )
        (second,) = advance_lane_record_divergence(
            *_UNRECORDED_MAP_ENTRY, streaks_path=streaks_path,
        )

        assert first.consecutive_digests == 1
        assert any(
            r.levelname == 'WARNING' and str(streaks_path) in r.getMessage()
            for r in caplog.records
        )
        assert second.consecutive_digests == 2
