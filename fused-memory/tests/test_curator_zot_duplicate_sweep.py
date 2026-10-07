"""Tests for the post-ZOT duplicate sweep (task 5491)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from fused_memory.middleware.curator_zot_duplicate_sweep import (
    DuplicateFinding,
    build_duplicate_metadata,
    select_near_duplicate,
    sweep_zot_duplicate,
)
from fused_memory.middleware.task_curator import embedding_text


def _hit(task_id, score, title='dup title'):
    return {'task_id': task_id, 'title': title, 'score': score}


class TestSelectNearDuplicate:
    def test_single_eligible_hit_is_returned_as_finding(self):
        finding = select_near_duplicate(
            [_hit('42', 0.80, title='Existing task')],
            self_task_id='99',
            statuses={'42': 'pending'},
            threshold=0.65,
        )
        assert finding == DuplicateFinding(
            task_id='99',
            duplicate_task_id='42',
            duplicate_title='Existing task',
            score=0.80,
        )

    def test_self_hit_is_dropped_even_at_perfect_score(self):
        assert select_near_duplicate(
            [_hit('99', 1.0)],
            self_task_id='99',
            statuses={'99': 'pending'},
            threshold=0.65,
        ) is None

    def test_self_hit_compares_as_str(self):
        assert select_near_duplicate(
            [_hit(7, 1.0)],
            self_task_id='7',
            statuses={'7': 'pending'},
            threshold=0.65,
        ) is None

    def test_hit_exactly_at_threshold_is_kept(self):
        finding = select_near_duplicate(
            [_hit('42', 0.65)],
            self_task_id='99',
            statuses={'42': 'pending'},
            threshold=0.65,
        )
        assert finding is not None
        assert finding.duplicate_task_id == '42'

    def test_hit_just_below_threshold_is_dropped(self):
        assert select_near_duplicate(
            [_hit('42', 0.6499)],
            self_task_id='99',
            statuses={'42': 'pending'},
            threshold=0.65,
        ) is None

    def test_cancelled_hit_is_dropped(self):
        assert select_near_duplicate(
            [_hit('42', 0.9)],
            self_task_id='99',
            statuses={'42': 'cancelled'},
            threshold=0.65,
        ) is None

    def test_hit_absent_from_statuses_is_dropped(self):
        assert select_near_duplicate(
            [_hit('42', 0.9)],
            self_task_id='99',
            statuses={},
            threshold=0.65,
        ) is None

    @pytest.mark.parametrize('status', ['unknown', 'archived', '', 'PENDING'])
    def test_status_outside_vocabulary_is_dropped(self, status):
        assert select_near_duplicate(
            [_hit('42', 0.9)],
            self_task_id='99',
            statuses={'42': status},
            threshold=0.65,
        ) is None

    @pytest.mark.parametrize('status', ['in-progress', 'done', 'blocked', 'deferred'])
    def test_non_pending_live_statuses_are_kept(self, status):
        finding = select_near_duplicate(
            [_hit('42', 0.9)],
            self_task_id='99',
            statuses={'42': status},
            threshold=0.65,
        )
        assert finding is not None
        assert finding.duplicate_task_id == '42'

    def test_highest_scoring_eligible_hit_wins(self):
        finding = select_near_duplicate(
            [_hit('1', 0.70), _hit('2', 0.92), _hit('3', 0.81)],
            self_task_id='99',
            statuses={'1': 'pending', '2': 'in-progress', '3': 'done'},
            threshold=0.65,
        )
        assert finding is not None
        assert finding.duplicate_task_id == '2'
        assert finding.score == 0.92

    def test_highest_scoring_ineligible_hit_does_not_mask_eligible_one(self):
        finding = select_near_duplicate(
            [_hit('1', 0.70), _hit('2', 0.95)],
            self_task_id='99',
            statuses={'1': 'pending', '2': 'cancelled'},
            threshold=0.65,
        )
        assert finding is not None
        assert finding.duplicate_task_id == '1'

    def test_empty_hits_returns_none(self):
        assert select_near_duplicate(
            [], self_task_id='99', statuses={}, threshold=0.65,
        ) is None

    def test_all_ineligible_returns_none(self):
        assert select_near_duplicate(
            [_hit('99', 1.0), _hit('1', 0.9), _hit('2', 0.5)],
            self_task_id='99',
            statuses={'1': 'cancelled', '2': 'pending'},
            threshold=0.65,
        ) is None

    @pytest.mark.parametrize(
        'malformed',
        [
            {'title': 'no id', 'score': 0.9},
            {'task_id': None, 'title': 'none id', 'score': 0.9},
            {'task_id': '', 'title': 'empty id', 'score': 0.9},
            {'task_id': '5', 'title': 'no score'},
            {'task_id': '5', 'title': 'none score', 'score': None},
            {'task_id': '5', 'title': 'str score', 'score': 'high'},
            {'task_id': '5', 'title': 'bool score', 'score': True},
        ],
    )
    def test_malformed_hit_is_skipped_not_raised(self, malformed):
        finding = select_near_duplicate(
            [malformed, _hit('42', 0.80)],
            self_task_id='99',
            statuses={'5': 'pending', '42': 'pending'},
            threshold=0.65,
        )
        assert finding is not None
        assert finding.duplicate_task_id == '42'

    def test_missing_title_becomes_empty_string(self):
        finding = select_near_duplicate(
            [{'task_id': '42', 'score': 0.8}],
            self_task_id='99',
            statuses={'42': 'pending'},
            threshold=0.65,
        )
        assert finding is not None
        assert finding.duplicate_title == ''


class _FakeSearchCurator:
    def __init__(self, hits=None, *, raises: BaseException | None = None):
        self._hits = list(hits or [])
        self._raises = raises
        self.calls: list[dict] = []

    async def search_corpus(self, query, project_id, *, limit=10, score_threshold=0.3):
        self.calls.append({
            'query': query,
            'project_id': project_id,
            'limit': limit,
            'score_threshold': score_threshold,
        })
        if self._raises is not None:
            raise self._raises
        return list(self._hits)


class _RecordingStatusReader:
    def __init__(self, statuses=None, *, raises: BaseException | None = None):
        self._statuses = dict(statuses or {})
        self._raises = raises
        self.calls: list[list[str]] = []

    async def __call__(self, ids):
        self.calls.append(list(ids))
        if self._raises is not None:
            raise self._raises
        return {i: self._statuses[i] for i in ids if i in self._statuses}


async def _sweep(curator, read_statuses, **overrides):
    kwargs = {
        'project_id': 'proj',
        'task_id': '99',
        'title': 'Fix the flaky merge gate',
        'description': 'It flakes under load',
        'files_to_modify': ['a.py', 'b.py'],
        'read_statuses': read_statuses,
        'threshold': 0.65,
        'limit': 5,
    }
    kwargs.update(overrides)
    return await sweep_zot_duplicate(curator, **kwargs)


class TestSweepZotDuplicate:
    @pytest.mark.asyncio
    async def test_flags_the_near_duplicate_not_the_self_hit(self):
        curator = _FakeSearchCurator([_hit('99', 1.0, 'self'), _hit('42', 0.80, 'Existing')])
        reader = _RecordingStatusReader({'99': 'pending', '42': 'pending'})
        finding = await _sweep(curator, reader)
        assert finding == DuplicateFinding(
            task_id='99', duplicate_task_id='42', duplicate_title='Existing', score=0.80,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('hits', [[], [_hit('99', 1.0)], [_hit(99, 1.0)]])
    async def test_no_candidate_hits_returns_none_without_status_read(self, hits):
        curator = _FakeSearchCurator(hits)
        reader = _RecordingStatusReader({'99': 'pending'})
        assert await _sweep(curator, reader) is None
        assert reader.calls == []

    @pytest.mark.asyncio
    async def test_status_read_is_scoped_to_non_self_hit_ids(self):
        curator = _FakeSearchCurator([_hit('99', 1.0), _hit(42, 0.8), _hit('17', 0.7)])
        reader = _RecordingStatusReader({'42': 'pending', '17': 'done'})
        await _sweep(curator, reader)
        assert len(reader.calls) == 1
        assert sorted(reader.calls[0]) == ['17', '42']
        assert all(isinstance(i, str) for i in reader.calls[0])

    @pytest.mark.asyncio
    async def test_query_and_knobs_are_forwarded(self):
        curator = _FakeSearchCurator([])
        reader = _RecordingStatusReader()
        await _sweep(curator, reader, threshold=0.71, limit=3)
        assert curator.calls == [{
            'query': embedding_text('Fix the flaky merge gate', 'It flakes under load', ['a.py', 'b.py']),
            'project_id': 'proj',
            'limit': 3,
            'score_threshold': 0.71,
        }]

    @pytest.mark.asyncio
    async def test_search_error_yields_none(self):
        curator = _FakeSearchCurator(raises=RuntimeError('qdrant down'))
        reader = _RecordingStatusReader()
        assert await _sweep(curator, reader) is None
        assert reader.calls == []

    @pytest.mark.asyncio
    async def test_status_read_error_yields_none(self):
        curator = _FakeSearchCurator([_hit('42', 0.9)])
        reader = _RecordingStatusReader(raises=RuntimeError('db locked'))
        assert await _sweep(curator, reader) is None

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        curator = _FakeSearchCurator(raises=asyncio.CancelledError())
        reader = _RecordingStatusReader()
        with pytest.raises(asyncio.CancelledError):
            await _sweep(curator, reader)


class TestBuildDuplicateMetadata:
    _FINDING = DuplicateFinding(
        task_id='99', duplicate_task_id='42', duplicate_title='Existing', score=0.8,
    )

    def test_single_namespaced_key(self):
        metadata = build_duplicate_metadata(self._FINDING, zot_escalation_id='esc-curator-1')
        assert list(metadata) == ['x_zot_duplicate_candidate']

    def test_value_carries_the_finding(self):
        value = build_duplicate_metadata(
            self._FINDING, zot_escalation_id='esc-curator-1',
        )['x_zot_duplicate_candidate']
        assert value['duplicate_task_id'] == '42'
        assert isinstance(value['duplicate_task_id'], str)
        assert value['duplicate_title'] == 'Existing'
        assert value['score'] == 0.8
        assert isinstance(value['score'], float)
        assert value['zot_escalation_id'] == 'esc-curator-1'

    def test_absent_escalation_id_is_none(self):
        value = build_duplicate_metadata(
            self._FINDING, zot_escalation_id=None,
        )['x_zot_duplicate_candidate']
        assert value['zot_escalation_id'] is None

    def test_flagged_at_is_iso_utc(self):
        value = build_duplicate_metadata(
            self._FINDING, zot_escalation_id=None,
        )['x_zot_duplicate_candidate']
        parsed = datetime.fromisoformat(value['flagged_at'])
        assert parsed.utcoffset() == timedelta(0)
