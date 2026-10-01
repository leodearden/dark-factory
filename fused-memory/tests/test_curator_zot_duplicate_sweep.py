"""Tests for the post-ZOT duplicate sweep (task 5491)."""

from __future__ import annotations

import pytest

from fused_memory.middleware.curator_zot_duplicate_sweep import (
    DuplicateFinding,
    select_near_duplicate,
)


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
