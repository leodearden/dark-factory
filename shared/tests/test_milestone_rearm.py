"""Tests for shared.milestone_rearm — a predicate re-arm keeps the link's scheduled fire time.

plans/completion-driven-triggers-prd.md C1/D1: the re-arm advances
``milestone.at`` (the scheduler's one timer); the mint's run-label strip and the
chain-state gauge read the scheduled fire time through ``scheduled_fire_at``.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from shared.milestone_rearm import rearmed_milestone, scheduled_fire_at

SCHEDULED = '2026-09-02T12:00:00+00:00'
FIRST_CHECK = '2026-09-02T18:00:00+00:00'
SECOND_CHECK = '2026-09-03T00:00:00+00:00'


def _dated() -> dict[str, Any]:
    return {'mode': 'dated', 'at': SCHEDULED, 'x_note': 'kept'}


class TestRearmedMilestone:
    def test_first_rearm_advances_at_and_keeps_the_scheduled_fire_time(self) -> None:
        rearmed = rearmed_milestone(_dated(), FIRST_CHECK)

        assert rearmed['at'] == FIRST_CHECK
        assert rearmed['x_note'] == 'kept'
        assert scheduled_fire_at(rearmed) == SCHEDULED

    def test_a_second_rearm_leaves_the_scheduled_fire_time_where_the_first_put_it(self) -> None:
        rearmed = rearmed_milestone(rearmed_milestone(_dated(), FIRST_CHECK), SECOND_CHECK)

        assert rearmed['at'] == SECOND_CHECK
        assert scheduled_fire_at(rearmed) == SCHEDULED

    def test_the_persisted_shape_adds_only_the_scheduled_fire_at_key(self) -> None:
        original = _dated()

        assert rearmed_milestone(original, FIRST_CHECK) == {
            **original,
            'at': FIRST_CHECK,
            'scheduled_fire_at': SCHEDULED,
        }

    def test_the_input_milestone_is_not_mutated(self) -> None:
        original = _dated()
        before = copy.deepcopy(original)

        rearmed_milestone(original, FIRST_CHECK)

        assert original == before


class TestScheduledFireAt:
    def test_a_never_rearmed_dated_milestone_fires_at_its_at(self) -> None:
        assert scheduled_fire_at(_dated()) == SCHEDULED

    @pytest.mark.parametrize(
        'milestone',
        [
            None,
            'x',
            {'mode': 'delayed', 'after_secs': 60},
            {'mode': 'dated', 'at': 1788350400},
        ],
        ids=['none', 'non-mapping', 'delayed', 'non-str-at'],
    )
    def test_no_readable_fire_time_reads_none(self, milestone: object) -> None:
        assert scheduled_fire_at(milestone) is None
