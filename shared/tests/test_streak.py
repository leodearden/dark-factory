"""Tests for ``shared.streak.KeyedStreaks`` through its public API.

Contract: plans/review-all-s14-shared-leaves-prd.md, decision 11, invariant I7
(fires once per run) and boundary test B10.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Hashable

import pytest

from shared.streak import KeyedStreaks, StreakStep


def test_b10_fires_at_the_third_failure_of_each_run():
    streaks: KeyedStreaks[str] = KeyedStreaks(threshold=3)

    steps = [streaks.failure('k') for _ in range(5)]
    assert steps == [
        StreakStep(1, False),
        StreakStep(2, False),
        StreakStep(3, True),
        StreakStep(4, False),
        StreakStep(5, False),
    ]

    streaks.success('k')
    assert streaks.count('k') == 0

    second = [streaks.failure('k') for _ in range(3)]
    assert second == [StreakStep(1, False), StreakStep(2, False), StreakStep(3, True)]


@pytest.mark.parametrize('threshold', [1, 2, 3, 7])
def test_fires_exactly_once_per_run_at_the_threshold(threshold: int):
    streaks: KeyedStreaks[str] = KeyedStreaks(threshold=threshold)

    for _ in range(2):
        run = [streaks.failure('k') for _ in range(2 * threshold + 1)]
        assert [step.count for step in run if step.fired] == [threshold]
        streaks.success('k')


def test_count_tracks_the_current_run():
    streaks: KeyedStreaks[str] = KeyedStreaks(threshold=3)
    assert streaks.count('k') == 0

    for _ in range(4):
        streaks.failure('k')
    assert streaks.count('k') == 4

    streaks.success('k')
    assert streaks.count('k') == 0


def test_success_on_an_untracked_key_is_a_noop():
    streaks: KeyedStreaks[str] = KeyedStreaks(threshold=3)

    streaks.success('never-seen')

    assert streaks.count('never-seen') == 0


@pytest.mark.parametrize('bad', [0, -1, 2.5, True, '3'])
def test_rejects_a_threshold_that_could_never_fire(bad):
    with pytest.raises(ValueError) as excinfo:
        KeyedStreaks(threshold=bad)

    assert 'threshold' in str(excinfo.value)
    assert repr(bad) in str(excinfo.value)


def test_accepts_threshold_one():
    assert KeyedStreaks(threshold=1).failure('k') == StreakStep(1, True)


def test_tuple_keys_are_independent_per_component():
    streaks: KeyedStreaks[tuple[str, str]] = KeyedStreaks(threshold=2)
    first, second = ('task-1', 'dep-a'), ('task-1', 'dep-b')

    assert [streaks.failure(first) for _ in range(2)] == [
        StreakStep(1, False),
        StreakStep(2, True),
    ]
    assert streaks.count(second) == 0
    assert streaks.failure(second) == StreakStep(1, False)


def test_str_and_tuple_keys_do_not_collide():
    streaks: KeyedStreaks[Hashable] = KeyedStreaks(threshold=3)

    streaks.failure('a')
    streaks.failure(('a',))
    streaks.failure(('a',))

    assert streaks.count('a') == 1
    assert streaks.count(('a',)) == 2


def test_success_resets_only_its_own_key():
    streaks: KeyedStreaks[str] = KeyedStreaks(threshold=5)
    for _ in range(2):
        streaks.failure('a')
        streaks.failure('b')

    streaks.success('a')

    assert streaks.count('a') == 0
    assert streaks.count('b') == 2
    assert streaks.failure('b') == StreakStep(3, False)


def test_tracked_keys_holds_only_keys_in_a_failure_run():
    streaks: KeyedStreaks[Hashable] = KeyedStreaks(threshold=3)
    assert streaks.tracked_keys == frozenset()

    streaks.failure('a')
    streaks.failure(('b', 1))
    assert streaks.tracked_keys == frozenset({'a', ('b', 1)})

    streaks.count('never')
    assert 'never' not in streaks.tracked_keys

    streaks.success('a')
    snapshot = streaks.tracked_keys
    assert snapshot == frozenset({('b', 1)})
    assert isinstance(snapshot, frozenset)


@pytest.mark.parametrize('field', ['count', 'fired'])
def test_streak_step_is_a_frozen_value(field: str):
    step = StreakStep(2, False)
    assert step == StreakStep(count=2, fired=False)

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(step, field, 3)
