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


def test_forget_drops_the_named_keys_and_ignores_unknown_ones():
    streaks: KeyedStreaks[Hashable] = KeyedStreaks(threshold=3)
    streaks.failure('a')
    streaks.failure('b')
    streaks.failure(('c', 1))

    streaks.forget(key for key in ['a', ('c', 1), 'never-seen'])

    assert streaks.count('a') == 0
    assert streaks.count(('c', 1)) == 0
    assert streaks.count('never-seen') == 0
    assert streaks.count('b') == 1


def test_a_forgotten_key_starts_a_fresh_run():
    streaks: KeyedStreaks[str] = KeyedStreaks(threshold=2)
    assert [streaks.failure('k') for _ in range(2)][-1] == StreakStep(2, True)

    streaks.forget(['k'])

    assert [streaks.failure('k') for _ in range(2)] == [
        StreakStep(1, False),
        StreakStep(2, True),
    ]


def test_retain_keeps_only_keys_the_predicate_accepts():
    streaks: KeyedStreaks[tuple[str, str]] = KeyedStreaks(threshold=3)
    for key in [('t1', 'd1'), ('t1', 'd2'), ('t2', 'd1')]:
        streaks.failure(key)
    live = {'t1'}

    streaks.retain(lambda key: key[0] in live)

    assert streaks.count(('t2', 'd1')) == 0
    assert streaks.count(('t1', 'd1')) == 1
    assert streaks.count(('t1', 'd2')) == 1


def test_retain_with_a_raising_predicate_changes_nothing():
    streaks: KeyedStreaks[str] = KeyedStreaks(threshold=3)
    streaks.failure('a')
    streaks.failure('b')

    def keep(key: str) -> bool:
        if key == 'b':
            raise RuntimeError('predicate failed on b')
        return False

    with pytest.raises(RuntimeError):
        streaks.retain(keep)

    assert streaks.count('a') == 1
    assert streaks.count('b') == 1


@pytest.mark.parametrize('field', ['count', 'fired'])
def test_streak_step_is_a_frozen_value(field: str):
    step = StreakStep(2, False)
    assert step == StreakStep(count=2, fired=False)

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(step, field, 3)


def test_tracked_keys_is_empty_for_a_fresh_instance():
    assert KeyedStreaks(threshold=3).tracked_keys == frozenset()


def test_tracked_keys_lists_the_keys_in_a_live_run():
    streaks: KeyedStreaks[Hashable] = KeyedStreaks(threshold=3)

    streaks.failure('a')
    streaks.failure(('b', 1))

    assert streaks.tracked_keys == frozenset({'a', ('b', 1)})


def test_tracked_keys_drops_keys_whose_run_ended():
    streaks: KeyedStreaks[Hashable] = KeyedStreaks(threshold=3)
    for key in ['a', ('b', 1), 'c']:
        streaks.failure(key)

    streaks.success('a')
    assert streaks.tracked_keys == frozenset({('b', 1), 'c'})

    streaks.forget([('b', 1)])
    assert streaks.tracked_keys == frozenset({'c'})

    streaks.retain(lambda key: False)
    assert streaks.tracked_keys == frozenset()


def test_tracked_keys_is_a_snapshot():
    streaks: KeyedStreaks[str] = KeyedStreaks(threshold=3)
    streaks.failure('a')

    snapshot = streaks.tracked_keys
    streaks.failure('b')

    assert isinstance(snapshot, frozenset)
    assert snapshot == frozenset({'a'})
    assert streaks.tracked_keys == frozenset({'a', 'b'})
