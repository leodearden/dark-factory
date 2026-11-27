"""scripts/split_resampling.py: the resampling draws over a routing split's tasks."""
from __future__ import annotations

import random
import statistics

import pytest
import split_resampling


def test_permuted_arms_draws_every_arm_but_the_last_at_its_observed_size():
    draws = list(split_resampling.permuted_arms([3, 4, 5], 50, random.Random(7)))

    assert len(draws) == 50
    for drawn in draws:
        assert [len(arm) for arm in drawn] == [3, 4]
        members = [index for arm in drawn for index in arm]
        assert len(set(members)) == len(members)
        assert set(members) <= set(range(12))


def test_permuted_arms_is_reproducible_from_its_seed():
    first = list(split_resampling.permuted_arms([3, 4, 5], 50, random.Random(7)))
    second = list(split_resampling.permuted_arms([3, 4, 5], 50, random.Random(7)))

    assert first == second


def test_bootstrap_arms_keeps_each_arm_size_and_draws_only_its_own_members():
    members = [[0, 1, 2], [3, 4]]

    draws = list(split_resampling.bootstrap_arms(members, 200, random.Random(7)))

    assert len(draws) == 200
    for drawn in draws:
        assert [len(arm) for arm in drawn] == [3, 2]
        assert set(drawn[0]) <= {0, 1, 2}
        assert set(drawn[1]) <= {3, 4}


def test_bootstrap_arms_draws_with_replacement():
    draws = split_resampling.bootstrap_arms([[0, 1, 2]], 200, random.Random(7))

    assert any(len(set(arm)) < len(arm) for (arm,) in draws)


def test_percentile_interval_interpolates_as_type_7():
    assert split_resampling.percentile_interval(list(range(1, 102)), 0.90) == (6.0, 96.0)


def test_percentile_interval_reads_unsorted_values_as_their_sorted_order():
    values = list(range(1, 102))
    random.Random(3).shuffle(values)

    assert split_resampling.percentile_interval(values, 0.90) == (6.0, 96.0)


def test_a_wider_level_contains_the_narrower_interval_on_the_same_values():
    values = [random.Random(11).gauss(0.0, 1.0) for _ in range(500)]
    narrow_low, narrow_high = split_resampling.percentile_interval(values, 0.90)
    wide_low, wide_high = split_resampling.percentile_interval(values, 0.97)

    assert wide_low < narrow_low < narrow_high < wide_high


@pytest.mark.parametrize('level', [0.0, 1.0, 1.5, -0.1])
def test_percentile_interval_refuses_a_level_outside_the_open_unit_interval(level):
    with pytest.raises(ValueError, match=repr(level)):
        split_resampling.percentile_interval([1.0, 2.0, 3.0], level)


def test_percentile_interval_refuses_fewer_than_two_values():
    with pytest.raises(ValueError, match='2 values'):
        split_resampling.percentile_interval([1.0], 0.90)


def test_null_cuts_are_the_inclusive_p95_and_p99():
    values = [random.Random(5).expovariate(1.0) for _ in range(300)]
    cuts = statistics.quantiles(values, n=100, method='inclusive')

    assert split_resampling.null_cuts(values) == (cuts[94], cuts[98])


def test_null_cuts_refuse_fewer_than_two_values():
    with pytest.raises(ValueError, match='2 values'):
        split_resampling.null_cuts([1.0])
