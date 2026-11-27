"""Contract tests for ``shared.routing_split.split_bucket``.

The golden values pin the PRD formula
(plans/routing-experiment-substrate-prd.md §"Contract: split_bucket"), so a
silent change to the separator, slice width or byte order cannot re-shuffle
the arms of a live experiment.
"""

from __future__ import annotations

import pytest

from shared.routing_split import split_bucket

_TEN_THOUSAND_KEYS = [str(i) for i in range(1, 10_001)]


class TestGoldenValues:
    @pytest.mark.parametrize(
        ('salt', 'key', 'buckets', 'expected'),
        [
            ('routing-exp-1', '8373', 1_000_003, 153099),
            ('routing-exp-2', 'a1b2c3d4e5f60718', 1_000_003, 931072),
            ('t', '42', 7, 1),
            ('routing-exp-1', '1', 2, 0),
            ('routing-exp-1', '3', 2, 1),
        ],
    )
    def test_formula_is_pinned(self, salt, key, buckets, expected):
        assert split_bucket(salt, key, buckets) == expected


class TestRange:
    @pytest.mark.parametrize('buckets', [2, 3, 7])
    def test_every_result_is_a_valid_bucket(self, buckets):
        results = {split_bucket('t', str(i), buckets) for i in range(1, 501)}
        assert all(0 <= r < buckets for r in results)


class TestDeterminism:
    def test_same_inputs_give_same_bucket(self):
        assert split_bucket('routing-exp-1', '8373', 5) == split_bucket(
            'routing-exp-1', '8373', 5
        )


class TestBucketsBelowTwoRejected:
    @pytest.mark.parametrize('buckets', [1, 0, -3])
    def test_raises_naming_the_value(self, buckets):
        with pytest.raises(ValueError, match=repr(buckets)):
            split_bucket('t', '42', buckets)


class TestDistribution:
    def test_two_arms_are_balanced(self):
        in_arm_one = sum(split_bucket('t', k, 2) == 1 for k in _TEN_THOUSAND_KEYS)
        assert 0.48 <= in_arm_one / len(_TEN_THOUSAND_KEYS) <= 0.52

    def test_different_salts_split_independently(self):
        agree = sum(
            split_bucket('a', k, 2) == split_bucket('b', k, 2)
            for k in _TEN_THOUSAND_KEYS
        )
        assert 0.48 <= agree / len(_TEN_THOUSAND_KEYS) <= 0.52
