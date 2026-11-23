"""Tests for orchestrator.chain_telemetry — the deep merge-ahead chain aggregation core (task 3188 ε)."""

from __future__ import annotations

import dataclasses

import pytest

from orchestrator.chain_telemetry import ChainBucket, ChainTelemetry, summarize_chain_telemetry


def _histogram_counts(telemetry: ChainTelemetry) -> dict[int, tuple[int, int]]:
    return {k: (b.passed, b.total) for k, b in telemetry.chain_items_histogram.items()}


class TestSummarizeChainTelemetryHistogram:
    def test_buckets_by_chain_items_with_pass_rate(self):
        telemetry = summarize_chain_telemetry(
            [
                {'chain_items': 1, 'passed': True},
                {'chain_items': 1, 'passed': False},
                {'chain_items': 4, 'passed': True},
            ],
            [],
        )
        histogram = telemetry.chain_items_histogram
        assert histogram == {1: ChainBucket(1, 2), 4: ChainBucket(1, 1)}, f'{histogram=}'
        assert histogram[1].rate == 0.5, f'{histogram[1].rate=}'
        assert histogram[4].rate == 1.0, f'{histogram[4].rate=}'

    def test_empty_bucket_rate_is_none(self):
        assert ChainBucket(0, 0).rate is None, f'{ChainBucket(0, 0).rate=}'

    def test_payload_without_chain_items_is_not_bucketed(self):
        telemetry = summarize_chain_telemetry([{'passed': True}], [])
        assert telemetry.chain_items_histogram == {}, f'{telemetry.chain_items_histogram=}'

    @pytest.mark.parametrize(
        'untrusted', [None, '2', True, 0, -1, 2.0],
        ids=['none', 'str', 'bool', 'zero', 'negative', 'float'],
    )
    def test_untrusted_chain_items_is_skipped_not_coerced(self, untrusted):
        telemetry = summarize_chain_telemetry([{'chain_items': untrusted, 'passed': True}], [])
        assert telemetry.chain_items_histogram == {}, f'{untrusted=!r}: {telemetry=}'
        assert telemetry.deep_verifies == 0, f'{untrusted=!r}: {telemetry.deep_verifies=}'

    def test_histogram_iterates_in_ascending_chain_items(self):
        telemetry = summarize_chain_telemetry(
            [{'chain_items': k, 'passed': True} for k in (4, 1, 3)], [],
        )
        assert list(telemetry.chain_items_histogram) == [1, 3, 4], (
            f'{list(telemetry.chain_items_histogram)=}'
        )

    def test_empty_input_is_the_neutral_value(self):
        telemetry = summarize_chain_telemetry([], [])
        assert telemetry == ChainTelemetry(), f'{telemetry=}'

    def test_neutral_value_has_no_signal(self):
        neutral = ChainTelemetry()
        assert neutral.chain_items_histogram == {}, f'{neutral=}'
        assert (neutral.deep_verifies, neutral.deep_failures) == (0, 0), f'{neutral=}'
        assert neutral.deep_fail_rate is None, f'{neutral=}'
        assert neutral.items_landed_via_chain == 0, f'{neutral=}'
        assert neutral.items_landed_per_deep_verify is None, f'{neutral=}'
        assert neutral.excluded_legacy_depth == 0, f'{neutral=}'

    def test_values_are_frozen(self):
        pytest.raises(dataclasses.FrozenInstanceError, setattr, ChainTelemetry(), 'deep_verifies', 1)
        pytest.raises(dataclasses.FrozenInstanceError, setattr, ChainBucket(), 'total', 1)


class TestSummarizeChainTelemetryRates:
    def test_only_chain_items_of_two_or_more_are_deep(self):
        telemetry = summarize_chain_telemetry(
            [
                {'chain_items': 1, 'passed': False},
                {'chain_items': 2, 'passed': True},
                {'chain_items': 3, 'passed': False},
            ],
            [],
        )
        assert telemetry.deep_verifies == 2, f'{telemetry.deep_verifies=}'
        assert telemetry.deep_failures == 1, f'{telemetry.deep_failures=}'

    def test_missing_passed_counts_as_a_failure(self):
        telemetry = summarize_chain_telemetry([{'chain_items': 2}], [])
        assert telemetry.deep_failures == 1, f'{telemetry.deep_failures=}'

    def test_deep_fail_rate(self):
        telemetry = summarize_chain_telemetry(
            [{'chain_items': 2, 'passed': p} for p in (True, True, True, False)], [],
        )
        assert telemetry.deep_fail_rate == 0.25, f'{telemetry.deep_fail_rate=}'

    @pytest.mark.parametrize(
        'untrusted', [None, 0, '1', True, 1.0],
        ids=['none', 'zero', 'str', 'bool', 'float'],
    )
    def test_untrusted_landed_via_chain_is_not_summed(self, untrusted):
        telemetry = summarize_chain_telemetry(
            [{'chain_items': 2, 'passed': True}],
            [{'landed_via_chain': untrusted}, {'landed_via_chain': 1}],
        )
        assert telemetry.items_landed_via_chain == 1, (
            f'{untrusted=!r}: {telemetry.items_landed_via_chain=}'
        )

    def test_absent_landed_via_chain_is_not_summed(self):
        telemetry = summarize_chain_telemetry([], [{'state': 'done'}])
        assert telemetry.items_landed_via_chain == 0, f'{telemetry.items_landed_via_chain=}'

    def test_items_landed_per_deep_verify(self):
        telemetry = summarize_chain_telemetry(
            [{'chain_items': 3, 'passed': True}, {'chain_items': 3, 'passed': True}],
            [{'landed_via_chain': 1}] * 6,
        )
        assert telemetry.items_landed_via_chain == 6, f'{telemetry.items_landed_via_chain=}'
        assert telemetry.items_landed_per_deep_verify == 3.0, (
            f'{telemetry.items_landed_per_deep_verify=}'
        )

    def test_no_deep_verify_means_no_signal_not_zero(self):
        telemetry = summarize_chain_telemetry(
            [{'chain_items': 1, 'passed': True}, {'chain_items': 1, 'passed': False}],
            [{'landed_via_chain': 1}],
        )
        assert telemetry.deep_verifies == 0, f'{telemetry.deep_verifies=}'
        assert telemetry.deep_fail_rate is None, f'{telemetry.deep_fail_rate=}'
        assert telemetry.items_landed_per_deep_verify is None, (
            f'{telemetry.items_landed_per_deep_verify=}'
        )


class TestSummarizeChainTelemetryLegacyDepthExclusion:
    @pytest.mark.parametrize('depth', [3, '2'], ids=['int', 'str'])
    def test_pre_chain_items_deep_label_is_counted_and_not_bucketed(self, depth):
        telemetry = summarize_chain_telemetry([{'depth': depth, 'passed': True}], [])
        assert telemetry.excluded_legacy_depth == 1, f'{depth=!r}: {telemetry=}'
        assert telemetry.chain_items_histogram == {}, f'{depth=!r}: {telemetry=}'

    @pytest.mark.parametrize(
        'payload',
        [{'depth': 1}, {'depth': 0}, {'depth': None}, {}, {'depth': 'abc'}, {'depth': [3]}],
        ids=['one', 'zero', 'none', 'absent', 'unparseable', 'list'],
    )
    def test_shallow_or_unreadable_depth_is_not_a_legacy_label(self, payload):
        telemetry = summarize_chain_telemetry([{**payload, 'passed': True}], [])
        assert telemetry.excluded_legacy_depth == 0, f'{payload=}: {telemetry=}'

    def test_chain_items_overrides_the_depth_label(self):
        telemetry = summarize_chain_telemetry(
            [{'depth': 3, 'chain_items': 1, 'passed': True}], [],
        )
        assert telemetry.excluded_legacy_depth == 0, f'{telemetry=}'
        assert _histogram_counts(telemetry) == {1: (1, 1)}, f'{telemetry=}'

    def test_all_post_chain_items_traffic_excludes_nothing(self):
        telemetry = summarize_chain_telemetry(
            [{'depth': d, 'chain_items': k, 'passed': True} for d, k in ((2, 1), (5, 4))], [],
        )
        assert telemetry.excluded_legacy_depth == 0, f'{telemetry=}'
