"""Tests for agents/token_pricing.py — the single home of the per-token USD formula."""

from __future__ import annotations

import dataclasses

import pytest

from orchestrator.agents.token_pricing import (
    FALLBACK_PRICE,
    TokenUsage,
    price_fields,
    price_tokens,
)
from orchestrator.config import PriceEntry

# Measured counts of eval cell endpoint__46074e01 (a glm-5.3 workflow run).
MEASURED_USAGE = TokenUsage(
    input_tokens=1_193,
    output_tokens=12_537,
    cache_read_tokens=1_370_402,
    cache_create_tokens=13_702,
)

GLM_RATES = {
    'input_per_1m': 1.40,
    'output_per_1m': 4.40,
    'cached_input_per_1m': 0.26,
    'cache_write_per_1m': 1.40,
}


@pytest.fixture(params=['dict', 'price_entry'])
def entry_shape(request):
    """Build a price entry as a plain dict or as a PriceEntry."""
    if request.param == 'dict':
        return dict
    return lambda rates: PriceEntry(**rates)


class TestPriceTokens:
    def test_all_four_counts_are_priced_at_their_own_rates(self, entry_shape):
        priced = price_tokens(entry_shape(GLM_RATES), MEASURED_USAGE)

        assert priced.cost_usd == pytest.approx(
            (1_193 * 1.40 + 12_537 * 4.40 + 1_370_402 * 0.26 + 13_702 * 1.40) / 1e6,
        )
        assert priced.cost_usd == pytest.approx(0.43232032)
        assert priced.defaulted_rates == frozenset()

    def test_missing_cached_input_rate_prices_cache_reads_at_input_rate(self, entry_shape):
        entry = entry_shape({'input_per_1m': 1.0, 'output_per_1m': 4.0, 'cache_write_per_1m': 1.0})
        usage = TokenUsage(input_tokens=100, output_tokens=10, cache_read_tokens=1_000)

        priced = price_tokens(entry, usage)

        assert priced.cost_usd == pytest.approx((100 * 1.0 + 10 * 4.0 + 1_000 * 1.0) / 1e6)
        assert priced.defaulted_rates == frozenset({'cached_input_per_1m'})

    def test_missing_cache_write_rate_prices_cache_writes_at_input_rate(self, entry_shape):
        entry = entry_shape({'input_per_1m': 1.0, 'output_per_1m': 4.0, 'cached_input_per_1m': 0.1})
        usage = TokenUsage(input_tokens=100, output_tokens=10, cache_create_tokens=2_000)

        priced = price_tokens(entry, usage)

        assert priced.cost_usd == pytest.approx((100 * 1.0 + 10 * 4.0 + 2_000 * 1.0) / 1e6)
        assert priced.defaulted_rates == frozenset({'cache_write_per_1m'})

    def test_missing_cache_rates_without_cache_tokens_default_nothing(self, entry_shape):
        entry = entry_shape({'input_per_1m': 1.0, 'output_per_1m': 4.0})
        usage = TokenUsage(input_tokens=100, output_tokens=10)

        priced = price_tokens(entry, usage)

        assert priced.cost_usd == (100 * 1.0 + 10 * 4.0) / 1_000_000
        assert priced.defaulted_rates == frozenset()

    def test_fallback_price_prices_cache_tokens_at_its_input_rate(self):
        usage = TokenUsage(
            input_tokens=100, output_tokens=10, cache_read_tokens=1_000, cache_create_tokens=500,
        )

        priced = price_tokens(FALLBACK_PRICE, usage)

        input_rate = FALLBACK_PRICE['input_per_1m']
        assert priced.cost_usd == pytest.approx(
            ((100 + 1_000 + 500) * input_rate + 10 * FALLBACK_PRICE['output_per_1m']) / 1e6,
        )


class TestPriceFields:
    def test_set_cache_rates_join_input_and_output(self, entry_shape):
        assert price_fields(entry_shape(GLM_RATES)) == GLM_RATES

    def test_unset_cache_rates_are_omitted(self, entry_shape):
        rates = {'input_per_1m': 1.0, 'output_per_1m': 4.0, 'cached_input_per_1m': 0.1}

        assert price_fields(entry_shape(rates)) == rates


class TestTokenUsageCountsAreNonNegative:
    @pytest.mark.parametrize('field', [
        'input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_create_tokens',
    ])
    def test_a_negative_count_is_rejected_naming_the_field_and_value(self, field):
        with pytest.raises(ValueError, match=rf'TokenUsage\.{field} must be non-negative, got -1'):
            TokenUsage(**{field: -1})


class TestValuesAreFrozen:
    def test_token_usage_is_frozen(self):
        pytest.raises(dataclasses.FrozenInstanceError, setattr, MEASURED_USAGE, 'input_tokens', 0)

    def test_token_cost_is_frozen(self):
        priced = price_tokens(GLM_RATES, MEASURED_USAGE)
        pytest.raises(dataclasses.FrozenInstanceError, setattr, priced, 'cost_usd', 0.0)
