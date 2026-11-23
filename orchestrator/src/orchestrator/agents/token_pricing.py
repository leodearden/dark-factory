"""The single home of the per-token USD formula, shared by runtime estimation and eval pricing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Fallback USD/1M-token rate used ONLY when a model has no configured price.
# It is a DEFINED, logged rate: every caller that reaches for it warns.
FALLBACK_PRICE: dict[str, float] = {'input_per_1m': 2.0, 'output_per_1m': 8.0}

_PER_MILLION = 1_000_000


def rate(entry: Any, key: str) -> float:
    """Read *key* off a price *entry*, which may be a PriceEntry (attribute)
    or a plain dict (item) — so both ``config.prices`` values and plain
    dicts (e.g. FALLBACK_PRICE, test fixtures) work through the same
    accessor. Uses hasattr (not a getattr default) so a legitimate 0.0 rate
    is read correctly rather than masked by a fallback default.
    """
    return getattr(entry, key) if hasattr(entry, key) else entry[key]


def optional_rate(entry: Any, key: str) -> float | None:
    """Read an optional *key* off a price *entry* as rate does, answering
    None when it is absent or unset rather than raising."""
    return getattr(entry, key) if hasattr(entry, key) else entry.get(key)


@dataclass(frozen=True)
class TokenUsage:
    """One invocation's token counts, in AgentResult's convention:
    *input_tokens* EXCLUDES both cache counts."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_create_tokens: int = 0


@dataclass(frozen=True)
class TokenCost:
    """A priced usage, naming every cache rate key that had to default to
    input_per_1m because the entry carries none and tokens needed it."""

    cost_usd: float
    defaulted_rates: frozenset[str]


def price_tokens(entry: Any, usage: TokenUsage) -> TokenCost:
    """Price *usage* at *entry*'s per-1M rates. A missing cache rate prices
    its tokens at input_per_1m, an upper bound, and is reported as defaulted."""
    input_rate = rate(entry, 'input_per_1m')
    defaulted: set[str] = set()

    def cache_rate(key: str, tokens: int) -> float:
        own = optional_rate(entry, key)
        if own is not None:
            return own
        if tokens:
            defaulted.add(key)
        return input_rate

    cached_rate = cache_rate('cached_input_per_1m', usage.cache_read_tokens)
    write_rate = cache_rate('cache_write_per_1m', usage.cache_create_tokens)
    cost = (
        usage.input_tokens * input_rate
        + usage.cache_read_tokens * cached_rate
        + usage.cache_create_tokens * write_rate
        + usage.output_tokens * rate(entry, 'output_per_1m')
    ) / _PER_MILLION
    return TokenCost(cost_usd=cost, defaulted_rates=frozenset(defaulted))
