"""List prices for the OpenAI models fused-memory's callers price their calls by.

Same shape and None-for-unknown rule as
``fused-memory/scripts/score_write_triage_pairs.py::LIST_PRICES``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ListPrice:
    usd_per_mtok_input: float
    usd_per_mtok_output: float

    def usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.usd_per_mtok_input
            + output_tokens * self.usd_per_mtok_output
        ) / 1e6


LIST_PRICES_AS_OF = '2026-09-30'
LIST_PRICES: Mapping[str, ListPrice] = MappingProxyType({
    'gpt-6.1-sol': ListPrice(2.00, 10.00),
    'gpt-6-luna': ListPrice(0.10, 0.50),
})
"""USD per million tokens at list price, as of :data:`LIST_PRICES_AS_OF`.

Output tokens include reasoning tokens, which the Responses API bills as
output. No cached-input discount is applied, so a cost is an upper bound.
"""


def list_price(model: str) -> ListPrice | None:
    """The list price of *model*, or None when it is unpriced: never a guess."""
    return LIST_PRICES.get(model)
