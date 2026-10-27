"""What ONE Claude CLI process cost.

Since CLI 2.1.277, a ``--resume``'d process reports ``total_cost_usd`` as the
running total of its whole session, not as its own spend. Its ``usage`` block
still covers only that process, and the session transcript carries no cost at
all. So the previous running total has to be carried by whoever arms the
resume, and :func:`derive_process_cost` turns it into this process's cost:

- a fresh process costs what it reports;
- a resumed one costs its delta over the carried prior, when that delta is
  plausible against its own priced usage;
- otherwise it costs its priced usage, capped at what it reports.

:class:`ProcessCost` records which of those bases was used. The evidence for
the rates and the acceptance window is in
``shared/tests/fixtures/cumulative_cost/README.md``.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeGuard

__all__ = ['ProcessCost', 'ProcessCostBasis', 'derive_process_cost', 'next_resume_prior']

logger = logging.getLogger(__name__)


class ProcessCostBasis(enum.StrEnum):
    """How a :class:`ProcessCost` was obtained."""

    REPORTED = 'reported'
    SESSION_DELTA = 'session_delta'
    USAGE_ESTIMATE = 'usage_estimate'


@dataclass(frozen=True)
class ProcessCost:
    """One CLI process's cost, the basis it was derived on, and the CLI's own figure.

    ``delta_clamped`` is True when a prior total was supplied and the delta
    over it was rejected.
    """

    cost_usd: float
    basis: ProcessCostBasis
    reported_total_usd: float | None
    delta_clamped: bool = False


@dataclass(frozen=True)
class _ModelRates:
    """List prices in USD per 1M tokens."""

    input: float
    output: float
    cache_read: float
    cache_write_5m: float
    cache_write_1h: float


# Fitted exactly from full CLI stdout; 5m writes use the published 1.25 x input (unmeasured).
_LIST_RATES: Mapping[str, _ModelRates] = MappingProxyType({
    'claude-opus-5': _ModelRates(5, 25, 0.5, 6.25, 10),
    'claude-opus-5-5': _ModelRates(4, 20, 0.2, 5, 8),
    'claude-sonnet-5': _ModelRates(2, 10, 0.2, 2.5, 4),
    'claude-haiku-4-5': _ModelRates(1, 5, 0.1, 1.25, 2),
    'claude-fable-5-1': _ModelRates(10, 50, 0.25, 12.5, 20),
})

_TOKENS_PER_RATE_UNIT = 1_000_000
_DELTA_FLOOR_FRACTION = 0.97


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _reported_total(cli_result: Mapping[str, Any]) -> float | None:
    key = 'cost_usd' if 'cost_usd' in cli_result else 'total_cost_usd'
    raw = cli_result.get(key)
    return float(raw) if _is_number(raw) else None


def _entry_model(key: str, entry: Mapping[str, Any]) -> str:
    canonical = entry.get('canonicalModel')
    return canonical if isinstance(canonical, str) and canonical else key


def _pricing_model(cli_result: Mapping[str, Any], served_model_id: str | None) -> str | None:
    if served_model_id in _LIST_RATES:
        return served_model_id
    model_usage = cli_result.get('modelUsage')
    if not isinstance(model_usage, Mapping):
        return None
    costed = [
        (entry['costUSD'], _entry_model(key, entry))
        for key, entry in model_usage.items()
        if isinstance(entry, Mapping) and _is_number(entry.get('costUSD'))
    ]
    if not costed:
        return None
    return max(costed, key=lambda pair: pair[0])[1]


def _token_count(counts: Mapping[str, Any], key: str) -> int:
    value = counts.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _price_usage(usage: Mapping[str, Any], rates: _ModelRates) -> float:
    split = usage.get('cache_creation')
    write_1h = _token_count(split, 'ephemeral_1h_input_tokens') if isinstance(split, Mapping) else 0
    write_5m = max(_token_count(usage, 'cache_creation_input_tokens') - write_1h, 0)
    priced = (
        _token_count(usage, 'input_tokens') * rates.input
        + _token_count(usage, 'output_tokens') * rates.output
        + _token_count(usage, 'cache_read_input_tokens') * rates.cache_read
        + write_5m * rates.cache_write_5m
        + write_1h * rates.cache_write_1h
    )
    return priced / _TOKENS_PER_RATE_UNIT


def _usage_estimate(
    cli_result: Mapping[str, Any], served_model_id: str | None,
) -> tuple[float | None, str | None]:
    model = _pricing_model(cli_result, served_model_id)
    rates = _LIST_RATES.get(model) if model is not None else None
    usage = cli_result.get('usage')
    if rates is None or not isinstance(usage, Mapping):
        return None, model
    return _price_usage(usage, rates), model


def _accept_delta(
    reported: float | None, prior: float | None, estimate: float | None,
) -> float | None:
    if reported is None or prior is None:
        return None
    delta = reported - prior
    floor = estimate * _DELTA_FLOOR_FRACTION if estimate is not None else 0.0
    return delta if floor <= delta <= reported else None


def _fallback(
    reported: float | None, estimate: float | None, *, delta_clamped: bool,
) -> ProcessCost:
    if estimate is not None:
        cost = estimate if reported is None else min(estimate, reported)
        return ProcessCost(cost, ProcessCostBasis.USAGE_ESTIMATE, reported, delta_clamped)
    return ProcessCost(reported or 0.0, ProcessCostBasis.REPORTED, reported, delta_clamped)


def _warn_fallback(
    cli_result: Mapping[str, Any],
    cost: ProcessCost,
    *,
    prior: float | None,
    estimate: float | None,
    model: str | None,
) -> None:
    reported = cost.reported_total_usd
    reason = (
        'rejected the session delta'
        if cost.delta_clamped
        else 'had no priced usage, so the cost may include earlier processes'
    )
    logger.warning(
        'Resumed Claude CLI process %s and fell back to %s=%r: session=%r '
        'reported_total=%r prior_total=%r delta=%r usage_estimate=%r model=%r',
        reason,
        cost.basis.value,
        cost.cost_usd,
        cli_result.get('session_id'),
        reported,
        prior,
        reported - prior if reported is not None and prior is not None else None,
        estimate,
        model,
    )


def derive_process_cost(
    cli_result: Mapping[str, Any],
    *,
    resumed: bool,
    prior_total_usd: float | None,
    served_model_id: str | None = None,
) -> ProcessCost:
    """The cost of the CLI process whose result JSON is *cli_result*.

    *prior_total_usd* is the reported total of the result this process
    resumed; a non-resumed process ignores it. *served_model_id* is the model
    the CLI actually served, which prices the usage block when it is a known
    model. Never raises on a malformed result.
    """
    reported = _reported_total(cli_result)
    if not resumed:
        return ProcessCost(reported or 0.0, ProcessCostBasis.REPORTED, reported)
    estimate, model = _usage_estimate(cli_result, served_model_id)
    delta = _accept_delta(reported, prior_total_usd, estimate)
    if delta is not None:
        return ProcessCost(delta, ProcessCostBasis.SESSION_DELTA, reported)
    cost = _fallback(
        reported, estimate,
        delta_clamped=prior_total_usd is not None and reported is not None,
    )
    if cost.delta_clamped or estimate is None:
        _warn_fallback(cli_result, cost, prior=prior_total_usd, estimate=estimate, model=model)
    return cost


def next_resume_prior(
    *,
    armed_session_id: str | None,
    armed_prior_usd: float | None,
    result_session_id: str,
    result_reported_total_usd: float | None,
) -> float | None:
    """The prior total to pass when next resuming *result_session_id*.

    A running total never decreases within one session, so the same session
    keeps the larger figure; a different session never inherits a prior.
    """
    if armed_session_id != result_session_id or armed_prior_usd is None:
        return result_reported_total_usd
    if result_reported_total_usd is None:
        return armed_prior_usd
    return max(armed_prior_usd, result_reported_total_usd)
