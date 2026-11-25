"""Decide whether a verify host lifted fleet landings: on-blocks against off-blocks, with a block bootstrap.

Pure and stdlib only. Its input is one BlockLandings per trial block. Every case
in which no comparison can be made yields a reason and no number.
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ['DECISION_RULE', 'MIN_BLOCKS_FOR_INTERVAL', 'BlockLandings', 'Decision', 'Interval', 'decide']

MIN_BLOCKS_FOR_INTERVAL = 2
_PERCENTILES = (2.5, 97.5)

DECISION_RULE = (
    'decision: fleet landings per day (sum of landings / sum of days) over the on-blocks where a '
    'host verify ran, against every off-block; effect = on rate / off rate - 1; interval = '
    'nearest-rank 2.5th and 97.5th percentiles of a block bootstrap that resamples whole blocks '
    'with replacement within each condition (--resamples draws, fixed --seed), needing at least '
    f'{MIN_BLOCKS_FOR_INTERVAL} blocks per condition; host cost share = --host-cost-usd / '
    '--seat-bill-usd, the seat bill'
)


@dataclass(frozen=True)
class BlockLandings:
    """One trial block's fleet landings, its length, and how many host verifies started in it."""

    on: bool
    landings: int
    days: float
    host_verifies: int


@dataclass(frozen=True)
class Interval:
    low: float
    high: float
    undefined_resamples: int
    """Resamples whose off-blocks landed nothing, so their ratio is undefined; left out of the percentiles."""


@dataclass(frozen=True)
class Decision:
    on_rate: float | None
    off_rate: float | None
    on_blocks: int
    off_blocks: int
    effect: float | None
    interval: Interval | None
    reason: str | None
    excluded_on_blocks: tuple[int, ...]
    """Positions in the input of the on-blocks where no host verify ran."""
    host_cost_share: float


def _rate(group: Sequence[BlockLandings]) -> float | None:
    days = sum(block.days for block in group)
    return sum(block.landings for block in group) / days if group and days > 0 else None


def _nearest_rank(ordered: Sequence[float], percentile: float) -> float:
    rank = max(1, math.ceil(percentile / 100 * len(ordered)))
    return ordered[rank - 1]


def _bootstrap(
    on_group: Sequence[BlockLandings], off_group: Sequence[BlockLandings], resamples: int, seed: int
) -> Interval | None:
    rng = random.Random(seed)
    effects: list[float] = []
    for _ in range(resamples):
        on_rate = _rate(rng.choices(on_group, k=len(on_group)))
        off_rate = _rate(rng.choices(off_group, k=len(off_group)))
        if on_rate is not None and off_rate:
            effects.append(on_rate / off_rate - 1)
    if not effects:
        return None
    effects.sort()
    low, high = (_nearest_rank(effects, p) for p in _PERCENTILES)
    return Interval(low, high, resamples - len(effects))


def _no_effect_reason(on_rate: float | None, off_rate: float | None) -> str | None:
    reasons = []
    if on_rate is None:
        reasons.append('no verify ran on this host in any on-block')
    if off_rate is None:
        reasons.append('no off-block is available')
    elif off_rate == 0:
        reasons.append('the off-blocks landed nothing, so the off rate is 0')
    return f'no comparison: {"; ".join(reasons)}' if reasons else None


def _plural(count: int, noun: str) -> str:
    return f'{count} {noun}' if count == 1 else f'{count} {noun}s'


def decide(
    blocks: Sequence[BlockLandings],
    *,
    host_cost_usd: float,
    seat_bill_usd: float,
    resamples: int,
    seed: int,
) -> Decision:
    """The on-against-off decision over *blocks*; any effect or interval that cannot be computed is None, with a reason."""
    if seat_bill_usd <= 0:
        raise ValueError(f'the seat bill must be positive, got {seat_bill_usd}')
    if host_cost_usd < 0:
        raise ValueError(f'the host cost must not be negative, got {host_cost_usd}')
    if resamples <= 0:
        raise ValueError(f'resamples must be positive, got {resamples}')
    excluded = tuple(i for i, block in enumerate(blocks) if block.on and block.host_verifies == 0)
    on_group = [block for block in blocks if block.on and block.host_verifies > 0]
    off_group = [block for block in blocks if not block.on]
    on_rate, off_rate = _rate(on_group), _rate(off_group)
    reason = _no_effect_reason(on_rate, off_rate)
    effect = interval = None
    if reason is None and on_rate is not None and off_rate:
        effect = on_rate / off_rate - 1
        if min(len(on_group), len(off_group)) < MIN_BLOCKS_FOR_INTERVAL:
            reason = (
                f'no interval: a block bootstrap needs at least {MIN_BLOCKS_FOR_INTERVAL} blocks per '
                f'condition; there are {_plural(len(on_group), "on-block")} and '
                f'{_plural(len(off_group), "off-block")}'
            )
        else:
            interval = _bootstrap(on_group, off_group, resamples, seed)
            if interval is None:
                reason = 'no interval: every bootstrap resample drew off-blocks that landed nothing'
    return Decision(
        on_rate=on_rate,
        off_rate=off_rate,
        on_blocks=len(on_group),
        off_blocks=len(off_group),
        effect=effect,
        interval=interval,
        reason=reason,
        excluded_on_blocks=excluded,
        host_cost_share=host_cost_usd / seat_bill_usd,
    )
