"""How deep a merge verify chains.

Rationale: plans/merge-lane-policy-prd.md D6/D7; contract §6.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence

from orchestrator.config import DepthRow
from orchestrator.merge_verify_facts import ChainDepthBasis, ChainDepthReason

#: How many of the newest tip verdicts the pass estimate reads.
RECENT_WINDOW = 30


@dataclasses.dataclass(frozen=True)
class TipVerdict:
    """One tip verdict: a verify that covered *depth* items."""

    depth: int
    passed: bool

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError(f'a tip verdict covers at least one item; got depth {self.depth}')


@dataclasses.dataclass(frozen=True)
class ChainPolicyState:
    """The newest tip verdicts of the rounds the policy sized, oldest first."""

    recent: tuple[TipVerdict, ...] = ()

    def __post_init__(self) -> None:
        if len(self.recent) > RECENT_WINDOW:
            raise ValueError(
                f'the verdict window holds at most RECENT_WINDOW={RECENT_WINDOW} verdicts; '
                f'got {len(self.recent)}'
            )


def per_item_pass_estimate(recent: Sequence[TipVerdict]) -> float | None:
    """The chance one item passes: the window's pass rate spread over its mean depth."""
    if not recent:
        return None
    pass_rate = sum(verdict.passed for verdict in recent) / len(recent)
    mean_depth = sum(verdict.depth for verdict in recent) / len(recent)
    return pass_rate ** (1 / mean_depth)


def choose_depth(
    state: ChainPolicyState, queue_depth: int, depth_table: Sequence[DepthRow],
) -> tuple[int, ChainDepthReason]:
    """How many items the dispatching request's verify covers, itself first, and why.

    *queue_depth* counts the dispatching request and the chainable ones
    queued behind it.
    """
    estimate = per_item_pass_estimate(state.recent)
    if estimate is None:
        row: DepthRow | None = depth_table[0]
        basis = ChainDepthBasis.NO_ESTIMATE
    else:
        row = next((r for r in depth_table if estimate >= r.min_pass_estimate), None)
        basis = ChainDepthBasis.PASS_ESTIMATE
    reason = ChainDepthReason(
        basis=basis,
        pass_estimate=estimate,
        row=None if row is None else (row.min_pass_estimate, row.depth),
        queue_depth=queue_depth,
    )
    return (1 if row is None else min(row.depth, queue_depth)), reason


def on_tip_result(state: ChainPolicyState, depth: int, passed: bool) -> ChainPolicyState:
    """*state* with the verdict of one sized round over *depth* items in its window."""
    recent = (*state.recent, TipVerdict(depth, passed))[-RECENT_WINDOW:]
    return dataclasses.replace(state, recent=recent)


def seed_from_rows(rows: Iterable[Mapping[str, object]]) -> ChainPolicyState:
    """A state whose window holds the verdicts of *rows*, ``merge_verify`` row
    data oldest first, keeping only the last attempt per merge commit."""
    last_attempt: dict[object, tuple[int, int, Mapping[str, object]]] = {}
    for position, row in enumerate(rows):
        if not _is_window_sample(row):
            continue
        key = row.get('merge_sha') or ('unkeyed', position)
        held = last_attempt.get(key)
        if held is None or _attempt(row) >= held[0]:
            last_attempt[key] = (_attempt(row), position, row)
    kept = sorted(last_attempt.values(), key=lambda entry: entry[1])
    verdicts = tuple(TipVerdict(_chain_items(row), row['passed'] is True) for _, _, row in kept)
    return ChainPolicyState(recent=verdicts[-RECENT_WINDOW:])


def _is_window_sample(row: Mapping[str, object]) -> bool:
    """A verdict from a round the policy sized, or from a row written before
    the policy existed."""
    if not isinstance(row.get('passed'), bool):
        return False
    if 'chain_depth_reason' not in row:
        return True
    reason = row['chain_depth_reason']
    return isinstance(reason, Mapping) and _is_known_basis(reason.get('basis'))


def _is_known_basis(basis: object) -> bool:
    try:
        ChainDepthBasis(basis)
    except ValueError:
        return False
    return True


def _attempt(row: Mapping[str, object]) -> int:
    attempt = row.get('attempt')
    return attempt if isinstance(attempt, int) else 0


def _chain_items(row: Mapping[str, object]) -> int:
    items = row.get('chain_items')
    return items if isinstance(items, int) and items >= 1 else 1
