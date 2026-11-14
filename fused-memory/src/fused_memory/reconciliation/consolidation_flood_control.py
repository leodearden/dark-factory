"""The per-cycle flood-control policy for consolidation candidates.

This module is the single home of that policy (``plans/memory-auto-consolidation-prd.md``
D6). Two callers share it: the auto-consolidation executor
(``reconciliation/consolidation_auto_executor.py``), which passes
``cap = max_auto_per_cycle + max_gate_filings_per_cycle``, and task 4916's
detector-report submitter, which passes its own cap. Both read their numbers
from the one ``consolidation_auto`` config block. Like
``reconciliation/consolidation_auto.py`` it imports only the standard library;
the config type is referenced under ``TYPE_CHECKING`` only.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from fused_memory.config.schema import ConsolidationAutoConfig

K = TypeVar('K', bound=Hashable)


@dataclass(frozen=True)
class FloodCandidate(Generic[K]):
    key: K
    size: int
    category: str
    canonical_absent: bool


@dataclass(frozen=True)
class FloodPlan(Generic[K]):
    """The ranked partition of one cycle's candidates.

    ``selected`` is the evaluation slice in rank order; ``deferred`` holds the
    ranked remainder (every ranked candidate when the backlog tripped);
    ``excluded`` holds candidates whose category carries no weight. Nothing is
    dropped: the three tuples together are exactly the input.
    """

    selected: tuple[FloodCandidate[K], ...]
    deferred: tuple[FloodCandidate[K], ...]
    excluded: tuple[FloodCandidate[K], ...]
    backlog_tripped: bool
    cap: int
    backlog_threshold: int


def _score(candidate: FloodCandidate, config: ConsolidationAutoConfig) -> float:
    bonus = config.canonical_absent_bonus if candidate.canonical_absent else 1.0
    return candidate.size * config.category_weights[candidate.category] * bonus


def rank_and_cap(
    candidates: Iterable[FloodCandidate[K]],
    *,
    cap: int,
    config: ConsolidationAutoConfig,
) -> FloodPlan[K]:
    """Rank candidates by size x category weight x canonical-absent bonus.

    The sort is stable, so equal scores keep the caller's input order. The
    backlog trips when the candidate count, excluded categories included,
    exceeds ``cap x config.backlog_multiplier``; a tripped plan selects nothing.
    Every config leaf is read per call.
    """
    if cap < 0:
        raise ValueError(f'cap must be >= 0, got {cap}')
    pool = tuple(candidates)
    for candidate in pool:
        if candidate.size < 0:
            raise ValueError(
                f'candidate {candidate.key!r} size must be >= 0, got {candidate.size}',
            )
    weights = config.category_weights
    excluded = tuple(c for c in pool if c.category not in weights)
    ranked = tuple(
        sorted(
            (c for c in pool if c.category in weights),
            key=lambda c: _score(c, config),
            reverse=True,
        ),
    )
    backlog_threshold = cap * config.backlog_multiplier
    backlog_tripped = len(pool) > backlog_threshold
    if backlog_tripped:
        selected, deferred = (), ranked
    else:
        selected, deferred = ranked[:cap], ranked[cap:]
    return FloodPlan(
        selected=selected,
        deferred=deferred,
        excluded=excluded,
        backlog_tripped=backlog_tripped,
        cap=cap,
        backlog_threshold=backlog_threshold,
    )
