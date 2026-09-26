#!/usr/bin/env python3
"""Review an admitted model's per-role performance against a baseline model.

Compares a candidate model's per-role outcomes, merge-integrity events, spend
and role containment against a baseline model, over FIXED windows around an
admission: [apply, apply + days) for both models, and
[apply - baseline_days, apply) for the baseline. Written for the D6
claude-fable-5-1 day-14 review (task 5441), parameterized by model, roles and
windows like the audit it builds on.

Layering: this module composes scripts/audit_model_admission.py's scans with
scripts/escalation_ladder.py; neither imports it back.

STRICTLY READ-ONLY, like the audit. It also asserts no threshold: it measures
and renders. The verdict belongs to the reader and to the milestone task.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from audit_model_admission import InvocationRecord

# The CLI's result subtypes for a run killed at its dispatch caps. The _usd
# spelling is orchestrator/src/orchestrator/routing.py::PROBE_BUDGET_EXHAUSTED_SUBTYPE.
TURN_CAP_SUBTYPE = 'error_max_turns'
BUDGET_CAP_SUBTYPE = 'error_max_budget_usd'

MERGE_DONE_STATE = 'done'
UNKNOWN_KEY = '-'

_Number = TypeVar('_Number', int, float)
_Key = TypeVar('_Key', bound=Hashable)


def nearest_rank(values: Iterable[_Number], pct: float) -> _Number | None:
    """The *pct*-th percentile by nearest rank: the ceil(pct/100 * n)-th smallest.

    Nearest rank always returns an OBSERVED value, so a reader can reproduce it
    by hand from the per-run rows. None for no values.
    """
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def _tally(keys: Iterable[_Key | None]) -> tuple[tuple[_Key | str, int], ...]:
    """Count *keys*, sorted, with None counted last under :data:`UNKNOWN_KEY`."""
    counts = Counter(keys)
    known = sorted(key for key in counts if key is not None)
    ordered: list[_Key | None] = [*known, *([None] if None in counts else [])]
    return tuple((UNKNOWN_KEY if key is None else key, counts[key]) for key in ordered)


@dataclass(frozen=True)
class OutcomeSummary:
    """How one arm's runs ended. Counts are runs; percentiles are nearest-rank.

    ``merge_states`` tallies each run's attributed merge outcome and
    ``dispatch_caps`` each run's dispatch max_turns, '-' for unknown.
    """

    runs: int
    succeeded: int
    turn_cap_kills: int
    budget_kills: int
    timed_out: int
    no_end_event: int
    over_flat_ceiling: int
    cost_total_usd: float
    cost_usd_p50: float | None
    cost_usd_p95: float | None
    turns_p50: int | None
    turns_p95: int | None
    duration_ms_p50: int | None
    duration_ms_p95: int | None
    duration_ms_max: int | None
    merge_states: tuple[tuple[str, int], ...]
    dispatch_caps: tuple[tuple[int | str, int], ...]

    @property
    def resolved(self) -> int:
        """Runs whose attributed merge finished done: a view of ``merge_states``."""
        return dict(self.merge_states).get(MERGE_DONE_STATE, 0)


def summarize_outcomes(records: Sequence[InvocationRecord]) -> OutcomeSummary:
    """Summarize *records*. A run with unknown turns is left out of the turns
    percentiles, not counted as zero turns."""
    costs = [r.cost_usd for r in records]
    durations = [r.duration_ms for r in records]
    turns = [r.turns for r in records if r.turns is not None]
    return OutcomeSummary(
        runs=len(records),
        succeeded=sum(1 for r in records if r.succeeded is True),
        turn_cap_kills=sum(1 for r in records if r.subtype == TURN_CAP_SUBTYPE),
        budget_kills=sum(1 for r in records if r.subtype == BUDGET_CAP_SUBTYPE),
        timed_out=sum(1 for r in records if r.timed_out is True),
        no_end_event=sum(1 for r in records if r.succeeded is None and r.subtype is None),
        over_flat_ceiling=sum(1 for r in records if r.at_or_over_flat_role_ceiling is True),
        cost_total_usd=sum(costs, 0.0),
        cost_usd_p50=nearest_rank(costs, 50),
        cost_usd_p95=nearest_rank(costs, 95),
        turns_p50=nearest_rank(turns, 50),
        turns_p95=nearest_rank(turns, 95),
        duration_ms_p50=nearest_rank(durations, 50),
        duration_ms_p95=nearest_rank(durations, 95),
        duration_ms_max=max(durations, default=None),
        merge_states=_tally(r.merge_outcome.state if r.merge_outcome else None
                            for r in records),
        dispatch_caps=_tally(r.dispatch_max_turns for r in records),
    )
