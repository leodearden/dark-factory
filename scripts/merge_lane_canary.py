"""Pure figure and predicate layer of the task-8532 merge-lane canary readout.

`plans/merge-lane-policy-prd.md` D12 names the predicates the lane must meet a
week after its policy leaves deploy. This module computes every figure behind
them, and the predicates themselves, from values the readout CLI
(`scripts/merge_lane_canary_readout.py`) has already read.

STRICTLY PURE: no database, no git, no journal, no clock. Event inputs are
dicts in `merge_lane_throughput.load_events`' shape,
``{'timestamp': <iso +00:00>, 'task_id': ..., 'data': {...}}``; everything
read from git or the journal arrives as one of the small frozen records below.

An unmeasured predicate never holds: a canary must not pass on missing signal,
the same rule as `scripts/merge_deep_canary_predicate.py`'s "a leg with no
value FAILs".
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

Event = Mapping[str, Any]


def _data(event: Event) -> Mapping[str, Any]:
    data = event.get('data')
    return data if isinstance(data, Mapping) else {}


# ---------------------------------------------------------------------------
# Predicates.
# ---------------------------------------------------------------------------


class Comparison(StrEnum):
    LT = 'lt'
    LE = 'le'
    EQ0 = 'eq0'


@dataclass(frozen=True)
class Predicate:
    """One D12 row: a measured value held against its limit."""

    key: str
    label: str
    value: float | None
    limit: float
    comparison: Comparison
    rule: str
    baseline: str
    chain_traced: bool

    @property
    def holds(self) -> bool:
        if self.value is None:
            return False
        match self.comparison:
            case Comparison.LT:
                return self.value < self.limit
            case Comparison.LE:
                return self.value <= self.limit
            case Comparison.EQ0:
                return self.value == 0

    @property
    def rendered_value(self) -> str:
        return 'not measured' if self.value is None else f'{self.value:.4g}'


# ---------------------------------------------------------------------------
# The alpha refusal: a window whose rows predate the D10 instrumentation.
# ---------------------------------------------------------------------------


def _carried(events: Sequence[Event], key: str) -> bool:
    return any(_data(e).get(key) is not None for e in events)


def missing_alpha_fields(
    verify: Sequence[Event],
    cancelled: Sequence[Event],
    voided: Sequence[Event],
    finalized: Sequence[Event],
) -> tuple[str, ...]:
    """Name each alpha field no row of the window carries, in a fixed order.

    A field counts as present when at least one row carries it non-null;
    ``merge_verify_cancelled`` is present when the window holds any such row.
    """
    done = [e for e in finalized if _data(e).get('state') == 'done']
    presence = (
        ('merge_verify.runner_reason', _carried(verify, 'runner_reason')),
        ('merge_verify.base_relation', _carried(verify, 'base_relation')),
        ('merge_verify_cancelled', bool(cancelled)),
        ('verdict_voided.item_commit', _carried(voided, 'item_commit')),
        ('verdict_voided.killer', _carried(voided, 'killer')),
        ('merge_finalized.queue_to_land_secs', _carried(done, 'queue_to_land_secs')),
    )
    return tuple(name for name, present in presence if not present)
