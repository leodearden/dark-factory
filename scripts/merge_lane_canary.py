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

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from itertools import pairwise
from typing import Any

Event = Mapping[str, Any]


def _data(event: Event) -> Mapping[str, Any]:
    data = event.get('data')
    return data if isinstance(data, Mapping) else {}


def _instant(raw: object) -> datetime | None:
    """An events-table timestamp as tz-aware UTC, or None when absent or unparseable."""
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _at(event: Event) -> datetime | None:
    return _instant(event.get('timestamp'))


def _is_number(raw: object) -> bool:
    return isinstance(raw, (int, float)) and not isinstance(raw, bool)


def _str_or_none(raw: object) -> str | None:
    return raw if isinstance(raw, str) and raw else None


def _times_by_task(
    events: Sequence[Event], keep: Callable[[Event], bool] = lambda _e: True,
) -> dict[str, list[datetime]]:
    by_task: dict[str, list[datetime]] = defaultdict(list)
    for event in events:
        moment = _at(event)
        if moment is not None and keep(event):
            by_task[str(event.get('task_id'))].append(moment)
    return by_task


def _share(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


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


# ---------------------------------------------------------------------------
# The landing ledger and the gate/void ratios. Counting rules are the
# merge-throughput report's (plans/portfolio-review-2026-11-01/followups/
# merge-throughput-seats/B-ledger.md, population rules and P5).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Landing:
    task_id: str
    at: datetime
    merge_sha: str | None
    via_chain: bool


def _finalize_landings(finalized: Sequence[Event]) -> list[Landing]:
    rows = []
    for event in finalized:
        data, moment = _data(event), _at(event)
        if data.get('state') != 'done' or moment is None:
            continue
        rows.append(Landing(
            task_id=str(event.get('task_id')), at=moment,
            merge_sha=_str_or_none(data.get('merge_sha')),
            via_chain=_is_number(data.get('landed_via_chain')),
        ))
    return rows


def _train_landings(train_merged: Sequence[Event]) -> list[Landing]:
    rows = []
    for event in train_merged:
        data, moment = _data(event), _at(event)
        members = data.get('member_task_ids')
        if moment is None or not isinstance(members, list):
            continue
        sha = _str_or_none(data.get('merge_commit_sha'))
        rows.extend(Landing(str(m), moment, sha, via_chain=False) for m in members)
    return rows


def landings(finalized: Sequence[Event], train_merged: Sequence[Event]) -> tuple[Landing, ...]:
    """Landings = ``merge_finalized`` state done ∪ ``train_merged`` member_task_ids, by time.

    A train member that also has a done finalize row is counted once, as the finalize.
    ``via_chain`` is True iff ``landed_via_chain`` is a non-bool number.
    """
    done = _finalize_landings(finalized)
    finalized_tasks = {row.task_id for row in done}
    members = [row for row in _train_landings(train_merged) if row.task_id not in finalized_tasks]
    return tuple(sorted([*done, *members], key=lambda row: row.at))


def _reverifies(event: Event) -> bool:
    return _data(event).get('reverify') is True


def gates_per_landing(gate_rows: Sequence[Event], ledger: Sequence[Landing]) -> float | None:
    """``merge_reverify_gate`` rows with reverify true, per landing."""
    return _share(sum(1 for e in gate_rows if _reverifies(e)), len(ledger))


@dataclass(frozen=True)
class VoidTally:
    total: int
    before_host: int
    at_adoption: int
    per_landing: float | None


def voids_per_landing(voided: Sequence[Event], ledger: Sequence[Landing]) -> VoidTally:
    """``verdict_voided`` rows per landing; point dispatch voids were discarded before a host."""
    points = [_data(e).get('point') for e in voided]
    return VoidTally(
        total=len(points),
        before_host=points.count('dispatch'),
        at_adoption=points.count('adoption'),
        per_landing=_share(len(points), len(ledger)),
    )


@dataclass(frozen=True)
class GateConditionals:
    gated: tuple[bool, ...]
    p_gate_after_gate: float | None
    p_gate_after_ungated: float | None


def _is_gated(
    landing: Landing, gates: Mapping[str, list[datetime]], queued: Mapping[str, list[datetime]],
) -> bool:
    submitted = max((t for t in queued.get(landing.task_id, ()) if t < landing.at), default=None)
    return any(
        (submitted is None or t > submitted) and t <= landing.at
        for t in gates.get(landing.task_id, ())
    )


def _conditional(gated: Sequence[bool], previous: bool) -> float | None:
    successors = [after for before, after in pairwise(gated) if before is previous]
    return _share(sum(successors), len(successors))


def gate_conditionals(
    ledger: Sequence[Landing], gate_rows: Sequence[Event], queued: Sequence[Event],
) -> GateConditionals:
    """P(gate | previous landing gated) and P(gate | previous not gated), report P5.

    A landing is gated iff a reverify=true gate row for its task is stamped
    after that task's last ``merge_queued`` before the landing, and at or
    before the landing: a gate from an earlier submission does not count.
    """
    gates = _times_by_task(gate_rows, _reverifies)
    submissions = _times_by_task(queued)
    gated = tuple(_is_gated(landing, gates, submissions) for landing in ledger)
    return GateConditionals(
        gated=gated,
        p_gate_after_gate=_conditional(gated, previous=True),
        p_gate_after_ungated=_conditional(gated, previous=False),
    )
