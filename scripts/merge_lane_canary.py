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

import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from itertools import pairwise
from typing import Any

import merge_lane_throughput as mlt
from merge_deep_canary_predicate import pct

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


def _number(raw: object) -> float | None:
    """*raw* as a float, or None for a missing, bool or non-numeric value."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw)


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
            via_chain=_number(data.get('landed_via_chain')) is not None,
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


# ---------------------------------------------------------------------------
# Time-based figures. Rules: the merge-throughput report's C-laptop and
# B-ledger §4b (heartbeat weighting) and D-regimes §3.2 (union-busy).
# ---------------------------------------------------------------------------

HEARTBEAT_GAP_CAP = timedelta(seconds=600)
HOURS_PER_WEEK = 168.0

Interval = tuple[datetime, datetime]


def _weighted_heartbeats(heartbeats: Sequence[Event], hi: datetime) -> list[tuple[float, Event]]:
    """Each heartbeat with the seconds to the next one (the last: to *hi*), capped at 600 s."""
    timed = sorted(
        ((moment, e) for e in heartbeats if (moment := _at(e)) is not None),
        key=lambda pair: pair[0],
    )
    if not timed:
        return []
    following = [moment for moment, _ in timed[1:]] + [hi]
    return [
        (max(0.0, min(nxt - moment, HEARTBEAT_GAP_CAP).total_seconds()), event)
        for (moment, event), nxt in zip(timed, following, strict=True)
    ]


def _holds_finished_lease(heartbeat: Event, host: str) -> bool:
    occupancy = _data(heartbeat).get('occupancy')
    leases = occupancy.get('leases') if isinstance(occupancy, Mapping) else None
    return isinstance(leases, list) and any(
        isinstance(lease, Mapping) and lease.get('host') == host and lease.get('activity') == 'finished'
        for lease in leases
    )


def host_finished_share(heartbeats: Sequence[Event], hi: datetime, host: str) -> float | None:
    """Heartbeat-weighted share of time *host* held a lease whose verify had finished."""
    weighted = _weighted_heartbeats(heartbeats, hi)
    finished = sum(weight for weight, e in weighted if _holds_finished_lease(e, host))
    return _share(finished, sum(weight for weight, _ in weighted))


def _interval(event: Event, duration_key: str) -> Interval | None:
    """``[timestamp - data[duration_key] ms, timestamp]``: a verify event is stamped at its end."""
    end, millis = _at(event), _number(_data(event).get(duration_key))
    if end is None or millis is None:
        return None
    return (end - timedelta(milliseconds=millis), end)


def _intervals(events: Sequence[Event], duration_key: str) -> list[Interval]:
    return [span for e in events if (span := _interval(e, duration_key)) is not None]


def _union_hours(intervals: Iterable[Interval], lo: datetime, hi: datetime) -> float:
    """Hours of ``[lo, hi)`` covered by at least one interval; overlaps count once."""
    covered = timedelta(0)
    reach = lo
    for start, end in sorted(intervals):
        start, end = max(start, reach), min(end, hi)
        if end > start:
            covered += end - start
            reach = end
    return covered.total_seconds() / 3600.0


@dataclass(frozen=True)
class HostHours:
    hours: float
    per_week: float


def host_verify_hours(
    verify: Sequence[Event], cancelled: Sequence[Event], lo: datetime, hi: datetime, host: str,
) -> HostHours:
    """Union of *host*'s completed (``runner``) and cancelled (``host``) merge-verify time in ``[lo, hi)``."""
    spans = [
        *_intervals([e for e in verify if _data(e).get('runner') == host], 'duration_ms'),
        *_intervals([e for e in cancelled if _data(e).get('host') == host], 'elapsed_ms'),
    ]
    hours = _union_hours(spans, lo, hi)
    window_hours = (hi - lo).total_seconds() / 3600.0
    return HostHours(hours=hours, per_week=hours * HOURS_PER_WEEK / window_hours)


def union_busy_hours(verify: Sequence[Event], lo: datetime, hi: datetime) -> float:
    """Hours in ``[lo, hi)`` during which any host ran a completed merge verify."""
    return _union_hours(_intervals(verify, 'duration_ms'), lo, hi)


@dataclass(frozen=True)
class QueueToLand:
    p50_hours: float | None
    n: int
    n_missing: int


def queue_to_land(finalized: Sequence[Event]) -> QueueToLand:
    """Median ``queue_to_land_secs`` of done landings, per merge_lane_throughput's policy signals."""
    series = mlt.compute_policy_signals([], [], [], [dict(e) for e in finalized])['queue_to_land']
    p50 = series['p50']
    return QueueToLand(
        p50_hours=None if p50 is None else p50 / 3600.0, n=series['n'], n_missing=series['n_missing'],
    )


# ---------------------------------------------------------------------------
# Chain figures over merge_verify payloads: chain_items, chain_task_ids and
# chain_depth_reason {basis, pass_estimate, row: [threshold, depth], queue_depth}.
# ---------------------------------------------------------------------------

DEEP_CHAIN_MIN_ITEMS = 2
UNSIZED = 'unsized'


def _chain_items(event: Event) -> int | None:
    raw = _data(event).get('chain_items')
    return raw if isinstance(raw, int) and not isinstance(raw, bool) else None


def _duration_secs(event: Event) -> float | None:
    millis = _number(_data(event).get('duration_ms'))
    return None if millis is None else millis / 1000.0


def deep_chain_p90_secs(verify: Sequence[Event]) -> float | None:
    """Nearest-rank p90 of deep verifies' (``chain_items`` >= 2) duration, the 3190 script's rule."""
    secs = [
        duration for e in verify
        if (_chain_items(e) or 0) >= DEEP_CHAIN_MIN_ITEMS and (duration := _duration_secs(e)) is not None
    ]
    return pct(secs, 0.9)


def chain_items_distribution(verify: Sequence[Event]) -> dict[int, int]:
    """``chain_items`` -> number of merge verifies, per merge_lane_throughput.compute_chains."""
    return mlt.compute_chains([], [dict(e) for e in verify])['chain_items']['distribution']


def _depth_reason(event: Event) -> Mapping[str, Any] | None:
    reason = _data(event).get('chain_depth_reason')
    return reason if isinstance(reason, Mapping) else None


def depth_reason_tally(verify: Sequence[Event]) -> dict[str, int]:
    """Merge verifies by ``chain_depth_reason.basis``; a null reason is :data:`UNSIZED`."""
    return dict(Counter(
        UNSIZED if (reason := _depth_reason(e)) is None else str(reason.get('basis'))
        for e in verify
    ))


class Starvation(StrEnum):
    FULL = 'full'
    TOO_FEW_WAITING = 'too_few_waiting'
    OTHER = 'other'


def _policy_depth(reason: Mapping[str, Any]) -> int | None:
    row = reason.get('row')
    depth = row[1] if isinstance(row, list) and len(row) == 2 else None
    return depth if isinstance(depth, int) and not isinstance(depth, bool) else None


def _starvation(event: Event) -> Starvation | None:
    """How a policy-sized verify's chain compares with the depth its policy row chose."""
    reason, items = _depth_reason(event), _chain_items(event)
    depth = None if reason is None else _policy_depth(reason)
    if reason is None or depth is None or items is None:
        return None
    if items >= depth:
        return Starvation.FULL
    waiting = _number(reason.get('queue_depth'))
    return Starvation.TOO_FEW_WAITING if waiting is not None and waiting < depth else Starvation.OTHER


@dataclass(frozen=True)
class DepthStarved:
    sized: int
    starved: int
    too_few_waiting: int
    other: int
    share_of_verifies: float | None


def depth_starved(verify: Sequence[Event]) -> DepthStarved:
    """Policy-sized verifies whose chain fell short of the policy row's depth.

    ``too_few_waiting``: the queue held fewer items than that depth;
    ``share_of_verifies`` is starved / every merge verify passed in.
    """
    tally = Counter(kind for e in verify if (kind := _starvation(e)) is not None)
    starved = tally[Starvation.TOO_FEW_WAITING] + tally[Starvation.OTHER]
    return DepthStarved(
        sized=sum(tally.values()), starved=starved,
        too_few_waiting=tally[Starvation.TOO_FEW_WAITING], other=tally[Starvation.OTHER],
        share_of_verifies=_share(starved, len(verify)),
    )


def _red_chain_ends(verify: Sequence[Event]) -> dict[str, list[datetime]]:
    """Member task -> end of each failed deep verify whose chain carried it."""
    ends: dict[str, list[datetime]] = defaultdict(list)
    for event in verify:
        data, moment = _data(event), _at(event)
        members = data.get('chain_task_ids')
        if (_chain_items(event) or 0) < DEEP_CHAIN_MIN_ITEMS or data.get('passed') is not False:
            continue
        if moment is not None and isinstance(members, list):
            for member in members:
                ends[str(member)].append(moment)
    return ends


def _follows_unretired_red(
    task: str, solo_start: datetime,
    red_ends: Mapping[str, list[datetime]], finalizes: Mapping[str, list[datetime]],
) -> bool:
    return any(
        red_end <= solo_start
        and not any(red_end < retired < solo_start for retired in finalizes.get(task, ()))
        for red_end in red_ends.get(task, ())
    )


def _next_finalize_state(finalized: Sequence[Event], task: str, after: datetime) -> object:
    later = [
        (moment, _data(e).get('state')) for e in finalized
        if str(e.get('task_id')) == task and (moment := _at(e)) is not None and moment > after
    ]
    return min(later, key=lambda pair: pair[0])[1] if later else None


@dataclass(frozen=True)
class PostRedSolos:
    count: int
    hours: float
    share_of_verify_hours: float | None
    next_finalize_blocked: int


def _hours(span: Interval) -> float:
    return (span[1] - span[0]).total_seconds() / 3600.0


def post_red_chain_solos(
    verify: Sequence[Event], finalized: Sequence[Event], lo: datetime, hi: datetime,
) -> PostRedSolos:
    """Solo (``chain_items`` 1) verifies in ``[lo, hi)`` re-verifying a member of a red chain.

    The red chain is an earlier failed deep verify that ended at or before the
    solo started (it may predate *lo*: pass a lookback), and the member's
    request was not retired (no ``merge_finalized`` row) between the two.
    """
    red_ends, finalizes = _red_chain_ends(verify), _times_by_task(finalized)
    in_window = [
        (e, span) for e in verify
        if (span := _interval(e, 'duration_ms')) is not None and lo <= span[1] < hi
    ]
    solos = [
        (str(e.get('task_id')), span) for e, span in in_window
        if _chain_items(e) == 1
        and _follows_unretired_red(str(e.get('task_id')), span[0], red_ends, finalizes)
    ]
    solo_hours = sum(_hours(span) for _, span in solos)
    return PostRedSolos(
        count=len(solos), hours=solo_hours,
        share_of_verify_hours=_share(solo_hours, sum(_hours(span) for _, span in in_window)),
        next_finalize_blocked=sum(
            1 for task, span in solos if _next_finalize_state(finalized, task, span[1]) == 'blocked'
        ),
    )


@dataclass(frozen=True)
class ChainWall:
    n: int
    p50_secs: float
    max_secs: float


def per_item_chain_wall(verify: Sequence[Event], min_items: int = 8) -> ChainWall | None:
    """Verify wall time per chained item, over chains of at least *min_items*; None without one."""
    per_item = [
        secs / items for e in verify
        if (items := _chain_items(e)) is not None and items >= min_items
        and (secs := _duration_secs(e)) is not None
    ]
    if not per_item:
        return None
    return ChainWall(n=len(per_item), p50_secs=statistics.median(per_item), max_secs=max(per_item))
