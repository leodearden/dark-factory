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

import argparse
import math
import sqlite3
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

from audit_model_admission import (
    DEFAULT_RUNS_DB,
    MERGER_ROLE,
    InvocationRecord,
    RoleContainment,
    RoutingRejection,
    RoutingScan,
    ScopedCapScan,
    ServiceRestart,
    SpendInWindow,
    connect_ro,
    iso,
    load_events,
    markdown_table,
    parse_moment,
    roles_on_model,
    scan_invocations,
    scan_routing_decisions,
    scan_scoped_cap,
    spend_in_window,
)
from escalation_ladder import (
    DEFAULT_ESCALATIONS_DIR,
    STEWARD_ROLE,
    EscalationCorpus,
    L2TierMetrics,
    StewardDisposition,
    StewardDispositionSummary,
    l2_tier_metrics,
    load_escalation_corpus,
    summarize_steward_dispositions,
)

# The CLI's result subtypes for a run killed at its dispatch caps. The _usd
# spelling is orchestrator/src/orchestrator/routing.py::PROBE_BUDGET_EXHAUSTED_SUBTYPE.
TURN_CAP_SUBTYPE = 'error_max_turns'
BUDGET_CAP_SUBTYPE = 'error_max_budget_usd'

MERGE_DONE_STATE = 'done'
MERGE_CONFLICT_STATE = 'conflict'

# The drop guard's two witnesses. The prefix is
# orchestrator/src/orchestrator/merge_gates.py::DROPPED_PLAN_TARGETS_REASON_PREFIX;
# the outcome is the OutcomeKind.dropped_plan_targets that merge_queue.py passes
# to _emit_merge_attempt when the gate fires.
DROPPED_PLAN_TARGETS_REASON_PREFIX = 'Merge commit is missing plan target files'
DROPPED_PLAN_TARGETS_OUTCOME = 'dropped_plan_targets'
MERGE_ATTEMPT_EVENT = 'merge_attempt'
MERGE_FINALIZED_EVENT = 'merge_finalized'

# One of audit_model_admission.MODEL_REJECTION_REASONS: the per-model daily
# ceiling was spent, so the resolver fell through to another model.
CEILING_EXHAUSTED_REASON = 'model-ceiling-exhausted'
DAY = timedelta(days=1)
UNKNOWN_KEY = '-'

_Number = TypeVar('_Number', int, float)
_Key = TypeVar('_Key', str, int)


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


@dataclass(frozen=True)
class DropGuardEvent:
    """One firing of the gate that refuses a merge commit missing plan target files."""

    timestamp: str
    task_id: str | None
    source: str


def scan_drop_guard(
    conn: sqlite3.Connection, *, since: datetime, until: datetime
) -> tuple[DropGuardEvent, ...]:
    """Every drop-guard firing in ``[since, until)``, read from both witnesses.

    The merge_attempt row is the gate's own emission; the merge_finalized
    reason is what the workflow routes on (workflow.py short-circuits on the
    prefix). Reading both makes a firing that reached only one of them visible.
    """
    attempts = [
        DropGuardEvent(row.timestamp, row.task_id, MERGE_ATTEMPT_EVENT)
        for row in load_events(conn, MERGE_ATTEMPT_EVENT, since, until)
        if row.payload.get('outcome') == DROPPED_PLAN_TARGETS_OUTCOME
    ]
    finalized = [
        DropGuardEvent(row.timestamp, row.task_id, MERGE_FINALIZED_EVENT)
        for row in load_events(conn, MERGE_FINALIZED_EVENT, since, until)
        if str(row.payload.get('reason') or '').startswith(DROPPED_PLAN_TARGETS_REASON_PREFIX)
    ]
    return tuple(sorted([*attempts, *finalized], key=lambda event: event.timestamp))


@dataclass(frozen=True)
class ConflictReopen:
    """A merge conflict on a task AFTER a merger run on it had succeeded."""

    task_id: str | None
    run_completed_at: str
    reopened_at: str


def conflict_reopens(
    conn: sqlite3.Connection, records: Sequence[InvocationRecord], *, until: datetime
) -> tuple[ConflictReopen, ...]:
    """Every conflict finalized on a successful merger run's task after it
    completed and before *until*. One event load, from the earliest such run."""
    resolved = [r for r in records if r.role == MERGER_ROLE and r.succeeded is True]
    if not resolved:
        return ()
    earliest = datetime.fromisoformat(min(r.completed_at for r in resolved))
    conflicts: dict[str | None, list[str]] = {}
    for row in load_events(conn, MERGE_FINALIZED_EVENT, earliest, until):
        if row.payload.get('state') == MERGE_CONFLICT_STATE:
            conflicts.setdefault(row.task_id, []).append(row.timestamp)
    return tuple(
        ConflictReopen(task_id=r.task_id, run_completed_at=r.completed_at, reopened_at=at)
        for r in resolved
        for at in conflicts.get(r.task_id, ())
        if at > r.completed_at
    )


def daily_spend(
    conn: sqlite3.Connection,
    *,
    model: str,
    since: datetime,
    days: int,
    ceiling_usd: float | None,
) -> tuple[SpendInWindow, ...]:
    """*model*'s spend in each of *days* consecutive 24 h slices from *since*.

    Each slice is half-open, so adjacent days never double-count; see
    audit_model_admission.spend_in_window's docstring for why.
    """
    return tuple(
        spend_in_window(
            conn, model=model, window_start=since + k * DAY,
            window_end=since + (k + 1) * DAY, ceiling_usd=ceiling_usd,
        )
        for k in range(days)
    )


@dataclass(frozen=True)
class PeakSpend:
    """The largest trailing-24 h spend on a model, and the moment it was reached.

    ``at_or_over_ceiling`` is None without a ceiling: unknown, never a
    plausible-looking False (the SpendInWindow convention).
    """

    total_usd: float
    at: str
    ceiling_usd: float | None
    at_or_over_ceiling: bool | None


def peak_trailing_24h(
    conn: sqlite3.Connection,
    *,
    model: str,
    since: datetime,
    until: datetime,
    ceiling_usd: float | None,
) -> PeakSpend | None:
    """The maximum, over every *model* run completed in ``[since, until)`` at t,
    of the spend completed in the CLOSED window ``[t - 24h, t]``.

    Closed, unlike the daily slices, because it asks "could the resolver have
    tripped?", and shared/src/shared/cost_store.py::CostStore.model_cost_in_window
    sums with an inclusive BETWEEN. None when nothing ran.
    """
    runs = [
        (datetime.fromisoformat(completed_at), cost)
        for completed_at, cost in conn.execute(
            'SELECT completed_at, cost_usd FROM invocations WHERE model = ? '
            'AND completed_at >= ? AND completed_at < ? ORDER BY completed_at, id',
            (model, iso(since), iso(until)),
        )
    ]
    best: tuple[float, datetime] | None = None
    window_total, left = 0.0, 0
    for at, cost in runs:
        window_total += cost
        while runs[left][0] < at - DAY:
            window_total -= runs[left][1]
            left += 1
        if best is None or window_total > best[0]:
            best = (window_total, at)
    if best is None:
        return None
    total, at = best
    return PeakSpend(
        total_usd=total,
        at=iso(at),
        ceiling_usd=ceiling_usd,
        at_or_over_ceiling=None if ceiling_usd is None else total >= ceiling_usd,
    )


def model_rejections(scan: RoutingScan) -> tuple[RoutingRejection, ...]:
    """Every model rejection the resolver recorded, on any role."""
    return scan.rejections


def ceiling_trips(scan: RoutingScan) -> tuple[RoutingRejection, ...]:
    """The rejections that were a spent per-model ceiling, narrowed to that
    reason; ``resolved_model`` is the model the resolver fell through to."""
    trips = []
    for rejection in scan.rejections:
        reasons = tuple(
            reason for reason in rejection.reasons
            if reason.rpartition(':')[2] == CEILING_EXHAUSTED_REASON
        )
        if reasons:
            trips.append(RoutingRejection(
                timestamp=rejection.timestamp, task_id=rejection.task_id,
                role=rejection.role, resolved_model=rejection.resolved_model,
                reasons=reasons,
            ))
    return tuple(trips)


def scoped_hits_by_account(scan: ScopedCapScan) -> tuple[tuple[str, int, str], ...]:
    """(account, scoped cap hits, first hit's created_at), sorted by account."""
    by_account: dict[str, list[str]] = {}
    for hit in scan.scoped_hits:
        by_account.setdefault(hit.account_name, []).append(hit.created_at)
    return tuple(
        (account, len(stamps), min(stamps))
        for account, stamps in sorted(by_account.items())
    )


@dataclass(frozen=True)
class _TierBand:
    """The dispatch tiers an arm admits: ``[low, high)``, either end open when None.

    A run whose tier is unknown is admitted only by the unbounded band, so a
    tier-split arm never guesses where it belongs.
    """

    description: str
    low: int | None = None
    high: int | None = None

    def admits(self, tier: int | None) -> bool:
        if self.low is None and self.high is None:
            return True
        return tier is not None and (self.low is None or tier >= self.low) and (
            self.high is None or tier < self.high)


ANY_TIER = _TierBand('any')
IN_WINDOW = 'window'
BEFORE_APPLY = 'before apply'


@dataclass(frozen=True)
class _Scan:
    """Every run of one model completed in one window, read once."""

    model: str
    name: str
    start: datetime
    end: datetime
    records: tuple[InvocationRecord, ...]


@dataclass(frozen=True)
class Arm:
    """One comparison column: a model's runs of a role in a window, tier-filtered.

    ``dispositions`` is how the escalations the runs worked on left the
    steward, judged as of ``window_end``; None for a role that works none.
    """

    label: str
    model: str
    role: str
    window_start: str
    window_end: str
    tier_filter: str
    outcomes: OutcomeSummary
    dispositions: StewardDispositionSummary | None


def _arm(scan: _Scan, role: str, band: _TierBand, corpus: EscalationCorpus | None) -> Arm:
    """*scan*'s runs of *role* that *band* admits; dispositions when given a corpus."""
    records = [r for r in scan.records if r.role == role and band.admits(r.routing_tier)]
    suffix = '' if band is ANY_TIER else f', {band.description}'
    return Arm(
        label=f'{scan.model} {role}, {scan.name}{suffix}',
        model=scan.model,
        role=role,
        window_start=iso(scan.start),
        window_end=iso(scan.end),
        tier_filter=band.description,
        outcomes=summarize_outcomes(records),
        dispositions=None if corpus is None else summarize_steward_dispositions(
            (r.escalation_id for r in records), corpus, as_of=scan.end,
        ),
    )


def _steward_arms(
    layout: Sequence[tuple[_Scan, Sequence[_TierBand]]], corpus: EscalationCorpus
) -> tuple[tuple[Arm, ...], tuple[tuple[str, InvocationRecord], ...]]:
    """One steward arm per (scan, band) in *layout*, plus every steward run that
    no band on its scan admits (with its model), so none vanishes between them."""
    arms = tuple(_arm(scan, STEWARD_ROLE, band, corpus)
                 for scan, bands in layout for band in bands)
    in_no_arm = tuple(
        (scan.model, r) for scan, bands in layout for r in scan.records
        if r.role == STEWARD_ROLE and not any(band.admits(r.routing_tier) for band in bands)
    )
    return arms, in_no_arm


@dataclass(frozen=True)
class ReviewResult:
    """Every measurement the review renders, frozen against one read of each store.

    The review window is ``[window_start, window_end)`` and the baseline window
    ``[baseline_start, window_start)``; the scoped-cap, spend, rejection and
    role sections are all over the review window.
    """

    model: str
    baseline_model: str
    window_start: str
    window_end: str
    baseline_start: str
    steward_min_tier: int
    ceiling_usd: float | None
    merger_arms: tuple[Arm, ...]
    steward_arms: tuple[Arm, ...]
    steward_runs_in_no_arm: tuple[tuple[str, InvocationRecord], ...]
    drop_guard_window: tuple[DropGuardEvent, ...]
    drop_guard_baseline: tuple[DropGuardEvent, ...]
    conflict_reopens: tuple[ConflictReopen, ...]
    l2_window: L2TierMetrics
    l2_baseline: L2TierMetrics
    escalations_loaded: int
    escalations_skipped: int
    oldest_archive_date: str | None
    daily_spend: tuple[SpendInWindow, ...]
    peak_24h: PeakSpend | None
    model_rejections: tuple[RoutingRejection, ...]
    ceiling_trips: tuple[RoutingRejection, ...]
    scoped_hits: tuple[tuple[str, int, str], ...]
    unscoped_cap_hits: int
    restarts: tuple[ServiceRestart, ...]
    containment: RoleContainment


def review(
    conn: sqlite3.Connection,
    corpus: EscalationCorpus,
    *,
    model: str,
    baseline_model: str,
    expected_roles: Sequence[str],
    apply: datetime,
    days: int,
    baseline_days: int,
    ceiling_usd: float | None,
    steward_min_tier: int = 1,
    role_ceilings_secs: dict[str, int] | None = None,
) -> ReviewResult:
    """Measure *model* over ``[apply, apply + days)`` against *baseline_model*
    there and over ``[apply - baseline_days, apply)``.

    Each (model, window) pair is scanned ONCE and split by role and dispatch
    tier in Python. Only three pairs are read: no arm compares the candidate
    before its own admission.
    """
    end, baseline_start = apply + days * DAY, apply - baseline_days * DAY

    def scan(scan_model: str, name: str, start: datetime, stop: datetime) -> _Scan:
        return _Scan(scan_model, name, start, stop, scan_invocations(
            conn, model=scan_model, since=start, until=stop,
            role_ceilings_secs=role_ceilings_secs,
        ))

    candidate = scan(model, IN_WINDOW, apply, end)
    baseline_in_window = scan(baseline_model, IN_WINDOW, apply, end)
    baseline_before = scan(baseline_model, BEFORE_APPLY, baseline_start, apply)
    at_or_above = _TierBand(f'tier >= {steward_min_tier}', low=steward_min_tier)
    below = _TierBand(f'tier < {steward_min_tier}', high=steward_min_tier)
    steward_arms, in_no_arm = _steward_arms(
        ((candidate, (at_or_above,)), (baseline_in_window, (below,)),
         (baseline_before, (at_or_above, below))),
        corpus,
    )
    routing = scan_routing_decisions(conn, model=model, since=apply, until=end)
    scoped = scan_scoped_cap(conn, model=model, since=apply, until=end)
    return ReviewResult(
        model=model,
        baseline_model=baseline_model,
        window_start=iso(apply),
        window_end=iso(end),
        baseline_start=iso(baseline_start),
        steward_min_tier=steward_min_tier,
        ceiling_usd=ceiling_usd,
        merger_arms=tuple(_arm(s, MERGER_ROLE, ANY_TIER, None)
                          for s in (candidate, baseline_before, baseline_in_window)),
        steward_arms=steward_arms,
        steward_runs_in_no_arm=in_no_arm,
        drop_guard_window=scan_drop_guard(conn, since=apply, until=end),
        drop_guard_baseline=scan_drop_guard(conn, since=baseline_start, until=apply),
        conflict_reopens=conflict_reopens(conn, candidate.records, until=end),
        l2_window=l2_tier_metrics(corpus, since=apply, until=end),
        l2_baseline=l2_tier_metrics(corpus, since=baseline_start, until=apply),
        escalations_loaded=len(corpus.records),
        escalations_skipped=corpus.skipped,
        oldest_archive_date=corpus.oldest_archive_date,
        daily_spend=daily_spend(conn, model=model, since=apply, days=days,
                                ceiling_usd=ceiling_usd),
        peak_24h=peak_trailing_24h(conn, model=model, since=apply, until=end,
                                   ceiling_usd=ceiling_usd),
        model_rejections=model_rejections(routing),
        ceiling_trips=ceiling_trips(routing),
        scoped_hits=scoped_hits_by_account(scoped),
        unscoped_cap_hits=scoped.unscoped_cap_hit_count,
        restarts=scoped.restarts,
        containment=roles_on_model(conn, model=model, since=apply, until=end,
                                   expected_roles=expected_roles),
    )


# --- rendering: markdown only, every number read from the frozen result ---


def _rate(count: int, of: int) -> str:
    """'k/n (x%)', so the denominator of every rate stays visible."""
    return f'{count}/{of} (-)' if of == 0 else f'{count}/{of} ({count / of:.1%})'


def _usd(value: float | None) -> str:
    return '-' if value is None else f'{value:.2f}'


def _secs(duration_ms: int | None) -> str:
    return '-' if duration_ms is None else f'{duration_ms / 1000:.0f}'


def _known(value: Any) -> str:
    return '-' if value is None else str(value)


def _breakdown(tally: tuple[tuple[Any, int], ...]) -> str:
    return ', '.join(f'{key}: {count}' for key, count in tally) or 'none'


def _span(start: str, end: str) -> str:
    return f'[{start}, {end})'


def _preamble(result: ReviewResult) -> list[str]:
    ceiling = ('not supplied' if result.ceiling_usd is None
               else f'${result.ceiling_usd:.2f}')
    return [
        f'Candidate `{result.model}` against baseline `{result.baseline_model}`. '
        f'Review window {_span(result.window_start, result.window_end)}; baseline '
        f'window {_span(result.baseline_start, result.window_start)}. Steward arms '
        f'split at dispatch tier {result.steward_min_tier}. Per-model daily '
        f'ceiling: {ceiling}.',
        '',
        'Rates read k/n (x%). Percentiles are nearest-rank, so each is an observed '
        "run's value. Durations are in seconds. '-' is unknown: no end event, no "
        'dispatch decision or no attributed merge.',
        '',
    ]


def _arms_table(arms: Sequence[Arm]) -> list[str]:
    return markdown_table(
        ['arm', 'model', 'role', 'window', 'tier filter'],
        [(a.label, a.model, a.role, _span(a.window_start, a.window_end), a.tier_filter)
         for a in arms],
    )


_ENDINGS_HEADER = ['arm', 'runs', 'succeeded', 'turn-cap kills', 'budget kills',
                   'timed out', 'no end event']


def _endings_cells(arm: Arm) -> list[Any]:
    o = arm.outcomes
    return [arm.label, o.runs, _rate(o.succeeded, o.runs), _rate(o.turn_cap_kills, o.runs),
            _rate(o.budget_kills, o.runs), _rate(o.timed_out, o.runs),
            _rate(o.no_end_event, o.runs)]


def _distribution_table(arms: Sequence[Arm]) -> list[str]:
    return markdown_table(
        ['arm', 'cost total $', 'cost p50 $', 'cost p95 $', 'turns p50', 'turns p95',
         'duration p50 s', 'duration p95 s', 'duration max s'],
        [(a.label, _usd(a.outcomes.cost_total_usd), _usd(a.outcomes.cost_usd_p50),
          _usd(a.outcomes.cost_usd_p95), _known(a.outcomes.turns_p50),
          _known(a.outcomes.turns_p95), _secs(a.outcomes.duration_ms_p50),
          _secs(a.outcomes.duration_ms_p95), _secs(a.outcomes.duration_ms_max))
         for a in arms],
    )


def _drop_guard_table(events: Sequence[DropGuardEvent]) -> list[str]:
    return markdown_table(['timestamp', 'task', 'witness'],
                          [(e.timestamp, _known(e.task_id), e.source) for e in events])


def _merger_section(result: ReviewResult) -> list[str]:
    window = _span(result.window_start, result.window_end)
    baseline = _span(result.baseline_start, result.window_start)
    arms = result.merger_arms
    out = [f'### 1. Merger: `{result.model}` in {window}, against '
           f'`{result.baseline_model}` in {baseline} and in {window}', '']
    out += [*_arms_table(arms), '']
    out += [*markdown_table(
        [*_ENDINGS_HEADER, 'resolved (merge done)', 'over flat ceiling'],
        [(*_endings_cells(a), _rate(a.outcomes.resolved, a.outcomes.runs),
          _rate(a.outcomes.over_flat_ceiling, a.outcomes.runs)) for a in arms],
    ), '']
    out += [*_distribution_table(arms), '']
    out += [*markdown_table(
        ['arm', 'merge states', 'dispatch max_turns'],
        [(a.label, _breakdown(a.outcomes.merge_states),
          _breakdown(a.outcomes.dispatch_caps)) for a in arms],
    ), '']
    out += [f'Drop-guard firings in {window}:', '',
            *_drop_guard_table(result.drop_guard_window), '']
    out += [f'Drop-guard firings in {baseline}:', '',
            *_drop_guard_table(result.drop_guard_baseline), '']
    out += [f'Conflicts finalized after a successful `{result.model}` merger run, '
            f'before {result.window_end}:', '']
    out += [*markdown_table(
        ['task', 'run completed', 're-opened at'],
        [(_known(r.task_id), r.run_completed_at, r.reopened_at)
         for r in result.conflict_reopens],
    ), '']
    return out


def _dispositions_table(arms: Sequence[Arm]) -> list[str]:
    rows = []
    for arm in arms:
        d = arm.dispositions
        if d is None:
            continue
        rows.append((
            arm.label, d.unlinked_runs, *(n for _, n in d.counts), d.decided,
            _rate(d.count(StewardDisposition.RESOLVED_IN_PLACE), d.decided),
            _rate(d.count(StewardDisposition.PROMOTED_TO_L1), d.decided),
        ))
    return markdown_table(
        ['arm', 'runs naming no escalation', *(d.value for d in StewardDisposition),
         'decided', 'in-place share', 'promoted share'],
        rows,
    )


def _l2_table(metrics: Sequence[L2TierMetrics]) -> list[str]:
    return markdown_table(
        ['L2 window', 'days', 'filed', 'filed/day', 'watcher filed', 'watcher filed/day',
         'resolved', 'close_only share'],
        [(_span(iso(m.since), iso(m.until)), f'{m.days:g}', m.filed,
          f'{m.filed_per_day:.2f}', m.watcher_filed, f'{m.watcher_filed_per_day:.2f}',
          m.resolved, _rate(m.close_only, m.resolved)) for m in metrics],
    )


def _steward_section(result: ReviewResult) -> list[str]:
    window = _span(result.window_start, result.window_end)
    baseline = _span(result.baseline_start, result.window_start)
    arms = result.steward_arms
    out = [f'### 2. Steward and the L2 tier: `{result.model}` in {window}, against '
           f'`{result.baseline_model}` in {window} and in {baseline}', '']
    out += [*_arms_table(arms), '']
    out += [*markdown_table(_ENDINGS_HEADER, [_endings_cells(a) for a in arms]), '']
    out += [*_distribution_table(arms), '']
    out += [*markdown_table(
        ['arm', 'dispatch max_turns'],
        [(a.label, _breakdown(a.outcomes.dispatch_caps)) for a in arms],
    ), '']
    out += ["How the escalations each arm's runs worked on left the steward, judged "
            "as of the arm's window end. Shares are over the decided ones (neither "
            'missing nor pending).', '', *_dispositions_table(arms), '']
    out += ['Steward runs no arm admits (tier outside every band on their window, '
            'or unknown):', '']
    out += [*markdown_table(
        ['model', 'task', 'completed_at', 'tier'],
        [(m, _known(r.task_id), r.completed_at, _known(r.routing_tier))
         for m, r in result.steward_runs_in_no_arm],
    ), '']
    out += [*_l2_table((result.l2_baseline, result.l2_window)), '']
    out += [f'Escalation corpus: {result.escalations_loaded} records loaded, '
            f'{result.escalations_skipped} skipped; oldest archive date '
            f'{result.oldest_archive_date or "none"} (records resolved before it '
            'have been pruned).', '']
    return out


def _cost_section(result: ReviewResult) -> list[str]:
    window = _span(result.window_start, result.window_end)
    peak = result.peak_24h
    out = [f'### 3. Spend and caps on `{result.model}` over {window}', '',
           'Daily spend, half-open 24 h slices:', '']
    out += [*markdown_table(
        ['slice start', 'slice end', 'runs', 'total $', 'ceiling $', 'headroom $',
         'at/over ceiling'],
        [(s.window_start, s.window_end, s.invocation_count, _usd(s.total_usd),
          _usd(s.ceiling_usd), _usd(s.headroom_usd), _known(s.at_or_over_ceiling))
         for s in result.daily_spend],
    ), '']
    out += ['Peak trailing-24 h spend (closed window, the resolver\'s rule):', '']
    out += [*markdown_table(
        ['peak $', 'reached at', 'ceiling $', 'at/over ceiling'],
        [] if peak is None else [(_usd(peak.total_usd), peak.at, _usd(peak.ceiling_usd),
                                  _known(peak.at_or_over_ceiling))],
    ), '']
    out += ['Model rejections, any role:', '']
    out += [*markdown_table(
        ['timestamp', 'task', 'role', 'resolved to', 'reasons'],
        [(r.timestamp, _known(r.task_id), r.role, r.resolved_model, ', '.join(r.reasons))
         for r in result.model_rejections],
    ), '']
    out += ['Per-model ceiling trips, and the model each fell through to:', '']
    out += [*markdown_table(
        ['timestamp', 'task', 'role', 'fell through to'],
        [(t.timestamp, _known(t.task_id), t.role, t.resolved_model)
         for t in result.ceiling_trips],
    ), '']
    out += [f'Cap hits scoped to `{result.model}`, by account:', '']
    out += [*markdown_table(['account', 'scoped hits', 'first hit'],
                            result.scoped_hits), '']
    out += [f'Account-level (unscoped) cap hits: {result.unscoped_cap_hits}', '',
            'Service restarts:', '']
    out += [*markdown_table(
        ['timestamp', 'service', 'reason'],
        [(r.timestamp, r.service, r.reason or '-') for r in result.restarts],
    ), '']
    return out


def _leakage_section(result: ReviewResult) -> list[str]:
    containment = result.containment
    out = [f'### 4. Roles observed on `{result.model}` in '
           f'{_span(result.window_start, result.window_end)}', '',
           f'Admitted roles: {", ".join(containment.expected_roles)}', '']
    out += [*markdown_table(
        ['role', 'runs', 'total $', 'admitted'],
        [(u.role, u.count, _usd(u.total_usd), u.role in containment.expected_roles)
         for u in containment.by_role],
    ), '']
    out += [f'Roles outside the admitted set: '
            f'{", ".join(containment.unexpected_roles) or "none"}', '']
    return out


def render_markdown(result: ReviewResult) -> str:
    """Render the four sections, each heading stating the windows it covers."""
    return '\n'.join([
        *_preamble(result),
        *_merger_section(result),
        *_steward_section(result),
        *_cost_section(result),
        *_leakage_section(result),
    ])


def _positive_int(spec: str) -> int:
    """A positive integer, rejected at the boundary: a zero-day window renders
    truthful-looking zeros that read as a measurement."""
    if not spec.isdigit() or int(spec) <= 0:
        raise argparse.ArgumentTypeError(f'expected a positive integer, got {spec!r}')
    return int(spec)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Review an admitted model's per-role outcomes, merge integrity, "
                    'spend and containment against a baseline model over fixed '
                    'windows around the admission. Strictly read-only.',
    )
    parser.add_argument('--model', required=True, help='the admitted (candidate) model')
    parser.add_argument('--baseline-model', default='opus',
                        help='the model the candidate replaced (default: opus)')
    parser.add_argument('--expect-roles', required=True,
                        help='comma-separated roles the model was admitted for')
    parser.add_argument('--apply', required=True, type=parse_moment,
                        help='ISO-8601 instant the admission was applied')
    parser.add_argument('--days', default=14, type=_positive_int,
                        help='review window length from --apply (default: 14)')
    parser.add_argument('--baseline-days', default=30, type=_positive_int,
                        help='baseline window length before --apply (default: 30)')
    parser.add_argument('--ceiling', default=None, type=float,
                        help="the candidate's per-model daily ceiling in USD; omitted "
                             'means unknown, rendered as "-"')
    parser.add_argument('--steward-min-tier', default=1, type=int,
                        help='dispatch tier the steward arms split at (default: 1)')
    parser.add_argument('--runs-db', default=DEFAULT_RUNS_DB, type=Path)
    parser.add_argument('--escalations-dir', default=DEFAULT_ESCALATIONS_DIR, type=Path)
    args = parser.parse_args(argv)

    corpus = load_escalation_corpus(args.escalations_dir)
    conn = connect_ro(args.runs_db)
    try:
        result = review(
            conn, corpus,
            model=args.model,
            baseline_model=args.baseline_model,
            expected_roles=tuple(
                r.strip() for r in args.expect_roles.split(',') if r.strip()
            ),
            apply=args.apply,
            days=args.days,
            baseline_days=args.baseline_days,
            ceiling_usd=args.ceiling,
            steward_min_tier=args.steward_min_tier,
        )
    finally:
        conn.close()
    print(render_markdown(result))
    return 0


if __name__ == '__main__':
    sys.exit(main())
