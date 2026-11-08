#!/usr/bin/env python3
"""Census: how much task-verify slot time a cancel-on-red of still-waiting legs would save (task 5630).

Reads `verify_admission_plan` events from an orchestrator runs.db and measures
the admission legs granted after a sibling unit had already gone red: the
slot time a "cancel a plan's still-waiting legs once a sibling is red" change
would have avoided. This is plans/verify-admission-fifo-prd.md task δ. The
definitions, the decision and its rationale live in
plans/verify-admission-fifo-prd.delta-decision.md.

  verify_admission_cancel_on_red_census.py --db dark_factory=data/orchestrator/runs.db [--db NAME=PATH ...]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# Copied from orchestrator/src/orchestrator/event_store.py::EventType.verify_admission_plan, not
# imported: scripts/tests run under `uv run --project shared`, which installs no orchestrator.
PLAN_EVENT_TYPE = 'verify_admission_plan'
MEASURED_ORDER = 'arrival'
MEASURED_OUTCOME = 'returned'
SECS_PER_DAY = 86400.0
SECS_PER_HOUR = 3600.0
TASK_ROLE = 'task'
BACKGROUND_ROLE = 'background'
INTERVAL_QUANTILES = (0.025, 0.975)
DECISION_SHARE = 0.03
MIN_WINDOW_DAYS = 14.0
MIN_MULTI_MODULE_RED_PLANS = 40


@dataclass(frozen=True)
class Leg:
    module: str | None
    arrived_at: float
    granted_at: float
    held_secs: float
    rc: int
    timed_out: bool

    @property
    def red(self) -> bool:
        return self.rc != 0 or self.timed_out

    @property
    def ended_at(self) -> float:
        return self.granted_at + self.held_secs


@dataclass(frozen=True)
class Plan:
    plan_id: str
    role: str
    legs: tuple[Leg, ...]


@dataclass(frozen=True)
class AvoidableLeg:
    leg: Leg
    cutoff: float

    @property
    def queued_at_red(self) -> bool:
        return self.leg.arrived_at <= self.cutoff


Grouping = Callable[[int, Leg], Hashable]


def _module_key(_index: int, leg: Leg) -> Hashable:
    return leg.module


def _leg_key(index: int, _leg: Leg) -> Hashable:
    return index


BY_MODULE: Grouping = _module_key
BY_LEG: Grouping = _leg_key


@dataclass(frozen=True)
class Unit:
    """The attempts sharing one grouping key, in grant order; the final attempt is the unit's verdict."""

    attempts: tuple[Leg, ...]

    @property
    def final(self) -> Leg:
        return self.attempts[-1]

    @property
    def red(self) -> bool:
        return self.final.red

    @property
    def ended_at(self) -> float:
        return self.final.ended_at

    @property
    def retried_attempts(self) -> tuple[Leg, ...]:
        return self.attempts[:-1]


def plan_units(plan: Plan, group: Grouping = BY_MODULE) -> dict[Hashable, Unit]:
    keyed: dict[Hashable, list[Leg]] = {}
    for index, leg in enumerate(plan.legs):
        keyed.setdefault(group(index, leg), []).append(leg)
    return {
        key: Unit(attempts=tuple(sorted(legs, key=lambda leg: leg.granted_at)))
        for key, legs in keyed.items()
    }


def _red_sibling_cutoff(key: Hashable, units: dict[Hashable, Unit]) -> float | None:
    red_ends = [unit.ended_at for other, unit in units.items() if other != key and unit.red]
    return min(red_ends, default=None)


def avoidable_legs(plan: Plan, group: Grouping = BY_MODULE) -> tuple[AvoidableLeg, ...]:
    units = plan_units(plan, group)
    cutoffs = {key: _red_sibling_cutoff(key, units) for key in units}
    found: list[AvoidableLeg] = []
    for index, leg in enumerate(plan.legs):
        cutoff = cutoffs[group(index, leg)]
        if cutoff is not None and leg.granted_at > cutoff:
            found.append(AvoidableLeg(leg=leg, cutoff=cutoff))
    return tuple(found)


@dataclass(frozen=True)
class Sample:
    plans: tuple[Plan, ...]
    window_start: str
    window_end: str
    window_days: float


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def _parse_leg(raw: Mapping[str, Any]) -> Leg:
    return Leg(
        module=raw['module'],
        arrived_at=_epoch(raw['arrived_at']),
        granted_at=_epoch(raw['granted_at']),
        held_secs=float(raw['held_secs']),
        rc=int(raw['rc']),
        timed_out=bool(raw['timed_out']),
    )


def _is_measured(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get('order') == MEASURED_ORDER
        and payload.get('outcome') == MEASURED_OUTCOME
        and bool(payload.get('legs'))
    )


def _plan_rows(db_path: Path) -> list[tuple[str, dict[str, Any]]]:
    conn = sqlite3.connect(f'{db_path.resolve().as_uri()}?mode=ro', uri=True)
    try:
        rows = conn.execute(
            'SELECT timestamp, data FROM events WHERE event_type = ? ORDER BY id', (PLAN_EVENT_TYPE,),
        ).fetchall()
    finally:
        conn.close()
    return [(timestamp, json.loads(data)) for timestamp, data in rows]


def load_arrival_plans(db_path: Path) -> Sample:
    kept = [(timestamp, payload) for timestamp, payload in _plan_rows(db_path) if _is_measured(payload)]
    plans = tuple(
        Plan(
            plan_id=payload['plan_id'],
            role=payload['role'],
            legs=tuple(_parse_leg(raw) for raw in payload['legs']),
        )
        for _, payload in kept
    )
    if not kept:
        return Sample(plans=plans, window_start='', window_end='', window_days=0.0)
    start, end = kept[0][0], kept[-1][0]
    return Sample(
        plans=plans,
        window_start=start,
        window_end=end,
        window_days=(_epoch(end) - _epoch(start)) / SECS_PER_DAY,
    )


@dataclass(frozen=True)
class PlanCounts:
    total: int
    task: int
    background: int


@dataclass(frozen=True)
class CostSide:
    """What a cancel-on-red would have thrown away, over the headline (task-role, BY_MODULE) legs."""

    avoidable_legs: int
    avoidable_legs_queued_at_red: int
    avoidable_modules_red: int
    plans_losing_a_red: int


@dataclass(frozen=True)
class RetriedAttempts:
    """The non-final attempts of every role's modules, whatever made verify re-run them; `timed_out` counts the timeouts."""

    attempts: int
    timed_out: int
    held_secs: float
    share_of_busy: float
    modules: int
    modules_final_green: int


@dataclass(frozen=True)
class Measurement:
    project: str
    window_start: str
    window_end: str
    window_days: float
    plans: PlanCounts
    multi_module_red_plans: int
    busy_secs: float
    avoidable_secs: float
    avoidable_share: float
    share_interval: tuple[float, float]
    avoidable_hours_per_day: float
    literal_avoidable_secs: float
    literal_avoidable_share: float
    background_avoidable_secs: float
    imposed_wait_secs: float
    cost: CostSide
    retried_attempts: RetriedAttempts


Contribution = tuple[float, float]


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _busy_secs(plan: Plan) -> float:
    return math.fsum(leg.held_secs for leg in plan.legs)


def _avoidable_secs(plan: Plan, group: Grouping = BY_MODULE) -> float:
    return math.fsum(item.leg.held_secs for item in avoidable_legs(plan, group))


def _contribution(plan: Plan) -> Contribution:
    """(slot-busy seconds, headline avoidable seconds) of one plan; only task plans enter the numerator."""
    return _busy_secs(plan), (_avoidable_secs(plan) if plan.role == TASK_ROLE else 0.0)


def _share(contributions: Iterable[Contribution]) -> float:
    pairs = list(contributions)
    return _ratio(math.fsum(avoidable for _, avoidable in pairs), math.fsum(busy for busy, _ in pairs))


def _percentile(ordered: Sequence[float], q: float) -> float:
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _bootstrap_interval(contributions: Sequence[Contribution], resamples: int, seed: int) -> tuple[float, float]:
    if resamples < 1:
        raise ValueError(f'resamples must be at least 1, got {resamples}')
    if not contributions:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(contributions)
    shares = sorted(
        _share(contributions[i] for i in rng.choices(range(n), k=n)) for _ in range(resamples)
    )
    low_q, high_q = INTERVAL_QUANTILES
    return _percentile(shares, low_q), _percentile(shares, high_q)


def _covered_secs(span: tuple[float, float], intervals: Iterable[tuple[float, float]]) -> float:
    """Measure of the union of *intervals*, clipped to *span*."""
    start, end = span
    clipped = sorted(
        (max(low, start), min(high, end)) for low, high in intervals if min(high, end) > max(low, start)
    )
    covered, reach = 0.0, start
    for low, high in clipped:
        if high > reach:
            covered += high - max(low, reach)
            reach = high
    return covered


def _imposed_wait_secs(plans: Sequence[Plan]) -> float:
    """Seconds of headline avoidable holds during which at least one OTHER plan's leg was waiting."""
    waits = [(index, (leg.arrived_at, leg.granted_at)) for index, plan in enumerate(plans) for leg in plan.legs]
    total = 0.0
    for index, plan in enumerate(plans):
        if plan.role != TASK_ROLE:
            continue
        others = [wait for owner, wait in waits if owner != index]
        for item in avoidable_legs(plan):
            total += _covered_secs((item.leg.granted_at, item.leg.ended_at), others)
    return total


def _is_multi_module_red(plan: Plan) -> bool:
    units = plan_units(plan).values()
    return len(units) >= 2 and any(unit.red for unit in units)


def _cost_side(task_plans: Iterable[Plan]) -> CostSide:
    legs = queued = modules_red = plans_losing = 0
    for plan in task_plans:
        found = avoidable_legs(plan)
        units = plan_units(plan)
        red_modules = {item.leg.module for item in found if units[item.leg.module].red}
        legs += len(found)
        queued += sum(item.queued_at_red for item in found)
        modules_red += len(red_modules)
        plans_losing += bool(red_modules)
    return CostSide(
        avoidable_legs=legs,
        avoidable_legs_queued_at_red=queued,
        avoidable_modules_red=modules_red,
        plans_losing_a_red=plans_losing,
    )


def _retried_attempts(plans: Iterable[Plan], busy_secs: float) -> RetriedAttempts:
    retried = [unit for plan in plans for unit in plan_units(plan).values() if unit.retried_attempts]
    attempts = [leg for unit in retried for leg in unit.retried_attempts]
    held = math.fsum(leg.held_secs for leg in attempts)
    return RetriedAttempts(
        attempts=len(attempts),
        timed_out=sum(leg.timed_out for leg in attempts),
        held_secs=held,
        share_of_busy=_ratio(held, busy_secs),
        modules=len(retried),
        modules_final_green=sum(not unit.red for unit in retried),
    )


def measure(project: str, sample: Sample, *, resamples: int, seed: int) -> Measurement:
    task_plans = [plan for plan in sample.plans if plan.role == TASK_ROLE]
    contributions = [_contribution(plan) for plan in sample.plans]
    busy = math.fsum(busy for busy, _ in contributions)
    avoidable = math.fsum(avoidable for _, avoidable in contributions)
    literal = math.fsum(_avoidable_secs(plan, BY_LEG) for plan in task_plans)
    return Measurement(
        project=project,
        window_start=sample.window_start,
        window_end=sample.window_end,
        window_days=sample.window_days,
        plans=PlanCounts(
            total=len(sample.plans),
            task=len(task_plans),
            background=sum(plan.role == BACKGROUND_ROLE for plan in sample.plans),
        ),
        multi_module_red_plans=sum(_is_multi_module_red(plan) for plan in task_plans),
        busy_secs=busy,
        avoidable_secs=avoidable,
        avoidable_share=_ratio(avoidable, busy),
        share_interval=_bootstrap_interval(contributions, resamples, seed),
        avoidable_hours_per_day=_ratio(avoidable / SECS_PER_HOUR, sample.window_days),
        literal_avoidable_secs=literal,
        literal_avoidable_share=_ratio(literal, busy),
        background_avoidable_secs=math.fsum(
            _avoidable_secs(plan) for plan in sample.plans if plan.role == BACKGROUND_ROLE
        ),
        imposed_wait_secs=_imposed_wait_secs(sample.plans),
        cost=_cost_side(task_plans),
        retried_attempts=_retried_attempts(sample.plans, busy),
    )


class Verdict(Enum):
    FILE = 'file'
    DECLINE = 'decline'
    THIN_DATA = 'thin_data'


def _side_of_decision_share(share: float) -> Verdict:
    return Verdict.FILE if share >= DECISION_SHARE else Verdict.DECLINE


def verdict(m: Measurement) -> Verdict:
    if m.multi_module_red_plans < MIN_MULTI_MODULE_RED_PLANS or m.window_days <= 0:
        return Verdict.THIN_DATA
    if m.window_days >= MIN_WINDOW_DAYS:
        return _side_of_decision_share(m.avoidable_share)
    low, high = m.share_interval
    if _side_of_decision_share(low) is _side_of_decision_share(high):
        return _side_of_decision_share(low)
    return Verdict.THIN_DATA


def _named_db(value: str) -> tuple[str, Path]:
    name, sep, path = value.partition('=')
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f'expected NAME=PATH, got {value!r}')
    return name, Path(path)


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f'expected a positive integer, got {value!r}')
    return number


def _report(name: str, db_path: Path, resamples: int, seed: int) -> dict[str, Any]:
    m = measure(name, load_arrival_plans(db_path), resamples=resamples, seed=seed)
    return {**asdict(m), 'verdict': verdict(m).value}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--db', type=_named_db, action='append', required=True, metavar='NAME=PATH')
    parser.add_argument('--resamples', type=_positive_int, default=2000)
    parser.add_argument('--seed', type=int, default=5630)
    args = parser.parse_args(argv)
    reports = []
    for name, db_path in args.db:
        try:
            reports.append(_report(name, db_path, args.resamples, args.seed))
        except sqlite3.OperationalError as exc:
            print(f'{db_path}: {exc}', file=sys.stderr)
            return 1
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
