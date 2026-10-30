#!/usr/bin/env python3
"""Census: how much task-verify slot time a cancel-on-red of still-waiting legs would save (task 5630).

Reads `verify_admission_plan` events from an orchestrator runs.db and measures
the admission legs granted after a sibling unit had already gone red: the
slot time a "cancel a plan's still-waiting legs once a sibling is red" change
would have avoided. This is plans/verify-admission-fifo-prd.md task δ. The
definitions, the decision and its rationale live in
plans/verify-admission-fifo-prd.delta-decision.md.
"""
from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass


@dataclass(frozen=True)
class Leg:
    module: str | None
    arrived_at: float
    granted_at: float
    held_secs: float
    red: bool

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
