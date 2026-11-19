"""Measure, over a window, what CPU each cgroup used and wanted and what its cpu.weight entitles it to among its siblings.

Read-only. Usually pointed at the user@1000.service tree described in
docs/cgroup-slice-topology.md.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping


@dataclasses.dataclass(frozen=True)
class Claim:
    weight: int | None
    demand: float


def fair_shares(capacity: float, claims: Mapping[str, Claim]) -> dict[str, float] | None:
    """cgroup-v2 cpu.weight water-filling; None when any sibling has no weight."""
    unsatisfied: dict[str, tuple[int, float]] = {}
    for name, claim in claims.items():
        if claim.weight is None:
            return None
        unsatisfied[name] = (claim.weight, claim.demand)
    shares: dict[str, float] = {}
    remaining = capacity
    while satisfied := _satisfied_at(remaining, unsatisfied):
        for name in satisfied:
            _, shares[name] = unsatisfied.pop(name)
            remaining -= shares[name]
    shares.update(_split_by_weight(remaining, unsatisfied))
    return {name: shares[name] for name in claims}


def _split_by_weight(capacity: float, claims: Mapping[str, tuple[int, float]]) -> dict[str, float]:
    total = sum(weight for weight, _ in claims.values())
    return {name: capacity * weight / total for name, (weight, _) in claims.items()}


def _satisfied_at(capacity: float, claims: Mapping[str, tuple[int, float]]) -> list[str]:
    proportional = _split_by_weight(capacity, claims)
    return [name for name, (_, demand) in claims.items() if proportional[name] >= demand]
