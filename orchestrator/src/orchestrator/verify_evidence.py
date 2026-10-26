"""What a verify verdict attests: the identity of the plan it ran and the outcome
of each unit, as typed values and as event fields."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from orchestrator.verify_spec_shared import canonical_json

_PROSE_RUN_KEYS = frozenset({'reason'})

#: The unit module of a run that had no ModuleConfig (the global commands).
GLOBAL_MODULE = '__global__'


class CacheMode(StrEnum):
    OFF = 'off'
    LOG = 'log'
    ON = 'on'


class UnitTier(StrEnum):
    TREE = 'tree'
    MODULE = 'module'
    TEST = 'test'


class UnitOutcome(StrEnum):
    RAN_GREEN = 'ran_green'
    RAN_RED = 'ran_red'
    CACHE_HIT = 'cache_hit'
    WOULD_HIT = 'would_hit'
    UNAVAILABLE = 'unavailable'


class VerifySite(StrEnum):
    """Which workflow site emitted a ``workflow_verify`` row, and what each reader makes of it.

    Pinned by orchestrator/tests/test_workflow_verify_site_readers.py:

    - merge_lane/disposition.py::_branch_pre_merge_verify_green (I5) ignores pre-merge reds.
    - merge_lane/completion.py::merge_completion_eligible: a current-run pre-merge green
      satisfies its passed-row half.
    - stranded_verified_green.py::last_verified_green_tip: a pre-merge green at the
      post-rebase tip becomes the latest green tip, and a red is skipped.
    - verify_checkpoint.py::green_checkpoint_at_tip: a pre-merge green lets a redispatch
      at that tip skip VERIFY.
    """

    IMPLEMENT = 'implement'
    PRE_MERGE = 'pre_merge'


def counts_as_branch_verdict(data: Mapping[str, Any]) -> bool:
    """False only for a pre-merge red: it was measured on main's tip after the rebase."""
    return not (data.get('site') == VerifySite.PRE_MERGE and not data.get('passed'))


@dataclass(frozen=True)
class VerifyUnit:
    """One verified unit and how its verdict was reached; ``key`` is its cache key, when it has one."""

    unit: str
    tier: UnitTier
    key: str | None
    outcome: UnitOutcome
    evidence_event_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'unit': self.unit,
            'tier': str(self.tier),
            'key': self.key,
            'outcome': str(self.outcome),
            'evidence_event_id': self.evidence_event_id,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> VerifyUnit:
        return cls(**{**d, 'tier': UnitTier(d['tier']), 'outcome': UnitOutcome(d['outcome'])})


def module_leg_unit(module_prefix: str | None, leg: str, rc: int) -> VerifyUnit:
    """The unit for one module leg that ran and exited *rc*."""
    return VerifyUnit(
        unit=f'module:{module_prefix or GLOBAL_MODULE}:{leg}',
        tier=UnitTier.MODULE,
        key=None,
        outcome=UnitOutcome.RAN_GREEN if rc == 0 else UnitOutcome.RAN_RED,
    )


def units_from_wire(units: Iterable[VerifyUnit | Mapping[str, Any]]) -> tuple[VerifyUnit, ...]:
    """*units* as records. ``verify_runner.py::result_from_dict`` rebuilds a
    VerifyResult generically, so its units arrive as plain dicts."""
    return tuple(unit if isinstance(unit, VerifyUnit) else VerifyUnit.from_dict(unit) for unit in units)


def plan_hash(plan: Mapping[str, Any] | None) -> str | None:
    """sha256 of the plan's rendered runs minus their prose; None when no plan ran.

    Keys are denied rather than allowed, so a new execution-relevant run field
    can only make two hashes more distinct, never collide them.
    """
    if plan is None:
        return None
    runs = [
        {key: value for key, value in run.items() if key not in _PROSE_RUN_KEYS}
        for run in plan['runs']
    ]
    return hashlib.sha256(canonical_json(runs).encode()).hexdigest()


class VerifyEvidence(Protocol):
    @property
    def plan(self) -> Mapping[str, Any] | None: ...
    @property
    def cache_mode(self) -> CacheMode: ...
    @property
    def units(self) -> tuple[VerifyUnit, ...] | None: ...


def verify_event_fields(result: VerifyEvidence | None) -> dict[str, Any]:
    """The always-present slice of a verify event that *result* attests; all None without a verdict."""
    if result is None:
        return {'plan_hash': None, 'cache_mode': None, 'units': None}
    return {
        'plan_hash': plan_hash(result.plan),
        'cache_mode': str(result.cache_mode),
        'units': None if result.units is None else [unit.to_dict() for unit in result.units],
    }
