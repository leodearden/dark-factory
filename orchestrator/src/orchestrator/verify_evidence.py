"""What a verify verdict attests: the identity of the plan it ran and the outcome
of each unit, as typed values and as event fields, and the ``workflow_verify``
row that records a branch verdict at its tip or the TipUnresolved reason it has none."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Any, Protocol

from orchestrator.verify_cache.modes import CacheMode
from orchestrator.verify_spec_shared import canonical_json

_PROSE_RUN_KEYS = frozenset({'reason'})

#: The unit module of a run that had no ModuleConfig (the global commands).
GLOBAL_MODULE = '__global__'


class UnitTier(StrEnum):
    TREE = 'tree'
    MODULE = 'module'
    TEST = 'test'


class Leg(StrEnum):
    """One check of a module; the values are ``verify.py::CheckRun`` labels."""

    TEST = 'test'
    LINT = 'lint'
    TYPE = 'type'


class UnitOutcome(StrEnum):
    RAN_GREEN = 'ran_green'
    RAN_RED = 'ran_red'
    CACHE_HIT = 'cache_hit'
    WOULD_HIT = 'would_hit'
    UNAVAILABLE = 'unavailable'


class VerifySite(StrEnum):
    """Which site emitted a ``workflow_verify`` row, and what each reader makes of it.

    Pinned by orchestrator/tests/test_workflow_verify_site_readers.py:

    - merge_lane/disposition.py::classify_merge_failure_disposition (I5) ignores pre-merge
      reds: one never erases a branch green, and alone it confirms none.
    - merge_lane/completion.py::merge_completion_eligible: a current-run pre-merge green
      satisfies its passed-row half.
    - stranded_verified_green.py::last_verified_green_tip: a pre-merge green at the
      post-rebase tip becomes the latest green tip, and a red is skipped.
    - verify_checkpoint.py::green_checkpoint_at_tip: a pre-merge green lets a redispatch
      at that tip skip VERIFY.
    - A merge_request(verified_green=True) vouch is a green at the tip the server
      resolved, and every reader counts it.
    """

    IMPLEMENT = 'implement'
    PRE_MERGE = 'pre_merge'
    MCP_MERGE_REQUEST = 'mcp:merge_request'


class TipUnresolved(Enum):
    """Why a ``workflow_verify`` row has no tip."""

    HEAD_UNREADABLE = 'head_unreadable'
    NOT_CAPTURED = 'not_captured'
    BRANCH_UNRESOLVED = 'branch_unresolved'


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


def module_leg_name(module_prefix: str | None, leg: str) -> str:
    """The unit name of one module leg, e.g. ``module:scripts:type``."""
    return f'module:{module_prefix or GLOBAL_MODULE}:{leg}'


def module_leg_unit(module_prefix: str | None, leg: str, rc: int) -> VerifyUnit:
    """The unit for one module leg that ran and exited *rc*."""
    return VerifyUnit(
        unit=module_leg_name(module_prefix, leg),
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
    runs = [_execution_spec(run) for run in plan['runs']]
    return hashlib.sha256(canonical_json(runs).encode()).hexdigest()


def run_spec_hash(run: Mapping[str, Any]) -> str:
    """sha256 of one rendered run minus its prose, by the same deny-list as :func:`plan_hash`."""
    return hashlib.sha256(canonical_json(_execution_spec(run)).encode()).hexdigest()


def _execution_spec(run: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if key not in _PROSE_RUN_KEYS}


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


class TimedVerifyEvidence(VerifyEvidence, Protocol):
    """VerifyEvidence plus the run's own wall time and admission wait, which a workflow_verify row records."""

    @property
    def duration_secs(self) -> float: ...
    @property
    def slot_wait_secs(self) -> float | None: ...


def workflow_verify_event_data(
    passed: bool,
    *,
    tip_sha: str | TipUnresolved,
    base_sha: str | None,
    branch: str,
    result: TimedVerifyEvidence | None,
    site: VerifySite,
    runner: str,
) -> dict[str, Any]:
    """The single constructor of a ``workflow_verify`` row's data, for every emitter."""
    unresolved = isinstance(tip_sha, TipUnresolved)
    if not unresolved and not tip_sha:
        raise ValueError(
            'a workflow_verify tip_sha is a commit sha or a TipUnresolved reason, never '
            f'empty: got tip_sha={tip_sha!r} for branch={branch!r} site={site}',
        )
    return {
        'passed': passed,
        'tip_sha': None if unresolved else tip_sha,
        'tip_sha_unresolved': unresolved,
        'tip_sha_unresolved_reason': tip_sha.value if unresolved else None,
        'base_sha': base_sha,
        'branch': branch,
        'duration_ms': None if result is None else round(result.duration_secs * 1000),
        'slot_wait_secs': None if result is None else result.slot_wait_secs,
        'site': str(site),
        'runner': runner,
        **verify_event_fields(result),
    }
