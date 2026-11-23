"""Builders for the verify-result cache's records: unit keys, decisions, replay plans,
the legs and units a verify ran, recorded entries, and a verify result to carry them."""

from __future__ import annotations

import hashlib
from typing import Any

from orchestrator.config import VerifyCacheConfig
from orchestrator.verify import VerifyResult
from orchestrator.verify_cache.keys import UnitKey
from orchestrator.verify_cache.modes import CacheMode
from orchestrator.verify_cache.policy import CacheDecision, Decision, LegResult
from orchestrator.verify_cache.registry import Entry, Registry
from orchestrator.verify_cache.replay import ReplayPlan, UnitDecision
from orchestrator.verify_evidence import Leg, UnitTier, VerifyUnit, module_leg_unit

MERGE_ON = VerifyCacheConfig(merge_mode=CacheMode.ON)
MERGE_LOG = VerifyCacheConfig(merge_mode=CacheMode.LOG)


def unit_key(unit_id: str, *, tier: UnitTier = UnitTier.MODULE, token: str = 'a') -> UnitKey:
    spec = hashlib.sha256(unit_id.encode()).hexdigest()
    return UnitKey.of(unit_id, tier, spec, [('spec', spec), ('tree:x', token)])


def unit_decision(decision: Decision, entry: Entry | None = None, *, key: UnitKey | None = None) -> UnitDecision:
    return UnitDecision(key, CacheDecision(decision, entry, f'{decision} because the test said so'))


def merge_replay(
    modules: dict[str, UnitDecision], *, tree: UnitDecision | None = None, modes: VerifyCacheConfig = MERGE_ON,
) -> ReplayPlan:
    """A merge-role replay plan of *modules*; the tree misses unless *tree* says otherwise."""
    tree = tree or unit_decision(Decision.MISS, key=unit_key('tree', tier=UnitTier.TREE))
    return ReplayPlan(role='merge', modes=modes, tree=tree, modules=modules)


def ran_unit(unit_id: str, rc: int = 0) -> VerifyUnit:
    module, leg = unit_id.removeprefix('module:').rsplit(':', 1)
    return module_leg_unit(module, leg, rc)


def clean_leg(leg: Leg, **overrides: Any) -> LegResult:
    """A leg every admission rule admits, but for *overrides*."""
    fields: dict[str, Any] = {
        'leg': leg, 'rc': 0, 'timed_out': False, 'trivial': False, 'flake_suppressed': False,
        'junit_tests': 2 if leg is Leg.TEST else None, 'collected': 2 if leg is Leg.TEST else None,
        'never_cache': False, **overrides,
    }
    return LegResult(**fields)


async def record_entry(registry: Registry, key: UnitKey, evidence: int = 7) -> Entry:
    return await registry.record_green(key, evidence_event_id=evidence, run_id='r0', sha='s0')


async def recorded_entries(registry: Registry, *keys: UnitKey) -> dict[str, Entry]:
    """The live entry of each of *keys*, by unit id."""
    view = await registry.view(keys)
    return {unit_id: entry for (unit_id, _key_digest), entry in view.entries.items()}


def verify_result(**overrides: Any) -> VerifyResult:
    return VerifyResult(passed=True, test_output='', lint_output='', type_output='', summary='', **overrides)
