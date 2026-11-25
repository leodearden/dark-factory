"""Whether a verify unit may replay a recorded green, and whether a green may be recorded."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final

from orchestrator.config import VerifyCacheConfig
from orchestrator.verify_cache.keys import UnitKey
from orchestrator.verify_cache.modes import CacheMode, VerifyRole
from orchestrator.verify_cache.registry import Candidate, Entry, Registry, RegistryView
from orchestrator.verify_cache.wire import typed_field, wire_fields
from orchestrator.verify_evidence import Leg

logger = logging.getLogger(__name__)

NEVER_CACHE_REFUSAL: Final = 'the module is never-cache'


class Decision(StrEnum):
    HIT = 'hit'
    MISS = 'miss'
    WOULD_HIT = 'would_hit'
    UNAVAILABLE = 'unavailable'
    DISABLED = 'disabled'


_CARRIES_ENTRY = frozenset({Decision.HIT, Decision.WOULD_HIT})


@dataclass(frozen=True)
class CacheDecision:
    """One unit's decision, the entry it would replay (HIT and WOULD_HIT only), and why."""

    decision: Decision
    entry: Entry | Candidate | None
    reason: str

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError(f'a {self.decision} decision needs a reason')
        if (self.entry is not None) != (self.decision in _CARRIES_ENTRY):
            raise ValueError(f'a {self.decision} decision cannot carry entry {self.entry!r}')


@dataclass(frozen=True)
class Unavailable:
    """What a verify's cache decision could not read (the registry, or the code identity its keys carry); *reason* says why."""

    reason: str


async def consult(source: Registry | RegistryView | None, keys: Sequence[UnitKey]) -> RegistryView | Unavailable:
    """One batched registry read for a verify's *keys*, or the view a remote
    verify was shipped, answered as is; never raises (I2)."""
    if source is None:
        return Unavailable('no registry on this host (no_registry_on_runner)')
    if isinstance(source, RegistryView):
        return source
    try:
        return await source.view(keys)
    except Exception as exc:
        logger.warning('verify_cache.consult unavailable: %s: %s', type(exc).__name__, exc)
        return Unavailable(f'{type(exc).__name__}: {exc}')


def decide(
    modes: VerifyCacheConfig,
    role: VerifyRole,
    key: UnitKey,
    view: RegistryView | Unavailable,
    *,
    never_cache: bool,
    now: datetime,
) -> CacheDecision:
    """Whether *key*'s unit replays its recorded green in *role*; never raises (I2)."""
    try:
        return _decide(modes.mode_for(role), role, key, view, never_cache=never_cache, now=now)
    except Exception as exc:
        logger.warning('verify_cache.decide failed for %s: %s: %s', key.unit_id, type(exc).__name__, exc)
        return CacheDecision(Decision.UNAVAILABLE, None, f'decision failed: {type(exc).__name__}: {exc}')


@dataclass(frozen=True)
class LegResult:
    """What one leg's run says about recording its green. Built from a CheckRun
    and its VerifyResult: ``flake_suppressed`` is ``flake_suppression is not
    None``, and ``collected`` is the number of items pytest selected to run."""

    leg: Leg
    rc: int
    timed_out: bool
    trivial: bool
    flake_suppressed: bool
    junit_tests: int | None
    collected: int | None
    never_cache: bool


@dataclass(frozen=True)
class Admission:
    admitted: bool
    reason: str

    @classmethod
    def from_wire(cls, raw: object, *, where: str = 'Admission') -> Admission:
        """The admission *raw* encodes; TypeError naming the field it breaks."""
        fields = wire_fields(raw, _ADMISSION_FIELDS, where)
        return cls(typed_field(fields, 'admitted', bool, where), typed_field(fields, 'reason', str, where))


_ADMISSION_FIELDS = frozenset(f.name for f in dataclasses.fields(Admission))


def admissible(modes: VerifyCacheConfig, role: VerifyRole, leg: LegResult) -> Admission:
    """Whether *leg*'s green may be recorded in *role*'s registry."""
    refusal = _refusal(modes.mode_for(role), role, leg)
    if refusal is not None:
        return Admission(False, refusal)
    checked = 'rc 0, not timed out, not trivial, no suppressed flake'
    if leg.leg is Leg.TEST:
        checked += f', junit reported {leg.junit_tests} of {leg.collected} collected tests'
    return Admission(True, f'clean green {leg.leg} leg: {checked}')


def _decide(
    mode: CacheMode, role: VerifyRole, key: UnitKey, view: RegistryView | Unavailable, *, never_cache: bool, now: datetime,
) -> CacheDecision:
    if mode is CacheMode.OFF:
        return _miss(f'verify cache is off for the {role} role')
    if never_cache:
        return _miss(f'{key.unit_id} belongs to a never-cache module')
    if isinstance(view, Unavailable):
        return CacheDecision(Decision.UNAVAILABLE, None, f'registry unavailable: {view.reason}')
    if view.policy.is_disabled(now):
        return CacheDecision(
            Decision.DISABLED, None,
            f'verify cache disabled until {view.policy.disabled_until}: {view.policy.reason}',
        )
    entry = view.entries.get((key.unit_id, key.key))
    if entry is None:
        unshipped = '; the shipped view was truncated, so an unshipped green may exist' if view.truncated else ''
        return _miss(f'no recorded green for {key.unit_id} at key {key.key}{unshipped}')
    replay = f'the green of {entry.unit_id} at key {entry.key} (evidence event {entry.evidence_event_id})'
    if mode is CacheMode.ON:
        return CacheDecision(Decision.HIT, entry, f'replaying {replay}')
    return CacheDecision(Decision.WOULD_HIT, entry, f'log mode: would replay {replay}')


def _miss(reason: str) -> CacheDecision:
    return CacheDecision(Decision.MISS, None, reason)


def _refusal(mode: CacheMode, role: VerifyRole, leg: LegResult) -> str | None:
    if mode is CacheMode.OFF:
        return f'verify cache is off for the {role} role'
    if leg.never_cache:
        return NEVER_CACHE_REFUSAL
    if leg.trivial:
        return 'a trivial verify ran nothing'
    if leg.rc != 0:
        return f'the leg exited {leg.rc}'
    if leg.timed_out:
        return 'the leg timed out'
    if leg.flake_suppressed:
        return 'a flake was suppressed in this verify'
    if leg.leg is not Leg.TEST:
        return None
    if leg.junit_tests is None or leg.collected is None:
        return f'test counts unknown: junit_tests={leg.junit_tests} collected={leg.collected}'
    if leg.junit_tests != leg.collected:
        return f'junit reported {leg.junit_tests} tests but {leg.collected} were collected'
    return None
