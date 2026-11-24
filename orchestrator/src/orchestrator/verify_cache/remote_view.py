"""The view of the dispatcher's registry a remote verify carries: built on the dispatcher,
decided against on the runner, and settled on the dispatcher (plans/verify-result-cache-prd.md
Amendments A5 and A6)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any

from orchestrator.config import VerifyCacheConfig
from orchestrator.verify_cache.keys import TREE_UNIT_ID
from orchestrator.verify_cache.modes import CacheMode, VerifyRole
from orchestrator.verify_cache.registry import Candidate, KillSwitch, Registry, RegistryView
from orchestrator.verify_evidence import module_unit_ids
from orchestrator.verify_spec_shared import VerifyCommand

logger = logging.getLogger(__name__)

_VIEW_FIELDS = frozenset({'mode', 'disabled_until', 'disabled_reason', 'candidates', 'truncated'})
_CANDIDATE_FIELDS = frozenset({'unit_id', 'key', 'entry_id', 'evidence_event_id'})


@dataclass(frozen=True)
class ShippedView:
    """The mode a remote verify may act under, the kill switch, the most recent greens of
    each unit it will run, and whether the dispatcher held greens it left out."""

    mode: CacheMode
    disabled_until: datetime | None
    disabled_reason: str | None
    candidates: tuple[Candidate, ...]
    truncated: bool

    def __post_init__(self) -> None:
        KillSwitch(self.disabled_until, self.disabled_reason)

    @property
    def kill_switch(self) -> KillSwitch:
        return KillSwitch(self.disabled_until, self.disabled_reason)

    def registry_view(self) -> RegistryView:
        entries = {(candidate.unit_id, candidate.key): candidate for candidate in self.candidates}
        return RegistryView(MappingProxyType(entries), self.kill_switch, self.truncated)

    def to_wire(self) -> dict[str, Any]:
        return {
            'mode': self.mode.value,
            'disabled_until': None if self.disabled_until is None else self.disabled_until.isoformat(),
            'disabled_reason': self.disabled_reason,
            'candidates': [
                {
                    'unit_id': candidate.unit_id, 'key': candidate.key,
                    'entry_id': candidate.entry_id, 'evidence_event_id': candidate.evidence_event_id,
                }
                for candidate in self.candidates
            ],
            'truncated': self.truncated,
        }

    @classmethod
    def from_wire(cls, raw: object) -> ShippedView:
        """The view *raw* encodes; TypeError or ValueError naming the field it breaks."""
        fields = _exact_fields(raw, _VIEW_FIELDS, 'ShippedView')
        disabled_until = _typed(fields, 'disabled_until', str, 'ShippedView', optional=True)
        return cls(
            mode=_mode(_typed(fields, 'mode', str, 'ShippedView')),
            disabled_until=None if disabled_until is None else _aware(disabled_until),
            disabled_reason=_typed(fields, 'disabled_reason', str, 'ShippedView', optional=True),
            candidates=tuple(
                _candidate_from_wire(item, f'ShippedView.candidates[{index}]')
                for index, item in enumerate(_typed(fields, 'candidates', list, 'ShippedView'))
            ),
            truncated=_typed(fields, 'truncated', bool, 'ShippedView'),
        )


def planned_unit_ids(commands: Iterable[VerifyCommand]) -> tuple[str, ...]:
    """The units a remote verify of *commands* can key: every set leg of every command, and the tree."""
    return (*(unit_id for command in commands for unit_id in module_unit_ids(command.prefix, command)), TREE_UNIT_ID)


async def ship_view(
    registry: Registry,
    modes: VerifyCacheConfig,
    role: VerifyRole,
    unit_ids: Sequence[str],
    *,
    fits: Callable[[ShippedView], bool],
) -> ShippedView | None:
    """The view a remote *role* verify of *unit_ids* carries, at the deepest uniform
    per-unit depth that *fits*; None, logged, when the registry cannot be read."""
    mode = modes.remote_mode_for(role)
    if mode is CacheMode.OFF:
        return ShippedView(CacheMode.OFF, None, None, (), False)
    cap = modes.remote_candidates_per_unit
    try:
        switch = (await registry.policy_state()).kill_switch
        read = await registry.candidates(unit_ids, per_unit=cap + 1)
    except Exception as exc:
        logger.warning('verify_cache.ship_view: no view shipped for the %s role: %s: %s', role, type(exc).__name__, exc)
        return None
    by_unit: dict[str, list[Candidate]] = {}
    for candidate in read:
        by_unit.setdefault(candidate.unit_id, []).append(candidate)
    held = max((len(greens) for greens in by_unit.values()), default=0)
    deepest = min(cap, held)
    view = _at_depth(mode, switch, by_unit, deepest, truncated=held > cap)
    if fits(view):
        return view
    return _at_depth(mode, switch, by_unit, _deepest_fitting(mode, switch, by_unit, deepest, fits), truncated=True)


def _deepest_fitting(
    mode: CacheMode,
    switch: KillSwitch,
    by_unit: Mapping[str, Sequence[Candidate]],
    too_deep: int,
    fits: Callable[[ShippedView], bool],
) -> int:
    """The deepest depth below *too_deep* whose view fits, by bisection (a view only
    shrinks as its depth does); 0 when none does."""
    low, high = 0, too_deep - 1
    while low < high:
        middle = (low + high + 1) // 2
        if fits(_at_depth(mode, switch, by_unit, middle, truncated=True)):
            low = middle
        else:
            high = middle - 1
    return low


def _at_depth(
    mode: CacheMode, switch: KillSwitch, by_unit: Mapping[str, Sequence[Candidate]], depth: int, *, truncated: bool,
) -> ShippedView:
    candidates = tuple(candidate for greens in by_unit.values() for candidate in greens[:depth])
    return ShippedView(mode, switch.disabled_until, switch.reason, candidates, truncated)


def _candidate_from_wire(raw: object, where: str) -> Candidate:
    fields = _exact_fields(raw, _CANDIDATE_FIELDS, where)
    return Candidate(
        unit_id=_typed(fields, 'unit_id', str, where),
        key=_typed(fields, 'key', str, where),
        entry_id=_typed(fields, 'entry_id', int, where),
        evidence_event_id=_typed(fields, 'evidence_event_id', int, where),
    )


def _exact_fields(raw: object, expected: frozenset[str], where: str) -> Mapping[str, object]:
    if not isinstance(raw, dict):
        raise TypeError(f'{where} must be a JSON object, got {type(raw).__name__}')
    if set(raw) != expected:
        raise TypeError(
            f'{where} keys: missing {sorted(expected - set(raw))}, unexpected {sorted(set(raw) - expected)}',
        )
    return raw


def _typed(fields: Mapping[str, object], name: str, kind: type, where: str, *, optional: bool = False) -> Any:
    value = fields[name]
    if value is None and optional:
        return None
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise TypeError(f'{where}.{name} must be {kind.__name__}, got {type(value).__name__} {value!r}')
    return value


def _mode(value: str) -> CacheMode:
    try:
        return CacheMode(value)
    except ValueError:
        raise ValueError(f'ShippedView.mode {value!r} is not one of {[mode.value for mode in CacheMode]}') from None


def _aware(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError(f'ShippedView.disabled_until must be timezone-aware, got {value!r}')
    return parsed
