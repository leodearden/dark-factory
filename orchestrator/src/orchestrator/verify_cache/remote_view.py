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
from orchestrator.verify_cache.recorder import (
    CacheRecord,
    CacheReport,
    ConsumeOutcome,
    UnbackedReplay,
    consume_replayed,
)
from orchestrator.verify_cache.registry import Candidate, KillSwitch, Registry, RegistryView
from orchestrator.verify_cache.wire import optional_field, typed_field, wire_fields
from orchestrator.verify_evidence import UnitOutcome, VerifyUnit, module_unit_ids
from orchestrator.verify_spec_shared import VerifyCommand

logger = logging.getLogger(__name__)

_VIEW_FIELDS = frozenset({'mode', 'disabled_until', 'disabled_reason', 'candidates', 'truncated'})


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
            'candidates': [candidate.to_wire() for candidate in self.candidates],
            'truncated': self.truncated,
        }

    @classmethod
    def from_wire(cls, raw: object) -> ShippedView:
        """The view *raw* encodes; TypeError or ValueError naming the field it breaks."""
        fields = wire_fields(raw, _VIEW_FIELDS, 'ShippedView')
        disabled_until = optional_field(fields, 'disabled_until', str, 'ShippedView')
        return cls(
            mode=_mode(typed_field(fields, 'mode', str, 'ShippedView')),
            disabled_until=None if disabled_until is None else _aware(disabled_until),
            disabled_reason=optional_field(fields, 'disabled_reason', str, 'ShippedView'),
            candidates=tuple(
                Candidate.from_wire(item, where=f'ShippedView.candidates[{index}]')
                for index, item in enumerate(typed_field(fields, 'candidates', list, 'ShippedView'))
            ),
            truncated=typed_field(fields, 'truncated', bool, 'ShippedView'),
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


@dataclass(frozen=True)
class RemoteSettlement:
    """What a remote verify leaves its dispatcher: the record its greens are recorded
    from (None when there is none to record), and whether a live entry backs every replay."""

    record: CacheRecord | None
    outcome: ConsumeOutcome


@dataclass(frozen=True)
class RemoteCache:
    """The dispatcher's half of a remote verify's cache: it ships a view of *registry*
    under *modes*, and settles what the runner reports against the live registry."""

    registry: Registry
    modes: VerifyCacheConfig

    @classmethod
    def of(cls, registry: Registry | None, modes: VerifyCacheConfig) -> RemoteCache | None:
        return None if registry is None else cls(registry, modes)

    async def ship(
        self, role: VerifyRole, unit_ids: Sequence[str], *, fits: Callable[[ShippedView], bool],
    ) -> ShippedView | None:
        return await ship_view(self.registry, self.modes, role, unit_ids, fits=fits)

    async def settle(
        self,
        view: ShippedView | None,
        role: VerifyRole,
        units: Sequence[VerifyUnit] | None,
        report: CacheReport | None,
        *,
        landing_sha: str | None,
    ) -> RemoteSettlement:
        """Back every replay among *units* against the live registry, consuming the backing
        entries for *landing_sha* when there is one (I6), and rebuild the record of *report*
        under the mode *view* shipped. Never raises; the caller maps the outcome to a verdict."""
        record = None if view is None or report is None else report.record(role, self.modes.with_mode(role, view.mode))
        claimed = _claimed_replays(units, report)
        refused = _refused(view, role, claimed)
        if refused is not None:
            return RemoteSettlement(record, refused)
        if record is None:
            return RemoteSettlement(None, ConsumeOutcome((), (), None, 'no replayed unit to consume'))
        return RemoteSettlement(record, await consume_replayed(self.registry, record, landing_sha=landing_sha))


def _claimed_replays(units: Sequence[VerifyUnit] | None, report: CacheReport | None) -> tuple[UnbackedReplay, ...]:
    """Each unit the verdict replays, with the entry the runner's report cites for it."""
    reported = {} if report is None else {
        unit.unit.unit: unit for unit in (*report.units, *(() if report.tree is None else (report.tree,)))
    }

    def cited(unit_id: str) -> tuple[str, str] | None:
        unit = reported.get(unit_id)
        if unit is None or unit.unit.outcome is not UnitOutcome.CACHE_HIT or unit.entry is None:
            return None
        return (unit.entry.unit_id, unit.entry.key)

    return tuple(
        UnbackedReplay(unit.unit, cited(unit.unit)) for unit in units or () if unit.outcome is UnitOutcome.CACHE_HIT
    )


def _refused(view: ShippedView | None, role: VerifyRole, claimed: Sequence[UnbackedReplay]) -> ConsumeOutcome | None:
    """Why no replay among *claimed* can be backed, whatever the registry holds; None when each may be."""
    if not claimed:
        return None
    if view is None:
        return ConsumeOutcome.refused(claimed, f'no view was shipped to the {role} verify, so it could replay nothing')
    if view.mode is not CacheMode.ON:
        return ConsumeOutcome.refused(claimed, f'the {role} view was shipped in {view.mode} mode, which replays nothing')
    uncited = [replay for replay in claimed if replay.cited is None]
    if uncited:
        return ConsumeOutcome.refused(uncited, 'the runner reported no recorded green for the replay of')
    return None


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
