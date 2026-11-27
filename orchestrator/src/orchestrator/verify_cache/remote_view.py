"""The view of the dispatcher's registry a remote verify carries: built on the dispatcher,
decided against on the runner, and settled on the dispatcher (plans/verify-result-cache-prd.md
Amendments A5 and A6)."""

from __future__ import annotations

import dataclasses
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
    CitedReplay,
    ConsumeOutcome,
    ReportedUnit,
    consume_replayed,
)
from orchestrator.verify_cache.registry import Candidate, KillSwitch, Registry, RegistryView
from orchestrator.verify_cache.wire import optional_field, typed_field, wire_fields
from orchestrator.verify_evidence import (
    Leg,
    UnitOutcome,
    VerifyUnit,
    module_leg_name,
    module_unit_ids,
)
from orchestrator.verify_spec_shared import VerifyCommand

logger = logging.getLogger(__name__)

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


_VIEW_FIELDS = frozenset(f.name for f in dataclasses.fields(ShippedView))


def planned_unit_ids(commands: Iterable[VerifyCommand]) -> tuple[str, ...]:
    """The units a remote verify of *commands* can key: every set leg of every command, and the tree."""
    return (*(unit_id for command in commands for unit_id in module_unit_ids(command.prefix, command)), TREE_UNIT_ID)


def never_cache_unit_ids(commands: Iterable[VerifyCommand]) -> frozenset[str]:
    """Every leg's unit id, set or not, of each never-cache command among *commands*."""
    return frozenset(
        module_leg_name(command.prefix, leg) for command in commands if command.verify_cache == 'never' for leg in Leg
    )


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
        never_cache: frozenset[str],
        landing_sha: str | None,
    ) -> RemoteSettlement:
        """Back every replay among *units* against the live registry, consuming the backing
        entries for *landing_sha* when there is one (I6), and rebuild the record of *report*
        under the mode *view* shipped. This is also the dispatcher's own check of the
        never-cache rule: a unit of *never_cache*, and the tree beside any the verify
        settled (the set the runner decided over), neither replays nor records, whatever
        the runner reports. Never raises; the caller maps the outcome to a verdict."""
        in_verify = never_cache & _settled_unit_ids(units, report)
        record = None if view is None or report is None else report.record(
            role, self.modes.with_mode(role, view.mode), never_cache=in_verify,
        )
        claims = _claims(units, report)
        refused = _refused(view, role, claims, in_verify)
        if refused is not None:
            return RemoteSettlement(record, refused)
        if record is None:
            return RemoteSettlement(None, ConsumeOutcome((), (), None, 'no replayed unit to consume'))
        return RemoteSettlement(record, await consume_replayed(self.registry, record, landing_sha=landing_sha))


@dataclass(frozen=True)
class _Claim:
    """A unit the verdict replays, and what the runner's report says of it; None when it says nothing."""

    verdict: VerifyUnit
    reported: ReportedUnit | None

    @property
    def citation(self) -> Candidate | None:
        """The green the report cites for this replay; None when it cites none."""
        if self.reported is None or self.reported.unit.outcome is not UnitOutcome.CACHE_HIT:
            return None
        return self.reported.entry

    @property
    def cites_its_own_unit(self) -> bool:
        """Whether the cited green is this unit's own, or the tree's that replays every unit."""
        citation = self.citation
        return citation is not None and citation.unit_id in (self.verdict.unit, TREE_UNIT_ID)

    @property
    def report_agrees(self) -> bool:
        """Whether the report names the evidence event the verdict does (the one its event row records)."""
        return self.reported is not None and self.reported.unit.evidence_event_id == self.verdict.evidence_event_id

    @property
    def replay(self) -> CitedReplay:
        citation = self.citation
        return CitedReplay(self.verdict.unit, None if citation is None else (citation.unit_id, citation.key))


def _reported_units(report: CacheReport | None) -> tuple[ReportedUnit, ...]:
    if report is None:
        return ()
    return (*report.units, *(() if report.tree is None else (report.tree,)))


def _settled_unit_ids(units: Sequence[VerifyUnit] | None, report: CacheReport | None) -> frozenset[str]:
    """Every unit the runner settled, in its verdict or its report."""
    return frozenset({unit.unit for unit in units or ()} | {unit.unit.unit for unit in _reported_units(report)})


def _claims(units: Sequence[VerifyUnit] | None, report: CacheReport | None) -> tuple[_Claim, ...]:
    """Each unit the verdict replays, with what the runner's report says of it."""
    reported = {unit.unit.unit: unit for unit in _reported_units(report)}
    return tuple(
        _Claim(unit, reported.get(unit.unit)) for unit in units or () if unit.outcome is UnitOutcome.CACHE_HIT
    )


def _refused(
    view: ShippedView | None, role: VerifyRole, claims: Sequence[_Claim], never_cache: frozenset[str],
) -> ConsumeOutcome | None:
    """Why no replay among *claims* can be backed, whatever the registry holds; None when each may be."""
    if not claims:
        return None
    replays = [claim.replay for claim in claims]
    never = [
        replay for replay in replays
        if replay.unit_id in never_cache or (never_cache and replay.unit_id == TREE_UNIT_ID)
    ]
    if never:
        return ConsumeOutcome.refused(never, 'a never-cache module cannot replay')
    if view is None:
        return ConsumeOutcome.refused(replays, f'no view was shipped to the {role} verify, so it could replay nothing')
    if view.mode is not CacheMode.ON:
        return ConsumeOutcome.refused(replays, f'the {role} view was shipped in {view.mode} mode, which replays nothing')
    uncited = [claim.replay for claim in claims if claim.citation is None]
    if uncited:
        return ConsumeOutcome.refused(uncited, 'the runner reported no recorded green for the replay of')
    foreign = [claim.replay for claim in claims if not claim.cites_its_own_unit]
    if foreign:
        return ConsumeOutcome.refused(foreign, "the runner cited another unit's green for the replay of")
    disagreeing = [claim.replay for claim in claims if not claim.report_agrees]
    if disagreeing:
        return ConsumeOutcome.refused(
            disagreeing, 'the runner reported another evidence event than its verdict for the replay of',
        )
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
