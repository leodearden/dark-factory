"""What a verify result leaves for the verify-result cache, and writing it."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterable
from dataclasses import dataclass

from orchestrator.config import VerifyCacheConfig
from orchestrator.verify_cache import policy
from orchestrator.verify_cache.keys import UnitKey
from orchestrator.verify_cache.modes import CacheMode, VerifyRole
from orchestrator.verify_cache.policy import Admission, LegResult
from orchestrator.verify_cache.registry import Entry, Registry
from orchestrator.verify_evidence import UnitOutcome, VerifyUnit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LegRun:
    """One leg that ran, under the unit id it verifies."""

    unit_id: str
    leg: LegResult


@dataclass(frozen=True)
class RecordedUnit:
    """A settled unit, its key, the leg that ran it, and the entry it replayed or would have."""

    unit: VerifyUnit
    key: UnitKey | None
    leg: LegResult | None
    entry: Entry | None


@dataclass(frozen=True)
class CacheRecord:
    """Everything a verify leaves for recording its greens: the legs that ran, and once settled, its units."""

    role: VerifyRole
    modes: VerifyCacheConfig
    legs: tuple[LegRun, ...]
    units: tuple[RecordedUnit, ...] = ()
    tree: RecordedUnit | None = None

    @classmethod
    def concat(cls, records: Iterable[CacheRecord | None]) -> CacheRecord | None:
        """The legs of *records* as one record; None when any is None, or there are none."""
        collected = list(records)
        if not collected or any(record is None for record in collected):
            return None
        present = [record for record in collected if record is not None]
        first = present[0]
        return cls(
            role=first.role,
            modes=first.modes,
            legs=tuple(leg for record in present for leg in record.legs),
            units=tuple(unit for record in present for unit in record.units),
        )


@dataclass(frozen=True)
class VerifyEvidenceRef:
    """The verify event a recorded green cites: its row id and the facts that row records."""

    event_id: int
    run_id: str
    sha: str
    passed: bool


@dataclass(frozen=True)
class RecordOutcome:
    recorded: tuple[Entry, ...]
    reason: str


@dataclass(frozen=True)
class ConsumeOutcome:
    """The entries consumed for a landing, the replayed units no live entry backs, and any registry error."""

    consumed: tuple[Entry, ...]
    unbacked: tuple[str, ...]
    error: str | None
    reason: str

    @property
    def backs_every_hit(self) -> bool:
        return self.error is None and not self.unbacked


async def record_greens(
    registry: Registry, record: CacheRecord, *, flake_suppressed: bool, evidence: VerifyEvidenceRef | None,
) -> RecordOutcome:
    """Record every admitted green of *record*, citing *evidence*; never raises."""
    if evidence is None:
        return RecordOutcome((), 'no verify evidence event id: the store returned no row for a green to cite')
    admissions = [(unit, _admission(record, unit, flake_suppressed=flake_suppressed)) for unit in record.units]
    keys = [unit.key for unit, admission in admissions if admission.admitted and unit.key is not None]
    tree = _tree_admission(record, admissions, evidence)
    if tree.admitted and record.tree is not None and record.tree.key is not None:
        keys.append(record.tree.key)
    try:
        recorded = tuple([
            await registry.record_green(key, evidence_event_id=evidence.event_id, run_id=evidence.run_id, sha=evidence.sha)
            for key in keys
        ])
    except Exception as exc:
        logger.warning('verify_cache.record_greens failed: %s: %s', type(exc).__name__, exc)
        return RecordOutcome((), f'registry unwritable: {type(exc).__name__}: {exc}')
    refused = '; '.join(f'{unit.unit.unit}: {admission.reason}' for unit, admission in admissions if not admission.admitted)
    return RecordOutcome(recorded, f'recorded {len(recorded)}; refused [{refused}]; tree: {tree.reason}')


async def consume_replayed(registry: Registry, record: CacheRecord, *, landing_sha: str) -> ConsumeOutcome:
    """Write the live entry of every replayed unit of *record* as consumed for *landing_sha* (I6); never raises."""
    hits = [unit for unit in (*record.units, *_tree_of(record)) if unit.unit.outcome is UnitOutcome.CACHE_HIT]
    if not hits:
        return ConsumeOutcome((), (), None, 'no replayed unit to consume')
    cited = {(entry.unit_id, entry.key): entry for unit in hits if (entry := unit.entry) is not None}
    try:
        live = (await registry.view([_entry_key(entry) for entry in cited.values()])).entries
        unbacked = tuple(
            unit.unit.unit for unit in hits if unit.entry is None or (unit.entry.unit_id, unit.entry.key) not in live
        )
        if unbacked:
            return ConsumeOutcome((), unbacked, None, f'no live entry backs the replay of {", ".join(unbacked)}')
        consumed = tuple(live[cited_key] for cited_key in cited)
        await registry.consume(landing_sha, consumed)
    except Exception as exc:
        logger.warning('verify_cache.consume_replayed failed: %s: %s', type(exc).__name__, exc)
        error = f'{type(exc).__name__}: {exc}'
        return ConsumeOutcome((), (), error, f'registry unreadable: {error}')
    return ConsumeOutcome(consumed, (), None, f'consumed {len(consumed)} entries for {landing_sha}')


def _admission(record: CacheRecord, unit: RecordedUnit, *, flake_suppressed: bool) -> Admission:
    if unit.unit.outcome is UnitOutcome.CACHE_HIT:
        return Admission(False, 'a replay is not new evidence')
    if unit.leg is None:
        return Admission(False, 'no leg ran for this unit')
    return policy.admissible(record.modes, record.role, dataclasses.replace(unit.leg, flake_suppressed=flake_suppressed))


def _tree_admission(
    record: CacheRecord, admissions: list[tuple[RecordedUnit, Admission]], evidence: VerifyEvidenceRef,
) -> Admission:
    blocking = [
        unit.unit.unit for unit, admission in admissions
        if not admission.admitted and unit.unit.outcome is not UnitOutcome.CACHE_HIT
    ]
    if record.modes.mode_for(record.role) is CacheMode.OFF:
        return Admission(False, f'verify cache is off for the {record.role} role')
    if record.tree is None or record.tree.key is None:
        return Admission(False, 'the tree unit has no key')
    if record.tree.unit.outcome is UnitOutcome.CACHE_HIT:
        return Admission(False, 'the tree green was replayed')
    if not evidence.passed:
        return Admission(False, 'the verify verdict did not pass')
    if not admissions:
        return Admission(False, 'no unit ran')
    if blocking:
        return Admission(False, f'units neither admitted nor replayed: {", ".join(blocking)}')
    return Admission(True, 'a passing verdict whose every unit was admitted or replayed')


def _tree_of(record: CacheRecord) -> tuple[RecordedUnit, ...]:
    return () if record.tree is None else (record.tree,)


def _entry_key(entry: Entry) -> UnitKey:
    return UnitKey(entry.unit_id, entry.tier, entry.key, entry.spec_hash, entry.inputs)
