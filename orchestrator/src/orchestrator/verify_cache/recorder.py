"""What a verify result leaves for the verify-result cache, and writing it."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from orchestrator.config import VerifyCacheConfig
from orchestrator.verify_cache.keys import UnitKey
from orchestrator.verify_cache.modes import VerifyRole
from orchestrator.verify_cache.policy import LegResult
from orchestrator.verify_cache.registry import Entry
from orchestrator.verify_evidence import VerifyUnit


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
