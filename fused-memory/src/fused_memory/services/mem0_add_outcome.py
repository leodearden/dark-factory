"""Read a Mem0 add, raised or not, the one way every write path reads it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fused_memory.models.enums import SourceStore


@dataclass(frozen=True)
class Mem0AddOutcome:
    memory_ids: tuple[str, ...]
    empty_result_error: str | None

    def __post_init__(self) -> None:
        if self.empty_result_error is not None and self.landed:
            raise ValueError(
                'Mem0AddOutcome: an empty_result_error is only valid with zero '
                f'memory_ids, got memory_ids={self.memory_ids!r} and '
                f'empty_result_error={self.empty_result_error!r}'
            )

    @property
    def landed(self) -> bool:
        return bool(self.memory_ids)

    @property
    def stores_written(self) -> list[SourceStore]:
        return [SourceStore.mem0] if self.landed else []


RAISED_MEM0_ADD = Mem0AddOutcome(memory_ids=(), empty_result_error=None)


@dataclass(frozen=True)
class Mem0LegResult:
    """One Mem0 write leg: what the add returned, or why it raised and whether it was deferred."""

    outcome: Mem0AddOutcome
    error: str | None = None
    pending_retry_reason: str | None = None

    def __post_init__(self) -> None:
        if self.error is not None and self.pending_retry_reason is not None:
            raise ValueError(
                'Mem0LegResult: a leg either failed or was deferred, not both, got '
                f'error={self.error!r} and pending_retry_reason={self.pending_retry_reason!r}'
            )
        if (
            self.error is not None or self.pending_retry_reason is not None
        ) and self.outcome != RAISED_MEM0_ADD:
            raise ValueError(
                'Mem0LegResult: an error or pending_retry_reason means the add raised, '
                f'so the outcome must be RAISED_MEM0_ADD, got memory_ids='
                f'{self.outcome.memory_ids!r} and empty_result_error='
                f'{self.outcome.empty_result_error!r}'
            )

    @property
    def pending_retry(self) -> bool:
        return self.pending_retry_reason is not None


def read_mem0_add_result(
    result: Mapping[str, Any] | None,
    *,
    operation: str,
    zero_ids_anomalous: bool,
) -> Mem0AddOutcome:
    memory_ids = tuple(
        r['id']
        for r in (result or {}).get('results', [])
        if isinstance(r, dict) and 'id' in r
    )
    empty_result_error = (
        f'empty_result: mem0 {operation} returned zero memory_ids'
        if not memory_ids and zero_ids_anomalous
        else None
    )
    return Mem0AddOutcome(memory_ids=memory_ids, empty_result_error=empty_result_error)
