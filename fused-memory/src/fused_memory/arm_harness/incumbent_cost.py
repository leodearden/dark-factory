"""The incumbent's measured LLM spend: production per-write telemetry and the controls' replay unit cost.

Provenance of the committed figures: plans/local-memory-models-eval-llm/README.md.
"""

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from itertools import pairwise
from typing import Self

from pydantic import AwareDatetime, Field, TypeAdapter, model_validator

from fused_memory.arm_harness.frozen_model import FrozenModel

GRAPHITI_BACKEND = 'graphiti'
_TOKEN_COLUMNS = ('input_tokens', 'output_tokens', 'total_tokens', 'llm_calls')
_AWARE_DATETIME: TypeAdapter[datetime] = TypeAdapter(AwareDatetime)

_StampedRow = tuple[datetime, Mapping[str, object]]


class LlmWriteOperation(StrEnum):
    ADD_MEMORY = 'add_memory'
    ADD_EPISODE = 'add_episode'


_LLM_WRITE_OPERATIONS = frozenset(operation.value for operation in LlmWriteOperation)


class TelemetryWindowError(ValueError):
    """The telemetry cannot bound a measurement window, so nothing is costed."""


class TelemetryAccountingError(ValueError):
    """An LLM-bearing write inside the window has no token record: its cost is unknown, not zero."""


class LlmWriteTelemetry(FrozenModel):
    created_at: AwareDatetime
    operation: LlmWriteOperation
    project_id: str = Field(min_length=1)
    success: bool
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    llm_calls: int = Field(ge=0)

    @classmethod
    def from_telemetry_row(cls, row: Mapping[str, object]) -> Self:
        """Parse one ``telemetry_query.py`` row, refusing a total that is not input + output."""
        write = cls.model_validate({field: row.get(field) for field in cls.model_fields})
        total = row.get('total_tokens')
        if total != write.input_tokens + write.output_tokens:
            raise ValueError(
                f'telemetry row at {row.get("created_at")}: total_tokens {total} != '
                f'input_tokens {write.input_tokens} + output_tokens {write.output_tokens}'
            )
        return write


class TelemetryWindow(FrozenModel):
    """The LLM writes from the first token-bearing one up to, not including, ``end``."""

    start: AwareDatetime
    end: AwareDatetime
    writes: tuple[LlmWriteTelemetry, ...]

    @model_validator(mode='after')
    def _bounds_its_writes(self) -> Self:
        if self.end <= self.start:
            raise ValueError(
                f'window end {self.end.isoformat()} is not after its start {self.start.isoformat()}'
            )
        if not self.writes:
            raise ValueError(f'window [{self.start.isoformat()}, {self.end.isoformat()}) is empty')
        self._require_sorted()
        self._require_start_is_first_write()
        self._require_writes_before_end()
        return self

    def _require_sorted(self) -> None:
        for earlier, later in pairwise(self.writes):
            if later.created_at < earlier.created_at:
                raise ValueError(
                    f'window writes are not sorted by created_at: {later.created_at.isoformat()} '
                    f'follows {earlier.created_at.isoformat()}'
                )

    def _require_start_is_first_write(self) -> None:
        first = self.writes[0].created_at
        if self.start != first:
            raise ValueError(
                f"window start {self.start.isoformat()} is not its first write's created_at "
                f'{first.isoformat()}'
            )

    def _require_writes_before_end(self) -> None:
        late = [write.created_at for write in self.writes if write.created_at >= self.end]
        if late:
            raise ValueError(
                f'{len(late)} write(s) at or after the window end {self.end.isoformat()}, '
                f'the first at {late[0].isoformat()}'
            )


def select_llm_writes(rows: Iterable[Mapping[str, object]], *, until: datetime) -> TelemetryWindow:
    """The window of graphiti LLM writes from where token telemetry began up to ``until``."""
    if until.tzinfo is None:
        raise TelemetryWindowError(f'until {until.isoformat()} is naive; the journal stamps UTC')
    stamped = _llm_writes_by_time(rows)
    start = _first_tokened_at(stamped, until)
    _require_coverage(stamped, start)
    in_window = [(moment, row) for moment, row in stamped if start <= moment < until]
    _require_accounted(in_window)
    return TelemetryWindow(
        start=start,
        end=until,
        writes=tuple(LlmWriteTelemetry.from_telemetry_row(row) for _, row in in_window),
    )


def _llm_writes_by_time(rows: Iterable[Mapping[str, object]]) -> list[_StampedRow]:
    stamped = [
        (_AWARE_DATETIME.validate_python(row.get('created_at')), row)
        for row in rows
        if row.get('backend') == GRAPHITI_BACKEND and row.get('operation') in _LLM_WRITE_OPERATIONS
    ]
    return sorted(stamped, key=lambda pair: pair[0])


def _has_any_tokens(row: Mapping[str, object]) -> bool:
    return any(row.get(column) is not None for column in _TOKEN_COLUMNS)


def _first_tokened_at(stamped: Sequence[_StampedRow], until: datetime) -> datetime:
    for moment, row in stamped:
        if moment < until and _has_any_tokens(row):
            return moment
    raise TelemetryWindowError(
        f'no token-bearing LLM write lies before until {until.isoformat()}: there is no window'
    )


def _require_coverage(stamped: Sequence[_StampedRow], start: datetime) -> None:
    if not any(moment < start for moment, _ in stamped):
        raise TelemetryWindowError(
            f'no untokened LLM write precedes the first token-bearing one at {start.isoformat()}: '
            'the telemetry does not reach back past where token recording began'
        )


def _require_accounted(in_window: Sequence[_StampedRow]) -> None:
    unaccounted = [
        moment
        for moment, row in in_window
        if any(row.get(column) is None for column in _TOKEN_COLUMNS)
    ]
    if unaccounted:
        raise TelemetryAccountingError(
            f'{len(unaccounted)} LLM write(s) in the window lack a complete token record, the '
            f'first at {unaccounted[0].isoformat()}; their tokens are unknown, not zero'
        )
