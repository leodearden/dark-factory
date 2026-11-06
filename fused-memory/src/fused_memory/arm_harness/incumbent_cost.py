"""The incumbent's measured LLM spend: production per-attempt telemetry and the controls' replay unit cost.

Provenance of the committed figures: plans/local-memory-models-eval-llm/README.md.
"""

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, date, datetime
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Literal, Self, TypeVar

from pydantic import AwareDatetime, Field, TypeAdapter, ValidationError, model_validator
from shared.memory_eval_metrics import canonical_json_text

from fused_memory.arm_harness.arm_spec import ArmId, LlmArmSpec, TokenPricing
from fused_memory.arm_harness.frozen_model import FrozenModel
from fused_memory.arm_harness.metrics_record import LlmMetricId, MetricsRecord

GRAPHITI_BACKEND = 'graphiti'
PROJECTION_DAYS = 30
INCUMBENT_COST_SCHEMA_VERSION = 1
INCUMBENT_COST_FILENAME = 'incumbent-cost.json'
PRODUCTION_TELEMETRY_FILENAME = 'production-telemetry.jsonl'
_SECONDS_PER_DAY = 86400
_TOKEN_COLUMNS = ('input_tokens', 'output_tokens', 'total_tokens', 'llm_calls')
_AWARE_DATETIME: TypeAdapter[datetime] = TypeAdapter(AwareDatetime)

_StampedRow = tuple[datetime, Mapping[str, object]]
_GroupKey = TypeVar('_GroupKey', date, str)


class LlmWriteOperation(StrEnum):
    ADD_MEMORY = 'add_memory'
    ADD_EPISODE = 'add_episode'


_LLM_WRITE_OPERATIONS = frozenset(operation.value for operation in LlmWriteOperation)


class TelemetryRowError(ValueError):
    """A telemetry row cannot be parsed: the dump is malformed."""


class TelemetryWindowError(ValueError):
    """The telemetry cannot bound a measurement window, so nothing is costed."""


class TelemetryAccountingError(ValueError):
    """An LLM attempt inside the window has no token record: its cost is unknown, not zero."""


class IncumbentCostError(ValueError):
    """The inputs cannot yield a measured incumbent cost, so none is written."""


class LlmAttemptTelemetry(FrozenModel):
    """One graphiti attempt at an LLM-bearing write; a retried write yields one per attempt."""

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
        where = f'telemetry row at {row.get("created_at")}'
        try:
            attempt = cls.model_validate({field: row.get(field) for field in cls.model_fields})
        except ValidationError as error:
            raise TelemetryRowError(f'{where}: {error}') from error
        total = row.get('total_tokens')
        if total != attempt.input_tokens + attempt.output_tokens:
            raise TelemetryRowError(
                f'{where}: total_tokens {total} != input_tokens {attempt.input_tokens} + '
                f'output_tokens {attempt.output_tokens}'
            )
        return attempt


class TelemetryWindow(FrozenModel):
    """The LLM attempts from the first token-bearing one up to, not including, ``end``."""

    start: AwareDatetime
    end: AwareDatetime
    attempts: tuple[LlmAttemptTelemetry, ...]

    @model_validator(mode='after')
    def _bounds_its_attempts(self) -> Self:
        if self.end <= self.start:
            raise ValueError(
                f'window end {self.end.isoformat()} is not after its start {self.start.isoformat()}'
            )
        if not self.attempts:
            raise ValueError(f'window [{self.start.isoformat()}, {self.end.isoformat()}) is empty')
        self._require_sorted()
        self._require_start_is_first_attempt()
        self._require_attempts_before_end()
        return self

    def _require_sorted(self) -> None:
        for earlier, later in pairwise(self.attempts):
            if later.created_at < earlier.created_at:
                raise ValueError(
                    f'window attempts are not sorted by created_at: '
                    f'{later.created_at.isoformat()} follows {earlier.created_at.isoformat()}'
                )

    def _require_start_is_first_attempt(self) -> None:
        first = self.attempts[0].created_at
        if self.start != first:
            raise ValueError(
                f"window start {self.start.isoformat()} is not its first attempt's created_at "
                f'{first.isoformat()}'
            )

    def _require_attempts_before_end(self) -> None:
        late = [attempt.created_at for attempt in self.attempts if attempt.created_at >= self.end]
        if late:
            raise ValueError(
                f'{len(late)} attempt(s) at or after the window end {self.end.isoformat()}, '
                f'the first at {late[0].isoformat()}'
            )


def select_llm_attempts(
    rows: Iterable[Mapping[str, object]], *, until: datetime
) -> TelemetryWindow:
    """The window of graphiti LLM attempts from where token telemetry began up to ``until``."""
    if until.tzinfo is None:
        raise TelemetryWindowError(f'until {until.isoformat()} is naive; the journal stamps UTC')
    stamped = _by_time(rows)
    _require_coverage_to(stamped, until)
    llm_attempts = [(moment, row) for moment, row in stamped if _is_llm_attempt(row)]
    start = _first_tokened_at(llm_attempts, until)
    _require_coverage_from(llm_attempts, start)
    in_window = [(moment, row) for moment, row in llm_attempts if start <= moment < until]
    _require_accounted(in_window)
    return TelemetryWindow(
        start=start,
        end=until,
        attempts=tuple(LlmAttemptTelemetry.from_telemetry_row(row) for _, row in in_window),
    )


def _by_time(rows: Iterable[Mapping[str, object]]) -> list[_StampedRow]:
    return sorted(((_created_at(row), row) for row in rows), key=lambda pair: pair[0])


def _created_at(row: Mapping[str, object]) -> datetime:
    try:
        return _AWARE_DATETIME.validate_python(row.get('created_at'))
    except ValidationError as error:
        raise TelemetryRowError(
            f'telemetry row created_at {row.get("created_at")!r}: {error}'
        ) from error


def _is_llm_attempt(row: Mapping[str, object]) -> bool:
    return row.get('backend') == GRAPHITI_BACKEND and row.get('operation') in _LLM_WRITE_OPERATIONS


def _has_any_tokens(row: Mapping[str, object]) -> bool:
    return any(row.get(column) is not None for column in _TOKEN_COLUMNS)


def _require_coverage_to(stamped: Sequence[_StampedRow], until: datetime) -> None:
    if not stamped:
        raise TelemetryWindowError(
            f'the telemetry holds no row, so it cannot reach until {until.isoformat()}'
        )
    latest = stamped[-1][0]
    if latest < until:
        raise TelemetryWindowError(
            f'the telemetry ends at {latest.isoformat()}, before until {until.isoformat()}: '
            'the window would count days the dump never saw'
        )


def _first_tokened_at(llm_attempts: Sequence[_StampedRow], until: datetime) -> datetime:
    for moment, row in llm_attempts:
        if moment < until and _has_any_tokens(row):
            return moment
    raise TelemetryWindowError(
        f'no token-bearing LLM attempt lies before until {until.isoformat()}: there is no window'
    )


def _require_coverage_from(llm_attempts: Sequence[_StampedRow], start: datetime) -> None:
    if not any(moment < start for moment, _ in llm_attempts):
        raise TelemetryWindowError(
            f'no untokened LLM attempt precedes the first token-bearing one at '
            f'{start.isoformat()}: the telemetry does not reach back past where token '
            'recording began'
        )


def _require_accounted(in_window: Sequence[_StampedRow]) -> None:
    unaccounted = [
        moment
        for moment, row in in_window
        if any(row.get(column) is None for column in _TOKEN_COLUMNS)
    ]
    if unaccounted:
        raise TelemetryAccountingError(
            f'{len(unaccounted)} LLM attempt(s) in the window lack a complete token record, the '
            f'first at {unaccounted[0].isoformat()}; their tokens are unknown, not zero'
        )


class DailySpend(FrozenModel):
    day: date
    attempts: int = Field(gt=0)
    usd: float = Field(ge=0)


class ProjectSpend(FrozenModel):
    project_id: str = Field(min_length=1)
    attempts: int = Field(gt=0)
    usd: float = Field(ge=0)


class ProductionCost(FrozenModel):
    """The window's LLM spend; failed attempts count, because their tokens were spent."""

    window_start: AwareDatetime
    window_end: AwareDatetime
    days: float = Field(gt=0)
    attempts: int = Field(gt=0)
    failed_attempts: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    usd: float = Field(ge=0)
    usd_per_ok_attempt: float
    tokens_per_attempt: float
    usd_per_day: float
    projected_usd_per_30_days: float
    by_day: tuple[DailySpend, ...]
    by_project: tuple[ProjectSpend, ...]

    @model_validator(mode='after')
    def _carries_its_derivation(self) -> Self:
        if self.failed_attempts >= self.attempts:
            raise ValueError(
                f'failed_attempts {self.failed_attempts} leaves no ok attempt among attempts '
                f'{self.attempts}'
            )
        _require_derived(
            'days', self.days, 'window length', _days_between(self.window_start, self.window_end)
        )
        _require_derived(
            'by_day attempts', sum(d.attempts for d in self.by_day), 'attempts', self.attempts
        )
        _require_derived(
            'by_project attempts', sum(p.attempts for p in self.by_project), 'attempts', self.attempts
        )
        _require_derived(
            'usd_per_ok_attempt',
            self.usd_per_ok_attempt,
            'usd / (attempts - failed_attempts)',
            self.usd / (self.attempts - self.failed_attempts),
        )
        _require_derived(
            'tokens_per_attempt',
            self.tokens_per_attempt,
            '(input_tokens + output_tokens) / attempts',
            (self.input_tokens + self.output_tokens) / self.attempts,
        )
        _require_derived('usd_per_day', self.usd_per_day, 'usd / days', self.usd / self.days)
        _require_derived(
            'projected_usd_per_30_days',
            self.projected_usd_per_30_days,
            f'usd_per_day * {PROJECTION_DAYS}',
            self.usd_per_day * PROJECTION_DAYS,
        )
        return self


def _require_derived(field: str, value: float, derivation: str, derived: float) -> None:
    if value != derived:
        raise ValueError(f'{field} {value} != {derivation} {derived}')


def _days_between(start: datetime, end: datetime) -> float:
    return (end - start).total_seconds() / _SECONDS_PER_DAY


def _usd(attempt: LlmAttemptTelemetry, pricing: TokenPricing) -> float:
    return pricing.usd_for(attempt.input_tokens, attempt.output_tokens)


def production_cost(window: TelemetryWindow, pricing: TokenPricing) -> ProductionCost:
    attempts = window.attempts
    ok_attempts = sum(1 for attempt in attempts if attempt.success)
    if ok_attempts == 0:
        raise IncumbentCostError(
            f'the window from {window.start.isoformat()} holds {len(attempts)} attempt(s) and '
            'no ok attempt to cost'
        )
    usd = sum(_usd(attempt, pricing) for attempt in attempts)
    days = _days_between(window.start, window.end)
    usd_per_day = usd / days
    input_tokens = sum(attempt.input_tokens for attempt in attempts)
    output_tokens = sum(attempt.output_tokens for attempt in attempts)
    return ProductionCost(
        window_start=window.start,
        window_end=window.end,
        days=days,
        attempts=len(attempts),
        failed_attempts=len(attempts) - ok_attempts,
        llm_calls=sum(attempt.llm_calls for attempt in attempts),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usd=usd,
        usd_per_ok_attempt=usd / ok_attempts,
        tokens_per_attempt=(input_tokens + output_tokens) / len(attempts),
        usd_per_day=usd_per_day,
        projected_usd_per_30_days=usd_per_day * PROJECTION_DAYS,
        by_day=tuple(
            DailySpend(day=day, attempts=len(usds), usd=sum(usds))
            for day, usds in _usd_by(attempts, pricing, _utc_day)
        ),
        by_project=tuple(
            ProjectSpend(project_id=project_id, attempts=len(usds), usd=sum(usds))
            for project_id, usds in _usd_by(attempts, pricing, _project_id)
        ),
    )


def _utc_day(attempt: LlmAttemptTelemetry) -> date:
    return attempt.created_at.astimezone(UTC).date()


def _project_id(attempt: LlmAttemptTelemetry) -> str:
    return attempt.project_id


def _usd_by(
    attempts: Sequence[LlmAttemptTelemetry],
    pricing: TokenPricing,
    key: Callable[[LlmAttemptTelemetry], _GroupKey],
) -> list[tuple[_GroupKey, list[float]]]:
    groups: defaultdict[_GroupKey, list[float]] = defaultdict(list)
    for attempt in attempts:
        groups[key(attempt)].append(_usd(attempt, pricing))
    return sorted(groups.items(), key=lambda group: group[0])


class ReplayUnitCost(FrozenModel):
    arm_id: ArmId
    n: int = Field(gt=0)
    tokens_per_episode: float = Field(ge=0)
    usd_per_episode: float = Field(ge=0)


def replay_unit_cost(records: Sequence[MetricsRecord]) -> ReplayUnitCost:
    """One complete control run's token and usd cost per replayed episode."""
    arm_id = _one_complete_control_run(records)
    tokens = _one_record(records, LlmMetricId.TOKENS_PER_EPISODE, arm_id)
    usd = _one_record(records, LlmMetricId.USD_PER_EPISODE, arm_id)
    return ReplayUnitCost(
        arm_id=arm_id,
        n=usd.metric.n,
        tokens_per_episode=tokens.metric.value,
        usd_per_episode=usd.metric.value,
    )


def _one_complete_control_run(records: Sequence[MetricsRecord]) -> str:
    arm_ids = sorted({record.arm_id for record in records})
    if len(arm_ids) != 1:
        raise IncumbentCostError(f'a replay unit cost reads one arm run; got arm ids {arm_ids}')
    for record in records:
        where = f'{record.arm_id!r} record {record.metric.metric_id!r}'
        if record.arm_role != 'control':
            raise IncumbentCostError(f'{where} has arm_role {record.arm_role!r}, not control')
        if record.delta_of is not None:
            raise IncumbentCostError(f'{where} is a delta record, not a measurement')
        if record.incomplete:
            raise IncumbentCostError(f'{where} is incomplete')
    return arm_ids[0]


def _one_record(
    records: Sequence[MetricsRecord], metric_id: LlmMetricId, arm_id: str
) -> MetricsRecord:
    matching = [record for record in records if record.metric.metric_id == metric_id]
    if len(matching) != 1:
        raise IncumbentCostError(
            f'control run {arm_id!r} must report one {metric_id.value!r} record; '
            f'got {len(matching)}'
        )
    return matching[0]


class IncumbentCost(FrozenModel):
    schema_version: Literal[1]
    pricing: TokenPricing
    pricing_arm_id: ArmId
    production: ProductionCost
    replay: tuple[ReplayUnitCost, ...] = Field(min_length=1)


def derive_incumbent_cost(
    window: TelemetryWindow,
    *,
    pricing_spec: LlmArmSpec,
    control_records: Sequence[Sequence[MetricsRecord]],
) -> IncumbentCost:
    pricing = pricing_spec.pricing
    if pricing is None:
        raise IncumbentCostError(
            f'pricing spec {pricing_spec.arm_id!r} carries no pricing: it cannot price the '
            'incumbent'
        )
    if not control_records:
        raise IncumbentCostError('no control run given: the replay unit cost needs one')
    replay = tuple(replay_unit_cost(records) for records in control_records)
    _require_distinct_arms(replay)
    return IncumbentCost(
        schema_version=INCUMBENT_COST_SCHEMA_VERSION,
        pricing=pricing,
        pricing_arm_id=pricing_spec.arm_id,
        production=production_cost(window, pricing),
        replay=replay,
    )


def _require_distinct_arms(replay: Sequence[ReplayUnitCost]) -> None:
    arm_ids = [unit.arm_id for unit in replay]
    repeated = sorted({arm_id for arm_id in arm_ids if arm_ids.count(arm_id) > 1})
    if repeated:
        raise IncumbentCostError(f'control runs repeat arm(s) {repeated}: give each arm once')


def serialize_incumbent_cost(cost: IncumbentCost) -> str:
    return canonical_json_text(cost.model_dump(mode='json'))


def load_incumbent_cost(path: Path | str) -> IncumbentCost:
    return IncumbentCost.model_validate_json(Path(path).read_text())


def serialize_llm_attempts(attempts: Sequence[LlmAttemptTelemetry]) -> str:
    """One canonical (sorted-key) JSON line per attempt."""
    return ''.join(
        json.dumps(attempt.model_dump(mode='json'), sort_keys=True, ensure_ascii=False) + '\n'
        for attempt in attempts
    )


def load_llm_attempts(path: Path | str) -> tuple[LlmAttemptTelemetry, ...]:
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    return tuple(
        LlmAttemptTelemetry.model_validate_json(line) for line in lines if line.strip()
    )
