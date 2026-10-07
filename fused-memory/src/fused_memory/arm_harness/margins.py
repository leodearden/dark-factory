"""Pre-registered non-inferiority margins, derived from two control runs' MetricsRecords.

    margin_m = max(2 * sigma_control(m), floor_m)

``sigma_control`` is the run pair's sample sd ``|a - b| / sqrt(2)``, or, for a metric a
control pair observes only once, that observation's per-episode standard error. The
floor is a proportion's resolution quantum ``1 / min(denominator)`` and is absent for
every other kind. Rationale: plans/local-memory-models-eval-preregistration.md.
"""

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Self

from pydantic import model_validator
from shared.memory_eval_metrics import MetricDirection

from fused_memory.arm_harness.arm_spec import ArmAxis
from fused_memory.arm_harness.frozen_model import FrozenModel
from fused_memory.arm_harness.metrics_record import (
    METRIC_IDS_BY_AXIS,
    EmbeddingMetricId,
    IndexConfiguration,
    LlmMetricId,
    MetricsRecord,
)

EPISODE_MEAN_TOLERANCE = 1e-9


class SigmaSource(StrEnum):
    RUN_PAIR = 'run-pair'
    EPISODE_SE = 'episode-standard-error'


class GatedMetric(FrozenModel):
    metric_id: str
    direction: MetricDirection
    sigma_source: SigmaSource


def _gated(metric_id: str, direction: MetricDirection, source: SigmaSource) -> GatedMetric:
    return GatedMetric(metric_id=metric_id, direction=direction, sigma_source=source)


GATED_METRICS: Mapping[str, GatedMetric] = MappingProxyType({
    gated.metric_id: gated
    for gated in (
        _gated(LlmMetricId.CONFORMANCE_RATE, 'lower_is_worse', SigmaSource.RUN_PAIR),
        _gated(LlmMetricId.EPISODE_FAILURE_RATE, 'higher_is_worse', SigmaSource.RUN_PAIR),
        _gated(LlmMetricId.RETRIEVAL_UTILITY, 'lower_is_worse', SigmaSource.RUN_PAIR),
        _gated(LlmMetricId.GRAPH_SAMENESS, 'lower_is_worse', SigmaSource.EPISODE_SE),
        _gated(EmbeddingMetricId.KNOWN_ITEM_RECALL_AT_5, 'lower_is_worse', SigmaSource.RUN_PAIR),
        _gated(EmbeddingMetricId.KNOWN_ITEM_RECALL_AT_10, 'lower_is_worse', SigmaSource.RUN_PAIR),
        _gated(EmbeddingMetricId.MRR, 'lower_is_worse', SigmaSource.RUN_PAIR),
    )
})


class MarginDerivationError(ValueError):
    """The control records cannot yield a margin, so none is pre-registered."""


class MarginEntry(FrozenModel):
    metric_id: str
    index_configuration: IndexConfiguration | None
    direction: MetricDirection
    sigma_source: SigmaSource
    control_values: tuple[float, ...]
    reference_value: float
    sigma: float
    floor: float | None
    margin: float

    @model_validator(mode='after')
    def _carries_its_derivation(self) -> Self:
        values = (*self.control_values, self.reference_value, self.sigma, self.margin)
        if not all(math.isfinite(v) for v in (*values, *_present(self.floor))):
            raise ValueError(f'margin entry {self.metric_id!r} carries a non-finite value')
        if self.sigma < 0:
            raise ValueError(f'margin entry {self.metric_id!r}: sigma {self.sigma} is negative')
        derived = margin_formula(self.sigma, self.floor)
        if self.margin != derived:
            raise ValueError(
                f'margin entry {self.metric_id!r}: margin {self.margin} != '
                f'max(2 * sigma {self.sigma}, floor {self.floor}) = {derived}'
            )
        return self

    def admits(self, candidate_value: float) -> bool:
        """Whether ``candidate_value`` is no worse than the reference by more than the margin."""
        if not math.isfinite(candidate_value):
            raise ValueError(
                f'{self.metric_id!r}: candidate value {candidate_value} cannot be judged'
            )
        if self.direction == 'lower_is_worse':
            return candidate_value >= self.reference_value - self.margin
        return candidate_value <= self.reference_value + self.margin


def margin_formula(sigma: float, floor: float | None) -> float:
    return max(2 * sigma, floor or 0.0)


def _present(value: float | None) -> tuple[float, ...]:
    return () if value is None else (value,)


_RecordKey = tuple[str, IndexConfiguration | None]


def derive_margins(
    records_a: Sequence[MetricsRecord],
    records_b: Sequence[MetricsRecord],
    *,
    episode_values: Mapping[str, Sequence[float]],
) -> tuple[MarginEntry, ...]:
    """One margin per gated metric (and index configuration) of the two control runs' axis."""
    axis = _require_comparable_records(records_a, records_b)
    gated = tuple(g for g in GATED_METRICS.values() if g.metric_id in METRIC_IDS_BY_AXIS[axis])
    _require_episode_values_are_used(episode_values, gated)
    run_a, run_b = _ControlRun.of(records_a), _ControlRun.of(records_b)
    entries = [
        entry
        for gate in gated
        for entry in _entries_for(gate, run_a, run_b, episode_values)
    ]
    return tuple(sorted(entries, key=lambda e: (e.metric_id, e.index_configuration or '')))


def _require_comparable_records(
    records_a: Sequence[MetricsRecord], records_b: Sequence[MetricsRecord]
) -> ArmAxis:
    arm_a, arm_b = _single_arm(records_a, 'a'), _single_arm(records_b, 'b')
    if arm_a == arm_b:
        raise MarginDerivationError(
            f'both control runs are arm {arm_a!r}; margins need two distinct control runs'
        )
    every = (*records_a, *records_b)
    for record in every:
        _require_control_measurement(record)
        _require_direction_agrees(record)
        _require_finite_value(record)
    for field in ('corpus_sha', 'code_sha', 'axis'):
        _require_shared(every, field)
    return every[0].axis


def _single_arm(records: Sequence[MetricsRecord], run: str) -> str:
    arms = sorted({record.arm_id for record in records})
    if len(arms) != 1:
        raise MarginDerivationError(f'control run {run} must be one arm\'s records; got {arms}')
    return arms[0]


def _require_control_measurement(record: MetricsRecord) -> None:
    where = f'{record.arm_id!r} {record.metric.metric_id!r}'
    if record.arm_role != 'control':
        raise MarginDerivationError(f'{where} is a {record.arm_role} record, not a control')
    if record.delta_of is not None:
        raise MarginDerivationError(
            f'{where} is a delta record ({record.delta_of}), not a control measurement'
        )


def _require_direction_agrees(record: MetricsRecord) -> None:
    gated = GATED_METRICS.get(record.metric.metric_id)
    declared = record.metric.direction
    if gated is not None and declared is not None and declared != gated.direction:
        raise MarginDerivationError(
            f'{record.arm_id!r} declares {record.metric.metric_id!r} {declared}, '
            f'but the gated table says {gated.direction}'
        )


def _require_finite_value(record: MetricsRecord) -> None:
    if not math.isfinite(record.metric.value):
        raise MarginDerivationError(
            f'{record.arm_id!r} {record.metric.metric_id!r} has non-finite value '
            f'{record.metric.value}'
        )


def _require_shared(records: Iterable[MetricsRecord], field: str) -> None:
    values = sorted({str(getattr(record, field)) for record in records})
    if len(values) != 1:
        raise MarginDerivationError(f'control records must share one {field}; got {values}')


@dataclass(frozen=True)
class _ControlRun:
    arm_id: str
    keyed: Mapping[_RecordKey, MetricsRecord]

    @classmethod
    def of(cls, records: Sequence[MetricsRecord]) -> '_ControlRun':
        keyed: dict[_RecordKey, MetricsRecord] = {}
        for record in records:
            key = (record.metric.metric_id, record.index_configuration)
            if key in keyed:
                raise MarginDerivationError(f'arm {record.arm_id!r} reports {key} twice')
            keyed[key] = record
        return cls(arm_id=records[0].arm_id, keyed=MappingProxyType(keyed))

    def get(self, key: _RecordKey) -> MetricsRecord | None:
        return self.keyed.get(key)

    def require(self, key: _RecordKey) -> MetricsRecord:
        record = self.keyed.get(key)
        if record is None:
            raise MarginDerivationError(
                f'gated metric {key} is missing from control run {self.arm_id!r}; a run-pair '
                'sigma needs it in both runs'
            )
        return record


def _require_episode_values_are_used(
    episode_values: Mapping[str, Sequence[float]], gated: Sequence[GatedMetric]
) -> None:
    takers = {g.metric_id for g in gated if g.sigma_source is SigmaSource.EPISODE_SE}
    unused = sorted(set(episode_values) - takers)
    if unused:
        raise MarginDerivationError(
            f'episode values given for {unused}, whose sigma is not the episode standard '
            f'error (episode-SE metrics: {sorted(takers)})'
        )


def _entries_for(
    gate: GatedMetric,
    run_a: _ControlRun,
    run_b: _ControlRun,
    episode_values: Mapping[str, Sequence[float]],
) -> tuple[MarginEntry, ...]:
    keys = sorted(
        {key for key in (*run_a.keyed, *run_b.keyed) if key[0] == gate.metric_id},
        key=lambda key: (key[0], key[1] or ''),
    )
    if not keys:
        raise MarginDerivationError(
            f'gated metric {gate.metric_id!r} is reported by neither control run'
        )
    if gate.sigma_source is SigmaSource.EPISODE_SE:
        return tuple(
            _episode_se_entry(
                gate,
                key[1],
                tuple(r for r in (run_a.get(key), run_b.get(key)) if r is not None),
                episode_values.get(gate.metric_id),
            )
            for key in keys
        )
    return tuple(
        _run_pair_entry(gate, key[1], run_a.require(key), run_b.require(key)) for key in keys
    )


def _floor(records: Sequence[MetricsRecord]) -> float | None:
    kinds = sorted({record.metric.kind for record in records})
    if len(kinds) != 1:
        metric_id = records[0].metric.metric_id
        raise MarginDerivationError(f'{metric_id!r} is reported as different kinds {kinds}')
    denominators = [r.metric.denominator for r in records if r.metric.denominator is not None]
    if kinds[0] != 'proportion':
        return None
    return 1 / min(denominators)


def _run_pair_entry(
    gate: GatedMetric,
    configuration: IndexConfiguration | None,
    record_a: MetricsRecord,
    record_b: MetricsRecord,
) -> MarginEntry:
    value_a, value_b = record_a.metric.value, record_b.metric.value
    sigma = abs(value_a - value_b) / math.sqrt(2)
    floor = _floor((record_a, record_b))
    return MarginEntry(
        metric_id=gate.metric_id,
        index_configuration=configuration,
        direction=gate.direction,
        sigma_source=gate.sigma_source,
        control_values=(value_a, value_b),
        reference_value=statistics.mean((value_a, value_b)),
        sigma=sigma,
        floor=floor,
        margin=margin_formula(sigma, floor),
    )


def _episode_se_entry(
    gate: GatedMetric,
    configuration: IndexConfiguration | None,
    reports: Sequence[MetricsRecord],
    values: Sequence[float] | None,
) -> MarginEntry:
    if len(reports) != 1:
        arms = [record.arm_id for record in reports]
        raise MarginDerivationError(
            f'{gate.metric_id!r} must be reported by exactly one control run; got {arms}'
        )
    (record,) = reports
    episode = _checked_episode_values(record, values)
    sigma = statistics.stdev(episode) / math.sqrt(len(episode))
    floor = _floor(reports)
    return MarginEntry(
        metric_id=gate.metric_id,
        index_configuration=configuration,
        direction=gate.direction,
        sigma_source=gate.sigma_source,
        control_values=(record.metric.value,),
        reference_value=record.metric.value,
        sigma=sigma,
        floor=floor,
        margin=margin_formula(sigma, floor),
    )


def _checked_episode_values(
    record: MetricsRecord, values: Sequence[float] | None
) -> tuple[float, ...]:
    metric = record.metric
    if values is None:
        raise MarginDerivationError(
            f'{metric.metric_id!r} takes the episode standard error, but no episode values '
            'were given'
        )
    episode = tuple(values)
    if not all(math.isfinite(value) for value in episode):
        raise MarginDerivationError(
            f'{metric.metric_id!r} episode values are not all finite: {list(episode)}'
        )
    if len(episode) < 2:
        raise MarginDerivationError(
            f'{metric.metric_id!r} has {len(episode)} episode value(s); a standard error '
            'needs at least 2'
        )
    if len(episode) != metric.n:
        raise MarginDerivationError(
            f'{metric.metric_id!r} has {len(episode)} episode values, but the record of '
            f'{record.arm_id!r} counts n={metric.n}'
        )
    mean = statistics.mean(episode)
    if abs(mean - metric.value) > EPISODE_MEAN_TOLERANCE:
        raise MarginDerivationError(
            f'{metric.metric_id!r} episode values average {mean}, but the record of '
            f'{record.arm_id!r} says {metric.value}'
        )
    return episode
