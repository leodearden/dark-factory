"""MetricsRecord: one artifact per (arm, metric[, index configuration]) (PRD §Contract).

The record wraps the imported M1 ``Metric``, so M1's kind/value/n/direction rules
apply verbatim. It adds the arm's identity and the three SHAs. Serialization
follows the null convention stated in ``shared/src/shared/memory_eval_metrics.py``'s
module docstring. ``preregistration_sha`` is this record's one required-nullable
field.
"""

from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, model_validator
from shared.memory_eval_metrics import Metric, canonical_json_text
from shared.safe_io import atomic_write_text

from fused_memory.arm_harness.arm_spec import (
    ArmAxis,
    ArmId,
    ArmRole,
    ContentSha,
    EmbeddingArmSpec,
    GitSha,
    LlmArmSpec,
    require_preregistration_matches_role,
)

METRICS_SCHEMA_VERSION = 1
METRICS_DIRNAME = 'metrics'


class LlmMetricId(StrEnum):
    CONFORMANCE_RATE = 'conformance-rate'
    EPISODE_FAILURE_RATE = 'episode-failure-rate'
    EPISODE_LATENCY_P50 = 'episode-latency-p50'
    EPISODE_LATENCY_P95 = 'episode-latency-p95'
    GRAPH_SAMENESS = 'graph-sameness'
    RETRIEVAL_UTILITY = 'retrieval-utility'
    TOKENS_PER_EPISODE = 'tokens-per-episode'
    USD_PER_EPISODE = 'usd-per-episode'


class EmbeddingMetricId(StrEnum):
    KNOWN_ITEM_RECALL_AT_5 = 'known-item-recall@5'
    KNOWN_ITEM_RECALL_AT_10 = 'known-item-recall@10'
    MRR = 'mrr'
    QUERY_EMBED_LATENCY_P95 = 'query-embed-latency-p95'
    REEMBED_THROUGHPUT = 'reembed-throughput'


LLM_METRIC_IDS: frozenset[str] = frozenset(LlmMetricId)
EMBEDDING_METRIC_IDS: frozenset[str] = frozenset(EmbeddingMetricId)
METRIC_IDS_BY_AXIS: Mapping[ArmAxis, frozenset[str]] = MappingProxyType({
    'llm': LLM_METRIC_IDS,
    'embedding': EMBEDDING_METRIC_IDS,
})
PER_INDEX_CONFIGURATION_METRIC_IDS: frozenset[str] = frozenset({
    EmbeddingMetricId.KNOWN_ITEM_RECALL_AT_5,
    EmbeddingMetricId.KNOWN_ITEM_RECALL_AT_10,
    EmbeddingMetricId.MRR,
})


class IndexConfiguration(StrEnum):
    WITH_INDICES = 'with-indices'
    EMBEDDING_ONLY = 'embedding-only'


class DeltaOf(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    minuend_arm_id: ArmId
    subtrahend_arm_id: ArmId

    @model_validator(mode='after')
    def _distinct_arms(self) -> Self:
        if self.minuend_arm_id == self.subtrahend_arm_id:
            raise ValueError(f'delta of arm {self.minuend_arm_id!r} with itself is not a delta')
        return self


UtcDatetime = Annotated[AwareDatetime, AfterValidator(lambda value: value.astimezone(UTC))]


class MetricsRecord(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    schema_version: Literal[1]
    arm_id: ArmId
    axis: ArmAxis
    arm_role: ArmRole
    measured_at: UtcDatetime
    code_sha: GitSha
    corpus_sha: ContentSha
    preregistration_sha: GitSha | None
    incomplete: bool
    index_configuration: IndexConfiguration | None = None
    delta_of: DeltaOf | None = None
    metric: Metric

    @model_validator(mode='after')
    def _record_rules(self) -> Self:
        require_preregistration_matches_role(self.arm_id, self.arm_role, self.preregistration_sha)
        self._require_metric_on_axis()
        self._require_index_configuration_iff_per_configuration()
        return self

    def _require_metric_on_axis(self) -> None:
        allowed = METRIC_IDS_BY_AXIS[self.axis]
        if self.metric.metric_id not in allowed:
            raise ValueError(
                f'metric {self.metric.metric_id!r} is not a {self.axis}-axis metric '
                f'(allowed: {", ".join(sorted(allowed))})'
            )

    def _require_index_configuration_iff_per_configuration(self) -> None:
        per_configuration = self.metric.metric_id in PER_INDEX_CONFIGURATION_METRIC_IDS
        if per_configuration and self.index_configuration is None:
            raise ValueError(
                f'metric {self.metric.metric_id!r} is reported per index configuration, '
                'so index_configuration is required'
            )
        if not per_configuration and self.index_configuration is not None:
            raise ValueError(
                f'metric {self.metric.metric_id!r} is not reported per index configuration, '
                f'so index_configuration must be absent (got {self.index_configuration.value!r})'
            )


def record_for(
    spec: LlmArmSpec | EmbeddingArmSpec,
    metric: Metric,
    *,
    measured_at: datetime,
    incomplete: bool,
    index_configuration: IndexConfiguration | None = None,
    delta_of: DeltaOf | None = None,
) -> MetricsRecord:
    return MetricsRecord(
        schema_version=METRICS_SCHEMA_VERSION,
        arm_id=spec.arm_id,
        axis=spec.axis,
        arm_role=spec.arm_role,
        measured_at=measured_at,
        code_sha=spec.code_sha,
        corpus_sha=spec.corpus_sha,
        preregistration_sha=spec.preregistration_sha,
        incomplete=incomplete,
        index_configuration=index_configuration,
        delta_of=delta_of,
        metric=metric,
    )


def serialize_metrics_record(record: MetricsRecord) -> str:
    payload = record.model_dump(mode='json', exclude_none=True)
    payload['preregistration_sha'] = record.preregistration_sha
    return canonical_json_text(payload)


def metrics_record_path(record: MetricsRecord, run_dir: Path) -> Path:
    suffix = f'.{record.index_configuration.value}' if record.index_configuration else ''
    return run_dir / METRICS_DIRNAME / f'{record.metric.metric_id}{suffix}.json'


def write_metrics_record(record: MetricsRecord, run_dir: Path) -> Path:
    text = serialize_metrics_record(record)
    MetricsRecord.model_validate_json(text)
    path = metrics_record_path(record, run_dir)
    atomic_write_text(path, text, mkdir=True)
    return path


def load_metrics_record(path: Path | str) -> MetricsRecord:
    return MetricsRecord.model_validate_json(Path(path).read_text())
