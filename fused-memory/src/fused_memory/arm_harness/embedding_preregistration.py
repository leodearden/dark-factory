"""The embedding axis's pre-registered decision quantities, from the incumbent control pair, and one candidate judged against them.

Margins come only from ``margins.derive_margins`` (preregistration §3 and §6).
The query-latency envelope is ``p95_bound = search_timeout / LATENCY_HEADROOM``,
anchored on the search timeout the runs recorded. Only the with-indices margins
and the envelope decide non-inferiority. The embedding-only, Mem0 and throughput
rows are reported (preregistration §6).

A query the arm could not embed is the arm's failure: a known-item query reads as
a miss in its metric, and a latency query, having no latency, keeps the envelope
from admitting the arm. A run whose re-embed left texts unembedded probed a
partial store, so it measured that run rather than the model and is not judged.
"""

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator
from shared.memory_eval_metrics import canonical_json_text

from fused_memory.arm_harness.arm_spec import ArmId, ContentSha, GitSha
from fused_memory.arm_harness.embedding_run_manifest import EmbeddingRunManifest
from fused_memory.arm_harness.frozen_model import FrozenModel
from fused_memory.arm_harness.instrument_checks import (
    CheckResult,
    InstrumentCheckId,
    check_failed,
    check_passed,
)
from fused_memory.arm_harness.margins import MarginEntry, derive_margins
from fused_memory.arm_harness.metrics_record import (
    EmbeddingMetricId,
    IndexConfiguration,
    MetricsRecord,
)
from fused_memory.arm_harness.preregistration import LATENCY_HEADROOM

EMBEDDING_PREREGISTRATION_INPUTS_FILENAME = 'embedding-preregistration-inputs.json'
DECIDING_CONFIGURATION = IndexConfiguration.WITH_INDICES
REPORTED_METRIC_IDS = (
    EmbeddingMetricId.MEM0_KNOWN_ITEM_RECALL_AT_5,
    EmbeddingMetricId.MEM0_KNOWN_ITEM_RECALL_AT_10,
    EmbeddingMetricId.MEM0_MRR,
    EmbeddingMetricId.REEMBED_THROUGHPUT,
)
_MS_PER_S = 1000

_RecordKey = tuple[str, IndexConfiguration | None]


class EmbeddingPreregistrationError(ValueError):
    """The runs cannot yield, or be judged against, a valid embedding pre-registration."""


# --- symmetry -------------------------------------------------------------------------


def check_embedding_run_symmetry(runs: Sequence[EmbeddingRunManifest]) -> CheckResult:
    """Every run shares its settings, corpus_sha and code_sha; the embedder is the variable."""
    _require_distinct_runs(runs)
    check_id = InstrumentCheckId.ARM_CONFIG_SYMMETRY
    values = [(run.spec.arm_id, _symmetric_values(run)) for run in runs]
    first = values[0][1]
    asymmetric = [
        field for field in first if any(by_field[field] != first[field] for _, by_field in values)
    ]
    if asymmetric:
        lines = [
            f'{field}: ' + ', '.join(f'{arm_id}={by_field[field]}' for arm_id, by_field in values)
            for field in asymmetric
        ]
        detail = f'embedding arms must run symmetrically; differing: {"; ".join(lines)}'
        return check_failed(check_id, detail, asymmetric)
    return check_passed(check_id, f'{len(runs)} embedding arms ran symmetrically at one code sha')


def _require_distinct_runs(runs: Sequence[EmbeddingRunManifest]) -> None:
    if len(runs) < 2:
        raise EmbeddingPreregistrationError(
            f'a symmetry check needs at least two runs, got {len(runs)}'
        )
    counts = Counter(run.spec.arm_id for run in runs)
    repeated = sorted(arm_id for arm_id, count in counts.items() if count > 1)
    if repeated:
        raise EmbeddingPreregistrationError(f'runs repeat arm ids {repeated}; one run per arm')


def _symmetric_values(run: EmbeddingRunManifest) -> dict[str, object]:
    settings = {f'settings.{name}': value for name, value in run.settings.model_dump().items()}
    return {**settings, 'corpus_sha': run.spec.corpus_sha, 'code_sha': run.spec.code_sha}


def _require_complete_build(run: EmbeddingRunManifest) -> None:
    graph, replica = run.graph_reembed.failures, run.replica_reembed.failures
    if graph or replica:
        raise EmbeddingPreregistrationError(
            f'arm {run.spec.arm_id!r} could not embed {len(graph)} graph texts and '
            f'{len(replica)} Mem0 records, so its probes searched a partial store: the run '
            'measured itself, not the model; re-run it'
        )


# --- the query-latency envelope ---------------------------------------------------------


class QueryLatencyEnvelope(FrozenModel):
    search_timeout_s: float = Field(gt=0)
    headroom: float = Field(ge=1)
    p95_bound_ms: float

    @model_validator(mode='after')
    def _bound_is_derived(self) -> Self:
        derived = self.search_timeout_s * _MS_PER_S / self.headroom
        if self.p95_bound_ms != derived:
            raise ValueError(
                f'p95_bound_ms {self.p95_bound_ms} != search_timeout_s {self.search_timeout_s} '
                f'* {_MS_PER_S} / headroom {self.headroom} = {derived}'
            )
        return self

    def admits(self, p95_ms: float) -> bool:
        """Whether a warm query-embed p95 under load lies strictly inside the bound."""
        if not math.isfinite(p95_ms):
            raise ValueError(f'query latency p95 {p95_ms} cannot be judged against the envelope')
        return p95_ms < self.p95_bound_ms


def query_latency_envelope(search_timeout_s: float) -> QueryLatencyEnvelope:
    return QueryLatencyEnvelope(
        search_timeout_s=search_timeout_s,
        headroom=LATENCY_HEADROOM,
        p95_bound_ms=search_timeout_s * _MS_PER_S / LATENCY_HEADROOM,
    )


def _envelope_admits_run(
    envelope: QueryLatencyEnvelope, p95_ms: float, failed_queries: int
) -> bool:
    """A failed latency query has no latency to fall inside the bound, so it lies outside."""
    return failed_queries == 0 and envelope.admits(p95_ms)


# --- derivation -------------------------------------------------------------------------


class EmbeddingPreregistrationInputs(FrozenModel):
    schema_version: Literal[1]
    control_arm_ids: tuple[ArmId, ArmId]
    code_sha: GitSha
    corpus_sha: ContentSha
    margins: tuple[MarginEntry, ...]
    query_latency_envelope: QueryLatencyEnvelope
    incumbent_query_latency_p95_ms: float

    @model_validator(mode='after')
    def _is_a_valid_preregistration(self) -> Self:
        arm_a, arm_b = self.control_arm_ids
        if arm_a == arm_b:
            raise ValueError(f'control_arm_ids name arm {arm_a!r} twice; a control pair is two')
        _require_incumbent_inside(
            self.query_latency_envelope, self.incumbent_query_latency_p95_ms, failed_queries=0
        )
        return self


def derive_embedding_preregistration_inputs(
    run_a: EmbeddingRunManifest,
    records_a: Sequence[MetricsRecord],
    run_b: EmbeddingRunManifest,
    records_b: Sequence[MetricsRecord],
) -> EmbeddingPreregistrationInputs:
    """Refuses unless both are symmetric control runs whose incumbent sits inside its envelope."""
    for run, records in ((run_a, records_a), (run_b, records_b)):
        _require_control_run(run, records)
    symmetry = check_embedding_run_symmetry((run_a, run_b))
    if not symmetry.passed:
        raise EmbeddingPreregistrationError(f'the control pair is asymmetric: {symmetry.detail}')
    envelope = query_latency_envelope(run_a.settings.search_timeout_s)
    p95_ms = max(_query_latency_p95_ms(records_a), _query_latency_p95_ms(records_b))
    failed = run_a.query_failures.query_latency + run_b.query_failures.query_latency
    _require_incumbent_inside(envelope, p95_ms, failed_queries=failed)
    return EmbeddingPreregistrationInputs(
        schema_version=1,
        control_arm_ids=(run_a.spec.arm_id, run_b.spec.arm_id),
        code_sha=run_a.spec.code_sha,
        corpus_sha=run_a.spec.corpus_sha,
        margins=derive_margins(records_a, records_b, episode_values={}),
        query_latency_envelope=envelope,
        incumbent_query_latency_p95_ms=p95_ms,
    )


def _require_control_run(run: EmbeddingRunManifest, records: Sequence[MetricsRecord]) -> None:
    if run.spec.arm_role != 'control':
        raise EmbeddingPreregistrationError(
            f'arm {run.spec.arm_id!r} is a {run.spec.arm_role} run; the pre-registration '
            'takes two control runs'
        )
    _require_records_of(run, records)
    _require_complete_build(run)


def _require_records_of(run: EmbeddingRunManifest, records: Sequence[MetricsRecord]) -> None:
    arms = sorted({record.arm_id for record in records})
    if arms != [run.spec.arm_id]:
        raise EmbeddingPreregistrationError(
            f'the records given for run {run.spec.arm_id!r} are of arms {arms}'
        )


def _require_incumbent_inside(
    envelope: QueryLatencyEnvelope, p95_ms: float, *, failed_queries: int
) -> None:
    if not _envelope_admits_run(envelope, p95_ms, failed_queries):
        raise EmbeddingPreregistrationError(
            f'the incumbent query-embed p95 {p95_ms} ms, with {failed_queries} failed latency '
            f'queries, is not inside the envelope bound {envelope.p95_bound_ms} ms: an envelope '
            'the incumbent fails is not a valid pre-registration'
        )


def _query_latency_p95_ms(records: Sequence[MetricsRecord]) -> float:
    return _value_of(_keyed(records), (EmbeddingMetricId.QUERY_EMBED_LATENCY_P95, None))


def _keyed(records: Sequence[MetricsRecord]) -> dict[_RecordKey, MetricsRecord]:
    keyed: dict[_RecordKey, MetricsRecord] = {}
    for record in records:
        key = (record.metric.metric_id, record.index_configuration)
        if key in keyed:
            raise EmbeddingPreregistrationError(f'arm {record.arm_id!r} reports {key} twice')
        keyed[key] = record
    return keyed


def _value_of(keyed: Mapping[_RecordKey, MetricsRecord], key: _RecordKey) -> float:
    record = keyed.get(key)
    if record is None:
        metric_id, configuration = key
        where = f' ({configuration.value})' if configuration else ''
        raise EmbeddingPreregistrationError(f'no {metric_id}{where} record to judge')
    return record.metric.value


def serialize_embedding_preregistration_inputs(inputs: EmbeddingPreregistrationInputs) -> str:
    return canonical_json_text(inputs.model_dump(mode='json'))


def load_embedding_preregistration_inputs(path: Path | str) -> EmbeddingPreregistrationInputs:
    return EmbeddingPreregistrationInputs.model_validate_json(Path(path).read_text())


# --- comparison --------------------------------------------------------------------------


class MarginVerdict(FrozenModel):
    metric_id: str
    index_configuration: IndexConfiguration | None
    reference_value: float
    margin: float
    candidate_value: float
    admits: bool


class EnvelopeVerdict(FrozenModel):
    p95_bound_ms: float
    candidate_p95_ms: float
    failed_queries: int = Field(ge=0)
    admits: bool


class ReportedRow(FrozenModel):
    metric_id: str
    value: float
    n: int


class EmbeddingComparison(FrozenModel):
    arm_id: ArmId
    margins: tuple[MarginVerdict, ...]
    envelope: EnvelopeVerdict
    reported: tuple[ReportedRow, ...]
    non_inferior: bool

    @model_validator(mode='after')
    def _verdict_follows_the_deciding_rows(self) -> Self:
        deciding = [row for row in self.margins if row.index_configuration is DECIDING_CONFIGURATION]
        if not deciding:
            raise ValueError(f'arm {self.arm_id!r}: no {DECIDING_CONFIGURATION.value} margin decides')
        verdict = all(row.admits for row in deciding) and self.envelope.admits
        if self.non_inferior != verdict:
            raise ValueError(
                f'arm {self.arm_id!r}: non_inferior {self.non_inferior} disagrees with its '
                f'{DECIDING_CONFIGURATION.value} margins and envelope ({verdict})'
            )
        return self


def compare_embedding_arm(
    inputs: EmbeddingPreregistrationInputs,
    candidate_run: EmbeddingRunManifest,
    candidate_records: Sequence[MetricsRecord],
) -> EmbeddingComparison:
    _require_candidate_of(inputs, candidate_run, candidate_records)
    keyed = _keyed(candidate_records)
    margins = tuple(_margin_verdict(entry, keyed) for entry in inputs.margins)
    envelope = _envelope_verdict(
        inputs.query_latency_envelope,
        _query_latency_p95_ms(candidate_records),
        candidate_run.query_failures.query_latency,
    )
    deciding = (row.admits for row in margins if row.index_configuration is DECIDING_CONFIGURATION)
    return EmbeddingComparison(
        arm_id=candidate_run.spec.arm_id,
        margins=margins,
        envelope=envelope,
        reported=tuple(
            ReportedRow(metric_id=metric_id, value=record.metric.value, n=record.metric.n)
            for metric_id in REPORTED_METRIC_IDS
            if (record := keyed.get((metric_id, None))) is not None
        ),
        non_inferior=all(deciding) and envelope.admits,
    )


def _require_candidate_of(
    inputs: EmbeddingPreregistrationInputs,
    run: EmbeddingRunManifest,
    records: Sequence[MetricsRecord],
) -> None:
    spec = run.spec
    if spec.arm_role != 'candidate':
        raise EmbeddingPreregistrationError(
            f'arm {spec.arm_id!r} is a {spec.arm_role} run; only a candidate is compared'
        )
    ran_at = _symmetric_values(run)
    for field, expected in _recorded_symmetric_values(inputs).items():
        if ran_at[field] != expected:
            raise EmbeddingPreregistrationError(
                f'arm {spec.arm_id!r} ran at {field} {ran_at[field]}, but the '
                f'pre-registration was derived at {expected}'
            )
    _require_records_of(run, records)
    _require_complete_build(run)


def _recorded_symmetric_values(inputs: EmbeddingPreregistrationInputs) -> dict[str, object]:
    """The symmetric values the inputs record, under ``_symmetric_values``'s keys."""
    return {
        'code_sha': inputs.code_sha,
        'corpus_sha': inputs.corpus_sha,
        'settings.search_timeout_s': inputs.query_latency_envelope.search_timeout_s,
    }


def _margin_verdict(
    entry: MarginEntry, keyed: Mapping[_RecordKey, MetricsRecord]
) -> MarginVerdict:
    value = _value_of(keyed, (entry.metric_id, entry.index_configuration))
    return MarginVerdict(
        metric_id=entry.metric_id,
        index_configuration=entry.index_configuration,
        reference_value=entry.reference_value,
        margin=entry.margin,
        candidate_value=value,
        admits=entry.admits(value),
    )


def _envelope_verdict(
    envelope: QueryLatencyEnvelope, p95_ms: float, failed_queries: int
) -> EnvelopeVerdict:
    return EnvelopeVerdict(
        p95_bound_ms=envelope.p95_bound_ms,
        candidate_p95_ms=p95_ms,
        failed_queries=failed_queries,
        admits=_envelope_admits_run(envelope, p95_ms, failed_queries),
    )
