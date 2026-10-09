"""EmbeddingRunManifest: the ``run.json`` an embedding arm run writes last, as its commit marker.

A run whose instrument check fails writes none, so a manifest holds only passed
checks. Its text form follows the same null convention as ``run_manifest``'s.
"""

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator
from shared.memory_eval_metrics import canonical_json_text

from fused_memory.arm_harness.arm_spec import EmbeddingArmSpec
from fused_memory.arm_harness.frozen_model import FrozenModel
from fused_memory.arm_harness.graph_copy import GraphReembed
from fused_memory.arm_harness.instrument_checks import CheckResult
from fused_memory.arm_harness.mem0_replica import ReplicaReembed
from fused_memory.arm_harness.metrics_record import IndexConfiguration, UtcDatetime
from fused_memory.arm_harness.run_manifest import EffectiveEmbedder, null_convention_payload

EMBEDDING_RUN_MANIFEST_SCHEMA_VERSION = 1


class EmbeddingRunSettings(FrozenModel):
    embed_batch_size: int = Field(gt=0)
    embed_concurrency: int = Field(gt=0)
    query_concurrency: int = Field(gt=0)
    search_k: int = Field(gt=0)
    transcript_queries: int = Field(ge=0)
    mem0_project_id: str = Field(min_length=1)


class QueryFailureCounts(FrozenModel):
    """Queries the arm could not embed. Each still counts in its metric's n, as a miss."""

    known_item: dict[IndexConfiguration, int]
    mem0_known_item: int = Field(ge=0)
    query_latency: int = Field(ge=0)

    @model_validator(mode='after')
    def _one_count_per_configuration(self) -> Self:
        if set(self.known_item) != set(IndexConfiguration):
            raise ValueError(
                f'known_item counts cover {sorted(self.known_item)}, '
                f'not every index configuration {sorted(IndexConfiguration)}'
            )
        negative = {key: count for key, count in self.known_item.items() if count < 0}
        if negative:
            raise ValueError(f'known_item counts are negative: {negative}')
        return self


class EmbeddingRunManifest(FrozenModel):
    schema_version: Literal[1]
    spec: EmbeddingArmSpec
    settings: EmbeddingRunSettings
    effective_embedder: EffectiveEmbedder
    graph_reembed: GraphReembed
    replica_reembed: ReplicaReembed
    query_failures: QueryFailureCounts
    check_results: tuple[CheckResult, ...]
    started_at: UtcDatetime
    finished_at: UtcDatetime

    @model_validator(mode='after')
    def _manifest_rules(self) -> Self:
        if self.finished_at < self.started_at:
            raise ValueError(
                f'finished_at {self.finished_at} precedes started_at {self.started_at}'
            )
        if self.effective_embedder.dimensions != self.spec.embedding_dim:
            raise ValueError(
                f'arm {self.spec.arm_id!r} declares dimension {self.spec.embedding_dim}, '
                f'but its effective embedder has {self.effective_embedder.dimensions}'
            )
        failed = [check.check_id for check in self.check_results if not check.passed]
        if failed:
            raise ValueError(f'a committed embedding run holds only passed checks; failed: {failed}')
        return self


def serialize_embedding_run_manifest(manifest: EmbeddingRunManifest) -> str:
    return canonical_json_text(null_convention_payload(manifest))


def load_embedding_run_manifest(path: Path | str) -> EmbeddingRunManifest:
    return EmbeddingRunManifest.model_validate_json(Path(path).read_text())
