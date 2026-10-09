"""One embedding arm run, end to end: refusals, the graph half, the Mem0 replica, query latency, then the records and ``run.json``.

Every refusal and pre-run check comes before the first store call. The graph
half (embedding_graph_phase.py) raises ``EmbeddingRunCheckFailed`` on a failed
instrument check, so such a run writes nothing. A query the arm cannot embed is
a measurement, not an error: it reads as a miss and is counted in the manifest.
``run.json`` is written last and is the run's commit marker.
"""

import asyncio
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from shared.memory_eval_metrics import Metric
from shared.safe_io import atomic_write_text

from fused_memory.arm_harness.arm_config import embedding_arm_config
from fused_memory.arm_harness.arm_embedder import (
    QUERY_LATENCY_CONCURRENCY,
    ArmEmbedder,
    QueryEmbedError,
    QueryLatency,
    measure_query_latency,
)
from fused_memory.arm_harness.arm_spec import EmbeddingArmSpec, LlmArmSpec
from fused_memory.arm_harness.embedding_graph_phase import (
    SEARCH_K,
    ArmGraphClient,
    EmbeddingSearchBackend,
    GraphPhase,
    Sleep,
    run_graph_phase,
)
from fused_memory.arm_harness.embedding_run_manifest import (
    EMBEDDING_RUN_MANIFEST_SCHEMA_VERSION,
    EmbeddingRunManifest,
    EmbeddingRunSettings,
    QueryFailureCounts,
    serialize_embedding_run_manifest,
)
from fused_memory.arm_harness.graph_copy import GraphReembed
from fused_memory.arm_harness.mem0_replica import (
    Mem0Snapshot,
    ReplicaClient,
    ReplicaHit,
    ReplicaReembed,
    build_replica,
    load_snapshot,
    search_replica,
    snapshot_sha,
)
from fused_memory.arm_harness.metrics_record import (
    EmbeddingMetricId,
    IndexConfiguration,
    MetricsRecord,
    record_for,
    write_metrics_record,
)
from fused_memory.arm_harness.probe_set import (
    Mem0KnownItem,
    Mem0SnapshotPin,
    ProbeSet,
    probe_set_sha,
    serialize_probe_set,
)
from fused_memory.arm_harness.retrieval import (
    ProbeTally,
    Rank,
    mrr_metric,
    recall_metric,
    tally_probe,
)
from fused_memory.arm_harness.run import RUN_MANIFEST_FILENAME, require_pre_run_checks
from fused_memory.arm_harness.run_manifest import EffectiveEmbedder
from fused_memory.config.schema import FusedMemoryConfig

logger = logging.getLogger(__name__)

Mem0Ranker = Callable[[str, Sequence[ReplicaHit]], Rank]
"""(topic, replica hits in rank order) -> the topic canonical's 1-based rank, or None."""


class EmbeddingRunRefused(ValueError):
    """The inputs cannot yield a measurement of this arm, so no store was touched."""


@dataclass(frozen=True)
class _KnownItemMetricIds:
    recall_at_5: EmbeddingMetricId
    recall_at_10: EmbeddingMetricId
    mrr: EmbeddingMetricId


_GRAPH_METRIC_IDS = _KnownItemMetricIds(
    EmbeddingMetricId.KNOWN_ITEM_RECALL_AT_5,
    EmbeddingMetricId.KNOWN_ITEM_RECALL_AT_10,
    EmbeddingMetricId.MRR,
)
_MEM0_METRIC_IDS = _KnownItemMetricIds(
    EmbeddingMetricId.MEM0_KNOWN_ITEM_RECALL_AT_5,
    EmbeddingMetricId.MEM0_KNOWN_ITEM_RECALL_AT_10,
    EmbeddingMetricId.MEM0_MRR,
)


async def run_embedding_arm(
    spec: LlmArmSpec | EmbeddingArmSpec,
    probe_set: ProbeSet,
    *,
    graph_client: ArmGraphClient,
    backend: EmbeddingSearchBackend,
    qdrant: ReplicaClient,
    embedder: ArmEmbedder,
    mem0_snapshot: Path,
    mem0_project_id: str,
    rank_mem0: Mem0Ranker,
    base_config: FusedMemoryConfig,
    run_dir: Path,
    repo_root: Path,
    sleep: Sleep = asyncio.sleep,
) -> EmbeddingRunManifest:
    arm = _require_embedding_spec(spec, embedder)
    pre_run = await asyncio.to_thread(require_pre_run_checks, arm, repo_root)
    snapshot = _require_runnable(arm, probe_set, mem0_snapshot, run_dir)
    started_at = datetime.now(UTC)
    graph = await run_graph_phase(arm, probe_set, graph_client, backend, embedder, sleep=sleep)
    replica, mem0 = await _replica_phase(
        arm, probe_set.mem0_known_items, qdrant, embedder, snapshot, rank_mem0, mem0_project_id
    )
    latency = await measure_query_latency(
        embedder, probe_set.transcript.queries, QUERY_LATENCY_CONCURRENCY
    )
    manifest = EmbeddingRunManifest(
        schema_version=EMBEDDING_RUN_MANIFEST_SCHEMA_VERSION,
        spec=arm,
        settings=_settings(embedder, probe_set, mem0_project_id),
        effective_embedder=_effective_embedder(arm, base_config),
        graph_reembed=graph.reembed,
        replica_reembed=replica,
        query_failures=_failure_counts(graph, mem0, latency),
        check_results=pre_run + graph.checks,
        started_at=started_at,
        finished_at=datetime.now(UTC),
    )
    _write_artifacts(run_dir, manifest, _records(manifest, graph.probes, mem0, latency))
    return manifest


# --- refusals: everything here runs before any store call ------------------------------


def _require_embedding_spec(
    spec: LlmArmSpec | EmbeddingArmSpec, embedder: ArmEmbedder
) -> EmbeddingArmSpec:
    if not isinstance(spec, EmbeddingArmSpec):
        raise EmbeddingRunRefused(
            f'arm {spec.arm_id!r} is an {spec.axis} arm; an embedding run drives embedding arms'
        )
    if embedder.spec != spec:
        raise EmbeddingRunRefused(
            f'arm {spec.arm_id!r} was handed the embedder of arm {embedder.spec.arm_id!r}'
        )
    return spec


def _require_runnable(
    arm: EmbeddingArmSpec, probe_set: ProbeSet, mem0_snapshot: Path, run_dir: Path
) -> Mem0Snapshot:
    """The pinned Mem0 snapshot, once the run dir is empty and the probe set is the arm's."""
    if run_dir.exists() and any(run_dir.iterdir()):
        held = sorted(path.name for path in run_dir.iterdir())
        raise EmbeddingRunRefused(f'{run_dir} is not empty ({held}); a run dir is never reused')
    probe_sha = probe_set_sha(serialize_probe_set(probe_set).encode())
    if arm.corpus_sha != probe_sha:
        raise EmbeddingRunRefused(
            f'arm {arm.arm_id!r} names corpus_sha {arm.corpus_sha}, '
            f'but its probe set hashes to {probe_sha}'
        )
    return _pinned_snapshot(mem0_snapshot, probe_set.mem0_snapshot)


def _pinned_snapshot(path: Path, pin: Mem0SnapshotPin) -> Mem0Snapshot:
    actual = snapshot_sha(path)
    if actual != pin.sha256:
        raise EmbeddingRunRefused(
            f'Mem0 snapshot {path} hashes to {actual}, not the probe set pin {pin.sha256}'
        )
    return load_snapshot(path)


# --- the Mem0 replica ---------------------------------------------------------------------


async def _replica_phase(
    arm: EmbeddingArmSpec,
    items: Sequence[Mem0KnownItem],
    qdrant: ReplicaClient,
    embedder: ArmEmbedder,
    snapshot: Mem0Snapshot,
    rank_mem0: Mem0Ranker,
    project_id: str,
) -> tuple[ReplicaReembed, ProbeTally]:
    name = arm.scratch_group_id
    reembed = await build_replica(qdrant, name, snapshot.records, embedder, dim=arm.embedding_dim)
    outcomes = [
        await _mem0_rank(qdrant, embedder, name, project_id, item, rank_mem0) for item in items
    ]
    return reembed, tally_probe(outcomes)


async def _mem0_rank(
    qdrant: ReplicaClient,
    embedder: ArmEmbedder,
    name: str,
    project_id: str,
    item: Mem0KnownItem,
    rank_mem0: Mem0Ranker,
) -> tuple[Rank, bool]:
    try:
        vector, _ = await embedder.embed_query(item.phrasing)
    except QueryEmbedError as error:
        logger.warning('%s: Mem0 topic %s reads as a miss: %s', name, item.topic, error)
        return None, True
    hits = await search_replica(qdrant, name, vector, limit=SEARCH_K, project_id=project_id)
    return rank_mem0(item.topic, hits), False


# --- the manifest and the records --------------------------------------------------------


def _settings(
    embedder: ArmEmbedder, probe_set: ProbeSet, mem0_project_id: str
) -> EmbeddingRunSettings:
    return EmbeddingRunSettings(
        embed_batch_size=embedder.settings.batch_size,
        embed_concurrency=embedder.settings.concurrency,
        query_concurrency=QUERY_LATENCY_CONCURRENCY,
        search_k=SEARCH_K,
        transcript_queries=len(probe_set.transcript.queries),
        mem0_project_id=mem0_project_id,
    )


def _effective_embedder(arm: EmbeddingArmSpec, base_config: FusedMemoryConfig) -> EffectiveEmbedder:
    embedder = embedding_arm_config(arm, base_config).embedder
    return EffectiveEmbedder(model=embedder.model, dimensions=embedder.dimensions)


def _failure_counts(
    graph: GraphPhase, mem0: ProbeTally, latency: QueryLatency
) -> QueryFailureCounts:
    return QueryFailureCounts(
        known_item={configuration: probe.failures for configuration, probe in graph.probes.items()},
        mem0_known_item=mem0.failures,
        query_latency=latency.failures,
    )


def _records(
    manifest: EmbeddingRunManifest,
    probes: Mapping[IndexConfiguration, ProbeTally],
    mem0: ProbeTally,
    latency: QueryLatency,
) -> tuple[MetricsRecord, ...]:
    def record(metric: Metric, configuration: IndexConfiguration | None = None) -> MetricsRecord:
        return record_for(
            manifest.spec,
            metric,
            measured_at=manifest.finished_at,
            incomplete=False,
            index_configuration=configuration,
        )

    per_configuration = [
        record(metric, configuration)
        for configuration, probe in probes.items()
        for metric in _known_item_metrics(probe.ranks, _GRAPH_METRIC_IDS)
    ]
    store_level = (
        *_known_item_metrics(mem0.ranks, _MEM0_METRIC_IDS),
        latency.metric,
        _reembed_throughput(manifest.graph_reembed, manifest.replica_reembed),
    )
    return (*per_configuration, *(record(metric) for metric in store_level if metric is not None))


def _known_item_metrics(ranks: Sequence[Rank], ids: _KnownItemMetricIds) -> tuple[Metric, ...]:
    metrics = (
        recall_metric(ids.recall_at_5, ranks, 5),
        recall_metric(ids.recall_at_10, ranks, 10),
        mrr_metric(ranks, metric_id=ids.mrr),
    )
    return tuple(metric for metric in metrics if metric is not None)


def _reembed_throughput(graph: GraphReembed, replica: ReplicaReembed) -> Metric | None:
    """Vectors embedded per embedding second, both stores together; write seconds excluded."""
    seconds = graph.embed_seconds + replica.embed_seconds
    if seconds <= 0:
        return None
    vectors = graph.written + replica.written
    return Metric(
        metric_id=EmbeddingMetricId.REEMBED_THROUGHPUT,
        kind='scalar',
        value=vectors / seconds,
        n=vectors,
    )


def _write_artifacts(
    run_dir: Path, manifest: EmbeddingRunManifest, records: Sequence[MetricsRecord]
) -> None:
    for metrics_record in records:
        write_metrics_record(metrics_record, run_dir)
    atomic_write_text(
        run_dir / RUN_MANIFEST_FILENAME, serialize_embedding_run_manifest(manifest), mkdir=True
    )
