"""The graph half of an embedding arm run: the frozen reference re-checked, the arm's re-embedded copy made, and each index configuration entered, verified and probed.

A failed instrument check raises ``EmbeddingRunCheckFailed`` at once, so a probe
never runs on a configuration that does not differ in fact. Every known-item
search is the production ``GraphitiBackend.search`` on the arm's scratch graph.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from fused_memory.arm_harness.arm_backend import build_scratch_indices
from fused_memory.arm_harness.arm_embedder import ArmEmbedder, QueryEmbedError
from fused_memory.arm_harness.arm_spec import EmbeddingArmSpec
from fused_memory.arm_harness.checks import check_index_configuration
from fused_memory.arm_harness.graph_copy import (
    GraphReembed,
    adopt_group_id,
    copy_reference_graph,
    reembed_graph,
)
from fused_memory.arm_harness.instrument_checks import (
    CheckResult,
    InstrumentCheckId,
    check_failed,
    check_passed,
)
from fused_memory.arm_harness.metrics_record import IndexConfiguration
from fused_memory.arm_harness.probe_set import FrozenReference, KnownItem, ProbeSet
from fused_memory.arm_harness.retrieval import (
    RETRIEVAL_UTILITY_K,
    ProbeTally,
    Rank,
    known_item_rank,
    provenance_matcher,
    tally_probe,
)
from fused_memory.arm_harness.scratch_indices import drop_all_indices
from fused_memory.arm_harness.topology import (
    IntegrityVerdict,
    Topology,
    check_reembed_integrity,
    read_topology,
    topology_hash,
)
from fused_memory.backends.falkor_indices import IndexProvisionResult

logger = logging.getLogger(__name__)

SEARCH_K = RETRIEVAL_UTILITY_K
CONFIGURATION_ORDER = (IndexConfiguration.EMBEDDING_ONLY, IndexConfiguration.WITH_INDICES)
"""Embedding-only first: whatever indices COPY carried are dropped, then built afresh."""

Sleep = Callable[[float], Awaitable[object]]


class ArmScratchGraph(Protocol):
    """The slice of falkordb's AsyncGraph a run uses on the reference and on the arm's copy."""

    async def query(self, q: str, params: dict[str, Any] | None = None) -> Any: ...

    async def ro_query(self, q: str, params: dict[str, Any] | None = None) -> Any: ...

    async def list_indices(self) -> Any: ...

    async def drop_node_range_index(self, label: str, attribute: str) -> object: ...

    async def drop_node_fulltext_index(self, label: str, attribute: str) -> object: ...

    async def drop_edge_range_index(self, label: str, attribute: str) -> object: ...

    async def drop_edge_fulltext_index(self, label: str, attribute: str) -> object: ...

    async def copy(self, clone: str, /) -> object: ...


class ArmGraphClient(Protocol):
    """The slice of falkordb's async FalkorDB client a run uses."""

    async def list_graphs(self) -> list[str]: ...

    def select_graph(self, graph_id: str, /) -> ArmScratchGraph: ...


class EmbeddingSearchBackend(Protocol):
    """GraphitiBackend's production search, embedding each query through the arm, and its index build."""

    async def search(
        self, query: str, group_ids: list[str] | None = None, num_results: int = 10
    ) -> list[Any]: ...

    async def ensure_indices(self, *, group_id: str) -> IndexProvisionResult: ...


class EmbeddingRunCheckFailed(RuntimeError):
    """An instrument check failed mid-run, so the run stopped and wrote no ``run.json``."""

    def __init__(self, arm_id: str, check_result: CheckResult) -> None:
        self.check_result = check_result
        super().__init__(f'arm {arm_id!r} stopped: {check_result.check_id}: {check_result.detail}')


@dataclass(frozen=True)
class GraphPhase:
    reembed: GraphReembed
    probes: Mapping[IndexConfiguration, ProbeTally]
    checks: tuple[CheckResult, ...]
    """Frozen reference, copy integrity, each configuration's check, integrity after the probes."""


async def run_graph_phase(
    arm: EmbeddingArmSpec,
    probe_set: ProbeSet,
    client: ArmGraphClient,
    backend: EmbeddingSearchBackend,
    embedder: ArmEmbedder,
    *,
    sleep: Sleep = asyncio.sleep,
) -> GraphPhase:
    name = arm.scratch_group_id
    reference, frozen = await _require_frozen_reference(arm, client, probe_set.reference)
    await copy_reference_graph(client, probe_set.reference.graph, name)
    graph = client.select_graph(name)
    await adopt_group_id(graph, name)
    reembed = await reembed_graph(graph, name, embedder)
    copied = await _require_integrity(arm, graph, reference)
    probes: dict[IndexConfiguration, ProbeTally] = {}
    configuration_checks: list[CheckResult] = []
    for configuration in CONFIGURATION_ORDER:
        await _enter_configuration(configuration, graph, backend, name)
        check = await check_index_configuration(graph, name, configuration, sleep=sleep)
        configuration_checks.append(_require_passed(arm, check))
        probes[configuration] = await _known_item_probe(backend, name, probe_set.known_items)
    kept = await _require_integrity(arm, graph, reference)
    checks = (frozen, copied, *configuration_checks, kept)
    return GraphPhase(reembed=reembed, probes=probes, checks=checks)


async def _require_frozen_reference(
    arm: EmbeddingArmSpec, client: ArmGraphClient, reference: FrozenReference
) -> tuple[Topology, CheckResult]:
    live = await read_topology(client.select_graph(reference.graph), reference.graph)
    live_hash = topology_hash(*live)
    check_id = InstrumentCheckId.FROZEN_REFERENCE_UNCHANGED
    if live_hash != reference.topology_hash:
        check = check_failed(
            check_id,
            f'{reference.graph!r} hashes to {live_hash} live, not the probe set pin '
            f'{reference.topology_hash}: the graph every arm copies has changed',
            (reference.graph,),
        )
    else:
        check = check_passed(check_id, f'{reference.graph!r} still hashes to {live_hash}')
    return live, _require_passed(arm, check)


async def _require_integrity(
    arm: EmbeddingArmSpec, graph: ArmScratchGraph, reference: Topology
) -> CheckResult:
    name = arm.scratch_group_id
    verdict = check_reembed_integrity(reference, await read_topology(graph, name))
    return _require_passed(arm, _integrity_check(name, verdict))


def _integrity_check(name: str, verdict: IntegrityVerdict) -> CheckResult:
    check_id = InstrumentCheckId.REEMBED_INTEGRITY
    if verdict.identical:
        return check_passed(
            check_id,
            f'{name!r} keeps the reference topology: {verdict.node_count} nodes, '
            f'{verdict.edge_count} edges',
        )
    detail = (
        f'{name!r} differs from the reference: missing nodes {list(verdict.missing_node_uuids)}, '
        f'extra nodes {list(verdict.extra_node_uuids)}, changed nodes '
        f'{list(verdict.changed_node_uuids)}, changed edges {list(verdict.changed_edge_uuids)}'
    )
    offenders = (
        *verdict.missing_node_uuids,
        *verdict.extra_node_uuids,
        *verdict.changed_node_uuids,
        *verdict.changed_edge_uuids,
    )
    return check_failed(check_id, detail, offenders)


def _require_passed(arm: EmbeddingArmSpec, check: CheckResult) -> CheckResult:
    if not check.passed:
        raise EmbeddingRunCheckFailed(arm.arm_id, check)
    return check


async def _enter_configuration(
    configuration: IndexConfiguration,
    graph: ArmScratchGraph,
    backend: EmbeddingSearchBackend,
    name: str,
) -> None:
    if configuration is IndexConfiguration.EMBEDDING_ONLY:
        await drop_all_indices(graph, name)
    else:
        await build_scratch_indices(backend, name)


async def _known_item_probe(
    backend: EmbeddingSearchBackend, name: str, items: Sequence[KnownItem]
) -> ProbeTally:
    """Sequential over the items, so one query's latency never competes with another's."""
    return tally_probe([await _known_item_rank(backend, name, item) for item in items])


async def _known_item_rank(
    backend: EmbeddingSearchBackend, name: str, item: KnownItem
) -> tuple[Rank, bool]:
    try:
        results = await backend.search(item.query, group_ids=[name], num_results=SEARCH_K)
    except QueryEmbedError as error:
        logger.warning('%s: known item %s reads as a miss: %s', name, item.episode_uuid, error)
        return None, True
    return known_item_rank(results, provenance_matcher(item.episode_uuid)), False
