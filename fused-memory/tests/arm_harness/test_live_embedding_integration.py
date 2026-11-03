"""Live FalkorDB behaviour the embedding-run fakes assume: COPY, group adoption, re-embed, index drop and rebuild.

Every graph is a fresh ``evalmem_test_`` scratch graph, deleted in ``finally``.
Deselected by the default addopts; run it with ``uv run pytest -m integration
tests/arm_harness/test_live_embedding_integration.py``.
"""

import contextlib
import hashlib
import math
import uuid
from collections.abc import AsyncIterator, Iterable
from typing import Any

import pytest
from _fm_helpers import FALKOR_HOST, FALKOR_PORT, await_index_operational, falkor_skipif
from falkordb.asyncio import FalkorDB
from falkordb.asyncio.graph import AsyncGraph
from graphiti_core.embedder import EmbedderClient

from arm_harness._fakes import PROTECTED_GRAPHS, embedding_spec
from arm_harness.test_live_replay_integration import (
    CONCURRENT_TEST_GRAPH_PREFIXES,
    INDEX_DEFINITION_QUERY,
    _protected_index_definitions,
)
from fused_memory.arm_harness.arm_backend import build_scratch_indices, scratch_graphiti_backend
from fused_memory.arm_harness.arm_embedder import ArmEmbedder, EmbedSettings
from fused_memory.arm_harness.checks import check_index_configuration
from fused_memory.arm_harness.graph_copy import adopt_group_id, copy_reference_graph, reembed_graph
from fused_memory.arm_harness.metrics_record import IndexConfiguration
from fused_memory.arm_harness.scratch_indices import drop_all_indices
from fused_memory.arm_harness.teardown import delete_scratch_graph
from fused_memory.arm_harness.topology import check_reembed_integrity, read_topology, topology_hash
from fused_memory.config.schema import (
    FalkorDBProviderConfig,
    FusedMemoryConfig,
    GraphitiBackendConfig,
)

pytestmark = [falkor_skipif(), pytest.mark.timeout(300)]

ARM_DIM = 4
INCUMBENT_DIM = 8
FACT = 'the merge worker waits\non the index lock'
RAW_SCALE = 30.0
"""Arm vectors come back at norm 30, as granite's do, so the unit seam has work to do."""

SEED_CYPHER = (
    'CREATE (a:Entity {uuid: "n-merge", name: "merge worker", summary: "s", group_id: $g, '
    'name_embedding: vecf32($v)}), '
    '(b:Entity {uuid: "n-lock", name: "index lock", summary: "s", group_id: $g, '
    'name_embedding: vecf32($v)}), '
    '(e:Episodic {uuid: "ep-1", name: "ep-1", content: "the merge worker waits on the lock", '
    'group_id: $g}), '
    '(a)-[:RELATES_TO {uuid: "r-1", name: "WAITS_ON", fact: $fact, group_id: $g, '
    'episodes: ["ep-1"], fact_embedding: vecf32($v)}]->(b), '
    '(e)-[:MENTIONS {uuid: "m-1", group_id: $g}]->(a)'
)
VECTORS_CYPHER = (
    'MATCH (n:Entity) RETURN n.uuid AS uuid, n.name_embedding AS vector '
    'UNION ALL MATCH ()-[r:RELATES_TO]->() RETURN r.uuid AS uuid, r.fact_embedding AS vector'
)


class DeterministicInner(EmbedderClient):
    """A served model stand-in: each text's raw vector is a hash of it, at norm RAW_SCALE."""

    async def create(self, input_data: Any) -> list[float]:
        [text] = input_data
        return _raw(text)

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return [_raw(text) for text in input_data_list]


def _raw(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    components = [digest[index] - 127.5 for index in range(ARM_DIM)]
    norm = math.sqrt(sum(value * value for value in components))
    return [value * RAW_SCALE / norm for value in components]


def _scratch_name() -> str:
    return f'evalmem_test_{uuid.uuid4().hex[:8]}'


def _base(mock_config: FusedMemoryConfig) -> FusedMemoryConfig:
    falkordb = FalkorDBProviderConfig(uri=f'redis://{FALKOR_HOST}:{FALKOR_PORT}')
    return mock_config.model_copy(
        update={'graphiti': GraphitiBackendConfig(falkordb=falkordb)}, deep=True
    )


@contextlib.asynccontextmanager
async def _index_builder(base: FusedMemoryConfig) -> AsyncIterator[Any]:
    backend = scratch_graphiti_backend(base)
    await backend.initialize(skip_maintenance=True)
    try:
        yield backend
    finally:
        await backend.close()


async def _definitions(graph: AsyncGraph) -> list[Any]:
    return (await graph.ro_query(INDEX_DEFINITION_QUERY)).result_set


async def _vectors(graph: AsyncGraph) -> dict[str, list[float]]:
    return {key: list(vector) for key, vector in (await graph.ro_query(VECTORS_CYPHER)).result_set}


def _norm(vector: Iterable[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


async def _seed_reference(graph: AsyncGraph, name: str, builder: Any) -> None:
    await graph.query(SEED_CYPHER, {'g': name, 'v': [0.5] * INCUMBENT_DIM, 'fact': FACT})
    await build_scratch_indices(builder, name)
    await await_index_operational(graph)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_copy_is_re_embedded_and_moved_through_both_index_configurations(mock_config):
    client = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    reference, copy = _scratch_name(), _scratch_name()
    graphs_before = set(await client.list_graphs())
    protected_before = await _protected_index_definitions(client, graphs_before)
    base = _base(mock_config)
    try:
        async with _index_builder(base) as builder:
            source = client.select_graph(reference)
            await _seed_reference(source, reference, builder)
            source_definitions = await _definitions(source)

            # (2) COPY keeps the topology; whether it carries the source's index catalog
            await copy_reference_graph(client, reference, copy)
            target = client.select_graph(copy)
            reference_topology = await read_topology(source, reference)
            assert topology_hash(*await read_topology(target, copy)) == topology_hash(
                *reference_topology
            )
            assert await _definitions(target) == source_definitions, (
                'GRAPH.COPY no longer carries the source index catalog'
            )

            # (3) group adoption and re-embed at the arm's dimension, unit-normalised
            await adopt_group_id(target, copy)
            groups = (await target.ro_query('MATCH (n) RETURN DISTINCT n.group_id')).result_set
            assert groups == [[copy]]
            spec = embedding_spec(embedding_dim=ARM_DIM, scratch_group_id=copy)
            embedder = ArmEmbedder(
                DeterministicInner(), spec, EmbedSettings(batch_size=2, concurrency=1)
            )
            reembed = await reembed_graph(target, copy, embedder)
            assert reembed.written == 3
            assert reembed.raw_norms is not None
            assert reembed.raw_norms.max == pytest.approx(RAW_SCALE)
            vectors = await _vectors(target)
            assert sorted(vectors) == ['n-lock', 'n-merge', 'r-1']
            for vector in vectors.values():
                assert len(vector) == ARM_DIM
                assert _norm(vector) == pytest.approx(1.0, abs=1e-6)
            assert vectors['r-1'] == pytest.approx(
                [value / RAW_SCALE for value in _raw(FACT.replace('\n', ' '))],
                abs=1e-6,
            )
            integrity = check_reembed_integrity(
                reference_topology, await read_topology(target, copy)
            )
            assert integrity.identical, integrity
            scaled = [value * RAW_SCALE for value in vectors['n-merge']]
            distance = (await target.ro_query(
                'MATCH (n:Entity {uuid: "n-merge"}) '
                'RETURN vec.cosineDistance(n.name_embedding, vecf32($v))',
                {'v': scaled},
            )).result_set[0][0]
            assert distance == pytest.approx(0.0, abs=1e-6)

            # (4) embedding-only: an empty catalog, and the fulltext probe agrees
            dropped = await drop_all_indices(target, copy)
            assert dropped
            assert (await target.list_indices()).result_set == []
            bare = await check_index_configuration(
                target, copy, IndexConfiguration.EMBEDDING_ONLY
            )
            assert bare.passed, bare.detail
            assert await _definitions(source) == source_definitions

            # (5) with-indices: ensure_indices rebuilds, and the fulltext probe agrees
            await build_scratch_indices(builder, copy)
            await await_index_operational(target)
            indexed = await check_index_configuration(
                target, copy, IndexConfiguration.WITH_INDICES
            )
            assert indexed.passed, indexed.detail
    finally:
        for name in (reference, copy):
            with contextlib.suppress(Exception):
                await delete_scratch_graph(client, name)
        graphs_after = set(await client.list_graphs())
        protected_after = await _protected_index_definitions(client, graphs_after)
        await client.aclose()

    # (6) only scratch graphs came and went; no protected graph's index catalog moved
    changed = graphs_before ^ graphs_after
    assert not changed & set(PROTECTED_GRAPHS)
    assert not {reference, copy} & graphs_after
    assert {name for name in changed if not name.startswith(CONCURRENT_TEST_GRAPH_PREFIXES)} == set()
    assert protected_after == protected_before
