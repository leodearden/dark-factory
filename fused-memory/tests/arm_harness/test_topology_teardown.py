"""Frozen-graph re-embed integrity (boundary row 5's harness logic) and guarded teardown."""

import re
from types import SimpleNamespace
from typing import Any

import pytest

from arm_harness._fakes import PROTECTED_GRAPHS, embedding_spec, llm_spec
from fused_memory.arm_harness.arm_spec import LlmArmSpec
from fused_memory.arm_harness.scratch_guard import GuardCheckpoint, ScratchGuardError
from fused_memory.arm_harness.teardown import (
    delete_replica_collection,
    delete_scratch_graph,
    teardown_arm,
)
from fused_memory.arm_harness.topology import (
    EDGE_CYPHER,
    NODE_CYPHER,
    Topology,
    TopologyEdge,
    TopologyNode,
    check_reembed_integrity,
    read_topology,
    topology_hash,
)

SCRATCH = 'evalmem_frozen_ref'
NON_SCRATCH_NAMES = [*PROTECTED_GRAPHS, 'fused_dark_factory', 'evalmem_*', ['evalmem_a']]


class FakeTopologyGraph:
    """An AsyncGraph double answering only the two topology queries, read-only."""

    def __init__(self, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
        self._rows = {NODE_CYPHER: nodes, EDGE_CYPHER: edges}
        self.queries: list[str] = []

    async def ro_query(self, cypher: str, params: dict | None = None) -> Any:
        self.queries.append(cypher)
        return SimpleNamespace(result_set=[[row] for row in self._rows[cypher]])


def _node_rows(*, dim: int | None = None) -> list[dict[str, Any]]:
    rows = [
        {'uuid': 'n1', 'labels': ['Entity'], 'name': 'alice'},
        {'uuid': 'n2', 'labels': ['Entity', 'Person'], 'name': 'bob'},
        {'uuid': 'ep1', 'labels': ['Episodic'], 'name': 'episode one'},
    ]
    if dim is not None:
        for row in rows:
            row['name_embedding'] = [0.5] * dim
    return rows


def _edge_rows(*, dim: int | None = None) -> list[dict[str, Any]]:
    rows = [
        {
            'uuid': 'e1', 'rel_type': 'RELATES_TO', 'source_uuid': 'n1', 'target_uuid': 'n2',
            'name': 'KNOWS', 'fact': 'alice knows bob',
        },
        {
            'uuid': 'm1', 'rel_type': 'MENTIONS', 'source_uuid': 'ep1', 'target_uuid': 'n1',
            'name': None, 'fact': None,
        },
    ]
    if dim is not None:
        for row in rows:
            row['fact_embedding'] = [0.25] * dim
    return rows


async def _read(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> Topology:
    return await read_topology(FakeTopologyGraph(nodes, edges), SCRATCH)


# --- read_topology -------------------------------------------------------------------


def test_topology_queries_project_only_identity_and_content():
    for cypher in (NODE_CYPHER, EDGE_CYPHER):
        assert 'embedding' not in cypher
        assert not re.search(r'RETURN\s+[a-z]+\s*$', cypher)


@pytest.mark.asyncio
async def test_read_topology_runs_both_queries_read_only():
    graph = FakeTopologyGraph(_node_rows(), _edge_rows())

    await read_topology(graph, SCRATCH)

    assert sorted(graph.queries) == sorted([NODE_CYPHER, EDGE_CYPHER])


@pytest.mark.asyncio
async def test_read_topology_drops_embedding_properties_from_the_records():
    nodes, edges = await _read(_node_rows(dim=1536), _edge_rows(dim=1536))

    assert TopologyNode(uuid='n2', labels=('Entity', 'Person'), name='bob') in nodes
    assert TopologyEdge(
        uuid='m1', rel_type='MENTIONS', source_uuid='ep1', target_uuid='n1', name=None, fact=None
    ) in edges
    assert all('embedding' not in repr(record) for record in (*nodes, *edges))


@pytest.mark.asyncio
@pytest.mark.parametrize('name', PROTECTED_GRAPHS)
async def test_read_topology_refuses_a_live_graph_before_any_query(name):
    graph = FakeTopologyGraph(_node_rows(), _edge_rows())

    with pytest.raises(ScratchGuardError) as raised:
        await read_topology(graph, name)

    assert raised.value.checkpoint is GuardCheckpoint.TOPOLOGY_READ
    assert graph.queries == []


# --- boundary row 5: re-embed integrity ------------------------------------------------


@pytest.mark.asyncio
async def test_embedding_only_differences_hash_identically_across_dimensions():
    reference = await _read(_node_rows(dim=1536), _edge_rows(dim=1536))
    candidate = await _read(_node_rows(dim=1024), _edge_rows(dim=1024))
    bare = await _read(_node_rows(), _edge_rows())

    assert topology_hash(*reference) == topology_hash(*candidate) == topology_hash(*bare)
    verdict = check_reembed_integrity(reference, candidate)
    assert verdict.identical
    assert (verdict.node_count, verdict.edge_count) == (3, 2)
    assert verdict.missing_node_uuids == verdict.extra_node_uuids == ()
    assert verdict.changed_node_uuids == verdict.changed_edge_uuids == ()


@pytest.mark.asyncio
async def test_record_and_label_order_do_not_change_the_hash():
    reference = await _read(_node_rows(), _edge_rows())
    shuffled_nodes = list(reversed(_node_rows()))
    shuffled_nodes[0] = shuffled_nodes[0] | {'labels': ['Episodic']}
    shuffled_nodes[1]['labels'] = ['Person', 'Entity']
    shuffled = await _read(shuffled_nodes, list(reversed(_edge_rows())))

    assert topology_hash(*reference) == topology_hash(*shuffled)


def test_topology_hash_is_sha256_hex():
    assert re.fullmatch(r'[0-9a-f]{64}', topology_hash((), ()))


def _without(rows: list[dict[str, Any]], uuid: str) -> list[dict[str, Any]]:
    return [row for row in rows if row['uuid'] != uuid]


def _changed(rows: list[dict[str, Any]], uuid: str, **update: Any) -> list[dict[str, Any]]:
    return [row | update if row['uuid'] == uuid else row for row in rows]


_ADDED_EDGE = {
    'uuid': 'e9', 'rel_type': 'RELATES_TO', 'source_uuid': 'n2', 'target_uuid': 'n1',
    'name': 'LIKES', 'fact': 'bob likes alice',
}
_EXTRA_NODE = {'uuid': 'n9', 'labels': ['Entity'], 'name': 'carol'}

_CHANGES = {
    'removed node': ((_without(_node_rows(), 'n2'), _edge_rows()), 'missing_node_uuids', 'n2'),
    'added node': (([*_node_rows(), _EXTRA_NODE], _edge_rows()), 'extra_node_uuids', 'n9'),
    'added edge': ((_node_rows(), [*_edge_rows(), _ADDED_EDGE]), 'changed_edge_uuids', 'e9'),
    'changed endpoint': (
        (_node_rows(), _changed(_edge_rows(), 'e1', target_uuid='ep1')), 'changed_edge_uuids', 'e1'
    ),
    'changed fact': (
        (_node_rows(), _changed(_edge_rows(), 'e1', fact='alice met bob')),
        'changed_edge_uuids',
        'e1',
    ),
    'changed node name': (
        (_changed(_node_rows(), 'n1', name='alicia'), _edge_rows()), 'changed_node_uuids', 'n1'
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize('change', sorted(_CHANGES))
async def test_a_topology_change_changes_the_hash_and_names_the_uuid(change):
    (nodes, edges), verdict_field, uuid = _CHANGES[change]
    reference = await _read(_node_rows(dim=1536), _edge_rows(dim=1536))
    candidate = await _read(nodes, edges)

    assert topology_hash(*reference) != topology_hash(*candidate)
    verdict = check_reembed_integrity(reference, candidate)
    assert not verdict.identical
    assert getattr(verdict, verdict_field) == (uuid,)


# --- teardown --------------------------------------------------------------------------


class FakeFalkorClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def select_graph(self, name: object) -> Any:
        self.calls.append(('select_graph', name))

        async def delete() -> None:
            self.calls.append(('delete', name))

        return SimpleNamespace(delete=delete)


class FakeQdrantClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def delete_collection(self, name: object) -> bool:
        self.calls.append(('delete_collection', name))
        return True


@pytest.mark.asyncio
async def test_delete_scratch_graph_deletes_exactly_the_named_graph():
    client = FakeFalkorClient()

    await delete_scratch_graph(client, 'evalmem_qwen3_8b')

    assert client.calls == [('select_graph', 'evalmem_qwen3_8b'), ('delete', 'evalmem_qwen3_8b')]


@pytest.mark.asyncio
@pytest.mark.parametrize('name', NON_SCRATCH_NAMES, ids=repr)
async def test_delete_scratch_graph_refuses_a_non_scratch_name_without_any_call(name):
    client = FakeFalkorClient()

    with pytest.raises(ScratchGuardError) as raised:
        await delete_scratch_graph(client, name)

    assert raised.value.checkpoint is GuardCheckpoint.TEARDOWN_GRAPH
    assert client.calls == []


@pytest.mark.asyncio
async def test_delete_replica_collection_deletes_exactly_the_named_collection():
    client = FakeQdrantClient()

    await delete_replica_collection(client, 'evalmem_bge_m3')

    assert client.calls == [('delete_collection', 'evalmem_bge_m3')]


@pytest.mark.asyncio
@pytest.mark.parametrize('name', NON_SCRATCH_NAMES, ids=repr)
async def test_delete_replica_collection_refuses_a_non_scratch_name_without_any_call(name):
    client = FakeQdrantClient()

    with pytest.raises(ScratchGuardError) as raised:
        await delete_replica_collection(client, name)

    assert raised.value.checkpoint is GuardCheckpoint.TEARDOWN_COLLECTION
    assert client.calls == []


@pytest.mark.asyncio
async def test_teardown_arm_deletes_the_scratch_graph_and_its_replica_collection():
    falkor, qdrant = FakeFalkorClient(), FakeQdrantClient()

    await teardown_arm(falkor, qdrant, embedding_spec())

    assert falkor.calls == [('select_graph', 'evalmem_bge_m3'), ('delete', 'evalmem_bge_m3')]
    assert qdrant.calls == [('delete_collection', 'evalmem_bge_m3')]


@pytest.mark.asyncio
async def test_teardown_arm_without_qdrant_deletes_only_the_graph():
    falkor = FakeFalkorClient()

    await teardown_arm(falkor, None, llm_spec())

    assert falkor.calls == [('select_graph', 'evalmem_qwen3_8b'), ('delete', 'evalmem_qwen3_8b')]


@pytest.mark.asyncio
async def test_teardown_arm_rechecks_a_name_that_bypassed_spec_validation():
    bypassed = LlmArmSpec.model_construct(
        **(llm_spec().model_dump() | {'scratch_group_id': 'dark_factory'})
    )
    falkor, qdrant = FakeFalkorClient(), FakeQdrantClient()

    with pytest.raises(ScratchGuardError):
        await teardown_arm(falkor, qdrant, bypassed)

    assert falkor.calls == qdrant.calls == []
