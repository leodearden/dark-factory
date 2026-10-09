"""An arm's re-embedded copy of the frozen reference graph (arm_harness/graph_copy.py)."""

import copy
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from arm_harness._fakes import PROTECTED_GRAPHS
from fused_memory.arm_harness import graph_copy
from fused_memory.arm_harness.arm_embedder import DocumentEmbeddings
from fused_memory.arm_harness.graph_copy import (
    GraphReembed,
    ReembedCensusError,
    adopt_group_id,
    copy_reference_graph,
    reembed_graph,
)
from fused_memory.arm_harness.normalization import NormStats
from fused_memory.arm_harness.scratch_guard import GuardCheckpoint, ScratchGuardError

REFERENCE = 'evalmem_lme_ref_incumbent_a'
TARGET = 'evalmem_lme_emb_granite'
INCUMBENT_VECTOR = (0.5,) * 4


def _result(*rows: Sequence[Any]) -> SimpleNamespace:
    return SimpleNamespace(result_set=[list(row) for row in rows])


class FakeGraph:
    """An in-memory FalkorDB graph answering graph_copy's Cypher constants; every query recorded."""

    def __init__(
        self,
        name: str,
        *,
        client: 'FakeGraphClient',
        nodes: Mapping[str, dict[str, Any]] | None = None,
        edges: Mapping[str, dict[str, Any]] | None = None,
        lossy: frozenset[str] = frozenset(),
    ) -> None:
        self.name = name
        self.client = client
        self.nodes = {uuid: dict(props) for uuid, props in (nodes or {}).items()}
        self.edges = {uuid: dict(props) for uuid, props in (edges or {}).items()}
        self.lossy = lossy
        self.queries: list[tuple[str, str, dict[str, Any] | None]] = []

    def _entities(self) -> dict[str, dict[str, Any]]:
        return {uuid: props for uuid, props in self.nodes.items() if 'Entity' in props['labels']}

    def _facts(self) -> dict[str, dict[str, Any]]:
        return {uuid: props for uuid, props in self.edges.items() if props['type'] == 'RELATES_TO'}

    def _set_all(self, records: Mapping[str, dict[str, Any]], key: str, value: Any):
        for props in records.values():
            props[key] = value
        return _result([len(records)])

    def _write(self, records: Mapping[str, dict[str, Any]], key: str, rows: list[dict[str, Any]]):
        written = 0
        for row in rows:
            if row['uuid'] in records and row['uuid'] not in self.lossy:
                records[row['uuid']][key] = tuple(row['v'])
                written += 1
        return _result([written])

    def _non_null(self, records: Mapping[str, dict[str, Any]], key: str):
        return _result([sum(1 for props in records.values() if props.get(key) is not None)])

    def _handlers(self) -> Mapping[str, Callable[[dict[str, Any]], SimpleNamespace]]:
        return {
            graph_copy.ADOPT_NODE_GROUP_CYPHER: lambda p: self._set_all(
                self.nodes, 'group_id', p['group_id']),
            graph_copy.ADOPT_EDGE_GROUP_CYPHER: lambda p: self._set_all(
                self.edges, 'group_id', p['group_id']),
            graph_copy.READ_ENTITY_NAMES_CYPHER: lambda p: _result(
                *([uuid, props['name']] for uuid, props in self._entities().items())),
            graph_copy.READ_EDGE_FACTS_CYPHER: lambda p: _result(
                *([uuid, props['fact']] for uuid, props in self._facts().items())),
            graph_copy.NULL_NAME_EMBEDDINGS_CYPHER: lambda p: self._set_all(
                self._entities(), 'name_embedding', None),
            graph_copy.NULL_FACT_EMBEDDINGS_CYPHER: lambda p: self._set_all(
                self._facts(), 'fact_embedding', None),
            graph_copy.WRITE_NAME_EMBEDDINGS_CYPHER: lambda p: self._write(
                self._entities(), 'name_embedding', p['rows']),
            graph_copy.WRITE_FACT_EMBEDDINGS_CYPHER: lambda p: self._write(
                self._facts(), 'fact_embedding', p['rows']),
            graph_copy.COUNT_NAME_EMBEDDINGS_CYPHER: lambda p: self._non_null(
                self._entities(), 'name_embedding'),
            graph_copy.COUNT_FACT_EMBEDDINGS_CYPHER: lambda p: self._non_null(
                self._facts(), 'fact_embedding'),
        }

    async def query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        self.queries.append(('query', cypher, params))
        return self._handlers()[cypher](params or {})

    async def ro_query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        self.queries.append(('ro_query', cypher, params))
        return self._handlers()[cypher](params or {})

    async def copy(self, clone: str) -> None:
        self.client.calls.append(('copy', self.name, clone))
        self.client.graphs[clone] = FakeGraph(
            clone, client=self.client, nodes=copy.deepcopy(self.nodes),
            edges=copy.deepcopy(self.edges),
        )


class FakeGraphClient:
    def __init__(self) -> None:
        self.graphs: dict[str, FakeGraph] = {}
        self.calls: list[tuple[str, ...]] = []

    async def list_graphs(self) -> list[str]:
        self.calls.append(('list_graphs',))
        return list(self.graphs)

    def select_graph(self, name: str) -> FakeGraph:
        self.calls.append(('select_graph', name))
        return self.graphs[name]


def _reference_graph(client: FakeGraphClient, *, entities: int = 5, facts: int = 3) -> FakeGraph:
    nodes = {
        f'n{i}': {'labels': ['Entity'], 'name': f'entity\n{i}', 'group_id': REFERENCE,
                  'name_embedding': INCUMBENT_VECTOR}
        for i in range(entities)
    }
    nodes['ep0'] = {'labels': ['Episodic'], 'content': 'an episode', 'group_id': REFERENCE}
    edges = {
        f'r{i}': {'type': 'RELATES_TO', 'fact': f'fact {i}', 'group_id': REFERENCE,
                  'fact_embedding': INCUMBENT_VECTOR, 'episodes': ['ep0']}
        for i in range(facts)
    }
    edges['m0'] = {'type': 'MENTIONS', 'group_id': REFERENCE}
    graph = FakeGraph(REFERENCE, client=client, nodes=nodes, edges=edges)
    client.graphs[REFERENCE] = graph
    return graph


def _scratch_graph(**kwargs) -> FakeGraph:
    client = FakeGraphClient()
    reference = _reference_graph(client, **kwargs)
    graph = FakeGraph(TARGET, client=client, nodes=reference.nodes, edges=reference.edges)
    client.graphs[TARGET] = graph
    return graph


class FakeEmbedder:
    """An ArmEmbedder stand-in: every text embeds to (1, 0, 0) with raw norm 2 + its index."""

    def __init__(self, *, failing: frozenset[str] = frozenset(), seconds: float = 0.5) -> None:
        self.failing = failing
        self.seconds = seconds
        self.calls: list[list[tuple[str, str]]] = []

    async def embed_documents(self, items: Sequence[tuple[str, str]]) -> DocumentEmbeddings:
        self.calls.append(list(items))
        ok = [(index, key) for index, (key, _) in enumerate(items) if key not in self.failing]
        return DocumentEmbeddings(
            vectors=MappingProxyType({key: (1.0, 0.0, 0.0) for _, key in ok}),
            raw_norms=MappingProxyType({key: 2.0 + index for index, key in ok}),
            embed_seconds=self.seconds,
            failures=tuple((key, 'RuntimeError') for key, _ in items if key in self.failing),
        )


class StepClock:
    def __init__(self, step: float) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def _writes(graph: FakeGraph, cypher: str) -> list[list[dict[str, Any]]]:
    return [params['rows'] for kind, text, params in graph.queries
            if text == cypher and params is not None]


# --- copy -------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_copy_reference_graph_copies_the_reference_into_a_fresh_target():
    client = FakeGraphClient()
    reference = _reference_graph(client)

    await copy_reference_graph(client, REFERENCE, TARGET)

    assert ('copy', REFERENCE, TARGET) in client.calls
    assert client.graphs[TARGET].nodes == reference.nodes
    assert client.graphs[TARGET].edges == reference.edges


@pytest.mark.asyncio
async def test_copy_reference_graph_refuses_to_copy_a_graph_onto_itself():
    client = FakeGraphClient()
    _reference_graph(client)

    with pytest.raises(ValueError, match=REFERENCE):
        await copy_reference_graph(client, REFERENCE, REFERENCE)

    assert not [call for call in client.calls if call[0] == 'copy']


@pytest.mark.asyncio
async def test_copy_reference_graph_refuses_an_existing_target_rather_than_reuse_it():
    client = FakeGraphClient()
    _reference_graph(client)
    client.graphs[TARGET] = FakeGraph(TARGET, client=client)

    with pytest.raises(ValueError, match=TARGET):
        await copy_reference_graph(client, REFERENCE, TARGET)

    assert not [call for call in client.calls if call[0] == 'copy']


@pytest.mark.asyncio
@pytest.mark.parametrize('protected', PROTECTED_GRAPHS)
@pytest.mark.parametrize('position', ['reference', 'target'])
async def test_copy_reference_graph_guards_both_names_before_any_call(protected, position):
    client = FakeGraphClient()
    names = {'reference': REFERENCE, 'target': TARGET} | {position: protected}

    with pytest.raises(ScratchGuardError) as caught:
        await copy_reference_graph(client, names['reference'], names['target'])

    assert caught.value.checkpoint is GuardCheckpoint.GRAPH_COPY
    assert client.calls == []


# --- group id ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adopt_group_id_sets_the_copys_name_on_every_node_and_edge():
    graph = _scratch_graph()

    adoption = await adopt_group_id(graph, TARGET)

    assert (adoption.node_count, adoption.edge_count) == (6, 4)
    assert {props['group_id'] for props in graph.nodes.values()} == {TARGET}
    assert {props['group_id'] for props in graph.edges.values()} == {TARGET}
    assert all(kind == 'query' and params == {'group_id': TARGET}
               for kind, _, params in graph.queries)


@pytest.mark.asyncio
@pytest.mark.parametrize('protected', PROTECTED_GRAPHS)
async def test_adopt_group_id_refuses_a_live_graph_before_any_query(protected):
    graph = _scratch_graph()

    with pytest.raises(ScratchGuardError) as caught:
        await adopt_group_id(graph, protected)

    assert caught.value.checkpoint is GuardCheckpoint.REEMBED
    assert graph.queries == []


# --- re-embed ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reembed_embeds_exactly_what_graphiti_embeds():
    graph = _scratch_graph()
    embedder = FakeEmbedder()

    await reembed_graph(graph, TARGET, embedder, write_batch_size=2)

    assert embedder.calls == [
        [(f'n{i}', f'entity\n{i}') for i in range(5)],
        [(f'r{i}', f'fact {i}') for i in range(3)],
    ]
    reads = [text for kind, text, _ in graph.queries if kind == 'ro_query']
    assert reads[:2] == [graph_copy.READ_ENTITY_NAMES_CYPHER, graph_copy.READ_EDGE_FACTS_CYPHER]


@pytest.mark.asyncio
async def test_reembed_nulls_every_incumbent_vector_before_writing_any():
    graph = _scratch_graph()

    await reembed_graph(graph, TARGET, FakeEmbedder(), write_batch_size=2)

    texts = [text for _, text, _ in graph.queries]
    first_write = min(
        texts.index(graph_copy.WRITE_NAME_EMBEDDINGS_CYPHER),
        texts.index(graph_copy.WRITE_FACT_EMBEDDINGS_CYPHER),
    )
    assert texts.index(graph_copy.NULL_NAME_EMBEDDINGS_CYPHER) < first_write
    assert texts.index(graph_copy.NULL_FACT_EMBEDDINGS_CYPHER) < first_write


@pytest.mark.asyncio
async def test_reembed_writes_unit_vectors_in_batches_through_vecf32():
    graph = _scratch_graph()

    await reembed_graph(graph, TARGET, FakeEmbedder(), write_batch_size=2)

    name_writes = _writes(graph, graph_copy.WRITE_NAME_EMBEDDINGS_CYPHER)
    fact_writes = _writes(graph, graph_copy.WRITE_FACT_EMBEDDINGS_CYPHER)
    assert [len(rows) for rows in name_writes] == [2, 2, 1]
    assert [len(rows) for rows in fact_writes] == [2, 1]
    assert name_writes[0][0] == {'uuid': 'n0', 'v': [1.0, 0.0, 0.0]}
    assert 'vecf32(row.v)' in graph_copy.WRITE_NAME_EMBEDDINGS_CYPHER
    assert 'vecf32(row.v)' in graph_copy.WRITE_FACT_EMBEDDINGS_CYPHER
    assert {props['name_embedding'] for uuid, props in graph.nodes.items()
            if uuid.startswith('n')} == {(1.0, 0.0, 0.0)}


@pytest.mark.asyncio
async def test_reembed_leaves_a_failed_item_null_not_incumbent():
    graph = _scratch_graph()

    reembed = await reembed_graph(
        graph, TARGET, FakeEmbedder(failing=frozenset({'n3', 'r1'})), write_batch_size=2
    )

    assert graph.nodes['n3']['name_embedding'] is None
    assert graph.edges['r1']['fact_embedding'] is None
    assert reembed.failures == (('n3', 'RuntimeError'), ('r1', 'RuntimeError'))
    assert reembed.written == 6


@pytest.mark.asyncio
async def test_reembed_reports_counts_norms_and_timings():
    graph = _scratch_graph()

    reembed = await reembed_graph(
        graph, TARGET, FakeEmbedder(seconds=0.5), write_batch_size=2, clock=StepClock(0.25)
    )

    assert isinstance(reembed, GraphReembed)
    assert (reembed.entity_count, reembed.edge_count, reembed.written) == (5, 3, 8)
    assert reembed.failures == ()
    assert reembed.raw_norms == NormStats(count=8, min=2.0, median=3.5, max=6.0)
    assert reembed.embed_seconds == pytest.approx(1.0)
    assert reembed.write_seconds == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_reembed_result_is_frozen():
    reembed = await reembed_graph(_scratch_graph(), TARGET, FakeEmbedder(), write_batch_size=2)

    with pytest.raises(ValueError):
        reembed.written = 0  # type: ignore[misc]


@pytest.mark.asyncio
async def test_a_census_short_of_the_written_count_is_refused_naming_both_counts():
    graph = _scratch_graph()
    graph.lossy = frozenset({'n2'})

    with pytest.raises(ReembedCensusError) as caught:
        await reembed_graph(graph, TARGET, FakeEmbedder(), write_batch_size=2)

    message = str(caught.value)
    assert 'name_embedding' in message
    assert '5' in message
    assert '4' in message


@pytest.mark.asyncio
@pytest.mark.parametrize('protected', PROTECTED_GRAPHS)
async def test_reembed_refuses_a_live_graph_before_any_query(protected):
    graph = _scratch_graph()
    embedder = FakeEmbedder()

    with pytest.raises(ScratchGuardError) as caught:
        await reembed_graph(graph, protected, embedder, write_batch_size=2)

    assert caught.value.checkpoint is GuardCheckpoint.REEMBED
    assert graph.queries == []
    assert embedder.calls == []
