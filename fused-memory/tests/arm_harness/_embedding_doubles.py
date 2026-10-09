"""In-memory doubles of every store and endpoint one embedding arm run touches.

FalkorDB, the GraphitiBackend search, Qdrant and the arm's served model, each built
consistently over one small frozen reference graph and Mem0 snapshot. The library
tests (test_embedding_run.py) hand them to ``run_embedding_arm`` directly; the CLI
tests (test_embedding_cli.py) hand them to ``harness.main`` through ``HarnessDeps``.
"""

import copy
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

from graphiti_core.embedder import EmbedderClient
from qdrant_client.models import Filter, PointStruct, VectorParams

from fused_memory.arm_harness import checks, graph_copy, topology
from fused_memory.arm_harness.mem0_replica import Mem0Record, Mem0Snapshot
from fused_memory.arm_harness.metrics_record import IndexConfiguration
from fused_memory.arm_harness.probe_set import FrozenReference, KnownItem
from fused_memory.arm_harness.topology import read_topology, topology_hash
from fused_memory.backends.falkor_indices import IndexProvisionResult, IndexSpec

REFERENCE = 'evalmem_lme_ref_incumbent_a'
SOURCE_COLLECTION = 'fused_dark_factory'
PROJECT = 'dark_factory'
DIM = 4
SECONDS_PER_CALL = 0.25
INCUMBENT_VECTOR = (0.5,) * 1536

EMBEDDING_ONLY = IndexConfiguration.EMBEDDING_ONLY
WITH_INDICES = IndexConfiguration.WITH_INDICES

KNOWN_ITEM_QUERIES = {'ep-a': 'alpha query', 'ep-b': 'bravo query', 'ep-c': 'charlie query'}
SEARCH_RANKS: Mapping[IndexConfiguration, Mapping[str, int | None]] = {
    EMBEDDING_ONLY: {'ep-a': 1, 'ep-b': 7, 'ep-c': None},
    WITH_INDICES: {'ep-a': 1, 'ep-b': 2, 'ep-c': 4},
}

MEM0_IDS = (
    '0b6f8a52-3a50-4c11-9d39-0f5d1c1e7a01',
    '4c1d2e3f-5a6b-4c7d-8e9f-0a1b2c3d4e02',
    '9e8d7c6b-5a49-4382-9170-6f5e4d3c2b03',
)

REFERENCE_CATALOG: frozenset[IndexSpec] = frozenset({
    ('Entity', 'NODE', 'uuid', 'RANGE'),
    ('Entity', 'NODE', 'name', 'RANGE'),
    ('Entity', 'NODE', 'name', 'FULLTEXT'),
    ('Entity', 'NODE', 'summary', 'FULLTEXT'),
    ('Episodic', 'NODE', 'content', 'FULLTEXT'),
    ('RELATES_TO', 'RELATIONSHIP', 'uuid', 'RANGE'),
    ('RELATES_TO', 'RELATIONSHIP', 'fact', 'FULLTEXT'),
})
CATALOG_HEADER = ((1, 'label'), (1, 'properties'), (1, 'types'), (1, 'entitytype'))


def _result(*rows: Sequence[Any]) -> SimpleNamespace:
    return SimpleNamespace(result_set=[list(row) for row in rows])


# --- the FalkorDB double: one in-memory graph per name, every call logged by phase ------


class FakeGraph:
    """An in-memory FalkorDB graph answering the Cypher an embedding run sends.

    ``log`` is shared with every other double, so the run's order across stores is one
    list. ``answers_fulltext_unindexed`` makes the fulltext probe find its seed with no
    index, as a stale index would.
    """

    def __init__(
        self,
        name: str,
        *,
        log: list[str],
        nodes: Mapping[str, dict[str, Any]],
        edges: Mapping[str, dict[str, Any]],
        catalog: Iterable[IndexSpec] = (),
        answers_fulltext_unindexed: bool = False,
        lossy_copy: frozenset[str] = frozenset(),
    ) -> None:
        self.name = name
        self.log = log
        self.nodes = {uuid: dict(props) for uuid, props in nodes.items()}
        self.edges = {uuid: dict(props) for uuid, props in edges.items()}
        self.catalog: set[IndexSpec] = set(catalog)
        self.answers_fulltext_unindexed = answers_fulltext_unindexed
        self.lossy_copy = lossy_copy
        self.client: FakeGraphClient | None = None

    # graph_copy -------------------------------------------------------------------------

    def _entities(self) -> dict[str, dict[str, Any]]:
        return {uuid: p for uuid, p in self.nodes.items() if 'Entity' in p['labels']}

    def _facts(self) -> dict[str, dict[str, Any]]:
        return {uuid: p for uuid, p in self.edges.items() if p['type'] == 'RELATES_TO'}

    def _set_all(self, records: Mapping[str, dict[str, Any]], key: str, value: Any):
        for props in records.values():
            props[key] = value
        return _result([len(records)])

    def _write(self, records: Mapping[str, dict[str, Any]], key: str, rows: list[dict[str, Any]]):
        for row in rows:
            records[row['uuid']][key] = tuple(row['v'])
        return _result([len(rows)])

    def _non_null(self, records: Mapping[str, dict[str, Any]], key: str):
        return _result([sum(1 for p in records.values() if p.get(key) is not None)])

    # topology ---------------------------------------------------------------------------

    def _node_rows(self):
        return _result(*(
            [{'uuid': uuid, 'labels': list(p['labels']), 'name': p.get('name')}]
            for uuid, p in self.nodes.items()
        ))

    def _edge_rows(self):
        return _result(*(
            [{'uuid': uuid, 'rel_type': p['type'], 'source_uuid': p['source'],
              'target_uuid': p['target'], 'name': p.get('name'), 'fact': p.get('fact')}]
            for uuid, p in self.edges.items()
        ))

    # the fulltext index probe -----------------------------------------------------------

    def _seed(self, params: Mapping[str, Any]):
        self.nodes[params['uuid']] = {'labels': ['Entity'], 'name': params['name']}
        return _result()

    def _fulltext(self, params: Mapping[str, Any]):
        indexed = any(spec[0] == 'Entity' and spec[3] == 'FULLTEXT' for spec in self.catalog)
        if not (indexed or self.answers_fulltext_unindexed):
            return _result()
        return _result(*(
            [uuid] for uuid, p in self._entities().items() if p.get('name') == params['token']
        ))

    def _cleanup(self, params: Mapping[str, Any]):
        del self.nodes[params['uuid']]
        return _result()

    def _handlers(self) -> Mapping[str, tuple[str, Callable[[dict[str, Any]], Any]]]:
        return {
            graph_copy.ADOPT_NODE_GROUP_CYPHER: ('adopt', lambda p: self._set_all(
                self.nodes, 'group_id', p['group_id'])),
            graph_copy.ADOPT_EDGE_GROUP_CYPHER: ('adopt', lambda p: self._set_all(
                self.edges, 'group_id', p['group_id'])),
            graph_copy.READ_ENTITY_NAMES_CYPHER: ('reembed', lambda p: _result(
                *([uuid, props['name']] for uuid, props in self._entities().items()))),
            graph_copy.READ_EDGE_FACTS_CYPHER: ('reembed', lambda p: _result(
                *([uuid, props['fact']] for uuid, props in self._facts().items()))),
            graph_copy.NULL_NAME_EMBEDDINGS_CYPHER: ('reembed', lambda p: self._set_all(
                self._entities(), 'name_embedding', None)),
            graph_copy.NULL_FACT_EMBEDDINGS_CYPHER: ('reembed', lambda p: self._set_all(
                self._facts(), 'fact_embedding', None)),
            graph_copy.WRITE_NAME_EMBEDDINGS_CYPHER: ('reembed', lambda p: self._write(
                self._entities(), 'name_embedding', p['rows'])),
            graph_copy.WRITE_FACT_EMBEDDINGS_CYPHER: ('reembed', lambda p: self._write(
                self._facts(), 'fact_embedding', p['rows'])),
            graph_copy.COUNT_NAME_EMBEDDINGS_CYPHER: ('reembed', lambda p: self._non_null(
                self._entities(), 'name_embedding')),
            graph_copy.COUNT_FACT_EMBEDDINGS_CYPHER: ('reembed', lambda p: self._non_null(
                self._facts(), 'fact_embedding')),
            topology.NODE_CYPHER: (f'topology:{self.name}', lambda p: self._node_rows()),
            topology.EDGE_CYPHER: (f'topology:{self.name}', lambda p: self._edge_rows()),
            checks.SEED_CYPHER: ('index-probe', self._seed),
            checks.PROBE_CYPHER: ('index-probe', self._fulltext),
            checks.CLEANUP_CYPHER: ('index-probe', self._cleanup),
        }

    async def query(self, q: str, params: dict[str, Any] | None = None) -> Any:
        if self.name == REFERENCE:
            raise AssertionError(f'a write to the frozen reference: {q}')
        phase, handler = self._handlers()[q]
        self.log.append(phase)
        return handler(params or {})

    async def ro_query(self, q: str, params: dict[str, Any] | None = None) -> Any:
        phase, handler = self._handlers()[q]
        self.log.append(phase)
        return handler(params or {})

    # the index catalog ------------------------------------------------------------------

    async def list_indices(self) -> SimpleNamespace:
        self.log.append('drop-indices')
        grouped: dict[tuple[str, str], dict[str, list[str]]] = {}
        for label, entity_type, field, index_type in sorted(self.catalog):
            grouped.setdefault((label, entity_type), {}).setdefault(field, []).append(index_type)
        rows = [
            [label, list(types), types, entity_type]
            for (label, entity_type), types in grouped.items()
        ]
        return SimpleNamespace(header=[list(column) for column in CATALOG_HEADER], result_set=rows)

    async def _drop(self, label: str, entity_type: str, field: str, index_type: str) -> None:
        self.log.append('drop-indices')
        self.catalog.remove((label, entity_type, field, index_type))

    async def drop_node_range_index(self, label: str, attribute: str) -> None:
        await self._drop(label, 'NODE', attribute, 'RANGE')

    async def drop_node_fulltext_index(self, label: str, attribute: str) -> None:
        await self._drop(label, 'NODE', attribute, 'FULLTEXT')

    async def drop_edge_range_index(self, label: str, attribute: str) -> None:
        await self._drop(label, 'RELATIONSHIP', attribute, 'RANGE')

    async def drop_edge_fulltext_index(self, label: str, attribute: str) -> None:
        await self._drop(label, 'RELATIONSHIP', attribute, 'FULLTEXT')

    async def copy(self, clone: str) -> None:
        assert self.client is not None
        self.log.append('copy')
        kept = {uuid: p for uuid, p in self.edges.items() if uuid not in self.lossy_copy}
        self.client.add(FakeGraph(
            clone, log=self.log, nodes=copy.deepcopy(self.nodes), edges=copy.deepcopy(kept),
            catalog=self.catalog, answers_fulltext_unindexed=self.answers_fulltext_unindexed,
        ))


class FakeGraphClient:
    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.graphs: dict[str, FakeGraph] = {}

    def add(self, graph: FakeGraph) -> None:
        graph.client = self
        self.graphs[graph.name] = graph

    async def list_graphs(self) -> list[str]:
        self.log.append('presence')
        return list(self.graphs)

    def select_graph(self, name: str) -> FakeGraph:
        return self.graphs[name]


def reference_graph(log: list[str], **kwargs: Any) -> FakeGraph:
    nodes: dict[str, dict[str, Any]] = {
        f'n{i}': {'labels': ['Entity'], 'name': f'entity\n{i}', 'group_id': REFERENCE,
                  'name_embedding': INCUMBENT_VECTOR}
        for i in range(4)
    }
    nodes |= {
        uuid: {'labels': ['Episodic'], 'name': uuid, 'content': query, 'group_id': REFERENCE}
        for uuid, query in KNOWN_ITEM_QUERIES.items()
    }
    edges: dict[str, dict[str, Any]] = {
        f'r{i}': {'type': 'RELATES_TO', 'source': f'n{i}', 'target': f'n{i + 1}',
                  'name': 'RELATES', 'fact': f'fact\n{i}', 'group_id': REFERENCE,
                  'fact_embedding': INCUMBENT_VECTOR, 'episodes': [episode]}
        for i, episode in enumerate(KNOWN_ITEM_QUERIES)
    }
    edges['m0'] = {'type': 'MENTIONS', 'source': 'ep-a', 'target': 'n0', 'group_id': REFERENCE}
    return FakeGraph(REFERENCE, log=log, nodes=nodes, edges=edges, catalog=REFERENCE_CATALOG,
                     **kwargs)


# --- the search backend, the Qdrant double and the arm's endpoint ----------------------


class FakeSearchBackend:
    """GraphitiBackend's search and ensure_indices over the fake graphs.

    A search embeds its query through the arm's real QueryEmbedder, as the production
    path does, then answers by the scratch graph's index state: with any index it ranks
    by SEARCH_RANKS[with-indices], with none by SEARCH_RANKS[embedding-only].
    """

    def __init__(
        self,
        client: FakeGraphClient,
        query_embedder: EmbedderClient,
        *,
        builds_indices: bool = True,
        strays_on_build: bool = False,
    ) -> None:
        self.client = client
        self.query_embedder = query_embedder
        self.builds_indices = builds_indices
        self.strays_on_build = strays_on_build
        self.searches: list[dict[str, Any]] = []

    async def search(
        self, query: str, group_ids: list[str] | None = None, num_results: int = 10
    ) -> list[Any]:
        assert group_ids is not None
        graph = self.client.graphs[group_ids[0]]
        graph.log.append('search')
        self.searches.append({'query': query, 'group_ids': group_ids, 'num_results': num_results})
        await self.query_embedder.create(input_data=[query])
        configuration = WITH_INDICES if graph.catalog else EMBEDDING_ONLY
        target = next(uuid for uuid, q in KNOWN_ITEM_QUERIES.items() if q == query)
        rank = SEARCH_RANKS[configuration][target]
        results = [SimpleNamespace(episodes=['unrelated']) for _ in range(num_results)]
        if rank is not None:
            results[rank - 1] = SimpleNamespace(episodes=[target])
        return results

    async def ensure_indices(self, *, group_id: str) -> IndexProvisionResult:
        graph = self.client.graphs[group_id]
        graph.log.append('ensure-indices')
        if self.builds_indices:
            graph.catalog = set(REFERENCE_CATALOG)
        if self.strays_on_build:
            graph.nodes['stray'] = {'labels': ['Entity'], 'name': 'stray'}
        created = tuple(sorted(REFERENCE_CATALOG))
        return IndexProvisionResult(
            created=created, already_present=0, failed=(), expected_total=len(created),
            statements=(),
        )


class FakeQdrant:
    """The AsyncQdrantClient slice a replica build and search use; live names are refused."""

    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.points: dict[str, list[SimpleNamespace]] = {}
        self.vector_params: dict[str, VectorParams] = {}

    async def collection_exists(self, collection_name: str) -> bool:
        self.log.append('presence')
        return collection_name in self.points

    async def create_collection(
        self, collection_name: str, vectors_config: VectorParams | None = None
    ) -> bool:
        assert collection_name.startswith('evalmem_'), collection_name
        assert vectors_config is not None
        self.log.append('replica-build')
        self.points[collection_name] = []
        self.vector_params[collection_name] = vectors_config
        return True

    async def upsert(
        self, collection_name: str, points: list[PointStruct], wait: bool = True
    ) -> None:
        assert collection_name.startswith('evalmem_'), collection_name
        self.log.append('replica-build')
        self.points[collection_name].extend(
            SimpleNamespace(id=point.id, payload=point.payload, vector=point.vector)
            for point in points
        )

    async def query_points(
        self,
        collection_name: str,
        query: list[float] | None = None,
        query_filter: Filter | None = None,
        limit: int = 10,
        with_payload: bool = True,
    ) -> SimpleNamespace:
        assert query is not None
        self.log.append('replica-search')
        ranked = sorted(
            self.points[collection_name],
            key=lambda point: -sum(a * b for a, b in zip(point.vector, query, strict=True)),
        )
        return SimpleNamespace(points=ranked[:limit])


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class FakeEndpoint(EmbedderClient):
    """The arm's served model: each call takes SECONDS_PER_CALL; ``poison`` texts raise."""

    def __init__(self, *, poison: frozenset[str] = frozenset()) -> None:
        self.clock = FakeClock()
        self.poison = poison

    def _vector(self, text: str) -> list[float]:
        if text in self.poison:
            raise RuntimeError(f'cannot embed {text!r}')
        return [1.0 + len(text) % 7, 2.0, 3.0, 30.0]

    async def create(self, input_data):
        (text,) = input_data
        assert isinstance(text, str)
        self.clock.now += SECONDS_PER_CALL
        return self._vector(text)

    async def create_batch(self, input_data_list):
        self.clock.now += SECONDS_PER_CALL
        return [self._vector(text) for text in input_data_list]


def quiet(graph: FakeGraph) -> FakeGraph:
    """A read of ``graph`` that leaves no trace in the shared log (test setup only)."""
    return FakeGraph(graph.name, log=[], nodes=graph.nodes, edges=graph.edges)


async def frozen_reference(graph: FakeGraph) -> FrozenReference:
    """The pin a probe set holds for ``graph``, read without a trace in the log."""
    reference = await read_topology(quiet(graph), graph.name)
    return FrozenReference(
        graph=graph.name,
        node_count=len(reference.nodes),
        edge_count=len(reference.edges),
        topology_hash=topology_hash(*reference),
    )


KNOWN_ITEMS = tuple(
    KnownItem(episode_uuid=uuid, query=query) for uuid, query in sorted(KNOWN_ITEM_QUERIES.items())
)


def mem0_snapshot() -> Mem0Snapshot:
    return Mem0Snapshot(
        source=SOURCE_COLLECTION,
        excluded_empty=0,
        records=tuple(
            Mem0Record(id=point_id, data=f'memory\n{index}',
                       payload={'data': f'memory\n{index}', 'user_id': PROJECT})
            for index, point_id in enumerate(MEM0_IDS)
        ),
    )
