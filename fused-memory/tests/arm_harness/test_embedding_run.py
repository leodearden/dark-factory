"""One embedding arm run, end to end (arm_harness/embedding_run.py, embedding_run_manifest.py)."""

import copy
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from graphiti_core.embedder import EmbedderClient
from qdrant_client.models import Filter, PointStruct, VectorParams

from arm_harness._fakes import embedding_spec, llm_spec, make_prereg_repo
from fused_memory.arm_harness import checks, graph_copy, topology
from fused_memory.arm_harness.arm_embedder import ArmEmbedder, EmbedSettings, QueryEmbedder
from fused_memory.arm_harness.embedding_graph_phase import EmbeddingRunCheckFailed
from fused_memory.arm_harness.embedding_run import EmbeddingRunRefused, run_embedding_arm
from fused_memory.arm_harness.embedding_run_manifest import (
    EmbeddingRunManifest,
    load_embedding_run_manifest,
)
from fused_memory.arm_harness.instrument_checks import InstrumentCheckId
from fused_memory.arm_harness.mem0_replica import (
    Mem0Record,
    Mem0Snapshot,
    ReplicaHit,
    snapshot_sha,
    write_snapshot,
)
from fused_memory.arm_harness.metrics_record import (
    EmbeddingMetricId,
    IndexConfiguration,
    MetricsRecord,
    load_metrics_records,
)
from fused_memory.arm_harness.probe_set import (
    FrozenReference,
    KnownItem,
    Mem0KnownItem,
    Mem0SnapshotPin,
    ProbeSet,
    TranscriptPin,
    probe_set_sha,
    serialize_probe_set,
)
from fused_memory.arm_harness.run import PreRunCheckError
from fused_memory.arm_harness.topology import read_topology, topology_hash
from fused_memory.backends.falkor_indices import IndexProvisionResult, IndexSpec
from fused_memory.config.schema import FusedMemoryConfig

REFERENCE = 'evalmem_lme_ref_incumbent_a'
SCRATCH = 'evalmem_lme_emb_granite'
SOURCE_COLLECTION = 'fused_dark_factory'
PROJECT = 'dark_factory'
DIM = 4
SEARCH_K = 10
EMBED_SETTINGS = EmbedSettings(batch_size=2, concurrency=1)
SECONDS_PER_CALL = 0.25
INCUMBENT_VECTOR = (0.5,) * 1536

EMBEDDING_ONLY = IndexConfiguration.EMBEDDING_ONLY
WITH_INDICES = IndexConfiguration.WITH_INDICES

KNOWN_ITEM_QUERIES = {'ep-a': 'alpha query', 'ep-b': 'bravo query', 'ep-c': 'charlie query'}
SEARCH_RANKS: Mapping[IndexConfiguration, Mapping[str, int | None]] = {
    EMBEDDING_ONLY: {'ep-a': 1, 'ep-b': 7, 'ep-c': None},
    WITH_INDICES: {'ep-a': 1, 'ep-b': 2, 'ep-c': 4},
}
MEM0_RANKS: Mapping[str, int | None] = {'topic-one': 1, 'topic-two': 6}
TRANSCRIPT_QUERIES = ('how is a run committed', 'which graph is frozen', 'why cosine')

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
        self.log.append('copy')
        return list(self.graphs)

    def select_graph(self, name: str) -> FakeGraph:
        return self.graphs[name]


def _reference_graph(log: list[str], **kwargs: Any) -> FakeGraph:
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
        self.log.append('replica-build')
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


class RecordingRanker:
    """Stands in for E1's canonical_hit(...).rank: a fixed rank per topic, every call kept."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[ReplicaHit, ...]]] = []

    def __call__(self, topic: str, hits: Sequence[ReplicaHit]) -> int | None:
        self.calls.append((topic, tuple(hits)))
        return MEM0_RANKS[topic]


# --- the world one run sees -------------------------------------------------------------


class World:
    """Every input and double one ``run_embedding_arm`` call takes, built consistently."""

    def __init__(
        self,
        tmp_path: Path,
        base_config: FusedMemoryConfig,
        *,
        poison: frozenset[str] = frozenset(),
        reference_kwargs: Mapping[str, Any] | None = None,
        backend_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.repo = make_prereg_repo(tmp_path / 'repo')
        self.base_config = base_config
        self.log: list[str] = []
        self.client = FakeGraphClient(self.log)
        self.reference = _reference_graph(self.log, **(reference_kwargs or {}))
        self.client.add(self.reference)
        self.qdrant = FakeQdrant(self.log)
        self.snapshot_path = tmp_path / 'snapshot.jsonl'
        write_snapshot(self.snapshot_path, _snapshot())
        self.endpoint = FakeEndpoint(poison=poison)
        self.ranker = RecordingRanker()
        self.run_dir = tmp_path / 'runs' / 'granite'
        self.backend_kwargs = dict(backend_kwargs or {})

    async def probe_set(self) -> ProbeSet:
        reference = await read_topology(_quiet(self.reference), REFERENCE)
        return ProbeSet(
            corpus_sha='d' * 64,
            reference=FrozenReference(
                graph=REFERENCE,
                node_count=len(reference.nodes),
                edge_count=len(reference.edges),
                topology_hash=topology_hash(*reference),
            ),
            query_words=2,
            known_items=tuple(
                KnownItem(episode_uuid=uuid, query=query)
                for uuid, query in sorted(KNOWN_ITEM_QUERIES.items())
            ),
            uncited_episodes=0,
            transcript=TranscriptPin(path='t.jsonl', sha256='e' * 64, queries=TRANSCRIPT_QUERIES),
            mem0_snapshot=Mem0SnapshotPin(
                source=SOURCE_COLLECTION,
                sha256=snapshot_sha(self.snapshot_path),
                point_count=len(MEM0_IDS),
                excluded_empty=0,
            ),
            mem0_known_items=(
                Mem0KnownItem(topic='topic-one', phrasing='where do runs live', held_out=False,
                              canonical_content_hash='1' * 16,
                              canonical_last_known_id=MEM0_IDS[0]),
                Mem0KnownItem(topic='topic-two', phrasing='what is a control pair',
                              held_out=True, canonical_content_hash='2' * 16,
                              canonical_last_known_id=None),
            ),
        )

    def spec(self, probe_set: ProbeSet, **overrides: Any):
        data = {
            'arm_id': 'granite-embedding-english-r2',
            'embedding_dim': DIM,
            'code_sha': self.repo.with_prereg,
            'preregistration_sha': self.repo.with_prereg,
            'corpus_sha': probe_set_sha(serialize_probe_set(probe_set).encode()),
            'scratch_group_id': SCRATCH,
        }
        return embedding_spec(**(data | overrides))

    async def run(self, probe_set: ProbeSet | None = None, **spec_overrides: Any):
        probe_set = probe_set or await self.probe_set()
        spec = self.spec(probe_set, **spec_overrides)
        return await self.run_spec(spec, probe_set)

    async def run_spec(self, spec: Any, probe_set: ProbeSet) -> EmbeddingRunManifest:
        embed_spec = spec if spec.axis == 'embedding' else self.spec(probe_set)
        self.embedder = ArmEmbedder(
            self.endpoint, embed_spec, EMBED_SETTINGS, clock=self.endpoint.clock
        )
        self.backend = FakeSearchBackend(
            self.client, QueryEmbedder(self.embedder), **self.backend_kwargs
        )
        return await run_embedding_arm(
            spec,
            probe_set,
            graph_client=self.client,
            backend=self.backend,
            qdrant=self.qdrant,
            embedder=self.embedder,
            mem0_snapshot=self.snapshot_path,
            mem0_project_id=PROJECT,
            rank_mem0=self.ranker,
            base_config=self.base_config,
            run_dir=self.run_dir,
            repo_root=self.repo.root,
            sleep=_no_sleep,
        )

    def phases(self) -> list[str]:
        """The run's store calls by phase, consecutive repeats collapsed."""
        collapsed: list[str] = []
        for phase in self.log:
            if not collapsed or collapsed[-1] != phase:
                collapsed.append(phase)
        return collapsed


async def _no_sleep(seconds: float) -> None:
    return None


def _quiet(graph: FakeGraph) -> FakeGraph:
    """A read of ``graph`` that leaves no trace in the shared log (test setup only)."""
    silent = FakeGraph(graph.name, log=[], nodes=graph.nodes, edges=graph.edges)
    return silent


def _snapshot() -> Mem0Snapshot:
    return Mem0Snapshot(
        source=SOURCE_COLLECTION,
        excluded_empty=0,
        records=tuple(
            Mem0Record(id=point_id, data=f'memory\n{index}',
                       payload={'data': f'memory\n{index}', 'user_id': PROJECT})
            for index, point_id in enumerate(MEM0_IDS)
        ),
    )


def _by_metric(records: Sequence[MetricsRecord]) -> dict[tuple[str, str | None], MetricsRecord]:
    return {
        (record.metric.metric_id,
         record.index_configuration.value if record.index_configuration else None): record
        for record in records
    }


@pytest.fixture
def make_world(tmp_path, mock_config) -> Callable[..., World]:
    return lambda **kwargs: World(tmp_path, mock_config, **kwargs)


EXPECTED_PHASES = [
    f'topology:{REFERENCE}',
    'copy',
    'adopt',
    'reembed',
    f'topology:{SCRATCH}',
    'drop-indices',
    'index-probe',
    'search',
    'ensure-indices',
    'index-probe',
    'search',
    f'topology:{SCRATCH}',
    'replica-build',
    'replica-search',
]


# --- the happy path -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_run_moves_through_every_phase_in_order(make_world):
    world = make_world()

    await world.run()

    assert world.phases() == EXPECTED_PHASES


@pytest.mark.asyncio
async def test_each_known_item_search_is_the_production_call_on_the_scratch_graph(make_world):
    world = make_world()

    await world.run()

    assert [call['query'] for call in world.backend.searches] == (
        [KNOWN_ITEM_QUERIES[uuid] for uuid in sorted(KNOWN_ITEM_QUERIES)] * 2
    )
    assert {tuple(call['group_ids']) for call in world.backend.searches} == {(SCRATCH,)}
    assert {call['num_results'] for call in world.backend.searches} == {SEARCH_K}


@pytest.mark.asyncio
async def test_the_scratch_copy_is_reembedded_and_the_reference_is_untouched(make_world):
    world = make_world()
    reference_before = copy.deepcopy((world.reference.nodes, world.reference.edges))

    await world.run()

    scratch = world.client.graphs[SCRATCH]
    assert (world.reference.nodes, world.reference.edges) == reference_before
    assert {p['group_id'] for p in scratch.nodes.values()} == {SCRATCH}
    vectors = [p['name_embedding'] for p in scratch._entities().values()]
    vectors += [p['fact_embedding'] for p in scratch._facts().values()]
    assert all(len(vector) == DIM for vector in vectors)
    assert all(math.isclose(math.hypot(*vector), 1.0) for vector in vectors)


@pytest.mark.asyncio
async def test_the_graph_ends_in_with_indices_and_the_replica_is_cosine_at_native_dim(make_world):
    world = make_world()

    await world.run()

    assert world.client.graphs[SCRATCH].catalog == set(REFERENCE_CATALOG)
    params = world.qdrant.vector_params[SCRATCH]
    assert (params.size, params.distance.value) == (DIM, 'Cosine')
    assert sorted(str(point.id) for point in world.qdrant.points[SCRATCH]) == sorted(MEM0_IDS)


@pytest.mark.asyncio
async def test_known_item_records_are_written_per_index_configuration(make_world):
    world = make_world()

    manifest = await world.run()

    records = _by_metric(load_metrics_records(world.run_dir))
    expected = {
        EMBEDDING_ONLY: (1 / 3, 2 / 3, (1 + 1 / 7) / 3),
        WITH_INDICES: (3 / 3, 3 / 3, (1 + 1 / 2 + 1 / 4) / 3),
    }
    for configuration, (at_5, at_10, mrr) in expected.items():
        key = configuration.value
        assert records[(EmbeddingMetricId.KNOWN_ITEM_RECALL_AT_5, key)].metric.value == at_5
        assert records[(EmbeddingMetricId.KNOWN_ITEM_RECALL_AT_10, key)].metric.value == at_10
        assert records[(EmbeddingMetricId.MRR, key)].metric.value == pytest.approx(mrr)
        assert records[(EmbeddingMetricId.MRR, key)].metric.n == len(KNOWN_ITEM_QUERIES)
    assert manifest.query_failures.known_item == {EMBEDDING_ONLY: 0, WITH_INDICES: 0}


@pytest.mark.asyncio
async def test_mem0_records_rank_the_replica_hits_through_the_injected_ranker(make_world):
    world = make_world()

    await world.run()

    records = _by_metric(load_metrics_records(world.run_dir))
    assert records[(EmbeddingMetricId.MEM0_KNOWN_ITEM_RECALL_AT_5, None)].metric.value == 1 / 2
    assert records[(EmbeddingMetricId.MEM0_KNOWN_ITEM_RECALL_AT_10, None)].metric.value == 1.0
    assert records[(EmbeddingMetricId.MEM0_MRR, None)].metric.value == pytest.approx(
        (1 + 1 / 6) / 2
    )
    assert [topic for topic, _ in world.ranker.calls] == ['topic-one', 'topic-two']
    for _, hits in world.ranker.calls:
        assert 0 < len(hits) <= SEARCH_K
        assert all(isinstance(hit, ReplicaHit) and hit.id in MEM0_IDS for hit in hits)


@pytest.mark.asyncio
async def test_reembed_throughput_counts_both_stores_over_their_embed_seconds(make_world):
    world = make_world()

    manifest = await world.run()

    record = _by_metric(load_metrics_records(world.run_dir))[
        (EmbeddingMetricId.REEMBED_THROUGHPUT, None)
    ]
    graph, replica = manifest.graph_reembed, manifest.replica_reembed
    vectors = graph.written + replica.written
    assert vectors == 4 + 3 + len(MEM0_IDS)
    assert record.metric.kind == 'scalar'
    assert record.metric.n == vectors
    assert record.metric.value == pytest.approx(
        vectors / (graph.embed_seconds + replica.embed_seconds)
    )


@pytest.mark.asyncio
async def test_query_embed_latency_is_timed_over_the_transcript_queries(make_world):
    world = make_world()

    manifest = await world.run()

    record = _by_metric(load_metrics_records(world.run_dir))[
        (EmbeddingMetricId.QUERY_EMBED_LATENCY_P95, None)
    ]
    assert record.metric.value == pytest.approx(SECONDS_PER_CALL * 1000)
    assert record.metric.n == len(TRANSCRIPT_QUERIES)
    assert manifest.query_failures.query_latency == 0


@pytest.mark.asyncio
async def test_every_record_carries_the_spec_identity_and_the_finish_time(make_world):
    world = make_world()

    manifest = await world.run()

    records = load_metrics_records(world.run_dir)
    assert len(records) == 11
    for record in records:
        assert (record.arm_id, record.axis, record.arm_role) == (
            manifest.spec.arm_id, 'embedding', 'candidate'
        )
        assert (record.code_sha, record.corpus_sha, record.preregistration_sha) == (
            manifest.spec.code_sha, manifest.spec.corpus_sha, manifest.spec.preregistration_sha
        )
        assert record.measured_at == manifest.finished_at
        assert record.incomplete is False


@pytest.mark.asyncio
async def test_run_json_is_written_last_and_round_trips(make_world):
    world = make_world()

    manifest = await world.run()

    run_json = world.run_dir / 'run.json'
    assert load_embedding_run_manifest(run_json) == manifest
    metrics = list((world.run_dir / 'metrics').glob('*.json'))
    assert run_json.stat().st_mtime_ns >= max(path.stat().st_mtime_ns for path in metrics)


@pytest.mark.asyncio
async def test_the_manifest_records_settings_embedder_reembeds_and_checks(make_world):
    world = make_world()

    manifest = await world.run()

    settings = manifest.settings
    assert (settings.embed_batch_size, settings.embed_concurrency) == (2, 1)
    assert (settings.query_concurrency, settings.search_k) == (3, SEARCH_K)
    assert settings.transcript_queries == len(TRANSCRIPT_QUERIES)
    assert settings.mem0_project_id == PROJECT
    assert settings.search_timeout_s == world.base_config.queue.search_timeout_seconds
    assert (manifest.effective_embedder.model, manifest.effective_embedder.dimensions) == (
        manifest.spec.model_id, DIM
    )
    assert (manifest.graph_reembed.entity_count, manifest.graph_reembed.edge_count) == (4, 3)
    assert manifest.graph_reembed.raw_norms is not None
    assert manifest.replica_reembed.written == len(MEM0_IDS)
    assert [check.check_id for check in manifest.check_results] == [
        InstrumentCheckId.CODE_SHA_MATCHES_CHECKOUT,
        InstrumentCheckId.PREREGISTRATION_SHA,
        InstrumentCheckId.FROZEN_REFERENCE_UNCHANGED,
        InstrumentCheckId.REEMBED_INTEGRITY,
        InstrumentCheckId.INDEX_CONFIGURATION,
        InstrumentCheckId.INDEX_CONFIGURATION,
        InstrumentCheckId.REEMBED_INTEGRITY,
    ]
    assert all(check.passed for check in manifest.check_results)
    assert manifest.started_at <= manifest.finished_at


# --- failures that are measurements, not errors -----------------------------------------


@pytest.mark.asyncio
async def test_a_known_item_query_the_arm_cannot_embed_reads_as_a_miss_and_is_counted(make_world):
    world = make_world(poison=frozenset({KNOWN_ITEM_QUERIES['ep-c']}))

    manifest = await world.run()

    records = _by_metric(load_metrics_records(world.run_dir))
    at_10 = records[(EmbeddingMetricId.KNOWN_ITEM_RECALL_AT_10, WITH_INDICES.value)]
    assert at_10.metric.value == 2 / 3
    assert at_10.metric.n == len(KNOWN_ITEM_QUERIES)
    assert manifest.query_failures.known_item == {EMBEDDING_ONLY: 1, WITH_INDICES: 1}


@pytest.mark.asyncio
async def test_a_mem0_query_the_arm_cannot_embed_reads_as_a_miss_and_is_counted(make_world):
    world = make_world(poison=frozenset({'what is a control pair'}))

    manifest = await world.run()

    records = _by_metric(load_metrics_records(world.run_dir))
    assert records[(EmbeddingMetricId.MEM0_KNOWN_ITEM_RECALL_AT_10, None)].metric.value == 1 / 2
    assert [topic for topic, _ in world.ranker.calls] == ['topic-one']
    assert manifest.query_failures.mem0_known_item == 1


# --- refusals: nothing touched, nothing written -------------------------------------------


@pytest.mark.asyncio
async def test_a_failed_pre_run_check_refuses_before_any_store_call(make_world):
    world = make_world()

    with pytest.raises(PreRunCheckError):
        await world.run(code_sha=world.repo.without_prereg)

    assert world.log == []
    assert not world.run_dir.exists()


@pytest.mark.asyncio
async def test_an_llm_spec_is_refused(make_world):
    world = make_world()
    probe_set = await world.probe_set()

    with pytest.raises(EmbeddingRunRefused, match='embedding'):
        await world.run_spec(llm_spec(scratch_group_id=SCRATCH), probe_set)

    assert world.log == []


@pytest.mark.asyncio
async def test_a_spec_whose_corpus_sha_is_not_the_probe_sets_is_refused(make_world):
    world = make_world()

    with pytest.raises(EmbeddingRunRefused, match='corpus_sha'):
        await world.run(corpus_sha='f' * 64)

    assert world.log == []
    assert not world.run_dir.exists()


@pytest.mark.asyncio
async def test_a_mem0_snapshot_that_is_not_the_pinned_one_is_refused(make_world):
    world = make_world()
    probe_set = await world.probe_set()
    world.snapshot_path.write_text(world.snapshot_path.read_text() + '\n')

    with pytest.raises(EmbeddingRunRefused, match='snapshot'):
        await world.run(probe_set)

    assert world.log == []


@pytest.mark.asyncio
async def test_a_run_dir_that_is_not_empty_is_refused(make_world):
    world = make_world()
    world.run_dir.mkdir(parents=True)
    (world.run_dir / 'leftover.json').write_text('{}')

    with pytest.raises(EmbeddingRunRefused, match='leftover|not empty|already'):
        await world.run()

    assert world.log == []


@pytest.mark.asyncio
async def test_a_changed_frozen_reference_fails_its_check_and_makes_no_copy(make_world):
    world = make_world()
    probe_set = await world.probe_set()
    world.reference.nodes['n0']['name'] = 'renamed'

    with pytest.raises(EmbeddingRunCheckFailed) as raised:
        await world.run(probe_set)

    assert raised.value.check_result.check_id is InstrumentCheckId.FROZEN_REFERENCE_UNCHANGED
    assert not raised.value.check_result.passed
    assert 'copy' not in world.log
    assert not (world.run_dir / 'run.json').exists()


# --- failed checks: the run stops, no run.json ---------------------------------------------


@pytest.mark.asyncio
async def test_a_copy_that_lost_topology_fails_integrity_before_any_index_change(make_world):
    world = make_world(reference_kwargs={'lossy_copy': frozenset({'m0'})})

    with pytest.raises(EmbeddingRunCheckFailed) as raised:
        await world.run()

    assert raised.value.check_result.check_id is InstrumentCheckId.REEMBED_INTEGRITY
    assert 'm0' in raised.value.check_result.offenders
    assert 'drop-indices' not in world.log
    assert not (world.run_dir / 'run.json').exists()


@pytest.mark.asyncio
async def test_embedding_only_that_still_answers_fulltext_fails_before_its_probe(make_world):
    world = make_world(reference_kwargs={'answers_fulltext_unindexed': True})

    with pytest.raises(EmbeddingRunCheckFailed) as raised:
        await world.run()

    assert raised.value.check_result.check_id is InstrumentCheckId.INDEX_CONFIGURATION
    assert 'search' not in world.log
    assert not (world.run_dir / 'run.json').exists()


@pytest.mark.asyncio
async def test_with_indices_that_built_nothing_fails_before_its_probe(make_world):
    world = make_world(backend_kwargs={'builds_indices': False})

    with pytest.raises(EmbeddingRunCheckFailed) as raised:
        await world.run()

    assert raised.value.check_result.check_id is InstrumentCheckId.INDEX_CONFIGURATION
    after_build = world.log[world.log.index('ensure-indices'):]
    assert 'search' not in after_build
    assert not (world.run_dir / 'run.json').exists()


@pytest.mark.asyncio
async def test_integrity_is_rechecked_after_the_probes(make_world):
    world = make_world(backend_kwargs={'strays_on_build': True})

    with pytest.raises(EmbeddingRunCheckFailed) as raised:
        await world.run()

    assert raised.value.check_result.check_id is InstrumentCheckId.REEMBED_INTEGRITY
    assert 'stray' in raised.value.check_result.offenders
    assert 'replica-build' not in world.log
    assert not (world.run_dir / 'run.json').exists()
