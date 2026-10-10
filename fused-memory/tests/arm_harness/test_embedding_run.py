"""One embedding arm run, end to end (arm_harness/embedding_run.py, embedding_run_manifest.py)."""

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from arm_harness._embedding_doubles import (
    DIM,
    EMBEDDING_ONLY,
    KNOWN_ITEM_QUERIES,
    KNOWN_ITEMS,
    MEM0_IDS,
    PROJECT,
    REFERENCE,
    REFERENCE_CATALOG,
    SECONDS_PER_CALL,
    SOURCE_COLLECTION,
    WITH_INDICES,
    FakeEndpoint,
    FakeGraph,
    FakeGraphClient,
    FakeQdrant,
    FakeSearchBackend,
    frozen_reference,
    mem0_snapshot,
    reference_graph,
)
from arm_harness._fakes import embedding_spec, llm_spec, make_prereg_repo
from fused_memory.arm_harness.arm_embedder import ArmEmbedder, EmbedSettings, QueryEmbedder
from fused_memory.arm_harness.embedding_graph_phase import EmbeddingRunCheckFailed
from fused_memory.arm_harness.embedding_run import EmbeddingRunRefused, run_embedding_arm
from fused_memory.arm_harness.embedding_run_manifest import (
    EmbeddingRunManifest,
    load_embedding_run_manifest,
)
from fused_memory.arm_harness.instrument_checks import InstrumentCheckId
from fused_memory.arm_harness.mem0_replica import ReplicaHit, snapshot_sha, write_snapshot
from fused_memory.arm_harness.metrics_record import (
    EmbeddingMetricId,
    MetricsRecord,
    load_metrics_records,
)
from fused_memory.arm_harness.probe_set import (
    Mem0KnownItem,
    Mem0SnapshotPin,
    ProbeSet,
    TranscriptPin,
    probe_set_sha,
    serialize_probe_set,
)
from fused_memory.arm_harness.run import PreRunCheckError
from fused_memory.config.schema import FusedMemoryConfig

SCRATCH = 'evalmem_lme_emb_granite'
SEARCH_K = 10
EMBED_SETTINGS = EmbedSettings(batch_size=2, concurrency=1)

MEM0_RANKS: Mapping[str, int | None] = {'topic-one': 1, 'topic-two': 6}
TRANSCRIPT_QUERIES = ('how is a run committed', 'which graph is frozen', 'why cosine')


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
        self.reference = reference_graph(self.log, **(reference_kwargs or {}))
        self.client.add(self.reference)
        self.qdrant = FakeQdrant(self.log)
        self.snapshot_path = tmp_path / 'snapshot.jsonl'
        write_snapshot(self.snapshot_path, mem0_snapshot())
        self.probe_set_path = tmp_path / 'probe-set.json'
        self.endpoint = FakeEndpoint(poison=poison)
        self.ranker = RecordingRanker()
        self.run_dir = tmp_path / 'runs' / 'granite'
        self.backend_kwargs = dict(backend_kwargs or {})

    async def probe_set(self) -> ProbeSet:
        return ProbeSet(
            corpus_sha='d' * 64,
            reference=await frozen_reference(self.reference),
            query_words=2,
            known_items=KNOWN_ITEMS,
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

    def write_probe_set(self, probe_set: ProbeSet, *, trailer: str = '') -> None:
        text = serialize_probe_set(probe_set) + trailer
        self.probe_set_path.write_text(text, encoding='utf-8')

    def spec(self, **overrides: Any):
        """The arm's spec, naming the sha of the probe set file as written."""
        data = {
            'arm_id': 'granite-embedding-english-r2',
            'embedding_dim': DIM,
            'code_sha': self.repo.with_prereg,
            'preregistration_sha': self.repo.with_prereg,
            'corpus_sha': probe_set_sha(self.probe_set_path.read_bytes()),
            'scratch_group_id': SCRATCH,
        }
        return embedding_spec(**(data | overrides))

    async def run(self, probe_set: ProbeSet | None = None, **spec_overrides: Any):
        self.write_probe_set(probe_set or await self.probe_set())
        return await self.run_spec(self.spec(**spec_overrides))

    async def run_spec(self, spec: Any) -> EmbeddingRunManifest:
        embed_spec = spec if spec.axis == 'embedding' else self.spec()
        self.embedder = ArmEmbedder(
            self.endpoint, embed_spec, EMBED_SETTINGS, clock=self.endpoint.clock
        )
        self.backend = FakeSearchBackend(
            self.client, QueryEmbedder(self.embedder), **self.backend_kwargs
        )
        return await run_embedding_arm(
            spec,
            self.probe_set_path,
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
    'presence',
    f'topology:{REFERENCE}',
    'presence',
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
    'presence',
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
    world.write_probe_set(await world.probe_set())

    with pytest.raises(EmbeddingRunRefused, match='embedding'):
        await world.run_spec(llm_spec(scratch_group_id=SCRATCH))

    assert world.log == []


@pytest.mark.asyncio
async def test_a_spec_whose_corpus_sha_is_not_the_probe_sets_is_refused(make_world):
    world = make_world()

    with pytest.raises(EmbeddingRunRefused, match='corpus_sha'):
        await world.run(corpus_sha='f' * 64)

    assert world.log == []
    assert not world.run_dir.exists()


@pytest.mark.asyncio
async def test_a_probe_set_is_pinned_by_its_file_bytes_not_by_a_reserialization(make_world):
    world = make_world()
    world.write_probe_set(await world.probe_set(), trailer='\n')

    manifest = await world.run_spec(world.spec())

    assert manifest.spec.corpus_sha == probe_set_sha(world.probe_set_path.read_bytes())


@pytest.mark.asyncio
async def test_a_spec_naming_the_reserialized_sha_of_a_reformatted_probe_set_is_refused(
    make_world,
):
    world = make_world()
    probe_set = await world.probe_set()
    world.write_probe_set(probe_set, trailer='\n')
    reserialized = probe_set_sha(serialize_probe_set(probe_set).encode())

    with pytest.raises(EmbeddingRunRefused, match='corpus_sha'):
        await world.run_spec(world.spec(corpus_sha=reserialized))

    assert world.log == []


@pytest.mark.asyncio
async def test_an_unreadable_probe_set_is_refused_naming_it(make_world):
    world = make_world()
    world.probe_set_path.write_text('{"not": "a probe set"}', encoding='utf-8')

    with pytest.raises(EmbeddingRunRefused, match='probe-set.json'):
        await world.run_spec(world.spec())

    assert world.log == []


@pytest.mark.asyncio
async def test_a_stale_scratch_graph_is_refused_before_the_graph_phase(make_world):
    world = make_world()
    world.client.add(FakeGraph(SCRATCH, log=world.log, nodes={}, edges={}))

    with pytest.raises(EmbeddingRunRefused, match='teardown') as raised:
        await world.run()

    assert 'scratch graph' in str(raised.value)
    assert 'Mem0 replica' not in str(raised.value)
    assert world.phases() == ['presence']
    assert not world.run_dir.exists()


@pytest.mark.asyncio
async def test_a_stale_mem0_replica_is_refused_before_the_graph_phase(make_world):
    world = make_world()
    world.qdrant.points[SCRATCH] = []

    with pytest.raises(EmbeddingRunRefused, match='teardown') as raised:
        await world.run()

    assert 'Mem0 replica' in str(raised.value)
    assert world.phases() == ['presence']
    assert SCRATCH not in world.client.graphs
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
