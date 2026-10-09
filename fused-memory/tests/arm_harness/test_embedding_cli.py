"""The six embedding-axis subcommands of ``harness.py``: their wiring, exit codes and artifacts."""

import hashlib
import json
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import yaml
from _fm_helpers import load_script_module

from arm_harness._fakes import (
    CODE_SHA,
    PREREG_SHA,
    PreregRepo,
    embedding_control,
    embedding_records,
    embedding_run_manifest,
    embedding_slate_arm,
    embedding_spec,
    episode_outcome,
    incumbent_control_spec,
    llm_spec,
    make_prereg_repo,
    run_manifest_for,
    slate_arm,
    write_embedding_run,
    write_run,
)
from fused_memory.arm_harness.arm_embedder import EmbedSettings, QueryEmbedder
from fused_memory.arm_harness.arm_spec import EmbeddingArmSpec, load_arm_spec
from fused_memory.arm_harness.embedding_preregistration import (
    compare_embedding_arm,
    derive_embedding_preregistration_inputs,
    load_embedding_preregistration_inputs,
    serialize_embedding_preregistration_inputs,
)
from fused_memory.arm_harness.mem0_replica import (
    Mem0Record,
    Mem0Snapshot,
    ReplicaHit,
    load_snapshot,
    snapshot_sha,
    write_snapshot,
)
from fused_memory.arm_harness.probe_set import (
    FrozenReference,
    KnownItem,
    Mem0KnownItem,
    Mem0SnapshotPin,
    ProbeSet,
    TranscriptPin,
    build_probe_set,
    load_probe_set,
    probe_set_sha,
    serialize_probe_set,
)
from fused_memory.arm_harness.run import write_outcomes
from fused_memory.arm_harness.slate import embedding_candidate_spec, load_embedding_slate
from fused_memory.arm_harness.topology import (
    EDGE_CYPHER,
    NODE_CYPHER,
    TopologyEdge,
    TopologyNode,
    topology_hash,
)
from fused_memory.config.schema import EmbedderConfig, FusedMemoryConfig

LME_DIR = Path(__file__).parents[2] / 'scripts' / 'local_memory_models_eval'
STAMP = '20261009T120000Z'
REFERENCE = 'evalmem_lme_ref_incumbent_a'
SOURCE_COLLECTION = 'fused_dark_factory'
PROJECT = 'dark_factory'

EPISODES = {
    'replay-a': 'the merge worker stands off while the index lock is held by a commit',
    'replay-b': 'graphiti embeds every entity name with newlines folded into spaces',
    'replay-c': 'an uncited episode whose facts were all deduplicated away',
}
CITED = ('replay-a', 'replay-b')
NODE_ROWS = (
    *({'uuid': uuid, 'labels': ['Episodic'], 'name': uuid} for uuid in EPISODES),
    {'uuid': 'n-merge', 'labels': ['Entity'], 'name': 'merge worker'},
    {'uuid': 'n-lock', 'labels': ['Entity'], 'name': 'index lock'},
)
EDGE_ROWS = (
    {'uuid': 'e-1', 'rel_type': 'RELATES_TO', 'source_uuid': 'n-merge', 'target_uuid': 'n-lock',
     'name': 'WAITS_ON', 'fact': 'the merge worker waits on the index lock'},
)

HASHED_RECORD = Mem0Record(
    id='0b6f8a52-3a50-4c11-9d39-0f5d1c1e7a01',
    data='Never run git stash in a dark-factory checkout.',
    payload={'user_id': PROJECT, 'data': 'Never run git stash in a dark-factory checkout.'},
)
ID_RECORD = Mem0Record(
    id='4c1d2e3f-5a6b-4c7d-8e9f-0a1b2c3d4e02',
    data='Anchor ad-hoc probes on the checkout root.',
    payload={'user_id': PROJECT, 'data': 'Anchor ad-hoc probes on the checkout root.'},
)
BY_ID_PHRASINGS = (('where do probes run from', False), ('which dir does a probe start in', True))
TRANSCRIPT_QUERIES = ('why cosine', 'which graph is frozen', 'how is one run committed last')


@pytest.fixture
def harness(monkeypatch) -> ModuleType:
    """The CLI module, loaded with its own directory importable, as a direct run resolves it."""
    monkeypatch.syspath_prepend(str(LME_DIR))
    return load_script_module(LME_DIR / 'harness.py', mod_name='lme_harness')


def _result(rows: Sequence[Sequence[Any]]) -> SimpleNamespace:
    return SimpleNamespace(result_set=[list(row) for row in rows])


# --- fake live dependencies ----------------------------------------------------------


class FakeGraph:
    def __init__(self, name: str, falkor: 'FakeFalkor') -> None:
        self._name = name
        self._falkor = falkor

    async def ro_query(self, q: str, params: dict[str, Any] | None = None) -> Any:
        self._falkor.calls.append(('ro_query', self._name))
        return _result(self._falkor.rows(q))

    async def query(self, q: str, params: dict[str, Any] | None = None) -> Any:
        self._falkor.calls.append(('query', self._name))
        return _result([])

    async def copy(self, clone: str, /) -> None:
        self._falkor.calls.append(('copy', self._name))


class FakeFalkor:
    """Answers every graph's ro_query with ``rows(cypher)``; every call is logged."""

    def __init__(self, rows: Callable[[str], Sequence[Sequence[Any]]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, str]] = []

    async def list_graphs(self) -> list[str]:
        return [REFERENCE]

    def select_graph(self, graph_id: str) -> FakeGraph:
        self.calls.append(('select_graph', graph_id))
        return FakeGraph(graph_id, self)


class FakeQdrant:
    """Scrolls ``points`` in pages of two; a write of any kind is logged in ``writes``."""

    def __init__(self, points: Sequence[SimpleNamespace] = ()) -> None:
        self.points = list(points)
        self.scrolled: list[str] = []
        self.writes: list[str] = []

    async def scroll(self, collection_name: str, *, limit: int = 10, offset: Any = None,
                     with_payload: bool = True, with_vectors: bool = False) -> Any:
        self.scrolled.append(collection_name)
        start = offset or 0
        page = self.points[start : start + 2]
        return page, (start + 2 if start + 2 < len(self.points) else None)

    async def create_collection(self, collection_name: str, **kwargs: Any) -> None:
        self.writes.append(f'create_collection {collection_name}')

    async def upsert(self, collection_name: str, **kwargs: Any) -> None:
        self.writes.append(f'upsert {collection_name}')

    async def delete_collection(self, collection_name: str) -> None:
        self.writes.append(f'delete_collection {collection_name}')


@dataclass(frozen=True)
class FakeArmEmbedder:
    spec: EmbeddingArmSpec
    settings: EmbedSettings


@dataclass
class EmbedLive:
    """A ``deps`` factory over fakes; ``log`` records what each dependency was asked for."""

    base_config: FusedMemoryConfig
    falkor: FakeFalkor = field(default_factory=lambda: FakeFalkor(_reference_rows))
    qdrant: FakeQdrant = field(default_factory=FakeQdrant)
    backend: object = field(default_factory=object)
    factory_calls: int = 0
    backend_opens: list[tuple[EmbeddingArmSpec, FusedMemoryConfig, Any]] = field(
        default_factory=list
    )
    embedders: list[FakeArmEmbedder] = field(default_factory=list)

    def factory(self, harness: ModuleType) -> Callable[[], Any]:
        def build() -> Any:
            self.factory_calls += 1
            return harness.HarnessDeps(
                base_config=self.base_config,
                episode_reader=_unused,
                open_arm_backend=_unused,
                open_journal=_unused,
                open_falkordb=_yielding(self.falkor),
                open_qdrant=_yielding(self.qdrant),
                open_embedding_arm_backend=self._open_backend,
                build_arm_embedder=self._build_embedder,
            )

        return build

    @asynccontextmanager
    async def _open_backend(
        self, spec: EmbeddingArmSpec, base: FusedMemoryConfig, query_embedder: Any
    ) -> AsyncIterator[object]:
        self.backend_opens.append((spec, base, query_embedder))
        yield self.backend

    def _build_embedder(
        self, spec: EmbeddingArmSpec, base: FusedMemoryConfig, *, settings: EmbedSettings
    ) -> FakeArmEmbedder:
        embedder = FakeArmEmbedder(spec, settings)
        self.embedders.append(embedder)
        return embedder


def _unused(*args: Any) -> Any:
    raise AssertionError('the embedding subcommands never touch the LLM replay dependencies')


def _yielding(value: Any) -> Callable[[], Any]:
    @asynccontextmanager
    async def open_() -> AsyncIterator[Any]:
        yield value

    return open_


def _offline() -> Any:
    raise AssertionError('this subcommand is offline post-processing: it never builds live deps')


@pytest.fixture
def live(mock_config: FusedMemoryConfig) -> EmbedLive:
    return EmbedLive(base_config=mock_config)


def _reference_rows(cypher: str) -> list[list[Any]]:
    if cypher == NODE_CYPHER:
        return [[row] for row in NODE_ROWS]
    if cypher == EDGE_CYPHER:
        return [[row] for row in EDGE_ROWS]
    if 'Episodic' in cypher:
        return [[uuid, content] for uuid, content in EPISODES.items()]
    if 'r.episodes' in cypher:
        return [[uuid] for uuid in CITED]
    raise AssertionError(f'unexpected cypher {cypher!r}')


def _reference_hash() -> str:
    nodes = [
        TopologyNode(uuid=row['uuid'], labels=tuple(sorted(row['labels'])), name=row['name'])
        for row in NODE_ROWS
    ]
    return topology_hash(nodes, [TopologyEdge(**row) for row in EDGE_ROWS])


def _refused(harness: ModuleType, argv: list[str], deps: Any, capsys: Any, code: int) -> str:
    assert harness.main(argv, deps=deps) == code
    err = capsys.readouterr().err
    assert err.startswith('error: ')
    return err


# --- mem0-snapshot ---------------------------------------------------------------------


def _point(record: Mem0Record) -> SimpleNamespace:
    return SimpleNamespace(id=record.id, payload=dict(record.payload), vector=None)


def test_mem0_snapshot_writes_the_collection_read_only_and_prints_its_sha(
    harness, live, tmp_path, capsys
):
    empty = SimpleNamespace(id='9e8d7c6b-5a49-4382-9170-6f5e4d3c2b03', payload={'data': ' '})
    live.qdrant = FakeQdrant([_point(ID_RECORD), empty, _point(HASHED_RECORD)])
    out = tmp_path / 'mem0-source.jsonl'

    code = harness.main(
        ['mem0-snapshot', '--collection', SOURCE_COLLECTION, '--out', str(out)],
        deps=live.factory(harness),
    )

    assert code == harness.EXIT_OK
    snapshot = load_snapshot(out)
    assert snapshot.source == SOURCE_COLLECTION
    assert snapshot.records == (HASHED_RECORD, ID_RECORD)
    assert snapshot.excluded_empty == 1
    assert snapshot_sha(out) in capsys.readouterr().out
    assert set(live.qdrant.scrolled) == {SOURCE_COLLECTION}
    assert live.qdrant.writes == []


def test_mem0_snapshot_never_overwrites_an_existing_snapshot(harness, live, tmp_path, capsys):
    out = tmp_path / 'mem0-source.jsonl'
    out.write_text('committed\n')

    err = _refused(
        harness,
        ['mem0-snapshot', '--collection', SOURCE_COLLECTION, '--out', str(out)],
        live.factory(harness), capsys, harness.EXIT_REFUSED,
    )

    assert str(out) in err
    assert out.read_text() == 'committed\n'
    assert live.factory_calls == 0


# --- probe-set -------------------------------------------------------------------------


def _registry_entry(topic: str, project: str, content_hash: str, last_known_id: str | None,
                    phrasings: Sequence[tuple[str, bool]]) -> dict[str, Any]:
    return {
        'topic': topic,
        'project_id': project,
        'derived_from': 'hand',
        'canonical': {'content_hash': content_hash, 'last_known_id': last_known_id},
        'phrasings': [{'text': text, 'held_out': held_out} for text, held_out in phrasings],
    }


@dataclass(frozen=True)
class ProbeInputs:
    reference_json: Path
    control_a: Path
    transcript: Path
    snapshot: Path
    registry: Path
    out: Path

    def argv(self) -> list[str]:
        return [
            'probe-set',
            '--reference-json', str(self.reference_json),
            '--control-a-outcomes', str(self.control_a / 'outcomes.jsonl'),
            '--transcript-corpus', str(self.transcript),
            '--mem0-snapshot', str(self.snapshot),
            '--registry', str(self.registry),
            '--out', str(self.out),
        ]


def _probe_inputs(
    harness: ModuleType, tmp_path: Path, *, reference_hash: str | None = None,
    control_scratch: str = REFERENCE,
) -> ProbeInputs:
    content_key = harness.load_probe_module().content_key
    reference_json = tmp_path / 'frozen-reference.json'
    reference_json.write_text(json.dumps({
        'graph': REFERENCE, 'node_count': len(NODE_ROWS), 'edge_count': len(EDGE_ROWS),
        'topology_hash': reference_hash or _reference_hash(),
    }))
    control = incumbent_control_spec(scratch_group_id=control_scratch)
    control_a = write_run(tmp_path / 'control-a', run_manifest_for(control), [])
    write_outcomes(control_a, [
        *(episode_outcome(uuid.removeprefix('replay-')) for uuid in EPISODES),
        episode_outcome('d', ok=False),
    ])
    transcript = tmp_path / 'corpus-20261009T000000Z.jsonl'
    transcript.write_text(''.join(
        json.dumps({'schema_version': 1, 'result_status': status, 'query': query}) + '\n'
        for status, query in (('ok', TRANSCRIPT_QUERIES[0]), ('error', 'never answered'),
                              ('ok', TRANSCRIPT_QUERIES[1]), ('ok', TRANSCRIPT_QUERIES[2]))
    ))
    snapshot = tmp_path / 'mem0-source.jsonl'
    write_snapshot(snapshot, Mem0Snapshot(
        source=SOURCE_COLLECTION, excluded_empty=1, records=(HASHED_RECORD, ID_RECORD)
    ))
    registry = tmp_path / 'registry.json'
    registry.write_text(json.dumps({'schema_version': 1, 'entries': [
        _registry_entry('topic-hashed', PROJECT, content_key(HASHED_RECORD.data), 'gone-id',
                        [('is git stash safe here', False), ('may I stash my wip', True)]),
        _registry_entry('topic-by-id', PROJECT, 'deadbeefdeadbeef', ID_RECORD.id, BY_ID_PHRASINGS),
        _registry_entry('topic-absent', PROJECT, 'feedfacefeedface', 'missing-id',
                        [('an absent canonical', True)]),
        _registry_entry('topic-other-project', 'reify', content_key(HASHED_RECORD.data), None,
                        [('another project entirely', True)]),
    ]}))
    return ProbeInputs(reference_json, control_a, transcript, snapshot, registry,
                       tmp_path / 'probe-set.json')


def _expected_probe_set(harness: ModuleType, inputs: ProbeInputs) -> ProbeSet:
    content_key = harness.load_probe_module().content_key
    snapshot = load_snapshot(inputs.snapshot)
    return build_probe_set(
        corpus_sha=incumbent_control_spec().corpus_sha,
        reference=FrozenReference.model_validate_json(inputs.reference_json.read_text()),
        episodes=list(EPISODES.items()),
        cited_episode_uuids=set(CITED),
        control_ok_episode_uuids=set(EPISODES),
        transcript=TranscriptPin(
            path=inputs.transcript.name,
            sha256=hashlib.sha256(inputs.transcript.read_bytes()).hexdigest(),
            queries=TRANSCRIPT_QUERIES,
        ),
        mem0_snapshot=Mem0SnapshotPin(
            source=SOURCE_COLLECTION, sha256=snapshot_sha(inputs.snapshot),
            point_count=len(snapshot.records), excluded_empty=1,
        ),
        mem0_known_items=[
            Mem0KnownItem(topic='topic-hashed', phrasing=phrasing, held_out=held_out,
                          canonical_content_hash=content_key(HASHED_RECORD.data),
                          canonical_last_known_id='gone-id')
            for phrasing, held_out in (('is git stash safe here', False),
                                       ('may I stash my wip', True))
        ] + [
            Mem0KnownItem(topic='topic-by-id', phrasing=phrasing, held_out=held_out,
                          canonical_content_hash='deadbeefdeadbeef',
                          canonical_last_known_id=ID_RECORD.id)
            for phrasing, held_out in BY_ID_PHRASINGS
        ],
    )


def test_probe_set_reads_the_reference_read_only_and_writes_the_derived_probe(
    harness, live, tmp_path, capsys, monkeypatch
):
    inputs = _probe_inputs(harness, tmp_path)
    probe = harness.load_probe_module()
    registry_reads: list[Path] = []
    real_load = probe.load_topic_registry

    def recording_load(path: Any) -> Any:
        registry_reads.append(Path(path))
        return real_load(path)

    monkeypatch.setattr(probe, 'load_topic_registry', recording_load)

    code = harness.main(inputs.argv(), deps=live.factory(harness))

    assert code == harness.EXIT_OK
    expected = _expected_probe_set(harness, inputs)
    assert inputs.out.read_text() == serialize_probe_set(expected)
    assert expected.query_words == 4
    assert [item.episode_uuid for item in expected.known_items] == list(CITED)
    assert probe_set_sha(inputs.out.read_bytes()) in capsys.readouterr().out
    assert registry_reads == [inputs.registry]
    assert {kind for kind, _ in live.falkor.calls} == {'select_graph', 'ro_query'}
    assert {graph for _, graph in live.falkor.calls} == {REFERENCE}


def test_probe_set_whose_reference_moved_exits_3_writing_nothing(harness, live, tmp_path, capsys):
    inputs = _probe_inputs(harness, tmp_path, reference_hash='f' * 64)

    code = harness.main(inputs.argv(), deps=live.factory(harness))

    assert code == harness.EXIT_CHECK_FAILED
    assert 'FAIL frozen-reference-unchanged' in capsys.readouterr().out
    assert not inputs.out.exists()


def test_probe_set_refuses_a_control_a_that_did_not_build_the_reference(
    harness, live, tmp_path, capsys
):
    inputs = _probe_inputs(harness, tmp_path, control_scratch='evalmem_lme_ref_other')

    err = _refused(harness, inputs.argv(), live.factory(harness), capsys, harness.EXIT_REFUSED)

    assert REFERENCE in err
    assert not inputs.out.exists()


def test_probe_set_never_overwrites_an_existing_probe_set(harness, live, tmp_path, capsys):
    inputs = _probe_inputs(harness, tmp_path)
    inputs.out.write_text('committed\n')

    _refused(harness, inputs.argv(), live.factory(harness), capsys, harness.EXIT_REFUSED)

    assert inputs.out.read_text() == 'committed\n'
    assert live.factory_calls == 0


# --- embed-specs -----------------------------------------------------------------------

CANDIDATE_ARMS = (
    embedding_slate_arm(),
    embedding_slate_arm(arm_id='granite-embedding-english-r2', port=8415, dims=768,
                        served_model_name='granite-embedding-english-r2', query_prefix=None),
)
CONTROL_SCRATCH = {'incumbent-embed-a': 'evalmem_lme_emb_ctl_a',
                   'incumbent-embed-b': 'evalmem_lme_emb_ctl_b'}


def _probe_set_file(path: Path, snapshot_sha256: str = 'e' * 64) -> Path:
    probe_set = ProbeSet(
        corpus_sha='b' * 64,
        reference=FrozenReference(graph=REFERENCE, node_count=5, edge_count=1,
                                  topology_hash=_reference_hash()),
        query_words=4,
        known_items=(KnownItem(episode_uuid='replay-a', query='the merge worker stands'),),
        uncited_episodes=2,
        transcript=TranscriptPin(path='corpus.jsonl', sha256='d' * 64, queries=TRANSCRIPT_QUERIES),
        mem0_snapshot=Mem0SnapshotPin(source=SOURCE_COLLECTION, sha256=snapshot_sha256,
                                      point_count=2, excluded_empty=0),
        mem0_known_items=(
            Mem0KnownItem(topic='topic-by-id', phrasing='where do probes run from',
                          held_out=False, canonical_content_hash='deadbeefdeadbeef',
                          canonical_last_known_id=ID_RECORD.id),
        ),
    )
    path.write_text(serialize_probe_set(probe_set))
    return path


def _arms_manifest(path: Path) -> Path:
    arms = [
        slate_arm().model_dump() | {'axis': 'llm'},
        *(arm.model_dump() | {'axis': 'embedding'} for arm in CANDIDATE_ARMS),
    ]
    path.write_text(yaml.safe_dump({'arms': arms}))
    return path


def _specs_argv(manifest: Path, probe_set: Path, out_dir: Path) -> list[str]:
    return ['embed-specs', '--arms-manifest', str(manifest), '--probe-set', str(probe_set),
            '--code-sha', CODE_SHA, '--preregistration-sha', PREREG_SHA,
            '--out-dir', str(out_dir)]


def test_embed_specs_writes_each_slate_candidate_and_both_controls(harness, live, tmp_path):
    live.base_config = live.base_config.model_copy(update={
        'embedder': EmbedderConfig(model='text-embedding-3-large', dimensions=3072),
    })
    manifest = _arms_manifest(tmp_path / 'arms.yaml')
    probe_set = _probe_set_file(tmp_path / 'probe-set.json')
    out_dir = tmp_path / 'specs'

    code = harness.main(_specs_argv(manifest, probe_set, out_dir), deps=live.factory(harness))

    assert code == harness.EXIT_OK
    corpus = probe_set_sha(probe_set.read_bytes())
    slate = load_embedding_slate(manifest)
    assert [arm.arm_id for arm in slate] == [arm.arm_id for arm in CANDIDATE_ARMS]
    for arm in slate:
        assert load_arm_spec(out_dir / f'{arm.arm_id}.json') == embedding_candidate_spec(
            arm, code_sha=CODE_SHA, corpus_sha=corpus, preregistration_sha=PREREG_SHA
        )
    for arm_id, scratch in CONTROL_SCRATCH.items():
        control = load_arm_spec(out_dir / f'{arm_id}.json')
        assert control == embedding_control(
            arm_id, model_id='text-embedding-3-large', embedding_dim=3072,
            corpus_sha=corpus, scratch_group_id=scratch,
        )
    assert len(list(out_dir.iterdir())) == len(slate) + len(CONTROL_SCRATCH)


def test_embed_specs_never_overwrites_a_spec_and_then_writes_none(
    harness, live, tmp_path, capsys
):
    out_dir = tmp_path / 'specs'
    out_dir.mkdir()
    kept = out_dir / 'incumbent-embed-b.json'
    kept.write_text('committed\n')
    argv = _specs_argv(_arms_manifest(tmp_path / 'arms.yaml'),
                       _probe_set_file(tmp_path / 'probe-set.json'), out_dir)

    err = _refused(harness, argv, live.factory(harness), capsys, harness.EXIT_REFUSED)

    assert str(kept) in err
    assert [path.name for path in out_dir.iterdir()] == [kept.name]


def test_a_failed_spec_write_leaves_no_spec(harness, live, tmp_path, monkeypatch):
    out_dir = tmp_path / 'specs'
    argv = _specs_argv(_arms_manifest(tmp_path / 'arms.yaml'),
                       _probe_set_file(tmp_path / 'probe-set.json'), out_dir)
    real_write = harness.atomic_write_text

    def disk_full_on_a_control(path: Path, text: str, **kwargs: Any) -> None:
        if Path(path).name == 'incumbent-embed-b.json':
            raise OSError(f'disk full writing {path}')
        real_write(path, text, **kwargs)

    monkeypatch.setattr(harness, 'atomic_write_text', disk_full_on_a_control)
    with pytest.raises(OSError, match='disk full'):
        harness.main(argv, deps=live.factory(harness))

    assert list(out_dir.iterdir()) == []


def test_embed_specs_refuses_a_misshapen_arms_manifest_naming_it(
    harness, live, tmp_path, capsys
):
    manifest = tmp_path / 'arms.yaml'
    manifest.write_text('arms: {not: a list}\n')
    argv = _specs_argv(manifest, _probe_set_file(tmp_path / 'probe-set.json'), tmp_path / 'out')

    err = _refused(harness, argv, live.factory(harness), capsys, harness.EXIT_REFUSED)

    assert str(manifest) in err
    assert not (tmp_path / 'out').exists()


# --- embed-run -------------------------------------------------------------------------


@dataclass(frozen=True)
class RunInputs:
    repo: PreregRepo
    spec: EmbeddingArmSpec
    spec_path: Path
    probe_set: Path
    snapshot: Path
    out_root: Path

    def argv(self, spec_path: Path | None = None) -> list[str]:
        return ['embed-run', '--arm-spec', str(spec_path or self.spec_path),
                '--probe-set', str(self.probe_set), '--mem0-snapshot', str(self.snapshot),
                '--out-root', str(self.out_root), '--repo-root', str(self.repo.root)]

    @property
    def run_dir(self) -> Path:
        return self.out_root / self.spec.arm_id / STAMP


def _spec_file(path: Path, spec: Any, **raw_overrides: Any) -> Path:
    path.write_text(json.dumps(spec.model_dump(mode='json') | raw_overrides))
    return path


@pytest.fixture
def run_inputs(tmp_path: Path, monkeypatch) -> RunInputs:
    monkeypatch.setenv('MEMORY_EVAL_RUN_STAMP', STAMP)
    repo = make_prereg_repo(tmp_path / 'repo')
    snapshot = tmp_path / 'mem0-source.jsonl'
    write_snapshot(snapshot, Mem0Snapshot(source=SOURCE_COLLECTION, excluded_empty=0,
                                          records=(HASHED_RECORD, ID_RECORD)))
    probe_set = _probe_set_file(tmp_path / 'probe-set.json', snapshot_sha(snapshot))
    spec = embedding_spec(
        arm_id='granite-embedding-english-r2', model_id='granite-embedding-english-r2',
        embedding_dim=768, code_sha=repo.with_prereg, preregistration_sha=repo.with_prereg,
        corpus_sha=probe_set_sha(probe_set.read_bytes()),
        scratch_group_id='evalmem_lme_emb_granite_embedding_english_r2',
    )
    return RunInputs(repo, spec, _spec_file(tmp_path / 'spec.json', spec), probe_set,
                     snapshot, tmp_path / 'runs')


def test_embed_run_hands_every_live_dependency_to_run_embedding_arm(
    harness, live, run_inputs, monkeypatch, capsys
):
    captured: dict[str, Any] = {}

    async def recording_run(spec: Any, probe_set: Any, **kwargs: Any) -> Any:
        captured.update(spec=spec, probe_set=probe_set, **kwargs)
        return embedding_run_manifest(spec)

    monkeypatch.setattr(harness, 'run_embedding_arm', recording_run)

    code = harness.main(run_inputs.argv(), deps=live.factory(harness))

    assert code == harness.EXIT_OK
    [embedder] = live.embedders
    [(opened_spec, opened_base, query_embedder)] = live.backend_opens
    assert embedder.spec == opened_spec == captured['spec'] == run_inputs.spec
    assert isinstance(embedder.settings, EmbedSettings)
    assert isinstance(query_embedder, QueryEmbedder) and query_embedder.arm is embedder
    assert opened_base is captured['base_config'] is live.base_config
    assert captured['probe_set'] == load_probe_set(run_inputs.probe_set)
    assert captured['graph_client'] is live.falkor
    assert captured['qdrant'] is live.qdrant
    assert captured['backend'] is live.backend
    assert captured['embedder'] is embedder
    assert captured['mem0_snapshot'] == run_inputs.snapshot
    assert captured['mem0_project_id'] == PROJECT
    assert captured['run_dir'] == run_inputs.run_dir
    assert captured['repo_root'] == run_inputs.repo.root
    assert f'run: {run_inputs.run_dir}' in capsys.readouterr().out


def test_the_mem0_ranker_is_e1s_own_canonical_hit_over_the_pinned_canonical(
    harness, live, run_inputs, monkeypatch
):
    probe = harness.load_probe_module()
    entries: list[Any] = []
    real_hit = probe.canonical_hit

    def recording_hit(results: list, entry: Any, k: int, **kwargs: Any) -> Any:
        entries.append(entry)
        return real_hit(results, entry, k, **kwargs)

    captured: dict[str, Any] = {}

    async def recording_run(spec: Any, probe_set: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return embedding_run_manifest(spec)

    monkeypatch.setattr(probe, 'canonical_hit', recording_hit)
    monkeypatch.setattr(harness, 'run_embedding_arm', recording_run)
    assert harness.main(run_inputs.argv(), deps=live.factory(harness)) == harness.EXIT_OK

    hits = (ReplicaHit(id='other', content='unrelated'),
            ReplicaHit(id=ID_RECORD.id, content='reworded since the fixture was hashed'))
    assert captured['rank_mem0']('topic-by-id', hits) == 2
    [entry] = entries
    assert entry.canonical.content_hash == 'deadbeefdeadbeef'
    assert entry.canonical.last_known_id == ID_RECORD.id


def test_embed_run_refuses_an_llm_spec_before_any_dependency(
    harness, live, run_inputs, tmp_path, capsys
):
    llm = _spec_file(tmp_path / 'llm.json', llm_spec())

    err = _refused(harness, run_inputs.argv(llm), live.factory(harness), capsys,
                   harness.EXIT_REFUSED)

    assert 'llm' in err
    assert live.factory_calls == 0


def test_embed_run_whose_corpus_sha_is_not_the_probe_sets_exits_6(
    harness, live, run_inputs, tmp_path, capsys
):
    stale = _spec_file(tmp_path / 'stale.json', run_inputs.spec, corpus_sha='0' * 64)

    err = _refused(harness, run_inputs.argv(stale), live.factory(harness), capsys,
                   harness.EXIT_CORPUS_INTEGRITY)

    assert str(run_inputs.probe_set) in err
    assert live.factory_calls == 0


def test_embed_run_on_a_protected_graph_exits_5(harness, live, run_inputs, tmp_path, capsys):
    guarded = _spec_file(tmp_path / 'guarded.json', run_inputs.spec, scratch_group_id='dark_factory')

    err = _refused(harness, run_inputs.argv(guarded), live.factory(harness), capsys,
                   harness.EXIT_SCRATCH_GUARD)

    assert 'ScratchGuardError' in err
    assert live.factory_calls == 0


def test_embed_run_whose_reference_moved_exits_3_with_no_run_json(
    harness, live, run_inputs, capsys
):
    live.falkor = FakeFalkor(
        lambda cypher: [[{**row, 'name': 'moved'}] for row in NODE_ROWS] if cypher == NODE_CYPHER
        else _reference_rows(cypher)
    )

    err = _refused(harness, run_inputs.argv(), live.factory(harness), capsys,
                   harness.EXIT_CHECK_FAILED)

    assert 'frozen-reference-unchanged' in err
    assert not (run_inputs.run_dir / 'run.json').exists()
    assert {kind for kind, _ in live.falkor.calls} <= {'select_graph', 'ro_query'}


# --- embed-preregister and embed-compare -----------------------------------------------

CONTROL_A = embedding_control('incumbent-embed-a')
CONTROL_B = embedding_control('incumbent-embed-b', scratch_group_id='evalmem_lme_emb_ctl_b')
GRANITE = embedding_spec(
    arm_id='granite-embedding-english-r2', model_id='granite-embedding-english-r2',
    embedding_dim=768, scratch_group_id='evalmem_lme_emb_granite_embedding_english_r2',
)
QWEN = embedding_spec(
    arm_id='qwen3-embedding-0.6b', model_id='qwen3-embedding-0.6b',
    scratch_group_id='evalmem_lme_emb_qwen3_embedding_0_6b',
)
CONTROL_B_VALUES = {'with_indices': (0.78, 0.89, 0.68), 'embedding_only': (0.66, 0.84, 0.58),
                    'latency_p95_ms': 140.0}


def _embedding_run(root: Path, spec: EmbeddingArmSpec, **values: Any) -> Path:
    return write_embedding_run(
        root / spec.arm_id, embedding_run_manifest(spec), embedding_records(spec, **values)
    )


def _control_pair(root: Path) -> tuple[Path, Path]:
    return _embedding_run(root, CONTROL_A), _embedding_run(root, CONTROL_B, **CONTROL_B_VALUES)


def _preregister_argv(run_a: Path, run_b: Path, out: Path) -> list[str]:
    return ['embed-preregister', '--run-a', str(run_a), '--run-b', str(run_b), '--out', str(out)]


def _derived_inputs() -> Any:
    return derive_embedding_preregistration_inputs(
        embedding_run_manifest(CONTROL_A), embedding_records(CONTROL_A),
        embedding_run_manifest(CONTROL_B), embedding_records(CONTROL_B, **CONTROL_B_VALUES),
    )


def test_embed_preregister_writes_the_inputs_derived_from_the_control_pair(
    harness, tmp_path, capsys
):
    run_a, run_b = _control_pair(tmp_path)
    out = tmp_path / 'embedding-preregistration-inputs.json'

    code = harness.main(_preregister_argv(run_a, run_b, out), deps=_offline)

    assert code == harness.EXIT_OK
    inputs = _derived_inputs()
    assert out.read_text() == serialize_embedding_preregistration_inputs(inputs)
    lines = capsys.readouterr().out.splitlines()
    assert sum(line.startswith('margin ') for line in lines) == len(inputs.margins)
    assert any(str(inputs.query_latency_envelope.p95_bound_ms) in line for line in lines)
    assert lines[-1] == f'wrote: {out}'


def test_embed_preregister_never_overwrites_its_output(harness, tmp_path, capsys):
    run_a, run_b = _control_pair(tmp_path)
    out = tmp_path / 'inputs.json'
    out.write_text('committed\n')

    _refused(harness, _preregister_argv(run_a, run_b, out), _offline, capsys,
             harness.EXIT_REFUSED)

    assert out.read_text() == 'committed\n'


def test_embed_preregister_refuses_a_candidate_given_as_a_control(harness, tmp_path, capsys):
    run_a = _embedding_run(tmp_path, CONTROL_A)
    candidate = _embedding_run(tmp_path, GRANITE)
    out = tmp_path / 'inputs.json'

    err = _refused(harness, _preregister_argv(run_a, candidate, out), _offline, capsys,
                   harness.EXIT_REFUSED)

    assert 'candidate' in err
    assert not out.exists()


def test_embed_compare_prints_a_row_per_margin_then_the_envelope_and_verdict(
    harness, tmp_path, capsys
):
    inputs = _derived_inputs()
    prereg = tmp_path / 'inputs.json'
    prereg.write_text(serialize_embedding_preregistration_inputs(inputs))
    runs = {GRANITE: {}, QWEN: {'with_indices': (0.5, 0.6, 0.4)}}
    run_dirs = [_embedding_run(tmp_path, spec, **values) for spec, values in runs.items()]
    argv = ['embed-compare', '--preregistration', str(prereg)]
    for run_dir in run_dirs:
        argv += ['--run', str(run_dir)]

    code = harness.main(argv, deps=_offline)

    assert code == harness.EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    assert load_embedding_preregistration_inputs(prereg) == inputs
    for spec, values in runs.items():
        comparison = compare_embedding_arm(
            inputs, embedding_run_manifest(spec), embedding_records(spec, **values)
        )
        rows = [line for line in lines if line.startswith(f'| {spec.arm_id} |')]
        assert len(rows) == len(inputs.margins)
        for row, verdict in zip(rows, comparison.margins, strict=True):
            assert verdict.metric_id in row and str(verdict.candidate_value) in row
        envelope = next(line for line in lines if line.startswith(f'envelope {spec.arm_id}:'))
        assert str(comparison.envelope.candidate_p95_ms) in envelope
        assert f'non_inferior {spec.arm_id}: {comparison.non_inferior}' in lines
    assert f'non_inferior {QWEN.arm_id}: False' in lines
