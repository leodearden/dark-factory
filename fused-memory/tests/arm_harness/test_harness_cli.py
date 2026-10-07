"""The thin arm-harness CLI: argument-to-call wiring, exit codes, and the scratch guard at its surface."""

import asyncio
import dataclasses
import importlib
import json
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from _fm_helpers import load_script_module
from _mock_openai_server import mock_openai_server
from shared.cli_boundary import EXIT_STDOUT_FAILED
from shared.memory_eval_metrics import Metric, canonical_json_text
from shared.safe_io import atomic_write_text

from arm_harness._fakes import (
    FINISHED_AT,
    FakeArmGraph,
    PreregRepo,
    RecordingJournal,
    embedding_spec,
    fail,
    incumbent_control_spec,
    llm_spec,
    make_prereg_repo,
    run_manifest_for,
)
from fused_memory.arm_harness.arm_spec import LlmArmSpec
from fused_memory.arm_harness.checks import CLEANUP_CYPHER, PROBE_CYPHER
from fused_memory.arm_harness.conformance import ConformanceLedger
from fused_memory.arm_harness.corpus import corpus_sha
from fused_memory.arm_harness.instrument_checks import InstrumentCheckId
from fused_memory.arm_harness.llm_metrics import (
    GRAPH_SAMENESS_DETAILS_FILENAME,
    EpisodeSameness,
    GraphSamenessDetails,
)
from fused_memory.arm_harness.metrics_record import (
    IndexConfiguration,
    LlmMetricId,
    MetricsRecord,
    load_metrics_record,
    load_metrics_records,
    record_for,
    write_metrics_record,
)
from fused_memory.arm_harness.preregistration import (
    PREREGISTRATION_INPUTS_FILENAME,
    derive_preregistration_inputs,
    load_preregistration_inputs,
    serialize_preregistration_inputs,
)
from fused_memory.arm_harness.replay_types import ArmAbort, EpisodeOutcome, ReplaySettings
from fused_memory.arm_harness.run import (
    ABORT_FILENAME,
    OUTCOMES_FILENAME,
    RUN_MANIFEST_FILENAME,
    load_outcomes,
    write_outcomes,
)
from fused_memory.arm_harness.run_manifest import (
    RunManifest,
    load_run_manifest,
    serialize_run_manifest,
)
from fused_memory.arm_harness.topology import EDGE_CYPHER, NODE_CYPHER, read_topology, topology_hash
from fused_memory.backends.llm_token_usage import LlmTokenUsage
from fused_memory.config.schema import FusedMemoryConfig

LME_DIR = Path(__file__).parents[2] / 'scripts' / 'local_memory_models_eval'
STAMP = '20261005T120000Z'
CORPUS_GRAPH = 'dark_factory'
MANIFEST_N = 6
VALID_ENTITIES = json.dumps({'entities': [{'name': 'Alice'}, {'name': 'Bob'}]})
OFF_SCHEMA = json.dumps({'unexpected': 1})


@pytest.fixture
def harness(monkeypatch) -> ModuleType:
    """The CLI module, loaded with its own directory importable, as a direct run resolves it."""
    monkeypatch.syspath_prepend(str(LME_DIR))
    return load_script_module(LME_DIR / 'harness.py', mod_name='lme_harness')


@pytest.fixture
def build_corpus(harness) -> ModuleType:
    """δ's module, the very object the CLI imported as its sibling."""
    return importlib.import_module('build_corpus')


@pytest.fixture
def repo(tmp_path: Path) -> PreregRepo:
    return make_prereg_repo(tmp_path / 'repo')


# --- fake live dependencies ----------------------------------------------------------


class FakeReader:
    def __init__(self, population: list[Any], graph_name: str, log: list[str]) -> None:
        self._population = population
        log.append(graph_name)

    async def fetch_population(self) -> list[Any]:
        return list(self._population)


class FakeScratchGraph:
    def __init__(self, name: str, client: 'FakeFalkor') -> None:
        self._name = name
        self._client = client

    async def query(self, q: str, params: dict[str, Any] | None = None) -> Any:
        self._client.calls.append(('query', self._name))
        return SimpleNamespace(result_set=[])

    async def ro_query(self, q: str, params: dict[str, Any] | None = None) -> Any:
        self._client.calls.append(('ro_query', self._name))
        return SimpleNamespace(result_set=self._client.rows(self._name, q))

    async def delete(self) -> None:
        self._client.calls.append(('delete', self._name))


class FakeFalkor:
    """A falkordb-client double: ``rows(graph, cypher)`` answers each ro_query; every call is logged."""

    def __init__(self, rows: Callable[[str, str], list[Any]] = lambda graph, cypher: []) -> None:
        self.rows = rows
        self.calls: list[tuple[str, str]] = []

    def select_graph(self, graph_id: str) -> FakeScratchGraph:
        self.calls.append(('select_graph', graph_id))
        return FakeScratchGraph(graph_id, self)


class FakeQdrant:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def delete_collection(self, collection_name: str) -> None:
        self.deleted.append(collection_name)


@dataclass
class DepsLog:
    """What the fake deps were asked for, in order."""

    factory_calls: int = 0
    reader_graphs: list[str] = field(default_factory=list)
    backend_opens: list[tuple[LlmArmSpec, ReplaySettings]] = field(default_factory=list)
    journal_paths: list[Path] = field(default_factory=list)
    falkordb_opens: int = 0
    qdrant_opens: int = 0


def _opener(value: Any, on_open: Callable[..., None]) -> Callable[..., Any]:
    @asynccontextmanager
    async def open_(*args: Any) -> AsyncIterator[Any]:
        on_open(*args)
        yield value

    return open_


def _ledger() -> ConformanceLedger:
    ledger = ConformanceLedger()
    for _ in range(3):
        ledger.record_valid()
    return ledger


@dataclass
class FakeLive:
    """Builds a ``deps`` factory over fakes, logging every call into ``log``."""

    base_config: FusedMemoryConfig
    population: list[Any] = field(default_factory=list)
    graph: FakeArmGraph = field(default_factory=FakeArmGraph)
    journal: RecordingJournal = field(default_factory=RecordingJournal)
    falkor: FakeFalkor = field(default_factory=FakeFalkor)
    qdrant: FakeQdrant = field(default_factory=FakeQdrant)
    log: DepsLog = field(default_factory=DepsLog)

    def factory(self, harness: ModuleType) -> Callable[[], Any]:
        log = self.log

        def build() -> Any:
            log.factory_calls += 1
            return harness.HarnessDeps(
                base_config=self.base_config,
                episode_reader=lambda name: FakeReader(self.population, name, log.reader_graphs),
                open_arm_backend=_opener(
                    (self.graph, _ledger()),
                    lambda spec, base, settings: log.backend_opens.append((spec, settings)),
                ),
                open_journal=_opener(self.journal, log.journal_paths.append),
                open_falkordb=_opener(self.falkor, lambda: _count(log, 'falkordb_opens')),
                open_qdrant=_opener(self.qdrant, lambda: _count(log, 'qdrant_opens')),
            )

        return build


def _count(log: DepsLog, name: str) -> None:
    setattr(log, name, getattr(log, name) + 1)


@pytest.fixture
def live(mock_config: FusedMemoryConfig) -> FakeLive:
    return FakeLive(base_config=mock_config)


# --- inputs on disk ------------------------------------------------------------------


def _population(build_corpus: ModuleType, count: int = 8) -> list[Any]:
    months = ('2026-04', '2026-05')
    return [
        build_corpus.EpisodeRecord(
            uuid=f'{i:08d}-0000-0000-0000-000000000000',
            name=f'ep-{i}',
            group_id=CORPUS_GRAPH,
            source_description='add_memory:temporal_facts',
            created_at=f'{months[i % 2]}-16T12:00:00+00:00',
            content=f'content of episode {i}',
        )
        for i in range(count)
    ]


@dataclass(frozen=True)
class Corpus:
    manifest_path: Path
    manifest: dict[str, Any]
    sha: str
    population: list[Any]

    @property
    def episode_ids(self) -> tuple[str, ...]:
        return tuple(entry['uuid'] for entry in self.manifest['episodes'])


@pytest.fixture
def corpus(build_corpus: ModuleType, tmp_path: Path) -> Corpus:
    """A manifest δ itself built over a synthetic population, so δ's verify_manifest says ok."""
    population = _population(build_corpus)
    selection = build_corpus.select(population, MANIFEST_N, seed='cli-test')
    manifest = build_corpus.build_manifest(
        selection, population, n=MANIFEST_N, seed='cli-test', graph=CORPUS_GRAPH
    )
    path = tmp_path / 'corpus_manifest.json'
    path.write_text(build_corpus.serialize_manifest(manifest), encoding='utf-8')
    return Corpus(
        manifest_path=path,
        manifest=manifest,
        sha=corpus_sha(path.read_bytes()),
        population=population,
    )


def _write_spec(path: Path, spec_data: Mapping[str, Any]) -> Path:
    path.write_text(json.dumps(dict(spec_data)), encoding='utf-8')
    return path


def _spec_file(tmp_path: Path, spec: LlmArmSpec, **raw_overrides: Any) -> Path:
    return _write_spec(
        tmp_path / f'{spec.arm_id}.json', spec.model_dump(mode='json') | raw_overrides
    )


@dataclass(frozen=True)
class RunInputs:
    spec: LlmArmSpec
    spec_path: Path
    out_root: Path

    @property
    def run_dir(self) -> Path:
        return self.out_root / self.spec.arm_id / STAMP


@pytest.fixture
def run_inputs(tmp_path: Path, repo: PreregRepo, corpus: Corpus, monkeypatch) -> RunInputs:
    """A control arm at the repo's clean HEAD over the corpus fixture, with the stamp pinned."""
    monkeypatch.setenv('MEMORY_EVAL_RUN_STAMP', STAMP)
    spec = incumbent_control_spec(code_sha=repo.with_prereg, corpus_sha=corpus.sha)
    return RunInputs(spec=spec, spec_path=_spec_file(tmp_path, spec), out_root=tmp_path / 'runs')


def _run_argv(
    inputs: RunInputs,
    corpus: Corpus,
    repo: PreregRepo,
    *extra: str,
    spec_path: Path | None = None,
) -> list[str]:
    return [
        'run',
        '--arm-spec', str(spec_path or inputs.spec_path),
        '--manifest', str(corpus.manifest_path),
        '--out-root', str(inputs.out_root),
        '--concurrency', '2',
        '--index-configuration', 'embedding-only',
        '--repo-root', str(repo.root),
        *extra,
    ]


# --- exit codes are named constants ----------------------------------------------------


def test_every_exit_code_is_a_named_constant(harness):
    assert harness.EXIT_OK == 0
    assert harness.EXIT_RUN_FAILED == EXIT_STDOUT_FAILED == 1
    assert harness.EXIT_REFUSED == 2
    assert harness.EXIT_CHECK_FAILED == 3
    assert harness.EXIT_ABORTED == 4
    assert harness.EXIT_SCRATCH_GUARD == 5
    assert harness.EXIT_CORPUS_INTEGRITY == 6


# --- (a) boundary row 2 at the CLI surface ---------------------------------------------


def _guarded_argv(spec_path: Path) -> dict[str, list[str]]:
    run_args = [
        '--manifest', 'unread.json', '--out-root', 'unwritten', '--concurrency', '1',
        '--index-configuration', 'with-indices',
    ]
    return {
        'run': ['run', '--arm-spec', str(spec_path), *run_args],
        'smoke': ['smoke', '--arm-spec', str(spec_path)],
        'index-check': ['index-check', '--arm-spec', str(spec_path), '--expect', 'with-indices'],
        'teardown': ['teardown', '--arm-spec', str(spec_path)],
        'teardown-collection': ['teardown', '--arm-spec', str(spec_path), '--collection'],
    }


@pytest.mark.parametrize(
    'subcommand', ['run', 'smoke', 'index-check', 'teardown', 'teardown-collection']
)
def test_a_protected_graph_in_the_spec_is_refused_before_any_dependency(
    harness, live, tmp_path, capsys, subcommand
):
    spec_path = _spec_file(tmp_path, incumbent_control_spec(), scratch_group_id='dark_factory')

    code = harness.main(_guarded_argv(spec_path)[subcommand], deps=live.factory(harness))

    assert code == harness.EXIT_SCRATCH_GUARD
    err = capsys.readouterr().err
    assert 'ScratchGuardError' in err
    assert "'arm-spec'" in err
    assert "'dark_factory'" in err
    assert live.log.factory_calls == 0
    assert live.falkor.calls == []


@pytest.mark.parametrize(
    ('reference', 'candidate'), [('dark_factory', 'evalmem_b'), ('evalmem_a', 'reify')]
)
def test_integrity_refuses_a_protected_graph_name_before_any_dependency(
    harness, live, capsys, reference, candidate
):
    argv = ['integrity', '--reference', reference, '--candidate', candidate]

    code = harness.main(argv, deps=live.factory(harness))

    assert code == harness.EXIT_SCRATCH_GUARD
    err = capsys.readouterr().err
    assert 'ScratchGuardError' in err
    assert "'topology-read'" in err
    assert live.log.factory_calls == 0
    assert live.falkor.calls == []


# --- (b) run writes <out-root>/<arm_id>/<STAMP>/run.json ---------------------------------


def test_run_replays_the_verified_corpus_into_a_stamped_run_dir(
    harness, live, run_inputs, corpus, repo, mock_config
):
    live.population = corpus.population

    code = harness.main(_run_argv(run_inputs, corpus, repo), deps=live.factory(harness))

    assert code == harness.EXIT_OK
    manifest = load_run_manifest(run_inputs.run_dir / RUN_MANIFEST_FILENAME)
    assert manifest.spec == run_inputs.spec
    assert manifest.episode_ids == corpus.episode_ids
    assert manifest.incomplete is False
    assert live.log.reader_graphs == [CORPUS_GRAPH]
    assert live.log.journal_paths == [run_inputs.run_dir / 'journal']
    ((opened_spec, settings),) = live.log.backend_opens
    assert opened_spec == run_inputs.spec
    assert settings.concurrency == 2
    assert settings.index_configuration is IndexConfiguration.EMBEDDING_ONLY
    assert settings.episode_timeout_s == mock_config.queue.backend_write_timeout_seconds
    assert {call['group_id'] for call in live.graph.add_calls} == {run_inputs.spec.scratch_group_id}
    assert [call['content'] for call in live.graph.add_calls] == [
        _content_of(corpus, episode_id) for episode_id in corpus.episode_ids
    ]


def _content_of(corpus: Corpus, episode_id: str) -> str:
    (record,) = [record for record in corpus.population if record.uuid == episode_id]
    return record.content


def test_run_limit_replays_only_the_first_manifest_episodes(
    harness, live, run_inputs, corpus, repo
):
    live.population = corpus.population

    code = harness.main(
        _run_argv(run_inputs, corpus, repo, '--limit', '2'), deps=live.factory(harness)
    )

    assert code == harness.EXIT_OK
    manifest = load_run_manifest(run_inputs.run_dir / RUN_MANIFEST_FILENAME)
    assert manifest.episode_ids == corpus.episode_ids[:2]


def test_run_passes_reference_outcomes_through_to_the_run(
    harness, live, run_inputs, corpus, repo, tmp_path
):
    live.population = corpus.population
    reference_path = write_outcomes(tmp_path / 'reference', _reference(corpus.episode_ids))

    code = harness.main(
        _run_argv(run_inputs, corpus, repo, '--reference-outcomes', str(reference_path)),
        deps=live.factory(harness),
    )

    assert code == harness.EXIT_OK
    manifest = load_run_manifest(run_inputs.run_dir / RUN_MANIFEST_FILENAME)
    reference_checks = [
        check for check in manifest.check_results
        if check.check_id is InstrumentCheckId.REFERENCE_NONEMPTY
    ]
    assert [check.passed for check in reference_checks] == [True]


def _reference(episode_ids: tuple[str, ...]) -> list[EpisodeOutcome]:
    return [
        EpisodeOutcome(
            episode_id=episode_id,
            ok=True,
            error_class=None,
            duration_ms=10.0,
            tokens=LlmTokenUsage(input_tokens=30, output_tokens=10, llm_calls=1),
            replay_episode_uuid=f'ref-{episode_id}',
            entity_names=('alice',),
            edge_triples=(),
        )
        for episode_id in episode_ids
    ]


def test_run_refuses_a_stamped_run_dir_that_already_holds_artifacts(
    harness, live, run_inputs, corpus, repo
):
    live.population = corpus.population
    run_inputs.run_dir.mkdir(parents=True)
    (run_inputs.run_dir / 'outcomes.jsonl').write_text('{}\n')

    code = harness.main(_run_argv(run_inputs, corpus, repo), deps=live.factory(harness))

    assert code == harness.EXIT_REFUSED
    assert live.log.backend_opens == []
    assert not (run_inputs.run_dir / RUN_MANIFEST_FILENAME).exists()


# --- (c) every outcome has its own exit code ------------------------------------------


def test_a_failed_post_run_instrument_check_exits_3_with_the_run_recorded(
    harness, live, run_inputs, corpus, repo
):
    live.population = corpus.population
    live.graph = FakeArmGraph(usage=None)

    code = harness.main(_run_argv(run_inputs, corpus, repo), deps=live.factory(harness))

    assert code == harness.EXIT_CHECK_FAILED
    manifest = load_run_manifest(run_inputs.run_dir / RUN_MANIFEST_FILENAME)
    failed = [check.check_id for check in manifest.check_results if not check.passed]
    assert failed == [InstrumentCheckId.TOKEN_COST_ACCOUNTING]


def test_an_inv4_abort_exits_4_with_an_incomplete_run_and_abort_json(
    harness, live, run_inputs, corpus, repo
):
    live.population = corpus.population
    live.graph = FakeArmGraph(default=fail)

    code = harness.main(_run_argv(run_inputs, corpus, repo), deps=live.factory(harness))

    assert code == harness.EXIT_ABORTED
    manifest = load_run_manifest(run_inputs.run_dir / RUN_MANIFEST_FILENAME)
    assert manifest.incomplete is True
    assert (run_inputs.run_dir / ABORT_FILENAME).is_file()


def test_a_pre_run_refusal_exits_2_before_any_backend_journal_or_artifact(
    harness, live, run_inputs, corpus, repo, tmp_path, capsys
):
    live.population = corpus.population
    stale = run_inputs.spec.model_copy(update={'code_sha': repo.without_prereg})
    (tmp_path / 'stale').mkdir()
    stale_path = _spec_file(tmp_path / 'stale', stale)

    code = harness.main(
        _run_argv(run_inputs, corpus, repo, spec_path=stale_path), deps=live.factory(harness)
    )

    assert code == harness.EXIT_REFUSED
    assert InstrumentCheckId.CODE_SHA_MATCHES_CHECKOUT.value in capsys.readouterr().err
    assert live.log.backend_opens == []
    assert live.log.journal_paths == []
    assert not run_inputs.run_dir.exists()


def test_corpus_drift_reported_by_deltas_verifier_exits_6_before_any_backend(
    harness, live, run_inputs, corpus, repo, capsys
):
    drifted_id = corpus.episode_ids[0]
    live.population = [
        record if record.uuid != drifted_id else _with_content(record, 'edited body')
        for record in corpus.population
    ]

    code = harness.main(_run_argv(run_inputs, corpus, repo), deps=live.factory(harness))

    assert code == harness.EXIT_CORPUS_INTEGRITY
    err = capsys.readouterr().err
    assert 'hash_drift' in err
    assert drifted_id in err
    assert live.log.reader_graphs == [CORPUS_GRAPH]
    assert live.log.backend_opens == []
    assert not run_inputs.run_dir.exists()


def _with_content(record: Any, content: str) -> Any:
    return dataclasses.replace(record, content=content)


def test_a_spec_whose_corpus_sha_is_not_the_manifests_exits_6_before_reading_the_store(
    harness, live, run_inputs, corpus, repo, tmp_path, capsys
):
    live.population = corpus.population
    other = run_inputs.spec.model_copy(update={'corpus_sha': 'f' * 64})
    (tmp_path / 'other').mkdir()
    other_path = _spec_file(tmp_path / 'other', other)

    code = harness.main(
        _run_argv(run_inputs, corpus, repo, spec_path=other_path), deps=live.factory(harness)
    )

    assert code == harness.EXIT_CORPUS_INTEGRITY
    err = capsys.readouterr().err
    assert 'f' * 64 in err
    assert corpus.sha in err
    assert live.log.factory_calls == 0


# --- (d) smoke ------------------------------------------------------------------------


def test_smoke_passes_against_a_schema_valid_endpoint(harness, live, tmp_path, capsys):
    with mock_openai_server() as server:
        server.chat_content = VALID_ENTITIES
        spec_path = _spec_file(tmp_path, llm_spec(base_url=server.base_url))
        code = harness.main(['smoke', '--arm-spec', str(spec_path)], deps=live.factory(harness))
        hits = server.requests_to('/chat/completions')

    assert code == harness.EXIT_OK
    assert hits, 'the arm base_url received no traffic'
    out = capsys.readouterr().out
    assert InstrumentCheckId.ENDPOINT_CONFORMANCE.value in out
    assert InstrumentCheckId.VALIDATOR_NEGATIVE_CONTROL.value in out


def test_smoke_failure_is_a_non_zero_check_failure(harness, live, tmp_path):
    with mock_openai_server() as server:
        server.chat_content = OFF_SCHEMA
        spec_path = _spec_file(tmp_path, llm_spec(base_url=server.base_url))
        code = harness.main(['smoke', '--arm-spec', str(spec_path)], deps=live.factory(harness))

    assert code == harness.EXIT_CHECK_FAILED


# --- index-check ----------------------------------------------------------------------


def _fulltext_rows(answering: bool) -> Callable[[str, str], list[Any]]:
    def rows(graph: str, cypher: str) -> list[Any]:
        return [['probe-node']] if answering and cypher == PROBE_CYPHER else []

    return rows


@pytest.mark.parametrize(
    ('answering', 'expect', 'expected_code'),
    [
        (True, 'with-indices', 'EXIT_OK'),
        (True, 'embedding-only', 'EXIT_CHECK_FAILED'),
    ],
)
def test_index_check_probes_only_the_arms_scratch_graph(
    harness, live, tmp_path, answering, expect, expected_code
):
    spec = llm_spec()
    live.falkor = FakeFalkor(rows=_fulltext_rows(answering))

    code = harness.main(
        ['index-check', '--arm-spec', str(_spec_file(tmp_path, spec)), '--expect', expect],
        deps=live.factory(harness),
    )

    assert code == getattr(harness, expected_code)
    assert {graph for _, graph in live.falkor.calls} == {spec.scratch_group_id}


class _CleanupFailingGraph(FakeScratchGraph):
    async def query(self, q: str, params: dict[str, Any] | None = None) -> Any:
        if q == CLEANUP_CYPHER:
            raise ConnectionError('connection dropped')
        return await super().query(q, params)


class _CleanupFailingFalkor(FakeFalkor):
    def select_graph(self, graph_id: str) -> FakeScratchGraph:
        self.calls.append(('select_graph', graph_id))
        return _CleanupFailingGraph(graph_id, self)


def test_index_check_whose_probe_node_survives_exits_1_naming_it(harness, live, tmp_path, capsys):
    spec = llm_spec()
    live.falkor = _CleanupFailingFalkor()

    code = harness.main(
        ['index-check', '--arm-spec', str(_spec_file(tmp_path, spec)), '--expect', 'embedding-only'],
        deps=live.factory(harness),
    )

    assert code == harness.EXIT_RUN_FAILED
    err = capsys.readouterr().err
    assert 'ProbeCleanupError' in err
    assert spec.scratch_group_id in err


# --- integrity ------------------------------------------------------------------------


def _topology_rows(changed_graph: str | None) -> Callable[[str, str], list[Any]]:
    def rows(graph: str, cypher: str) -> list[Any]:
        if cypher == NODE_CYPHER:
            return [
                [{'uuid': 'n1', 'labels': ['Entity'], 'name': 'alice'}],
                [{'uuid': 'n2', 'labels': ['Entity'], 'name': 'bob'}],
            ]
        if cypher == EDGE_CYPHER:
            fact = 'alice knows carol' if graph == changed_graph else 'alice knows bob'
            return [[{
                'uuid': 'e1', 'rel_type': 'RELATES_TO', 'source_uuid': 'n1',
                'target_uuid': 'n2', 'name': 'KNOWS', 'fact': fact,
            }]]
        return []

    return rows


@pytest.mark.parametrize(
    ('changed_graph', 'expected_code'),
    [(None, 'EXIT_OK'), ('evalmem_candidate', 'EXIT_CHECK_FAILED')],
)
def test_integrity_compares_two_scratch_topologies(
    harness, live, capsys, changed_graph, expected_code
):
    live.falkor = FakeFalkor(rows=_topology_rows(changed_graph))
    argv = ['integrity', '--reference', 'evalmem_reference', '--candidate', 'evalmem_candidate']

    code = harness.main(argv, deps=live.factory(harness))

    assert code == getattr(harness, expected_code)
    assert {graph for _, graph in live.falkor.calls} == {'evalmem_reference', 'evalmem_candidate'}
    assert {kind for kind, _ in live.falkor.calls} <= {'select_graph', 'ro_query'}
    verdict = json.loads(capsys.readouterr().out)
    assert verdict['identical'] is (changed_graph is None)
    assert verdict['changed_edge_uuids'] == ([] if changed_graph is None else ['e1'])


# --- parity-check ---------------------------------------------------------------------

MEASURED_AT = FINISHED_AT


def _scalar_record(spec: LlmArmSpec, metric_id: LlmMetricId, value: float) -> MetricsRecord:
    metric = Metric(metric_id=metric_id, kind='scalar', value=value, n=3)
    return record_for(spec, metric, measured_at=MEASURED_AT, incomplete=False)


def _write_run(run_dir: Path, manifest: RunManifest, records: list[MetricsRecord]) -> Path:
    atomic_write_text(
        run_dir / RUN_MANIFEST_FILENAME, serialize_run_manifest(manifest), mkdir=True
    )
    for record in records:
        write_metrics_record(record, run_dir)
    return run_dir


def _accounted(spec: LlmArmSpec, tokens: float = 900.0) -> list[MetricsRecord]:
    return [
        _scalar_record(spec, LlmMetricId.TOKENS_PER_EPISODE, tokens),
        _scalar_record(spec, LlmMetricId.USD_PER_EPISODE, 0.001),
    ]


def _parity_specs() -> tuple[LlmArmSpec, LlmArmSpec]:
    openai = incumbent_control_spec(
        arm_id='incumbent-openai', client_class='openai', scratch_group_id='evalmem_par_a'
    )
    generic = incumbent_control_spec(
        arm_id='incumbent-generic', client_class='openai_generic', scratch_group_id='evalmem_par_b'
    )
    return openai, generic


def test_parity_check_writes_delta_records_under_run_a_parity(harness, live, tmp_path):
    spec_a, spec_b = _parity_specs()
    run_a = _write_run(tmp_path / 'a', run_manifest_for(spec_a), _accounted(spec_a, 900.0))
    run_b = _write_run(tmp_path / 'b', run_manifest_for(spec_b), _accounted(spec_b, 850.0))

    code = harness.main(
        ['parity-check', '--run-a', str(run_a), '--run-b', str(run_b)],
        deps=live.factory(harness),
    )

    assert code == harness.EXIT_OK
    deltas = {
        record.metric.metric_id: record
        for record in map(load_metrics_record, sorted((run_a / 'parity').rglob('*.json')))
    }
    assert set(deltas) == {LlmMetricId.TOKENS_PER_EPISODE, LlmMetricId.USD_PER_EPISODE}
    tokens = deltas[LlmMetricId.TOKENS_PER_EPISODE]
    assert tokens.metric.value == pytest.approx(50.0)
    assert tokens.delta_of is not None
    assert tokens.delta_of.minuend_arm_id == spec_a.arm_id
    assert tokens.delta_of.subtrahend_arm_id == spec_b.arm_id
    assert live.log.factory_calls == 0


def test_parity_check_refuses_an_incomplete_run(harness, live, tmp_path, capsys):
    spec_a, spec_b = _parity_specs()
    aborted = run_manifest_for(
        spec_b,
        incomplete=True,
        abort=ArmAbort(arm_id=spec_b.arm_id, item_ids=('e1',), error_classes=('RuntimeError',)),
    )
    run_a = _write_run(tmp_path / 'a', run_manifest_for(spec_a), _accounted(spec_a))
    run_b = _write_run(tmp_path / 'b', aborted, _accounted(spec_b))

    code = harness.main(
        ['parity-check', '--run-a', str(run_a), '--run-b', str(run_b)],
        deps=live.factory(harness),
    )

    assert code == harness.EXIT_REFUSED
    assert 'incomplete' in capsys.readouterr().err
    assert not (run_a / 'parity').exists()


def test_parity_check_refuses_an_interrupted_run_without_run_json(
    harness, live, tmp_path, capsys
):
    spec_a, _ = _parity_specs()
    run_a = _write_run(tmp_path / 'a', run_manifest_for(spec_a), _accounted(spec_a))
    interrupted = tmp_path / 'b'
    interrupted.mkdir()

    code = harness.main(
        ['parity-check', '--run-a', str(run_a), '--run-b', str(interrupted)],
        deps=live.factory(harness),
    )

    assert code == harness.EXIT_REFUSED
    assert RUN_MANIFEST_FILENAME in capsys.readouterr().err


# --- control-check --------------------------------------------------------------------


def _control(arm_id: str, scratch: str, **overrides: Any) -> LlmArmSpec:
    return incumbent_control_spec(arm_id=arm_id, scratch_group_id=scratch, **overrides)


def test_control_check_passes_two_symmetric_accounted_control_runs(
    harness, live, tmp_path, capsys
):
    spec_a, spec_b = _control('ctrl-a', 'evalmem_ctrl_a'), _control('ctrl-b', 'evalmem_ctrl_b')
    run_a = _write_run(tmp_path / 'a', run_manifest_for(spec_a), _accounted(spec_a))
    run_b = _write_run(tmp_path / 'b', run_manifest_for(spec_b), _accounted(spec_b))
    reference_path = write_outcomes(tmp_path / 'reference', _reference(('e1', 'e2')))

    code = harness.main(
        ['control-check', '--run', str(run_a), '--run', str(run_b),
         '--reference-outcomes', str(reference_path)],
        deps=live.factory(harness),
    )

    assert code == harness.EXIT_OK
    out = capsys.readouterr().out
    for check_id in (
        InstrumentCheckId.ARM_CONFIG_SYMMETRY,
        InstrumentCheckId.SINGLE_CODE_SHA,
        InstrumentCheckId.TOKEN_COST_ACCOUNTING,
        InstrumentCheckId.REFERENCE_NONEMPTY,
    ):
        assert check_id.value in out
    assert live.log.factory_calls == 0


def test_control_check_fails_on_an_asymmetric_temperature(harness, live, tmp_path, capsys):
    spec_a = _control('ctrl-a', 'evalmem_ctrl_a')
    spec_b = _control('ctrl-b', 'evalmem_ctrl_b', params={'temperature': 0.7, 'max_tokens': 4096})
    run_a = _write_run(tmp_path / 'a', run_manifest_for(spec_a), _accounted(spec_a))
    run_b = _write_run(tmp_path / 'b', run_manifest_for(spec_b), _accounted(spec_b))

    code = harness.main(
        ['control-check', '--run', str(run_a), '--run', str(run_b)], deps=live.factory(harness)
    )

    assert code == harness.EXIT_CHECK_FAILED
    assert 'temperature' in capsys.readouterr().out


def test_control_check_refuses_two_runs_of_one_arm(harness, live, tmp_path):
    spec = _control('ctrl-a', 'evalmem_ctrl_a')
    run_a = _write_run(tmp_path / 'a', run_manifest_for(spec), _accounted(spec))
    run_b = _write_run(tmp_path / 'b', run_manifest_for(spec), _accounted(spec))

    code = harness.main(
        ['control-check', '--run', str(run_a), '--run', str(run_b)], deps=live.factory(harness)
    )

    assert code == harness.EXIT_REFUSED


def test_control_check_refuses_a_limited_run_against_a_full_one(harness, live, tmp_path, capsys):
    spec_a, spec_b = _control('ctrl-a', 'evalmem_ctrl_a'), _control('ctrl-b', 'evalmem_ctrl_b')
    limited = run_manifest_for(spec_b, episode_ids=('e1',))
    run_a = _write_run(tmp_path / 'a', run_manifest_for(spec_a), _accounted(spec_a))
    run_b = _write_run(tmp_path / 'b', limited, _accounted(spec_b))

    code = harness.main(
        ['control-check', '--run', str(run_a), '--run', str(run_b)], deps=live.factory(harness)
    )

    assert code == harness.EXIT_REFUSED
    captured = capsys.readouterr()
    assert "lacks ['e2', 'e3']" in captured.err
    assert captured.out == ''


def test_control_check_refuses_an_embedding_arm_run(harness, live, tmp_path, capsys):
    spec_a = _control('ctrl-a', 'evalmem_ctrl_a')
    run_a = _write_run(tmp_path / 'a', run_manifest_for(spec_a), _accounted(spec_a))
    run_b = _write_run(tmp_path / 'b', run_manifest_for(embedding_spec()), [])

    code = harness.main(
        ['control-check', '--run', str(run_a), '--run', str(run_b)], deps=live.factory(harness)
    )

    assert code == harness.EXIT_REFUSED
    assert 'embedding arm' in capsys.readouterr().err


# --- unreadable inputs are refusals (exit 2), never tracebacks -------------------------

_RUN_CORRUPTIONS = {
    'run-json-not-json': (RUN_MANIFEST_FILENAME, '{not json'),
    'run-json-missing-fields': (RUN_MANIFEST_FILENAME, '{"schema_version": 1}'),
    'metrics-record-not-json': ('metrics/tokens-per-episode.json', '{not json'),
}


def _comparison_argv(command: str, run_a: Path, run_b: Path) -> list[str]:
    if command == 'parity-check':
        return ['parity-check', '--run-a', str(run_a), '--run-b', str(run_b)]
    return ['control-check', '--run', str(run_a), '--run', str(run_b)]


@pytest.mark.parametrize('corruption', sorted(_RUN_CORRUPTIONS))
@pytest.mark.parametrize('command', ['parity-check', 'control-check'])
def test_an_unreadable_run_is_refused_naming_it(
    harness, live, tmp_path, capsys, command, corruption
):
    spec_a, spec_b = _control('ctrl-a', 'evalmem_ctrl_a'), _control('ctrl-b', 'evalmem_ctrl_b')
    run_a = _write_run(tmp_path / 'a', run_manifest_for(spec_a), _accounted(spec_a))
    run_b = _write_run(tmp_path / 'b', run_manifest_for(spec_b), _accounted(spec_b))
    relative, text = _RUN_CORRUPTIONS[corruption]
    (run_b / relative).write_text(text)

    code = harness.main(_comparison_argv(command, run_a, run_b), deps=live.factory(harness))

    assert code == harness.EXIT_REFUSED
    assert f'{run_b} is not a readable run' in capsys.readouterr().err
    assert not (run_a / harness.PARITY_DIRNAME).exists()


@pytest.mark.parametrize('content', [None, '{not json\n'], ids=['missing', 'not-json'])
def test_control_check_refuses_an_unreadable_reference(harness, live, tmp_path, capsys, content):
    spec_a, spec_b = _control('ctrl-a', 'evalmem_ctrl_a'), _control('ctrl-b', 'evalmem_ctrl_b')
    run_a = _write_run(tmp_path / 'a', run_manifest_for(spec_a), _accounted(spec_a))
    run_b = _write_run(tmp_path / 'b', run_manifest_for(spec_b), _accounted(spec_b))
    reference = tmp_path / 'reference.jsonl'
    if content is not None:
        reference.write_text(content)

    code = harness.main(
        ['control-check', '--run', str(run_a), '--run', str(run_b),
         '--reference-outcomes', str(reference)],
        deps=live.factory(harness),
    )

    assert code == harness.EXIT_REFUSED
    assert f'{reference} is not a readable outcomes file' in capsys.readouterr().err


def test_run_refuses_an_unreadable_reference_before_any_backend(
    harness, live, run_inputs, corpus, repo, tmp_path, capsys
):
    live.population = corpus.population
    reference = tmp_path / 'reference.jsonl'
    reference.write_text('{not json\n')

    code = harness.main(
        _run_argv(run_inputs, corpus, repo, '--reference-outcomes', str(reference)),
        deps=live.factory(harness),
    )

    assert code == harness.EXIT_REFUSED
    assert f'{reference} is not a readable outcomes file' in capsys.readouterr().err
    assert live.log.factory_calls == 0
    assert not run_inputs.run_dir.exists()


def test_control_check_needs_at_least_two_runs(harness, live, tmp_path):
    spec = _control('ctrl-a', 'evalmem_ctrl_a')
    run_a = _write_run(tmp_path / 'a', run_manifest_for(spec), _accounted(spec))

    with pytest.raises(SystemExit) as raised:
        harness.main(['control-check', '--run', str(run_a)], deps=live.factory(harness))

    assert raised.value.code == 2


# --- (e) teardown ---------------------------------------------------------------------


def test_teardown_deletes_only_the_arms_scratch_graph(harness, live, tmp_path):
    spec = llm_spec()

    code = harness.main(
        ['teardown', '--arm-spec', str(_spec_file(tmp_path, spec))], deps=live.factory(harness)
    )

    assert code == harness.EXIT_OK
    assert live.falkor.calls == [
        ('select_graph', spec.scratch_group_id),
        ('delete', spec.scratch_group_id),
    ]
    assert live.log.qdrant_opens == 0
    assert live.qdrant.deleted == []


def test_teardown_collection_also_deletes_the_same_named_replica(harness, live, tmp_path):
    spec = llm_spec()

    code = harness.main(
        ['teardown', '--arm-spec', str(_spec_file(tmp_path, spec)), '--collection'],
        deps=live.factory(harness),
    )

    assert code == harness.EXIT_OK
    assert live.falkor.calls == [
        ('select_graph', spec.scratch_group_id),
        ('delete', spec.scratch_group_id),
    ]
    assert live.qdrant.deleted == [spec.scratch_group_id]


# --- preregister ----------------------------------------------------------------------

PREREG_EPISODES = ('e1', 'e2', 'e3', 'e4')
PREREG_JACCARDS = (0.6, 0.8, 1.0, 0.8)


def _prereg_outcomes(calls: tuple[int, ...]) -> list[EpisodeOutcome]:
    return [
        EpisodeOutcome(
            episode_id=episode_id,
            ok=True,
            error_class=None,
            duration_ms=3000.0,
            tokens=LlmTokenUsage(input_tokens=1000 * n, output_tokens=100 * n, llm_calls=n),
            replay_episode_uuid=f'replay-{episode_id}',
            entity_names=('alice',),
            edge_triples=(),
        )
        for episode_id, n in zip(PREREG_EPISODES, calls, strict=True)
    ]


def _prereg_records(
    spec: LlmArmSpec, *, p95_ms: float, sameness: bool
) -> list[MetricsRecord]:
    def proportion(metric_id: LlmMetricId, hits: int, total: int, direction: Any) -> Metric:
        return Metric(
            metric_id=metric_id, kind='proportion', value=hits / total, n=total,
            denominator=total, direction=direction,
        )

    metrics = [
        proportion(LlmMetricId.CONFORMANCE_RATE, 26, 26, 'lower_is_worse'),
        proportion(LlmMetricId.EPISODE_FAILURE_RATE, 0, 4, 'higher_is_worse'),
        proportion(LlmMetricId.RETRIEVAL_UTILITY, 3, 4, 'lower_is_worse'),
        Metric(metric_id=LlmMetricId.EPISODE_LATENCY_P95, kind='scalar', value=p95_ms, n=4),
    ]
    if sameness:
        mean = sum(PREREG_JACCARDS) / len(PREREG_JACCARDS)
        metrics.append(Metric(metric_id=LlmMetricId.GRAPH_SAMENESS, kind='scalar', value=mean, n=4))
    measured = [record_for(spec, m, measured_at=MEASURED_AT, incomplete=False) for m in metrics]
    return measured + _accounted(spec)


def _sameness_details_json() -> str:
    details = GraphSamenessDetails(
        episodes=tuple(
            EpisodeSameness(
                episode_id=episode_id, arm_entity_count=1, ref_entity_count=1,
                arm_edge_count=0, ref_edge_count=0, entity_jaccard=jaccard,
                edge_triple_jaccard=1.0,
            )
            for episode_id, jaccard in zip(PREREG_EPISODES, PREREG_JACCARDS, strict=True)
        ),
        excluded_arm_ids=(),
        excluded_reference_ids=(),
    )
    return canonical_json_text(details.model_dump(mode='json'))


def _control_run_dir(
    root: Path, spec: LlmArmSpec, calls: tuple[int, ...], *, p95_ms: float = 4000.0,
    sameness: bool = False,
) -> Path:
    manifest = run_manifest_for(spec, episode_ids=PREREG_EPISODES)
    run_dir = _write_run(root / spec.arm_id, manifest, _prereg_records(
        spec, p95_ms=p95_ms, sameness=sameness
    ))
    write_outcomes(run_dir, _prereg_outcomes(calls))
    if sameness:
        atomic_write_text(run_dir / GRAPH_SAMENESS_DETAILS_FILENAME, _sameness_details_json())
    return run_dir


def _control_pair(tmp_path: Path, **b_overrides: Any) -> tuple[Path, Path]:
    spec_a, spec_b = _control('ctrl-a', 'evalmem_ctrl_a'), _control('ctrl-b', 'evalmem_ctrl_b')
    run_a = _control_run_dir(tmp_path, spec_a, (3, 5, 8, 10))
    run_b = _control_run_dir(tmp_path, spec_b, (4, 4, 6, 6), **({'sameness': True} | b_overrides))
    return run_a, run_b


def _preregister_argv(run_a: Path, run_b: Path, out: Path) -> list[str]:
    return ['preregister', '--run-a', str(run_a), '--run-b', str(run_b), '--out', str(out)]


def test_preregister_writes_the_inputs_derived_from_the_control_pair(
    harness, live, tmp_path, capsys
):
    run_a, run_b = _control_pair(tmp_path)
    out = tmp_path / 'out' / PREREGISTRATION_INPUTS_FILENAME

    code = harness.main(_preregister_argv(run_a, run_b, out), deps=live.factory(harness))

    assert code == harness.EXIT_OK
    expected = derive_preregistration_inputs(
        load_run_manifest(run_a / RUN_MANIFEST_FILENAME),
        load_metrics_records(run_a),
        load_outcomes(run_a / OUTCOMES_FILENAME),
        load_run_manifest(run_b / RUN_MANIFEST_FILENAME),
        load_metrics_records(run_b),
        load_outcomes(run_b / OUTCOMES_FILENAME),
        GraphSamenessDetails.model_validate_json(
            (run_b / GRAPH_SAMENESS_DETAILS_FILENAME).read_text()
        ),
    )
    assert load_preregistration_inputs(out) == expected
    assert out.read_text() == serialize_preregistration_inputs(expected)
    lines = capsys.readouterr().out.splitlines()
    margin_lines = [line for line in lines if line.startswith('margin ')]
    assert len(margin_lines) == len(expected.margins)
    for entry, line in zip(expected.margins, margin_lines, strict=True):
        assert entry.metric_id in line
    assert any('60000.0' in line for line in lines)
    assert live.log.factory_calls == 0


def _refused_preregister(
    harness: ModuleType, live: FakeLive, run_a: Path, run_b: Path, out: Path, capsys: Any
) -> str:
    code = harness.main(_preregister_argv(run_a, run_b, out), deps=live.factory(harness))

    assert code == harness.EXIT_REFUSED
    assert not out.exists()
    assert live.log.factory_calls == 0
    err = capsys.readouterr().err
    assert err.startswith('error: ')
    return err


def test_preregister_refuses_a_run_b_without_graph_sameness_details(
    harness, live, tmp_path, capsys
):
    run_a, run_b = _control_pair(tmp_path)
    (run_b / GRAPH_SAMENESS_DETAILS_FILENAME).unlink()

    err = _refused_preregister(harness, live, run_a, run_b, tmp_path / 'f.json', capsys)

    assert GRAPH_SAMENESS_DETAILS_FILENAME in err
    assert '--reference-outcomes' in err


def test_preregister_refuses_an_incumbent_outside_the_envelope(harness, live, tmp_path, capsys):
    run_a, run_b = _control_pair(tmp_path, p95_ms=61000.0)

    err = _refused_preregister(harness, live, run_a, run_b, tmp_path / 'f.json', capsys)

    assert 'not a valid pre-registration' in err


def test_preregister_refuses_a_run_dir_without_run_json(harness, live, tmp_path, capsys):
    run_a, _ = _control_pair(tmp_path)
    interrupted = tmp_path / 'interrupted'
    interrupted.mkdir()

    err = _refused_preregister(harness, live, run_a, interrupted, tmp_path / 'f.json', capsys)

    assert RUN_MANIFEST_FILENAME in err


def test_preregister_refuses_missing_outcomes(harness, live, tmp_path, capsys):
    run_a, run_b = _control_pair(tmp_path)
    (run_a / OUTCOMES_FILENAME).unlink()

    err = _refused_preregister(harness, live, run_a, run_b, tmp_path / 'f.json', capsys)

    assert OUTCOMES_FILENAME in err


def test_preregister_refuses_an_embedding_arm_run(harness, live, tmp_path, capsys):
    run_a, _ = _control_pair(tmp_path)
    run_b = _write_run(tmp_path / 'emb', run_manifest_for(embedding_spec()), [])

    err = _refused_preregister(harness, live, run_a, run_b, tmp_path / 'f.json', capsys)

    assert 'embedding arm' in err


# --- topology -------------------------------------------------------------------------


def test_topology_prints_the_scratch_graphs_hash(harness, live, capsys):
    live.falkor = FakeFalkor(rows=_topology_rows(None))

    code = harness.main(['topology', '--graph', 'evalmem_x'], deps=live.factory(harness))

    assert code == harness.EXIT_OK
    expected = asyncio.run(
        read_topology(FakeFalkor(rows=_topology_rows(None)).select_graph('evalmem_x'), 'evalmem_x')
    )
    assert json.loads(capsys.readouterr().out) == {
        'graph': 'evalmem_x',
        'node_count': 2,
        'edge_count': 1,
        'topology_hash': topology_hash(*expected),
    }
    assert {graph for _, graph in live.falkor.calls} == {'evalmem_x'}
    assert {kind for kind, _ in live.falkor.calls} <= {'select_graph', 'ro_query'}


def test_topology_refuses_a_protected_graph_before_any_dependency(harness, live, capsys):
    code = harness.main(['topology', '--graph', 'dark_factory'], deps=live.factory(harness))

    assert code == harness.EXIT_SCRATCH_GUARD
    assert live.log.factory_calls == 0
    assert live.falkor.calls == []
