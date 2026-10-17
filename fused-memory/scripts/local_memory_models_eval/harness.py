#!/usr/bin/env python3
"""Arm-runner harness CLI for the local-memory-models eval (task ε).

Argument wiring and live dependencies only; every behaviour lives in
``fused_memory.arm_harness``. Subcommands:

  run            replay δ's verified corpus through one LLM arm into
                 <out-root>/<arm_id>/<STAMP>/ (STAMP: $MEMORY_EVAL_RUN_STAMP or now)
  smoke          one schema-constrained request to the arm endpoint, plus a negative control
  index-check    whether the arm's scratch graph answers fulltext, as --expect claims
  integrity      whether a re-embedded scratch graph kept the reference topology
  parity-check   client-class parity deltas between two complete runs
  control-check  symmetry, code sha, token/cost and reference checks over control runs
  preregister    the pre-registration inputs (margins, envelope, call profile) of a control pair
  screen         η's screening verdict over one sweep's evidence root (offline post-processing)
  incumbent-cost the incumbent's measured LLM spend from production telemetry and the
                 controls (offline)
  topology       a scratch graph's node/edge counts and topology hash
  teardown       delete the arm's scratch graph and, with --collection, its replica

The embedding axis (task ι):

  mem0-snapshot      a frozen, read-only JSONL snapshot of one Mem0 collection
  probe-set          the embedding probe set, read from the frozen reference graph
  embed-specs        the two incumbent control specs and one spec per slate arm
  embed-run          one embedding arm, end to end, into <out-root>/<arm_id>/<STAMP>/
  embed-preregister  the embedding pre-registration inputs of a control pair
  embed-compare      each candidate run judged against those inputs (offline)

Exit codes, run-directory layout and the live check:
README.md §"Arm-runner harness (task ε)" beside this script.
"""

import argparse
import asyncio
import contextlib
import dataclasses
import hashlib
import importlib.util
import itertools
import json
import sys
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, TypeVar
from urllib.parse import urlparse

import build_corpus
from falkordb.asyncio import FalkorDB
from graphiti_core.embedder import EmbedderClient
from qdrant_client import AsyncQdrantClient
from redis.exceptions import RedisError
from shared.cli_boundary import LoudArgumentParser, run_cli
from shared.memory_eval_metrics import canonical_json_text, run_stamp
from shared.safe_io import atomic_write_text

from fused_memory.arm_harness.arm_backend import (
    IndexBuildError,
    open_arm_backend,
    open_embedding_arm_backend,
)
from fused_memory.arm_harness.arm_embedder import (
    ArmEmbedder,
    EmbedSettings,
    QueryEmbedder,
    build_arm_embedder,
)
from fused_memory.arm_harness.arm_spec import EmbeddingArmSpec, LlmArmSpec, load_arm_spec
from fused_memory.arm_harness.checks import (
    ProbeCleanupError,
    check_index_configuration,
    control_variance_check,
    smoke_endpoint,
)
from fused_memory.arm_harness.comparison import RunComparabilityError, client_class_parity
from fused_memory.arm_harness.conformance import ConformanceLedger
from fused_memory.arm_harness.corpus import (
    CorpusIntegrityError,
    corpus_sha,
    select_replay_items,
)
from fused_memory.arm_harness.embedding_graph_phase import (
    SEARCH_K,
    ArmScratchGraph,
    EmbeddingRunCheckFailed,
    EmbeddingSearchBackend,
    frozen_reference_check,
)
from fused_memory.arm_harness.embedding_preregistration import (
    EmbeddingComparison,
    EmbeddingPreregistrationError,
    EmbeddingPreregistrationInputs,
    compare_embedding_arm,
    derive_embedding_preregistration_inputs,
    load_embedding_preregistration_inputs,
    serialize_embedding_preregistration_inputs,
)
from fused_memory.arm_harness.embedding_run import (
    EmbeddingRunRefused,
    Mem0Ranker,
    run_embedding_arm,
)
from fused_memory.arm_harness.embedding_run_manifest import (
    EmbeddingRunManifest,
    load_embedding_run_manifest,
)
from fused_memory.arm_harness.graph_copy import ReembedCensusError
from fused_memory.arm_harness.incumbent_cost import (
    INCUMBENT_COST_FILENAME,
    PRODUCTION_TELEMETRY_FILENAME,
    IncumbentCost,
    IncumbentCostError,
    TelemetryAccountingError,
    TelemetryRowError,
    TelemetryWindow,
    TelemetryWindowError,
    derive_incumbent_cost,
    select_llm_attempts,
    serialize_incumbent_cost,
    serialize_llm_attempts,
)
from fused_memory.arm_harness.instrument_checks import CheckResult
from fused_memory.arm_harness.llm_metrics import TokenAccountingError
from fused_memory.arm_harness.margins import MarginDerivationError, MarginEntry
from fused_memory.arm_harness.mem0_replica import (
    CollectionReader,
    Mem0Snapshot,
    ReplicaClient,
    ReplicaHit,
    load_snapshot,
    snapshot_collection,
    snapshot_sha,
    write_snapshot,
)
from fused_memory.arm_harness.metrics_record import (
    IndexConfiguration,
    MetricsRecord,
    load_metrics_records,
    write_metrics_record,
)
from fused_memory.arm_harness.normalization import VectorInvariantError
from fused_memory.arm_harness.preregistration import (
    PreregistrationError,
    PreregistrationInputs,
    derive_preregistration_inputs,
    load_preregistration_inputs,
    serialize_preregistration_inputs,
)
from fused_memory.arm_harness.probe_set import (
    TRANSCRIPT_QUERY_CAP,
    FrozenReference,
    Mem0KnownItem,
    Mem0SnapshotPin,
    ProbeSet,
    ProbeSetError,
    TranscriptPin,
    build_probe_set,
    probe_set_sha,
    read_probe_set,
    serialize_probe_set,
)
from fused_memory.arm_harness.replay import ArmGraph, ReplayJournal
from fused_memory.arm_harness.replay_types import (
    EpisodeOutcome,
    ReplayItem,
    ReplaySettings,
    default_replay_settings,
)
from fused_memory.arm_harness.retrieval import Rank
from fused_memory.arm_harness.run import (
    OUTCOMES_FILENAME,
    RUN_MANIFEST_FILENAME,
    PreRunCheckError,
    load_outcomes,
    require_pre_run_checks,
    run_llm_arm,
)
from fused_memory.arm_harness.run_manifest import RunManifest, load_run_manifest
from fused_memory.arm_harness.scratch_guard import (
    GuardCheckpoint,
    ScratchGuardError,
    require_scratch_name,
)
from fused_memory.arm_harness.scratch_indices import IndexDropError
from fused_memory.arm_harness.screening import (
    ScreeningVerdict,
    derive_screening_verdict,
    serialize_screening_verdict,
)
from fused_memory.arm_harness.screening_evidence import (
    ArmEvidencePaths,
    ScreeningEvidenceError,
    load_arm_evidence,
)
from fused_memory.arm_harness.slate import (
    EmbeddingSlateArm,
    SlateArm,
    embedding_candidate_spec,
    embedding_control_spec,
    load_embedding_slate,
    load_llm_slate,
)
from fused_memory.arm_harness.teardown import CollectionClient, teardown_arm
from fused_memory.arm_harness.topology import (
    IntegrityVerdict,
    Topology,
    check_reembed_integrity,
    read_topology,
    topology_hash,
)
from fused_memory.arm_harness.transcript_queries import iter_transcript_queries
from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.services.write_journal import WriteJournal

EXIT_OK = 0
EXIT_RUN_FAILED = 1
"""The run could not complete; agrees with ``shared.cli_boundary.EXIT_STDOUT_FAILED``."""
EXIT_REFUSED = 2
EXIT_CHECK_FAILED = 3
EXIT_ABORTED = 4
EXIT_SCRATCH_GUARD = 5
EXIT_CORPUS_INTEGRITY = 6

REPO_ROOT = Path(__file__).resolve().parents[3]
JOURNAL_DIRNAME = 'journal'
PARITY_DIRNAME = 'parity'
MIN_CONTROL_RUNS = 2

PROBE_SCRIPT_PATH = Path(__file__).resolve().parents[1] / 'memory_eval_retrieval_probe.py'
MEM0_PROJECT_ID = 'dark_factory'
EMBED_SETTINGS = EmbedSettings(batch_size=64, concurrency=4)
EMBEDDING_CONTROL_SCRATCH_GROUPS = {
    'incumbent-embed-a': 'evalmem_lme_emb_ctl_a',
    'incumbent-embed-b': 'evalmem_lme_emb_ctl_b',
}
REFERENCE_EPISODES_CYPHER = 'MATCH (e:Episodic) RETURN e.uuid, e.content'
CITED_EPISODES_CYPHER = (
    'MATCH ()-[r:RELATES_TO]->() UNWIND r.episodes AS uuid RETURN DISTINCT uuid'
)


class EpisodeSource(Protocol):
    async def fetch_population(self) -> list[build_corpus.EpisodeRecord]: ...


class ScratchGraph(ArmScratchGraph, Protocol):
    """The slice of a falkordb ``AsyncGraph`` an embedding run, the probes and teardown use."""

    async def delete(self) -> None: ...


class ScratchGraphClient(Protocol):
    async def list_graphs(self) -> list[str]: ...

    def select_graph(self, graph_id: str, /) -> ScratchGraph: ...


class QdrantClient(CollectionClient, CollectionReader, ReplicaClient, Protocol):
    """The slice of ``AsyncQdrantClient`` teardown, a Mem0 snapshot and an arm's replica use."""


class ArmEmbedderBuilder(Protocol):
    def __call__(
        self, spec: EmbeddingArmSpec, base_config: FusedMemoryConfig, /, *, settings: EmbedSettings
    ) -> ArmEmbedder: ...


@dataclasses.dataclass(frozen=True)
class HarnessDeps:
    """Every live resource a subcommand touches; ``build_live_deps`` builds the real ones."""

    base_config: FusedMemoryConfig
    episode_reader: Callable[[str], EpisodeSource]
    open_arm_backend: Callable[
        [LlmArmSpec, FusedMemoryConfig, ReplaySettings],
        AbstractAsyncContextManager[tuple[ArmGraph, ConformanceLedger]],
    ]
    open_journal: Callable[[Path], AbstractAsyncContextManager[ReplayJournal]]
    open_falkordb: Callable[[], AbstractAsyncContextManager[ScratchGraphClient]]
    open_qdrant: Callable[[], AbstractAsyncContextManager[QdrantClient]]
    open_embedding_arm_backend: Callable[
        [EmbeddingArmSpec, FusedMemoryConfig, EmbedderClient],
        AbstractAsyncContextManager[EmbeddingSearchBackend],
    ]
    build_arm_embedder: ArmEmbedderBuilder


def build_live_deps() -> HarnessDeps:
    """The real dependencies, configured by ``FusedMemoryConfig()`` (its ``CONFIG_PATH``)."""
    base = FusedMemoryConfig()
    falkor = base.graphiti.falkordb
    address = urlparse(falkor.uri)
    host, port = address.hostname or 'localhost', address.port or 6379

    def episode_reader(graph_name: str) -> EpisodeSource:
        return build_corpus.EpisodeReader(graph_name=graph_name, host=host, port=port)

    @contextlib.asynccontextmanager
    async def open_falkordb() -> AsyncIterator[ScratchGraphClient]:
        client = FalkorDB(host=host, port=port, password=falkor.password)
        try:
            yield client
        finally:
            await client.aclose()

    @contextlib.asynccontextmanager
    async def open_qdrant() -> AsyncIterator[QdrantClient]:
        client = AsyncQdrantClient(url=base.mem0.qdrant_url)
        try:
            yield client
        finally:
            await client.close()

    return HarnessDeps(
        base_config=base,
        episode_reader=episode_reader,
        open_arm_backend=open_arm_backend,
        open_journal=_open_journal,
        open_falkordb=open_falkordb,
        open_qdrant=open_qdrant,
        open_embedding_arm_backend=open_embedding_arm_backend,
        build_arm_embedder=build_arm_embedder,
    )


@contextlib.asynccontextmanager
async def _open_journal(path: Path) -> AsyncIterator[ReplayJournal]:
    journal = WriteJournal(path)
    await journal.initialize()
    try:
        yield journal
    finally:
        await journal.close()


DepsFactory = Callable[[], HarnessDeps]


class _Refusal(Exception):
    """The inputs cannot yield a meaningful result, so nothing was run or written."""

    def __init__(self, exit_code: int, message: str) -> None:
        self.exit_code = exit_code
        super().__init__(message)


# --- inputs ---------------------------------------------------------------------------


def _load_spec(path: Path) -> LlmArmSpec | EmbeddingArmSpec:
    """The arm spec at ``path``; a non-scratch graph raises ``ScratchGuardError`` raw."""
    try:
        return load_arm_spec(path)
    except (OSError, ValueError) as error:
        raise _Refusal(EXIT_REFUSED, f'{path} is not a valid arm spec: {error}') from error


def _load_llm_spec(path: Path, command: str) -> LlmArmSpec:
    spec = _load_spec(path)
    if not isinstance(spec, LlmArmSpec):
        raise _Refusal(EXIT_REFUSED, f'{command} drives LLM arms; {path} is a {spec.axis} arm')
    return spec


def _load_embedding_spec(path: Path, command: str) -> EmbeddingArmSpec:
    spec = _load_spec(path)
    if not isinstance(spec, EmbeddingArmSpec):
        raise _Refusal(
            EXIT_REFUSED, f'{command} drives embedding arms; {path} is a {spec.axis} arm'
        )
    return spec


def _read_manifest(path: Path, spec: LlmArmSpec) -> dict[str, Any]:
    """δ's manifest, refused unless its bytes hash to the spec's ``corpus_sha``."""
    try:
        manifest_bytes = path.read_bytes()
    except OSError as error:
        raise _Refusal(EXIT_REFUSED, f'cannot read manifest {path}: {error}') from error
    actual = corpus_sha(manifest_bytes)
    if actual != spec.corpus_sha:
        raise _Refusal(
            EXIT_CORPUS_INTEGRITY,
            f'arm {spec.arm_id!r} names corpus_sha {spec.corpus_sha}, '
            f'but {path} hashes to {actual}',
        )
    return json.loads(manifest_bytes)


def _corpus_graph(manifest: dict[str, Any]) -> str:
    criteria = manifest.get('criteria')
    graph = criteria.get('graph') if isinstance(criteria, dict) else None
    if not isinstance(graph, str) or not graph:
        raise _Refusal(EXIT_CORPUS_INTEGRITY, 'the manifest records no criteria.graph to read')
    return graph


async def _verified_items(live: HarnessDeps, manifest: dict[str, Any]) -> tuple[ReplayItem, ...]:
    """The manifest's replay items, after δ's own verdict on the fetched population is ok."""
    population = await live.episode_reader(_corpus_graph(manifest)).fetch_population()
    report = build_corpus.verify_manifest(manifest, population)
    if not report.ok:
        raise _Refusal(
            EXIT_CORPUS_INTEGRITY,
            f'build_corpus.verify_manifest: {report.status} — '
            f'{json.dumps(dataclasses.asdict(report), sort_keys=True)}',
        )
    return select_replay_items(manifest, population, content_hash=build_corpus.content_hash)


def _fresh_run_dir(run_dir: Path) -> Path:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise _Refusal(
            EXIT_REFUSED,
            f'{run_dir} already holds artifacts; pin a new $MEMORY_EVAL_RUN_STAMP or move it',
        )
    return run_dir


_Manifest = TypeVar('_Manifest', RunManifest, EmbeddingRunManifest)


def _load_run_with(
    run_dir: Path, load_manifest: Callable[[Path], _Manifest]
) -> tuple[_Manifest, tuple[MetricsRecord, ...]]:
    """A committed run's manifest and records; a spec naming a protected graph raises raw."""
    manifest_path = run_dir / RUN_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise _Refusal(
            EXIT_REFUSED,
            f'{run_dir} has no {RUN_MANIFEST_FILENAME}: an interrupted run, or not a run dir',
        )
    try:
        return load_manifest(manifest_path), load_metrics_records(run_dir)
    except (OSError, ValueError) as error:
        raise _Refusal(EXIT_REFUSED, f'{run_dir} is not a readable run: {error}') from error


def _load_run(run_dir: Path) -> tuple[RunManifest, tuple[MetricsRecord, ...]]:
    return _load_run_with(run_dir, load_run_manifest)


def _load_embedding_run(run_dir: Path) -> tuple[EmbeddingRunManifest, tuple[MetricsRecord, ...]]:
    return _load_run_with(run_dir, load_embedding_run_manifest)


def _load_control_run(run_dir: Path) -> tuple[RunManifest, tuple[MetricsRecord, ...]]:
    run, records = _load_run(run_dir)
    if not isinstance(run.spec, LlmArmSpec):
        raise _Refusal(
            EXIT_REFUSED,
            f'control runs are LLM arm runs; {run_dir} is a {run.spec.axis} arm',
        )
    return run, records


def _load_reference(path: Path | None) -> tuple[EpisodeOutcome, ...] | None:
    return None if path is None else _read_outcomes(path)


def _read_outcomes(path: Path) -> tuple[EpisodeOutcome, ...]:
    try:
        return load_outcomes(path)
    except (OSError, ValueError) as error:
        raise _Refusal(EXIT_REFUSED, f'{path} is not a readable outcomes file: {error}') from error


def _fresh_output(path: Path) -> Path:
    if path.exists():
        raise _Refusal(
            EXIT_REFUSED,
            f'{path} already exists; derived artifacts are never overwritten, so write '
            'to a new path',
        )
    return path


def _write_all_or_none(*outputs: tuple[Path, str]) -> None:
    """Write each output in turn; if one fails, remove those already written."""
    written: list[Path] = []
    try:
        for path, text in outputs:
            atomic_write_text(path, text, mkdir=True)
            written.append(path)
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise


def _scratch_graph(
    client: ScratchGraphClient, name: str, checkpoint: GuardCheckpoint
) -> ScratchGraph:
    return client.select_graph(require_scratch_name(name, checkpoint=checkpoint))


def _read_probe_set(path: Path) -> tuple[ProbeSet, str]:
    """The probe set at ``path`` and its sha, which every embedding arm's corpus_sha names."""
    try:
        return read_probe_set(path)
    except (OSError, ValueError) as error:
        raise _Refusal(EXIT_REFUSED, f'{path} is not a readable probe set: {error}') from error


def _arm_probe_set(path: Path, spec: EmbeddingArmSpec) -> ProbeSet:
    probe_set, sha = _read_probe_set(path)
    if sha != spec.corpus_sha:
        raise _Refusal(
            EXIT_CORPUS_INTEGRITY,
            f'arm {spec.arm_id!r} names corpus_sha {spec.corpus_sha}, but {path} hashes to {sha}',
        )
    return probe_set


def load_probe_module() -> ModuleType:
    """E1's ``memory_eval_retrieval_probe``, loaded by path once (scripts/ is no package).

    The shape of ``scripts/retro_stamp_topics.py::_load_probe_module``: a module already
    in ``sys.modules`` is reused, so every caller shares one set of its classes.
    """
    mod_name = 'memory_eval_retrieval_probe'
    cached = sys.modules.get(mod_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(mod_name, PROBE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load {PROBE_SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


# --- output ---------------------------------------------------------------------------


def _print_checks(results: Sequence[CheckResult]) -> None:
    for check in results:
        verdict = 'PASS' if check.passed else 'FAIL'
        print(f'{verdict} {check.check_id.value}: {check.detail}')


def _checks_exit_code(results: Sequence[CheckResult]) -> int:
    return EXIT_OK if all(check.passed for check in results) else EXIT_CHECK_FAILED


def _run_exit_code(manifest: RunManifest) -> int:
    if manifest.incomplete:
        return EXIT_ABORTED
    return _checks_exit_code(manifest.check_results)


# --- subcommands ----------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace, deps: DepsFactory) -> int:
    spec = _load_llm_spec(args.arm_spec, 'run')
    require_pre_run_checks(spec, args.repo_root)
    manifest = _read_manifest(args.manifest, spec)
    reference = _load_reference(args.reference_outcomes)
    run_dir = _fresh_run_dir(args.out_root / spec.arm_id / run_stamp())
    result = asyncio.run(_replay(args, spec, manifest, reference, run_dir, deps()))
    print(f'run: {run_dir}')
    _print_checks(result.check_results)
    if result.abort is not None:
        print(f'aborted: {result.abort.model_dump_json()}')
    return _run_exit_code(result)


async def _replay(
    args: argparse.Namespace,
    spec: LlmArmSpec,
    manifest: dict[str, Any],
    reference: Sequence[EpisodeOutcome] | None,
    run_dir: Path,
    live: HarnessDeps,
) -> RunManifest:
    items = (await _verified_items(live, manifest))[: args.limit]
    settings = default_replay_settings(
        live.base_config,
        concurrency=args.concurrency,
        index_configuration=args.index_configuration,
    )
    async with (
        live.open_journal(run_dir / JOURNAL_DIRNAME) as journal,
        live.open_arm_backend(spec, live.base_config, settings) as (graph, ledger),
    ):
        return await run_llm_arm(
            spec,
            items,
            graph=graph,
            conformance=ledger,
            journal=journal,
            settings=settings,
            run_dir=run_dir,
            repo_root=args.repo_root,
            reference=reference,
            base_config=live.base_config,
        )


def _cmd_smoke(args: argparse.Namespace, deps: DepsFactory) -> int:
    spec = _load_llm_spec(args.arm_spec, 'smoke')
    verdict = asyncio.run(smoke_endpoint(spec, deps().base_config))
    checks = (verdict.positive, verdict.negative_control)
    _print_checks(checks)
    return _checks_exit_code(checks)


def _cmd_index_check(args: argparse.Namespace, deps: DepsFactory) -> int:
    spec = _load_spec(args.arm_spec)
    result = asyncio.run(_index_check(deps(), spec.scratch_group_id, args.expect))
    _print_checks((result,))
    return _checks_exit_code((result,))


async def _index_check(live: HarnessDeps, scratch: str, expect: IndexConfiguration) -> CheckResult:
    async with live.open_falkordb() as client:
        graph = _scratch_graph(client, scratch, GuardCheckpoint.INDEX_PROBE)
        return await check_index_configuration(graph, scratch, expect)


def _cmd_integrity(args: argparse.Namespace, deps: DepsFactory) -> int:
    for name in (args.reference, args.candidate):
        require_scratch_name(name, checkpoint=GuardCheckpoint.TOPOLOGY_READ)
    verdict = asyncio.run(_integrity(deps(), args.reference, args.candidate))
    print(json.dumps(dataclasses.asdict(verdict), sort_keys=True, indent=2))
    return EXIT_OK if verdict.identical else EXIT_CHECK_FAILED


async def _integrity(live: HarnessDeps, reference: str, candidate: str) -> IntegrityVerdict:
    async with live.open_falkordb() as client:
        checkpoint = GuardCheckpoint.TOPOLOGY_READ
        reference_topology = await read_topology(
            _scratch_graph(client, reference, checkpoint), reference
        )
        candidate_topology = await read_topology(
            _scratch_graph(client, candidate, checkpoint), candidate
        )
    return check_reembed_integrity(reference_topology, candidate_topology)


def _cmd_parity_check(args: argparse.Namespace, deps: DepsFactory) -> int:
    run_a, records_a = _load_run(args.run_a)
    run_b, records_b = _load_run(args.run_b)
    deltas = client_class_parity(run_a, run_b, records_a, records_b)
    out_dir = args.run_a / PARITY_DIRNAME / run_b.spec.arm_id
    for record in deltas:
        print(write_metrics_record(record, out_dir))
    return EXIT_OK


def _cmd_control_check(args: argparse.Namespace, deps: DepsFactory) -> int:
    loaded = [_load_control_run(run_dir) for run_dir in args.run]
    reference = _load_reference(args.reference_outcomes)
    results = control_variance_check(
        [run for run, _ in loaded],
        {run.spec.arm_id: records for run, records in loaded},
        reference=reference,
    )
    _print_checks(results)
    return _checks_exit_code(results)


def _cmd_preregister(args: argparse.Namespace, deps: DepsFactory) -> int:
    out = _fresh_output(args.out)
    run_a, records_a = _load_control_run(args.run_a)
    run_b, records_b = _load_control_run(args.run_b)
    inputs = derive_preregistration_inputs(
        run_a,
        records_a,
        _read_outcomes(args.run_a / OUTCOMES_FILENAME),
        run_b,
        records_b,
        _read_outcomes(args.run_b / OUTCOMES_FILENAME),
    )
    atomic_write_text(out, serialize_preregistration_inputs(inputs), mkdir=True)
    _print_preregistration(inputs)
    print(f'wrote: {out}')
    return EXIT_OK


def _print_margins(margins: Sequence[MarginEntry]) -> None:
    for entry in margins:
        configuration = f' [{entry.index_configuration}]' if entry.index_configuration else ''
        print(
            f'margin {entry.metric_id}{configuration}: reference {entry.reference_value} '
            f'sigma {entry.sigma} ({entry.sigma_source}) floor {entry.floor} '
            f'margin {entry.margin} ({entry.direction})'
        )


def _print_preregistration(inputs: PreregistrationInputs) -> None:
    _print_margins(inputs.margins)
    envelope, profile = inputs.envelope, inputs.call_profile
    print(
        f'envelope: warm p95 under load < {envelope.p95_bound_ms} ms '
        f'(timeout {envelope.episode_timeout_s} s / headroom {envelope.headroom}); '
        f'incumbent p95 {inputs.incumbent_latency_p95_ms} ms, '
        f'max {inputs.incumbent_latency_max_ms} ms'
    )
    print(
        f'calls/episode: p50 {profile.calls_p50} p95 {profile.calls_p95} '
        f'max {profile.calls_max} over {profile.n_episodes} ok episodes'
    )


def _cmd_screen(args: argparse.Namespace, deps: DepsFactory) -> int:
    out = _fresh_output(args.out)
    slate = _read_slate(args.arms_manifest, load_llm_slate)
    evidence = {
        arm.arm_id: load_arm_evidence(ArmEvidencePaths(args.evidence_root, arm.arm_id), arm)
        for arm in slate
    }
    verdict = derive_screening_verdict(
        slate,
        evidence,
        _read_preregistration_inputs(args.preregistration_inputs),
        _read_outcomes(args.reference_outcomes),
    )
    atomic_write_text(out, serialize_screening_verdict(verdict), mkdir=True)
    _print_screening(verdict)
    print(f'wrote: {out}')
    return EXIT_OK


_Arm = TypeVar('_Arm', SlateArm, EmbeddingSlateArm)


def _read_slate(path: Path, load: Callable[[Path], tuple[_Arm, ...]]) -> tuple[_Arm, ...]:
    try:
        return load(path)
    except (OSError, ValueError) as error:
        raise _Refusal(EXIT_REFUSED, f'{path} is not a readable arms manifest: {error}') from error


def _read_preregistration_inputs(path: Path) -> PreregistrationInputs:
    try:
        return load_preregistration_inputs(path)
    except (OSError, ValueError) as error:
        raise _Refusal(
            EXIT_REFUSED, f'{path} is not a readable preregistration-inputs file: {error}'
        ) from error


def _print_screening(verdict: ScreeningVerdict) -> None:
    for arm in verdict.arms:
        for gate in arm.gates:
            unit = f' ({gate.unit.value})' if gate.unit else ''
            print(
                f'gate {arm.arm_id} {gate.gate.value}: {gate.verdict.value} value {gate.value} '
                f'bound {gate.bound} margin {gate.margin}{unit}'
            )
    print(f'survivors: {", ".join(verdict.survivors) or "none"}')
    print(f'outcome: {verdict.outcome.value}')


def _cmd_incumbent_cost(args: argparse.Namespace, deps: DepsFactory) -> int:
    telemetry_out = _fresh_output(args.out_dir / PRODUCTION_TELEMETRY_FILENAME)
    cost_out = _fresh_output(args.out_dir / INCUMBENT_COST_FILENAME)
    window = _select_window(args.telemetry, args.until)
    pricing_spec = _load_llm_spec(args.pricing_spec, 'incumbent-cost')
    control_records = [_load_control_run(run_dir)[1] for run_dir in args.control_run]
    cost = derive_incumbent_cost(
        window, pricing_spec=pricing_spec, control_records=control_records
    )
    _write_all_or_none(
        (telemetry_out, serialize_llm_attempts(window.attempts)),
        (cost_out, serialize_incumbent_cost(cost)),
    )
    _print_incumbent_cost(cost)
    print(f'wrote: {telemetry_out}')
    print(f'wrote: {cost_out}')
    return EXIT_OK


def _select_window(path: Path, until: datetime) -> TelemetryWindow:
    rows = _read_telemetry_rows(path)
    try:
        return select_llm_attempts(rows, until=until)
    except TelemetryRowError as error:
        raise _Refusal(EXIT_REFUSED, f'{path} holds a malformed telemetry row: {error}') from error


def _read_telemetry_rows(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError as error:
        raise _Refusal(EXIT_REFUSED, f'cannot read telemetry {path}: {error}') from error
    rows = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise _Refusal(EXIT_REFUSED, f'{path} line {number} is not JSON: {error}') from error
        if not isinstance(row, dict):
            raise _Refusal(EXIT_REFUSED, f'{path} line {number} is not a JSON object')
        rows.append(row)
    return rows


def _print_incumbent_cost(cost: IncumbentCost) -> None:
    production = cost.production
    print(
        f'window: {production.window_start.isoformat()} to {production.window_end.isoformat()} '
        f'({production.days} days)'
    )
    print(
        f'attempts {production.attempts} ({production.failed_attempts} failed), '
        f'llm_calls {production.llm_calls}, tokens/attempt {production.tokens_per_attempt}'
    )
    print(
        f'usd {production.usd} at {cost.pricing_arm_id} pricing: usd/day {production.usd_per_day}, '
        f'projected usd/30 days {production.projected_usd_per_30_days}'
    )
    for unit in cost.replay:
        print(
            f'replay {unit.arm_id}: usd/episode {unit.usd_per_episode} '
            f'tokens/episode {unit.tokens_per_episode} (n {unit.n})'
        )


def _cmd_topology(args: argparse.Namespace, deps: DepsFactory) -> int:
    require_scratch_name(args.graph, checkpoint=GuardCheckpoint.TOPOLOGY_READ)
    topology = asyncio.run(_topology(deps(), args.graph))
    summary = {
        'graph': args.graph,
        'node_count': len(topology.nodes),
        'edge_count': len(topology.edges),
        'topology_hash': topology_hash(*topology),
    }
    print(canonical_json_text(summary), end='')
    return EXIT_OK


async def _topology(live: HarnessDeps, graph_name: str) -> Topology:
    async with live.open_falkordb() as client:
        graph = _scratch_graph(client, graph_name, GuardCheckpoint.TOPOLOGY_READ)
        return await read_topology(graph, graph_name)


def _cmd_teardown(args: argparse.Namespace, deps: DepsFactory) -> int:
    spec = _load_spec(args.arm_spec)
    asyncio.run(_teardown(deps(), spec, collection=args.collection))
    print(f'deleted {spec.scratch_group_id!r}' + (' and its replica' if args.collection else ''))
    return EXIT_OK


async def _teardown(
    live: HarnessDeps, spec: LlmArmSpec | EmbeddingArmSpec, *, collection: bool
) -> None:
    async with live.open_falkordb() as falkor:
        if not collection:
            await teardown_arm(falkor, None, spec)
            return
        async with live.open_qdrant() as qdrant:
            await teardown_arm(falkor, qdrant, spec)


# --- the embedding axis ----------------------------------------------------------------


def _cmd_mem0_snapshot(args: argparse.Namespace, deps: DepsFactory) -> int:
    out = _fresh_output(args.out)
    snapshot = asyncio.run(_mem0_snapshot(deps(), args.collection))
    write_snapshot(out, snapshot)
    print(
        f'{len(snapshot.records)} records from {snapshot.source!r}, '
        f'{snapshot.excluded_empty} without text left out'
    )
    print(f'sha256: {snapshot_sha(out)}')
    print(f'wrote: {out}')
    return EXIT_OK


async def _mem0_snapshot(live: HarnessDeps, collection: str) -> Mem0Snapshot:
    async with live.open_qdrant() as qdrant:
        return await snapshot_collection(qdrant, collection)


def _cmd_probe_set(args: argparse.Namespace, deps: DepsFactory) -> int:
    out = _fresh_output(args.out)
    reference = _read_frozen_reference(args.reference_json)
    corpus = _reference_corpus_sha(args.control_a_outcomes, reference)
    replayed = {
        outcome.replay_episode_uuid
        for outcome in _read_outcomes(args.control_a_outcomes)
        if outcome.ok and outcome.replay_episode_uuid is not None
    }
    snapshot_pin, snapshot = _read_mem0_snapshot(args.mem0_snapshot)
    mem0_items = _mem0_known_items(args.registry, snapshot)
    transcript = _transcript_pin(args.transcript_corpus)
    read = asyncio.run(_read_reference(deps(), reference))
    if not read.frozen.passed:
        _print_checks((read.frozen,))
        return EXIT_CHECK_FAILED
    probe_set = build_probe_set(
        corpus_sha=corpus,
        reference=reference,
        episodes=read.episodes,
        cited_episode_uuids=read.cited,
        control_ok_episode_uuids=replayed,
        transcript=transcript,
        mem0_snapshot=snapshot_pin,
        mem0_known_items=mem0_items,
    )
    text = serialize_probe_set(probe_set)
    atomic_write_text(out, text, mkdir=True)
    _print_probe_set(probe_set)
    print(f'sha256: {probe_set_sha(text.encode())}')
    print(f'wrote: {out}')
    return EXIT_OK


def _read_frozen_reference(path: Path) -> FrozenReference:
    try:
        return FrozenReference.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as error:
        raise _Refusal(EXIT_REFUSED, f'{path} is not a readable frozen reference: {error}') from error


def _reference_corpus_sha(outcomes: Path, reference: FrozenReference) -> str:
    """δ's corpus_sha, from the run.json beside control A's outcomes, once that run built the reference."""
    run, _ = _load_control_run(outcomes.parent)
    if run.spec.scratch_group_id != reference.graph:
        raise _Refusal(
            EXIT_REFUSED,
            f'{outcomes.parent} replayed into {run.spec.scratch_group_id!r}, not the reference '
            f'graph {reference.graph!r}, so its outcomes are not the reference\'s episodes',
        )
    return run.spec.corpus_sha


def _read_mem0_snapshot(path: Path) -> tuple[Mem0SnapshotPin, Mem0Snapshot]:
    try:
        snapshot, sha = load_snapshot(path), snapshot_sha(path)
    except (OSError, ValueError) as error:
        raise _Refusal(EXIT_REFUSED, f'{path} is not a readable Mem0 snapshot: {error}') from error
    pin = Mem0SnapshotPin(
        source=snapshot.source,
        sha256=sha,
        point_count=len(snapshot.records),
        excluded_empty=snapshot.excluded_empty,
    )
    return pin, snapshot


def _mem0_known_items(registry_path: Path, snapshot: Mem0Snapshot) -> list[Mem0KnownItem]:
    """Each phrasing of E1's project topics whose canonical the snapshot holds, by hash or else id."""
    probe = load_probe_module()
    try:
        registry = probe.load_topic_registry(registry_path)
    except probe.RegistryError as error:
        raise _Refusal(EXIT_REFUSED, f'{registry_path}: {error}') from error
    held_hashes = {probe.content_key(record.data) for record in snapshot.records}
    held_ids = {record.id for record in snapshot.records}
    return [
        Mem0KnownItem(
            topic=entry.topic,
            phrasing=phrasing.text,
            held_out=phrasing.held_out,
            canonical_content_hash=entry.canonical.content_hash,
            canonical_last_known_id=entry.canonical.last_known_id,
        )
        for entry in registry.entries
        if entry.project_id == MEM0_PROJECT_ID
        and (
            entry.canonical.content_hash in held_hashes
            or entry.canonical.last_known_id in held_ids
        )
        for phrasing in entry.phrasings
    ]


def _transcript_pin(path: Path) -> TranscriptPin:
    try:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        queries = tuple(itertools.islice(iter_transcript_queries(path), TRANSCRIPT_QUERY_CAP))
    except (OSError, ValueError) as error:
        raise _Refusal(EXIT_REFUSED, f'{path} is not a readable transcript corpus: {error}') from error
    return TranscriptPin(path=path.name, sha256=sha, queries=queries)


@dataclasses.dataclass(frozen=True)
class _ReferenceRead:
    frozen: CheckResult
    episodes: tuple[tuple[str, str], ...]
    cited: frozenset[str]


async def _read_reference(live: HarnessDeps, reference: FrozenReference) -> _ReferenceRead:
    """The reference's live hash check, Episodic (uuid, content) pairs and cited uuids; ro_query only."""
    async with live.open_falkordb() as client:
        graph = _scratch_graph(client, reference.graph, GuardCheckpoint.TOPOLOGY_READ)
        topology = await read_topology(graph, reference.graph)
        episodes = await graph.ro_query(REFERENCE_EPISODES_CYPHER)
        cited = await graph.ro_query(CITED_EPISODES_CYPHER)
    return _ReferenceRead(
        frozen=frozen_reference_check(reference, topology_hash(*topology)),
        episodes=tuple((uuid, content) for uuid, content in episodes.result_set),
        cited=frozenset(uuid for (uuid,) in cited.result_set),
    )


def _print_probe_set(probe_set: ProbeSet) -> None:
    topics = {item.topic for item in probe_set.mem0_known_items}
    print(
        f'query words {probe_set.query_words}: {len(probe_set.known_items)} known items, '
        f'{probe_set.uncited_episodes} uncited episodes left out'
    )
    print(
        f'transcript queries {len(probe_set.transcript.queries)}; Mem0 known items '
        f'{len(probe_set.mem0_known_items)} over {len(topics)} topics'
    )


def _cmd_embed_specs(args: argparse.Namespace, deps: DepsFactory) -> int:
    slate = _read_slate(args.arms_manifest, load_embedding_slate)
    _, corpus = _read_probe_set(args.probe_set)
    arm_ids = [*EMBEDDING_CONTROL_SCRATCH_GROUPS, *(arm.arm_id for arm in slate)]
    if len(set(arm_ids)) != len(arm_ids):
        raise _Refusal(
            EXIT_REFUSED, f'{args.arms_manifest} reuses a control arm id: arms {arm_ids}'
        )
    outputs = {arm_id: _fresh_output(args.out_dir / f'{arm_id}.json') for arm_id in arm_ids}
    specs = _embedding_specs(args, slate, corpus, deps().base_config)
    _write_all_or_none(*((outputs[spec.arm_id], _spec_text(spec)) for spec in specs))
    for spec in specs:
        print(f'wrote: {outputs[spec.arm_id]}')
    return EXIT_OK


def _embedding_specs(
    args: argparse.Namespace,
    slate: Sequence[EmbeddingSlateArm],
    corpus: str,
    base_config: FusedMemoryConfig,
) -> tuple[EmbeddingArmSpec, ...]:
    """The incumbent controls, at the base config's embedder, then each slate arm."""
    incumbent = base_config.embedder
    try:
        controls = tuple(
            embedding_control_spec(
                arm_id,
                model_id=incumbent.model,
                embedding_dim=incumbent.dimensions,
                code_sha=args.code_sha,
                corpus_sha=corpus,
                scratch_group_id=scratch,
            )
            for arm_id, scratch in EMBEDDING_CONTROL_SCRATCH_GROUPS.items()
        )
        candidates = tuple(
            embedding_candidate_spec(
                arm,
                code_sha=args.code_sha,
                corpus_sha=corpus,
                preregistration_sha=args.preregistration_sha,
            )
            for arm in slate
        )
    except ValueError as error:
        raise _Refusal(EXIT_REFUSED, f'the embedding specs are not valid: {error}') from error
    return (*controls, *candidates)


def _spec_text(spec: EmbeddingArmSpec) -> str:
    return canonical_json_text(spec.model_dump(mode='json'))


def _cmd_embed_run(args: argparse.Namespace, deps: DepsFactory) -> int:
    spec = _load_embedding_spec(args.arm_spec, 'embed-run')
    require_pre_run_checks(spec, args.repo_root)
    probe_set = _arm_probe_set(args.probe_set, spec)
    run_dir = _fresh_run_dir(args.out_root / spec.arm_id / run_stamp())
    manifest = asyncio.run(_embed_run(args, spec, probe_set, run_dir, deps()))
    print(f'run: {run_dir}')
    _print_checks(manifest.check_results)
    return _checks_exit_code(manifest.check_results)


async def _embed_run(
    args: argparse.Namespace,
    spec: EmbeddingArmSpec,
    probe_set: ProbeSet,
    run_dir: Path,
    live: HarnessDeps,
) -> EmbeddingRunManifest:
    embedder = live.build_arm_embedder(spec, live.base_config, settings=EMBED_SETTINGS)
    async with (
        live.open_falkordb() as falkor,
        live.open_qdrant() as qdrant,
        live.open_embedding_arm_backend(
            spec, live.base_config, QueryEmbedder(embedder)
        ) as backend,
    ):
        return await run_embedding_arm(
            spec,
            args.probe_set,
            graph_client=falkor,
            backend=backend,
            qdrant=qdrant,
            embedder=embedder,
            mem0_snapshot=args.mem0_snapshot,
            mem0_project_id=MEM0_PROJECT_ID,
            rank_mem0=_mem0_ranker(probe_set.mem0_known_items),
            base_config=live.base_config,
            run_dir=run_dir,
            repo_root=args.repo_root,
        )


@dataclasses.dataclass(frozen=True)
class _PinnedTopic:
    """A registry topic as the probe set pinned it: the part of an entry canonical_hit matches on."""

    topic: str
    canonical: Any


def _mem0_ranker(items: Sequence[Mem0KnownItem]) -> Mem0Ranker:
    """E1's own canonical_hit, over the canonical the probe set pinned for each topic."""
    probe = load_probe_module()
    topics = {
        item.topic: _PinnedTopic(
            topic=item.topic,
            canonical=probe.Canonical(
                content_hash=item.canonical_content_hash,
                last_known_id=item.canonical_last_known_id,
            ),
        )
        for item in items
    }

    def rank(topic: str, hits: Sequence[ReplicaHit]) -> Rank:
        return probe.canonical_hit(list(hits), topics[topic], SEARCH_K).rank

    return rank


def _cmd_embed_preregister(args: argparse.Namespace, deps: DepsFactory) -> int:
    out = _fresh_output(args.out)
    run_a, records_a = _load_embedding_run(args.run_a)
    run_b, records_b = _load_embedding_run(args.run_b)
    inputs = derive_embedding_preregistration_inputs(run_a, records_a, run_b, records_b)
    atomic_write_text(out, serialize_embedding_preregistration_inputs(inputs), mkdir=True)
    _print_margins(inputs.margins)
    envelope = inputs.query_latency_envelope
    print(
        f'query-latency envelope: warm p95 under load < {envelope.p95_bound_ms} ms '
        f'(search timeout {envelope.search_timeout_s} s / headroom {envelope.headroom}); '
        f'incumbent p95 {inputs.incumbent_query_latency_p95_ms} ms'
    )
    print(f'wrote: {out}')
    return EXIT_OK


def _cmd_embed_compare(args: argparse.Namespace, deps: DepsFactory) -> int:
    inputs = _read_embedding_preregistration(args.preregistration)
    comparisons = [
        compare_embedding_arm(inputs, *_load_embedding_run(run_dir)) for run_dir in args.run
    ]
    print('| arm | metric | index configuration | reference | margin | candidate | admits |')
    print('|---|---|---|---|---|---|---|')
    for comparison in comparisons:
        _print_margin_rows(comparison)
    for comparison in comparisons:
        _print_comparison_verdict(comparison)
    return EXIT_OK


def _read_embedding_preregistration(path: Path) -> EmbeddingPreregistrationInputs:
    try:
        return load_embedding_preregistration_inputs(path)
    except (OSError, ValueError) as error:
        raise _Refusal(
            EXIT_REFUSED, f'{path} is not a readable embedding pre-registration: {error}'
        ) from error


def _print_margin_rows(comparison: EmbeddingComparison) -> None:
    for row in comparison.margins:
        configuration = row.index_configuration.value if row.index_configuration else '-'
        print(
            f'| {comparison.arm_id} | {row.metric_id} | {configuration} | {row.reference_value} '
            f'| {row.margin} | {row.candidate_value} | {row.admits} |'
        )


def _print_comparison_verdict(comparison: EmbeddingComparison) -> None:
    arm_id, envelope = comparison.arm_id, comparison.envelope
    print(
        f'envelope {arm_id}: p95 {envelope.candidate_p95_ms} ms, '
        f'{envelope.failed_queries} failed queries, bound {envelope.p95_bound_ms} ms, '
        f'admits {envelope.admits}'
    )
    for row in comparison.reported:
        print(f'reported {arm_id} {row.metric_id}: {row.value} (n {row.n})')
    print(f'non_inferior {arm_id}: {comparison.non_inferior}')


# --- parser and entry point -----------------------------------------------------------


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f'must be >= 1, got {value}')
    return value


def _aware_timestamp(text: str) -> datetime:
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f'not an ISO-8601 timestamp: {text!r}') from error
    if moment.tzinfo is None:
        raise argparse.ArgumentTypeError(f'{text!r} has no UTC offset; write it with +00:00')
    return moment


def _add_arm_spec(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--arm-spec', type=Path, required=True, help='ArmSpec JSON file')


def _build_parser() -> argparse.ArgumentParser:
    parser = LoudArgumentParser(
        description=(__doc__ or '').split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest='command', required=True)

    run = commands.add_parser('run', help='replay the corpus through one LLM arm')
    _add_arm_spec(run)
    run.add_argument('--manifest', type=Path, required=True, help="δ's corpus_manifest.json")
    run.add_argument('--out-root', type=Path, required=True)
    run.add_argument('--concurrency', type=_positive_int, required=True)
    run.add_argument(
        '--index-configuration',
        type=IndexConfiguration,
        choices=list(IndexConfiguration),
        required=True,
    )
    run.add_argument('--reference-outcomes', type=Path, help="the reference arm's outcomes.jsonl")
    run.add_argument('--repo-root', type=Path, default=REPO_ROOT, help='checkout code_sha names')
    run.add_argument('--limit', type=_positive_int, help='replay only the first N episodes')
    run.set_defaults(handler=_cmd_run)

    smoke = commands.add_parser('smoke', help='endpoint conformance smoke with negative control')
    _add_arm_spec(smoke)
    smoke.set_defaults(handler=_cmd_smoke)

    index_check = commands.add_parser('index-check', help='probe the scratch fulltext index')
    _add_arm_spec(index_check)
    index_check.add_argument(
        '--expect', type=IndexConfiguration, choices=list(IndexConfiguration), required=True
    )
    index_check.set_defaults(handler=_cmd_index_check)

    integrity = commands.add_parser('integrity', help='compare two scratch topologies')
    integrity.add_argument('--reference', required=True, help='reference scratch graph')
    integrity.add_argument('--candidate', required=True, help='re-embedded scratch graph')
    integrity.set_defaults(handler=_cmd_integrity)

    parity = commands.add_parser('parity-check', help='client-class parity deltas (a - b)')
    parity.add_argument('--run-a', type=Path, required=True)
    parity.add_argument('--run-b', type=Path, required=True)
    parity.set_defaults(handler=_cmd_parity_check)

    control = commands.add_parser('control-check', help='instrument checks over control runs')
    control.add_argument('--run', type=Path, action='append', required=True)
    control.add_argument('--reference-outcomes', type=Path)
    control.set_defaults(handler=_cmd_control_check)

    preregister = commands.add_parser(
        'preregister', help='derive the pre-registration inputs from two control runs'
    )
    preregister.add_argument('--run-a', type=Path, required=True, help='the reference control')
    preregister.add_argument(
        '--run-b', type=Path, required=True, help="the control run with A's reference outcomes"
    )
    preregister.add_argument(
        '--out', type=Path, required=True, help='the inputs JSON to write; never overwritten'
    )
    preregister.set_defaults(handler=_cmd_preregister)

    screen = commands.add_parser(
        'screen', help="derive η's screening verdict from a sweep's evidence root (offline)"
    )
    screen.add_argument('--evidence-root', type=Path, required=True, help="screen_slate's root")
    screen.add_argument(
        '--arms-manifest', type=Path, required=True, help='the arms.yaml the slate is read from'
    )
    screen.add_argument(
        '--preregistration-inputs', type=Path, required=True, help="ζ's committed inputs"
    )
    screen.add_argument(
        '--reference-outcomes', type=Path, required=True, help="control A's outcomes.jsonl"
    )
    screen.add_argument(
        '--out', type=Path, required=True, help='the verdict JSON to write; never overwritten'
    )
    screen.set_defaults(handler=_cmd_screen)

    incumbent = commands.add_parser(
        'incumbent-cost',
        help="the incumbent's measured LLM spend from production telemetry (offline)",
    )
    incumbent.add_argument(
        '--telemetry', type=Path, required=True, help="telemetry_query.py's JSONL dump"
    )
    incumbent.add_argument(
        '--until', type=_aware_timestamp, required=True, help='window end, ISO-8601 with offset'
    )
    incumbent.add_argument(
        '--pricing-spec', type=Path, required=True, help='the metered control spec to price by'
    )
    incumbent.add_argument(
        '--control-run', type=Path, action='append', required=True, help='a control run dir'
    )
    incumbent.add_argument(
        '--out-dir', type=Path, required=True, help='where both outputs go; never overwritten'
    )
    incumbent.set_defaults(handler=_cmd_incumbent_cost)

    topology = commands.add_parser('topology', help="a scratch graph's topology hash")
    topology.add_argument('--graph', required=True, help='scratch graph name')
    topology.set_defaults(handler=_cmd_topology)

    teardown = commands.add_parser('teardown', help="delete the arm's scratch graph")
    _add_arm_spec(teardown)
    teardown.add_argument('--collection', action='store_true', help='also its Qdrant replica')
    teardown.set_defaults(handler=_cmd_teardown)
    _add_embedding_commands(commands)
    return parser


def _add_embedding_commands(commands: Any) -> None:
    snapshot = commands.add_parser(
        'mem0-snapshot', help='a frozen, read-only snapshot of one Mem0 collection'
    )
    snapshot.add_argument('--collection', required=True, help='the live collection to read')
    snapshot.add_argument('--out', type=Path, required=True, help='JSONL to write; never overwritten')
    snapshot.set_defaults(handler=_cmd_mem0_snapshot)

    probe = commands.add_parser('probe-set', help='the embedding probe set (reads the reference)')
    probe.add_argument('--reference-json', type=Path, required=True, help="ζ's frozen-reference.json")
    probe.add_argument(
        '--control-a-outcomes',
        type=Path,
        required=True,
        help="control A's outcomes.jsonl; the run.json beside it names the corpus and graph",
    )
    probe.add_argument(
        '--transcript-corpus', type=Path, required=True, help='a transcript corpus-<STAMP>.jsonl'
    )
    probe.add_argument('--mem0-snapshot', type=Path, required=True, help="mem0-snapshot's JSONL")
    probe.add_argument('--registry', type=Path, required=True, help="E1's topic registry JSON")
    probe.add_argument('--out', type=Path, required=True, help='JSON to write; never overwritten')
    probe.set_defaults(handler=_cmd_probe_set)

    specs = commands.add_parser('embed-specs', help='the control and slate embedding arm specs')
    specs.add_argument('--arms-manifest', type=Path, required=True, help='the arms.yaml slate')
    specs.add_argument('--probe-set', type=Path, required=True, help='its sha is every corpus_sha')
    specs.add_argument('--code-sha', required=True, help='the clean commit every arm runs at')
    specs.add_argument('--preregistration-sha', required=True, help="the candidates' prereg commit")
    specs.add_argument('--out-dir', type=Path, required=True, help='<arm_id>.json each; never overwritten')
    specs.set_defaults(handler=_cmd_embed_specs)

    embed_run = commands.add_parser('embed-run', help='one embedding arm, end to end')
    _add_arm_spec(embed_run)
    embed_run.add_argument('--probe-set', type=Path, required=True)
    embed_run.add_argument('--mem0-snapshot', type=Path, required=True, help='the pinned snapshot')
    embed_run.add_argument('--out-root', type=Path, required=True)
    embed_run.add_argument('--repo-root', type=Path, default=REPO_ROOT, help='checkout code_sha names')
    embed_run.set_defaults(handler=_cmd_embed_run)

    preregister = commands.add_parser(
        'embed-preregister', help='the embedding pre-registration inputs of a control pair'
    )
    preregister.add_argument('--run-a', type=Path, required=True, help='a control run dir')
    preregister.add_argument('--run-b', type=Path, required=True, help='the other control run dir')
    preregister.add_argument('--out', type=Path, required=True, help='JSON to write; never overwritten')
    preregister.set_defaults(handler=_cmd_embed_preregister)

    compare = commands.add_parser(
        'embed-compare', help='candidate runs against the embedding pre-registration (offline)'
    )
    compare.add_argument('--preregistration', type=Path, required=True, help="embed-preregister's JSON")
    compare.add_argument('--run', type=Path, action='append', required=True, help='a candidate run')
    compare.set_defaults(handler=_cmd_embed_compare)


def _report(exit_code: int, message: str) -> int:
    print(f'error: {message}', file=sys.stderr)
    return exit_code


def _named(error: Exception) -> str:
    return f'{type(error).__name__}: {error}'


def main(argv: list[str] | None = None, *, deps: DepsFactory = build_live_deps) -> int:
    """Run one subcommand; ``deps`` is called only once its inputs have been validated."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == 'control-check' and len(args.run) < MIN_CONTROL_RUNS:
        parser.error(f'control-check needs at least {MIN_CONTROL_RUNS} --run directories')
    try:
        return args.handler(args, deps)
    except _Refusal as refusal:
        return _report(refusal.exit_code, str(refusal))
    except ScratchGuardError as error:
        return _report(EXIT_SCRATCH_GUARD, _named(error))
    except (
        PreRunCheckError,
        RunComparabilityError,
        PreregistrationError,
        MarginDerivationError,
        TokenAccountingError,
        ScreeningEvidenceError,
        TelemetryWindowError,
        TelemetryAccountingError,
        IncumbentCostError,
        EmbeddingRunRefused,
        ProbeSetError,
        EmbeddingPreregistrationError,
    ) as error:
        return _report(EXIT_REFUSED, _named(error))
    except EmbeddingRunCheckFailed as error:
        return _report(EXIT_CHECK_FAILED, _named(error))
    except (CorpusIntegrityError, build_corpus.CorpusBuildError) as error:
        return _report(EXIT_CORPUS_INTEGRITY, _named(error))
    except (
        IndexBuildError,
        IndexDropError,
        ReembedCensusError,
        VectorInvariantError,
        ProbeCleanupError,
        RedisError,
    ) as error:
        return _report(EXIT_RUN_FAILED, _named(error))


if __name__ == '__main__':
    sys.exit(run_cli(main))
