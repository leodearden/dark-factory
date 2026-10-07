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
  teardown       delete the arm's scratch graph and, with --collection, its replica

Exit codes, run-directory layout and the live check:
README.md §"Arm-runner harness (task ε)" beside this script.
"""

import argparse
import asyncio
import contextlib
import dataclasses
import json
import sys
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import build_corpus
from falkordb.asyncio import FalkorDB
from qdrant_client import AsyncQdrantClient
from redis.exceptions import RedisError
from shared.cli_boundary import LoudArgumentParser, run_cli
from shared.memory_eval_metrics import run_stamp

from fused_memory.arm_harness.arm_backend import IndexBuildError, open_arm_backend
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
from fused_memory.arm_harness.instrument_checks import CheckResult
from fused_memory.arm_harness.metrics_record import (
    IndexConfiguration,
    MetricsRecord,
    load_metrics_records,
    write_metrics_record,
)
from fused_memory.arm_harness.replay import ArmGraph, ReplayJournal
from fused_memory.arm_harness.replay_types import (
    EpisodeOutcome,
    ReplayItem,
    ReplaySettings,
    default_replay_settings,
)
from fused_memory.arm_harness.run import (
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
from fused_memory.arm_harness.teardown import CollectionClient, teardown_arm
from fused_memory.arm_harness.topology import (
    IntegrityVerdict,
    check_reembed_integrity,
    read_topology,
)
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


class EpisodeSource(Protocol):
    async def fetch_population(self) -> list[build_corpus.EpisodeRecord]: ...


class ScratchGraph(Protocol):
    """The slice of a falkordb ``AsyncGraph`` the index probe, topology read and teardown use."""

    async def query(self, q: str, params: dict[str, Any] | None = None) -> Any: ...

    async def ro_query(self, q: str, params: dict[str, Any] | None = None) -> Any: ...

    async def delete(self) -> None: ...


class ScratchGraphClient(Protocol):
    def select_graph(self, graph_id: str, /) -> ScratchGraph: ...


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
    open_qdrant: Callable[[], AbstractAsyncContextManager[CollectionClient]]


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
    async def open_qdrant() -> AsyncIterator[CollectionClient]:
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


def _load_run(run_dir: Path) -> tuple[RunManifest, tuple[MetricsRecord, ...]]:
    """A committed run's manifest and records; a spec naming a protected graph raises raw."""
    manifest_path = run_dir / RUN_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise _Refusal(
            EXIT_REFUSED,
            f'{run_dir} has no {RUN_MANIFEST_FILENAME}: an interrupted run, or not a run dir',
        )
    try:
        return load_run_manifest(manifest_path), load_metrics_records(run_dir)
    except (OSError, ValueError) as error:
        raise _Refusal(EXIT_REFUSED, f'{run_dir} is not a readable run: {error}') from error


def _load_control_run(run_dir: Path) -> tuple[RunManifest, tuple[MetricsRecord, ...]]:
    run, records = _load_run(run_dir)
    if not isinstance(run.spec, LlmArmSpec):
        raise _Refusal(
            EXIT_REFUSED,
            f'control-check compares LLM control runs; {run_dir} is a {run.spec.axis} arm',
        )
    return run, records


def _load_reference(path: Path | None) -> tuple[EpisodeOutcome, ...] | None:
    if path is None:
        return None
    try:
        return load_outcomes(path)
    except (OSError, ValueError) as error:
        raise _Refusal(EXIT_REFUSED, f'{path} is not a readable outcomes file: {error}') from error


def _scratch_graph(
    client: ScratchGraphClient, name: str, checkpoint: GuardCheckpoint
) -> ScratchGraph:
    return client.select_graph(require_scratch_name(name, checkpoint=checkpoint))


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


# --- parser and entry point -----------------------------------------------------------


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f'must be >= 1, got {value}')
    return value


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

    teardown = commands.add_parser('teardown', help="delete the arm's scratch graph")
    _add_arm_spec(teardown)
    teardown.add_argument('--collection', action='store_true', help='also its Qdrant replica')
    teardown.set_defaults(handler=_cmd_teardown)
    return parser


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
    except (PreRunCheckError, RunComparabilityError) as error:
        return _report(EXIT_REFUSED, _named(error))
    except (CorpusIntegrityError, build_corpus.CorpusBuildError) as error:
        return _report(EXIT_CORPUS_INTEGRITY, _named(error))
    except (IndexBuildError, ProbeCleanupError, RedisError) as error:
        return _report(EXIT_RUN_FAILED, _named(error))


if __name__ == '__main__':
    sys.exit(run_cli(main))
