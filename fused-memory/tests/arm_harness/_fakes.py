"""Shared arm-harness test doubles: valid spec builders and public-Protocol fakes."""

import asyncio
import dataclasses
import json
import subprocess
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fused_memory.arm_harness.arm_spec import EmbeddingArmSpec, LlmArmSpec, LlmParams
from fused_memory.arm_harness.conformance import ConformanceCounts
from fused_memory.arm_harness.instrument_checks import PREREGISTRATION_DOC_PATH
from fused_memory.arm_harness.llm_metrics import llm_axis_records
from fused_memory.arm_harness.metrics_record import MetricsRecord, write_metrics_record
from fused_memory.arm_harness.preregistration import PreregistrationInputs, latency_envelope
from fused_memory.arm_harness.replay_types import ArmAbort, ArmRunResult, EpisodeOutcome
from fused_memory.arm_harness.run import RUN_MANIFEST_FILENAME, write_outcomes
from fused_memory.arm_harness.run_manifest import RunManifest, serialize_run_manifest
from fused_memory.arm_harness.screening_evidence import (
    SCREENING_RUN_SHAPE,
    ArmCommands,
    ArmEvidence,
    ArmEvidencePaths,
    CommandRecord,
    ScreeningStage,
    TapBinding,
    VramEvidence,
    write_arm_commands,
    write_screening_spec,
)
from fused_memory.arm_harness.slate import SlateArm, arm_endpoint, candidate_spec
from fused_memory.arm_harness.usage_tap import CallRecord
from fused_memory.backends.llm_token_usage import (
    AttributingTokenUsageTracker,
    LlmTokenUsage,
    TokenMeasurement,
    measure_llm_tokens,
)

CODE_SHA = 'a' * 40
CORPUS_SHA = 'b' * 64
PREREG_SHA = 'c' * 40
UNREACHABLE_BASE_URL = 'http://127.0.0.1:9/v1'
PROTECTED_GRAPHS = (
    'dark_factory',
    'reify',
    'know_live',
    'solar_challenge_platform',
    'autopilot_video',
    'pump_web_ui',
    'my_solar_challenge',
    'probe_e1_master',
    '_probe',
)
"""The live graphs of plans/local-memory-models-eval-prd.md §Hazards, as test inputs only."""


def llm_spec(*, base_url: str = UNREACHABLE_BASE_URL, **overrides) -> LlmArmSpec:
    """A local-stack candidate LLM arm; override any field by keyword."""
    data = {
        'arm_id': 'qwen3-8b-vllm',
        'axis': 'llm',
        'model_id': 'qwen3-8b',
        'serving': {'stack': 'vllm', 'base_url': base_url},
        'client_class': 'openai_generic',
        'structured_output_mode': 'json_schema',
        'params': {'temperature': 0.0, 'max_tokens': 4096},
        'pricing': None,
        'code_sha': CODE_SHA,
        'corpus_sha': CORPUS_SHA,
        'preregistration_sha': PREREG_SHA,
        'scratch_group_id': 'evalmem_qwen3_8b',
        'arm_role': 'candidate',
    }
    return LlmArmSpec.model_validate(data | overrides)


def incumbent_control_spec(**overrides) -> LlmArmSpec:
    """The metered incumbent control arm (stack 'openai', priced, no prereg sha, default client)."""
    data = {
        'arm_id': 'incumbent-ctrl-a',
        'model_id': 'gpt-4.1-mini',
        'client_class': 'openai',
        'serving': {'stack': 'openai', 'base_url': 'https://api.openai.com/v1'},
        'pricing': {'usd_per_mtok_input': 0.4, 'usd_per_mtok_output': 1.6},
        'preregistration_sha': None,
        'scratch_group_id': 'evalmem_ctrl_a',
        'arm_role': 'control',
    }
    return llm_spec(**(data | overrides))


def embedding_spec(*, base_url: str = UNREACHABLE_BASE_URL, **overrides) -> EmbeddingArmSpec:
    """A local-stack candidate embedding arm; override any field by keyword."""
    data = {
        'arm_id': 'bge-m3',
        'axis': 'embedding',
        'model_id': 'BAAI/bge-m3',
        'serving': {'stack': 'tei', 'base_url': base_url},
        'embedding_dim': 1024,
        'code_sha': CODE_SHA,
        'corpus_sha': CORPUS_SHA,
        'preregistration_sha': PREREG_SHA,
        'scratch_group_id': 'evalmem_bge_m3',
        'arm_role': 'candidate',
    }
    return EmbeddingArmSpec.model_validate(data | overrides)


STARTED_AT = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
FINISHED_AT = datetime(2026, 10, 5, 13, 30, tzinfo=UTC)


def run_manifest_for(spec: LlmArmSpec | EmbeddingArmSpec, **overrides) -> RunManifest:
    """A complete run of ``spec`` over three episodes; override any field by keyword."""
    data = {
        'schema_version': 1,
        'spec': spec,
        'settings_summary': {
            'concurrency': 4,
            'index_configuration': 'with-indices',
            'episode_timeout_s': 120.0,
        },
        'effective_embedder': {'model': 'text-embedding-3-small', 'dimensions': 1536},
        'graphiti_max_coroutines': 5,
        'graphiti_semaphore_limit': 20,
        'episode_ids': ('e1', 'e2', 'e3'),
        'incomplete': False,
        'abort': None,
        'check_results': (),
        'started_at': STARTED_AT,
        'finished_at': FINISHED_AT,
    }
    return RunManifest.model_validate(data | overrides)


def fake_add_result(
    episode_uuid: str,
    entity_names: tuple[str, ...] = (),
    edges: tuple[tuple[str, str, str], ...] = (),
) -> SimpleNamespace:
    """A minimal AddEpisodeResults-shaped value: episode.uuid, nodes, edges (by node name)."""
    nodes = [SimpleNamespace(uuid=f'node-{name}', name=name) for name in entity_names]
    entity_edges = [
        SimpleNamespace(
            uuid=f'edge-{source}-{relation}-{target}',
            source_node_uuid=f'node-{source}',
            target_node_uuid=f'node-{target}',
            name=relation,
            episodes=[episode_uuid],
        )
        for source, relation, target in edges
    ]
    return SimpleNamespace(
        episode=SimpleNamespace(uuid=episode_uuid), nodes=nodes, edges=entity_edges
    )


Behaviour = Callable[[dict[str, Any]], Awaitable[Any]]


async def succeed(call: dict[str, Any]) -> Any:
    return fake_add_result(f'replay-{call["name"]}', entity_names=('alice', 'bob'))


async def fail(call: dict[str, Any]) -> Any:
    raise RuntimeError(f'extraction failed for {call["name"]}')


async def hang(call: dict[str, Any]) -> Any:
    await asyncio.Event().wait()


def slow(behaviour: Behaviour, seconds: float) -> Behaviour:
    async def delayed(call: dict[str, Any]) -> Any:
        await asyncio.sleep(seconds)
        return await behaviour(call)

    return delayed


class FakeArmGraph:
    """A public-``ArmGraph`` double: per-episode-name behaviours, recorded calls, overlap.

    ``usage`` is recorded on an attributing tracker inside each add_episode, so the
    real ``measure_llm_tokens`` window credits it to that episode.
    """

    def __init__(
        self,
        behaviours: Mapping[str, Behaviour] | None = None,
        *,
        default: Behaviour = succeed,
        usage: tuple[int, int] | None = (30, 10),
        llm_client: Any = None,
        search_results: Callable[[str], list[Any]] | None = None,
    ) -> None:
        self._behaviours = dict(behaviours or {})
        self._default = default
        self._usage = usage
        self.llm_client = llm_client or SimpleNamespace(
            token_tracker=AttributingTokenUsageTracker()
        )
        self._search_results = search_results or (lambda query: [])
        self.add_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.events: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def add_episode(self, **call: Any) -> Any:
        self.add_calls.append(call)
        self.events.append('add_episode')
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            result = await self._behaviours.get(call['name'], self._default)(call)
            if self._usage is not None:
                self.llm_client.token_tracker.record('extract_nodes', *self._usage)
            return result
        finally:
            self.in_flight -= 1

    async def search(
        self, query: str, group_ids: list[str] | None = None, num_results: int = 10
    ) -> list[Any]:
        self.search_calls.append(
            {'query': query, 'group_ids': group_ids, 'num_results': num_results}
        )
        self.events.append('search')
        return self._search_results(query)[:num_results]

    def token_probe(self) -> AbstractAsyncContextManager[TokenMeasurement]:
        return measure_llm_tokens(self.llm_client)


class RecordingJournal:
    """A journal double recording each log call's keyword arguments, in order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def log_write_op(self, **kwargs: Any) -> None:
        self.calls.append(('log_write_op', kwargs))

    async def log_backend_op(self, **kwargs: Any) -> None:
        self.calls.append(('log_backend_op', kwargs))


def git(root: Path, *args: str) -> str:
    """Run real git in ``root`` with a throwaway identity; returns stripped stdout."""
    completed = subprocess.run(
        ['git', '-C', str(root), '-c', 'user.name=T', '-c', 'user.email=t@e.example',
         '-c', 'commit.gpgsign=false', *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@dataclasses.dataclass(frozen=True)
class PreregRepo:
    root: Path
    without_prereg: str
    with_prereg: str


def make_prereg_repo(root: Path) -> PreregRepo:
    """A real two-commit repo at ``root``: the second (HEAD) commit adds the preregistration doc."""
    root.mkdir(parents=True)
    git(root, 'init', '-q', '-b', 'main')
    (root / 'seed.txt').write_text('seed\n')
    git(root, 'add', '-A')
    git(root, 'commit', '-q', '--no-verify', '-m', 'seed')
    without_prereg = git(root, 'rev-parse', 'HEAD')
    doc = root / PREREGISTRATION_DOC_PATH
    doc.parent.mkdir(parents=True)
    doc.write_text('# preregistration\n')
    git(root, 'add', '-A')
    git(root, 'commit', '-q', '--no-verify', '-m', 'prereg')
    with_prereg = git(root, 'rev-parse', 'HEAD')
    return PreregRepo(root=root, without_prereg=without_prereg, with_prereg=with_prereg)


# --- η screening evidence ------------------------------------------------------------

TAP_LISTEN_URL = 'http://127.0.0.1:8418'
TAP_BASE_URL = f'{TAP_LISTEN_URL}/v1'
SCREENING_PARAMS = LlmParams(temperature=0.0, max_tokens=4096)
EVIDENCE_STAMP = '20261007T120000Z'
WHISPER_WRITER = {'pid': 9024, 'process_name': 'python', 'used_mib': 4050}
SCREENING_EPISODE_IDS = tuple(f'e{i:02d}' for i in range(1, SCREENING_RUN_SHAPE.limit + 1))


def slate_arm(**overrides) -> SlateArm:
    """qwen3.5-9b as arms.yaml declares it; override any field by keyword."""
    data = {
        'arm_id': 'qwen3.5-9b',
        'stack': 'vllm',
        'port': 8410,
        'served_model_name': 'qwen3.5-9b',
        'structured_output_mode': 'json_schema',
        'quant': 'awq',
        'reasoning': 'on',
        'max_model_len': 32768,
    }
    return SlateArm.model_validate(data | overrides)


def screening_spec(arm: SlateArm) -> LlmArmSpec:
    return candidate_spec(
        arm,
        base_url=TAP_BASE_URL,
        code_sha=CODE_SHA,
        corpus_sha=CORPUS_SHA,
        preregistration_sha=PREREG_SHA,
        params=SCREENING_PARAMS,
    )


def health_report(
    arm_id: str = 'qwen3.5-9b',
    *,
    reasoning: str | None = 'on',
    vram_verdict: str = 'PASS',
    arm_footprint_mib: int = 16637,
    budget_mib: int = 18019,
    pollution: str = 'CLEAN',
    baseline_consumers: tuple[dict[str, Any], ...] = (WHISPER_WRITER,),
    top_level_entities_named: int | None = 4,
    schema_version: int = 6,
) -> dict[str, Any]:
    """α's ``lms_healthcheck --arm X --output`` report, trimmed to what screening reads plus noise."""
    row = {
        'arm_id': arm_id,
        'axis': 'llm',
        'stack': 'vllm',
        'verdict': 'PASS',
        'reason': 'ok',
        'latency_ms': 2900.0,
        'reasoning': reasoning,
        'top_level_entities_named': top_level_entities_named,
    }
    return {
        'schema_version': schema_version,
        'measured_at': '2026-10-07T12:30:00+00:00',
        'arms': [row],
        'vram': {
            'total_mib': 24576,
            'budget_mib': budget_mib,
            'arm_footprint_mib': arm_footprint_mib,
            'verdict': vram_verdict,
            'reason': f'the arm took {arm_footprint_mib} MiB of a {budget_mib} MiB budget',
            'baseline_consumers': list(baseline_consumers),
            'pollution': pollution,
            'pollution_reason': '',
        },
        'overall': 'PASS',
        'latency_caveat': 'single sample',
    }


def command_record(
    stage: ScreeningStage, *, exit_code: int = 0, output_tail: str = '', argv: tuple[str, ...] = ()
) -> CommandRecord:
    return CommandRecord(
        stage=stage,
        argv=argv or ('uv', 'run', stage.value),
        exit_code=exit_code,
        started_at=STARTED_AT,
        finished_at=FINISHED_AT,
        output_tail=output_tail,
    )


def call(
    *,
    model: str = 'qwen3.5-9b',
    status: int = 200,
    prompt_tokens: int | None = 1500,
    max_tokens: int | None = 4096,
    completion_tokens: int | None = 200,
    finish_reason: str | None = 'stop',
    duration_ms: float = 2000.0,
    error_excerpt: str | None = None,
    outlived_session: bool = False,
) -> CallRecord:
    succeeded = 200 <= status < 300
    return CallRecord(
        started_at=STARTED_AT,
        duration_ms=duration_ms,
        method='POST',
        path='/v1/chat/completions',
        status=status,
        request_model=model,
        request_max_tokens=max_tokens,
        request_response_format='json_schema',
        prompt_tokens=prompt_tokens if succeeded else None,
        completion_tokens=completion_tokens if succeeded else None,
        finish_reason=finish_reason if succeeded else None,
        error_excerpt=None if succeeded else (error_excerpt or f'error {status}'),
        outlived_session=outlived_session,
    )


def write_call_records(path: Path, calls: tuple[CallRecord, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(record.model_dump_json() + '\n' for record in calls))


def episode_outcome(
    episode_id: str,
    *,
    ok: bool = True,
    duration_ms: float = 10000.0,
    entity_names: tuple[str, ...] = ('alice', 'bob'),
    llm_calls: int = 9,
) -> EpisodeOutcome:
    return EpisodeOutcome(
        episode_id=episode_id,
        ok=ok,
        error_class=None if ok else 'EpisodeOverBudget',
        duration_ms=duration_ms,
        tokens=LlmTokenUsage(input_tokens=1000, output_tokens=200, llm_calls=llm_calls)
        if ok
        else None,
        replay_episode_uuid=f'replay-{episode_id}' if ok else None,
        entity_names=entity_names if ok else (),
        edge_triples=(),
    )


def screening_outcomes(duration_ms: float = 10000.0) -> tuple[EpisodeOutcome, ...]:
    return tuple(
        episode_outcome(episode_id, duration_ms=duration_ms + index)
        for index, episode_id in enumerate(SCREENING_EPISODE_IDS)
    )


def screening_run_manifest(spec: LlmArmSpec, /, **overrides) -> RunManifest:
    """A screening-shaped run of ``spec``; override any field, ``spec`` included, by keyword."""
    data = {
        'settings_summary': {
            'concurrency': SCREENING_RUN_SHAPE.concurrency,
            'index_configuration': SCREENING_RUN_SHAPE.index_configuration.value,
            'episode_timeout_s': 120.0,
        },
        'episode_ids': SCREENING_EPISODE_IDS,
    } | overrides
    return run_manifest_for(data.pop('spec', spec), **data)


def screening_records(
    spec: LlmArmSpec,
    outcomes: tuple[EpisodeOutcome, ...],
    *,
    abort: ArmAbort | None = None,
    schema_valid: int = 180,
) -> tuple[MetricsRecord, ...]:
    """The records the pinned harness derives from ``outcomes``, through its own metric code."""
    result = ArmRunResult(arm_id=spec.arm_id, outcomes=outcomes, cancelled_ids=(), abort=abort)
    counts = ConformanceCounts(
        schema_valid=schema_valid, transport_errors=0, invalid_by_error_class={}
    )
    return llm_axis_records(
        spec, result, counts, reference=None, retrieval_ranks=None, measured_at=FINISHED_AT
    )


def write_screening_run(
    run_dir: Path, run: RunManifest, outcomes: tuple[EpisodeOutcome, ...]
) -> None:
    """A pinned harness run's artifacts, through the harness's own writers."""
    assert isinstance(run.spec, LlmArmSpec)
    run_dir.mkdir(parents=True, exist_ok=True)
    for record in screening_records(run.spec, outcomes, abort=run.abort):
        write_metrics_record(record, run_dir)
    write_outcomes(run_dir, outcomes)
    (run_dir / RUN_MANIFEST_FILENAME).write_text(serialize_run_manifest(run))


def _stage_records(
    start_exit: int, wait_ready_exit: int, smoke_exit: int, run_exit: int, smoke_tail: str = ''
) -> tuple[CommandRecord, ...]:
    if start_exit != 0:
        stages = ((ScreeningStage.START, start_exit),)
    elif wait_ready_exit != 0:
        stages = ((ScreeningStage.START, 0), (ScreeningStage.WAIT_READY, wait_ready_exit))
    else:
        stages = (
            (ScreeningStage.START, 0),
            (ScreeningStage.WAIT_READY, 0),
            (ScreeningStage.SMOKE, smoke_exit),
            (ScreeningStage.RUN, run_exit),
            (ScreeningStage.HEALTHCHECK, 0),
        )
    tail = ((ScreeningStage.STOP, 0), (ScreeningStage.TEARDOWN, 0))
    return tuple(
        command_record(
            stage,
            exit_code=code,
            output_tail=smoke_tail
            if stage is ScreeningStage.SMOKE and smoke_tail
            else f'{stage.value} said {code}',
        )
        for stage, code in stages + tail
    )


def _tap_for(arm: SlateArm) -> TapBinding:
    return TapBinding(listen_url=TAP_LISTEN_URL, upstream_url=arm_endpoint(arm))


def vram_evidence(report: dict[str, Any]) -> VramEvidence:
    return VramEvidence.model_validate({'row': report['arms'][0], 'vram': report['vram']})


def default_calls(arm: SlateArm) -> tuple[CallRecord, ...]:
    return tuple(call(model=arm.served_model_name, prompt_tokens=1000 + i) for i in range(5))


def arm_evidence(
    arm: SlateArm | None = None,
    *,
    start_exit: int = 0,
    wait_ready_exit: int = 0,
    smoke_exit: int = 0,
    smoke_tail: str = '',
    spec: LlmArmSpec | None = None,
    health: dict[str, Any] | None = None,
    calls: tuple[CallRecord, ...] | None = None,
    outcomes: tuple[EpisodeOutcome, ...] | None = None,
    records: tuple[MetricsRecord, ...] | None = None,
    abort: ArmAbort | None = None,
) -> ArmEvidence:
    """One arm's loaded screening evidence, built in memory; any piece can be swapped."""
    arm = arm or slate_arm()
    spec = spec or screening_spec(arm)
    commands = ArmCommands(
        arm_id=arm.arm_id,
        tap=_tap_for(arm),
        records=_stage_records(start_exit, wait_ready_exit, smoke_exit, 0, smoke_tail),
    )
    if start_exit != 0 or wait_ready_exit != 0:
        return ArmEvidence(
            arm=arm, spec=spec, commands=commands, served=False,
            vram=None, calls=(), run=None, records=(), outcomes=(),
        )
    outcomes = outcomes if outcomes is not None else screening_outcomes()
    return ArmEvidence(
        arm=arm,
        spec=spec,
        commands=commands,
        served=True,
        vram=vram_evidence(health or health_report(arm.arm_id, reasoning=arm.reasoning)),
        calls=calls if calls is not None else default_calls(arm),
        run=screening_run_manifest(spec, incomplete=abort is not None, abort=abort),
        records=records
        if records is not None
        else screening_records(spec, outcomes, abort=abort),
        outcomes=outcomes,
    )


def preregistration_inputs(**overrides) -> PreregistrationInputs:
    """ζ's committed inputs in shape: the 120 s envelope, the incumbent's p95 inside it."""
    data = {
        'schema_version': 1,
        'control_arm_ids': ('incumbent-generic-a', 'incumbent-generic-b'),
        'code_sha': CODE_SHA,
        'corpus_sha': CORPUS_SHA,
        'margins': (),
        'envelope': latency_envelope(120.0),
        'call_profile': {
            'n_episodes': 400,
            'calls_p50': 9,
            'calls_p95': 14,
            'calls_max': 39,
            'input_tokens_per_call': 1700.0,
            'output_tokens_per_call': 200.0,
        },
        'incumbent_latency_p95_ms': 42828.0,
        'incumbent_latency_max_ms': 92826.0,
    }
    return PreregistrationInputs.model_validate(data | overrides)


def write_arm_evidence(
    root: Path,
    arm: SlateArm,
    *,
    start_exit: int = 0,
    wait_ready_exit: int = 0,
    smoke_exit: int = 0,
    run_exit: int = 0,
    records: tuple[CommandRecord, ...] | None = None,
    tap: TapBinding | None = None,
    spec: LlmArmSpec | None = None,
    health: dict[str, Any] | None = None,
    write_health: bool = True,
    smoke_calls: tuple[CallRecord, ...] | None = None,
    calls: tuple[CallRecord, ...] | None = None,
    outcomes: tuple[EpisodeOutcome, ...] | None = None,
    run_overrides: Mapping[str, Any] | None = None,
    stamps: tuple[str, ...] = (EVIDENCE_STAMP,),
) -> ArmEvidencePaths:
    """One arm's screening evidence tree as the driver leaves it; any piece can be swapped."""
    paths = ArmEvidencePaths(root=root, arm_id=arm.arm_id)
    committed_spec = screening_spec(arm)
    write_screening_spec(paths.spec, spec or committed_spec)
    write_arm_commands(
        paths.commands,
        ArmCommands(
            arm_id=arm.arm_id,
            tap=tap or _tap_for(arm),
            records=records
            if records is not None
            else _stage_records(start_exit, wait_ready_exit, smoke_exit, run_exit),
        ),
    )
    if start_exit != 0 or wait_ready_exit != 0:
        return paths
    write_call_records(
        paths.smoke_calls,
        smoke_calls if smoke_calls is not None else (call(model=arm.served_model_name),),
    )
    write_call_records(paths.calls, calls if calls is not None else default_calls(arm))
    if write_health:
        report = health or health_report(arm.arm_id, reasoning=arm.reasoning)
        paths.health.write_text(json.dumps(report))
    run = screening_run_manifest(committed_spec, **dict(run_overrides or {}))
    for stamp in stamps:
        write_screening_run(
            paths.runs / stamp, run, outcomes if outcomes is not None else screening_outcomes()
        )
    return paths
