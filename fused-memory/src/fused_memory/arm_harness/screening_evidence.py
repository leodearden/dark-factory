"""What one arm's η screening left on disk, typed and validated: layout, command records, α's VRAM reading, the tap's calls and the pinned run."""

import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, NoReturn, TypeVar
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field
from shared.memory_eval_metrics import canonical_json_text
from shared.safe_io import atomic_write_text

from fused_memory.arm_harness.arm_spec import ArmId, LlmArmSpec, load_arm_spec
from fused_memory.arm_harness.frozen_model import FrozenModel
from fused_memory.arm_harness.llm_metrics import graph_sameness_details, nearest_rank
from fused_memory.arm_harness.metrics_record import (
    IndexConfiguration,
    LlmMetricId,
    MetricsRecord,
    UtcDatetime,
    load_metrics_records,
)
from fused_memory.arm_harness.replay_types import EpisodeOutcome
from fused_memory.arm_harness.run import OUTCOMES_FILENAME, RUN_MANIFEST_FILENAME, load_outcomes
from fused_memory.arm_harness.run_manifest import RunManifest, load_run_manifest
from fused_memory.arm_harness.slate import SlateArm
from fused_memory.arm_harness.usage_tap import CallRecord, load_call_records

LMS_CTL_EXIT_CARD_HELD = 5
"""Mirrors scripts/local-model-serving/lms_ctl.py::EXIT_CARD_HELD."""
HEALTH_REPORT_SCHEMA_VERSION = 6
"""Mirrors scripts/local-model-serving/lms_healthcheck.py::REPORT_SCHEMA_VERSION."""
CLEAN_READING = 'CLEAN'
LENGTH_FINISH_REASON = 'length'
_P50, _P95 = 0.50, 0.95

_Loaded = TypeVar('_Loaded')


class ScreeningEvidenceError(ValueError):
    """Screening evidence that measured nothing valid: it is refused, never scored."""


class ScreeningStage(StrEnum):
    RELEASE = 'release'
    START = 'start'
    WAIT_READY = 'wait-ready'
    SMOKE = 'smoke'
    RUN = 'run'
    HEALTHCHECK = 'healthcheck'
    STOP = 'stop'
    TEARDOWN = 'teardown'


class CommandRecord(FrozenModel):
    stage: ScreeningStage
    argv: tuple[str, ...]
    exit_code: int
    started_at: UtcDatetime
    finished_at: UtcDatetime
    output_tail: str


class TapBinding(FrozenModel):
    listen_url: str
    upstream_url: str


class ArmCommands(FrozenModel):
    arm_id: ArmId
    tap: TapBinding
    records: tuple[CommandRecord, ...]

    def record_of(self, stage: ScreeningStage) -> CommandRecord | None:
        return next((record for record in self.records if record.stage is stage), None)


class ScreeningRunShape(FrozenModel):
    limit: int = Field(gt=0)
    concurrency: int = Field(ge=1)
    index_configuration: IndexConfiguration


SCREENING_RUN_SHAPE = ScreeningRunShape(
    limit=20, concurrency=3, index_configuration=IndexConfiguration.WITH_INDICES
)


@dataclass(frozen=True)
class ArmEvidencePaths:
    root: Path
    arm_id: str

    @property
    def spec(self) -> Path:
        return self.root / 'specs' / f'{self.arm_id}.json'

    @property
    def arm_dir(self) -> Path:
        return self.root / 'arms' / self.arm_id

    @property
    def commands(self) -> Path:
        return self.arm_dir / 'commands.json'

    @property
    def health(self) -> Path:
        return self.arm_dir / 'health.json'

    @property
    def smoke_calls(self) -> Path:
        return self.arm_dir / 'smoke-calls.jsonl'

    @property
    def calls(self) -> Path:
        return self.arm_dir / 'calls.jsonl'

    @property
    def runs(self) -> Path:
        return self.root / 'runs' / self.arm_id


class _ForeignRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra='ignore')


class BaselineConsumer(_ForeignRecord):
    pid: int
    process_name: str
    used_mib: int


class HealthRow(_ForeignRecord):
    arm_id: str
    reasoning: str | None
    verdict: str
    top_level_entities_named: int | None = None


class VramReading(_ForeignRecord):
    verdict: Literal['PASS', 'FAIL']
    reason: str
    arm_footprint_mib: int
    budget_mib: int
    pollution: str
    baseline_consumers: tuple[BaselineConsumer, ...]


class _HealthReport(_ForeignRecord):
    schema_version: int
    arms: tuple[HealthRow, ...]
    vram: VramReading


class VramEvidence(FrozenModel):
    row: HealthRow
    vram: VramReading


@dataclass(frozen=True)
class ArmEvidence:
    arm: SlateArm
    spec: LlmArmSpec
    commands: ArmCommands
    served: bool
    vram: VramEvidence | None
    calls: tuple[CallRecord, ...]
    run: RunManifest | None
    records: tuple[MetricsRecord, ...]
    outcomes: tuple[EpisodeOutcome, ...]

    @property
    def own_model_calls(self) -> tuple[CallRecord, ...]:
        """The tap's calls for this arm's own served model, not the embedder's or anyone else's."""
        return tuple(call for call in self.calls if call.request_model == self.spec.model_id)

    def metric_value(self, metric_id: LlmMetricId) -> float | None:
        return next(
            (r.metric.value for r in self.records if r.metric.metric_id == metric_id), None
        )


def _refuse(where: object, invariant: str, offending: object) -> NoReturn:
    raise ScreeningEvidenceError(f'{where}: {invariant}; got {offending!r}')


RELEASE_RECORD_FILENAME = 'release.json'
"""The sweep's one leading stop-all, under the evidence root beside specs/, arms/ and runs/."""


def write_release_record(root: Path, record: CommandRecord) -> None:
    text = canonical_json_text(record.model_dump(mode='json'))
    atomic_write_text(root / RELEASE_RECORD_FILENAME, text, mkdir=True)


def write_arm_commands(path: Path | str, commands: ArmCommands) -> None:
    atomic_write_text(Path(path), canonical_json_text(commands.model_dump(mode='json')), mkdir=True)


def load_arm_commands(path: Path | str) -> ArmCommands:
    return _loaded(Path(path), lambda p: ArmCommands.model_validate_json(p.read_text()))


def write_screening_spec(path: Path | str, spec: LlmArmSpec) -> None:
    atomic_write_text(Path(path), canonical_json_text(spec.model_dump(mode='json')), mkdir=True)


def require_file(path: Path) -> Path:
    if not path.is_file():
        _refuse(path, 'this evidence file must exist', 'no such file')
    return path


def _guarded(where: Path, load: Callable[[], _Loaded]) -> _Loaded:
    try:
        return load()
    except (ValueError, OSError) as error:
        _refuse(where, 'this evidence must be readable', f'{type(error).__name__}: {error}')


def _loaded(path: Path, load: Callable[[Path], _Loaded]) -> _Loaded:
    require_file(path)
    return _guarded(path, lambda: load(path))


# --- α's VRAM reading ----------------------------------------------------------------


def require_health_schema(path: Path, version: object) -> None:
    if version != HEALTH_REPORT_SCHEMA_VERSION:
        _refuse(
            path,
            f'schema_version must be {HEALTH_REPORT_SCHEMA_VERSION} '
            '(scripts/local-model-serving/lms_healthcheck.py::REPORT_SCHEMA_VERSION)',
            version,
        )


def require_one_row_for(path: Path, rows: Sequence[HealthRow], arm_id: str) -> HealthRow:
    if len(rows) != 1:
        _refuse(path, 'an --arm report carries exactly one arm row', len(rows))
    if rows[0].arm_id != arm_id:
        _refuse(path, f'the arm row arm_id must be {arm_id!r}', rows[0].arm_id)
    return rows[0]


def require_clean_reading(path: Path, vram: VramReading) -> None:
    if vram.pollution != CLEAN_READING:
        _refuse(
            path,
            f'a polluted VRAM reading is void, not a result: pollution must be {CLEAN_READING}',
            vram.pollution,
        )


def require_resident_baseline(path: Path, vram: VramReading) -> None:
    if not vram.baseline_consumers:
        _refuse(
            path,
            'the VRAM gate is defined beside resident whisper-writer, so baseline_consumers '
            'must name a compute consumer',
            list(vram.baseline_consumers),
        )


def load_vram_evidence(path: Path | str, arm_id: str) -> VramEvidence:
    report_path = Path(path)
    payload = _loaded(report_path, lambda p: json.loads(p.read_text()))
    if not isinstance(payload, dict):
        _refuse(report_path, 'an lms_healthcheck report is a JSON object', type(payload).__name__)
    require_health_schema(report_path, payload.get('schema_version'))
    report = _guarded(report_path, lambda: _HealthReport.model_validate(payload))
    row = require_one_row_for(report_path, report.arms, arm_id)
    require_clean_reading(report_path, report.vram)
    require_resident_baseline(report_path, report.vram)
    return VramEvidence(row=row, vram=report.vram)


# --- one arm's evidence --------------------------------------------------------------


def require_commands_for(path: Path, commands: ArmCommands, arm: SlateArm) -> None:
    if commands.arm_id != arm.arm_id:
        _refuse(path, f'the commands must be arm {arm.arm_id!r}\'s', commands.arm_id)


def require_recorded(path: Path, commands: ArmCommands, stage: ScreeningStage) -> CommandRecord:
    record = commands.record_of(stage)
    if record is None:
        _refuse(
            path,
            f'arm {commands.arm_id!r} must record a {stage.value!r} stage',
            [r.stage.value for r in commands.records],
        )
    return record


def require_card_not_held(path: Path, arm_id: str, start: CommandRecord) -> None:
    if start.exit_code == LMS_CTL_EXIT_CARD_HELD:
        _refuse(
            path,
            f'arm {arm_id!r}: lms_ctl start exited EXIT_CARD_HELD '
            '(scripts/local-model-serving/lms_ctl.py::EXIT_CARD_HELD), so another process held '
            'the card and nothing was measured',
            start.exit_code,
        )


def require_candidate_spec_for(path: Path, spec: LlmArmSpec, arm: SlateArm) -> None:
    if spec.arm_id != arm.arm_id:
        _refuse(path, f'the spec must be slate arm {arm.arm_id!r}\'s', spec.arm_id)
    if spec.arm_role != 'candidate':
        _refuse(path, f'arm {arm.arm_id!r} is screened as a candidate', spec.arm_role)


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f'{parts.scheme}://{parts.netloc}'


def require_tap_to_arm(path: Path, tap: TapBinding, arm: SlateArm, spec: LlmArmSpec) -> None:
    upstream_port = urlsplit(tap.upstream_url).port
    if upstream_port != arm.port:
        invariant = f'the tap must forward to arm {arm.arm_id!r} on port {arm.port}'
        _refuse(path, invariant, upstream_port)
    spec_origin = _origin(spec.serving.base_url)
    if tap.listen_url != spec_origin:
        _refuse(path, f'the tap must listen where the spec points, {spec_origin}', tap.listen_url)


def require_screened_reasoning(path: Path, row: HealthRow, arm: SlateArm) -> None:
    if row.reasoning != arm.reasoning:
        _refuse(
            path,
            f'arm {arm.arm_id!r} must be screened in the manifest reasoning mode {arm.reasoning!r}',
            row.reasoning,
        )


def require_max_tokens_premise(
    path: Path, calls: Sequence[CallRecord], spec: LlmArmSpec
) -> None:
    offending = sorted(
        {
            str(call.request_max_tokens)
            for call in calls
            if call.request_model == spec.model_id
            and call.request_max_tokens != spec.params.max_tokens
        }
    )
    if offending:
        _refuse(
            path,
            f'every {spec.model_id!r} call must request max_tokens {spec.params.max_tokens}, '
            'the completion budget the context gate reserves',
            offending,
        )


def require_single_run(runs: Path) -> Path:
    stamps = sorted(p.name for p in runs.iterdir() if p.is_dir()) if runs.is_dir() else []
    if len(stamps) != 1:
        _refuse(
            runs,
            f'a served arm leaves exactly one run stamp dir, found {len(stamps)}',
            stamps,
        )
    return runs / stamps[0]


def require_run_of(run_dir: Path, run: RunManifest, spec: LlmArmSpec) -> None:
    if run.spec != spec:
        _refuse(
            run_dir,
            f'arm {spec.arm_id!r}: the run must be of the committed spec',
            run.spec.model_dump(mode='json'),
        )


def require_screening_shape(run_dir: Path, run: RunManifest, arm_id: str) -> None:
    actual = {
        'limit': len(run.episode_ids),
        'concurrency': run.settings_summary.concurrency,
        'index_configuration': run.settings_summary.index_configuration,
    }
    expected = SCREENING_RUN_SHAPE.model_dump()
    differing = {key: value for key, value in actual.items() if value != expected[key]}
    if differing:
        shape = SCREENING_RUN_SHAPE.model_dump(mode='json')
        _refuse(run_dir, f'arm {arm_id!r} must run the screening shape {shape}', differing)


def _served(commands: ArmCommands, start: CommandRecord) -> bool:
    wait_ready = commands.record_of(ScreeningStage.WAIT_READY)
    return start.exit_code == 0 and wait_ready is not None and wait_ready.exit_code == 0


def _load_llm_spec(path: Path) -> LlmArmSpec:
    spec = _loaded(path, load_arm_spec)
    if not isinstance(spec, LlmArmSpec):
        _refuse(path, 'a screened arm is an LLM arm', spec.axis)
    return spec


def load_arm_evidence(paths: ArmEvidencePaths, arm: SlateArm) -> ArmEvidence:
    commands = load_arm_commands(paths.commands)
    require_commands_for(paths.commands, commands, arm)
    start = require_recorded(paths.commands, commands, ScreeningStage.START)
    require_card_not_held(paths.commands, arm.arm_id, start)
    spec = _load_llm_spec(paths.spec)
    require_candidate_spec_for(paths.spec, spec, arm)
    require_tap_to_arm(paths.commands, commands.tap, arm, spec)
    if not _served(commands, start):
        return ArmEvidence(
            arm=arm, spec=spec, commands=commands, served=False,
            vram=None, calls=(), run=None, records=(), outcomes=(),
        )
    require_recorded(paths.commands, commands, ScreeningStage.SMOKE)
    vram = load_vram_evidence(paths.health, arm.arm_id)
    require_screened_reasoning(paths.health, vram.row, arm)
    calls = _loaded(paths.calls, load_call_records)
    require_max_tokens_premise(paths.calls, calls, spec)
    run_dir = require_single_run(paths.runs)
    run = _loaded(run_dir / RUN_MANIFEST_FILENAME, load_run_manifest)
    require_run_of(run_dir, run, spec)
    require_screening_shape(run_dir, run, arm.arm_id)
    return ArmEvidence(
        arm=arm,
        spec=spec,
        commands=commands,
        served=True,
        vram=vram,
        calls=calls,
        run=run,
        records=_guarded(run_dir, lambda: load_metrics_records(run_dir)),
        outcomes=_loaded(run_dir / OUTCOMES_FILENAME, load_outcomes),
    )


# --- reported, never gating ---------------------------------------------------------

class ErrorClassCount(FrozenModel):
    error_class: str
    count: int = Field(ge=1)


class ReportedEvidence(FrozenModel):
    """Measured beside the gates and never gating: survival reads the four gates only."""

    episode_failure_rate: float | None
    conformance_rate: float | None
    latency_p50_ms: float | None
    tokens_per_episode: float | None
    error_classes: tuple[ErrorClassCount, ...]
    incomplete: bool | None
    abort_item_count: int | None
    calls_per_ok_episode_p50: int | None
    calls_per_ok_episode_max: int | None
    own_model_calls: int
    call_duration_p50_ms: float | None
    call_duration_p95_ms: float | None
    length_truncations: int
    rejected_calls: int
    other_model_calls: int
    health_verdict: str | None
    top_level_entities_named: int | None
    graph_sameness_mean_entity_jaccard: float | None
    graph_sameness_n: int


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _ranked(values: Sequence[float], p: float) -> float | None:
    return nearest_rank(sorted(values), p) if values else None


def reported_evidence(
    evidence: ArmEvidence, reference_outcomes: Sequence[EpisodeOutcome]
) -> ReportedEvidence:
    own = evidence.own_model_calls
    durations = [call.duration_ms for call in own]
    calls_per_ok = sorted(
        outcome.tokens.llm_calls
        for outcome in evidence.outcomes
        if outcome.ok and outcome.tokens is not None
    )
    errors = Counter(o.error_class for o in evidence.outcomes if o.error_class is not None)
    sameness = graph_sameness_details(evidence.outcomes, reference_outcomes).episodes
    run, row = evidence.run, evidence.vram.row if evidence.vram else None
    return ReportedEvidence(
        episode_failure_rate=evidence.metric_value(LlmMetricId.EPISODE_FAILURE_RATE),
        conformance_rate=evidence.metric_value(LlmMetricId.CONFORMANCE_RATE),
        latency_p50_ms=evidence.metric_value(LlmMetricId.EPISODE_LATENCY_P50),
        tokens_per_episode=evidence.metric_value(LlmMetricId.TOKENS_PER_EPISODE),
        error_classes=tuple(
            ErrorClassCount(error_class=name, count=count) for name, count in sorted(errors.items())
        ),
        incomplete=run.incomplete if run else None,
        abort_item_count=len(run.abort.item_ids) if run and run.abort else None,
        calls_per_ok_episode_p50=nearest_rank(calls_per_ok, _P50) if calls_per_ok else None,
        calls_per_ok_episode_max=calls_per_ok[-1] if calls_per_ok else None,
        own_model_calls=len(own),
        call_duration_p50_ms=_ranked(durations, _P50),
        call_duration_p95_ms=_ranked(durations, _P95),
        length_truncations=sum(1 for c in own if c.finish_reason == LENGTH_FINISH_REASON),
        rejected_calls=sum(1 for call in own if not call.succeeded),
        other_model_calls=len(evidence.calls) - len(own),
        health_verdict=row.verdict if row else None,
        top_level_entities_named=row.top_level_entities_named if row else None,
        graph_sameness_mean_entity_jaccard=_mean(tuple(e.entity_jaccard for e in sameness)),
        graph_sameness_n=len(sameness),
    )
