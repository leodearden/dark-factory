"""One LLM arm run, end to end: checks, replay, probe, metrics and the run directory's artifacts.

``run_llm_arm`` is a sequence of named phases. Everything is validated before the
first artifact is written, so a refused run leaves its directory empty. ``run.json``
is written last and is the commit marker: a run directory without one is an
interrupted run. A failed post-run instrument check is recorded in ``run.json``,
not raised; the caller decides what that run is worth.
"""

import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from graphiti_core import helpers as graphiti_helpers
from shared.memory_eval_metrics import canonical_json_text
from shared.safe_io import atomic_write_text

from fused_memory.arm_harness.arm_spec import EmbeddingArmSpec, LlmArmSpec
from fused_memory.arm_harness.conformance import ConformanceLedger
from fused_memory.arm_harness.instrument_checks import (
    CheckResult,
    InstrumentCheckId,
    check_code_sha_matches_checkout,
    check_failed,
    check_preregistration_sha,
    check_reference_nonempty,
    check_token_cost_accounting,
)
from fused_memory.arm_harness.llm_metrics import (
    GRAPH_SAMENESS_DETAILS_FILENAME,
    GraphSamenessDetails,
    TokenAccountingError,
    graph_sameness_details,
    llm_axis_records,
)
from fused_memory.arm_harness.metrics_record import MetricsRecord, write_metrics_record
from fused_memory.arm_harness.replay import ArmGraph, ReplayJournal, replay_arm
from fused_memory.arm_harness.replay_types import (
    ArmRunResult,
    EpisodeOutcome,
    ReplayItem,
    ReplaySettings,
)
from fused_memory.arm_harness.retrieval import Rank, probe_retrieval_utility
from fused_memory.arm_harness.run_manifest import (
    RUN_MANIFEST_SCHEMA_VERSION,
    EffectiveEmbedder,
    RunManifest,
    SettingsSummary,
    serialize_run_manifest,
)
from fused_memory.config.schema import FusedMemoryConfig

RUN_MANIFEST_FILENAME = 'run.json'
OUTCOMES_FILENAME = 'outcomes.jsonl'
ABORT_FILENAME = 'abort.json'


class PreRunCheckError(RuntimeError):
    """A pre-run instrument check failed, so the arm was refused before any replay."""

    def __init__(self, arm_id: str, check_results: tuple[CheckResult, ...]) -> None:
        self.check_results = check_results
        failed = '; '.join(
            f'{check.check_id}: {check.detail}' for check in check_results if not check.passed
        )
        super().__init__(f'arm {arm_id!r} refused before replay: {failed}')


async def run_llm_arm(
    spec: LlmArmSpec,
    items: Sequence[ReplayItem],
    *,
    graph: ArmGraph,
    conformance: ConformanceLedger,
    journal: ReplayJournal,
    settings: ReplaySettings,
    run_dir: Path,
    repo_root: Path,
    reference: Sequence[EpisodeOutcome] | None,
    base_config: FusedMemoryConfig,
) -> RunManifest:
    pre_run = await asyncio.to_thread(require_pre_run_checks, spec, repo_root)
    started_at = datetime.now(UTC)
    result = await replay_arm(spec, items, graph=graph, journal=journal, settings=settings)
    ranks = await _retrieval_ranks(graph, spec, result, items)
    finished_at = datetime.now(UTC)
    records, token_failure = _measure(spec, result, conformance, reference, ranks, finished_at)
    manifest = RunManifest(
        schema_version=RUN_MANIFEST_SCHEMA_VERSION,
        spec=spec,
        settings_summary=_settings_summary(settings),
        effective_embedder=_effective_embedder(base_config),
        graphiti_max_coroutines=base_config.queue.graphiti_max_coroutines,
        graphiti_semaphore_limit=graphiti_helpers.SEMAPHORE_LIMIT,
        episode_ids=tuple(item.episode_id for item in items),
        incomplete=result.incomplete,
        abort=result.abort,
        check_results=pre_run + _post_run_checks(spec, records, token_failure, reference),
        started_at=started_at,
        finished_at=finished_at,
    )
    details = graph_sameness_details(result.outcomes, reference) if reference is not None else None
    _write_artifacts(run_dir, manifest, result.outcomes, records, details)
    return manifest


def require_pre_run_checks(
    spec: LlmArmSpec | EmbeddingArmSpec, repo_root: Path
) -> tuple[CheckResult, ...]:
    """The passed pre-run checks, or ``PreRunCheckError``; callable before any resource opens."""
    checks = (
        check_code_sha_matches_checkout(spec, repo_root),
        check_preregistration_sha(spec, repo_root),
    )
    if not all(check.passed for check in checks):
        raise PreRunCheckError(spec.arm_id, checks)
    return checks


async def _retrieval_ranks(
    graph: ArmGraph, spec: LlmArmSpec, result: ArmRunResult, items: Sequence[ReplayItem]
) -> tuple[Rank, ...] | None:
    """Probed only after every write has finished, and only when some episode landed."""
    if not any(outcome.ok for outcome in result.outcomes):
        return None
    return await probe_retrieval_utility(graph, spec, result.outcomes, items=items)


def _measure(
    spec: LlmArmSpec,
    result: ArmRunResult,
    conformance: ConformanceLedger,
    reference: Sequence[EpisodeOutcome] | None,
    ranks: tuple[Rank, ...] | None,
    measured_at: datetime,
) -> tuple[tuple[MetricsRecord, ...], CheckResult | None]:
    """The arm's records, or none plus a failed check when its token usage is unknown."""
    try:
        records = llm_axis_records(
            spec,
            result,
            conformance.snapshot(),
            reference=reference,
            retrieval_ranks=ranks,
            measured_at=measured_at,
        )
    except TokenAccountingError as error:
        detail = f'{type(error).__name__}: {error}'
        failed = check_failed(InstrumentCheckId.TOKEN_COST_ACCOUNTING, detail, error.episode_ids)
        return (), failed
    return records, None


def _post_run_checks(
    spec: LlmArmSpec,
    records: Sequence[MetricsRecord],
    token_failure: CheckResult | None,
    reference: Sequence[EpisodeOutcome] | None,
) -> tuple[CheckResult, ...]:
    token_check = token_failure or check_token_cost_accounting(spec, records)
    if reference is None:
        return (token_check,)
    return (token_check, check_reference_nonempty(reference))


def _settings_summary(settings: ReplaySettings) -> SettingsSummary:
    return SettingsSummary(
        concurrency=settings.concurrency,
        index_configuration=settings.index_configuration,
        episode_timeout_s=settings.episode_timeout_s,
    )


def _effective_embedder(base_config: FusedMemoryConfig) -> EffectiveEmbedder:
    """An LLM arm replaces only the LLM block, so the base config's embedder is the one used."""
    return EffectiveEmbedder(
        model=base_config.embedder.model, dimensions=base_config.embedder.dimensions
    )


def _write_artifacts(
    run_dir: Path,
    manifest: RunManifest,
    outcomes: Sequence[EpisodeOutcome],
    records: Sequence[MetricsRecord],
    details: GraphSamenessDetails | None,
) -> None:
    for record in records:
        write_metrics_record(record, run_dir)
    if details is not None:
        _write_json(run_dir / GRAPH_SAMENESS_DETAILS_FILENAME, details.model_dump(mode='json'))
    write_outcomes(run_dir, outcomes)
    if manifest.abort is not None:
        _write_json(run_dir / ABORT_FILENAME, manifest.abort.model_dump(mode='json'))
    atomic_write_text(run_dir / RUN_MANIFEST_FILENAME, serialize_run_manifest(manifest), mkdir=True)


def _write_json(path: Path, payload: object) -> None:
    atomic_write_text(path, canonical_json_text(payload), mkdir=True)


def _outcome_line(outcome: EpisodeOutcome) -> str:
    return json.dumps(outcome.model_dump(mode='json'), sort_keys=True, ensure_ascii=False)


def write_outcomes(run_dir: Path, outcomes: Sequence[EpisodeOutcome]) -> Path:
    """``outcomes.jsonl``: one canonical (sorted-key) JSON line per attempted episode."""
    path = run_dir / OUTCOMES_FILENAME
    atomic_write_text(path, ''.join(_outcome_line(o) + '\n' for o in outcomes), mkdir=True)
    return path


def load_outcomes(path: Path | str) -> tuple[EpisodeOutcome, ...]:
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    return tuple(EpisodeOutcome.model_validate_json(line) for line in lines if line.strip())
