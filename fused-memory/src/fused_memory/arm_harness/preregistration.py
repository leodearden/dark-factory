"""The LLM axis's pre-registered decision quantities, composed from the incumbent control pair.

Margins (``margins.derive_margins``), the warm-p95-under-load latency envelope
``p95_bound = episode_timeout / LATENCY_HEADROOM``, and the calls-per-episode profile the
reasoning-mode ruling is priced from. Rationale:
plans/local-memory-models-eval-preregistration.md.
"""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator
from shared.memory_eval_metrics import canonical_json_text

from fused_memory.arm_harness.arm_spec import ArmId, ContentSha, GitSha
from fused_memory.arm_harness.checks import control_variance_check
from fused_memory.arm_harness.frozen_model import FrozenModel
from fused_memory.arm_harness.llm_metrics import (
    GraphSamenessDetails,
    nearest_rank,
    ok_token_usages,
)
from fused_memory.arm_harness.margins import MarginEntry, derive_margins
from fused_memory.arm_harness.metrics_record import LlmMetricId, MetricsRecord
from fused_memory.arm_harness.replay_types import EpisodeOutcome
from fused_memory.arm_harness.run_manifest import RunManifest

LATENCY_HEADROOM = 2.0
PREREGISTRATION_INPUTS_FILENAME = 'preregistration-inputs.json'
_MS_PER_S = 1000
_P50, _P95 = 0.50, 0.95


class PreregistrationError(ValueError):
    """The control pair cannot yield a valid pre-registration, so none is written."""


class LatencyEnvelope(FrozenModel):
    episode_timeout_s: float = Field(gt=0)
    headroom: float = Field(ge=1)
    p95_bound_ms: float

    @model_validator(mode='after')
    def _bound_is_derived(self) -> Self:
        derived = self.episode_timeout_s * _MS_PER_S / self.headroom
        if self.p95_bound_ms != derived:
            raise ValueError(
                f'p95_bound_ms {self.p95_bound_ms} != episode_timeout_s '
                f'{self.episode_timeout_s} * {_MS_PER_S} / headroom {self.headroom} = {derived}'
            )
        return self

    def admits(self, p95_ms: float) -> bool:
        """Whether a warm p95 under load lies strictly inside the bound."""
        if not math.isfinite(p95_ms):
            raise ValueError(f'latency p95 {p95_ms} cannot be judged against the envelope')
        return p95_ms < self.p95_bound_ms


def latency_envelope(episode_timeout_s: float) -> LatencyEnvelope:
    return LatencyEnvelope(
        episode_timeout_s=episode_timeout_s,
        headroom=LATENCY_HEADROOM,
        p95_bound_ms=episode_timeout_s * _MS_PER_S / LATENCY_HEADROOM,
    )


class CallProfile(FrozenModel):
    n_episodes: int = Field(gt=0)
    calls_p50: int
    calls_p95: int
    calls_max: int
    input_tokens_per_call: float
    output_tokens_per_call: float


def call_profile(
    outcomes_a: Sequence[EpisodeOutcome], outcomes_b: Sequence[EpisodeOutcome]
) -> CallProfile:
    """LLM calls per ok episode over both control runs, percentiles by nearest rank."""
    usages = ok_token_usages((*outcomes_a, *outcomes_b))
    calls = sorted(usage.llm_calls for usage in usages)
    if not calls or sum(calls) == 0:
        raise PreregistrationError(
            f'a call profile needs an ok episode with LLM calls; got calls {calls}'
        )
    total_calls = sum(calls)
    return CallProfile(
        n_episodes=len(calls),
        calls_p50=nearest_rank(calls, _P50),
        calls_p95=nearest_rank(calls, _P95),
        calls_max=calls[-1],
        input_tokens_per_call=sum(u.input_tokens for u in usages) / total_calls,
        output_tokens_per_call=sum(u.output_tokens for u in usages) / total_calls,
    )


class PreregistrationInputs(FrozenModel):
    schema_version: Literal[1]
    control_arm_ids: tuple[ArmId, ArmId]
    code_sha: GitSha
    corpus_sha: ContentSha
    margins: tuple[MarginEntry, ...]
    envelope: LatencyEnvelope
    call_profile: CallProfile
    incumbent_latency_p95_ms: float
    incumbent_latency_max_ms: float


def derive_preregistration_inputs(
    run_a: RunManifest,
    records_a: Sequence[MetricsRecord],
    outcomes_a: Sequence[EpisodeOutcome],
    run_b: RunManifest,
    records_b: Sequence[MetricsRecord],
    outcomes_b: Sequence[EpisodeOutcome],
    sameness_b: GraphSamenessDetails,
) -> PreregistrationInputs:
    """Refuses unless the pair passes control-check and the incumbent sits inside its envelope."""
    _require_control_checks_pass(run_a, records_a, run_b, records_b)
    for run, outcomes in ((run_a, outcomes_a), (run_b, outcomes_b)):
        _require_outcomes_of(run, outcomes)
    envelope = latency_envelope(_shared_episode_timeout(run_a, run_b))
    margins = derive_margins(
        records_a,
        records_b,
        episode_values={LlmMetricId.GRAPH_SAMENESS: _sameness_values(run_b, sameness_b)},
    )
    p95_ms = max(_latency_p95_ms(records_a), _latency_p95_ms(records_b))
    if not envelope.admits(p95_ms):
        raise PreregistrationError(
            f'the incumbent latency p95 {p95_ms} ms is not inside the envelope bound '
            f'{envelope.p95_bound_ms} ms: an envelope the incumbent fails is not a valid '
            'pre-registration'
        )
    return PreregistrationInputs(
        schema_version=1,
        control_arm_ids=(run_a.spec.arm_id, run_b.spec.arm_id),
        code_sha=run_a.spec.code_sha,
        corpus_sha=run_a.spec.corpus_sha,
        margins=margins,
        envelope=envelope,
        call_profile=call_profile(outcomes_a, outcomes_b),
        incumbent_latency_p95_ms=p95_ms,
        incumbent_latency_max_ms=max(o.duration_ms for o in (*outcomes_a, *outcomes_b) if o.ok),
    )


def _require_control_checks_pass(
    run_a: RunManifest,
    records_a: Sequence[MetricsRecord],
    run_b: RunManifest,
    records_b: Sequence[MetricsRecord],
) -> None:
    results = control_variance_check(
        [run_a, run_b],
        {run_a.spec.arm_id: records_a, run_b.spec.arm_id: records_b},
        reference=None,
    )
    failed = [f'{check.check_id.value}: {check.detail}' for check in results if not check.passed]
    if failed:
        raise PreregistrationError(f'the control pair fails control-check: {"; ".join(failed)}')


def _require_outcomes_of(run: RunManifest, outcomes: Sequence[EpisodeOutcome]) -> None:
    attempted = [outcome.episode_id for outcome in outcomes]
    if sorted(attempted) != sorted(run.episode_ids):
        raise PreregistrationError(
            f'the outcomes given for {run.spec.arm_id!r} are not its run\'s episodes: they '
            f'lack {sorted(set(run.episode_ids) - set(attempted))} and add '
            f'{sorted(set(attempted) - set(run.episode_ids))}'
        )


def _shared_episode_timeout(run_a: RunManifest, run_b: RunManifest) -> float:
    timeout_a = run_a.settings_summary.episode_timeout_s
    timeout_b = run_b.settings_summary.episode_timeout_s
    if timeout_a != timeout_b:
        raise PreregistrationError(
            f'the control runs ran under different episode timeouts: '
            f'{run_a.spec.arm_id}={timeout_a}, {run_b.spec.arm_id}={timeout_b}'
        )
    return timeout_a


def _sameness_values(run_b: RunManifest, sameness_b: GraphSamenessDetails) -> tuple[float, ...]:
    if not sameness_b.episodes:
        raise PreregistrationError(
            f'the graph-sameness comparison of {run_b.spec.arm_id!r} against its reference '
            'holds no episode'
        )
    return tuple(episode.entity_jaccard for episode in sameness_b.episodes)


def _latency_p95_ms(records: Sequence[MetricsRecord]) -> float:
    values = [
        record.metric.value
        for record in records
        if record.metric.metric_id == LlmMetricId.EPISODE_LATENCY_P95
    ]
    if len(values) != 1:
        arms = sorted({record.arm_id for record in records})
        raise PreregistrationError(
            f'control run {arms} must report one {LlmMetricId.EPISODE_LATENCY_P95.value!r}; '
            f'got {values}'
        )
    return values[0]


def serialize_preregistration_inputs(inputs: PreregistrationInputs) -> str:
    return canonical_json_text(inputs.model_dump(mode='json'))


def load_preregistration_inputs(path: Path | str) -> PreregistrationInputs:
    return PreregistrationInputs.model_validate_json(Path(path).read_text())
