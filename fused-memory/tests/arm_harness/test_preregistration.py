"""The LLM axis's pre-registered decision quantities (arm_harness/preregistration.py)."""

import math
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from shared.memory_eval_metrics import Metric

from arm_harness._fakes import incumbent_control_spec, run_manifest_for
from fused_memory.arm_harness.arm_spec import LlmArmSpec
from fused_memory.arm_harness.comparison import RunComparabilityError
from fused_memory.arm_harness.llm_metrics import (
    EpisodeSameness,
    GraphSamenessDetails,
    TokenAccountingError,
)
from fused_memory.arm_harness.margins import derive_margins
from fused_memory.arm_harness.metrics_record import LlmMetricId, MetricsRecord, record_for
from fused_memory.arm_harness.preregistration import (
    LATENCY_HEADROOM,
    PREREGISTRATION_INPUTS_FILENAME,
    CallProfile,
    LatencyEnvelope,
    PreregistrationError,
    PreregistrationInputs,
    call_profile,
    derive_preregistration_inputs,
    latency_envelope,
    load_preregistration_inputs,
    serialize_preregistration_inputs,
)
from fused_memory.arm_harness.replay_types import ArmAbort, EpisodeOutcome
from fused_memory.arm_harness.run_manifest import RunManifest
from fused_memory.backends.llm_token_usage import LlmTokenUsage

MEASURED_AT = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
SPEC_A = incumbent_control_spec(arm_id='incumbent-ctrl-a', scratch_group_id='evalmem_ctrl_a')
SPEC_B = incumbent_control_spec(arm_id='incumbent-ctrl-b', scratch_group_id='evalmem_ctrl_b')
EPISODES = ('e1', 'e2', 'e3', 'e4')
JACCARDS = (0.6, 0.8, 1.0, 0.8)


def _outcome(
    episode_id: str,
    calls: int,
    *,
    ok: bool = True,
    duration_ms: float = 3000.0,
    tokens: LlmTokenUsage | None | str = 'derived',
) -> EpisodeOutcome:
    usage = (
        LlmTokenUsage(input_tokens=1000 * calls + 7, output_tokens=100 * calls, llm_calls=calls)
        if tokens == 'derived'
        else tokens
    )
    return EpisodeOutcome(
        episode_id=episode_id,
        ok=ok,
        error_class=None if ok else 'TimeoutError',
        duration_ms=duration_ms,
        tokens=usage if not isinstance(usage, str) else None,
        replay_episode_uuid=f'replay-{episode_id}' if ok else None,
        entity_names=('alice',) if ok else (),
        edge_triples=(),
    )


def _outcomes(calls: Sequence[int]) -> tuple[EpisodeOutcome, ...]:
    return tuple(_outcome(episode_id, n) for episode_id, n in zip(EPISODES, calls, strict=True))


OUTCOMES_A = _outcomes((3, 5, 8, 10))
OUTCOMES_B = _outcomes((4, 4, 6, 6))


def _proportion(metric_id: str, hits: int, total: int, direction) -> Metric:
    return Metric(
        metric_id=metric_id,
        kind='proportion',
        value=hits / total,
        n=total,
        denominator=total,
        direction=direction,
    )


def _scalar(metric_id: str, value: float, n: int = 4) -> Metric:
    return Metric(metric_id=metric_id, kind='scalar', value=value, n=n)


def _records(
    spec: LlmArmSpec, *, p95_ms: float = 4000.0, sameness: bool = False
) -> tuple[MetricsRecord, ...]:
    metrics = [
        _proportion(LlmMetricId.CONFORMANCE_RATE, 26, 26, 'lower_is_worse'),
        _proportion(LlmMetricId.EPISODE_FAILURE_RATE, 0, 4, 'higher_is_worse'),
        _proportion(LlmMetricId.RETRIEVAL_UTILITY, 3, 4, 'lower_is_worse'),
        _scalar(LlmMetricId.EPISODE_LATENCY_P50, 3000.0),
        _scalar(LlmMetricId.EPISODE_LATENCY_P95, p95_ms),
        _scalar(LlmMetricId.TOKENS_PER_EPISODE, 7000.0),
        _scalar(LlmMetricId.USD_PER_EPISODE, 0.0015),
    ]
    if sameness:
        metrics.append(_scalar(LlmMetricId.GRAPH_SAMENESS, sum(JACCARDS) / len(JACCARDS)))
    return tuple(
        record_for(spec, metric, measured_at=MEASURED_AT, incomplete=False) for metric in metrics
    )


def _sameness_details(jaccards: Sequence[float] = JACCARDS) -> GraphSamenessDetails:
    return GraphSamenessDetails(
        episodes=tuple(
            EpisodeSameness(
                episode_id=episode_id,
                arm_entity_count=1,
                ref_entity_count=1,
                arm_edge_count=0,
                ref_edge_count=0,
                entity_jaccard=jaccard,
                edge_triple_jaccard=1.0,
            )
            for episode_id, jaccard in zip(EPISODES, jaccards, strict=False)
        ),
        excluded_arm_ids=(),
        excluded_reference_ids=(),
    )


def _run(spec: LlmArmSpec, **overrides) -> RunManifest:
    return run_manifest_for(spec, **({'episode_ids': EPISODES} | overrides))


def _derive(**overrides) -> PreregistrationInputs:
    args = {
        'run_a': _run(SPEC_A),
        'records_a': _records(SPEC_A),
        'outcomes_a': OUTCOMES_A,
        'run_b': _run(SPEC_B),
        'records_b': _records(SPEC_B, sameness=True),
        'outcomes_b': OUTCOMES_B,
        'sameness_b': _sameness_details(),
    } | overrides
    return derive_preregistration_inputs(**args)


# --- latency envelope -----------------------------------------------------------------


def test_the_envelope_halves_the_production_episode_timeout():
    envelope = latency_envelope(episode_timeout_s=120.0)

    assert LATENCY_HEADROOM == 2.0
    assert envelope.episode_timeout_s == 120.0
    assert envelope.headroom == LATENCY_HEADROOM
    assert envelope.p95_bound_ms == 60000.0


def test_the_envelope_admits_a_p95_strictly_inside_its_bound():
    envelope = latency_envelope(episode_timeout_s=120.0)

    assert envelope.admits(59999.0)
    assert not envelope.admits(60000.0)
    with pytest.raises(ValueError, match='nan'):
        envelope.admits(math.nan)


@pytest.mark.parametrize(
    'fields',
    [
        {'episode_timeout_s': 120.0, 'headroom': 2.0, 'p95_bound_ms': 59000.0},
        {'episode_timeout_s': 120.0, 'headroom': 0.5, 'p95_bound_ms': 240000.0},
        {'episode_timeout_s': 0.0, 'headroom': 2.0, 'p95_bound_ms': 0.0},
    ],
)
def test_an_envelope_must_carry_its_own_derivation(fields):
    with pytest.raises(ValidationError):
        LatencyEnvelope.model_validate(fields)


# --- call profile ---------------------------------------------------------------------


def test_the_call_profile_pools_both_runs_ok_episodes_by_nearest_rank():
    failed = _outcome('e9', 40, ok=False)
    profile = call_profile(
        (*_outcomes((3, 5, 8, 10))[:2], failed),
        (*_outcomes((3, 5, 8, 10))[2:], failed),
    )

    calls = (3, 5, 8, 10)
    assert profile == CallProfile(
        n_episodes=4,
        calls_p50=5,
        calls_p95=10,
        calls_max=10,
        input_tokens_per_call=sum(1000 * c + 7 for c in calls) / sum(calls),
        output_tokens_per_call=sum(100 * c for c in calls) / sum(calls),
    )


def test_an_ok_episode_without_token_usage_is_a_token_accounting_error():
    with pytest.raises(TokenAccountingError) as raised:
        call_profile((_outcome('e1', 3, tokens=None),), OUTCOMES_B)

    assert raised.value.episode_ids == ('e1',)


def test_a_call_profile_needs_an_ok_episode():
    failed = (_outcome('e1', 3, ok=False),)

    with pytest.raises(PreregistrationError, match='ok episode'):
        call_profile(failed, failed)


# --- derive_preregistration_inputs ----------------------------------------------------


def test_the_inputs_compose_margins_envelope_and_call_profile_from_the_control_pair():
    inputs = _derive()

    assert inputs.schema_version == 1
    assert inputs.control_arm_ids == ('incumbent-ctrl-a', 'incumbent-ctrl-b')
    assert inputs.code_sha == SPEC_A.code_sha
    assert inputs.corpus_sha == SPEC_A.corpus_sha
    assert inputs.margins == derive_margins(
        _records(SPEC_A),
        _records(SPEC_B, sameness=True),
        episode_values={LlmMetricId.GRAPH_SAMENESS: JACCARDS},
    )
    assert inputs.envelope == latency_envelope(episode_timeout_s=120.0)
    assert inputs.call_profile == call_profile(OUTCOMES_A, OUTCOMES_B)
    assert inputs.incumbent_latency_p95_ms == 4000.0
    assert inputs.incumbent_latency_max_ms == 3000.0


def test_the_incumbent_p95_is_the_worse_of_the_two_runs():
    inputs = _derive(records_b=_records(SPEC_B, p95_ms=4500.0, sameness=True))

    assert inputs.incumbent_latency_p95_ms == 4500.0


def test_the_incumbent_max_is_the_slowest_ok_episode_of_either_run():
    slow = (*OUTCOMES_B[:3], _outcome('e4', 6, duration_ms=9000.0))

    assert _derive(outcomes_b=slow).incumbent_latency_max_ms == 9000.0


def _refused(**overrides) -> str:
    with pytest.raises(PreregistrationError) as raised:
        _derive(**overrides)
    assert isinstance(raised.value, ValueError)
    return str(raised.value)


def test_an_envelope_the_incumbent_fails_is_refused():
    message = _refused(records_b=_records(SPEC_B, p95_ms=60000.0, sameness=True))

    assert 'not a valid pre-registration' in message
    assert '60000' in message


def test_runs_with_different_episode_timeouts_are_refused():
    summary = {'concurrency': 4, 'index_configuration': 'with-indices', 'episode_timeout_s': 90.0}

    message = _refused(run_b=_run(SPEC_B, settings_summary=summary))

    assert '90.0' in message
    assert '120.0' in message


def test_a_failed_control_variance_check_is_refused():
    spec_b = incumbent_control_spec(
        arm_id='incumbent-ctrl-b',
        scratch_group_id='evalmem_ctrl_b',
        params={'temperature': 0.7, 'max_tokens': 4096},
    )

    message = _refused(run_b=_run(spec_b), records_b=_records(spec_b, sameness=True))

    assert 'arm-config-symmetry' in message
    assert 'params.temperature' in message


def test_an_incomplete_run_is_not_comparable():
    abort = ArmAbort(arm_id='incumbent-ctrl-b', item_ids=('e1',), error_classes=('X',))

    with pytest.raises(RunComparabilityError, match='incomplete'):
        _derive(run_b=_run(SPEC_B, incomplete=True, abort=abort))


def test_runs_over_different_episode_sets_are_not_comparable():
    with pytest.raises(RunComparabilityError, match='episode set'):
        _derive(run_b=_run(SPEC_B, episode_ids=EPISODES[:3]))


def test_outcomes_that_are_not_the_runs_episodes_are_refused():
    message = _refused(outcomes_a=OUTCOMES_A[:3])

    assert 'incumbent-ctrl-a' in message
    assert 'e4' in message


def test_an_empty_graph_sameness_comparison_is_refused():
    message = _refused(sameness_b=_sameness_details(()))

    assert 'graph-sameness' in message
    assert 'incumbent-ctrl-b' in message


# --- serialization --------------------------------------------------------------------


def test_the_inputs_round_trip_canonically(tmp_path):
    inputs = _derive()
    path = tmp_path / PREREGISTRATION_INPUTS_FILENAME
    path.write_text(serialize_preregistration_inputs(inputs))

    loaded = load_preregistration_inputs(path)

    assert loaded == inputs
    assert serialize_preregistration_inputs(loaded) == path.read_text()
    assert PREREGISTRATION_INPUTS_FILENAME == 'preregistration-inputs.json'
