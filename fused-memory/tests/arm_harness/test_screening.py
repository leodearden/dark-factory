"""The η survivor rule: four absolute gates per arm over its screening evidence (arm_harness/screening.py)."""

import dataclasses

import pytest
from fused_memory.arm_harness.screening import (
    SURVIVOR_CAP,
    ArmScreening,
    GateId,
    GateResult,
    GateVerdict,
    ScreeningOutcome,
    ScreeningVerdict,
    conformance_gate,
    context_gate,
    derive_screening_verdict,
    load_screening_verdict,
    reported_evidence,
    screen_arm,
    serialize_screening_verdict,
    throughput_gate,
    vram_gate,
)
from pydantic import ValidationError
from shared.memory_eval_metrics import Metric

from arm_harness._fakes import (
    CODE_SHA,
    FINISHED_AT,
    SCREENING_EPISODE_IDS,
    arm_evidence,
    call,
    episode_outcome,
    health_report,
    preregistration_inputs,
    screening_outcomes,
    screening_spec,
    slate_arm,
)
from fused_memory.arm_harness.metrics_record import LlmMetricId, record_for
from fused_memory.arm_harness.preregistration import latency_envelope
from fused_memory.arm_harness.replay_types import ArmAbort
from fused_memory.arm_harness.screening_evidence import ScreeningEvidenceError

ENVELOPE = latency_envelope(120.0)
QWEN = slate_arm()
PHI = slate_arm(
    arm_id='phi-4-14b', port=8412, served_model_name='phi-4-14b', reasoning='off',
    max_model_len=16384,
)
MOE = slate_arm(
    arm_id='moe-stretch', stack='llamacpp', port=8413, served_model_name='moe-stretch',
    structured_output_mode='json_object', quant='q4_k_xl', reasoning='off', max_model_len=16384,
)
SLATE = (QWEN, PHI, MOE)
INPUTS = preregistration_inputs()


def _durations(*durations: float):
    return tuple(
        episode_outcome(episode_id, duration_ms=duration)
        for episode_id, duration in zip(SCREENING_EPISODE_IDS, durations, strict=False)
    )


def _uniform(duration_ms: float):
    return _durations(*([duration_ms] * len(SCREENING_EPISODE_IDS)))


def _failing_throughput():
    return arm_evidence(MOE, outcomes=_uniform(ENVELOPE.p95_bound_ms))


# --- vocabulary -----------------------------------------------------------------------


def test_the_gates_are_the_four_preregistered_in_order():
    assert [gate.value for gate in GateId] == [
        'conformance-smoke', 'vram-fit', 'context-fit', 'throughput-floor',
    ]
    assert [verdict.value for verdict in GateVerdict] == ['PASS', 'FAIL', 'UNMEASURED']


def test_the_survivor_cap_is_the_prds_three():
    assert SURVIVOR_CAP == 3


# --- gate 1: conformance smoke --------------------------------------------------------


def test_a_smoke_that_exits_zero_passes():
    assert conformance_gate(arm_evidence()).verdict is GateVerdict.PASS


def test_a_failed_smoke_fails_and_carries_its_output():
    result = conformance_gate(arm_evidence(smoke_exit=3, smoke_tail='FAIL endpoint-conformance'))

    assert result.gate is GateId.CONFORMANCE_SMOKE
    assert result.verdict is GateVerdict.FAIL
    assert 'FAIL endpoint-conformance' in result.detail
    assert '3' in result.detail


# --- gate 2: VRAM fit -----------------------------------------------------------------


def test_a_passing_vram_reading_passes_with_its_mib_margin():
    result = vram_gate(arm_evidence(health=health_report(arm_footprint_mib=16637, budget_mib=18019)))

    assert result.verdict is GateVerdict.PASS
    assert (result.value, result.bound, result.margin) == (16637, 18019, 1382)
    assert result.unit == 'MiB'


def test_a_failing_vram_reading_fails_with_a_negative_margin():
    report = health_report(vram_verdict='FAIL', arm_footprint_mib=19000, budget_mib=18019)

    result = vram_gate(arm_evidence(health=report))

    assert result.verdict is GateVerdict.FAIL
    assert result.margin == -981


# --- gate 3: context fit --------------------------------------------------------------


def test_the_longest_own_model_prompt_is_measured_against_the_context_left_for_completion():
    calls = (call(prompt_tokens=1200), call(prompt_tokens=9000), call(prompt_tokens=300))

    result = context_gate(arm_evidence(QWEN, calls=calls))

    assert result.verdict is GateVerdict.PASS
    assert result.value == 9000
    assert result.bound == 32768 - 4096
    assert result.margin == 32768 - 4096 - 9000
    assert result.unit == 'tokens'


def test_a_prompt_exactly_at_the_bound_passes():
    calls = (call(model='phi-4-14b', prompt_tokens=16384 - 4096),)

    assert context_gate(arm_evidence(PHI, calls=calls)).verdict is GateVerdict.PASS


def test_a_prompt_past_the_bound_fails():
    calls = (call(model='phi-4-14b', prompt_tokens=16384 - 4096 + 1),)

    result = context_gate(arm_evidence(PHI, calls=calls))

    assert result.verdict is GateVerdict.FAIL
    assert result.margin == -1


def test_other_models_calls_do_not_enter_the_context_gate():
    calls = (call(prompt_tokens=500), call(model='text-embedding-3-small', prompt_tokens=99999))

    assert context_gate(arm_evidence(QWEN, calls=calls)).value == 500


def test_no_reported_own_model_prompt_is_unmeasured_not_a_pass():
    calls = (call(model='text-embedding-3-small', prompt_tokens=10),)

    result = context_gate(arm_evidence(QWEN, calls=calls))

    assert result.verdict is GateVerdict.UNMEASURED
    assert result.value is None
    assert 'qwen3.5-9b' in result.detail


def test_a_rejected_own_model_call_leaves_the_longest_prompt_unknown():
    rejected = call(status=400, error_excerpt='maximum context length is 32768 tokens')
    calls = (call(prompt_tokens=500), rejected)

    result = context_gate(arm_evidence(QWEN, calls=calls))

    assert result.verdict is GateVerdict.UNMEASURED
    assert '400' in result.detail
    assert 'maximum context length is 32768 tokens' in result.detail


def test_a_served_own_model_call_without_usage_leaves_the_longest_prompt_unknown():
    calls = (call(prompt_tokens=500), call(prompt_tokens=None))

    assert context_gate(arm_evidence(QWEN, calls=calls)).verdict is GateVerdict.UNMEASURED


# --- gate 4: throughput floor ---------------------------------------------------------


def test_a_p95_inside_the_envelope_passes_with_its_ms_margin():
    result = throughput_gate(arm_evidence(outcomes=_uniform(45000.0)), ENVELOPE)

    assert result.verdict is GateVerdict.PASS
    assert (result.value, result.bound, result.margin) == (45000.0, 60000.0, 15000.0)
    assert result.unit == 'ms'


def test_a_p95_on_the_bound_fails_because_the_bound_is_strict():
    result = throughput_gate(_failing_throughput(), ENVELOPE)

    assert result.verdict is GateVerdict.FAIL
    assert result.margin == 0


def test_no_ok_episode_fails_with_no_p95_and_names_what_was_attempted():
    failed = tuple(
        episode_outcome(episode_id, ok=False, duration_ms=120000.0 + index)
        for index, episode_id in enumerate(SCREENING_EPISODE_IDS[:5])
    )
    abort = ArmAbort(
        arm_id='qwen3.5-9b', item_ids=SCREENING_EPISODE_IDS[:5],
        error_classes=('EpisodeOverBudget',) * 5,
    )

    result = throughput_gate(arm_evidence(outcomes=failed, abort=abort), ENVELOPE)

    assert result.verdict is GateVerdict.FAIL
    assert result.value is None
    assert result.bound == 60000.0
    assert '0/5' in result.detail
    assert '120000' in result.detail


# --- unserved arms and the arm-level verdict -------------------------------------------


@pytest.mark.parametrize(
    ('stage', 'kwargs'), [('start', {'start_exit': 4}), ('wait-ready', {'wait_ready_exit': 1})]
)
def test_an_unserved_arm_is_unmeasured_on_every_gate(stage, kwargs):
    screening = screen_arm(arm_evidence(QWEN, **kwargs), ENVELOPE, screening_outcomes())

    assert screening.served is False
    assert [gate.gate for gate in screening.gates] == list(GateId)
    assert all(gate.verdict is GateVerdict.UNMEASURED for gate in screening.gates)
    assert all(stage in gate.detail for gate in screening.gates)
    exit_code = str(next(iter(kwargs.values())))
    assert all(exit_code in gate.detail for gate in screening.gates)
    assert screening.survives is False


def test_an_arm_survives_only_when_all_four_gates_pass():
    passing = screen_arm(arm_evidence(QWEN), ENVELOPE, screening_outcomes())
    failing = screen_arm(_failing_throughput(), ENVELOPE, screening_outcomes())

    assert passing.survives is True
    assert failing.survives is False
    assert (passing.arm_id, passing.stack, passing.reasoning) == ('qwen3.5-9b', 'vllm', 'on')


def test_an_arm_screening_cannot_claim_survival_its_gates_deny():
    passing = screen_arm(arm_evidence(QWEN), ENVELOPE, screening_outcomes())

    with pytest.raises(ValidationError, match='survives'):
        ArmScreening.model_validate(passing.model_dump() | {'survives': False})


def test_an_arm_screening_carries_exactly_the_four_gates_in_order():
    passing = screen_arm(arm_evidence(QWEN), ENVELOPE, screening_outcomes())
    gates = passing.model_dump()['gates']

    with pytest.raises(ValidationError, match='gates'):
        ArmScreening.model_validate(passing.model_dump() | {'gates': gates[::-1]})


def test_a_gate_margin_is_the_bound_minus_the_value():
    with pytest.raises(ValidationError, match='margin'):
        GateResult(
            gate=GateId.VRAM_FIT, verdict=GateVerdict.PASS, value=10.0, bound=20.0,
            margin=5.0, unit='MiB', detail='x',
        )


# --- the reported, non-gating block --------------------------------------------------


def test_reports_the_run_the_tap_and_the_health_row_without_gating_on_them():
    outcomes = (
        *_durations(1000.0, 2000.0, 3000.0),
        episode_outcome('e04', ok=False, duration_ms=120000.0),
        episode_outcome('e05', ok=False, duration_ms=120001.0),
    )
    calls = (
        call(prompt_tokens=100, duration_ms=100.0, finish_reason='length'),
        call(prompt_tokens=200, duration_ms=200.0),
        call(prompt_tokens=300, duration_ms=300.0),
        call(status=400),
        call(model='text-embedding-3-small'),
    )
    reference = (
        episode_outcome('e01', entity_names=('alice', 'bob')),
        episode_outcome('e02', entity_names=('alice',)),
    )
    evidence = arm_evidence(
        QWEN,
        outcomes=outcomes,
        calls=calls,
        health=health_report(top_level_entities_named=3),
    )

    reported = reported_evidence(evidence, reference)

    assert reported.episode_failure_rate == pytest.approx(2 / 5)
    assert reported.conformance_rate == 1.0
    assert reported.latency_p50_ms == 2000.0
    assert reported.tokens_per_episode == 1200.0
    assert [(c.error_class, c.count) for c in reported.error_classes] == [('EpisodeOverBudget', 2)]
    assert reported.incomplete is False
    assert reported.abort_item_count is None
    assert reported.calls_per_ok_episode_p50 == 9
    assert reported.calls_per_ok_episode_max == 9
    assert reported.own_model_calls == 4
    assert reported.call_duration_p50_ms == 200.0
    assert reported.call_duration_p95_ms == 2000.0
    assert reported.length_truncations == 1
    assert reported.rejected_calls == 1
    assert reported.other_model_calls == 1
    assert reported.health_verdict == 'PASS'
    assert reported.top_level_entities_named == 3
    assert reported.graph_sameness_mean_entity_jaccard == pytest.approx((1.0 + 0.5) / 2)
    assert reported.graph_sameness_n == 2


def test_an_aborted_run_reports_its_abort_size():
    abort = ArmAbort(
        arm_id='qwen3.5-9b', item_ids=('e01', 'e02', 'e03', 'e04', 'e05'),
        error_classes=('EpisodeOverBudget',) * 5,
    )

    reported = reported_evidence(arm_evidence(abort=abort), screening_outcomes())

    assert reported.incomplete is True
    assert reported.abort_item_count == 5


def test_an_unserved_arm_reports_nothing_measured():
    reported = reported_evidence(arm_evidence(start_exit=4), screening_outcomes())

    assert reported.episode_failure_rate is None
    assert reported.own_model_calls == 0
    assert reported.health_verdict is None
    assert reported.graph_sameness_n == 0


def test_reported_values_never_move_survival():
    spec = screening_spec(QWEN)
    records = (
        record_for(
            spec,
            Metric(metric_id=LlmMetricId.EPISODE_LATENCY_P95, kind='scalar', value=30000.0, n=20),
            measured_at=FINISHED_AT, incomplete=False,
        ),
        record_for(
            spec,
            Metric(
                metric_id=LlmMetricId.EPISODE_FAILURE_RATE, kind='proportion', value=1.0,
                n=20, denominator=20, direction='higher_is_worse',
            ),
            measured_at=FINISHED_AT, incomplete=False,
        ),
    )

    screening = screen_arm(arm_evidence(QWEN, records=records), ENVELOPE, screening_outcomes())

    assert screening.reported.episode_failure_rate == 1.0
    assert all(gate.verdict is GateVerdict.PASS for gate in screening.gates)
    assert screening.survives is True


# --- the slate verdict ----------------------------------------------------------------


def _evidence(**by_arm):
    evidence = {arm.arm_id: arm_evidence(arm) for arm in SLATE}
    return evidence | by_arm


def test_derives_the_survivors_in_slate_order_and_proceeds_to_theta():
    evidence = _evidence(**{'moe-stretch': _failing_throughput()})

    verdict = derive_screening_verdict(SLATE, evidence, INPUTS, screening_outcomes())

    assert verdict.schema_version == 1
    assert [arm.arm_id for arm in verdict.arms] == ['qwen3.5-9b', 'phi-4-14b', 'moe-stretch']
    assert verdict.survivors == ('qwen3.5-9b', 'phi-4-14b')
    assert verdict.outcome is ScreeningOutcome.PROCEED_TO_THETA
    assert (verdict.cap.cap, verdict.cap.candidates, verdict.cap.binds) == (3, 3, False)
    assert verdict.envelope == INPUTS.envelope
    assert verdict.code_sha == CODE_SHA
    assert verdict.corpus_sha == INPUTS.corpus_sha
    assert verdict.preregistration_sha == screening_spec(QWEN).preregistration_sha


def test_a_single_survivor_proceeds_without_error():
    evidence = _evidence(**{
        'qwen3.5-9b': arm_evidence(QWEN, start_exit=4),
        'moe-stretch': _failing_throughput(),
    })

    verdict = derive_screening_verdict(SLATE, evidence, INPUTS, screening_outcomes())

    assert verdict.survivors == ('phi-4-14b',)
    assert verdict.outcome is ScreeningOutcome.PROCEED_TO_THETA


def test_zero_survivors_is_a_negative_verdict_not_an_error():
    evidence = {arm.arm_id: arm_evidence(arm, smoke_exit=3) for arm in SLATE}

    verdict = derive_screening_verdict(SLATE, evidence, INPUTS, screening_outcomes())

    assert verdict.survivors == ()
    assert verdict.outcome is ScreeningOutcome.NEGATIVE_VERDICT


def test_refuses_evidence_that_is_not_the_slate():
    evidence = _evidence()
    del evidence['phi-4-14b']
    evidence['mistral-small'] = arm_evidence(slate_arm(arm_id='mistral-small'))

    with pytest.raises(ScreeningEvidenceError, match=r'(?s)phi-4-14b.*mistral-small'):
        derive_screening_verdict(SLATE, evidence, INPUTS, screening_outcomes())


def _with_spec(arm, **update):
    evidence = arm_evidence(arm)
    return dataclasses.replace(evidence, spec=evidence.spec.model_copy(update=update))


def test_refuses_arms_at_more_than_one_code_sha():
    evidence = _evidence(**{'phi-4-14b': _with_spec(PHI, code_sha='d' * 40)})

    with pytest.raises(ScreeningEvidenceError, match='code_sha'):
        derive_screening_verdict(
            SLATE, evidence, preregistration_inputs(), screening_outcomes()
        )


def test_refuses_arms_not_at_the_controls_code_sha():
    inputs = preregistration_inputs(code_sha='e' * 40)

    with pytest.raises(ScreeningEvidenceError, match=r'(?s)code_sha.*e{40}'):
        derive_screening_verdict(SLATE, _evidence(), inputs, screening_outcomes())


def test_refuses_arms_at_more_than_one_preregistration_sha():
    evidence = _evidence(**{'phi-4-14b': _with_spec(PHI, preregistration_sha='f' * 40)})

    with pytest.raises(ScreeningEvidenceError, match='preregistration_sha'):
        derive_screening_verdict(SLATE, evidence, INPUTS, screening_outcomes())


def test_refuses_to_rank_when_more_arms_pass_than_the_cap_admits():
    fourth = slate_arm(arm_id='fourth', port=8411, served_model_name='fourth')
    slate = (*SLATE, fourth)
    evidence = _evidence(fourth=arm_evidence(fourth))

    with pytest.raises(ScreeningEvidenceError, match='no ranking rule is pre-registered'):
        derive_screening_verdict(slate, evidence, INPUTS, screening_outcomes())


def test_a_larger_slate_within_the_cap_derives_and_says_the_cap_could_bind():
    fourth = slate_arm(arm_id='fourth', port=8411, served_model_name='fourth')
    slate = (*SLATE, fourth)
    evidence = _evidence(fourth=arm_evidence(fourth, smoke_exit=3))

    verdict = derive_screening_verdict(slate, evidence, INPUTS, screening_outcomes())

    assert len(verdict.survivors) == 3
    assert verdict.cap.binds is True


def test_the_verdict_refuses_survivors_its_arms_do_not_support():
    verdict = derive_screening_verdict(SLATE, _evidence(), INPUTS, screening_outcomes())

    with pytest.raises(ValidationError, match='survivors'):
        ScreeningVerdict.model_validate(verdict.model_dump() | {'survivors': ('qwen3.5-9b',)})


def test_the_verdict_refuses_an_outcome_its_survivors_do_not_support():
    verdict = derive_screening_verdict(SLATE, _evidence(), INPUTS, screening_outcomes())

    with pytest.raises(ValidationError, match='outcome'):
        ScreeningVerdict.model_validate(
            verdict.model_dump() | {'outcome': ScreeningOutcome.NEGATIVE_VERDICT}
        )


def test_the_verdict_serializes_canonically_and_round_trips(tmp_path):
    verdict = derive_screening_verdict(SLATE, _evidence(), INPUTS, screening_outcomes())
    text = serialize_screening_verdict(verdict)
    path = tmp_path / 'screening-verdict.json'
    path.write_text(text)

    assert serialize_screening_verdict(load_screening_verdict(path)) == text
    assert load_screening_verdict(path) == verdict
    assert text.endswith('\n')
    assert text == serialize_screening_verdict(
        derive_screening_verdict(SLATE, _evidence(), INPUTS, screening_outcomes())
    )
