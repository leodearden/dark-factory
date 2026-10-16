"""Cross-arm instrument checks and the client-class parity delta (boundary row 3's logic)."""

from datetime import UTC, datetime

import pytest
from shared.memory_eval_metrics import Metric

from arm_harness._fakes import (
    CODE_SHA,
    CORPUS_SHA,
    incumbent_control_spec,
    llm_spec,
    run_manifest_for,
)
from fused_memory.arm_harness.comparison import (
    RunComparabilityError,
    check_arm_config_symmetry,
    check_single_code_sha,
    client_class_parity,
    require_comparable_runs,
)
from fused_memory.arm_harness.instrument_checks import InstrumentCheckId
from fused_memory.arm_harness.metrics_record import DeltaOf, LlmMetricId, MetricsRecord, record_for
from fused_memory.arm_harness.replay_types import ArmAbort

EARLIER = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
LATER = datetime(2026, 10, 5, 14, 0, tzinfo=UTC)


def _parity_specs():
    openai = incumbent_control_spec(
        arm_id='incumbent-openai', client_class='openai', scratch_group_id='evalmem_par_a'
    )
    generic = incumbent_control_spec(
        arm_id='incumbent-generic',
        client_class='openai_generic',
        scratch_group_id='evalmem_par_b',
    )
    return openai, generic


# --- arm config symmetry -----------------------------------------------------------


def test_symmetric_arms_pass_though_model_serving_and_client_differ():
    runs = (run_manifest_for(llm_spec()), run_manifest_for(incumbent_control_spec()))

    result = check_arm_config_symmetry(runs)

    assert result.check_id is InstrumentCheckId.ARM_CONFIG_SYMMETRY
    assert result.passed, result.detail
    assert result.offenders == ()


_ARM_B = {'arm_id': 'qwen3-8b-b', 'scratch_group_id': 'evalmem_b'}
_ASYMMETRIES = {
    'params.temperature': {
        'spec': llm_spec(**_ARM_B, params={'temperature': 0.7, 'max_tokens': 4096})
    },
    'params.max_tokens': {
        'spec': llm_spec(**_ARM_B, params={'temperature': 0.0, 'max_tokens': 512})
    },
    'concurrency': {
        'settings_summary': {
            'concurrency': 8, 'index_configuration': 'with-indices', 'episode_timeout_s': 120.0
        }
    },
    'index_configuration': {
        'settings_summary': {
            'concurrency': 4, 'index_configuration': 'embedding-only', 'episode_timeout_s': 120.0
        }
    },
    'corpus_sha': {'spec': llm_spec(**_ARM_B, corpus_sha='e' * 64)},
    'effective_embedder': {
        'effective_embedder': {'model': 'text-embedding-3-large', 'dimensions': 3072}
    },
    'graphiti_max_coroutines': {'graphiti_max_coroutines': 9},
    'graphiti_semaphore_limit': {'graphiti_semaphore_limit': 3},
}


@pytest.mark.parametrize('field', sorted(_ASYMMETRIES))
def test_an_asymmetric_field_fails_naming_it_and_each_arms_value(field):
    arm_b = run_manifest_for(**({'spec': llm_spec(**_ARM_B)} | _ASYMMETRIES[field]))

    result = check_arm_config_symmetry((run_manifest_for(llm_spec()), arm_b))

    assert not result.passed
    assert result.offenders == (field,)
    assert field in result.detail
    assert 'qwen3-8b-vllm' in result.detail
    assert 'qwen3-8b-b' in result.detail


def test_symmetry_names_the_values_it_compared():
    other = llm_spec(**_ARM_B, params={'temperature': 0.7, 'max_tokens': 4096})

    result = check_arm_config_symmetry((run_manifest_for(llm_spec()), run_manifest_for(other)))

    assert '0.0' in result.detail
    assert '0.7' in result.detail


def test_symmetry_needs_at_least_two_runs():
    with pytest.raises(ValueError, match='two'):
        check_arm_config_symmetry((run_manifest_for(llm_spec()),))


# --- comparability: the precondition every cross-arm comparison shares -----------------

_ABORT_B = ArmAbort(arm_id='qwen3-8b-b', item_ids=('e1',), error_classes=('X',))
_INCOMPARABLE_SECOND_RUN = {
    'one arm twice': (lambda: run_manifest_for(llm_spec()), 'repeat arm ids'),
    'a limited run': (
        lambda: run_manifest_for(llm_spec(**_ARM_B), episode_ids=('e1', 'e2')),
        r"lacks \['e3'\]",
    ),
    'an extra episode': (
        lambda: run_manifest_for(llm_spec(**_ARM_B), episode_ids=('e1', 'e2', 'e3', 'e9')),
        r"adds \['e9'\]",
    ),
    'an aborted run': (
        lambda: run_manifest_for(llm_spec(**_ARM_B), incomplete=True, abort=_ABORT_B),
        'incomplete',
    ),
    'an incomplete run': (
        lambda: run_manifest_for(llm_spec(**_ARM_B), incomplete=True),
        'incomplete',
    ),
}


@pytest.mark.parametrize('case', sorted(_INCOMPARABLE_SECOND_RUN))
@pytest.mark.parametrize('check', [check_arm_config_symmetry, check_single_code_sha])
def test_incomparable_runs_are_refused_not_passed(check, case):
    second_run, message = _INCOMPARABLE_SECOND_RUN[case]

    with pytest.raises(RunComparabilityError, match=message):
        check((run_manifest_for(llm_spec()), second_run()))


def test_an_episode_set_difference_names_each_differing_arm():
    runs = (
        run_manifest_for(llm_spec()),
        run_manifest_for(llm_spec(**_ARM_B)),
        run_manifest_for(
            llm_spec(arm_id='qwen3-8b-c', scratch_group_id='evalmem_c'), episode_ids=('e1',)
        ),
    )

    with pytest.raises(RunComparabilityError) as raised:
        require_comparable_runs(runs)

    message = str(raised.value)
    assert 'qwen3-8b-c' in message
    assert 'qwen3-8b-b' not in message
    assert "lacks ['e2', 'e3']" in message


# --- single code sha -----------------------------------------------------------------


def test_one_code_sha_across_arms_passes():
    runs = (run_manifest_for(llm_spec()), run_manifest_for(incumbent_control_spec()))

    result = check_single_code_sha(runs)

    assert result.check_id is InstrumentCheckId.SINGLE_CODE_SHA
    assert result.passed, result.detail


def test_two_code_shas_fail_naming_each_arm():
    runs = (
        run_manifest_for(llm_spec()),
        run_manifest_for(incumbent_control_spec(code_sha='f' * 40)),
    )

    result = check_single_code_sha(runs)

    assert not result.passed
    assert result.offenders == (CODE_SHA, 'f' * 40)
    assert 'qwen3-8b-vllm' in result.detail
    assert 'incumbent-ctrl-a' in result.detail


# --- client-class parity -------------------------------------------------------------


def _record(spec, metric: Metric, measured_at: datetime = EARLIER) -> MetricsRecord:
    return record_for(spec, metric, measured_at=measured_at, incomplete=False)


def _scalar(metric_id: LlmMetricId, value: float, n: int = 3) -> Metric:
    return Metric(metric_id=metric_id, kind='scalar', value=value, n=n)


def _conformance(valid: int, received: int) -> Metric:
    return Metric(
        metric_id=LlmMetricId.CONFORMANCE_RATE,
        kind='proportion',
        value=valid / received,
        n=received,
        denominator=received,
        direction='lower_is_worse',
    )


def test_parity_emits_one_scalar_delta_per_shared_metric():
    spec_a, spec_b = _parity_specs()
    records_a = (
        _record(spec_a, _scalar(LlmMetricId.TOKENS_PER_EPISODE, 60.0)),
        _record(spec_a, _conformance(9, 10)),
        _record(spec_a, _scalar(LlmMetricId.USD_PER_EPISODE, 0.002)),
    )
    records_b = (
        _record(spec_b, _scalar(LlmMetricId.TOKENS_PER_EPISODE, 50.0), measured_at=LATER),
        _record(spec_b, _conformance(10, 10)),
        _record(spec_b, _scalar(LlmMetricId.EPISODE_LATENCY_P50, 30.0)),
    )

    deltas = client_class_parity(
        run_manifest_for(spec_a), run_manifest_for(spec_b), records_a, records_b
    )

    by_id = {record.metric.metric_id: record for record in deltas}
    assert set(by_id) == {'tokens-per-episode', 'conformance-rate'}
    assert by_id['tokens-per-episode'].metric.value == pytest.approx(10.0)
    assert by_id['conformance-rate'].metric.value == pytest.approx(-0.1)
    for record in deltas:
        assert record.metric.kind == 'scalar'
        assert record.delta_of == DeltaOf(
            minuend_arm_id='incumbent-openai', subtrahend_arm_id='incumbent-generic'
        )
        assert record.arm_id == 'incumbent-openai'
        assert (record.code_sha, record.corpus_sha) == (CODE_SHA, CORPUS_SHA)
        assert record.preregistration_sha is None
        assert record.incomplete is False
        assert MetricsRecord.model_validate(record.model_dump()) == record


def test_a_parity_delta_is_measured_when_both_sides_are_and_sized_by_the_smaller_side():
    spec_a, spec_b = _parity_specs()
    records_a = (_record(spec_a, _scalar(LlmMetricId.TOKENS_PER_EPISODE, 60.0, n=3)),)
    records_b = (
        _record(spec_b, _scalar(LlmMetricId.TOKENS_PER_EPISODE, 50.0, n=2), measured_at=LATER),
    )

    (delta,) = client_class_parity(
        run_manifest_for(spec_a), run_manifest_for(spec_b), records_a, records_b
    )

    assert delta.measured_at == LATER
    assert delta.metric.n == 2


def test_parity_ignores_episode_order():
    spec_a, spec_b = _parity_specs()

    deltas = client_class_parity(
        run_manifest_for(spec_a, episode_ids=('e1', 'e2', 'e3')),
        run_manifest_for(spec_b, episode_ids=('e3', 'e1', 'e2')),
        (_record(spec_a, _scalar(LlmMetricId.TOKENS_PER_EPISODE, 60.0)),),
        (_record(spec_b, _scalar(LlmMetricId.TOKENS_PER_EPISODE, 60.0)),),
    )

    assert deltas[0].metric.value == 0.0


def test_parity_refuses_an_incomplete_run():
    spec_a, spec_b = _parity_specs()

    with pytest.raises(RunComparabilityError, match='incumbent-openai.*incomplete'):
        client_class_parity(
            run_manifest_for(spec_a, incomplete=True), run_manifest_for(spec_b), (), ()
        )


def test_parity_refuses_an_aborted_run():
    spec_a, spec_b = _parity_specs()
    abort = ArmAbort(arm_id='incumbent-generic', item_ids=('e1',), error_classes=('X',))

    with pytest.raises(RunComparabilityError, match='incumbent-generic'):
        client_class_parity(
            run_manifest_for(spec_a),
            run_manifest_for(spec_b, incomplete=True, abort=abort),
            (),
            (),
        )


def test_parity_refuses_different_episode_sets_naming_the_difference():
    spec_a, spec_b = _parity_specs()

    with pytest.raises(RunComparabilityError, match=r"e3.*e4|e4.*e3") as raised:
        client_class_parity(
            run_manifest_for(spec_a, episode_ids=('e1', 'e2', 'e3')),
            run_manifest_for(spec_b, episode_ids=('e1', 'e2', 'e4')),
            (),
            (),
        )

    assert 'incumbent-openai' in str(raised.value)
    assert 'incumbent-generic' in str(raised.value)


def test_parity_refuses_records_of_another_arm():
    spec_a, spec_b = _parity_specs()

    with pytest.raises(RunComparabilityError, match='incumbent-generic'):
        client_class_parity(
            run_manifest_for(spec_a),
            run_manifest_for(spec_b),
            (_record(spec_b, _scalar(LlmMetricId.TOKENS_PER_EPISODE, 60.0)),),
            (),
        )


def test_parity_refuses_a_metric_reported_twice_by_one_arm():
    spec_a, spec_b = _parity_specs()
    twice = (
        _record(spec_a, _scalar(LlmMetricId.TOKENS_PER_EPISODE, 60.0)),
        _record(spec_a, _scalar(LlmMetricId.TOKENS_PER_EPISODE, 61.0)),
    )

    with pytest.raises(RunComparabilityError, match='tokens-per-episode'):
        client_class_parity(run_manifest_for(spec_a), run_manifest_for(spec_b), twice, ())
