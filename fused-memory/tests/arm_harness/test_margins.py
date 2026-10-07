"""Non-inferiority margins derived from two control runs' records (arm_harness/margins.py)."""

import math
import statistics
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from shared.memory_eval_metrics import Metric, MetricDirection

from arm_harness._fakes import embedding_spec, incumbent_control_spec, llm_spec
from fused_memory.arm_harness.arm_spec import EmbeddingArmSpec, LlmArmSpec
from fused_memory.arm_harness.margins import (
    GATED_METRICS,
    MarginDerivationError,
    MarginEntry,
    SigmaSource,
    derive_margins,
)
from fused_memory.arm_harness.metrics_record import (
    EMBEDDING_METRIC_IDS,
    LLM_METRIC_IDS,
    DeltaOf,
    EmbeddingMetricId,
    IndexConfiguration,
    LlmMetricId,
    MetricsRecord,
    record_for,
)

MEASURED_AT = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
SPEC_A = incumbent_control_spec(arm_id='incumbent-ctrl-a', scratch_group_id='evalmem_ctrl_a')
SPEC_B = incumbent_control_spec(arm_id='incumbent-ctrl-b', scratch_group_id='evalmem_ctrl_b')
SAMENESS_VALUES = (0.6, 0.8, 1.0)


def _proportion(metric_id: str, hits: int, total: int, direction: MetricDirection) -> Metric:
    return Metric(
        metric_id=metric_id,
        kind='proportion',
        value=hits / total,
        n=total,
        denominator=total,
        direction=direction,
    )


def _scalar(metric_id: str, value: float, n: int) -> Metric:
    return Metric(metric_id=metric_id, kind='scalar', value=value, n=n)


def _failure(hits: int, total: int = 200) -> Metric:
    return _proportion(LlmMetricId.EPISODE_FAILURE_RATE, hits, total, 'higher_is_worse')


def _conformance(hits: int, total: int) -> Metric:
    return _proportion(LlmMetricId.CONFORMANCE_RATE, hits, total, 'lower_is_worse')


def _sameness(value: float = 0.8, n: int = 3) -> Metric:
    return _scalar(LlmMetricId.GRAPH_SAMENESS, value, n)


BASE_LLM_METRICS = (
    _conformance(590, 600),
    _failure(2),
    _proportion(LlmMetricId.RETRIEVAL_UTILITY, 150, 200, 'lower_is_worse'),
    _scalar(LlmMetricId.EPISODE_LATENCY_P50, 2500.0, 198),
    _scalar(LlmMetricId.EPISODE_LATENCY_P95, 4000.0, 198),
    _scalar(LlmMetricId.TOKENS_PER_EPISODE, 9000.0, 198),
    _scalar(LlmMetricId.USD_PER_EPISODE, 0.002, 198),
)


def _records(
    spec: LlmArmSpec | EmbeddingArmSpec,
    *replacements: Metric,
    base: Sequence[Metric] = BASE_LLM_METRICS,
    omit: frozenset[str] = frozenset(),
) -> tuple[MetricsRecord, ...]:
    metrics = {m.metric_id: m for m in base} | {m.metric_id: m for m in replacements}
    return tuple(
        record_for(spec, metric, measured_at=MEASURED_AT, incomplete=False)
        for metric_id, metric in metrics.items()
        if metric_id not in omit
    )


def _derive(
    records_a: Sequence[MetricsRecord] | None = None,
    records_b: Sequence[MetricsRecord] | None = None,
    episode_values: Mapping[str, Sequence[float]] | None = None,
) -> tuple[MarginEntry, ...]:
    return derive_margins(
        _records(SPEC_A) if records_a is None else records_a,
        _records(SPEC_B, _sameness()) if records_b is None else records_b,
        episode_values=(
            {LlmMetricId.GRAPH_SAMENESS: SAMENESS_VALUES}
            if episode_values is None
            else episode_values
        ),
    )


def _entry(
    entries: Sequence[MarginEntry],
    metric_id: str,
    configuration: IndexConfiguration | None = None,
) -> MarginEntry:
    (entry,) = [
        e for e in entries if e.metric_id == metric_id and e.index_configuration == configuration
    ]
    return entry


# --- run-pair sigma -------------------------------------------------------------------


def test_run_pair_proportion_margin_is_twice_the_pair_sd_or_the_resolution_floor():
    entries = _derive(_records(SPEC_A, _failure(4)), _records(SPEC_B, _failure(2), _sameness()))

    entry = _entry(entries, LlmMetricId.EPISODE_FAILURE_RATE)
    sigma = abs(0.02 - 0.01) / math.sqrt(2)
    assert entry.sigma_source is SigmaSource.RUN_PAIR
    assert entry.direction == 'higher_is_worse'
    assert entry.control_values == (0.02, 0.01)
    assert entry.reference_value == statistics.mean((0.02, 0.01))
    assert entry.sigma == sigma
    assert entry.floor == 1 / 200
    assert entry.margin == max(2 * sigma, 1 / 200)
    assert entry.index_configuration is None


def test_equal_controls_give_zero_sigma_so_the_floor_is_the_margin():
    entries = _derive(_records(SPEC_A, _failure(0)), _records(SPEC_B, _failure(0), _sameness()))

    entry = _entry(entries, LlmMetricId.EPISODE_FAILURE_RATE)
    assert entry.sigma == 0.0
    assert entry.margin == entry.floor == 1 / 200
    assert math.isfinite(entry.margin)
    assert entry.margin > 0


def test_unequal_denominators_floor_at_the_coarser_resolution():
    entries = _derive(
        _records(SPEC_A, _conformance(590, 600)),
        _records(SPEC_B, _conformance(600, 610), _sameness()),
    )

    assert _entry(entries, LlmMetricId.CONFORMANCE_RATE).floor == 1 / 600


# --- episode standard error -----------------------------------------------------------


def test_graph_sameness_sigma_is_the_per_episode_standard_error_of_its_one_observation():
    entries = _derive()

    entry = _entry(entries, LlmMetricId.GRAPH_SAMENESS)
    sigma = statistics.stdev(SAMENESS_VALUES) / math.sqrt(3)
    assert entry.sigma_source is SigmaSource.EPISODE_SE
    assert entry.direction == 'lower_is_worse'
    assert entry.control_values == (0.8,)
    assert entry.reference_value == 0.8
    assert entry.sigma == sigma
    assert entry.floor is None
    assert entry.margin == 2 * sigma


# --- refusals -------------------------------------------------------------------------


def _refusal(**derive_args) -> str:
    with pytest.raises(MarginDerivationError) as raised:
        _derive(**derive_args)
    assert isinstance(raised.value, ValueError)
    return str(raised.value)


def test_a_run_pair_metric_missing_from_one_run_is_refused():
    message = _refusal(records_a=_records(SPEC_A, omit=frozenset({'retrieval-utility'})))

    assert 'retrieval-utility' in message
    assert 'incumbent-ctrl-a' in message


def test_a_gated_metric_reported_by_neither_run_is_refused():
    omit = frozenset({'retrieval-utility'})
    message = _refusal(
        records_a=_records(SPEC_A, omit=omit),
        records_b=_records(SPEC_B, _sameness(), omit=omit),
    )

    assert 'retrieval-utility' in message


def test_a_non_finite_record_value_is_refused():
    message = _refusal(records_b=_records(SPEC_B, _sameness(math.inf)))

    assert 'graph-sameness' in message
    assert 'inf' in message


def test_a_non_finite_episode_value_is_refused():
    message = _refusal(episode_values={LlmMetricId.GRAPH_SAMENESS: (0.6, math.nan, 1.0)})

    assert 'graph-sameness' in message
    assert 'nan' in message


@pytest.mark.parametrize(
    ('field', 'value_b'),
    [('corpus_sha', 'e' * 64), ('code_sha', 'd' * 40)],
)
def test_records_from_different_shas_are_refused(field, value_b):
    spec_b = incumbent_control_spec(
        arm_id='incumbent-ctrl-b', scratch_group_id='evalmem_ctrl_b', **{field: value_b}
    )

    message = _refusal(records_b=_records(spec_b, _sameness()))

    assert field in message
    assert value_b in message


def test_records_from_different_axes_are_refused():
    message = _refusal(records_b=_embedding_records(_embedding_control('emb-ctl-b'), 160))

    assert 'axis' in message
    assert 'embedding' in message


def test_candidate_records_are_refused():
    message = _refusal(records_a=_records(llm_spec()))

    assert 'candidate' in message
    assert 'qwen3-8b-vllm' in message


def test_delta_records_are_refused():
    delta = record_for(
        SPEC_A,
        _failure(2),
        measured_at=MEASURED_AT,
        incomplete=False,
        delta_of=DeltaOf(minuend_arm_id='incumbent-ctrl-a', subtrahend_arm_id='incumbent-x'),
    )
    records_a = (*_records(SPEC_A, omit=frozenset({'episode-failure-rate'})), delta)

    message = _refusal(records_a=records_a)

    assert 'delta' in message
    assert 'episode-failure-rate' in message


def test_two_runs_of_one_arm_are_refused():
    message = _refusal(records_b=_records(SPEC_A, _sameness()))

    assert 'incumbent-ctrl-a' in message


def test_a_run_reporting_one_metric_twice_is_refused():
    records_a = (*_records(SPEC_A), *_records(SPEC_A, base=(_failure(3),)))

    message = _refusal(records_a=records_a)

    assert 'episode-failure-rate' in message


def test_episode_values_whose_mean_disagrees_with_the_record_are_refused():
    message = _refusal(episode_values={LlmMetricId.GRAPH_SAMENESS: (0.6, 0.7, 1.0)})

    assert 'graph-sameness' in message
    assert '0.8' in message


def test_fewer_than_two_episode_values_are_refused():
    message = _refusal(
        records_b=_records(SPEC_B, _sameness(0.8, n=1)),
        episode_values={LlmMetricId.GRAPH_SAMENESS: (0.8,)},
    )

    assert 'graph-sameness' in message
    assert '1' in message


def test_episode_values_disagreeing_with_the_record_n_are_refused():
    message = _refusal(episode_values={LlmMetricId.GRAPH_SAMENESS: (0.7, 0.8, 0.8, 0.9)})

    assert 'graph-sameness' in message
    assert '4' in message


def test_an_episode_se_metric_reported_by_both_runs_is_refused():
    message = _refusal(records_a=_records(SPEC_A, _sameness()))

    assert 'graph-sameness' in message


def test_an_episode_se_metric_without_episode_values_is_refused():
    message = _refusal(episode_values={})

    assert 'graph-sameness' in message


def test_episode_values_for_a_metric_without_episode_se_sigma_are_refused():
    message = _refusal(
        episode_values={
            LlmMetricId.GRAPH_SAMENESS: SAMENESS_VALUES,
            LlmMetricId.RETRIEVAL_UTILITY: (0.7, 0.8),
        }
    )

    assert 'retrieval-utility' in message


def test_a_record_direction_contradicting_the_gated_table_is_refused():
    wrong = _proportion(LlmMetricId.RETRIEVAL_UTILITY, 150, 200, 'higher_is_worse')

    message = _refusal(records_a=_records(SPEC_A, wrong))

    assert 'retrieval-utility' in message
    assert 'higher_is_worse' in message
    assert 'lower_is_worse' in message


# --- MarginEntry ----------------------------------------------------------------------


def _hand_entry(**overrides) -> MarginEntry:
    data = {
        'metric_id': 'conformance-rate',
        'index_configuration': None,
        'direction': 'lower_is_worse',
        'sigma_source': SigmaSource.RUN_PAIR,
        'control_values': (0.75, 0.875),
        'reference_value': 0.8125,
        'sigma': 0.0625,
        'floor': 0.03125,
        'margin': 0.125,
    }
    return MarginEntry.model_validate(data | overrides)


def test_lower_is_worse_admits_down_to_reference_minus_margin():
    entry = _hand_entry()
    bound = entry.reference_value - entry.margin

    assert entry.admits(bound)
    assert entry.admits(1.0)
    assert not entry.admits(math.nextafter(bound, -math.inf))


def test_higher_is_worse_admits_up_to_reference_plus_margin():
    entry = _hand_entry(metric_id='episode-failure-rate', direction='higher_is_worse')
    bound = entry.reference_value + entry.margin

    assert entry.admits(bound)
    assert entry.admits(0.0)
    assert not entry.admits(math.nextafter(bound, math.inf))


def test_a_non_finite_candidate_value_is_refused_rather_than_judged():
    with pytest.raises(ValueError, match='nan'):
        _hand_entry().admits(math.nan)


@pytest.mark.parametrize(
    'overrides',
    [
        {'margin': 0.1},
        {'floor': 0.25},
        {'floor': None, 'margin': 0.0625},
        {'sigma': -0.0625, 'margin': 0.03125},
        {'reference_value': math.nan},
        {'control_values': (0.75, math.inf)},
        {'sigma': math.inf, 'margin': math.inf},
    ],
)
def test_a_hand_built_entry_must_carry_its_own_derivation(overrides):
    with pytest.raises(ValidationError):
        _hand_entry(**overrides)


# --- the gated table ------------------------------------------------------------------


def test_every_gated_metric_is_a_known_metric():
    assert set(GATED_METRICS) <= LLM_METRIC_IDS | EMBEDDING_METRIC_IDS
    for metric_id, gated in GATED_METRICS.items():
        assert gated.metric_id == metric_id


def test_the_llm_gated_set_is_quality_only():
    llm_gated = set(GATED_METRICS) & LLM_METRIC_IDS

    assert llm_gated == {
        'conformance-rate',
        'episode-failure-rate',
        'retrieval-utility',
        'graph-sameness',
    }
    for not_gated in (
        LlmMetricId.EPISODE_LATENCY_P50,
        LlmMetricId.EPISODE_LATENCY_P95,
        LlmMetricId.TOKENS_PER_EPISODE,
        LlmMetricId.USD_PER_EPISODE,
    ):
        assert not_gated not in GATED_METRICS


def test_only_graph_sameness_takes_the_episode_standard_error():
    by_source = {
        metric_id for metric_id, gated in GATED_METRICS.items()
        if gated.sigma_source is SigmaSource.EPISODE_SE
    }

    assert by_source == {'graph-sameness'}


def test_the_llm_axis_yields_exactly_one_entry_per_gated_metric():
    entries = _derive()

    assert [e.metric_id for e in entries] == sorted(set(GATED_METRICS) & LLM_METRIC_IDS)


# --- embedding axis: entries per index configuration ----------------------------------


def _embedding_control(arm_id: str) -> EmbeddingArmSpec:
    return embedding_spec(
        arm_id=arm_id,
        model_id='text-embedding-3-small',
        serving={'stack': 'openai', 'base_url': 'https://api.openai.com/v1'},
        embedding_dim=1536,
        preregistration_sha=None,
        arm_role='control',
        scratch_group_id=f'evalmem_{arm_id.replace("-", "_")}',
    )


EMBEDDING_BASE = (
    _proportion(EmbeddingMetricId.KNOWN_ITEM_RECALL_AT_5, 160, 200, 'lower_is_worse'),
    _proportion(EmbeddingMetricId.KNOWN_ITEM_RECALL_AT_10, 180, 200, 'lower_is_worse'),
    _scalar(EmbeddingMetricId.MRR, 0.625, 200),
)


def _embedding_records(spec: EmbeddingArmSpec, recall_at_5_hits: int) -> list[MetricsRecord]:
    metrics = (
        _proportion(EmbeddingMetricId.KNOWN_ITEM_RECALL_AT_5, recall_at_5_hits, 200,
                    'lower_is_worse'),
        *EMBEDDING_BASE[1:],
    )
    return [
        record_for(
            spec,
            metric,
            measured_at=MEASURED_AT,
            incomplete=False,
            index_configuration=configuration,
        )
        for metric in metrics
        for configuration in IndexConfiguration
    ] + [
        record_for(
            spec,
            _scalar(EmbeddingMetricId.QUERY_EMBED_LATENCY_P95, 40.0, 200),
            measured_at=MEASURED_AT,
            incomplete=False,
        )
    ]


def test_embedding_margins_are_keyed_per_index_configuration():
    entries = derive_margins(
        _embedding_records(_embedding_control('emb-ctl-a'), 160),
        _embedding_records(_embedding_control('emb-ctl-b'), 164),
        episode_values={},
    )

    keys = [(e.metric_id, e.index_configuration) for e in entries]
    assert keys == sorted(
        (metric_id, configuration)
        for metric_id in ('known-item-recall@5', 'known-item-recall@10', 'mrr')
        for configuration in IndexConfiguration
    )
    for configuration in IndexConfiguration:
        recall = _entry(entries, EmbeddingMetricId.KNOWN_ITEM_RECALL_AT_5, configuration)
        assert recall.control_values == (0.8, 0.82)
        assert recall.floor == 1 / 200
        mrr = _entry(entries, EmbeddingMetricId.MRR, configuration)
        assert mrr.floor is None
        assert mrr.margin == 0.0
