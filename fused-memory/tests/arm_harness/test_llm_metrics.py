"""LLM-axis metrics from synthetic replay outcomes, each against a hand-computed value."""

from datetime import UTC, datetime

import pytest
from fused_memory.arm_harness.llm_metrics import (
    GRAPH_SAMENESS_DETAILS_FILENAME,
    TokenAccountingError,
    episode_failure_rate_metric,
    graph_sameness_details,
    graph_sameness_metric,
    latency_metric,
    llm_axis_records,
    tokens_per_episode_metric,
    usd_per_episode_metric,
)
from shared.memory_eval_metrics import canonical_json_text

from arm_harness._fakes import (
    CODE_SHA,
    CORPUS_SHA,
    PREREG_SHA,
    embedding_spec,
    incumbent_control_spec,
    llm_spec,
)
from fused_memory.arm_harness.conformance import ConformanceCounts
from fused_memory.arm_harness.metrics_record import LlmMetricId, MetricsRecord
from fused_memory.arm_harness.replay import ArmAbort, ArmRunResult, EpisodeOutcome
from fused_memory.arm_harness.retrieval import RETRIEVAL_UTILITY_K
from fused_memory.backends.llm_token_usage import LlmTokenUsage

MEASURED_AT = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
CONFORMANCE = ConformanceCounts(
    schema_valid=9, transport_errors=2, invalid_by_error_class={'ValidationError': 1}
)


def _usage(input_tokens: int = 30, output_tokens: int = 10) -> LlmTokenUsage:
    return LlmTokenUsage(input_tokens=input_tokens, output_tokens=output_tokens, llm_calls=1)


def _ok(
    episode_id: str,
    *,
    duration_ms: float = 10.0,
    tokens: LlmTokenUsage | None = None,
    entities: tuple[str, ...] = (),
    edges: tuple[tuple[str, str, str], ...] = (),
) -> EpisodeOutcome:
    return EpisodeOutcome(
        episode_id=episode_id,
        ok=True,
        error_class=None,
        duration_ms=duration_ms,
        tokens=tokens if tokens is not None else _usage(),
        replay_episode_uuid=f'replay-{episode_id}',
        entity_names=entities,
        edge_triples=edges,
    )


def _failed(episode_id: str, *, duration_ms: float = 5.0) -> EpisodeOutcome:
    return EpisodeOutcome(
        episode_id=episode_id,
        ok=False,
        error_class='RuntimeError',
        duration_ms=duration_ms,
        tokens=None,
        replay_episode_uuid=None,
        entity_names=(),
        edge_triples=(),
    )


def _no_tokens(episode_id: str) -> EpisodeOutcome:
    return _ok(episode_id).model_copy(update={'tokens': None})


def _result(
    outcomes: tuple[EpisodeOutcome, ...],
    *,
    arm_id: str = 'qwen3-8b-vllm',
    cancelled_ids: tuple[str, ...] = (),
    abort: ArmAbort | None = None,
) -> ArmRunResult:
    return ArmRunResult(arm_id=arm_id, outcomes=outcomes, cancelled_ids=cancelled_ids, abort=abort)


def _by_id(records: tuple[MetricsRecord, ...]) -> dict[str, MetricsRecord]:
    ids = [record.metric.metric_id for record in records]
    assert len(ids) == len(set(ids)), f'duplicate metric ids: {ids}'
    return {record.metric.metric_id: record for record in records}


# --- episode-failure-rate ---------------------------------------------------------


def test_failure_rate_is_failed_over_attempted_and_higher_is_worse():
    metric = episode_failure_rate_metric(
        _result((_ok('e1'), _failed('e2'), _ok('e3'), _failed('e4')))
    )

    assert metric is not None
    assert metric.metric_id == LlmMetricId.EPISODE_FAILURE_RATE
    assert metric.kind == 'proportion'
    assert metric.value == pytest.approx(0.5)
    assert (metric.n, metric.denominator) == (4, 4)
    assert metric.direction == 'higher_is_worse'


def test_failure_rate_denominator_excludes_cancelled_ids_of_an_aborted_run():
    failures = tuple(_failed(f'f{i}') for i in range(5))
    abort = ArmAbort(
        arm_id='qwen3-8b-vllm',
        item_ids=tuple(o.episode_id for o in failures),
        error_classes=('RuntimeError',) * 5,
    )
    result = _result((_ok('e0'), *failures), cancelled_ids=('c1', 'c2'), abort=abort)

    metric = episode_failure_rate_metric(result)

    assert metric is not None
    assert (metric.n, metric.denominator) == (6, 6)
    assert metric.value == pytest.approx(5 / 6)


def test_failure_rate_is_absent_with_nothing_attempted():
    assert episode_failure_rate_metric(_result(())) is None


# --- latency ----------------------------------------------------------------------


def test_latency_is_nearest_rank_over_ok_outcomes_in_milliseconds():
    outcomes = tuple(
        _ok(f'e{i}', duration_ms=ms) for i, ms in enumerate([40.0, 10.0, 100.0, 30.0, 20.0])
    )

    p50 = latency_metric(LlmMetricId.EPISODE_LATENCY_P50, outcomes)
    p95 = latency_metric(LlmMetricId.EPISODE_LATENCY_P95, outcomes)

    assert p50 is not None and p95 is not None
    assert (p50.metric_id, p50.kind, p50.value, p50.n) == ('episode-latency-p50', 'scalar', 30.0, 5)
    assert (p95.metric_id, p95.kind, p95.value, p95.n) == ('episode-latency-p95', 'scalar', 100.0, 5)


def test_latency_ignores_failed_outcomes():
    outcomes = (_ok('e1', duration_ms=10.0), _failed('e2', duration_ms=9999.0))

    p95 = latency_metric(LlmMetricId.EPISODE_LATENCY_P95, outcomes)

    assert p95 is not None
    assert (p95.value, p95.n) == (10.0, 1)


def test_latency_is_absent_without_ok_outcomes():
    assert latency_metric(LlmMetricId.EPISODE_LATENCY_P50, (_failed('e1'),)) is None


def test_latency_refuses_a_non_latency_metric_id():
    with pytest.raises(ValueError, match='tokens-per-episode'):
        latency_metric(LlmMetricId.TOKENS_PER_EPISODE, (_ok('e1'),))


# --- tokens and cost ---------------------------------------------------------------


def test_tokens_per_episode_is_the_mean_total_over_ok_outcomes():
    outcomes = (
        _ok('e1', tokens=_usage(30, 10)),
        _ok('e2', tokens=_usage(50, 30)),
        _failed('e3'),
    )

    metric = tokens_per_episode_metric(outcomes)

    assert metric is not None
    assert (metric.metric_id, metric.kind, metric.n) == ('tokens-per-episode', 'scalar', 2)
    assert metric.value == pytest.approx(60.0)


def test_an_ok_outcome_without_tokens_raises_naming_the_episodes():
    outcomes = (_ok('e1'), _no_tokens('e2'), _no_tokens('e3'))

    with pytest.raises(TokenAccountingError, match=r"e2.*e3") as raised:
        tokens_per_episode_metric(outcomes)

    assert raised.value.episode_ids == ('e2', 'e3')


def test_tokens_per_episode_is_absent_without_ok_outcomes():
    assert tokens_per_episode_metric((_failed('e1'),)) is None


def test_usd_per_episode_prices_input_and_output_for_a_metered_arm():
    spec = incumbent_control_spec()  # 0.4 / 1.6 usd per million tokens
    outcomes = (
        _ok('e1', tokens=_usage(1_000_000, 0)),
        _ok('e2', tokens=_usage(0, 500_000)),
        _failed('e3'),
    )

    metric = usd_per_episode_metric(spec, outcomes)

    assert metric is not None
    assert (metric.metric_id, metric.kind, metric.n) == ('usd-per-episode', 'scalar', 2)
    assert metric.value == pytest.approx((0.4 + 0.8) / 2)


def test_usd_per_episode_is_exactly_zero_for_a_local_arm():
    metric = usd_per_episode_metric(llm_spec(), (_ok('e1', tokens=_usage(10**6, 10**6)),))

    assert metric is not None
    assert metric.value == 0.0
    assert metric.n == 1


def test_usd_per_episode_never_prices_a_missing_token_count_as_free():
    with pytest.raises(TokenAccountingError):
        usd_per_episode_metric(incumbent_control_spec(), (_ok('e1'), _no_tokens('e2')))


# --- graph sameness ----------------------------------------------------------------


def test_graph_sameness_is_mean_entity_jaccard_over_episodes_ok_on_both_sides():
    arm = (
        _ok('e1', entities=('alice', 'bob')),
        _ok('e2', entities=('carol',)),
        _ok('e3', entities=('dave',)),
        _failed('e4'),
    )
    reference = (
        _ok('e1', entities=('alice', 'eve')),
        _ok('e2', entities=('carol',)),
        _failed('e3'),
        _ok('e4', entities=('frank',)),
    )

    metric = graph_sameness_metric(arm, reference)

    assert metric is not None
    assert (metric.metric_id, metric.kind, metric.n) == ('graph-sameness', 'scalar', 2)
    assert metric.value == pytest.approx((1 / 3 + 1.0) / 2)
    assert metric.details_path == GRAPH_SAMENESS_DETAILS_FILENAME == 'graph_sameness_details.json'


def test_graph_sameness_counts_two_empty_entity_sets_as_identical():
    metric = graph_sameness_metric((_ok('e1'),), (_ok('e1'),))

    assert metric is not None
    assert metric.value == 1.0


def test_graph_sameness_compares_normalised_entity_names():
    metric = graph_sameness_metric(
        (_ok('e1', entities=('Alice   Smith',)),), (_ok('e1', entities=('alice smith',)),)
    )

    assert metric is not None
    assert metric.value == 1.0


def test_graph_sameness_is_absent_when_no_episode_is_ok_on_both_sides():
    assert graph_sameness_metric((_ok('e1'),), (_failed('e1'),)) is None


def test_graph_sameness_details_pin_per_episode_counts_jaccards_and_exclusions():
    arm = (
        _ok(
            'e1',
            entities=('alice', 'bob'),
            edges=(('alice', 'KNOWS', 'bob'), ('bob', 'LIKES', 'alice')),
        ),
        _ok('e2'),
        _failed('e3'),
    )
    reference = (
        _ok('e1', entities=('alice', 'eve'), edges=(('alice', 'KNOWS', 'bob'),)),
        _failed('e2'),
        _ok('e3', entities=('carol',)),
        _ok('e4'),
    )

    details = graph_sameness_details(arm, reference)

    assert details.excluded_arm_ids == ('e2', 'e3')
    assert details.excluded_reference_ids == ('e2', 'e3', 'e4')
    (episode,) = details.episodes
    assert episode.episode_id == 'e1'
    assert (episode.arm_entity_count, episode.ref_entity_count) == (2, 2)
    assert (episode.arm_edge_count, episode.ref_edge_count) == (2, 1)
    assert episode.entity_jaccard == pytest.approx(1 / 3)
    assert episode.edge_triple_jaccard == pytest.approx(1 / 2)


def test_graph_sameness_details_render_as_canonical_json():
    details = graph_sameness_details((_ok('e1', entities=('a',)),), (_ok('e1', entities=('a',)),))

    text = canonical_json_text(details.model_dump(mode='json'))

    assert '"entity_jaccard": 1.0' in text
    assert '"excluded_arm_ids": []' in text


# --- the composer ------------------------------------------------------------------


def _complete_result() -> ArmRunResult:
    return _result((
        _ok('e1', duration_ms=10.0, tokens=_usage(30, 10), entities=('alice',)),
        _ok('e2', duration_ms=20.0, tokens=_usage(50, 30), entities=('bob',)),
        _failed('e3'),
    ))


def test_llm_axis_records_compose_every_applicable_metric():
    reference = (_ok('e1', entities=('alice',)), _ok('e2', entities=('carol',)))

    records = _by_id(llm_axis_records(
        llm_spec(),
        _complete_result(),
        CONFORMANCE,
        reference=reference,
        retrieval_ranks=(1, 12),
        measured_at=MEASURED_AT,
    ))

    assert set(records) == {str(metric_id) for metric_id in LlmMetricId}
    assert records['episode-failure-rate'].metric.value == pytest.approx(1 / 3)
    assert records['conformance-rate'].metric.value == pytest.approx(9 / 10)
    assert records['episode-latency-p50'].metric.value == 10.0
    assert records['episode-latency-p95'].metric.value == 20.0
    assert records['tokens-per-episode'].metric.value == pytest.approx(60.0)
    assert records['usd-per-episode'].metric.value == 0.0
    assert records['graph-sameness'].metric.value == pytest.approx(0.5)
    retrieval = records['retrieval-utility'].metric
    assert (retrieval.kind, retrieval.value, retrieval.n) == ('proportion', 0.5, 2)


def test_retrieval_utility_is_recall_at_the_harness_k():
    assert RETRIEVAL_UTILITY_K == 10
    records = _by_id(llm_axis_records(
        llm_spec(),
        _complete_result(),
        CONFORMANCE,
        reference=None,
        retrieval_ranks=(10, 11),
        measured_at=MEASURED_AT,
    ))

    assert records['retrieval-utility'].metric.value == 0.5


def test_reference_and_ranks_absent_leave_their_records_absent_not_null():
    records = _by_id(llm_axis_records(
        llm_spec(),
        _complete_result(),
        CONFORMANCE,
        reference=None,
        retrieval_ranks=None,
        measured_at=MEASURED_AT,
    ))

    assert 'graph-sameness' not in records
    assert 'retrieval-utility' not in records
    assert 'tokens-per-episode' in records


@pytest.mark.parametrize(
    ('spec', 'preregistration_sha'),
    [(llm_spec(), PREREG_SHA), (incumbent_control_spec(), None)],
    ids=['candidate', 'control'],
)
def test_every_record_carries_the_spec_identity(spec, preregistration_sha):
    records = llm_axis_records(
        spec,
        _result(_complete_result().outcomes, arm_id=spec.arm_id),
        CONFORMANCE,
        reference=None,
        retrieval_ranks=None,
        measured_at=MEASURED_AT,
    )

    assert records
    for record in records:
        assert MetricsRecord.model_validate(record.model_dump()) == record
        assert (record.arm_id, record.arm_role) == (spec.arm_id, spec.arm_role)
        assert (record.code_sha, record.corpus_sha) == (CODE_SHA, CORPUS_SHA)
        assert record.preregistration_sha == preregistration_sha
        assert record.measured_at == MEASURED_AT
        assert record.incomplete is False


def test_records_of_an_aborted_run_are_marked_incomplete():
    failures = tuple(_failed(f'f{i}') for i in range(5))
    abort = ArmAbort(
        arm_id='qwen3-8b-vllm',
        item_ids=tuple(o.episode_id for o in failures),
        error_classes=('RuntimeError',) * 5,
    )
    result = _result((_ok('e0'), *failures), cancelled_ids=('c1',), abort=abort)

    records = llm_axis_records(
        llm_spec(), result, CONFORMANCE, reference=None, retrieval_ranks=None,
        measured_at=MEASURED_AT,
    )

    assert records
    assert all(record.incomplete for record in records)


def test_the_composer_surfaces_a_token_accounting_failure():
    result = _result((_ok('e1'), _no_tokens('e2')))

    with pytest.raises(TokenAccountingError):
        llm_axis_records(
            llm_spec(), result, CONFORMANCE, reference=None, retrieval_ranks=None,
            measured_at=MEASURED_AT,
        )


def test_an_embedding_spec_is_refused():
    with pytest.raises(TypeError, match='embedding'):
        llm_axis_records(
            embedding_spec(),  # type: ignore[arg-type]
            _complete_result(),
            CONFORMANCE,
            reference=None,
            retrieval_ranks=None,
            measured_at=MEASURED_AT,
        )
