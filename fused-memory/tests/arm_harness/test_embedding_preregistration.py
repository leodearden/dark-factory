"""The embedding axis's pre-registration and a candidate's comparison (arm_harness/embedding_preregistration.py)."""

from collections.abc import Sequence

import pytest
from fused_memory.arm_harness.embedding_preregistration import (
    EmbeddingPreregistrationError,
    QueryLatencyEnvelope,
    check_embedding_run_symmetry,
    compare_embedding_arm,
    derive_embedding_preregistration_inputs,
    load_embedding_preregistration_inputs,
    query_latency_envelope,
    serialize_embedding_preregistration_inputs,
)
from pydantic import ValidationError

from arm_harness._fakes import (
    CODE_SHA,
    CORPUS_SHA,
    PREREG_SHA,
    embedding_control,
    embedding_records,
    embedding_run_manifest,
    embedding_spec,
)
from fused_memory.arm_harness.embedding_run_manifest import EmbeddingRunManifest
from fused_memory.arm_harness.instrument_checks import InstrumentCheckId
from fused_memory.arm_harness.margins import derive_margins
from fused_memory.arm_harness.metrics_record import (
    EmbeddingMetricId,
    IndexConfiguration,
    MetricsRecord,
)
from fused_memory.arm_harness.preregistration import LATENCY_HEADROOM

CONTROL_A = embedding_control('incumbent-embed-a')
CONTROL_B = embedding_control('incumbent-embed-b', scratch_group_id='evalmem_lme_emb_ctl_b')
CANDIDATE = embedding_spec(
    arm_id='granite-embedding-english-r2',
    model_id='granite-embedding-english-r2',
    embedding_dim=768,
    scratch_group_id='evalmem_lme_emb_granite_embedding_english_r2',
)

WITH_INDICES = IndexConfiguration.WITH_INDICES
EMBEDDING_ONLY = IndexConfiguration.EMBEDDING_ONLY


def _settings(**overrides) -> dict[str, object]:
    settings = embedding_run_manifest(CONTROL_A).settings.model_dump()
    return settings | overrides


def _pair(
    *,
    run_a: EmbeddingRunManifest | None = None,
    run_b: EmbeddingRunManifest | None = None,
    records_a: Sequence[MetricsRecord] | None = None,
    records_b: Sequence[MetricsRecord] | None = None,
):
    return (
        run_a or embedding_run_manifest(CONTROL_A),
        records_a if records_a is not None else embedding_records(CONTROL_A),
        run_b or embedding_run_manifest(CONTROL_B),
        records_b
        if records_b is not None
        else embedding_records(
            CONTROL_B, with_indices=(0.78, 0.89, 0.68), embedding_only=(0.66, 0.84, 0.58),
            latency_p95_ms=140.0,
        ),
    )


def _inputs():
    return derive_embedding_preregistration_inputs(*_pair())


# --- symmetry -------------------------------------------------------------------------


def test_symmetry_passes_when_settings_and_shas_agree_whatever_the_embedder():
    runs = [embedding_run_manifest(CONTROL_A), embedding_run_manifest(CANDIDATE)]

    result = check_embedding_run_symmetry(runs)

    assert result.check_id is InstrumentCheckId.ARM_CONFIG_SYMMETRY
    assert result.passed


def test_symmetry_names_each_differing_field_and_every_arms_value():
    runs = [
        embedding_run_manifest(CONTROL_A),
        embedding_run_manifest(
            embedding_spec(arm_id='gte-modernbert-base', code_sha='d' * 40),
            settings=_settings(embed_batch_size=32),
        ),
    ]

    result = check_embedding_run_symmetry(runs)

    assert not result.passed
    assert set(result.offenders) == {'settings.embed_batch_size', 'code_sha'}
    assert 'incumbent-embed-a=64' in result.detail
    assert 'gte-modernbert-base=32' in result.detail


def test_symmetry_refuses_a_single_run_or_a_repeated_arm():
    with pytest.raises(EmbeddingPreregistrationError, match='two'):
        check_embedding_run_symmetry([embedding_run_manifest(CONTROL_A)])
    with pytest.raises(EmbeddingPreregistrationError, match='incumbent-embed-a'):
        check_embedding_run_symmetry(
            [embedding_run_manifest(CONTROL_A), embedding_run_manifest(CONTROL_A)]
        )


# --- the query-latency envelope ---------------------------------------------------------


def test_the_envelope_is_the_search_timeout_over_the_imported_headroom():
    envelope = query_latency_envelope(30.0)

    assert envelope.headroom == LATENCY_HEADROOM
    assert envelope.p95_bound_ms == 15000.0


def test_the_envelope_admits_strictly_inside_its_bound():
    envelope = query_latency_envelope(30.0)

    assert envelope.admits(14999.9)
    assert not envelope.admits(15000.0)
    with pytest.raises(ValueError, match='nan'):
        envelope.admits(float('nan'))


def test_an_envelope_whose_bound_is_not_derived_is_refused():
    with pytest.raises(ValidationError, match='p95_bound_ms'):
        QueryLatencyEnvelope(search_timeout_s=30.0, headroom=2.0, p95_bound_ms=14000.0)


# --- derivation -------------------------------------------------------------------------


def test_margins_are_exactly_derive_margins_with_no_episode_values():
    run_a, records_a, run_b, records_b = _pair()

    inputs = derive_embedding_preregistration_inputs(run_a, records_a, run_b, records_b)

    assert inputs.margins == derive_margins(records_a, records_b, episode_values={})
    assert len(inputs.margins) == 6
    assert {(m.metric_id, m.index_configuration) for m in inputs.margins} == {
        (metric_id, configuration)
        for metric_id in (
            EmbeddingMetricId.KNOWN_ITEM_RECALL_AT_5,
            EmbeddingMetricId.KNOWN_ITEM_RECALL_AT_10,
            EmbeddingMetricId.MRR,
        )
        for configuration in IndexConfiguration
    }


def test_the_inputs_carry_the_pair_its_shas_and_the_slower_controls_p95():
    inputs = _inputs()

    assert inputs.schema_version == 1
    assert inputs.control_arm_ids == ('incumbent-embed-a', 'incumbent-embed-b')
    assert (inputs.code_sha, inputs.corpus_sha) == (CODE_SHA, CORPUS_SHA)
    assert inputs.incumbent_query_latency_p95_ms == 140.0
    assert inputs.query_latency_envelope == query_latency_envelope(30.0)


def test_the_envelope_is_anchored_on_the_runs_recorded_search_timeout():
    run_a = embedding_run_manifest(CONTROL_A, settings=_settings(search_timeout_s=40.0))
    run_b = embedding_run_manifest(CONTROL_B, settings=_settings(search_timeout_s=40.0))

    inputs = derive_embedding_preregistration_inputs(*_pair(run_a=run_a, run_b=run_b))

    assert inputs.query_latency_envelope.p95_bound_ms == 20000.0


def test_controls_with_different_search_timeouts_are_refused():
    run_b = embedding_run_manifest(CONTROL_B, settings=_settings(search_timeout_s=15.0))

    with pytest.raises(EmbeddingPreregistrationError, match='search_timeout_s'):
        derive_embedding_preregistration_inputs(*_pair(run_b=run_b))


def test_a_candidate_run_given_as_a_control_is_refused():
    run_b = embedding_run_manifest(CANDIDATE)

    with pytest.raises(EmbeddingPreregistrationError, match='control'):
        derive_embedding_preregistration_inputs(
            *_pair(run_b=run_b, records_b=embedding_records(CANDIDATE))
        )


def test_one_control_run_twice_is_refused():
    with pytest.raises(EmbeddingPreregistrationError, match='incumbent-embed-a'):
        derive_embedding_preregistration_inputs(
            *_pair(run_b=embedding_run_manifest(CONTROL_A), records_b=embedding_records(CONTROL_A))
        )


def test_asymmetric_controls_are_refused():
    run_b = embedding_run_manifest(CONTROL_B, settings=_settings(embed_concurrency=8))

    with pytest.raises(EmbeddingPreregistrationError, match='settings.embed_concurrency'):
        derive_embedding_preregistration_inputs(*_pair(run_b=run_b))


def test_records_of_another_arm_than_their_run_are_refused():
    with pytest.raises(EmbeddingPreregistrationError, match='incumbent-embed-b'):
        derive_embedding_preregistration_inputs(
            *_pair(records_a=embedding_records(CONTROL_B))
        )


def test_an_incumbent_p95_outside_its_own_envelope_is_refused():
    records_b = embedding_records(CONTROL_B, latency_p95_ms=15000.0)

    with pytest.raises(EmbeddingPreregistrationError, match='envelope'):
        derive_embedding_preregistration_inputs(*_pair(records_b=records_b))


def test_serialization_is_canonical_and_round_trips(tmp_path):
    inputs = _inputs()
    path = tmp_path / 'embedding-preregistration-inputs.json'

    text = serialize_embedding_preregistration_inputs(inputs)
    path.write_text(text)

    assert serialize_embedding_preregistration_inputs(_inputs()) == text
    assert load_embedding_preregistration_inputs(path) == inputs


# --- comparison --------------------------------------------------------------------------


def _candidate(**record_overrides):
    run = embedding_run_manifest(CANDIDATE)
    return run, embedding_records(CANDIDATE, **record_overrides)


def test_a_candidate_inside_every_margin_and_the_envelope_is_non_inferior():
    comparison = compare_embedding_arm(_inputs(), *_candidate())

    assert comparison.arm_id == CANDIDATE.arm_id
    assert len(comparison.margins) == 6
    assert all(row.admits for row in comparison.margins)
    assert comparison.envelope.admits
    assert comparison.non_inferior


def test_each_margin_row_carries_the_candidate_value_and_the_margin_verdict():
    inputs = _inputs()

    comparison = compare_embedding_arm(inputs, *_candidate(with_indices=(0.5, 0.9, 0.7)))

    row = next(
        row for row in comparison.margins
        if (row.metric_id, row.index_configuration)
        == (EmbeddingMetricId.KNOWN_ITEM_RECALL_AT_5, WITH_INDICES)
    )
    entry = next(
        entry for entry in inputs.margins
        if (entry.metric_id, entry.index_configuration)
        == (EmbeddingMetricId.KNOWN_ITEM_RECALL_AT_5, WITH_INDICES)
    )
    assert row.candidate_value == 0.5
    assert (row.reference_value, row.margin) == (entry.reference_value, entry.margin)
    assert row.admits is entry.admits(0.5) is False
    assert not comparison.non_inferior


def test_an_embedding_only_shortfall_is_reported_but_does_not_decide():
    comparison = compare_embedding_arm(_inputs(), *_candidate(embedding_only=(0.1, 0.1, 0.1)))

    failing = [row for row in comparison.margins if not row.admits]
    assert failing
    assert {row.index_configuration for row in failing} == {EMBEDDING_ONLY}
    assert comparison.non_inferior


def test_a_candidate_outside_the_envelope_is_not_non_inferior():
    comparison = compare_embedding_arm(_inputs(), *_candidate(latency_p95_ms=15000.0))

    assert all(row.admits for row in comparison.margins)
    assert not comparison.envelope.admits
    assert comparison.envelope.candidate_p95_ms == 15000.0
    assert not comparison.non_inferior


def test_mem0_and_throughput_rows_are_reported():
    comparison = compare_embedding_arm(
        _inputs(), *_candidate(mem0=(0.4, 0.6, 0.3), throughput=55.0)
    )

    assert {(row.metric_id, row.value) for row in comparison.reported} == {
        (EmbeddingMetricId.MEM0_KNOWN_ITEM_RECALL_AT_5, 0.4),
        (EmbeddingMetricId.MEM0_KNOWN_ITEM_RECALL_AT_10, 0.6),
        (EmbeddingMetricId.MEM0_MRR, 0.3),
        (EmbeddingMetricId.REEMBED_THROUGHPUT, 55.0),
    }


def test_a_control_given_as_the_candidate_is_refused():
    with pytest.raises(EmbeddingPreregistrationError, match='control'):
        compare_embedding_arm(
            _inputs(), embedding_run_manifest(CONTROL_A), embedding_records(CONTROL_A)
        )


@pytest.mark.parametrize('field', ['code_sha', 'corpus_sha'])
def test_a_candidate_on_another_code_or_corpus_sha_is_refused(field):
    spec = embedding_spec(**{
        'arm_id': 'granite-embedding-english-r2',
        'preregistration_sha': PREREG_SHA,
        field: 'e' * (40 if field == 'code_sha' else 64),
    })

    with pytest.raises(EmbeddingPreregistrationError, match=field):
        compare_embedding_arm(_inputs(), embedding_run_manifest(spec), embedding_records(spec))


def test_a_candidate_missing_a_gated_record_is_refused():
    run, records = _candidate()
    kept = [
        record for record in records
        if (record.metric.metric_id, record.index_configuration)
        != (EmbeddingMetricId.MRR, WITH_INDICES)
    ]

    with pytest.raises(EmbeddingPreregistrationError, match='mrr'):
        compare_embedding_arm(_inputs(), run, kept)
