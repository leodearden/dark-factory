"""MetricsRecord: one artifact per (arm, metric), wrapping the imported M1 Metric."""

import json
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError
from shared.memory_eval_metrics import Metric, canonical_json_text

from arm_harness._fakes import (
    CODE_SHA,
    CORPUS_SHA,
    PREREG_SHA,
    embedding_spec,
    incumbent_control_spec,
    llm_spec,
)
from fused_memory.arm_harness.metrics_record import (
    EMBEDDING_METRIC_IDS,
    LLM_METRIC_IDS,
    DeltaOf,
    IndexConfiguration,
    MetricsRecord,
    load_metrics_record,
    load_metrics_records,
    record_for,
    serialize_metrics_record,
    write_metrics_record,
)

MEASURED_AT = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)


def _proportion(metric_id: str = 'conformance-rate') -> Metric:
    return Metric(
        metric_id=metric_id,
        kind='proportion',
        value=0.75,
        n=4,
        denominator=4,
        direction='lower_is_worse',
    )


def _scalar(metric_id: str = 'episode-latency-p50', value: float = 30.0) -> Metric:
    return Metric(metric_id=metric_id, kind='scalar', value=value, n=5)


def _record_data(**overrides) -> dict:
    data = {
        'schema_version': 1,
        'arm_id': 'qwen3-8b-vllm',
        'axis': 'llm',
        'arm_role': 'candidate',
        'measured_at': MEASURED_AT,
        'code_sha': CODE_SHA,
        'corpus_sha': CORPUS_SHA,
        'preregistration_sha': PREREG_SHA,
        'incomplete': False,
        'metric': _scalar(),
    }
    return data | overrides


def test_metric_id_vocabularies_are_closed_frozensets():
    assert isinstance(LLM_METRIC_IDS, frozenset)
    assert isinstance(EMBEDDING_METRIC_IDS, frozenset)
    assert {
        'conformance-rate',
        'episode-failure-rate',
        'episode-latency-p50',
        'episode-latency-p95',
        'graph-sameness',
        'retrieval-utility',
        'tokens-per-episode',
        'usd-per-episode',
    } == LLM_METRIC_IDS
    assert {
        'known-item-recall@5',
        'known-item-recall@10',
        'mrr',
        'query-embed-latency-p95',
        'reembed-throughput',
    } == EMBEDDING_METRIC_IDS


def test_index_configuration_values():
    assert IndexConfiguration.WITH_INDICES.value == 'with-indices'
    assert IndexConfiguration.EMBEDDING_ONLY.value == 'embedding-only'


def test_the_metric_schema_is_imported_not_restated():
    assert MetricsRecord.model_fields['metric'].annotation is Metric


@pytest.mark.parametrize(
    'metric',
    [
        {'metric_id': 'conformance-rate', 'kind': 'proportion', 'value': 0.5, 'n': 2,
         'direction': 'lower_is_worse'},
        {'metric_id': 'conformance-rate', 'kind': 'proportion', 'value': 0.5, 'n': 2,
         'denominator': 2},
        {'metric_id': 'episode-latency-p50', 'kind': 'scalar', 'value': 1.0, 'n': 1,
         'direction': 'higher_is_worse'},
    ],
)
def test_m1_invalid_metrics_are_rejected_by_m1s_own_validator(metric):
    with pytest.raises(ValidationError):
        MetricsRecord.model_validate(_record_data(metric=metric))


def test_record_for_copies_identity_and_shas_from_the_spec():
    spec = llm_spec()

    record = record_for(spec, _proportion(), measured_at=MEASURED_AT, incomplete=False)

    assert record.schema_version == 1
    assert record.arm_id == spec.arm_id
    assert record.axis == 'llm'
    assert record.arm_role == spec.arm_role
    assert record.code_sha == spec.code_sha
    assert record.corpus_sha == spec.corpus_sha
    assert record.preregistration_sha == spec.preregistration_sha
    assert record.index_configuration is None
    assert record.delta_of is None


def test_candidate_without_preregistration_sha_is_invalid_by_schema():
    with pytest.raises(ValidationError, match='preregistration_sha'):
        MetricsRecord.model_validate(_record_data(preregistration_sha=None))


def test_control_with_preregistration_sha_is_invalid_by_schema():
    with pytest.raises(ValidationError, match='preregistration_sha'):
        MetricsRecord.model_validate(_record_data(arm_role='control'))


@pytest.mark.parametrize(
    ('axis', 'metric'),
    [('llm', _scalar('mrr')), ('embedding', _scalar('episode-latency-p50'))],
)
def test_metric_outside_the_axis_vocabulary_is_rejected(axis, metric):
    with pytest.raises(ValidationError, match=metric.metric_id):
        MetricsRecord.model_validate(_record_data(axis=axis, metric=metric))


@pytest.mark.parametrize('metric_id', ['known-item-recall@5', 'known-item-recall@10', 'mrr'])
def test_per_index_configuration_metrics_require_one(metric_id):
    data = _record_data(axis='embedding', metric=_scalar(metric_id, 0.5))

    with pytest.raises(ValidationError, match='index_configuration'):
        MetricsRecord.model_validate(data)
    record = MetricsRecord.model_validate(
        data | {'index_configuration': IndexConfiguration.EMBEDDING_ONLY}
    )
    assert record.index_configuration is IndexConfiguration.EMBEDDING_ONLY


@pytest.mark.parametrize(
    ('axis', 'metric_id'),
    [('llm', 'retrieval-utility'), ('embedding', 'query-embed-latency-p95')],
)
def test_other_metrics_forbid_an_index_configuration(axis, metric_id):
    data = _record_data(
        axis=axis,
        metric=_scalar(metric_id),
        index_configuration=IndexConfiguration.WITH_INDICES,
    )

    with pytest.raises(ValidationError, match='index_configuration'):
        MetricsRecord.model_validate(data)


def test_delta_of_an_arm_with_itself_is_rejected():
    with pytest.raises(ValidationError, match='qwen3-8b-vllm'):
        DeltaOf(minuend_arm_id='qwen3-8b-vllm', subtrahend_arm_id='qwen3-8b-vllm')


def test_naive_measured_at_is_rejected():
    with pytest.raises(ValidationError, match='measured_at'):
        MetricsRecord.model_validate(_record_data(measured_at=datetime(2026, 10, 5, 12, 0)))


def test_measured_at_is_normalised_to_utc():
    offset = datetime(2026, 10, 5, 14, 0, tzinfo=timezone(timedelta(hours=2)))

    record = MetricsRecord.model_validate(_record_data(measured_at=offset))

    assert record.measured_at == MEASURED_AT
    assert record.measured_at.utcoffset() == timedelta(0)


def test_records_are_frozen():
    record = MetricsRecord.model_validate(_record_data())
    with pytest.raises(ValidationError):
        record.incomplete = True  # type: ignore[misc]


def test_serialization_follows_the_m1_null_convention():
    control = record_for(
        incumbent_control_spec(), _scalar(), measured_at=MEASURED_AT, incomplete=False
    )

    text = serialize_metrics_record(control)
    payload = json.loads(text)

    assert text == canonical_json_text(payload)
    assert 'preregistration_sha' in payload
    assert payload['preregistration_sha'] is None
    assert 'index_configuration' not in payload
    assert 'delta_of' not in payload
    assert payload['incomplete'] is False
    assert 'denominator' not in payload['metric']


def test_serialization_emits_structured_index_configuration_and_delta():
    record = MetricsRecord.model_validate(
        _record_data(
            axis='embedding',
            metric=_scalar('mrr', 0.4),
            index_configuration=IndexConfiguration.WITH_INDICES,
            delta_of=DeltaOf(minuend_arm_id='bge-m3', subtrahend_arm_id='incumbent'),
        )
    )

    payload = json.loads(serialize_metrics_record(record))

    assert payload['index_configuration'] == 'with-indices'
    assert payload['delta_of'] == {'minuend_arm_id': 'bge-m3', 'subtrahend_arm_id': 'incumbent'}
    assert payload['metric']['metric_id'] == 'mrr'


def test_write_places_one_file_per_metric_and_round_trips(tmp_path):
    record = record_for(llm_spec(), _proportion(), measured_at=MEASURED_AT, incomplete=True)

    path = write_metrics_record(record, tmp_path)

    assert path == tmp_path / 'metrics' / 'conformance-rate.json'
    assert path.read_text() == serialize_metrics_record(record)
    assert load_metrics_record(path) == record


def test_write_names_per_index_configuration_files_by_configuration(tmp_path):
    record = record_for(
        embedding_spec(),
        _scalar('known-item-recall@5', 0.5),
        measured_at=MEASURED_AT,
        incomplete=False,
        index_configuration=IndexConfiguration.EMBEDDING_ONLY,
    )

    path = write_metrics_record(record, tmp_path)

    assert path == tmp_path / 'metrics' / 'known-item-recall@5.embedding-only.json'
    assert load_metrics_record(path) == record


def test_write_validates_before_creating_anything(tmp_path):
    malformed = MetricsRecord.model_construct(**_record_data(preregistration_sha=None))

    with pytest.raises(ValidationError):
        write_metrics_record(malformed, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_load_metrics_records_reads_back_every_record_a_run_dir_holds(tmp_path):
    spec = embedding_spec()
    written = [
        record_for(spec, _scalar('mrr', 0.4), measured_at=MEASURED_AT, incomplete=False,
                   index_configuration=configuration)
        for configuration in IndexConfiguration
    ] + [record_for(spec, _scalar('reembed-throughput', 9.0), measured_at=MEASURED_AT,
                    incomplete=False)]
    for record in written:
        write_metrics_record(record, tmp_path)

    loaded = load_metrics_records(tmp_path)

    assert sorted(loaded, key=serialize_metrics_record) == sorted(
        written, key=serialize_metrics_record
    )


def test_load_metrics_records_of_a_run_dir_without_metrics_is_empty(tmp_path):
    assert load_metrics_records(tmp_path) == ()
