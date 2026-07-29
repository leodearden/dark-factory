"""Tests for shared.memory_eval_metrics — the M1 metric-series schema + writer.

Built bottom-up in TDD order, following shared/tests/test_capability_manifest.py's
convention (see docs/prds/memory-eval-program.md §3 M1):
  - TestMetricSeriesSchema: the happy-path shape and round-trip in isolation.
  - TestMetricRejectedAtEmit: the per-kind rejection matrix, raised AT EMIT.
  - TestArtifactWriter: the STAMP artifact layout, atomicity and window loader.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.memory_eval_metrics import (
    SCHEMA_VERSION,
    Corpus,
    Metric,
    MetricSchemaError,
    MetricSeries,
    TripwireItem,
    parse_metric_series,
    validate_metric_series,
)


def _make_metric(**overrides) -> dict:
    """A minimal valid ``proportion`` metric dict; override any field."""
    payload = {
        'metric_id': 'canonical-in-top-5',
        'kind': 'proportion',
        'value': 0.8,
        'n': 30,
        'denominator': 30,
    }
    payload.update(overrides)
    return payload


def _make_tripwire(**overrides) -> dict:
    payload = {
        'metric_id': 'topic-canonical-present',
        'kind': 'tripwire',
        'value': 1.0,
        'n': 2,
        'items': [
            {'item_key': 't-alpha', 'passed': True},
            {'item_key': 't-beta', 'passed': False},
        ],
    }
    payload.update(overrides)
    return payload


def _make_count(**overrides) -> dict:
    payload = {'metric_id': 'dangling-pointers', 'kind': 'count', 'value': 5.0, 'n': 30}
    payload.update(overrides)
    return payload


def _make_scalar(**overrides) -> dict:
    payload = {'metric_id': 'search-latency-p50-ms', 'kind': 'scalar', 'value': 41.5, 'n': 30}
    payload.update(overrides)
    return payload


def _make_series(**overrides) -> dict:
    payload = {
        'schema_version': SCHEMA_VERSION,
        'eval_id': 'e1-retrieval-health',
        'run_stamp': '20260701T031500Z',
        'corpus': {'project_id': 'dark_factory', 'counts': {'entities_and_relations': 1204}},
        'metrics': [_make_metric(), _make_count(), _make_tripwire(), _make_scalar()],
    }
    payload.update(overrides)
    return payload


class TestMetricSeriesSchema:
    def test_schema_version_is_one(self):
        assert SCHEMA_VERSION == 1

    def test_series_with_every_kind_parses(self):
        series = parse_metric_series(_make_series())
        assert isinstance(series, MetricSeries)
        assert series.eval_id == 'e1-retrieval-health'
        assert series.run_stamp == '20260701T031500Z'
        assert [m.kind for m in series.metrics] == ['proportion', 'count', 'tripwire', 'scalar']

    def test_round_trip_is_lossless(self):
        payload = _make_series()
        series = parse_metric_series(payload)
        assert series.model_dump(mode='json', exclude_none=True) == payload

    def test_corpus_carries_project_id_and_counts(self):
        series = parse_metric_series(_make_series())
        assert isinstance(series.corpus, Corpus)
        assert series.corpus.project_id == 'dark_factory'
        assert series.corpus.counts == {'entities_and_relations': 1204}

    def test_tripwire_items_parse_as_models(self):
        series = parse_metric_series(_make_series())
        tripwire = next(m for m in series.metrics if m.kind == 'tripwire')
        assert tripwire.items is not None
        assert all(isinstance(i, TripwireItem) for i in tripwire.items)
        assert [(i.item_key, i.passed) for i in tripwire.items] == [
            ('t-alpha', True),
            ('t-beta', False),
        ]

    def test_optional_fields_are_genuinely_optional(self):
        # A count metric carries none of denominator / items / details_path.
        series = parse_metric_series(_make_series(metrics=[_make_count()]))
        metric = series.metrics[0]
        assert metric.denominator is None
        assert metric.items is None
        assert metric.details_path is None

    def test_details_path_round_trips_when_present(self):
        payload = _make_series(metrics=[_make_count(details_path='report-20260701T031500Z.txt')])
        series = parse_metric_series(payload)
        assert series.metrics[0].details_path == 'report-20260701T031500Z.txt'
        assert series.model_dump(mode='json', exclude_none=True) == payload

    def test_models_are_frozen(self):
        series = parse_metric_series(_make_series())
        with pytest.raises(ValidationError):
            series.eval_id = 'other'  # type: ignore[misc]
        with pytest.raises(ValidationError):
            series.metrics[0].value = 0.1  # type: ignore[misc]

    @pytest.mark.parametrize('bad_version', [0, 2, 99, '1'])
    def test_wrong_schema_version_is_a_validation_failure(self, bad_version):
        # Pinned as a Literal so a wrong version fails loudly rather than
        # taking a silent branch (capability_manifest.py:219 convention).
        with pytest.raises(ValidationError):
            parse_metric_series(_make_series(schema_version=bad_version))

    def test_metric_constructs_directly(self):
        metric = Metric(**_make_count())
        assert metric.metric_id == 'dangling-pointers'
        assert metric.kind == 'count'
        assert metric.value == 5.0
        assert metric.n == 30


# (malformed metric dict, the metric_id the error message must name)
_MALFORMED_METRICS = {
    'unknown-kind': (_make_metric(kind='histogram'), 'canonical-in-top-5'),
    'proportion-value-above-one': (_make_metric(value=1.4), 'canonical-in-top-5'),
    'proportion-value-negative': (_make_metric(value=-0.1), 'canonical-in-top-5'),
    'proportion-missing-denominator': (
        {'metric_id': 'canonical-in-top-5', 'kind': 'proportion', 'value': 0.8, 'n': 30},
        'canonical-in-top-5',
    ),
    # 0.815 * 30 == 24.45 successes — a proportion no binomial outcome produces.
    'proportion-non-integral-successes': (_make_metric(value=0.815), 'canonical-in-top-5'),
    'proportion-n-denominator-mismatch': (_make_metric(n=30, denominator=20), 'canonical-in-top-5'),
    'count-negative-value': (_make_count(value=-1.0), 'dangling-pointers'),
    'count-non-integral-value': (_make_count(value=5.5), 'dangling-pointers'),
    'count-with-denominator': (_make_count(denominator=30), 'dangling-pointers'),
    'tripwire-missing-items': (
        {'metric_id': 'topic-canonical-present', 'kind': 'tripwire', 'value': 1.0, 'n': 2},
        'topic-canonical-present',
    ),
    'tripwire-empty-items': (_make_tripwire(items=[], n=0, value=0.0), 'topic-canonical-present'),
    # items hold exactly one failure, but value claims two.
    'tripwire-value-disagrees-with-items': (_make_tripwire(value=2.0), 'topic-canonical-present'),
    'tripwire-n-disagrees-with-items': (_make_tripwire(n=7), 'topic-canonical-present'),
    'negative-n': (_make_count(n=-1), 'dangling-pointers'),
    'empty-metric-id': (_make_count(metric_id=''), ''),
    'unknown-extra-field': (_make_count(bogus_field=1), 'dangling-pointers'),
}


class TestMetricRejectedAtEmit:
    """M1: a malformed metric is rejected AT EMIT TIME, not read time.

    By the time the dashboard reads an artifact there is nobody left to tell,
    so every one of these must fail inside the producing runner — which is what
    ``validate_metric_series`` on the raw payload proves.
    """

    def test_error_type_is_a_value_error(self):
        # Callers catch one exception type without importing pydantic.
        assert issubclass(MetricSchemaError, ValueError)

    @pytest.mark.parametrize(
        ('metric', 'offender'),
        list(_MALFORMED_METRICS.values()),
        ids=list(_MALFORMED_METRICS),
    )
    def test_malformed_metric_is_rejected(self, metric, offender):
        with pytest.raises(MetricSchemaError) as exc_info:
            validate_metric_series(_make_series(metrics=[metric]))
        # structured-facts-at-failure: the message names the offending metric.
        assert repr(offender) in str(exc_info.value)

    @pytest.mark.parametrize(
        'metric', list(m for m, _ in _MALFORMED_METRICS.values()), ids=list(_MALFORMED_METRICS)
    )
    def test_malformed_metric_cannot_even_be_constructed(self, metric):
        # The strict schema is what makes rejection unavoidable at emit: a
        # producer cannot build the model and defer the failure to a reader.
        with pytest.raises(ValidationError):
            parse_metric_series(_make_series(metrics=[metric]))

    def test_duplicate_metric_id_within_one_series_is_rejected(self):
        payload = _make_series(metrics=[_make_count(), _make_count(value=6.0)])
        with pytest.raises(MetricSchemaError) as exc_info:
            validate_metric_series(payload)
        assert repr('dangling-pointers') in str(exc_info.value)

    def test_distinct_metric_ids_of_the_same_kind_are_fine(self):
        payload = _make_series(
            metrics=[_make_count(), _make_count(metric_id='superseded-above-successor')]
        )
        assert validate_metric_series(payload) is None

    def test_valid_series_passes_validation(self):
        assert validate_metric_series(_make_series()) is None

    def test_validate_accepts_an_already_parsed_series(self):
        series = parse_metric_series(_make_series())
        assert validate_metric_series(series) is None

    def test_error_chains_the_underlying_validation_error(self):
        # `raise ... from exc` so the pydantic detail is not lost, while the
        # public surface stays a single non-pydantic exception type.
        with pytest.raises(MetricSchemaError) as exc_info:
            validate_metric_series(_make_series(metrics=[_make_metric(value=1.4)]))
        assert isinstance(exc_info.value.__cause__, ValidationError)

    def test_series_level_shape_errors_are_also_rejected(self):
        with pytest.raises(MetricSchemaError):
            validate_metric_series({'schema_version': 1, 'eval_id': 'e1'})
