"""Tests for shared.memory_eval_metrics — the M1 metric-series schema + writer.

Built bottom-up in TDD order, following shared/tests/test_capability_manifest.py's
convention (see docs/prds/memory-eval-program.md §3 M1):
  - TestMetricSeriesSchema: the happy-path shape and round-trip in isolation.
  - TestMetricRejectedAtEmit: the per-kind rejection matrix, raised AT EMIT.
  - TestArtifactWriter: the STAMP artifact layout, atomicity and window loader.
"""

from __future__ import annotations

import pytest

from shared.memory_eval_metrics import (
    SCHEMA_VERSION,
    Corpus,
    Metric,
    MetricSeries,
    TripwireItem,
    parse_metric_series,
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
        with pytest.raises(Exception):
            series.eval_id = 'other'  # type: ignore[misc]
        with pytest.raises(Exception):
            series.metrics[0].value = 0.1  # type: ignore[misc]

    @pytest.mark.parametrize('bad_version', [0, 2, 99, '1'])
    def test_wrong_schema_version_is_a_validation_failure(self, bad_version):
        # Pinned as a Literal so a wrong version fails loudly rather than
        # taking a silent branch (capability_manifest.py:219 convention).
        with pytest.raises(Exception):
            parse_metric_series(_make_series(schema_version=bad_version))

    def test_metric_constructs_directly(self):
        metric = Metric(**_make_count())
        assert metric.metric_id == 'dangling-pointers'
        assert metric.kind == 'count'
        assert metric.value == 5.0
        assert metric.n == 30
