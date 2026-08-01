"""Tests for shared.memory_eval_metrics — the M1 metric-series schema + writer.

Built bottom-up in TDD order, following shared/tests/test_capability_manifest.py's
convention (see docs/prds/memory-eval-program.md §3 M1):
  - TestMetricSeriesSchema: the happy-path shape and round-trip in isolation.
  - TestMetricRejectedAtEmit: the per-kind rejection matrix, raised AT EMIT.
  - TestArtifactWriter: the STAMP artifact layout, atomicity and window loader.
"""

from __future__ import annotations

import json
import re

import pytest
from pydantic import ValidationError

from shared.memory_eval_metrics import (
    SCHEMA_VERSION,
    Corpus,
    Metric,
    MetricSchemaError,
    MetricSeries,
    TripwireItem,
    eval_dir,
    load_metric_series,
    load_series_window,
    metrics_artifact_path,
    parse_metric_series,
    report_artifact_path,
    run_stamp,
    validate_metric_series,
    write_metric_series,
)


def _make_metric(**overrides) -> dict:
    """A minimal valid ``proportion`` metric dict; override any field."""
    payload = {
        'metric_id': 'canonical-in-top-5',
        'kind': 'proportion',
        'value': 0.8,
        'n': 30,
        'denominator': 30,
        'direction': 'lower_is_worse',
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
    payload = {
        'metric_id': 'dangling-pointers',
        'kind': 'count',
        'value': 5.0,
        'n': 30,
        'direction': 'higher_is_worse',
    }
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
        {
            'metric_id': 'canonical-in-top-5',
            'kind': 'proportion',
            'value': 0.8,
            'n': 30,
            'direction': 'lower_is_worse',
        },
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
    # M2 fires alarms on REGRESSIONS, and a two-sided exact test cannot tell a
    # collapse from a repair without being told which side is which — so the two
    # statistical kinds must declare it, and the two kinds with no two-sided test
    # must not (one home for the fact, no second source of truth).
    'proportion-missing-direction': (
        {
            'metric_id': 'canonical-in-top-5',
            'kind': 'proportion',
            'value': 0.8,
            'n': 30,
            'denominator': 30,
        },
        'canonical-in-top-5',
    ),
    'count-missing-direction': (
        {'metric_id': 'dangling-pointers', 'kind': 'count', 'value': 5.0, 'n': 30},
        'dangling-pointers',
    ),
    'direction-outside-the-vocabulary': (
        _make_count(direction='higher_is_better'),
        'dangling-pointers',
    ),
    'tripwire-with-direction': (
        _make_tripwire(direction='higher_is_worse'),
        'topic-canonical-present',
    ),
    'scalar-with-direction': (_make_scalar(direction='lower_is_worse'), 'search-latency-p50-ms'),
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


class TestArtifactWriter:
    """The STAMP artifact idiom (`cgl_eta_auto_apply_impl.py:42-46`) applied to M1.

    The stamp is in the filename precisely so a re-run never clobbers an earlier
    run's output — the series on disk IS the baseline window the limits
    evaluator reads, so a clobbered run is a silently shortened window.
    """

    def test_metrics_artifact_path_layout(self, tmp_path):
        path = metrics_artifact_path(tmp_path, 'e1-retrieval-health', '20260701T031500Z')
        assert path == tmp_path / 'e1-retrieval-health' / 'metrics-20260701T031500Z.json'

    def test_report_artifact_path_sits_beside_the_json(self, tmp_path):
        metrics = metrics_artifact_path(tmp_path, 'e1-retrieval-health', '20260701T031500Z')
        report = report_artifact_path(tmp_path, 'e1-retrieval-health', '20260701T031500Z')
        assert report.parent == metrics.parent
        assert report.name == 'report-20260701T031500Z.txt'

    def test_eval_dir_is_the_parent_of_both_artifact_paths(self, tmp_path):
        directory = eval_dir(tmp_path, 'e1-retrieval-health')
        assert directory == tmp_path / 'e1-retrieval-health'
        metrics = metrics_artifact_path(tmp_path, 'e1-retrieval-health', '20260701T031500Z')
        report = report_artifact_path(tmp_path, 'e1-retrieval-health', '20260701T031500Z')
        assert directory == metrics.parent == report.parent

    def test_eval_dir_does_not_validate_a_stamp_because_it_takes_none(self, tmp_path):
        # Unlike the two path helpers below, eval_dir has no stamp argument at
        # all — it is the helper a caller reaches for instead of threading a
        # throwaway placeholder stamp through metrics_artifact_path just to
        # discard it via .parent.
        assert eval_dir(tmp_path, 'e1-retrieval-health') == tmp_path / 'e1-retrieval-health'

    def test_metrics_artifact_path_rejects_a_malformed_stamp(self, tmp_path):
        with pytest.raises(MetricSchemaError):
            metrics_artifact_path(tmp_path, 'e1-retrieval-health', '2026-07-04T03:15:00Z')

    def test_report_artifact_path_rejects_a_malformed_stamp(self, tmp_path):
        with pytest.raises(MetricSchemaError):
            report_artifact_path(tmp_path, 'e1-retrieval-health', '2026-07-04T03:15:00Z')

    def test_run_stamp_honours_env_override(self, monkeypatch):
        monkeypatch.setenv('MEMORY_EVAL_RUN_STAMP', '20260704T031500Z')
        assert run_stamp() == '20260704T031500Z'

    def test_run_stamp_defaults_to_a_utc_stamp(self, monkeypatch):
        monkeypatch.delenv('MEMORY_EVAL_RUN_STAMP', raising=False)
        stamp = run_stamp()
        # Zero-padded UTC so lexicographic order == chronological order, which
        # is what lets the window loader sort by filename.
        assert re.fullmatch(r'\d{8}T\d{6}Z', stamp), stamp

    def test_write_creates_parents_and_returns_both_paths(self, tmp_path):
        root = tmp_path / 'memory-evals'
        metrics_path, report_path = write_metric_series(_make_series(), root)
        assert metrics_path.is_file()
        assert report_path.is_file()
        assert metrics_path == metrics_artifact_path(
            root, 'e1-retrieval-health', '20260701T031500Z'
        )
        assert report_path == report_artifact_path(root, 'e1-retrieval-health', '20260701T031500Z')

    def test_written_json_reloads_to_an_equal_model(self, tmp_path):
        metrics_path, _ = write_metric_series(_make_series(), tmp_path)
        assert load_metric_series(metrics_path) == parse_metric_series(_make_series())

    def test_report_is_human_readable(self, tmp_path):
        _, report_path = write_metric_series(_make_series(), tmp_path)
        text = report_path.read_text(encoding='utf-8')
        assert 'e1-retrieval-health' in text
        assert '20260701T031500Z' in text
        for metric in parse_metric_series(_make_series()).metrics:
            assert metric.metric_id in text

    def test_two_stamps_coexist_without_clobbering(self, tmp_path):
        first, _ = write_metric_series(_make_series(), tmp_path)
        second, _ = write_metric_series(_make_series(run_stamp='20260702T031500Z'), tmp_path)
        assert first != second
        assert first.is_file() and second.is_file()
        assert load_metric_series(first).run_stamp == '20260701T031500Z'
        assert load_metric_series(second).run_stamp == '20260702T031500Z'

    def test_bytes_are_stable_across_identical_writes(self, tmp_path):
        first, _ = write_metric_series(_make_series(), tmp_path)
        original = first.read_bytes()
        again, _ = write_metric_series(_make_series(), tmp_path)
        assert again.read_bytes() == original
        # sorted keys + trailing newline so the artifact diffs cleanly.
        assert original.endswith(b'\n')

    def test_explicit_stamp_overrides_the_series_stamp_in_the_filename(self, tmp_path):
        path, _ = write_metric_series(_make_series(), tmp_path, stamp='20260709T000000Z')
        assert path.name == 'metrics-20260709T000000Z.json'

    def test_malformed_series_writes_nothing_at_all(self, tmp_path):
        payload = _make_series(metrics=[_make_metric(value=1.4)])
        with pytest.raises(MetricSchemaError):
            write_metric_series(payload, tmp_path)
        eval_dir = tmp_path / 'e1-retrieval-health'
        # Not even a partial or temp file: validate-then-write ordering, and the
        # atomic writer unlinks its mkstemp temp on any failure.
        leftovers = list(eval_dir.rglob('*')) if eval_dir.exists() else []
        assert leftovers == []

    def test_malformed_series_does_not_corrupt_an_existing_artifact(self, tmp_path):
        good, _ = write_metric_series(_make_series(), tmp_path)
        original = good.read_bytes()
        with pytest.raises(MetricSchemaError):
            write_metric_series(_make_series(metrics=[_make_metric(value=1.4)]), tmp_path)
        assert good.read_bytes() == original
        assert not list(good.parent.glob('*.tmp'))

    def test_load_rejects_a_malformed_artifact_on_disk(self, tmp_path):
        path = tmp_path / 'e1' / 'metrics-20260701T031500Z.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(_make_series(metrics=[_make_metric(value=1.4)])), 'utf-8')
        with pytest.raises(MetricSchemaError):
            load_metric_series(path)


class TestSeriesWindowLoader:
    """The trailing baseline window the M2 rules (b)/(c) test the current run against."""

    def _seed(self, root, stamps):
        for stamp in stamps:
            write_metric_series(_make_series(run_stamp=stamp), root)

    def test_window_is_returned_in_chronological_order(self, tmp_path):
        self._seed(tmp_path, ['20260703T031500Z', '20260701T031500Z', '20260702T031500Z'])
        window = load_series_window(tmp_path, 'e1-retrieval-health', limit=3, before_stamp=None)
        assert [s.run_stamp for s in window] == [
            '20260701T031500Z',
            '20260702T031500Z',
            '20260703T031500Z',
        ]

    def test_limit_keeps_the_trailing_runs(self, tmp_path):
        self._seed(tmp_path, ['20260701T031500Z', '20260702T031500Z', '20260703T031500Z'])
        window = load_series_window(tmp_path, 'e1-retrieval-health', limit=2, before_stamp=None)
        assert [s.run_stamp for s in window] == ['20260702T031500Z', '20260703T031500Z']

    def test_missing_eval_dir_yields_an_empty_window(self, tmp_path):
        assert load_series_window(tmp_path, 'never-ran', limit=3, before_stamp=None) == []

    def test_window_excludes_non_metric_files(self, tmp_path):
        self._seed(tmp_path, ['20260701T031500Z'])
        eval_dir = tmp_path / 'e1-retrieval-health'
        (eval_dir / 'limits-current.json').write_text('{}', encoding='utf-8')
        window = load_series_window(tmp_path, 'e1-retrieval-health', limit=5, before_stamp=None)
        assert [s.run_stamp for s in window] == ['20260701T031500Z']

    def test_before_stamp_excludes_the_current_runs_own_artifact(self, tmp_path):
        """A run must not be pooled into the baseline it is about to be judged against.

        The natural runner ordering is write-then-evaluate, so by the time the
        baseline window is loaded the current run's own ``metrics-*.json`` is
        already on disk. Pooling it drags ``p0``/``lam`` toward the observed
        value and DAMPS the very regression the M2 rules (b)/(c) exist to catch
        — the worse the run, the more it moves the bar it is measured against.
        """
        self._seed(
            tmp_path,
            [
                '20260701T031500Z',
                '20260702T031500Z',
                '20260703T031500Z',
                '20260704T031500Z',
            ],
        )
        window = load_series_window(
            tmp_path,
            'e1-retrieval-health',
            limit=3,
            before_stamp='20260704T031500Z',
        )
        assert [s.run_stamp for s in window] == [
            '20260701T031500Z',
            '20260702T031500Z',
            '20260703T031500Z',
        ]

    def test_limit_is_applied_after_the_exclusion_not_before(self, tmp_path):
        """The window keeps its full length once the current run is dropped.

        The subtle way this fix can be got wrong is filtering AFTER slicing: the
        trailing ``limit`` files include the current run, dropping it leaves
        ``limit - 1``, and the evidence base quietly halves with nothing to
        signal it. Asking for 3 must yield the 3 newest strictly-earlier runs.
        """
        self._seed(
            tmp_path,
            [
                '20260701T031500Z',
                '20260702T031500Z',
                '20260703T031500Z',
                '20260704T031500Z',
                '20260705T031500Z',
                '20260706T031500Z',
            ],
        )
        window = load_series_window(
            tmp_path,
            'e1-retrieval-health',
            limit=3,
            before_stamp='20260706T031500Z',
        )
        assert [s.run_stamp for s in window] == [
            '20260703T031500Z',
            '20260704T031500Z',
            '20260705T031500Z',
        ]

    def test_before_stamp_also_excludes_a_later_run(self, tmp_path):
        """A baseline holding a FUTURE run is no more a baseline than one holding the present.

        The filter is a strict ``<`` over the zero-padded stamp, so a concurrent
        runner, a re-run with a pinned stamp, or clock skew — all the same
        contamination class — are refused by the same rule.
        """
        self._seed(tmp_path, ['20260701T031500Z', '20260702T031500Z', '20260709T031500Z'])
        window = load_series_window(
            tmp_path,
            'e1-retrieval-health',
            limit=5,
            before_stamp='20260703T031500Z',
        )
        assert [s.run_stamp for s in window] == ['20260701T031500Z', '20260702T031500Z']

    def test_before_stamp_none_keeps_the_whole_window(self, tmp_path):
        """``None`` is the explicit opt-out: "no current run to exclude", not "I forgot"."""
        self._seed(tmp_path, ['20260701T031500Z', '20260702T031500Z', '20260703T031500Z'])
        window = load_series_window(
            tmp_path,
            'e1-retrieval-health',
            limit=5,
            before_stamp=None,
        )
        assert [s.run_stamp for s in window] == [
            '20260701T031500Z',
            '20260702T031500Z',
            '20260703T031500Z',
        ]

    def test_omitting_before_stamp_is_a_type_error(self, tmp_path):
        """Required-but-nullable, so the footgun cannot be re-entered by forgetting it.

        A ``before_stamp=None`` DEFAULT would reinstate exactly the self-pooling
        this parameter exists to prevent, for anyone who does the natural thing.
        """
        self._seed(tmp_path, ['20260701T031500Z'])
        with pytest.raises(TypeError):
            load_series_window(tmp_path, 'e1-retrieval-health', limit=3)  # type: ignore[call-arg]


# Plausible-looking stamps that are not `%Y%m%dT%H%M%SZ`. Every one of them
# sorts somewhere other than where its date says, and none of them is exotic:
# the ISO-8601 spelling is what most callers would reach for first.
_MISSHAPEN_STAMPS = [
    '2026-07-04T03:15:00Z',  # ISO-8601 with separators — sorts before every 2026… stamp
    '20260704T031500',  # no trailing Z
    '20260704031500Z',  # no T
    '2026074T031500Z',  # month not zero-padded, so ordering breaks at the boundary
    '20260704T0315Z',  # seconds dropped — a different width
    '20260704T031500Z ',  # trailing whitespace, invisible in a diff
    'latest',  # a symbolic name
    '',  # empty
]


class TestStampShapeIsEnforcedNotAssumed:
    """The lexicographic-order-is-chronological-order invariant, made checkable.

    ``_STAMP_FORMAT``'s docstring has always DECLARED that the stamp is
    fixed-width zero-padded UTC, and both window operations rely on it — the
    ``sorted()`` that finds the trailing runs and the ``<`` that cuts the
    current one out. Nothing checked it. A differently-shaped stamp does not
    fail, it just sorts somewhere else: ``'2026-07-04T03:15:00Z'`` orders before
    EVERY ``20260…`` stamp, so it silently yields an empty baseline (as a
    cutoff) or a contaminated one (as a filename). That is the silent
    degradation this module argues against everywhere else, so the shape is now
    refused at each boundary a stamp can enter from.
    """

    @pytest.mark.parametrize('stamp', _MISSHAPEN_STAMPS)
    def test_the_schema_refuses_a_misshapen_run_stamp(self, stamp):
        # On the field, so an artifact that sorts into the wrong place cannot be
        # constructed, written OR read back — one rule, not a guard per caller.
        with pytest.raises(MetricSchemaError):
            validate_metric_series(_make_series(run_stamp=stamp))

    @pytest.mark.parametrize('stamp', _MISSHAPEN_STAMPS)
    def test_the_env_override_is_refused_at_the_source(self, stamp, monkeypatch):
        # The one place an operator-supplied string enters the module. Refusing
        # it here means the bad stamp never becomes a filename in the first
        # place, so the failure names the env var rather than a mystery window.
        monkeypatch.setenv('MEMORY_EVAL_RUN_STAMP', stamp)
        if not stamp:
            # An empty override is falsy, so it means "unset" — not a stamp at
            # all — and legitimately falls through to the clock.
            assert re.fullmatch(r'[0-9]{8}T[0-9]{6}Z', run_stamp())
            return
        with pytest.raises(MetricSchemaError) as exc_info:
            run_stamp()
        assert 'MEMORY_EVAL_RUN_STAMP' in str(exc_info.value)

    @pytest.mark.parametrize('stamp', _MISSHAPEN_STAMPS)
    def test_a_write_time_stamp_override_is_refused(self, tmp_path, stamp):
        # Overriding is exactly how a caller sidesteps the validated field, so
        # the override gets the same check — and, like every other malformed
        # write, leaves nothing behind.
        with pytest.raises(MetricSchemaError):
            write_metric_series(_make_series(), tmp_path, stamp=stamp)
        assert not (tmp_path / 'e1-retrieval-health').exists()

    @pytest.mark.parametrize('stamp', _MISSHAPEN_STAMPS)
    def test_a_misshapen_before_stamp_is_refused_rather_than_cutting_arbitrarily(
        self, tmp_path, stamp
    ):
        (tmp_path / 'e1-retrieval-health').mkdir()
        write_metric_series(_make_series(), tmp_path)
        with pytest.raises(MetricSchemaError):
            load_series_window(tmp_path, 'e1-retrieval-health', limit=3, before_stamp=stamp)

    @pytest.mark.parametrize('stamp', _MISSHAPEN_STAMPS)
    def test_a_misshapen_before_stamp_is_refused_even_when_the_eval_never_ran(
        self, tmp_path, stamp
    ):
        """Validated BEFORE the missing-directory short-circuit.

        Otherwise the first run of a new eval swallows the bad argument and only
        the second one reports it — a bug that surfaces a day late, in a
        different run, with nothing pointing back at the caller.
        """
        with pytest.raises(MetricSchemaError):
            load_series_window(tmp_path, 'never-ran', limit=3, before_stamp=stamp)

    def test_the_error_names_the_shape_it_wanted(self, tmp_path):
        # structured-facts-at-failure: the message must be actionable without
        # opening the module, so it carries the offending value AND the format.
        with pytest.raises(MetricSchemaError) as exc_info:
            load_series_window(
                tmp_path, 'e1-retrieval-health', limit=3, before_stamp='2026-07-04T03:15:00Z'
            )
        message = str(exc_info.value)
        assert '2026-07-04T03:15:00Z' in message
        assert '%Y%m%dT%H%M%SZ' in message
        assert 'before_stamp' in message

    def test_a_non_ascii_digit_stamp_is_refused(self):
        """``\\d`` would accept this; ``[0-9]`` does not.

        Arabic-Indic digits render as a plausible stamp and sort nowhere near
        their ASCII equivalents, so this is the same wrong-window failure in a
        shape a reviewer would never spot by eye.
        """
        with pytest.raises(MetricSchemaError):
            validate_metric_series(_make_series(run_stamp='٢٠٢٦٠٧٠٤T٠٣١٥٠٠Z'))

    def test_a_well_formed_stamp_still_passes_everywhere(self, tmp_path):
        # The guard must not be so tight that the real thing fails: one stamp
        # through all four boundaries.
        monkeypatch_free_stamp = '20260704T031500Z'
        validate_metric_series(_make_series(run_stamp=monkeypatch_free_stamp))
        path, _ = write_metric_series(_make_series(), tmp_path, stamp=monkeypatch_free_stamp)
        assert path.name == f'metrics-{monkeypatch_free_stamp}.json'
        assert (
            load_series_window(
                tmp_path, 'e1-retrieval-health', limit=3, before_stamp='20260705T031500Z'
            )
            != []
        )
