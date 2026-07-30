"""Two-way boundary suite over the COMMITTED memory-eval exemplars.

Everything here reads ``shared/tests/fixtures/memory_eval/`` — the same corpus
the unit suites use, following the two-consumer precedent of
``test_invocation_outcome.py`` + ``test_invocation_outcome_boundary.py``. The
exemplars are committed once and read from both sides rather than restated.

Three sections, deliberately:

* **Producer** — the committed series validate, the committed malformed ones
  are rejected, and re-emitting a parsed exemplar reproduces it byte for byte.
  That last one is what keeps the fixtures honest: they are exactly what the
  writer produces, so they cannot drift into hand-maintained fiction.
* **Consumer / evaluator** — the documented verdict per rule kind, over the
  committed baseline window.
* **Consumer / dashboard** — a reader using ONLY ``json.load`` and dict access,
  which reaches the same alarm set the evaluator returned. This pins the
  artifact contract the commissioned dashboard PRD consumes WITHOUT creating a
  code dependency on it, in either direction.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.memory_eval_limits import (
    LimitsConfig,
    evaluate_series,
    evaluate_tripwire,
    limits_artifact_path,
    scoped_grandfather_key,
    write_limits_artifact,
)
from shared.memory_eval_metrics import (
    MetricSchemaError,
    MetricSeries,
    load_metric_series,
    parse_metric_series,
    serialize_metric_series,
    validate_metric_series,
)

_FIXTURES = Path(__file__).parent / 'fixtures' / 'memory_eval'
_BASELINE_STAMPS = ('20260701T031500Z', '20260702T031500Z', '20260703T031500Z')
_REGRESSION_STAMP = '20260704T031500Z'
_QUIET_STAMP = '20260705T031500Z'
_DUAL_EVAL_ID = 'e1-dual-tripwire'
_DUAL_STAMPS = ('20260801T031500Z', '20260802T031500Z')

_SERIES_FILES = sorted(_FIXTURES.glob('e1-*/metrics-*.json'))
_MALFORMED_FILES = sorted(_FIXTURES.glob('malformed/*.json'))


def _ids(paths):
    return [f'{p.parent.name}/{p.name}' for p in paths]


# The recipe that produced the committed limits-current.json. Stated once, here,
# so the exemplar is reproducible rather than a mystery blob: the regression run
# judged against the three baseline runs, resuming from a grandfather set seeded
# by the first run's failures.
_EXEMPLAR_CONFIG = LimitsConfig(
    false_alarm_budget=1.0, runs_per_quarter=90, min_samples=10, baseline_window=3
)


def _series(stamp: str, *, eval_id: str = 'e1-retrieval-health'):
    return load_metric_series(_FIXTURES / eval_id / f'metrics-{stamp}.json')


def _baseline():
    return [_series(stamp) for stamp in _BASELINE_STAMPS]


def _seeded_grandfather() -> frozenset[str]:
    """The first baseline run's failures, snapshotted as known-bad.

    Scoped by metric_id, because that is the shape the persisted state carries —
    ``evaluate_tripwire`` speaks in bare item_keys for one metric, while
    ``evaluate_series`` persists ``"<metric_id>::<item_key>"``.
    """
    first = _series(_BASELINE_STAMPS[0])
    tripwire = next(m for m in first.metrics if m.kind == 'tripwire')
    _, snapshot = evaluate_tripwire(tripwire, None)
    return frozenset(scoped_grandfather_key(tripwire.metric_id, key) for key in snapshot)


def _seeded_snapshotted() -> frozenset[str]:
    """The tripwire ids the seed run snapshotted — the ledger beside the seed set.

    Carried explicitly rather than left to default, because that is the
    supported resume path: a run resumes from BOTH halves of the persisted
    state, and an exemplar that skipped one would document the wrong recipe.
    """
    first = _series(_BASELINE_STAMPS[0])
    return frozenset(m.metric_id for m in first.metrics if m.kind == 'tripwire')


def evaluate_exemplar():
    """Reproduce the committed limits artifact's evaluation. Used by step-20's generator."""
    return evaluate_series(
        _series(_REGRESSION_STAMP),
        _baseline(),
        _EXEMPLAR_CONFIG,
        _seeded_grandfather(),
        _seeded_snapshotted(),
    )


def _verdict(result, metric_id):
    return next(v for v in result.verdicts if v.metric_id == metric_id)


class TestCommittedExemplarsAreProducible:
    """Producer side: the fixtures are what the writer emits, not hand-drift."""

    def test_the_corpus_is_actually_present(self):
        # A glob that silently matched nothing would make every parametrized
        # test below vacuously pass.
        assert len(_SERIES_FILES) == 8
        assert len(_MALFORMED_FILES) == 2

    @pytest.mark.parametrize('path', _SERIES_FILES, ids=_ids(_SERIES_FILES))
    def test_every_committed_series_validates(self, path):
        series = parse_metric_series(json.loads(path.read_text()))
        validate_metric_series(series)

    @pytest.mark.parametrize('path', _SERIES_FILES, ids=_ids(_SERIES_FILES))
    def test_reemitting_a_committed_series_is_byte_identical(self, path):
        """The anti-drift assertion.

        If a fixture were hand-edited into a shape the writer would never
        produce, every test that reads it would be testing a file no runner can
        emit. Byte identity makes that impossible to do accidentally.
        """
        assert serialize_metric_series(load_metric_series(path)) == path.read_text()

    @pytest.mark.parametrize('path', _MALFORMED_FILES, ids=_ids(_MALFORMED_FILES))
    def test_every_committed_malformed_series_is_rejected(self, path):
        # Via validate_metric_series, which is the AT-EMIT contract: one
        # exception type, and a caller that need not import pydantic.
        with pytest.raises(MetricSchemaError):
            validate_metric_series(json.loads(path.read_text()))

    @pytest.mark.parametrize('path', _SERIES_FILES, ids=_ids(_SERIES_FILES))
    def test_a_committed_series_names_its_own_stamp(self, path):
        # The filename and the record must agree, or a window loaded by
        # filename order would be labelled wrongly in every downstream report.
        series = load_metric_series(path)
        assert path.name == f'metrics-{series.run_stamp}.json'
        assert path.parent.name == series.eval_id


class TestEvaluatorOverTheCommittedCorpus:
    """Consumer side: the documented verdict per rule kind."""

    def test_the_regression_run_alarms_on_every_rule_kind(self):
        result = evaluate_exemplar()
        assert _verdict(result, 'canonical-in-top-5').status == 'alarm'
        assert _verdict(result, 'dangling-pointers').status == 'alarm'
        assert _verdict(result, 'topic-canonical-present').status == 'alarm'

    def test_the_regression_run_leaves_steady_metrics_alone(self):
        # superseded-above-successor is flat at 2 across the window and the
        # current run. An evaluator that alarmed here would be alarming on
        # nothing at all.
        result = evaluate_exemplar()
        assert _verdict(result, 'superseded-above-successor').status == 'ok'
        assert _verdict(result, 'search-latency-p50-ms').status == 'ok'

    def test_the_regression_tripwire_alarm_names_the_new_failure(self):
        result = evaluate_exemplar()
        alarms = _verdict(result, 'topic-canonical-present').alarms
        assert [a.item_key for a in alarms] == ['t-worktree-lifecycle']

    def test_pre_existing_failures_do_not_alarm(self):
        # t-recon-watcher-triage fails in every run including the baseline. It
        # is known-bad, so it is the fix lineage's problem, not an alarm.
        result = evaluate_exemplar()
        alarmed = {a.item_key for a in result.alarms}
        assert 't-recon-watcher-triage' not in alarmed
        assert 'topic-canonical-present::t-recon-watcher-triage' in result.grandfather

    def test_the_ratchet_releases_an_item_fixed_in_the_regression_run(self):
        # t-routing-ladder was grandfathered and passes in run 04, so it leaves
        # the set — even though that same run is alarming elsewhere.
        result = evaluate_exemplar()
        released = 'topic-canonical-present::t-routing-ladder'
        assert released in _seeded_grandfather()
        assert released not in result.grandfather

    def test_the_quiet_run_alarms_nothing(self):
        result = evaluate_series(
            _series(_QUIET_STAMP), _baseline(), _EXEMPLAR_CONFIG, _seeded_grandfather()
        )
        assert result.alarms == ()
        assert {v.status for v in result.verdicts} == {'ok'}

    def test_a_second_pass_over_the_quiet_run_still_alarms_nothing(self):
        # Idempotence over the committed corpus: re-running an unchanged eval
        # must stay silent, or a restart would produce phantom alarms.
        first = evaluate_series(
            _series(_QUIET_STAMP), _baseline(), _EXEMPLAR_CONFIG, _seeded_grandfather()
        )
        second = evaluate_series(
            _series(_QUIET_STAMP), _baseline(), _EXEMPLAR_CONFIG, first.grandfather
        )
        assert second.alarms == ()
        assert second.grandfather_hash == first.grandfather_hash

    def test_the_thin_run_is_insufficient_rather_than_alarming(self):
        thin = _series(_REGRESSION_STAMP, eval_id='e1-thin')
        result = evaluate_series(thin, _baseline(), _EXEMPLAR_CONFIG, None)
        assert result.alarms == ()
        assert {v.status for v in result.verdicts} == {'insufficient_data'}


class TestTwoTripwireSeriesOverTheCommittedCorpus:
    """A committed series carrying TWO tripwire metrics, with a colliding item_key.

    ``e1-dual-tripwire`` exists because a single-tripwire corpus cannot see the
    per-metric scoping of the grandfather state at all: with one tripwire, a
    flat set and a scoped set behave identically. Both runs are the same data,
    so the whole class asserts one thing — evaluating unchanged data twice stays
    silent — which is the observable symptom of every way the scoping can break.

    ``t-shared`` appears in BOTH metrics, failing under
    ``topic-canonical-present`` and passing under ``successor-pointer-present``.
    That is the collision case: in a flat namespace the pass releases the
    failure's known-bad entry, and the next run alarms on data nobody touched.
    """

    CONFIG = LimitsConfig(
        false_alarm_budget=1.0, runs_per_quarter=90, min_samples=1, baseline_window=3
    )

    def _dual(self, stamp: str):
        return _series(stamp, eval_id=_DUAL_EVAL_ID)

    def test_the_first_run_snapshots_both_metrics_without_alarming(self):
        result = evaluate_series(self._dual(_DUAL_STAMPS[0]), [], self.CONFIG, None)
        assert result.alarms == ()
        assert {v.status for v in result.verdicts} == {'baseline_snapshot'}
        assert sorted(result.grandfather) == [
            'successor-pointer-present::t-beta',
            'topic-canonical-present::t-recon-watcher-triage',
            'topic-canonical-present::t-shared',
        ]

    def test_the_unchanged_second_run_alarms_nothing(self):
        first = evaluate_series(self._dual(_DUAL_STAMPS[0]), [], self.CONFIG, None)
        second = evaluate_series(self._dual(_DUAL_STAMPS[1]), [], self.CONFIG, first.grandfather)
        assert second.alarms == ()
        assert second.grandfather_hash == first.grandfather_hash

    def test_the_shared_item_key_stays_scoped_to_its_own_metric(self):
        first = evaluate_series(self._dual(_DUAL_STAMPS[0]), [], self.CONFIG, None)
        assert 'topic-canonical-present::t-shared' in first.grandfather
        assert 'successor-pointer-present::t-shared' not in first.grandfather

    def test_a_tripwire_added_mid_programme_snapshots_over_the_committed_corpus(self):
        """The e1-retrieval-health probe joining an eval that already has state.

        Run 1 evaluates ONLY successor-pointer-present; run 2 adds
        topic-canonical-present, whose failures are pre-existing rather than
        new. M2 grandfathers them, so the run is silent — and the metric that
        already had state keeps its own slice across the arrival.
        """
        first_run = self._dual(_DUAL_STAMPS[0])
        solo = MetricSeries(
            schema_version=first_run.schema_version,
            eval_id=first_run.eval_id,
            run_stamp=first_run.run_stamp,
            corpus=first_run.corpus,
            metrics=[m for m in first_run.metrics if m.metric_id == 'successor-pointer-present'],
        )
        first = evaluate_series(solo, [], self.CONFIG, None)
        assert sorted(first.grandfather) == ['successor-pointer-present::t-beta']

        added = evaluate_series(
            first_run, [], self.CONFIG, first.grandfather, first.snapshotted_metrics
        )
        assert added.alarms == ()
        assert _verdict(added, 'topic-canonical-present').status == 'baseline_snapshot'
        assert _verdict(added, 'successor-pointer-present').status == 'ok'
        assert sorted(added.grandfather) == [
            'successor-pointer-present::t-beta',
            'topic-canonical-present::t-recon-watcher-triage',
            'topic-canonical-present::t-shared',
        ]

    def test_the_dual_artifact_publishes_scoped_known_bad_keys(self, tmp_path):
        # The dashboard reads these strings, so the scoping has to be visible in
        # the published artifact rather than living only in memory.
        result = evaluate_series(self._dual(_DUAL_STAMPS[0]), [], self.CONFIG, None)
        path = limits_artifact_path(tmp_path, result.eval_id)
        write_limits_artifact(result, path)
        published = json.loads(path.read_text())['grandfather_set']
        assert all('::' in key for key in published)
        assert published == sorted(result.grandfather)


class TestDashboardShapedReader:
    """Consumer side: plain ``json.load`` + dict access reaches the same alarms.

    This class deliberately touches NO function from ``shared.memory_eval_*``
    for its reading — only :mod:`json` and dict indexing, exactly what a
    dashboard would do. It pins the on-disk contract the commissioned dashboard
    PRD depends on, while leaving that PRD free to be written later against a
    shape that is already fixed and tested.
    """

    ARTIFACT = _FIXTURES / 'e1-retrieval-health' / 'limits-current.json'

    def _read(self) -> dict:
        with self.ARTIFACT.open(encoding='utf-8') as fh:
            return json.load(fh)

    def test_the_exemplar_artifact_is_committed(self):
        assert self.ARTIFACT.is_file()

    def test_a_plain_reader_gets_the_alarm_list(self):
        data = self._read()
        alarms = data['alarms']
        assert {a['metric_id'] for a in alarms} == {
            'canonical-in-top-5',
            'dangling-pointers',
            'topic-canonical-present',
        }
        assert all(a['detail'] for a in alarms)

    def test_a_plain_reader_gets_the_rule_kinds(self):
        data = self._read()
        kinds = {v['metric_id']: v['rule_kind'] for v in data['verdicts']}
        assert kinds['canonical-in-top-5'] == 'proportion'
        assert kinds['dangling-pointers'] == 'count'
        assert kinds['topic-canonical-present'] == 'tripwire'
        assert kinds['search-latency-p50-ms'] == 'scalar'

    def test_a_plain_reader_gets_the_derived_alpha_and_its_inputs(self):
        data = self._read()
        assert data['alpha'] == pytest.approx(1 / 360)
        assert data['false_alarm_budget'] == 1.0
        assert data['runs_per_quarter'] == 90
        assert data['alarmed_metric_count'] == 4

    def test_a_plain_reader_gets_the_provenance(self):
        data = self._read()
        assert data['schema_version'] == 1
        assert data['eval_id'] == 'e1-retrieval-health'
        assert data['run_stamp'] == _REGRESSION_STAMP
        assert data['generator']
        assert data['baseline_run_stamps'] == list(_BASELINE_STAMPS)

    def test_a_plain_reader_gets_the_known_bad_list(self):
        data = self._read()
        assert data['grandfather_set'] == ['topic-canonical-present::t-recon-watcher-triage']
        assert len(data['grandfather_set_hash']) == 64

    def test_the_plain_reader_and_the_evaluator_agree(self):
        """The assertion that makes the other six mean something.

        The dashboard's view and the evaluator's return value must be the same
        alarms — that is the whole reason limits-current.json is one file
        serving both (INV-1/INV-5). If these ever diverge, the dashboard is
        re-implementing the limits rather than reading them.
        """
        from_disk = {(a['metric_id'], a['item_key']) for a in self._read()['alarms']}
        from_evaluator = {(a.metric_id, a.item_key) for a in evaluate_exemplar().alarms}
        assert from_disk == from_evaluator

    def test_the_committed_artifact_is_exactly_what_the_writer_emits(self, tmp_path):
        # Same anti-drift guarantee the series fixtures get: regenerate, and
        # the bytes must match. This is what lets the README say "regenerate,
        # do not hand-edit" and have it be enforced rather than hoped for.
        result = evaluate_exemplar()
        regenerated = limits_artifact_path(tmp_path, result.eval_id)
        write_limits_artifact(result, regenerated)
        assert regenerated.read_text() == self.ARTIFACT.read_text()
