"""The verdicts artifact: per-fingerprint classification over the committed corpus.

Everything is driven by alpha's committed exemplars under
``shared/tests/fixtures/memory_eval/`` and by the one sanctioned regeneration
recipe, ``test_memory_eval_boundary.evaluate_exemplar`` — the regression run
20260704 judged against the three baseline runs. Other runs are chained from a
real first run (the grandfather snapshot), exactly as the scheduled runner
chains them, rather than from a hand-built grandfather set.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from shared.memory_eval_verdicts import VerdictEntry, metric_fingerprint, verdict_entries
from test_memory_eval_boundary import evaluate_exemplar

from shared.memory_eval_limits import EvaluationResult, LimitsConfig, evaluate_series
from shared.memory_eval_metrics import MetricSeries, load_metric_series, metrics_artifact_path

_FIXTURES = Path(__file__).parent / 'fixtures' / 'memory_eval'
_EVAL_ID = 'e1-retrieval-health'
_BASELINE_STAMPS = ('20260701T031500Z', '20260702T031500Z', '20260703T031500Z')
_REGRESSION_STAMP = '20260704T031500Z'
_QUIET_STAMP = '20260705T031500Z'
_TRIPWIRE = 'topic-canonical-present'
_WINDOW = 'trailing-3'


def _series(stamp: str, *, eval_id: str = _EVAL_ID) -> MetricSeries:
    return load_metric_series(metrics_artifact_path(_FIXTURES, eval_id, stamp))


def _config() -> LimitsConfig:
    return evaluate_exemplar().config


def _baseline() -> list[MetricSeries]:
    return [_series(stamp) for stamp in _BASELINE_STAMPS]


def _first_run() -> EvaluationResult:
    """The grandfather snapshot: the first baseline run, with no prior state."""
    return evaluate_series(_series(_BASELINE_STAMPS[0]), [], _config(), None)


def _chained(stamp: str) -> EvaluationResult:
    first = _first_run()
    return evaluate_series(
        _series(stamp), _baseline(), _config(), first.grandfather, first.snapshotted_metrics,
    )


def _regression_entries() -> tuple[VerdictEntry, ...]:
    return verdict_entries(_series(_REGRESSION_STAMP), evaluate_exemplar())


def _whole_metric(entries: tuple[VerdictEntry, ...], metric_id: str) -> VerdictEntry:
    (entry,) = [e for e in entries if e.metric_id == metric_id]
    return entry


def _item(entries: tuple[VerdictEntry, ...], item: str) -> VerdictEntry:
    (entry,) = [e for e in entries if e.metric_id == _TRIPWIRE and e.item == item]
    return entry


class TestRegressionRunClassification:
    def test_whole_metric_rules_give_one_windowed_entry_each(self):
        entries = _regression_entries()
        expected = {
            'canonical-in-top-5': ('alarm', 'alarm'),
            'dangling-pointers': ('alarm', 'alarm'),
            'superseded-above-successor': ('no_alarm', 'ok'),
            'search-latency-p50-ms': ('no_alarm', 'ok'),
        }
        for metric_id, (verdict, status) in expected.items():
            entry = _whole_metric(entries, metric_id)
            assert (entry.verdict, entry.metric_status) == (verdict, status), metric_id
            assert entry.window == _WINDOW
            assert entry.item is None

    def test_the_tripwire_yields_one_entry_per_item(self):
        entries = _regression_entries()
        tripwire = [e for e in entries if e.metric_id == _TRIPWIRE]
        items = [i.item_key for m in _series(_REGRESSION_STAMP).metrics
                 if m.metric_id == _TRIPWIRE for i in m.items or []]
        assert [e.item for e in tripwire] == items
        assert all(e.window is None for e in tripwire)
        assert _item(entries, 't-worktree-lifecycle').verdict == 'alarm'
        assert _item(entries, 't-recon-watcher-triage').verdict == 'grandfathered'
        assert _item(entries, 't-routing-ladder').verdict == 'no_alarm'
        others = set(items) - {'t-worktree-lifecycle', 't-recon-watcher-triage'}
        assert {_item(entries, key).verdict for key in others} == {'no_alarm'}

    def test_one_alarm_entry_per_evaluator_alarm(self):
        result = evaluate_exemplar()
        alarm_fingerprints = [e.fingerprint for e in _regression_entries() if e.verdict == 'alarm']
        expected = {
            metric_fingerprint(result.eval_id, a.metric_id, item=a.item_key)
            if a.item_key is not None
            else metric_fingerprint(result.eval_id, a.metric_id, window=_WINDOW)
            for a in result.alarms
        }
        assert len(alarm_fingerprints) == len(result.alarms)
        assert set(alarm_fingerprints) == expected

    def test_every_entry_carries_stamp_value_and_limit_ref(self):
        result = evaluate_exemplar()
        values = {v.metric_id: v.value for v in result.verdicts}
        for entry in _regression_entries():
            assert entry.eval_id == _EVAL_ID
            assert entry.run_stamp == _REGRESSION_STAMP
            assert entry.value == values[entry.metric_id]
            assert isinstance(entry.limit_ref, str) and entry.limit_ref


class TestFingerprints:
    def test_fingerprints_are_stable_across_runs(self):
        alarmed = _regression_entries()
        quiet = verdict_entries(_series(_QUIET_STAMP), _chained(_QUIET_STAMP))
        recovered_item = _item(quiet, 't-worktree-lifecycle')
        recovered_metric = _whole_metric(quiet, 'canonical-in-top-5')
        assert recovered_item.verdict == 'no_alarm'
        assert recovered_metric.verdict == 'no_alarm'
        assert recovered_item.fingerprint == _item(alarmed, 't-worktree-lifecycle').fingerprint
        assert recovered_metric.fingerprint == _whole_metric(alarmed, 'canonical-in-top-5').fingerprint

    def test_distinct_identities_give_distinct_fingerprints(self):
        dual = _series('20260801T031500Z', eval_id='e1-dual-tripwire')
        dual_entries = verdict_entries(dual, evaluate_series(dual, [], _config(), None))
        shared_item = [e.fingerprint for e in dual_entries if e.item == 't-shared']
        assert len(shared_item) == 2
        assert shared_item[0] != shared_item[1]

        everything = [e.fingerprint for e in (*_regression_entries(), *dual_entries)]
        assert len(set(everything)) == len(everything)

    def test_the_identity_kind_and_eval_are_part_of_the_key(self):
        assert metric_fingerprint('e1', 'm', item='x') != metric_fingerprint('e1', 'm', window='x')
        assert metric_fingerprint('e1', 'm', item='x') != metric_fingerprint('e2', 'm', item='x')

    def test_fingerprints_are_opaque(self):
        for entry in _regression_entries():
            assert re.fullmatch('[0-9a-f]{64}', entry.fingerprint)
            assert entry.metric_id not in entry.fingerprint
            if entry.item is not None:
                assert entry.item not in entry.fingerprint

    @pytest.mark.parametrize('kwargs', [{}, {'item': 'x', 'window': 'trailing-3'}])
    def test_exactly_one_of_item_or_window(self, kwargs):
        with pytest.raises(ValueError):
            metric_fingerprint('e1', 'm', **kwargs)


class TestNonAlarmStatuses:
    def test_the_first_run_grandfathers_failures_and_alarms_nothing(self):
        first = _series(_BASELINE_STAMPS[0])
        entries = verdict_entries(first, _first_run())
        tripwire = next(m for m in first.metrics if m.metric_id == _TRIPWIRE)
        for item in tripwire.items or []:
            expected = 'no_alarm' if item.passed else 'grandfathered'
            assert _item(entries, item.item_key).verdict == expected
            assert _item(entries, item.item_key).metric_status == 'baseline_snapshot'
        assert [e for e in entries if e.verdict == 'alarm'] == []

    def test_a_thin_run_is_insufficient_data(self):
        thin = _series(_REGRESSION_STAMP, eval_id='e1-thin')
        entries = verdict_entries(thin, evaluate_series(thin, _baseline(), _config(), None))
        assert entries
        assert {e.verdict for e in entries} == {'insufficient_data'}

    def test_an_improvement_is_no_alarm_with_its_status_preserved(self):
        quiet = _series(_QUIET_STAMP)
        improved = quiet.model_copy(update={'metrics': [
            m.model_copy(update={'value': 0.0}) if m.metric_id == 'dangling-pointers' else m
            for m in quiet.metrics
        ]})
        generous = LimitsConfig(
            false_alarm_budget=90.0, runs_per_quarter=90, min_samples=10, baseline_window=3,
        )
        first = _first_run()
        result = evaluate_series(
            improved, _baseline(), generous, first.grandfather, first.snapshotted_metrics,
        )
        entry = _whole_metric(verdict_entries(improved, result), 'dangling-pointers')
        assert (entry.verdict, entry.metric_status) == ('no_alarm', 'improved')


class TestMismatchedInputs:
    def test_a_different_eval_is_refused(self):
        thin = _series(_REGRESSION_STAMP, eval_id='e1-thin')
        with pytest.raises(ValueError):
            verdict_entries(thin, evaluate_exemplar())

    def test_a_different_run_is_refused(self):
        with pytest.raises(ValueError):
            verdict_entries(_series(_QUIET_STAMP), evaluate_exemplar())
