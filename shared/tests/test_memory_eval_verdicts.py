"""The verdicts artifact: per-fingerprint classification over the committed corpus.

Everything is driven by alpha's committed exemplars under
``shared/tests/fixtures/memory_eval/`` and by the one sanctioned regeneration
recipe, ``test_memory_eval_boundary.evaluate_exemplar`` — the regression run
20260704 judged against the three baseline runs. Other runs are chained from a
real first run (the grandfather snapshot), exactly as the scheduled runner
chains them, rather than from a hand-built grandfather set.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from test_memory_eval_boundary import evaluate_exemplar

from shared.memory_eval_limits import EvaluationResult, LimitsConfig, evaluate_series
from shared.memory_eval_metrics import (
    MetricSeries,
    canonical_json_text,
    load_metric_series,
    metrics_artifact_path,
)
from shared.memory_eval_verdicts import (
    RUNNER_FAILURE_FINGERPRINT,
    RunnerFailureRecord,
    StormEscape,
    VerdictEntry,
    VerdictsArtifact,
    VerdictsSchemaError,
    build_verdicts_artifact,
    load_verdicts_artifact,
    metric_fingerprint,
    serialize_verdicts_artifact,
    storm_fingerprint,
    verdict_entries,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

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

    def test_a_tripwire_limit_ref_carries_the_full_grandfather_hash(self):
        result = evaluate_exemplar()
        tripwire_refs = {e.limit_ref for e in _regression_entries() if e.metric_id == _TRIPWIRE}
        assert tripwire_refs == {f'grandfather_set_hash={result.grandfather_hash}'}


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


def _regression_run() -> tuple[MetricSeries, EvaluationResult]:
    return _series(_REGRESSION_STAMP), evaluate_exemplar()


def _thin_run() -> tuple[MetricSeries, EvaluationResult]:
    thin = _series(_REGRESSION_STAMP, eval_id='e1-thin')
    return thin, evaluate_series(thin, _baseline(), _config(), None)


def _failure(**overrides) -> RunnerFailureRecord:
    fields = {
        'scope': 'e4-staleness-sweep', 'phase': 'produce',
        'error_type': 'ProducerExit', 'message': 'exited 3', 'exit_code': 3,
    }
    return RunnerFailureRecord(**(fields | overrides))


def _artifact(*, alarm_limit: int = 3, stamp: str = _REGRESSION_STAMP, failures=()):
    evaluations = [_regression_run()] if stamp == _REGRESSION_STAMP else []
    return build_verdicts_artifact(
        evaluations, run_stamp=stamp, alarm_limit=alarm_limit, runner_failures=failures,
    )


class TestStormEscape:
    def test_alarms_at_the_limit_do_not_trigger_the_storm(self):
        storm = _artifact(alarm_limit=3).storm_escape
        assert (storm.triggered, storm.alarm_count, storm.alarm_limit) == (False, 3, 3)
        assert storm.aggregate_fingerprint is None

    def test_alarms_over_the_limit_trigger_one_run_scoped_aggregate(self):
        storm = _artifact(alarm_limit=2).storm_escape
        assert (storm.triggered, storm.alarm_count, storm.alarm_limit) == (True, 3, 2)
        assert storm.aggregate_fingerprint == storm_fingerprint(_REGRESSION_STAMP)

    def test_the_aggregate_is_scoped_to_the_run(self):
        assert storm_fingerprint(_REGRESSION_STAMP) != storm_fingerprint(_QUIET_STAMP)
        rerun = _artifact(alarm_limit=2).storm_escape.aggregate_fingerprint
        assert rerun == _artifact(alarm_limit=2).storm_escape.aggregate_fingerprint

    @pytest.mark.parametrize('fields', [
        {'triggered': True, 'alarm_count': 3, 'alarm_limit': 2},
        {'triggered': False, 'alarm_count': 3, 'alarm_limit': 3, 'aggregate_fingerprint': 'f'},
        {'triggered': False, 'alarm_count': 3, 'alarm_limit': 2},
    ])
    def test_an_inconsistent_storm_block_is_refused(self, fields):
        with pytest.raises(ValueError):
            StormEscape(**fields)


class TestRunnerFailures:
    def test_failures_are_carried_with_one_program_fingerprint(self):
        artifact = _artifact(failures=[_failure(), _failure(scope='program', phase='runner')])
        assert [f.scope for f in artifact.runner_failures] == ['e4-staleness-sweep', 'program']
        assert artifact.runner_failure_fingerprint == RUNNER_FAILURE_FINGERPRINT

    def test_the_failure_fingerprint_is_stamp_free(self):
        later = _artifact(stamp=_QUIET_STAMP, failures=[_failure()])
        assert later.runner_failure_fingerprint == RUNNER_FAILURE_FINGERPRINT

    def test_no_failures_means_no_failure_fingerprint(self):
        artifact = _artifact()
        assert artifact.runner_failures == []
        assert artifact.runner_failure_fingerprint is None

    def test_a_failure_with_an_unknown_phase_is_refused(self):
        with pytest.raises(ValueError):
            _failure(phase='deploy')


class TestArtifactAssembly:
    def test_entries_from_several_evals_concatenate_in_input_order(self):
        regression, thin = _regression_run(), _thin_run()
        artifact = build_verdicts_artifact(
            [regression, thin], run_stamp=_REGRESSION_STAMP, alarm_limit=10,
        )
        assert artifact.entries == [*verdict_entries(*regression), *verdict_entries(*thin)]

    def test_an_evaluation_from_another_run_is_refused(self):
        with pytest.raises(ValueError):
            build_verdicts_artifact([_regression_run()], run_stamp=_QUIET_STAMP, alarm_limit=10)

    def test_the_artifact_names_its_generator_and_version(self):
        artifact = _artifact()
        assert artifact.schema_version == 1
        assert artifact.generator == 'shared.memory_eval_verdicts'
        assert artifact.run_stamp == _REGRESSION_STAMP


class TestStrictAtEmit:
    def test_an_entry_with_both_item_and_window_is_refused(self):
        entry = _regression_entries()[0]
        with pytest.raises(ValueError):
            VerdictEntry(**(entry.model_dump() | {'item': 'x'}))

    def test_an_unknown_verdict_is_refused(self):
        entry = _regression_entries()[0]
        with pytest.raises(ValueError):
            VerdictEntry(**(entry.model_dump() | {'verdict': 'meh'}))

    def test_a_schema_version_other_than_one_is_refused(self):
        payload = _artifact().model_dump() | {'schema_version': 2}
        with pytest.raises(ValueError):
            VerdictsArtifact(**payload)


def _mutated(text: str, mutate) -> str:
    data = json.loads(text)
    mutate(data)
    return json.dumps(data)


_MUTATIONS = {
    'schema_version': lambda d: d.update(schema_version=2),
    'extra_entry_field': lambda d: d['entries'][0].update(surprise=1),
    'unknown_verdict': lambda d: d['entries'][0].update(verdict='meh'),
    'item_and_window': lambda d: d['entries'][0].update(item='x'),
    'hand_edited_fingerprint': lambda d: d['entries'][0].update(fingerprint='0' * 64),
    'extra_top_level_field': lambda d: d.update(surprise=1),
}


class TestSerializeAndLoad:
    def test_serialize_then_load_round_trips(self, tmp_path):
        for artifact in (_artifact(), _artifact(alarm_limit=2, failures=[_failure()])):
            path = tmp_path / 'verdicts.json'
            path.write_text(serialize_verdicts_artifact(artifact), encoding='utf-8')
            assert load_verdicts_artifact(path) == artifact

    def test_serialization_is_canonical(self):
        text = serialize_verdicts_artifact(_artifact())
        assert text == canonical_json_text(json.loads(text))

    def test_inapplicable_optionals_are_absent_not_null(self):
        data = json.loads(serialize_verdicts_artifact(_artifact(failures=[_failure(exit_code=None)])))
        assert 'aggregate_fingerprint' not in data['storm_escape']
        assert 'exit_code' not in data['runner_failures'][0]
        assert all(('item' in e) != ('window' in e) for e in data['entries'])
        quiet = json.loads(serialize_verdicts_artifact(_artifact()))
        assert 'runner_failure_fingerprint' not in quiet
        assert quiet['runner_failures'] == []

    @pytest.mark.parametrize('name', sorted(_MUTATIONS))
    def test_a_malformed_artifact_is_refused_at_load(self, tmp_path, name):
        path = tmp_path / 'verdicts.json'
        path.write_text(_mutated(serialize_verdicts_artifact(_artifact()), _MUTATIONS[name]))
        with pytest.raises(VerdictsSchemaError):
            load_verdicts_artifact(path)

    def test_the_schema_error_is_a_value_error(self):
        assert issubclass(VerdictsSchemaError, ValueError)


class TestDashboardShapedRead:
    """Only ``json`` and dict access — the keys the dashboard's verdicts reader consumes."""

    def test_a_plain_reader_finds_entries_and_the_storm_block(self):
        data = json.loads(serialize_verdicts_artifact(_artifact(alarm_limit=2)))
        assert isinstance(data['entries'], list)
        assert isinstance(data['storm_escape'], dict)
        assert data['storm_escape']['triggered'] is True
        for entry in data['entries']:
            for key in ('eval_id', 'metric_id', 'verdict', 'fingerprint', 'value', 'limit_ref', 'run_stamp'):
                assert key in entry


_VERDICTS_FIXTURES = Path(__file__).parent / 'fixtures' / 'memory_eval_verdicts'
_EXEMPLAR_LIMITS = {'filed': 3, 'storm': 2}


def _exemplar_path(name: str) -> Path:
    return _VERDICTS_FIXTURES / name / 'verdicts-current.json'


def _read_exemplar(name: str) -> dict:
    return json.loads(_exemplar_path(name).read_text(encoding='utf-8'))


class TestCommittedVerdictsExemplars:
    """The committed regression-run exemplars: K=3 files per alarm, K=2 collapses to a storm."""

    @pytest.mark.parametrize('name', sorted(_EXEMPLAR_LIMITS))
    def test_the_exemplar_is_committed_and_valid(self, name):
        assert _exemplar_path(name).is_file()
        load_verdicts_artifact(_exemplar_path(name))

    @pytest.mark.parametrize('name', sorted(_EXEMPLAR_LIMITS))
    def test_the_exemplar_is_exactly_what_the_builder_emits(self, name):
        regenerated = serialize_verdicts_artifact(build_verdicts_artifact(
            [_regression_run()], run_stamp=_REGRESSION_STAMP, alarm_limit=_EXEMPLAR_LIMITS[name],
        ))
        assert regenerated == _exemplar_path(name).read_text(encoding='utf-8')

    def test_the_filed_exemplar_carries_three_fingerprinted_alarms(self):
        data = _read_exemplar('filed')
        alarms = [e for e in data['entries'] if e['verdict'] == 'alarm']
        assert len(alarms) == 3
        assert all(isinstance(e['fingerprint'], str) and e['fingerprint'] for e in alarms)
        assert data['storm_escape']['triggered'] is False

    def test_the_storm_exemplar_carries_a_triggered_aggregate(self):
        storm = _read_exemplar('storm')['storm_escape']
        assert storm['triggered'] is True
        assert isinstance(storm['aggregate_fingerprint'], str) and storm['aggregate_fingerprint']

    def test_the_exemplar_tree_cannot_be_mistaken_for_an_eval(self):
        dirs = sorted(p.name for p in _VERDICTS_FIXTURES.rglob('*') if p.is_dir())
        assert dirs == ['filed', 'storm']
        assert list(_VERDICTS_FIXTURES.rglob('metrics-*.json')) == []

    def test_alphas_fixture_root_still_has_no_verdicts_artifact(self):
        assert not (_FIXTURES / 'verdicts-current.json').exists()
