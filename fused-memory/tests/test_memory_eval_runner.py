"""Tests for memory_eval_runner.py — the scheduled memory-eval runner.

The script is loaded via importlib (``_fm_helpers.load_script_module``), as the
producer scripts' own tests do. No test touches a live store: producers are
injected as tiny fake scripts, and escalations are filed into a real
``EscalationQueue`` rooted under ``tmp_path``.
"""
from __future__ import annotations

import functools
import json
import shutil
import textwrap
import types
from pathlib import Path

import pytest
import yaml
from _fm_helpers import load_script_module
from shared.memory_eval_limits import LimitsConfig, limits_artifact_path, load_limits_artifact
from shared.memory_eval_metrics import load_metric_series

_FM_ROOT = Path(__file__).parent.parent
SCRIPT_PATH = _FM_ROOT / 'scripts' / 'memory_eval_runner.py'
COMMITTED_CONFIG = _FM_ROOT / 'config' / 'memory-eval.yaml'
_ALPHA_FIXTURES = _FM_ROOT.parent / 'shared' / 'tests' / 'fixtures' / 'memory_eval'
_E1 = 'e1-retrieval-health'
_STAMP = '20260801T031500Z'
_REQUIRED_KEYS = ('storm_alarm_limit', 'min_samples', 'baseline_window', 'runs_per_quarter')


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(SCRIPT_PATH, mod_name='memory_eval_runner')


def _write_config(tmp_path: Path, **changes) -> Path:
    data = yaml.safe_load(COMMITTED_CONFIG.read_text(encoding='utf-8'))
    for key, value in changes.items():
        if value is None:
            data.pop(key)
        else:
            data[key] = value
    path = tmp_path / 'memory-eval.yaml'
    path.write_text(yaml.safe_dump(data), encoding='utf-8')
    return path


class TestRunnerConfig:
    def test_the_committed_config_declares_a_daily_cadence(self):
        config = _mod().load_runner_config(COMMITTED_CONFIG)
        limits = config.limits_config()
        assert limits.runs_per_quarter == 90
        assert limits.min_samples == config.min_samples
        assert limits.baseline_window == config.baseline_window
        assert limits.false_alarm_budget == config.false_alarm_budget

    @pytest.mark.parametrize('key', _REQUIRED_KEYS)
    def test_a_missing_declaration_is_refused(self, tmp_path, key):
        with pytest.raises(ValueError):
            _mod().load_runner_config(_write_config(tmp_path, **{key: None}))

    def test_an_unknown_key_is_refused(self, tmp_path):
        with pytest.raises(ValueError):
            _mod().load_runner_config(_write_config(tmp_path, surprise=1))

    def test_a_non_positive_storm_limit_is_refused(self, tmp_path):
        with pytest.raises(ValueError):
            _mod().load_runner_config(_write_config(tmp_path, storm_alarm_limit=0))


class TestQueueDirResolution:
    def test_a_relative_setting_resolves_against_the_repo_root(self, tmp_path):
        resolved = _mod().resolve_recon_queue_dir('./data/reconciliation/escalations', tmp_path)
        assert resolved == tmp_path / 'data' / 'reconciliation' / 'escalations'

    def test_an_absolute_setting_is_left_alone(self, tmp_path):
        absolute = tmp_path / 'elsewhere'
        assert _mod().resolve_recon_queue_dir(str(absolute), Path('/repo')) == absolute


class TestPreflightExits:
    def _run(self, tmp_path: Path, queue_dir: Path, config: Path = COMMITTED_CONFIG) -> int:
        return _mod().main([
            '--config', str(config),
            '--artifact-root', str(tmp_path / 'artifacts'),
            '--queue-dir', str(queue_dir),
        ])

    def _no_artifacts(self, tmp_path: Path) -> bool:
        root = tmp_path / 'artifacts'
        return not root.exists() or not any(root.iterdir())

    def test_a_missing_queue_dir_exits_2_without_artifacts(self, tmp_path, capsys):
        queue_dir = tmp_path / 'no-such-queue'
        assert self._run(tmp_path, queue_dir) == 2
        assert str(queue_dir) in capsys.readouterr().err
        assert self._no_artifacts(tmp_path)

    def test_an_empty_queue_dir_exits_2_without_artifacts(self, tmp_path, capsys):
        queue_dir = tmp_path / 'empty-queue'
        queue_dir.mkdir()
        assert self._run(tmp_path, queue_dir) == 2
        assert str(queue_dir) in capsys.readouterr().err
        assert self._no_artifacts(tmp_path)

    def test_an_invalid_config_exits_2_without_artifacts(self, tmp_path, capsys):
        config = _write_config(tmp_path, storm_alarm_limit=None)
        assert self._run(tmp_path, tmp_path / 'queue', config) == 2
        assert str(config) in capsys.readouterr().err
        assert self._no_artifacts(tmp_path)


_FAKE_PRODUCER = textwrap.dedent("""
    import argparse, json, os, sys, time
    from shared.memory_eval_metrics import RUN_STAMP_ENV_VAR, write_metric_series

    parser = argparse.ArgumentParser()
    for flag in ('--root', '--eval-id', '--source', '--stderr'):
        parser.add_argument(flag, default='')
    parser.add_argument('--exit', type=int, default=0)
    parser.add_argument('--sleep', type=float, default=0.0)
    parser.add_argument('--no-write', action='store_true')
    args = parser.parse_args()
    time.sleep(args.sleep)
    print(f'fake producer {args.eval_id} ran', flush=True)
    if args.stderr:
        print(args.stderr, file=sys.stderr)
    if args.exit == 0 and not args.no_write:
        with open(args.source, encoding='utf-8') as fh:
            data = json.load(fh)
        data.update(eval_id=args.eval_id, run_stamp=os.environ[RUN_STAMP_ENV_VAR])
        write_metric_series(data, args.root)
    sys.exit(args.exit)
""")


@pytest.fixture
def fake_script(tmp_path: Path) -> Path:
    path = tmp_path / 'fake_producer.py'
    path.write_text(_FAKE_PRODUCER, encoding='utf-8')
    return path


def _alpha_series(stamp: str, eval_id: str = _E1) -> Path:
    return _ALPHA_FIXTURES / eval_id / f'metrics-{stamp}.json'


def _fake(script: Path, eval_id: str = _E1, *, source: Path | None = None, extra=()):
    source = source or _alpha_series('20260701T031500Z')

    def build_argv(root: Path, eval_id: str, project_id: str) -> list[str]:
        return ['--root', str(root), '--eval-id', eval_id, '--source', str(source), *extra]

    return _mod().Producer(eval_id=eval_id, script=script, build_argv=build_argv)


def _flag_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


class TestDefaultProducers:
    def test_three_producers_whose_scripts_exist(self):
        producers = _mod().DEFAULT_PRODUCERS
        assert [p.eval_id for p in producers] == [
            'e1-retrieval-health', 'e4-staleness-sweep', 'e6-corpus-health',
        ]
        assert [p.script.name for p in producers] == [
            'memory_eval_retrieval_probe.py',
            'memory_eval_staleness_sweep.py',
            'audit_duplicate_memories.py',
        ]
        for producer in producers:
            assert producer.script.is_file()
            assert producer.script.parent == _FM_ROOT / 'scripts'

    def test_e1_writes_under_the_root(self, tmp_path):
        e1 = _mod().DEFAULT_PRODUCERS[0]
        argv = e1.build_argv(tmp_path, e1.eval_id, 'dark_factory')
        assert _flag_value(argv, '--out-root') == str(tmp_path)
        assert '--eval-id' not in argv

    def test_e4_and_e6_are_told_the_registry_eval_id(self, tmp_path):
        for producer in _mod().DEFAULT_PRODUCERS[1:]:
            argv = producer.build_argv(tmp_path, producer.eval_id, 'dark_factory')
            assert _flag_value(argv, '--eval-id') == producer.eval_id
            assert _flag_value(argv, '--metrics-root') == str(tmp_path)
            assert _flag_value(argv, '--project-id') == 'dark_factory'

    def test_no_producer_is_ever_asked_to_mutate_or_skip_its_metrics(self, tmp_path):
        for producer in _mod().DEFAULT_PRODUCERS:
            argv = producer.build_argv(tmp_path, producer.eval_id, 'dark_factory')
            assert '--apply' not in argv
            assert '--no-metrics' not in argv


class TestRunProducer:
    def _run(self, producer, root: Path, timeout: float = 60.0):
        return _mod().run_producer(
            producer, root=root, stamp=_STAMP, project_id='dark_factory', timeout=timeout,
        )

    def test_a_producer_that_writes_its_series_succeeds_under_the_shared_stamp(
        self, tmp_path, fake_script, capsys,
    ):
        root = tmp_path / 'artifacts'
        assert self._run(_fake(fake_script), root) is None
        assert (root / _E1 / f'metrics-{_STAMP}.json').is_file()
        assert 'fake producer e1-retrieval-health ran' in capsys.readouterr().err

    def test_a_non_zero_exit_is_a_produce_failure_carrying_the_stderr_tail(
        self, tmp_path, fake_script,
    ):
        producer = _fake(fake_script, extra=['--exit', '3', '--stderr', 'store unreachable'])
        failure = self._run(producer, tmp_path / 'artifacts')
        assert (failure.scope, failure.phase, failure.exit_code) == (_E1, 'produce', 3)
        assert 'store unreachable' in failure.message

    def test_a_clean_exit_without_the_series_is_a_produce_failure(self, tmp_path, fake_script):
        root = tmp_path / 'artifacts'
        failure = self._run(_fake(fake_script, extra=['--no-write']), root)
        assert (failure.scope, failure.phase) == (_E1, 'produce')
        assert str(root / _E1 / f'metrics-{_STAMP}.json') in failure.message

    def test_a_producer_over_its_timeout_is_a_produce_failure(self, tmp_path, fake_script):
        producer = _fake(fake_script, extra=['--sleep', '30'])
        failure = self._run(producer, tmp_path / 'artifacts', timeout=0.5)
        assert (failure.scope, failure.phase, failure.error_type) == (
            _E1, 'produce', 'TimeoutExpired',
        )


_TRIPWIRE = 'topic-canonical-present'
_FIRST, _REGRESSION, _QUIET = '20260701T031500Z', '20260704T031500Z', '20260705T031500Z'
_ALL_STAMPS = (_FIRST, '20260702T031500Z', '20260703T031500Z', _REGRESSION, _QUIET)
_LIMITS = LimitsConfig(false_alarm_budget=1.0, runs_per_quarter=90, min_samples=10, baseline_window=3)


def _seed(root: Path, *stamps: str, eval_id: str = _E1) -> None:
    (root / eval_id).mkdir(parents=True, exist_ok=True)
    for stamp in stamps:
        shutil.copy(_alpha_series(stamp), root / eval_id / f'metrics-{stamp}.json')


def _failing(stamp: str) -> list[str]:
    series = load_metric_series(_alpha_series(stamp))
    tripwire = next(m for m in series.metrics if m.metric_id == _TRIPWIRE)
    return sorted(f'{_TRIPWIRE}::{i.item_key}' for i in tripwire.items or [] if not i.passed)


def _tripwire_status(run) -> str:
    return next(v.status for v in run.result.verdicts if v.metric_id == _TRIPWIRE)


def _alarm_keys(run) -> set[tuple[str, str | None]]:
    return {(a.metric_id, a.item_key) for a in run.result.alarms}


def _evaluate(root: Path, stamp: str, limits: LimitsConfig = _LIMITS):
    return _mod().evaluate_eval(root, _E1, stamp, limits)


class TestEvaluateEval:
    def test_the_first_run_is_the_grandfather_snapshot(self, tmp_path):
        _seed(tmp_path, _FIRST)
        run = _evaluate(tmp_path, _FIRST)
        assert _tripwire_status(run) == 'baseline_snapshot'
        assert run.result.alarms == ()
        persisted = load_limits_artifact(limits_artifact_path(tmp_path, _E1))
        assert persisted.grandfather_set == _failing(_FIRST)
        assert _TRIPWIRE in persisted.snapshotted_metric_ids
        assert run.limits_path == limits_artifact_path(tmp_path, _E1)

    def test_each_run_resumes_from_the_persisted_state(self, tmp_path):
        _seed(tmp_path, *_ALL_STAMPS[:4])
        _evaluate(tmp_path, _FIRST)
        run = _evaluate(tmp_path, _REGRESSION)
        tripwire_alarms = {key for metric, key in _alarm_keys(run) if metric == _TRIPWIRE}
        assert tripwire_alarms == {'t-worktree-lifecycle'}
        persisted = load_limits_artifact(limits_artifact_path(tmp_path, _E1))
        assert persisted.grandfather_set == [f'{_TRIPWIRE}::t-recon-watcher-triage']

    def test_deleting_the_limits_artifact_starts_a_fresh_snapshot(self, tmp_path):
        _seed(tmp_path, *_ALL_STAMPS[:4])
        _evaluate(tmp_path, _FIRST)
        limits_artifact_path(tmp_path, _E1).unlink()
        run = _evaluate(tmp_path, _REGRESSION)
        assert _tripwire_status(run) == 'baseline_snapshot'
        assert not [a for a in run.result.alarms if a.metric_id == _TRIPWIRE]

    def test_the_window_is_the_trailing_runs_before_the_current_stamp(self, tmp_path):
        _seed(tmp_path, *_ALL_STAMPS)
        narrow = LimitsConfig(
            false_alarm_budget=1.0, runs_per_quarter=90, min_samples=10, baseline_window=2,
        )
        run = _evaluate(tmp_path, _REGRESSION, narrow)
        assert run.result.baseline_run_stamps == ('20260702T031500Z', '20260703T031500Z')

    def test_evaluating_a_stamp_twice_is_idempotent(self, tmp_path):
        _seed(tmp_path, *_ALL_STAMPS[:4])
        _evaluate(tmp_path, _FIRST)
        first = _evaluate(tmp_path, _REGRESSION)
        second = _evaluate(tmp_path, _REGRESSION)
        assert _alarm_keys(first) == _alarm_keys(second)
        assert first.result.grandfather_hash == second.result.grandfather_hash

    def test_the_run_carries_its_artifact_paths_and_entries(self, tmp_path):
        _seed(tmp_path, _FIRST)
        run = _evaluate(tmp_path, _FIRST)
        assert run.metrics_path == tmp_path / _E1 / f'metrics-{_FIRST}.json'
        assert run.report_path == tmp_path / _E1 / f'report-{_FIRST}.txt'
        assert run.series.run_stamp == _FIRST
        assert run.entries and all(e.run_stamp == _FIRST for e in run.entries)


class TestEvaluateEvals:
    def test_a_hand_edited_grandfather_set_is_an_evaluate_failure(self, tmp_path):
        _seed(tmp_path, _FIRST, _REGRESSION)
        _evaluate(tmp_path, _FIRST)
        limits = limits_artifact_path(tmp_path, _E1)
        data = json.loads(limits.read_text(encoding='utf-8'))
        data['grandfather_set'] = data['grandfather_set'][:1]
        limits.write_text(json.dumps(data), encoding='utf-8')

        runs, failures = _mod().evaluate_evals(tmp_path, [_E1], _REGRESSION, _LIMITS)
        assert runs == []
        (failure,) = failures
        assert (failure.scope, failure.phase, failure.error_type) == (
            _E1, 'evaluate', 'LimitsSchemaError',
        )

    def test_healthy_evals_are_evaluated_beside_a_broken_one(self, tmp_path):
        _seed(tmp_path, _FIRST)
        runs, failures = _mod().evaluate_evals(tmp_path, ['e-missing', _E1], _FIRST, _LIMITS)
        assert [run.eval_id for run in runs] == [_E1]
        assert [(f.scope, f.phase) for f in failures] == [('e-missing', 'evaluate')]
