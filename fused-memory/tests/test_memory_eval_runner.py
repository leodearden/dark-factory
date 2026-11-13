"""Tests for memory_eval_runner.py — the scheduled memory-eval runner.

The script is loaded via importlib (``_fm_helpers.load_script_module``), as the
producer scripts' own tests do. No test touches a live store: producers are
injected as tiny fake scripts, and escalations are filed into a real
``EscalationQueue`` rooted under ``tmp_path``.
"""
from __future__ import annotations

import functools
import json
import os
import shutil
import sys
import textwrap
import types
from datetime import timedelta
from pathlib import Path

import pytest
import yaml
from _fm_helpers import load_script_module
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from shared.memory_eval_limits import LimitsConfig, limits_artifact_path, load_limits_artifact
from shared.memory_eval_metrics import (
    RUN_STAMP_ENV_VAR,
    load_metric_series,
    stamp_datetime,
    write_metric_series,
)
from shared.memory_eval_verdicts import (
    RunnerFailureRecord,
    RunnerPhase,
    StormEscape,
    VerdictsArtifact,
    build_verdicts_artifact,
    load_verdicts_artifact,
)

from fused_memory.eval.retrieval_probe import EVAL_ID as E1_PROBE_EVAL_ID

_FM_ROOT = Path(__file__).parent.parent
SCRIPT_PATH = _FM_ROOT / 'scripts' / 'memory_eval_runner.py'
COMMITTED_CONFIG = _FM_ROOT / 'config' / 'memory-eval.yaml'
_ALPHA_FIXTURES = _FM_ROOT.parent / 'shared' / 'tests' / 'fixtures' / 'memory_eval'
_E1 = 'e1-retrieval-health'
_STAMP = '20260801T031500Z'


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

    @pytest.mark.parametrize('key', tuple(_mod().RunnerConfig.model_fields))
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

    @pytest.mark.parametrize('content', [
        pytest.param('reconciliation: [unclosed\n', id='unparseable'),
        pytest.param('reconciliation: 5\n', id='invalid'),
    ])
    def test_a_bad_fused_memory_config_exits_2_naming_config_path(
        self, tmp_path, monkeypatch, capsys, content,
    ):
        bad = tmp_path / 'config.yaml'
        bad.write_text(content, encoding='utf-8')
        monkeypatch.setenv('CONFIG_PATH', str(bad))
        code = _mod().main([
            '--config', str(COMMITTED_CONFIG), '--artifact-root', str(tmp_path / 'artifacts'),
        ])
        assert code == 2
        assert f'CONFIG_PATH={bad}' in capsys.readouterr().err
        assert self._no_artifacts(tmp_path)

    def test_a_malformed_run_stamp_exits_2_without_artifacts(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv(RUN_STAMP_ENV_VAR, 'bogus')
        assert self._run(tmp_path, _seeded_queue_dir(tmp_path)) == 2
        assert RUN_STAMP_ENV_VAR in capsys.readouterr().err
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


_PRODUCER_CLI: dict[str, tuple[str, str]] = {
    'e1-retrieval-health': ('build_parser', 'out_root'),
    'e4-staleness-sweep': ('build_parser', 'metrics_root'),
    'e6-corpus-health': ('_build_parser', 'metrics_root'),
    'e7-write-after-miss': ('build_parser', 'metrics_root'),
}
"""eval_id -> (the producer script's parser factory, the dest its artifact root lands in)."""


def _write_after_miss_series(stamp: str, *, incidents: int, adjudicated: int):
    """An e7 series built by the producer's own ``build_series``, so the shape is the real one."""
    producer = load_script_module(
        _FM_ROOT / 'scripts' / 'memory_eval_write_after_miss.py',
        mod_name='memory_eval_write_after_miss',
    )
    incident = producer.Incident(
        write_row_id='write-row', search_row_ids=('search-row',), project_id='dark_factory',
        caller_agent_id='claude-task-1-implementer', matched_memory_id='existing',
        similarity=0.95, threshold=0.92, category='procedural_knowledge',
        write_created_at=stamp_datetime(stamp),
    )
    adjudication = producer.Adjudication(
        incidents=(incident,) * incidents,
        adjudicated=adjudicated,
        skipped={reason: 0 for reason in producer.SkipReason},
        undated_candidates=0,
    )
    window = producer.JournalWindow(writes=(), searches=(), unattributed_searches=0, unparsed_rows=0)
    end = stamp_datetime(stamp)
    return producer.build_series(
        adjudication, window, period=producer.Period(end - timedelta(hours=24), end),
        project_id='dark_factory', stamp=stamp, details_path=producer.incidents_filename(stamp),
    )


class TestDefaultProducers:
    def test_four_producers_whose_scripts_exist(self):
        producers = _mod().DEFAULT_PRODUCERS
        assert [p.eval_id for p in producers] == [
            'e1-retrieval-health', 'e4-staleness-sweep', 'e6-corpus-health', 'e7-write-after-miss',
        ]
        assert [p.script.name for p in producers] == [
            'memory_eval_retrieval_probe.py',
            'memory_eval_staleness_sweep.py',
            'audit_duplicate_memories.py',
            'memory_eval_write_after_miss.py',
        ]
        for producer in producers:
            assert producer.script.is_file()
            assert producer.script.parent == _FM_ROOT / 'scripts'

    def test_e1_is_registered_under_the_probe_s_own_eval_id(self):
        assert _mod().DEFAULT_PRODUCERS[0].eval_id == E1_PROBE_EVAL_ID

    def test_e1_writes_under_the_root_and_probes_its_own_project_set(self, tmp_path):
        e1 = _mod().DEFAULT_PRODUCERS[0]
        argv = e1.build_argv(tmp_path, e1.eval_id, 'dark_factory')
        assert _flag_value(argv, '--out-root') == str(tmp_path)
        assert '--eval-id' not in argv
        assert '--project-id' not in argv

    @pytest.mark.parametrize('eval_id', sorted(_PRODUCER_CLI))
    def test_the_real_producer_parses_the_registry_argv(self, tmp_path, eval_id):
        producer = next(p for p in _mod().DEFAULT_PRODUCERS if p.eval_id == eval_id)
        parser_name, root_dest = _PRODUCER_CLI[eval_id]
        parser = getattr(load_script_module(producer.script), parser_name)()
        args = vars(parser.parse_args(producer.build_argv(tmp_path, eval_id, 'dark_factory')))
        assert Path(args[root_dest]) == tmp_path
        assert args.get('eval_id', eval_id) == eval_id
        assert not args.get('apply') and not args.get('no_metrics')

    def test_write_after_miss_is_judged_by_the_proportion_rule(self, tmp_path):
        """The wiring, not a limit (G6): e7's own build_series output reaches alpha's proportion rule."""
        stamps = ('20260701T031500Z', '20260702T031500Z', '20260703T031500Z', '20260704T031500Z')
        for stamp in stamps:
            write_metric_series(_write_after_miss_series(stamp, incidents=2, adjudicated=20), tmp_path)
        limits = _mod().load_runner_config(COMMITTED_CONFIG).limits_config()

        run = _mod().evaluate_eval(tmp_path, 'e7-write-after-miss', stamps[-1], limits)

        verdict = next(v for v in run.result.verdicts if v.metric_id == 'write-after-miss')
        assert verdict.rule_kind == 'proportion'

    def test_every_sweep_producer_is_told_the_registry_eval_id(self, tmp_path):
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

    def test_a_producer_that_cannot_start_is_a_produce_failure(
        self, tmp_path, fake_script, monkeypatch,
    ):
        monkeypatch.setattr(sys, 'executable', str(tmp_path / 'no-such-python'))
        failure = self._run(_fake(fake_script), tmp_path / 'artifacts')
        assert (failure.scope, failure.phase, failure.error_type) == (
            _E1, 'produce', 'FileNotFoundError',
        )
        assert 'could not start' in failure.message


_TRIPWIRE = 'topic-canonical-present'
_FIRST, _REGRESSION, _QUIET = '20260701T031500Z', '20260704T031500Z', '20260705T031500Z'
_ALL_STAMPS = (_FIRST, '20260702T031500Z', '20260703T031500Z', _REGRESSION, _QUIET)
_LIMITS = LimitsConfig(false_alarm_budget=1.0, runs_per_quarter=90, min_samples=10, baseline_window=3)


def _seed(root: Path, *stamps: str, eval_id: str = _E1) -> None:
    (root / eval_id).mkdir(parents=True, exist_ok=True)
    for stamp in stamps:
        shutil.copy(_alpha_series(stamp), root / eval_id / f'metrics-{stamp}.json')


def _failing_items(stamp: str) -> list[str]:
    series = load_metric_series(_alpha_series(stamp))
    tripwire = next(m for m in series.metrics if m.metric_id == _TRIPWIRE)
    return [i.item_key for i in tripwire.items or [] if not i.passed]


def _failing(stamp: str) -> list[str]:
    return sorted(f'{_TRIPWIRE}::{item}' for item in _failing_items(stamp))


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


_PROGRAM_TASK = 'memory-eval-program'


def _regression_run(root: Path):
    _seed(root, *_ALL_STAMPS[:4])
    _evaluate(root, _FIRST)
    return _evaluate(root, _REGRESSION)


def _context(root: Path, stamp: str = _REGRESSION):
    return _mod().FilingContext(
        run_stamp=stamp,
        project_id='dark_factory',
        run_report_path=root / f'run-report-{stamp}.txt',
        verdicts_path=root / f'verdicts-{stamp}.json',
    )


def _artifact(run, *, k: int = 10, failures=()) -> VerdictsArtifact:
    return build_verdicts_artifact(
        [(run.series, run.result)], run_stamp=_REGRESSION, alarm_limit=k, runner_failures=failures,
    )


def _file(queue, run, root: Path, *, k: int = 10, failures=()):
    artifact = _artifact(run, k=k, failures=failures)
    outcome = _mod().file_run_escalations(
        queue, [run], artifact.storm_escape, artifact.runner_failures, _context(root),
    )
    return artifact, outcome


def _pending(queue: EscalationQueue):
    return [e for e in queue.get_pending() if e.category == 'eval_regression']


def _alarm_fingerprints(run) -> set[str]:
    return {e.fingerprint for e in run.entries if e.verdict == 'alarm'}


def _by_fingerprint(queue: EscalationQueue) -> dict:
    return {e.dedupe_fingerprint: e for e in _pending(queue)}


def _stored(queue: EscalationQueue, esc_id: str) -> Escalation:
    esc = queue.get(esc_id)
    assert esc is not None, esc_id
    return esc


def _failure(
    scope: str, phase: RunnerPhase = 'produce', exit_code: int | None = 3,
) -> RunnerFailureRecord:
    return RunnerFailureRecord(
        scope=scope, phase=phase, error_type='ProducerExit', message='exited 3', exit_code=exit_code,
    )


class TestPerAlarmFiling:
    def test_one_escalation_per_alarm_entry(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'escalations')
        run = _regression_run(tmp_path / 'root')
        _, outcome = _file(queue, run, tmp_path / 'root')
        assert len(outcome.filed) == 3
        assert set(_by_fingerprint(queue)) == _alarm_fingerprints(run)

    def test_each_filing_is_contract_shaped(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'escalations')
        run = _regression_run(tmp_path / 'root')
        _file(queue, run, tmp_path / 'root')
        for esc in _pending(queue):
            assert esc.category == 'eval_regression'
            assert (esc.severity, esc.level, esc.status) == ('blocking', 1, 'pending')
            assert esc.task_id == f'memory-eval-{_E1}'
            assert esc.agent_role == 'memory-eval-runner'
            assert esc.project_id == 'dark_factory'
            assert esc.evidence
            for entry in esc.evidence:
                assert set(entry) == {'observation', 'measured_at', 'ref'}
                assert entry['measured_at'] == f'run_stamp={_REGRESSION}'

    def test_the_summary_names_the_metric_item_and_value_against_baseline(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'escalations')
        root = tmp_path / 'root'
        run = _regression_run(root)
        _file(queue, run, root)
        filed = _by_fingerprint(queue)
        alarm_entries = [e for e in run.entries if e.verdict == 'alarm']
        alarms = {(a.metric_id, a.item_key): a for a in run.result.alarms}
        for entry in alarm_entries:
            alarm = alarms[(entry.metric_id, entry.item)]
            summary = filed[entry.fingerprint].summary
            assert _E1 in summary and entry.metric_id in summary
            if entry.item is not None:
                assert entry.item in summary
            assert f'{alarm.value:g}' in summary
            assert f'{alarm.baseline:g}' in summary

    def test_the_detail_cites_the_limit_window_artifacts_and_a_remediation(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'escalations')
        root = tmp_path / 'root'
        run = _regression_run(root)
        _file(queue, run, root)
        context = _context(root)
        for entry in (e for e in run.entries if e.verdict == 'alarm'):
            detail = _by_fingerprint(queue)[entry.fingerprint].detail
            for cited in (
                entry.metric_id, entry.rule_kind, str(run.limits_path), str(run.report_path),
                str(context.run_report_path), str(context.verdicts_path), 'Remediation:',
            ):
                assert cited in detail, cited
            if entry.rule_kind == 'tripwire':
                assert f'limit: {entry.limit_ref}' in detail
                assert run.result.grandfather_hash in detail
            else:
                assert 'alpha=' in detail and 'p_value=' in detail
                assert all(stamp in detail for stamp in run.result.baseline_run_stamps)


class TestFingerprintDedup:
    def test_a_rerun_while_pending_files_nothing(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'escalations')
        root = tmp_path / 'root'
        run = _regression_run(root)
        _file(queue, run, root)
        _, rerun = _file(queue, run, root)
        assert rerun.filed == ()
        assert set(rerun.suppressed) == _alarm_fingerprints(run)
        assert len(_pending(queue)) == 3

    def test_resolving_one_lets_exactly_that_fingerprint_refile(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'escalations')
        root = tmp_path / 'root'
        run = _regression_run(root)
        _file(queue, run, root)
        resolved = _pending(queue)[0]
        queue.resolve(resolved.id, 'fixed upstream')
        _, rerun = _file(queue, run, root)
        (refiled,) = rerun.filed
        assert _stored(queue, refiled).dedupe_fingerprint == resolved.dedupe_fingerprint


class TestStormEscape:
    def test_alarms_over_k_file_one_aggregate_and_no_per_metric_escalation(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'escalations')
        root = tmp_path / 'root'
        run = _regression_run(root)
        artifact, outcome = _file(queue, run, root, k=2)
        (only,) = _pending(queue)
        assert outcome.filed == (only.id,)
        assert only.dedupe_fingerprint == artifact.storm_escape.aggregate_fingerprint
        assert only.task_id == _PROGRAM_TASK
        assert '3' in only.summary
        assert str(_context(root).run_report_path) in only.summary
        assert not _alarm_fingerprints(run) & {e.dedupe_fingerprint for e in _pending(queue)}

    def test_a_rerun_of_the_storm_stamp_files_nothing(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'escalations')
        root = tmp_path / 'root'
        run = _regression_run(root)
        _file(queue, run, root, k=2)
        _, rerun = _file(queue, run, root, k=2)
        assert rerun.filed == ()
        assert len(_pending(queue)) == 1


class TestRunnerFailureFiling:
    def test_all_failures_in_a_run_collapse_to_one_escalation(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'escalations')
        root = tmp_path / 'root'
        failures = [_failure('e4-staleness-sweep'), _failure('e6-corpus-health', 'evaluate', None)]
        artifact = build_verdicts_artifact(
            [], run_stamp=_REGRESSION, alarm_limit=10, runner_failures=failures,
        )
        outcome = _mod().file_run_escalations(
            queue, [], artifact.storm_escape, artifact.runner_failures, _context(root),
        )
        (only,) = _pending(queue)
        assert outcome.filed == (only.id,)
        assert only.task_id == _PROGRAM_TASK
        assert only.dedupe_fingerprint == artifact.runner_failure_fingerprint
        assert len(only.evidence) == 2
        first, second = (entry['observation'] for entry in only.evidence)
        assert 'e4-staleness-sweep' in first and 'produce' in first and '3' in first
        assert 'e6-corpus-health' in second and 'evaluate' in second and 'ProducerExit' in second

    def test_a_later_failure_files_nothing_while_one_is_pending(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'escalations')
        root = tmp_path / 'root'
        quiet_storm = StormEscape(triggered=False, alarm_count=0, alarm_limit=10)
        file_run = _mod().file_run_escalations
        file_run(queue, [], quiet_storm, [_failure('e4-staleness-sweep')], _context(root))
        later = file_run(queue, [], quiet_storm, [_failure('e6-corpus-health')], _context(root, _QUIET))
        assert later.filed == ()
        assert len(_pending(queue)) == 1


class TestTheRunnerNeverResolves:
    def test_no_filed_record_changes_status_during_a_run(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'escalations')
        root = tmp_path / 'root'
        run = _regression_run(root)
        _, first = _file(queue, run, root)
        _file(queue, run, root, failures=[_failure('e4-staleness-sweep')])
        assert {_stored(queue, esc_id).status for esc_id in first.filed} == {'pending'}


def _config(**changes):
    return _mod().load_runner_config(COMMITTED_CONFIG).model_copy(update=changes)


def _seed_first_run(root: Path) -> None:
    """Baseline runs 0701-0703 on disk, with the 0701 grandfather snapshot persisted."""
    _seed(root, *_ALL_STAMPS[:3])
    _mod().evaluate_eval(root, _E1, _FIRST, _config().limits_config())


def _producing(fake_script: Path, stamp: str):
    return _fake(fake_script, source=_alpha_series(stamp))


def _failing_producer(fake_script: Path, eval_id: str = 'e4-staleness-sweep'):
    return _fake(fake_script, eval_id, extra=['--exit', '3', '--stderr', 'store unreachable'])


def _seeded_queue_dir(tmp_path: Path) -> Path:
    queue_dir = tmp_path / 'escalations'
    queue = EscalationQueue(queue_dir)
    queue.submit(Escalation(
        id=queue.make_id('unrelated'), task_id='unrelated', agent_role='test',
        severity='info', category='merge_error', summary='unrelated record',
    ))
    return queue_dir


def _main(root: Path, queue_dir: Path, producers) -> int:
    return _mod().main(
        ['--config', str(COMMITTED_CONFIG), '--artifact-root', str(root), '--queue-dir', str(queue_dir)],
        producers=producers,
    )


class TestRunProgram:
    def _regression(self, tmp_path, fake_script, **config_changes):
        root, queue = tmp_path / 'root', EscalationQueue(tmp_path / 'escalations')
        _seed_first_run(root)
        outcome = _mod().run_program(
            _config(**config_changes), [_producing(fake_script, _REGRESSION)], root, queue, _REGRESSION,
        )
        return root, queue, outcome

    def test_the_verdicts_are_persisted_stamped_and_current(self, tmp_path, fake_script):
        root, _, _ = self._regression(tmp_path, fake_script)
        stamped = root / f'verdicts-{_REGRESSION}.json'
        current = root / 'verdicts-current.json'
        assert stamped.read_bytes() == current.read_bytes()
        assert load_verdicts_artifact(current).run_stamp == _REGRESSION

    def test_filed_escalations_match_the_alarm_entries(self, tmp_path, fake_script):
        root, queue, _ = self._regression(tmp_path, fake_script)
        artifact = load_verdicts_artifact(root / 'verdicts-current.json')
        alarms = {e.fingerprint for e in artifact.entries if e.verdict == 'alarm'}
        assert alarms
        assert {e.dedupe_fingerprint for e in _pending(queue)} == alarms

    def test_a_storm_files_the_aggregate_the_artifact_names(self, tmp_path, fake_script):
        root, queue, _ = self._regression(tmp_path, fake_script, storm_alarm_limit=2)
        artifact = load_verdicts_artifact(root / 'verdicts-current.json')
        (only,) = _pending(queue)
        assert only.dedupe_fingerprint == artifact.storm_escape.aggregate_fingerprint

    def test_the_run_report_is_written_and_cited(self, tmp_path, fake_script):
        root, queue, _ = self._regression(tmp_path, fake_script)
        report = root / f'run-report-{_REGRESSION}.txt'
        assert report.is_file()
        assert all(str(report) in esc.detail for esc in _pending(queue))

    def test_a_first_run_reports_the_known_bad_baseline_and_files_nothing(
        self, tmp_path, fake_script,
    ):
        root, queue = tmp_path / 'root', EscalationQueue(tmp_path / 'escalations')
        _mod().run_program(_config(), [_producing(fake_script, _FIRST)], root, queue, _FIRST)
        report = (root / f'run-report-{_FIRST}.txt').read_text(encoding='utf-8')
        assert 'Initial-state baseline (known-bad, not alarmed)' in report
        for item in _failing_items(_FIRST):
            assert item in report
        assert _pending(queue) == []



class TestMain:
    def test_one_failing_producer_does_not_stop_the_others(
        self, tmp_path, fake_script, monkeypatch, capsys,
    ):
        monkeypatch.setenv(RUN_STAMP_ENV_VAR, _FIRST)
        root, queue_dir = tmp_path / 'root', _seeded_queue_dir(tmp_path)
        code = _main(root, queue_dir, [_failing_producer(fake_script), _producing(fake_script, _FIRST)])
        assert code == 1
        artifact = load_verdicts_artifact(root / 'verdicts-current.json')
        assert {e.eval_id for e in artifact.entries} == {_E1}
        assert [(f.scope, f.phase) for f in artifact.runner_failures] == [
            ('e4-staleness-sweep', 'produce'),
        ]
        (only,) = _pending(EscalationQueue(queue_dir))
        assert only.task_id == _PROGRAM_TASK
        summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert summary['failures'] == 1

    def test_a_run_with_alarms_but_no_failures_exits_0(self, tmp_path, fake_script, monkeypatch):
        monkeypatch.setenv(RUN_STAMP_ENV_VAR, _REGRESSION)
        root, queue_dir = tmp_path / 'root', _seeded_queue_dir(tmp_path)
        _seed_first_run(root)
        assert _main(root, queue_dir, [_producing(fake_script, _REGRESSION)]) == 0
        assert len(_pending(EscalationQueue(queue_dir))) == 3

    def test_an_unexpected_error_files_one_runner_failure_and_exits_1(self, tmp_path, monkeypatch):
        monkeypatch.setenv(RUN_STAMP_ENV_VAR, _FIRST)
        root = tmp_path / 'root-is-a-file'
        root.write_text('not a directory', encoding='utf-8')
        queue_dir = _seeded_queue_dir(tmp_path)
        assert _main(root, queue_dir, []) == 1
        (only,) = _pending(EscalationQueue(queue_dir))
        assert only.task_id == _PROGRAM_TASK
        assert 'runner' in only.evidence[0]['observation']

    def test_a_crash_before_any_artifact_cites_the_journal_not_unwritten_paths(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv(RUN_STAMP_ENV_VAR, _FIRST)
        root = tmp_path / 'root-is-a-file'
        root.write_text('not a directory', encoding='utf-8')
        queue_dir = _seeded_queue_dir(tmp_path)
        _main(root, queue_dir, [])
        (only,) = _pending(EscalationQueue(queue_dir))
        journal = _mod().JOURNAL_COMMAND
        for unwritten in (root / f'run-report-{_FIRST}.txt', root / f'verdicts-{_FIRST}.json'):
            assert str(unwritten) not in only.summary + only.detail
        assert 'run report: not written' in only.detail
        assert 'verdicts: not written' in only.detail
        assert journal in only.summary
        assert [entry['ref'] for entry in only.evidence] == [journal]

    def test_a_crash_after_the_verdicts_landed_cites_them_and_not_the_missing_report(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv(RUN_STAMP_ENV_VAR, _FIRST)
        root = tmp_path / 'root'
        (root / 'verdicts-current.json').mkdir(parents=True)
        queue_dir = _seeded_queue_dir(tmp_path)
        assert _main(root, queue_dir, []) == 1
        (only,) = _pending(EscalationQueue(queue_dir))
        stamped = root / f'verdicts-{_FIRST}.json'
        assert stamped.is_file()
        assert f'verdicts: {stamped}' in only.detail
        assert 'run report: not written' in only.detail
        assert str(root / f'run-report-{_FIRST}.txt') not in only.summary + only.detail

    @pytest.mark.skipif(os.geteuid() == 0, reason='root writes through a read-only directory')
    def test_a_crash_whose_own_filing_fails_still_exits_1_and_says_so(
        self, tmp_path, monkeypatch, capsys,
    ):
        monkeypatch.setenv(RUN_STAMP_ENV_VAR, _FIRST)
        root = tmp_path / 'root'
        (root / 'verdicts-current.json').mkdir(parents=True)
        queue_dir = _seeded_queue_dir(tmp_path)
        queue_dir.chmod(0o500)
        try:
            assert _main(root, queue_dir, []) == 1
        finally:
            queue_dir.chmod(0o700)
        assert 'could not file the runner-failure escalation' in capsys.readouterr().err
        assert _pending(EscalationQueue(queue_dir)) == []

    def test_a_relative_artifact_root_is_the_callers_not_the_producers(
        self, tmp_path, fake_script, monkeypatch,
    ):
        monkeypatch.setenv(RUN_STAMP_ENV_VAR, _FIRST)
        monkeypatch.chdir(tmp_path)
        queue_dir = _seeded_queue_dir(tmp_path)
        inner = _producing(fake_script, _FIRST)

        def build_argv(root: Path, eval_id: str, project_id: str) -> list[str]:
            if not root.is_absolute():
                pytest.fail(f'a producer runs in the repo root but was handed {root}')
            return inner.build_argv(root, eval_id, project_id)

        producer = _mod().Producer(eval_id=_E1, script=fake_script, build_argv=build_argv)
        assert _main(Path('artifacts'), queue_dir, [producer]) == 0
        assert (tmp_path / 'artifacts' / 'verdicts-current.json').is_file()

    def test_a_run_while_another_holds_the_root_exits_2_naming_the_holder(
        self, tmp_path, fake_script, monkeypatch, capsys,
    ):
        monkeypatch.setenv(RUN_STAMP_ENV_VAR, _FIRST)
        root, queue_dir = tmp_path / 'root', _seeded_queue_dir(tmp_path)
        producers = [_producing(fake_script, _FIRST)]
        with _mod().hold_run_lock(root, holder='pid=4242 run_stamp=20260630T031500Z'):
            assert _main(root, queue_dir, producers) == 2
        assert 'pid=4242 run_stamp=20260630T031500Z' in capsys.readouterr().err
        assert not (root / _E1).exists()
        assert not (root / 'verdicts-current.json').exists()
        assert _main(root, queue_dir, producers) == 0

    def test_a_rerun_at_the_pinned_stamp_files_nothing_and_rewrites_identical_verdicts(
        self, tmp_path, fake_script, monkeypatch,
    ):
        monkeypatch.setenv(RUN_STAMP_ENV_VAR, _REGRESSION)
        root, queue_dir = tmp_path / 'root', _seeded_queue_dir(tmp_path)
        _seed_first_run(root)
        producers = [_producing(fake_script, _REGRESSION)]
        _main(root, queue_dir, producers)
        first_bytes = (root / 'verdicts-current.json').read_bytes()
        first_pending = {e.id for e in _pending(EscalationQueue(queue_dir))}
        _main(root, queue_dir, producers)
        assert (root / 'verdicts-current.json').read_bytes() == first_bytes
        assert {e.id for e in _pending(EscalationQueue(queue_dir))} == first_pending

    def test_only_producer_eval_dirs_appear_under_the_root(self, tmp_path, fake_script, monkeypatch):
        monkeypatch.setenv(RUN_STAMP_ENV_VAR, _FIRST)
        root, queue_dir = tmp_path / 'root', _seeded_queue_dir(tmp_path)
        _main(root, queue_dir, [_failing_producer(fake_script), _producing(fake_script, _FIRST)])
        assert sorted(p.name for p in root.iterdir() if p.is_dir()) == [_E1]
