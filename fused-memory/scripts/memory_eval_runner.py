"""Scheduled memory-eval runner: produce, evaluate, publish verdicts, escalate.

Run once a day by ``scripts/fused-memory-memory-eval.timer``
(``docs/prds/memory-eval-program.md`` M2 and M3; decisions D1, D4 and D10).
Each run:

1. runs every registered producer as a read-only subprocess, all under one
   shared ``MEMORY_EVAL_RUN_STAMP``;
2. evaluates each eval's new series against its trailing window, resuming from
   that eval's ``limits-current.json``. An eval with no limits file is on its
   first run, and that run IS its grandfather snapshot (D1): it alarms nothing;
3. writes ``verdicts-<STAMP>.json``, ``verdicts-current.json`` and
   ``run-report-<STAMP>.txt`` at the artifact root;
4. files ``eval_regression`` escalations into the recon queue: one per new
   alarm fingerprint, or ONE aggregate when the run's alarms exceed the storm
   limit K, plus one escalation covering everything that failed to run.

Exit status: 0 when the run completed (an alarm is not a failure), 1 when
something failed to run (after filing), 2 when the config or the recon queue
failed preflight.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from escalation.categories import EscalationCategory
from escalation.models import Escalation, EvidenceEntry
from escalation.queue import EscalationQueue
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from shared.memory_eval_limits import (
    Alarm,
    EvaluationResult,
    LimitsConfig,
    evaluate_series,
    limits_artifact_path,
    load_limits_artifact,
    write_limits_artifact,
)
from shared.memory_eval_metrics import (
    RUN_STAMP_ENV_VAR,
    MetricSeries,
    load_metric_series,
    load_series_window,
    metrics_artifact_path,
    report_artifact_path,
)
from shared.memory_eval_verdicts import (
    RUNNER_FAILURE_FINGERPRINT,
    RunnerFailureRecord,
    StormEscape,
    VerdictEntry,
    verdict_entries,
)

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.utils.target_store_preflight import TargetStoreMissing, assert_queue_dir_populated

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / 'fused-memory' / 'config' / 'memory-eval.yaml'
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / 'fused-memory' / 'data' / 'memory-evals'
_STDERR_TAIL_CHARS = 2000

EXIT_OK = 0
EXIT_RUN_FAILED = 1
EXIT_PREFLIGHT = 2


class RunnerConfigError(ValueError):
    """The runner config file is missing, unparseable or not a valid declaration."""


class RunnerConfig(BaseModel):
    """Operator declarations; every field is required because none has a defensible default (G6)."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    project_id: str = Field(min_length=1)
    runs_per_quarter: int
    false_alarm_budget: float
    min_samples: int
    baseline_window: int
    storm_alarm_limit: int = Field(gt=0)
    producer_timeout_secs: float = Field(gt=0)

    @model_validator(mode='after')
    def _check_limits(self) -> RunnerConfig:
        self.limits_config()
        return self

    def limits_config(self) -> LimitsConfig:
        return LimitsConfig(
            false_alarm_budget=self.false_alarm_budget,
            runs_per_quarter=self.runs_per_quarter,
            min_samples=self.min_samples,
            baseline_window=self.baseline_window,
        )


def load_runner_config(path: str | Path) -> RunnerConfig:
    target = Path(path)
    try:
        return RunnerConfig.model_validate(yaml.safe_load(target.read_text(encoding='utf-8')))
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise RunnerConfigError(f'{target}: {exc}') from exc


def resolve_recon_queue_dir(queue_dir_setting: str, repo_root: Path) -> Path:
    """The recon queue setting is relative to the repo root, fused-memory.service's WorkingDirectory."""
    setting = Path(queue_dir_setting)
    return setting if setting.is_absolute() else repo_root / setting


ArgvBuilder = Callable[[Path, str, str], list[str]]
"""``(artifact_root, eval_id, project_id) -> producer arguments``."""


@dataclass(frozen=True)
class Producer:
    eval_id: str
    script: Path
    build_argv: ArgvBuilder


def _probe_argv(root: Path, eval_id: str, project_id: str) -> list[str]:
    return ['--out-root', str(root)]


def _sweep_argv(root: Path, eval_id: str, project_id: str) -> list[str]:
    return ['--metrics-root', str(root), '--eval-id', eval_id, '--project-id', project_id]


DEFAULT_PRODUCERS: tuple[Producer, ...] = (
    Producer('e1-retrieval-health', _SCRIPTS_DIR / 'memory_eval_retrieval_probe.py', _probe_argv),
    Producer('e4-staleness-sweep', _SCRIPTS_DIR / 'memory_eval_staleness_sweep.py', _sweep_argv),
    Producer('e6-corpus-health', _SCRIPTS_DIR / 'audit_duplicate_memories.py', _sweep_argv),
)
"""Every scheduled producer, run read-only: no argv here may carry --apply or --no-metrics."""


def _text(stream: str | bytes | None) -> str:
    if isinstance(stream, bytes):
        return stream.decode('utf-8', errors='replace')
    return stream or ''


def _relay(producer: Producer, *streams: str | bytes | None) -> None:
    print(f'--- producer {producer.eval_id} output ---', file=sys.stderr)
    for stream in streams:
        text = _text(stream)
        if text:
            sys.stderr.write(text if text.endswith('\n') else text + '\n')


def _produce_failure(
    producer: Producer,
    error_type: str,
    what: str,
    stderr: str | bytes | None = None,
    exit_code: int | None = None,
) -> RunnerFailureRecord:
    tail = _text(stderr)[-_STDERR_TAIL_CHARS:].strip()
    message = f'{producer.script.name}: {what}' + (f'; stderr tail:\n{tail}' if tail else '')
    return RunnerFailureRecord(
        scope=producer.eval_id, phase='produce', error_type=error_type,
        message=message, exit_code=exit_code,
    )


def run_producer(
    producer: Producer, *, root: Path, stamp: str, project_id: str, timeout: float,
) -> RunnerFailureRecord | None:
    """Run one producer under the shared stamp; None when its series landed."""
    argv = [sys.executable, str(producer.script), *producer.build_argv(root, producer.eval_id, project_id)]
    try:
        completed = subprocess.run(
            argv, cwd=REPO_ROOT, env=os.environ | {RUN_STAMP_ENV_VAR: stamp},
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        _relay(producer, exc.stdout, exc.stderr)
        return _produce_failure(producer, 'TimeoutExpired', f'exceeded {timeout}s', exc.stderr)
    except OSError as exc:
        return _produce_failure(producer, type(exc).__name__, f'could not start: {exc}')
    _relay(producer, completed.stdout, completed.stderr)
    if completed.returncode != 0:
        return _produce_failure(
            producer, 'ProducerExit', f'exited {completed.returncode}',
            completed.stderr, completed.returncode,
        )
    expected = metrics_artifact_path(root, producer.eval_id, stamp)
    if not expected.is_file():
        return _produce_failure(
            producer, 'MissingArtifact', f'exited 0 without writing {expected}', completed.stderr, 0,
        )
    return None


@dataclass(frozen=True)
class EvalRun:
    eval_id: str
    series: MetricSeries
    result: EvaluationResult
    metrics_path: Path
    report_path: Path
    limits_path: Path
    entries: tuple[VerdictEntry, ...]


def evaluate_eval(root: Path, eval_id: str, stamp: str, limits_config: LimitsConfig) -> EvalRun:
    """Judge this run's series, resuming from (and rewriting) the eval's limits-current.json.

    Nothing is carried in memory between runs: the grandfather state is re-read
    from disk every time, so a missing limits file means a first run, which is
    the grandfather snapshot (D1).
    """
    metrics_path = metrics_artifact_path(root, eval_id, stamp)
    series = load_metric_series(metrics_path)
    if (series.eval_id, series.run_stamp) != (eval_id, stamp):
        raise ValueError(
            f'{metrics_path} holds {series.eval_id}@{series.run_stamp}, not {eval_id}@{stamp}.'
        )
    window = load_series_window(
        root, eval_id, limit=limits_config.baseline_window, before_stamp=stamp,
    )
    limits_path = limits_artifact_path(root, eval_id)
    prior = load_limits_artifact(limits_path) if limits_path.is_file() else None
    result = evaluate_series(
        series,
        window,
        limits_config,
        frozenset(prior.grandfather_set) if prior is not None else None,
        prior.snapshotted_metric_ids if prior is not None else None,
    )
    write_limits_artifact(result, limits_path)
    return EvalRun(
        eval_id=eval_id,
        series=series,
        result=result,
        metrics_path=metrics_path,
        report_path=report_artifact_path(root, eval_id, stamp),
        limits_path=limits_path,
        entries=verdict_entries(series, result),
    )


def evaluate_evals(
    root: Path, eval_ids: Sequence[str], stamp: str, limits_config: LimitsConfig,
) -> tuple[list[EvalRun], list[RunnerFailureRecord]]:
    """Evaluate each eval in turn; one eval's failure is recorded and never stops the rest."""
    runs: list[EvalRun] = []
    failures: list[RunnerFailureRecord] = []
    for eval_id in eval_ids:
        try:
            runs.append(evaluate_eval(root, eval_id, stamp, limits_config))
        except Exception as exc:
            failures.append(RunnerFailureRecord(
                scope=eval_id, phase='evaluate', error_type=type(exc).__name__, message=str(exc),
            ))
    return runs, failures


AGENT_ROLE = 'memory-eval-runner'
PROGRAM_TASK_ID = 'memory-eval-program'

_REMEDIATION: dict[str, str] = {
    'tripwire': (
        'Remediation: a probe item that passed before now fails. Read the failing item in '
        'the M1 report, fix the regression in the corpus or retrieval path, and resolve '
        'this once a later run shows the item passing.'
    ),
    'proportion': (
        'Remediation: the rate moved the bad way beyond the derived alpha. Compare this '
        "run's M1 report with the baseline runs to find the change that moved it."
    ),
    'count': (
        'Remediation: the event count moved the bad way beyond the derived alpha. Compare '
        "this run's M1 report with the baseline runs to find what produced the events."
    ),
}


def eval_task_id(eval_id: str) -> str:
    return f'memory-eval-{eval_id}'


@dataclass(frozen=True)
class FilingContext:
    run_stamp: str
    project_id: str
    run_report_path: Path
    verdicts_path: Path

    @property
    def measured_at(self) -> str:
        return f'run_stamp={self.run_stamp}'


@dataclass(frozen=True)
class Filing:
    task_id: str
    fingerprint: str
    summary: str
    detail: str
    evidence: tuple[EvidenceEntry, ...]


@dataclass(frozen=True)
class FilingOutcome:
    filed: tuple[str, ...]
    suppressed: tuple[str, ...]


def file_once(queue: EscalationQueue, filing: Filing, *, project_id: str) -> str | None:
    """The one place this runner builds an Escalation; None when that fingerprint is still open."""
    if queue.has_open_l1(
        filing.task_id,
        category=EscalationCategory.EVAL_REGRESSION,
        dedupe_fingerprint=filing.fingerprint,
    ):
        return None
    return queue.submit(Escalation(
        id=queue.make_id(filing.task_id),
        task_id=filing.task_id,
        agent_role=AGENT_ROLE,
        severity='blocking',
        category=EscalationCategory.EVAL_REGRESSION,
        summary=filing.summary,
        detail=filing.detail,
        level=1,
        dedupe_fingerprint=filing.fingerprint,
        project_id=project_id,
        evidence=list(filing.evidence),
    ))


def _num(value: float | None) -> str:
    return 'n/a' if value is None else f'{value:g}'


def _subject(entry: VerdictEntry) -> str:
    return f'{entry.metric_id}[{entry.item}]' if entry.item is not None else entry.metric_id


def _matching_alarm(run: EvalRun, entry: VerdictEntry) -> Alarm:
    for alarm in run.result.alarms:
        if (alarm.metric_id, alarm.item_key) == (entry.metric_id, entry.item):
            return alarm
    raise ValueError(f'{run.eval_id}: no evaluator alarm matches entry {_subject(entry)}.')


def _limit_line(run: EvalRun, alarm: Alarm) -> str:
    if alarm.rule_kind == 'tripwire':
        return f'limit: grandfather_set_hash={run.result.grandfather_hash}'
    return f'limit: alpha={alarm.alpha!r} p_value={alarm.p_value!r}'


def alarm_filing(run: EvalRun, entry: VerdictEntry, context: FilingContext) -> Filing:
    alarm = _matching_alarm(run, entry)
    subject = _subject(entry)
    detail = [
        f'eval_id: {run.eval_id}',
        f'metric_id: {entry.metric_id}',
        *([f'item: {entry.item}'] if entry.item is not None else []),
        f'rule_kind: {entry.rule_kind}',
        *([f'direction: {alarm.direction}'] if alarm.direction is not None else []),
        f'value: {_num(alarm.value)}',
        f'baseline: {_num(alarm.baseline)}',
        _limit_line(run, alarm),
        f'baseline_run_stamps: {", ".join(run.result.baseline_run_stamps) or "none"}',
        f'evaluator: {alarm.detail}',
        f'limits artifact: {run.limits_path}',
        f'M1 metrics: {run.metrics_path}',
        f'M1 report: {run.report_path}',
        f'run report: {context.run_report_path}',
        f'verdicts: {context.verdicts_path}',
        _REMEDIATION[entry.rule_kind],
    ]
    return Filing(
        task_id=eval_task_id(run.eval_id),
        fingerprint=entry.fingerprint,
        summary=(
            f'memory-eval {run.eval_id}: {subject} regressed at {context.run_stamp} '
            f'(value {_num(alarm.value)} vs baseline {_num(alarm.baseline)})'
        ),
        detail='\n'.join(detail),
        evidence=(EvidenceEntry(
            observation=f'{run.eval_id} {subject}: {alarm.detail}',
            measured_at=context.measured_at,
            ref=str(run.metrics_path),
        ),),
    )


def storm_filing(
    storm: StormEscape, alarm_entries: Sequence[VerdictEntry], context: FilingContext,
) -> Filing:
    if storm.aggregate_fingerprint is None:
        raise ValueError('storm_filing: the storm block is not triggered.')
    detail = [
        f'{storm.alarm_count} alarm(s) in run {context.run_stamp} exceed storm_alarm_limit '
        f'K={storm.alarm_limit}; per-metric escalations were collapsed into this one.',
        *(f'- {e.eval_id} {_subject(e)}: value {_num(e.value)}' for e in alarm_entries),
        f'run report: {context.run_report_path}',
        f'verdicts: {context.verdicts_path}',
        'Remediation: this many alarms at once usually share one cause (a store outage, a '
        'corpus migration, a probe change). Find that cause from the run report first.',
    ]
    return Filing(
        task_id=PROGRAM_TASK_ID,
        fingerprint=storm.aggregate_fingerprint,
        summary=(
            f'memory-eval storm at {context.run_stamp}: {storm.alarm_count} alarms exceed '
            f'K={storm.alarm_limit}; see {context.run_report_path}'
        ),
        detail='\n'.join(detail),
        evidence=(EvidenceEntry(
            observation=f'{storm.alarm_count} alarm entries > storm_alarm_limit {storm.alarm_limit}',
            measured_at=context.measured_at,
            ref=str(context.verdicts_path),
        ),),
    )


def _failure_observation(failure: RunnerFailureRecord) -> str:
    exit_part = f' exit_code={failure.exit_code}' if failure.exit_code is not None else ''
    return f'{failure.scope} {failure.phase}: {failure.error_type}{exit_part}'


def failure_filing(failures: Sequence[RunnerFailureRecord], context: FilingContext) -> Filing:
    scopes = ', '.join(failure.scope for failure in failures)
    detail = [
        *(f'- {_failure_observation(f)}\n  {f.message}' for f in failures),
        f'run report: {context.run_report_path}',
        f'verdicts: {context.verdicts_path}',
        'Remediation: the named evals did not produce or judge a series this run, so they '
        'carry no verdict. Fix the cause in the journal (journalctl --user -u '
        'fused-memory-memory-eval.service); this stays open across runs until resolved.',
    ]
    return Filing(
        task_id=PROGRAM_TASK_ID,
        fingerprint=RUNNER_FAILURE_FINGERPRINT,
        summary=(
            f'memory-eval runner: {len(failures)} failure(s) at {context.run_stamp} '
            f'({scopes}); see {context.run_report_path}'
        ),
        detail='\n'.join(detail),
        evidence=tuple(
            EvidenceEntry(
                observation=_failure_observation(failure),
                measured_at=context.measured_at,
                ref=str(context.run_report_path),
            )
            for failure in failures
        ),
    )


def file_run_escalations(
    queue: EscalationQueue,
    evals: Sequence[EvalRun],
    storm: StormEscape,
    failures: Sequence[RunnerFailureRecord],
    context: FilingContext,
) -> FilingOutcome:
    """File this run's eval_regression escalations; never resolves anything."""
    alarms = [(run, entry) for run in evals for entry in run.entries if entry.verdict == 'alarm']
    if storm.triggered:
        filings = [storm_filing(storm, [entry for _, entry in alarms], context)]
    else:
        filings = [alarm_filing(run, entry, context) for run, entry in alarms]
    if failures:
        filings.append(failure_filing(failures, context))
    filed: list[str] = []
    suppressed: list[str] = []
    for filing in filings:
        esc_id = file_once(queue, filing, project_id=context.project_id)
        if esc_id is None:
            suppressed.append(filing.fingerprint)
        else:
            filed.append(esc_id)
    return FilingOutcome(filed=tuple(filed), suppressed=tuple(suppressed))


def _default_queue_dir() -> Path:
    setting = FusedMemoryConfig().reconciliation.escalation_queue_dir
    return resolve_recon_queue_dir(setting, REPO_ROOT)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='memory_eval_runner',
        description='Run the scheduled memory evals and file eval_regression escalations.',
    )
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--artifact-root', type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        '--queue-dir', type=Path, default=None,
        help="recon escalation queue (default: fused-memory's "
             'reconciliation.escalation_queue_dir, resolved against the repo root)',
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        load_runner_config(args.config)
    except RunnerConfigError as exc:
        print(f'memory_eval_runner: config refused: {exc}', file=sys.stderr)
        return EXIT_PREFLIGHT
    queue_dir = args.queue_dir if args.queue_dir is not None else _default_queue_dir()
    try:
        assert_queue_dir_populated(queue_dir, operation='memory_eval_runner')
    except TargetStoreMissing as exc:
        print(f'memory_eval_runner: recon escalation queue {queue_dir}: {exc}', file=sys.stderr)
        return EXIT_PREFLIGHT
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
