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
something failed to run (after filing), 2 when the config, the recon queue or
the run stamp failed preflight, or another run holds the artifact root's lock.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import traceback
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import yaml
from escalation.categories import EscalationCategory
from escalation.models import Escalation, EvidenceEntry
from escalation.queue import EscalationQueue
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from shared import safe_io
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
    run_stamp,
    validate_stamp,
)
from shared.memory_eval_verdicts import (
    RUNNER_FAILURE_FINGERPRINT,
    RunnerFailureRecord,
    StormEscape,
    VerdictEntry,
    VerdictsArtifact,
    build_verdicts_artifact,
    serialize_verdicts_artifact,
    verdict_entries,
)

from fused_memory.config.schema import DEFAULT_CONFIG_PATH as FUSED_MEMORY_DEFAULT_CONFIG_PATH
from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.eval.retrieval_probe import EVAL_ID as E1_EVAL_ID
from fused_memory.utils.target_store_preflight import (
    TargetStoreUnpopulated,
    assert_queue_dir_populated,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / 'fused-memory' / 'config' / 'memory-eval.yaml'
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / 'fused-memory' / 'data' / 'memory-evals'
VERDICTS_CURRENT_NAME = 'verdicts-current.json'
RUN_LOCK_NAME = '.runner.lock'
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
    """The project E4, E6 and E7 measure. E1 takes no project: it probes its topic registry's own census set."""
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
    Producer(E1_EVAL_ID, _SCRIPTS_DIR / 'memory_eval_retrieval_probe.py', _probe_argv),
    Producer('e4-staleness-sweep', _SCRIPTS_DIR / 'memory_eval_staleness_sweep.py', _sweep_argv),
    Producer('e6-corpus-health', _SCRIPTS_DIR / 'audit_duplicate_memories.py', _sweep_argv),
    Producer('e7-write-after-miss', _SCRIPTS_DIR / 'memory_eval_write_after_miss.py', _sweep_argv),
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


JOURNAL_COMMAND = 'journalctl --user -u fused-memory-memory-eval.service'


def eval_task_id(eval_id: str) -> str:
    return f'memory-eval-{eval_id}'


@dataclass(frozen=True)
class FilingContext:
    """The run every filing cites; an artifact path is None when the run crashed before writing it."""

    run_stamp: str
    project_id: str
    run_report_path: Path | None
    verdicts_path: Path | None

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


def _look_at(path: Path | None) -> str:
    return str(path) if path is not None else JOURNAL_COMMAND


def _artifact_lines(context: FilingContext) -> list[str]:
    return [
        f'{label}: {path}' if path is not None else f'{label}: not written; see {JOURNAL_COMMAND}'
        for label, path in (('run report', context.run_report_path), ('verdicts', context.verdicts_path))
    ]


def _num(value: float | None) -> str:
    return 'n/a' if value is None else f'{value:g}'


def _subject(entry: VerdictEntry) -> str:
    return f'{entry.metric_id}[{entry.item}]' if entry.item is not None else entry.metric_id


def _matching_alarm(run: EvalRun, entry: VerdictEntry) -> Alarm:
    for alarm in run.result.alarms:
        if (alarm.metric_id, alarm.item_key) == (entry.metric_id, entry.item):
            return alarm
    raise ValueError(f'{run.eval_id}: no evaluator alarm matches entry {_subject(entry)}.')


def _limit_line(entry: VerdictEntry, alarm: Alarm) -> str:
    if alarm.rule_kind == 'tripwire':
        return f'limit: {entry.limit_ref}'
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
        _limit_line(entry, alarm),
        f'baseline_run_stamps: {", ".join(run.result.baseline_run_stamps) or "none"}',
        f'evaluator: {alarm.detail}',
        f'limits artifact: {run.limits_path}',
        f'M1 metrics: {run.metrics_path}',
        f'M1 report: {run.report_path}',
        *_artifact_lines(context),
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
        *_artifact_lines(context),
        'Remediation: this many alarms at once usually share one cause (a store outage, a '
        'corpus migration, a probe change). Find that cause from the run report first.',
    ]
    return Filing(
        task_id=PROGRAM_TASK_ID,
        fingerprint=storm.aggregate_fingerprint,
        summary=(
            f'memory-eval storm at {context.run_stamp}: {storm.alarm_count} alarms exceed '
            f'K={storm.alarm_limit}; see {_look_at(context.run_report_path)}'
        ),
        detail='\n'.join(detail),
        evidence=(EvidenceEntry(
            observation=f'{storm.alarm_count} alarm entries > storm_alarm_limit {storm.alarm_limit}',
            measured_at=context.measured_at,
            ref=_look_at(context.verdicts_path),
        ),),
    )


def _failure_observation(failure: RunnerFailureRecord) -> str:
    exit_part = f' exit_code={failure.exit_code}' if failure.exit_code is not None else ''
    return f'{failure.scope} {failure.phase}: {failure.error_type}{exit_part}'


def failure_filing(failures: Sequence[RunnerFailureRecord], context: FilingContext) -> Filing:
    scopes = ', '.join(failure.scope for failure in failures)
    detail = [
        *(f'- {_failure_observation(f)}\n  {f.message}' for f in failures),
        *_artifact_lines(context),
        'Remediation: the named evals did not produce or judge a series this run, so they '
        f'carry no verdict. Fix the cause in the journal ({JOURNAL_COMMAND}); this stays '
        'open across runs until resolved.',
    ]
    return Filing(
        task_id=PROGRAM_TASK_ID,
        fingerprint=RUNNER_FAILURE_FINGERPRINT,
        summary=(
            f'memory-eval runner: {len(failures)} failure(s) at {context.run_stamp} '
            f'({scopes}); see {_look_at(context.run_report_path)}'
        ),
        detail='\n'.join(detail),
        evidence=tuple(
            EvidenceEntry(
                observation=_failure_observation(failure),
                measured_at=context.measured_at,
                ref=_look_at(context.run_report_path),
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
    parser.add_argument(
        '--artifact-root', type=lambda value: Path(value).resolve(), default=DEFAULT_ARTIFACT_ROOT,
        help='resolved against the current directory before any producer runs',
    )
    parser.add_argument(
        '--queue-dir', type=Path, default=None,
        help="recon escalation queue (default: fused-memory's "
             'reconciliation.escalation_queue_dir, resolved against the repo root)',
    )
    return parser.parse_args(argv)


def verdicts_path(root: Path, stamp: str) -> Path:
    return root / f'verdicts-{validate_stamp(stamp, what="verdicts stamp")}.json'


def run_report_path(root: Path, stamp: str) -> Path:
    return root / f'run-report-{validate_stamp(stamp, what="run report stamp")}.txt'


def _write_text(path: Path, text: str) -> None:
    safe_io.atomic_write_text(path, text, encoding='utf-8', mode=0o600)


def write_verdicts(artifact: VerdictsArtifact, root: Path) -> tuple[Path, Path]:
    """Write identical bytes to the stamped file, then to verdicts-current.json."""
    root.mkdir(parents=True, exist_ok=True)
    text = serialize_verdicts_artifact(artifact)
    stamped = verdicts_path(root, artifact.run_stamp)
    current = root / VERDICTS_CURRENT_NAME
    _write_text(stamped, text)
    _write_text(current, text)
    return stamped, current


def _eval_section(run: EvalRun) -> list[str]:
    counts = Counter(entry.verdict for entry in run.entries)
    lines = [f'{run.eval_id}: ' + ', '.join(f'{v}={counts[v]}' for v in sorted(counts))]
    lines.extend(
        f'  ALARM {_subject(e)} value {_num(e.value)} ({e.limit_ref})'
        for e in run.entries if e.verdict == 'alarm'
    )
    return lines


def _initial_state_section(evals: Sequence[EvalRun]) -> list[str]:
    snapshotted = [
        (run, entry) for run in evals for entry in run.entries
        if entry.metric_status == 'baseline_snapshot'
    ]
    if not snapshotted:
        return []
    known_bad = [
        f'  {run.eval_id} {_subject(entry)}'
        for run, entry in snapshotted if entry.verdict == 'grandfathered'
    ]
    return ['', 'Initial-state baseline (known-bad, not alarmed):', *(known_bad or ['  none'])]


def render_run_report(artifact: VerdictsArtifact, evals: Sequence[EvalRun]) -> str:
    storm = artifact.storm_escape
    lines = [f'memory-eval run {artifact.run_stamp}', '']
    for run in evals:
        lines.extend(_eval_section(run))
    lines.append(
        f'storm escape: {"TRIGGERED" if storm.triggered else "not triggered"} '
        f'({storm.alarm_count} alarm(s), K={storm.alarm_limit})'
    )
    if artifact.runner_failures:
        lines.append('runner failures:')
        for failure in artifact.runner_failures:
            lines.append(f'  {_failure_observation(failure)}')
            lines.extend(f'    {line}' for line in failure.message.splitlines())
    lines.extend(_initial_state_section(evals))
    return '\n'.join(lines) + '\n'


@dataclass(frozen=True)
class RunOutcome:
    artifact: VerdictsArtifact
    filing: FilingOutcome
    verdicts_path: Path
    run_report_path: Path


def _produce_all(
    config: RunnerConfig, producers: Sequence[Producer], root: Path, stamp: str,
) -> list[RunnerFailureRecord]:
    failures: list[RunnerFailureRecord] = []
    for producer in producers:
        failure = run_producer(
            producer, root=root, stamp=stamp, project_id=config.project_id,
            timeout=config.producer_timeout_secs,
        )
        if failure is not None:
            failures.append(failure)
    return failures


def run_program(
    config: RunnerConfig,
    producers: Sequence[Producer],
    root: Path,
    queue: EscalationQueue,
    stamp: str,
) -> RunOutcome:
    """produce -> evaluate -> write verdicts and run report -> file escalations."""
    produce_failures = _produce_all(config, producers, root, stamp)
    failed = {failure.scope for failure in produce_failures}
    produced = [p.eval_id for p in producers if p.eval_id not in failed]
    runs, evaluate_failures = evaluate_evals(root, produced, stamp, config.limits_config())
    artifact = build_verdicts_artifact(
        [(run.series, run.result) for run in runs],
        run_stamp=stamp,
        alarm_limit=config.storm_alarm_limit,
        runner_failures=[*produce_failures, *evaluate_failures],
    )
    stamped, _ = write_verdicts(artifact, root)
    report = run_report_path(root, stamp)
    _write_text(report, render_run_report(artifact, runs))
    context = FilingContext(
        run_stamp=stamp, project_id=config.project_id,
        run_report_path=report, verdicts_path=stamped,
    )
    filing = file_run_escalations(
        queue, runs, artifact.storm_escape, artifact.runner_failures, context,
    )
    return RunOutcome(artifact=artifact, filing=filing, verdicts_path=stamped, run_report_path=report)


def _if_written(path: Path) -> Path | None:
    return path if path.is_file() else None


def _file_runner_crash(
    queue: EscalationQueue, exc: Exception, config: RunnerConfig, root: Path, stamp: str,
) -> FilingOutcome:
    """Best effort: one phase='runner' failure escalation. A filing error is logged, never raised."""
    failure = RunnerFailureRecord(
        scope='program', phase='runner', error_type=type(exc).__name__, message=str(exc),
    )
    context = FilingContext(
        run_stamp=stamp, project_id=config.project_id,
        run_report_path=_if_written(run_report_path(root, stamp)),
        verdicts_path=_if_written(verdicts_path(root, stamp)),
    )
    filing = failure_filing([failure], context)
    try:
        esc_id = file_once(queue, filing, project_id=config.project_id)
    except Exception:
        print('memory_eval_runner: could not file the runner-failure escalation:', file=sys.stderr)
        traceback.print_exc()
        return FilingOutcome(filed=(), suppressed=())
    if esc_id is None:
        return FilingOutcome(filed=(), suppressed=(filing.fingerprint,))
    return FilingOutcome(filed=(esc_id,), suppressed=())


def _print_summary(stamp: str, filing: FilingOutcome, *, failures: int, alarms: int) -> None:
    """One JSON line on stdout, for the journal."""
    print(json.dumps({
        'run_stamp': stamp, 'filed': len(filing.filed), 'suppressed': len(filing.suppressed),
        'failures': failures, 'alarms': alarms,
    }, sort_keys=True))


class _PreflightRefused(Exception):
    """Carries the message main() prints before exiting EXIT_PREFLIGHT."""


class RunLockHeld(Exception):
    """Another run holds the artifact root: this one must write and file nothing."""


def hold_run_lock(root: Path, *, holder: str) -> TextIO:
    """Take the artifact root's exclusive run lock, naming *holder* in it; closing the handle releases it.

    Overlapping runs could both pass has_open_l1 before either submits, and could
    interleave one eval's limits-current.json read-modify-write.
    """
    root.mkdir(parents=True, exist_ok=True)
    path = root / RUN_LOCK_NAME
    handle = os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT, 0o600), 'r+', encoding='utf-8')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        held_by = handle.read().strip() or 'a run that has not named itself yet'
        handle.close()
        raise RunLockHeld(f'{path} is held by {held_by}') from None
    except BaseException:
        handle.close()
        raise
    handle.truncate(0)
    handle.write(holder + '\n')
    handle.flush()
    return handle


def _preflight(args: argparse.Namespace) -> tuple[RunnerConfig, EscalationQueue, str]:
    try:
        config = load_runner_config(args.config)
    except RunnerConfigError as exc:
        raise _PreflightRefused(f'config refused: {exc}') from exc
    try:
        queue_dir = args.queue_dir if args.queue_dir is not None else _default_queue_dir()
    except (RuntimeError, ValidationError) as exc:
        config_path = Path(os.environ.get('CONFIG_PATH', FUSED_MEMORY_DEFAULT_CONFIG_PATH)).absolute()
        raise _PreflightRefused(f'fused-memory config refused (CONFIG_PATH={config_path}): {exc}') from exc
    try:
        assert_queue_dir_populated(queue_dir, operation='memory_eval_runner')
    except TargetStoreUnpopulated as exc:
        raise _PreflightRefused(f'recon escalation queue {queue_dir}: {exc}') from exc
    try:
        stamp = run_stamp()
    except ValueError as exc:
        raise _PreflightRefused(f'run stamp refused: {exc}') from exc
    return config, EscalationQueue(queue_dir), stamp


def _refused(exc: Exception) -> int:
    print(f'memory_eval_runner: {exc}', file=sys.stderr)
    return EXIT_PREFLIGHT


def _crashed(
    queue: EscalationQueue, exc: Exception, config: RunnerConfig, root: Path, stamp: str,
) -> int:
    traceback.print_exc()
    filing = _file_runner_crash(queue, exc, config, root, stamp)
    _print_summary(stamp, filing, failures=1, alarms=0)
    return EXIT_RUN_FAILED


def _run_and_report(
    config: RunnerConfig, producers: Sequence[Producer], root: Path, queue: EscalationQueue, stamp: str,
) -> int:
    try:
        outcome = run_program(config, producers, root, queue, stamp)
    except Exception as exc:
        return _crashed(queue, exc, config, root, stamp)
    artifact = outcome.artifact
    _print_summary(
        stamp, outcome.filing,
        failures=len(artifact.runner_failures), alarms=artifact.storm_escape.alarm_count,
    )
    return EXIT_RUN_FAILED if artifact.runner_failures else EXIT_OK


def main(argv: Sequence[str] | None = None, *, producers: Sequence[Producer] = DEFAULT_PRODUCERS) -> int:
    args = _parse_args(argv)
    root = args.artifact_root
    try:
        config, queue, stamp = _preflight(args)
    except _PreflightRefused as exc:
        return _refused(exc)
    try:
        lock = hold_run_lock(root, holder=f'pid={os.getpid()} run_stamp={stamp}')
    except RunLockHeld as exc:
        return _refused(exc)
    except OSError as exc:
        return _crashed(queue, exc, config, root, stamp)
    with lock:
        return _run_and_report(config, producers, root, queue, stamp)


if __name__ == '__main__':
    sys.exit(main())
