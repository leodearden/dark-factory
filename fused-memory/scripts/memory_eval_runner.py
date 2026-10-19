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
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from shared.memory_eval_limits import LimitsConfig
from shared.memory_eval_metrics import RUN_STAMP_ENV_VAR, metrics_artifact_path
from shared.memory_eval_verdicts import RunnerFailureRecord

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
