"""Run the merge role's per-module test commands on one tree, uncached, and report each failure under its module."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from shared.proc_group import terminate_process_group

from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.offline_lane_records import EndedBy, classify_exit
from orchestrator.verify import (
    FailureKind,
    PytestFailure,
    _provision_main_tip_worktree,
    _resolve_verify_env,
    _target_subprocess_env,
    parse_pytest_failures,
)
from orchestrator.verify_cache.modes import VERIFY_CACHE_ENV, CacheMode

AUDIT_COMMAND = 'verify-cache-audit'
DEFAULT_LOG_DIR = '.df-verify-cache-audit'

# pytest.ExitCode.OK and pytest.ExitCode.TESTS_FAILED; named here so pytest is not imported.
PYTEST_OK = 0
PYTEST_TESTS_FAILED = 1

_NODE_SEPARATOR = '::'

# The verify-cache-audit lane command sets these only so the runner's config load resolves
# dark-factory-orchestrator.yaml's project_root to the lane tree; the gate's children never carry them.
RUNNER_ONLY_ENV = ('PROJECT_ROOT',)


@dataclass(frozen=True)
class AuditNode:
    """An audit failure as the lane sees it: the module prefix, then the module-relative pytest node id."""

    module: str
    test: str

    def __str__(self) -> str:
        return f'{self.module}{_NODE_SEPARATOR}{self.test}'

    @classmethod
    def parse(cls, node: str) -> AuditNode | None:
        """The audit node *node* names, or None for an id the audit never prints."""
        module, separator, test = node.partition(_NODE_SEPARATOR)
        if not separator or not module or not test or module.endswith('.py'):
            return None
        return cls(module, test)


class ModuleEnd(StrEnum):
    EXIT = 'exit'
    SIGNAL = 'signal'
    TIMEOUT = 'timeout'


@dataclass(frozen=True)
class ModuleRun:
    module: str
    returncode: int
    ended: ModuleEnd
    failures: tuple[PytestFailure, ...]
    log_path: Path

    @property
    def completed(self) -> bool:
        """Whether the module's pytest session reached a verdict: all passed, or named what failed."""
        if self.ended is not ModuleEnd.EXIT:
            return False
        return self.returncode == PYTEST_OK or (self.returncode == PYTEST_TESTS_FAILED and bool(self.failures))


def audit_modules(config: OrchestratorConfig) -> tuple[ModuleConfig, ...]:
    """The modules whose test leg the full-breadth merge role runs, in prefix order.

    Lint and type legs name no test, so the lane could only read their red as unconfirmable.
    """
    declared = (mc for mc in config.module_configs_or_empty.values() if mc.test_command)
    return tuple(sorted(declared, key=lambda mc: mc.prefix))


def module_env(config: OrchestratorConfig, mc: ModuleConfig, tree: Path) -> dict[str, str]:
    """*mc*'s child environment as the merge gate builds it, with the verify cache forced off."""
    extra = {**_resolve_verify_env(config, mc, role='merge'), VERIFY_CACHE_ENV: CacheMode.OFF.value}
    env = _target_subprocess_env(extra, worktree=tree)
    for name in RUNNER_ONLY_ENV:
        if name not in extra:
            env.pop(name, None)
    return env


def module_timeout(config: OrchestratorConfig, mc: ModuleConfig) -> float:
    if mc.verify_command_timeout_secs is not None:
        return mc.verify_command_timeout_secs
    return config.verify_command_timeout_secs


async def run_module(tree: Path, mc: ModuleConfig, env: dict[str, str], timeout: float, log_dir: Path) -> ModuleRun:
    """Run *mc*'s test command in *tree* with its output in its own log, killing its group past *timeout*."""
    if not mc.test_command:
        raise ValueError(f'module {mc.prefix!r} declares no test_command')
    log_path = log_dir / f'{mc.prefix.replace("/", "--")}.log'
    with log_path.open('wb') as log:
        proc = await asyncio.create_subprocess_exec(
            'sh', '-c', mc.test_command, cwd=str(tree), env=env,
            stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True,
        )
        returncode, ended = await _wait_bounded(proc, timeout)
    failures = await asyncio.to_thread(_parse_log, log_path)
    return ModuleRun(mc.prefix, returncode, ended, failures, log_path)


async def _wait_bounded(proc: asyncio.subprocess.Process, timeout: float) -> tuple[int, ModuleEnd]:
    pgid = proc.pid
    try:
        returncode = await asyncio.wait_for(proc.wait(), timeout)
    except TimeoutError:
        await terminate_process_group(proc, pgid)
        return await proc.wait(), ModuleEnd.TIMEOUT
    finally:
        if proc.returncode is None:
            await terminate_process_group(proc, pgid)
    return returncode, ModuleEnd.SIGNAL if classify_exit(returncode) is EndedBy.SIGNAL else ModuleEnd.EXIT


def _parse_log(log_path: Path) -> tuple[PytestFailure, ...]:
    return parse_pytest_failures(log_path.read_bytes().decode(errors='replace'))


def render_report(runs: Sequence[ModuleRun], elapsed: float) -> list[str]:
    """The report's lines: each failure module-qualified, each module's end, then a pytest-shaped summary.

    The summary line is the lane's proof of a complete session, so it is printed only when every module completed.
    """
    lines = [_failure_line(run.module, failure) for run in runs for failure in run.failures]
    lines += [f'# {run.module}: rc={run.returncode} ended={run.ended} log={run.log_path}' for run in runs]
    if runs and all(run.completed for run in runs):
        kinds = [failure.kind for run in runs for failure in run.failures]
        failed, errors = kinds.count(FailureKind.FAILED), kinds.count(FailureKind.ERROR)
        lines.append(f'== {failed} failed, {errors} errors in {elapsed:.1f}s ==')
    return lines


def _failure_line(module: str, failure: PytestFailure) -> str:
    line = f'{failure.kind.upper()} {AuditNode(module, failure.node_id)}'
    reason = _inert(failure.first_line)
    return f'{line} - {reason}' if reason else line


def _inert(text: str) -> str:
    """*text* on one line with no node separator, so no parser surface can lift an unqualified id out of it."""
    return ' '.join(text.replace(_NODE_SEPARATOR, ':').split())


async def run_audit(tree: Path, config: OrchestratorConfig, log_dir: Path) -> int:
    """Run every audit module on *tree* in turn, print the report, and return 0 iff every module exited 0.

    The tree is first provisioned as the merge gate provisions its own
    (verify.py::_provision_main_tip_worktree); a failed provision is a non-zero exit naming
    no test, so the lane reads it as Unconfirmable. A tree with no module to audit is a
    failure, never a vacuous green.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    if not await _provision_main_tip_worktree(tree, config, role='merge'):
        print(
            f'# verify_cold_preprovision_command failed in {tree}; no module ran '
            '(see the orchestrator log for its output tail)',
            flush=True,
        )
        return 1
    started = time.monotonic()
    runs = [
        await run_module(tree, mc, module_env(config, mc, tree), module_timeout(config, mc), log_dir)
        for mc in audit_modules(config)
    ]
    if not runs:
        print(f'# no module under {tree} declares a test_command', flush=True)
        return 1
    for line in render_report(runs, time.monotonic() - started):
        print(line, flush=True)
    return 0 if all(run.returncode == 0 for run in runs) else 1
