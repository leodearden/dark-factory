"""Run the merge role's per-module test commands on one tree, uncached, and report each failure under its module."""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.offline_lane_records import EndedBy, classify_exit
from orchestrator.verify import (
    FailureKind,
    PytestFailure,
    _provision_main_tip_worktree,
    _resolve_verify_env,
    _run_cmd,
    failure_sections,
    parse_pytest_failures,
)
from orchestrator.verify_cache.modes import VERIFY_CACHE_ENV, CacheMode

AUDIT_COMMAND = 'verify-cache-audit'
DEFAULT_LOG_DIR = '.df-verify-cache-audit'

# pytest.ExitCode.OK and pytest.ExitCode.TESTS_FAILED; named here so pytest is not imported.
PYTEST_OK = 0
PYTEST_TESTS_FAILED = 1

#: How much of one failure's pytest section the report quotes; the rest stays in the module's log.
SECTION_MAX_LINES = 60

_NODE_SEPARATOR = '::'
_QUOTE = '#   | '

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
    sections: Mapping[PytestFailure, str] = field(default_factory=dict)

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


def drop_runner_only_env(environ: MutableMapping[str, str]) -> None:
    """Remove from the runner's own *environ* what only its config load needed, before any child inherits it."""
    for name in RUNNER_ONLY_ENV:
        environ.pop(name, None)


def gate_path(path: str) -> str:
    """*path* without any Python environment's ``bin``, as the merge gate's children get it.

    The gate strips the orchestrator's own venv; the runner is itself started under the lane
    tree's venv inside the orchestrator's environment, so it strips every one it inherited.
    """
    return os.pathsep.join(entry for entry in path.split(os.pathsep) if entry and not _is_venv_bin(entry))


def _is_venv_bin(entry: str) -> bool:
    return (Path(entry).parent / 'pyvenv.cfg').is_file()


def module_env_overlay(config: OrchestratorConfig, mc: ModuleConfig) -> dict[str, str]:
    """What *mc*'s child gets on top of verify.py::_target_subprocess_env: the gate's PATH, its verify env, and the cache off."""
    path = {'PATH': gate_path(os.environ['PATH'])} if 'PATH' in os.environ else {}
    return {**path, **_resolve_verify_env(config, mc, role='merge'), VERIFY_CACHE_ENV: CacheMode.OFF.value}


def module_timeout(config: OrchestratorConfig, mc: ModuleConfig) -> float:
    if mc.verify_command_timeout_secs is not None:
        return mc.verify_command_timeout_secs
    return config.verify_command_timeout_secs


async def run_module(tree: Path, mc: ModuleConfig, env: dict[str, str], timeout: float, log_dir: Path) -> ModuleRun:
    """Run *mc*'s test command in *tree* as the merge gate runs a leg (verify.py::_run_cmd), its output in its own log."""
    if not mc.test_command:
        raise ValueError(f'module {mc.prefix!r} declares no test_command')
    log_path = log_dir / f'{mc.prefix.replace("/", "--")}.log'
    returncode, _, timed_out = await _run_cmd(mc.test_command, tree, timeout, env=env, log_path=log_path)
    failures, sections = await asyncio.to_thread(_parse_log, log_path)
    return ModuleRun(mc.prefix, returncode, _module_end(returncode, timed_out), failures, log_path, sections)


def _module_end(returncode: int, timed_out: bool) -> ModuleEnd:
    if timed_out:
        return ModuleEnd.TIMEOUT
    return ModuleEnd.SIGNAL if classify_exit(returncode) is EndedBy.SIGNAL else ModuleEnd.EXIT


def _parse_log(log_path: Path) -> tuple[tuple[PytestFailure, ...], dict[PytestFailure, str]]:
    output = log_path.read_bytes().decode(errors='replace') if log_path.exists() else ''
    failures = parse_pytest_failures(output)
    return failures, failure_sections(output, failures)


def render_report(runs: Sequence[ModuleRun], elapsed: float) -> list[str]:
    """The report's lines: each failure module-qualified, their quoted sections, each module's end, then a pytest-shaped summary.

    The summary line is the lane's proof of a complete session, so it is printed only when every module completed.
    """
    lines = [_failure_line(run.module, failure) for run in runs for failure in run.failures]
    lines += [line for run in runs for failure in run.failures for line in _section_lines(run, failure)]
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


def _section_lines(run: ModuleRun, failure: PytestFailure) -> list[str]:
    """*failure*'s pytest section, quoted so it outlives the tree in the lane's own record of this run."""
    body = run.sections.get(failure)
    if body is None:
        return []
    body_lines = body.strip('\n').splitlines()
    shown = body_lines[:SECTION_MAX_LINES]
    hidden = len(body_lines) - len(shown)
    return [
        f'# {failure.kind} {AuditNode(run.module, failure.node_id)}:',
        *(f'{_QUOTE}{_unlinked(line)}' for line in shown),
        *([f'{_QUOTE}... {hidden} more lines in {run.log_path}'] if hidden else []),
    ]


def _unlinked(text: str) -> str:
    """*text* with no node separator, so no parser surface, anchored or not, can lift an unqualified id out of it."""
    return text.replace(_NODE_SEPARATOR, ':')


def _inert(text: str) -> str:
    """*text* unlinked and on one line."""
    return ' '.join(_unlinked(text).split())


def _run_log_dir(log_root: Path) -> Path:
    """A fresh directory under *log_root* for one run, so the lane's detect and confirm never share a log."""
    log_root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f'{datetime.now(UTC):%Y%m%dT%H%M%SZ}-', dir=log_root))


async def run_audit(tree: Path, config: OrchestratorConfig, log_root: Path) -> int:
    """Run every audit module on *tree* in turn, print the report, and return 0 iff every module exited 0.

    The tree is first provisioned as the merge gate provisions its own
    (verify.py::_provision_main_tip_worktree); a failed provision is a non-zero exit naming
    no test, so the lane reads it as Unconfirmable. A tree with no module to audit is a
    failure, never a vacuous green. Each run's module logs go to a fresh directory under *log_root*.
    """
    log_dir = _run_log_dir(log_root)
    if not await _provision_main_tip_worktree(tree, config, role='merge'):
        print(
            f'# verify_cold_preprovision_command failed in {tree}; no module ran '
            '(see the orchestrator log for its output tail)',
            flush=True,
        )
        return 1
    started = time.monotonic()
    runs = [
        await run_module(tree, mc, module_env_overlay(config, mc), module_timeout(config, mc), log_dir)
        for mc in audit_modules(config)
    ]
    if not runs:
        print(f'# no module under {tree} declares a test_command', flush=True)
        return 1
    for line in render_report(runs, time.monotonic() - started):
        print(line, flush=True)
    return 0 if all(run.returncode == 0 for run in runs) else 1
