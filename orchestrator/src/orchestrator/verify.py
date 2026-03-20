"""Test/lint/typecheck runner for verification stages."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from orchestrator.config import ModuleConfig, OrchestratorConfig

logger = logging.getLogger(__name__)


@dataclass
class VerifyResult:
    passed: bool
    test_output: str
    lint_output: str
    type_output: str
    summary: str

    def failure_report(self) -> str:
        """Format all failures into a single report for the debugger."""
        sections = []
        if self.test_output and 'FAILED' in self.test_output:
            sections.append(f'## Test Failures\n\n```\n{self.test_output[-3000:]}\n```')
        if self.lint_output and self.lint_output.strip():
            sections.append(f'## Lint Issues\n\n```\n{self.lint_output[-2000:]}\n```')
        if self.type_output and 'error' in self.type_output.lower():
            sections.append(f'## Type Errors\n\n```\n{self.type_output[-2000:]}\n```')
        return '\n\n'.join(sections) if sections else self.summary


async def _run_cmd(cmd: str, cwd: Path, timeout: float = 300) -> tuple[int, str]:
    """Run a shell command, return (returncode, combined output)."""
    proc = None
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            executable='/bin/bash',
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode if proc.returncode is not None else 1, stdout.decode()
    except TimeoutError:
        if proc is not None:
            proc.kill()
        return 1, f'Command timed out after {timeout}s: {cmd}'
    except Exception as e:
        return 1, f'Command failed: {e}'


async def run_verification(
    worktree: Path,
    config: OrchestratorConfig,
    module_config: ModuleConfig | None = None,
) -> VerifyResult:
    """Run test suite, linter, and type checker. Return structured result.

    When *module_config* is provided, a ``None`` command means "skip that check"
    (the subproject doesn't define it).  When *module_config* is ``None``,
    global config commands are used for every check.
    """
    if module_config is not None:
        # Scoped: use module command; None → skip
        test_cmd = module_config.test_command
        lint_cmd = module_config.lint_command
        type_cmd = module_config.type_check_command
    else:
        # Global fallback
        test_cmd = config.test_command
        lint_cmd = config.lint_command
        type_cmd = config.type_check_command

    # Run non-None checks in parallel
    async def _run_or_skip(cmd: str | None) -> tuple[int, str]:
        if cmd is None:
            return 0, ''
        return await _run_cmd(cmd, worktree)

    (test_rc, test_out), (lint_rc, lint_out), (type_rc, type_out) = await asyncio.gather(
        _run_or_skip(test_cmd), _run_or_skip(lint_cmd), _run_or_skip(type_cmd)
    )

    passed = test_rc == 0 and lint_rc == 0 and type_rc == 0

    # Build summary
    parts = []
    if test_rc != 0:
        parts.append('tests failed')
    if lint_rc != 0:
        parts.append('lint issues')
    if type_rc != 0:
        parts.append('type errors')

    summary = 'All checks passed' if passed else f'Failures: {", ".join(parts)}'

    result = VerifyResult(
        passed=passed,
        test_output=test_out,
        lint_output=lint_out if lint_rc != 0 else '',
        type_output=type_out if type_rc != 0 else '',
        summary=summary,
    )

    logger.info(f'Verification {"passed" if passed else "failed"}: {summary}')
    return result


def _aggregate_results(results: list[VerifyResult]) -> VerifyResult:
    """Merge per-subproject VerifyResults into one."""
    if len(results) == 1:
        return results[0]

    passed = all(r.passed for r in results)
    test_output = '\n'.join(r.test_output for r in results if r.test_output)
    lint_output = '\n'.join(r.lint_output for r in results if r.lint_output)
    type_output = '\n'.join(r.type_output for r in results if r.type_output)

    parts = []
    if any('tests failed' in r.summary for r in results):
        parts.append('tests failed')
    if any('lint issues' in r.summary for r in results):
        parts.append('lint issues')
    if any('type errors' in r.summary for r in results):
        parts.append('type errors')

    summary = 'All checks passed' if passed else f'Failures: {", ".join(parts)}'

    return VerifyResult(
        passed=passed,
        test_output=test_output,
        lint_output=lint_output,
        type_output=type_output,
        summary=summary,
    )


async def run_scoped_verification(
    worktree: Path,
    config: OrchestratorConfig,
    module_configs: list[ModuleConfig],
) -> VerifyResult:
    """Run verification scoped to specific subprojects.

    If *module_configs* is empty, falls back to global ``run_verification``.
    Otherwise runs ``run_verification`` per ModuleConfig and aggregates.
    """
    if not module_configs:
        return await run_verification(worktree, config)

    results = await asyncio.gather(
        *(run_verification(worktree, config, mc) for mc in module_configs)
    )
    return _aggregate_results(list(results))
