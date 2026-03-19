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
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            executable='/bin/bash',
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode, stdout.decode()
    except TimeoutError:
        proc.kill()
        return 1, f'Command timed out after {timeout}s: {cmd}'
    except Exception as e:
        return 1, f'Command failed: {e}'


async def run_verification(
    worktree: Path,
    config: OrchestratorConfig,
    module_config: ModuleConfig | None = None,
) -> VerifyResult:
    """Run test suite, linter, and type checker. Return structured result."""
    test_cmd = (module_config and module_config.test_command) or config.test_command
    lint_cmd = (module_config and module_config.lint_command) or config.lint_command
    type_cmd = (module_config and module_config.type_check_command) or config.type_check_command

    # Run all three in parallel
    test_task = _run_cmd(test_cmd, worktree)
    lint_task = _run_cmd(lint_cmd, worktree)
    type_task = _run_cmd(type_cmd, worktree)

    (test_rc, test_out), (lint_rc, lint_out), (type_rc, type_out) = await asyncio.gather(
        test_task, lint_task, type_task
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
