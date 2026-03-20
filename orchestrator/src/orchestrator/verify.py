"""Test/lint/typecheck runner for verification stages."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from orchestrator.config import ModuleConfig, OrchestratorConfig

logger = logging.getLogger(__name__)


def _scope_command(cmd: str | None, tool_keyword: str, files: list[str]) -> str | None:
    """Narrow *cmd* to operate on *files* instead of whole directories.

    Finds *tool_keyword* in *cmd*, keeps everything up to and including it as
    the prefix, extracts any dash-prefixed flags from the remainder, and
    rebuilds the command as ``'{prefix} {files} {flags}'``.

    Returns:
        ``None`` when *cmd* is ``None`` or *files* is empty.
        The original *cmd* unchanged when *tool_keyword* is not found.
        The scoped command otherwise.
    """
    if cmd is None:
        return None
    if not files:
        return None

    idx = cmd.find(tool_keyword)
    if idx == -1:
        return cmd

    prefix = cmd[: idx + len(tool_keyword)]
    remainder = cmd[idx + len(tool_keyword):]

    # Preserve any dash-prefixed flags from the original remainder
    flags = [tok for tok in remainder.split() if tok.startswith('-')]

    parts = [prefix] + files
    if flags:
        parts += flags
    return ' '.join(parts)


def _is_test_file(path: str) -> bool:
    """Return True when *path* looks like a test file."""
    name = path.rsplit('/', 1)[-1]
    return (
        name.startswith('test_')
        or name.endswith('_test.py')
        or name == 'conftest.py'
        or '/tests/' in path
        or path.startswith('tests/')
    )


def scope_module_config(mc: ModuleConfig, task_files: list[str]) -> ModuleConfig:
    """Narrow *mc*'s commands to the specific *task_files* it covers.

    Filters *task_files* to those matching ``mc.prefix + '/'``, strips the
    prefix, then calls :func:`_scope_command` to replace broad path args with
    the specific files.

    Returns the original *mc* when no ``.py`` files from *task_files* fall
    under the prefix.
    """
    prefix = mc.prefix + '/'
    # Strip prefix and filter to .py files
    scoped: list[str] = []
    for f in task_files:
        if f.startswith(prefix):
            stripped = f[len(prefix):]
            if stripped.endswith('.py'):
                scoped.append(stripped)

    if not scoped:
        return mc

    test_files = [f for f in scoped if _is_test_file(f)]

    # Build scoped commands; None when no applicable files exist
    lint_cmd = _scope_command(mc.lint_command, 'ruff check', scoped)
    type_cmd = _scope_command(mc.type_check_command, 'pyright', scoped)
    test_cmd = _scope_command(mc.test_command, 'pytest', test_files) if test_files else None

    return ModuleConfig(
        prefix=mc.prefix,
        lint_command=lint_cmd,
        type_check_command=type_cmd,
        test_command=test_cmd,
        lock_depth=mc.lock_depth,
        max_per_module=mc.max_per_module,
        module_overrides=mc.module_overrides,
    )


def _build_fallback_config(task_files: list[str]) -> ModuleConfig | None:
    """Build a synthetic ModuleConfig from *task_files* when no module configs match.

    Filters to ``.py`` files, classifies into source vs test, and builds bare
    ``ruff check``/``pyright``/``pytest`` commands (no ``uv run`` wrapper —
    callers run these in the worktree root where venvs aren't needed for the
    fallback path).

    Returns ``None`` when no ``.py`` files are found.
    """
    py_files = [f for f in task_files if f.endswith('.py')]
    if not py_files:
        return None

    test_files = [f for f in py_files if _is_test_file(f)]

    lint_cmd = 'ruff check ' + ' '.join(py_files)
    type_cmd = 'pyright ' + ' '.join(py_files)
    test_cmd = ('pytest ' + ' '.join(test_files)) if test_files else None

    return ModuleConfig(
        prefix='__fallback__',
        lint_command=lint_cmd,
        type_check_command=type_cmd,
        test_command=test_cmd,
    )


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
