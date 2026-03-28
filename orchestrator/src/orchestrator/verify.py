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


def _strip_directory_flag(cmd: str | None, module_prefix: str) -> str | None:
    """Remove ``--directory <module_prefix>`` from a ``uv run`` command.

    When scoping to worktree-relative file paths, the ``--directory`` flag
    would cause tools to resolve paths relative to the module subdirectory,
    leading to double-prefixed paths that don't exist.  The ``--project``
    flag is kept so ``uv`` still activates the correct venv.
    """
    if cmd is None:
        return None
    cmd = cmd.replace(f'--directory {module_prefix} ', '')
    cmd = cmd.replace(f'--directory={module_prefix} ', '')
    # Handle flag at end of string (no trailing space)
    cmd = cmd.replace(f'--directory {module_prefix}', '')
    cmd = cmd.replace(f'--directory={module_prefix}', '')
    return ' '.join(cmd.split())


def scope_module_config(mc: ModuleConfig, task_files: list[str]) -> ModuleConfig:
    """Narrow *mc*'s commands to the specific *task_files* it covers.

    Filters *task_files* to ``.py`` files matching ``mc.prefix + '/'`` and
    keeps full worktree-relative paths.  The ``--directory`` flag is stripped
    from scoped commands so that tools resolve paths from the worktree root,
    where the full paths are valid.

    Returns the original *mc* when no ``.py`` files from *task_files* fall
    under the prefix.
    """
    prefix = mc.prefix + '/'
    # Keep full worktree-relative paths, filter to .py files under this module
    scoped: list[str] = []
    for f in task_files:
        if f.startswith(prefix) and f.endswith('.py'):
            scoped.append(f)

    if not scoped:
        return mc

    test_files = [f for f in scoped if _is_test_file(f)]

    # Build scoped commands with worktree-relative paths, then strip
    # --directory so tools resolve paths from the worktree root
    lint_cmd = _strip_directory_flag(
        _scope_command(mc.lint_command, 'ruff check', scoped), mc.prefix)
    type_cmd = _strip_directory_flag(
        _scope_command(mc.type_check_command, 'pyright', scoped), mc.prefix)
    test_cmd = _strip_directory_flag(
        _scope_command(mc.test_command, 'pytest', test_files), mc.prefix) if test_files else None

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


async def run_full_verification(
    project_root: Path,
    config: OrchestratorConfig,
) -> VerifyResult:
    """Run verification for ALL subprojects against the project root.

    Unlike run_scoped_verification, this runs full (unscoped) test suites
    for every subproject that has an orchestrator.yaml. Used by review
    checkpoints to check integration health across the whole codebase.
    """
    from orchestrator.config import _discover_module_configs

    module_configs = _discover_module_configs(project_root)
    if not module_configs:
        logger.info('Full verification: no subproject configs — using global')
        return await run_verification(project_root, config)

    logger.info(
        'Full verification: running %d subprojects in parallel',
        len(module_configs),
    )
    results = await asyncio.gather(
        *(run_verification(project_root, config, mc) for mc in module_configs.values())
    )
    return _aggregate_results(list(results))


async def run_scoped_verification(
    worktree: Path,
    config: OrchestratorConfig,
    module_configs: list[ModuleConfig],
    task_files: list[str] | None = None,
) -> VerifyResult:
    """Run verification scoped to specific subprojects and optionally to task files.

    Scoping modes (in priority order):

    1. **File-scoped within subprojects** — when *module_configs* is non-empty
       and *task_files* is provided, each ModuleConfig's commands are narrowed
       to the specific files via :func:`scope_module_config`.
    2. **Fallback-scoped** — when *module_configs* is empty and *task_files* is
       provided, a synthetic ModuleConfig is built via
       :func:`_build_fallback_config`, bypassing the global commands entirely.
    3. **Global** — when *task_files* is ``None`` (or falsy) with no
       module_configs, or when fallback returns ``None`` (no .py files).
    """
    if module_configs:
        # Apply file-level scoping within each subproject when task_files given
        if task_files:
            # Filter to files that still exist — tasks may delete files as part of their work
            existing_files = [f for f in task_files if (worktree / f).exists()]
            scoped = [scope_module_config(mc, existing_files) for mc in module_configs]
            n_files = len(existing_files)
            n_mods = len(scoped)
            logger.info('Verification mode: file-scoped (%d files across %d subprojects)', n_files, n_mods)
        else:
            scoped = module_configs
            logger.info('Verification mode: subproject-scoped (%d subprojects)', len(module_configs))
        results = await asyncio.gather(
            *(run_verification(worktree, config, mc) for mc in scoped)
        )
        return _aggregate_results(list(results))

    # No module_configs — try fallback or global
    if task_files:
        # Filter to files that still exist — tasks may delete files as part of their work
        existing_files = [f for f in task_files if (worktree / f).exists()]
        fallback = _build_fallback_config(existing_files)
        if fallback is not None:
            logger.info('Verification mode: fallback-scoped (%d files)', len(existing_files))
            return await run_verification(worktree, config, fallback)

    logger.info('Verification mode: global (no scope info)')
    return await run_verification(worktree, config)
