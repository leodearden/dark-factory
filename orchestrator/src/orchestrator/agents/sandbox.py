"""Bubblewrap (bwrap) filesystem sandbox for agent invocations."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_bwrap_available: bool | None = None


def is_bwrap_available() -> bool:
    """Probe whether bwrap can create sandboxes on this system.

    Result is cached for the lifetime of the process.
    """
    global _bwrap_available
    if _bwrap_available is not None:
        return _bwrap_available

    if not shutil.which('bwrap'):
        logger.warning('bwrap not found in PATH — sandboxing disabled')
        _bwrap_available = False
        return False

    try:
        result = subprocess.run(
            ['bwrap', '--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc',
             '--', '/bin/true'],
            capture_output=True, timeout=5,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(errors='replace').strip()
            logger.warning('bwrap probe failed (rc=%d): %s — sandboxing disabled',
                           result.returncode, stderr)
            _bwrap_available = False
        else:
            _bwrap_available = True
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning('bwrap probe error: %s — sandboxing disabled', exc)
        _bwrap_available = False

    return _bwrap_available


def _reset_probe() -> None:
    """Reset the cached probe result (for tests)."""
    global _bwrap_available
    _bwrap_available = None


def build_bwrap_command(
    inner_cmd: list[str],
    worktree: Path,
    writable_modules: list[str],
    writable_extras: list[str] | None = None,
) -> list[str]:
    """Construct a bwrap command that sandboxes an agent to specific modules.

    Strategy:
    - --ro-bind / / — read-only root (entire filesystem visible, nothing writable)
    - --bind <worktree>/<module> — writable overlay per locked module
    - --bind <worktree>/.task — always writable (agent artifacts)
    - --tmpfs /tmp — writable tmp
    - --dev /dev — device nodes
    - --proc /proc — proc filesystem
    - / is read-only; ~/.claude is read-only too except for whatever
      subpaths are supplied here via writable_extras (e.g.
      ~/.claude/fleet/, as computed by
      orchestrator.agents.write_set.compute_write_set), so
      ~/.claude/settings.json stays read-only (PRD
      deny-write-to-settings.json property).

    CALLER OBLIGATION — CLAUDE_CONFIG_DIR must be granted, not assumed:
    this function binds exactly the module dirs, <worktree>/.task, /tmp
    (tmpfs), /dev, /proc, and each writable_extra. Whoever redirects the
    CLI's OAuth/session state to a per-task CLAUDE_CONFIG_DIR MUST ALSO
    pass that directory in writable_extras. The orchestrator satisfies
    this via the .task bind; reconciliation — which lands on THIS branch
    whenever Landlock is unavailable — satisfies it via a per-run computed
    extra in cli_stage_runner.run_stage_via_cli, machine-checked by
    sandbox_guard.resolve_recon_sandbox_wrap. That check exists because the
    obligation went silently unmet from 2026-07-18 (task 2744) to
    2026-08-11 (task 4003): recon's config dir sat under
    <data_dir>/recon-config/, outside every bind above, so every recon
    stage was told where to write its transcript and then denied the write.
    Note this branch drops a writable_extra that is not an existing dir
    (see below), so a mistimed grant is silently vacuous here too.
    """
    cmd = [
        'bwrap',
        '--die-with-parent',
        '--dev', '/dev',
        '--proc', '/proc',
        '--tmpfs', '/tmp',
        '--ro-bind', '/', '/',
    ]

    # Writable module directories
    worktree_str = str(worktree.resolve())
    for module in writable_modules:
        module_path = os.path.join(worktree_str, module)
        # Ensure dir exists so bwrap can bind it
        os.makedirs(module_path, exist_ok=True)
        cmd.extend(['--bind', module_path, module_path])

    # .task is always writable (agent artifacts)
    task_dir = os.path.join(worktree_str, '.task')
    os.makedirs(task_dir, exist_ok=True)
    cmd.extend(['--bind', task_dir, task_dir])

    # Extra writable directories
    if writable_extras:
        for extra in writable_extras:
            if os.path.isdir(extra):
                cmd.extend(['--bind', extra, extra])
            else:
                # bwrap cannot --bind a nonexistent path, and compute_write_set
                # is intentionally pure (no makedirs) — existence/creation is
                # the backend's job, but this backend does not create extras
                # (unlike the writable_modules/.task loops above, which do).
                # Silently dropping the bind would let a requested writable
                # path (e.g. ~/.claude/fleet on first run) go missing with no
                # signal, contrary to the loud-over-silent-degradation norm —
                # so surface it instead of degrading silently.
                logger.warning(
                    'build_bwrap_command: writable extra %r is not a directory '
                    '— skipping bind (path missing or not created yet)', extra,
                )

    cmd.append('--')
    cmd.extend(inner_cmd)
    return cmd
