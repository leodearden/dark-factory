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
    - --bind $HOME/.claude — Claude Code config/OAuth (writable for session state)
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

    # Claude Code config/OAuth
    claude_dir = os.path.join(os.path.expanduser('~'), '.claude')
    if os.path.isdir(claude_dir):
        cmd.extend(['--bind', claude_dir, claude_dir])

    # Extra writable directories
    if writable_extras:
        for extra in writable_extras:
            if os.path.isdir(extra):
                cmd.extend(['--bind', extra, extra])

    cmd.append('--')
    cmd.extend(inner_cmd)
    return cmd
