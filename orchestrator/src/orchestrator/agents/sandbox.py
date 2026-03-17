"""Bubblewrap (bwrap) filesystem sandbox for agent invocations."""

from __future__ import annotations

import os
from pathlib import Path


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
