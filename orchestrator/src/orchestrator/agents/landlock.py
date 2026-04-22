"""Landlock LSM filesystem sandbox for agent invocations.

Alternative to bwrap that does not use user namespaces, so it sidesteps the
Bun v1.3.13 + kernel 6.17 self-init segfault observed in
``claude`` CLI launches inside ``bwrap`` on this host.

Shape mirrors ``orchestrator.agents.sandbox`` (bwrap): a cached availability
probe plus a command-wrapper function that returns a ``list[str]`` suitable
for ``asyncio.create_subprocess_exec``.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Landlock syscall numbers (x86_64). Landlock landed in kernel 5.13; this
# module assumes x86_64 — other arches would need additional arch gates.
SYS_landlock_create_ruleset = 444

_LANDLOCK_EXEC = Path(__file__).with_name('landlock_exec.py')

_landlock_available: bool | None = None


def _syscall_probe_abi() -> int:
    """Return the kernel's landlock ABI version, or -1 if unsupported.

    Uses ``landlock_create_ruleset(NULL, 0, LANDLOCK_CREATE_RULESET_VERSION=1)``
    which is defined to return the ABI version without creating a ruleset.
    """
    try:
        libc = ctypes.CDLL('libc.so.6', use_errno=True)
        return int(libc.syscall(SYS_landlock_create_ruleset, None, 0, 1))
    except OSError:
        return -1


def is_landlock_available() -> bool:
    """Probe whether landlock can restrict this process on this kernel.

    Result is cached for the lifetime of the process.
    """
    global _landlock_available
    if _landlock_available is not None:
        return _landlock_available

    abi = _syscall_probe_abi()
    if abi < 1:
        logger.warning('landlock not supported (abi=%d) — landlock sandbox disabled', abi)
        _landlock_available = False
    else:
        logger.debug('landlock ABI %d available', abi)
        _landlock_available = True
    return _landlock_available


def _reset_probe() -> None:
    """Reset the cached probe result (for tests)."""
    global _landlock_available
    _landlock_available = None


def build_landlock_command(
    inner_cmd: list[str],
    worktree: Path,
    writable_modules: list[str],
    writable_extras: list[str] | None = None,
) -> list[str]:
    """Construct a landlock-wrapped command.

    Strategy:
    - Invoke the ``landlock_exec`` helper (``python -m landlock_exec`` style)
    - Pass per-module and ``.task`` writable paths via ``--writable``
    - ``/`` is read-only, ``~/.claude`` is writable (handled inside the helper)

    Returns argv suitable for ``asyncio.create_subprocess_exec``.
    """
    worktree_str = str(worktree.resolve())
    writable_paths: list[str] = []

    for module in writable_modules:
        module_path = os.path.join(worktree_str, module)
        os.makedirs(module_path, exist_ok=True)
        writable_paths.append(module_path)

    task_dir = os.path.join(worktree_str, '.task')
    os.makedirs(task_dir, exist_ok=True)
    writable_paths.append(task_dir)

    if writable_extras:
        for extra in writable_extras:
            writable_paths.append(extra)

    cmd: list[str] = [sys.executable, str(_LANDLOCK_EXEC)]
    for path in writable_paths:
        cmd.extend(['--writable', path])
    cmd.append('--')
    cmd.extend(inner_cmd)
    return cmd
