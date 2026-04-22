"""Sandbox backend dispatcher.

Chooses between ``bwrap`` and ``landlock`` at wrap-time, based on a
process-global preference set once at startup from
``SandboxConfig.backend``. Each of the three agent backends
(claude/codex/gemini) calls ``wrap_command()`` at its sandbox point.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

Backend = Literal['auto', 'bwrap', 'landlock', 'none']

_preferred: Backend = 'auto'


def set_backend(name: Backend) -> None:
    """Set the preferred sandbox backend. Called once at orchestrator startup."""
    global _preferred
    _preferred = name


def get_backend() -> Backend:
    return _preferred


def resolve_active_backend() -> Backend:
    """Return the effective backend for the current process.

    - ``none`` → ``none``
    - ``auto`` → first available of landlock, bwrap, or ``none``
    - ``landlock`` / ``bwrap`` → verified available; else ``none`` with a warning

    Callers can use this to decide whether to take a sandboxed code path at
    all (e.g. skip manual cmd assembly when the result would be unsandboxed).
    """
    if _preferred == 'none':
        return 'none'
    if _preferred == 'auto':
        return _resolve_auto()
    if _preferred == 'landlock':
        from orchestrator.agents.landlock import is_landlock_available
        if is_landlock_available():
            return 'landlock'
        logger.warning('sandbox backend=landlock but unavailable — running unsandboxed')
        return 'none'
    if _preferred == 'bwrap':
        from orchestrator.agents.sandbox import is_bwrap_available
        if is_bwrap_available():
            return 'bwrap'
        logger.warning('sandbox backend=bwrap but unavailable — running unsandboxed')
        return 'none'
    logger.warning('unknown sandbox backend=%s — running unsandboxed', _preferred)
    return 'none'


def wrap_command(
    inner_cmd: list[str],
    cwd: Path,
    writable_modules: list[str],
    writable_extras: list[str] | None = None,
) -> list[str]:
    """Wrap ``inner_cmd`` with the preferred sandbox backend.

    Returns the (possibly unwrapped) command to exec. If the preferred
    backend is unavailable or ``none``, logs a warning and returns
    ``inner_cmd`` unchanged.
    """
    backend = _preferred
    if backend == 'none':
        logger.info('sandbox backend=none — running unsandboxed')
        return inner_cmd

    if backend == 'auto':
        backend = _resolve_auto()

    if backend == 'landlock':
        from orchestrator.agents.landlock import (
            build_landlock_command,
            is_landlock_available,
        )
        if is_landlock_available():
            return build_landlock_command(inner_cmd, cwd, writable_modules, writable_extras)
        logger.warning('sandbox backend=landlock but unavailable — running unsandboxed')
        return inner_cmd

    if backend == 'bwrap':
        from orchestrator.agents.sandbox import build_bwrap_command, is_bwrap_available
        if is_bwrap_available():
            return build_bwrap_command(inner_cmd, cwd, writable_modules, writable_extras)
        logger.warning('sandbox backend=bwrap but unavailable — running unsandboxed')
        return inner_cmd

    logger.warning('unknown sandbox backend=%s — running unsandboxed', backend)
    return inner_cmd


def _resolve_auto() -> Backend:
    """Pick the best available backend: landlock > bwrap > none."""
    from orchestrator.agents.landlock import is_landlock_available
    from orchestrator.agents.sandbox import is_bwrap_available

    if is_landlock_available():
        return 'landlock'
    if is_bwrap_available():
        return 'bwrap'
    logger.warning('auto: no sandbox backend available — running unsandboxed')
    return 'none'
