"""Which environment variables select a Python environment, and an environ with none of them.

A child process that works on another tree (a verify target, a dispatched
agent) must resolve that tree's own environment. Inheriting the orchestrator's
selection makes it resolve the orchestrator's venv instead: a target's
``uv sync`` once synced its deps into ``dark-factory/.venv`` and broke the
running orchestrator (the 2026-05-29 ghost-venv incident), and an agent's
``uv run --active`` from a 3.14-pinned project rebuilt main's venv
(2026-10-22, task 7399).

This is a denylist, not an allowlist: reify's cargo verify needs a broad,
evolving set of toolchain, sccache and jobserver vars to pass through.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

__all__ = ['without_python_env_selection']

_PYTHON_ENV_SELECTION_KEYS: frozenset[str] = frozenset({
    'VIRTUAL_ENV',
    'UV_PROJECT_ENVIRONMENT',
    'UV_PROJECT',
    'UV_ACTIVE',
    # The orchestrator unit may run --frozen; a child must stay free to sync.
    'UV_FROZEN',
    'UV_NO_SYNC',
    # A child's uv invocation tree is independent and starts the count fresh.
    'UV_RUN_RECURSION_DEPTH',
    'CONDA_PREFIX',
    'CONDA_DEFAULT_ENV',
    'PYTHONHOME',
})


def _without_venv_bin(path: str | None, venv: str | None) -> str | None:
    if not path or not venv:
        return path
    venv_bin = os.path.normpath(os.path.join(venv, 'bin'))
    kept = [p for p in path.split(os.pathsep) if p and os.path.normpath(p) != venv_bin]
    return os.pathsep.join(kept)


def without_python_env_selection(environ: Mapping[str, str]) -> dict[str, str]:
    env = {k: v for k, v in environ.items() if k not in _PYTHON_ENV_SELECTION_KEYS}
    stripped_path = _without_venv_bin(env.get('PATH'), environ.get('VIRTUAL_ENV'))
    if stripped_path is not None:
        env['PATH'] = stripped_path
    return env
