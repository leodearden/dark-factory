"""Pre-done validator hook for set_task_status(done) transitions.

Environment-variable-driven configuration: set
    FUSED_MEMORY_PREDONE_HOOK_<PROJECT_ID_UPPER>=<command> {id} [args…]
to enable the hook for a given project.  An empty/whitespace-only value
is treated as unset so operators can disable without ``unset``.

Only ``{id}`` is supported as a template placeholder — it is replaced with
the incoming task_id in each argv token after ``shlex.split``.  No general
template language is supported.

Subprocess invocation uses ``asyncio.create_subprocess_exec`` (argv form,
NOT shell=True) to prevent shell injection from agent-supplied task ids.
cwd is set to ``project_root`` so the hook can resolve project-relative paths.

Timeout defaults to 30 s (≤30 s suggested by the spec).  On timeout the
process is killed and the transition is refused (fail-closed).

Fail-closed on all configuration errors — ``pre_done_hook_misconfigured``
is returned when the command cannot be parsed (shlex failure), the parsed
argv is empty, or the subprocess cannot be launched at all (binary missing,
not executable, or any other OSError).  This prevents a misconfiguration
from silently disabling the gate.

Error shapes mirror ``_terminal_exit_error`` / ``_done_gate_error`` in
``task_interceptor.py`` — same ``{'success': False, 'error': …, 'task_id':
…, 'hint': …}`` skeleton so MCP callers can handle them uniformly.
"""

from __future__ import annotations

import asyncio
import os
import shlex

from fused_memory.models.scope import resolve_project_id

__all__ = [
    'resolve_hook_command',
    'run_hook',
]

_STDERR_CLIP = 2000  # max bytes/chars captured in the error response


# ── env-var resolution ────────────────────────────────────────────────────────


def resolve_hook_command(project_root: str) -> str | None:
    """Return the configured pre-done hook command for *project_root*, or None.

    Reads ``FUSED_MEMORY_PREDONE_HOOK_<PROJECT_ID_UPPER>`` from the
    environment.  An empty or whitespace-only value is treated as unset so
    operators can clear the hook with an empty assignment rather than
    ``unset``.

    Examples::

        FUSED_MEMORY_PREDONE_HOOK_REIFY='reify-audit --task {id} --pre-done'
        → returns 'reify-audit --task {id} --pre-done'

        FUSED_MEMORY_PREDONE_HOOK_DARK_FACTORY=''
        → returns None  (empty treated as unset)
    """
    project_id = resolve_project_id(project_root).upper()
    env_var = f'FUSED_MEMORY_PREDONE_HOOK_{project_id}'
    raw = os.environ.get(env_var, '')
    stripped = raw.strip()
    return stripped if stripped else None


# ── structured error helpers ──────────────────────────────────────────────────


def _pre_done_hook_rejected_error(
    task_id: str,
    exit_code: int,
    stderr: str,
    command: str,
) -> dict:
    """Structured error when the hook subprocess exits non-zero."""
    return {
        'success': False,
        'error': 'pre_done_hook_rejected',
        'task_id': task_id,
        'exit_code': exit_code,
        'stderr': stderr[:_STDERR_CLIP],
        'command': command,
        'hint': (
            f'Pre-done validator hook exited with code {exit_code}. '
            'Fix the underlying issue reported in stderr, then retry. '
            'To disable the hook temporarily, unset or clear the '
            f'FUSED_MEMORY_PREDONE_HOOK_* env var for this project.'
        ),
    }


def _pre_done_hook_timeout_error(
    task_id: str,
    timeout_seconds: float,
    command: str,
) -> dict:
    """Structured error when the hook subprocess times out."""
    return {
        'success': False,
        'error': 'pre_done_hook_timeout',
        'task_id': task_id,
        'timeout_seconds': timeout_seconds,
        'command': command,
        'hint': (
            f'Pre-done validator hook timed out after {timeout_seconds}s. '
            'The hook process was killed. Fix the hook or increase the timeout.'
        ),
    }


def _pre_done_hook_misconfigured_error(
    task_id: str,
    reason: str,
    command: str,
) -> dict:
    """Structured error when the hook command cannot be parsed or is empty."""
    return {
        'success': False,
        'error': 'pre_done_hook_misconfigured',
        'task_id': task_id,
        'reason': reason,
        'command': command,
        'hint': (
            'The FUSED_MEMORY_PREDONE_HOOK_* env var contains an invalid '
            f'shell command: {reason}. Fix the configuration and retry.'
        ),
    }


# ── subprocess invocation ─────────────────────────────────────────────────────


async def run_hook(
    task_id: str,
    project_root: str,
    *,
    timeout: float = 30.0,
) -> dict | None:
    """Invoke the configured pre-done hook for *project_root*.

    Returns ``None`` on success (hook passed or unconfigured).
    Returns a structured error dict on rejection, timeout, or misconfiguration.

    The hook is unconfigured when
    ``FUSED_MEMORY_PREDONE_HOOK_<PROJECT_ID_UPPER>`` is unset or empty —
    in that case this function is a no-op with no subprocess overhead.

    Template substitution: ``{id}`` in each argv token is replaced with
    *task_id*.  Only ``{id}`` is supported (no general template language).

    Subprocess is invoked with ``cwd=project_root`` so hooks can resolve
    project-relative paths.
    """
    command = resolve_hook_command(project_root)
    if command is None:
        return None

    # Parse command into argv tokens
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return _pre_done_hook_misconfigured_error(
            task_id,
            reason=f'shlex.split failed: {exc}',
            command=command,
        )

    if not argv:
        return _pre_done_hook_misconfigured_error(
            task_id,
            reason='command is empty after splitting',
            command=command,
        )

    # Substitute {id} in each token
    argv = [tok.replace('{id}', task_id) for tok in argv]

    # Launch subprocess (argv form — no shell to prevent injection).
    # Catch all OS-level launch failures (missing binary, not executable,
    # ENOEXEC, ETXTBSY, etc.) and return a structured misconfigured error
    # rather than propagating the OSError up through the gate chain.
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=project_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (FileNotFoundError, PermissionError, OSError) as exc:
        return _pre_done_hook_misconfigured_error(
            task_id,
            reason=f'failed to launch hook: {exc}',
            command=command,
        )

    try:
        _stdout, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return _pre_done_hook_timeout_error(task_id, timeout, command)

    returncode = proc.returncode
    assert returncode is not None, 'returncode must be set after communicate()'
    if returncode == 0:
        return None

    stderr_text = stderr_bytes.decode('utf-8', errors='replace')
    return _pre_done_hook_rejected_error(
        task_id,
        exit_code=returncode,
        stderr=stderr_text,
        command=command,
    )
