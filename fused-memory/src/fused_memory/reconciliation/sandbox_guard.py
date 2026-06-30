"""Sandbox guard for reconciliation stage/remediation agents (task 1935).

Provides ``resolve_recon_sandbox_wrap`` — a factory that returns a callable
suitable for the ``sandbox_wrap`` hook in ``shared.cli_invoke.invoke_claude_agent``.
The returned callable wraps the claude argv with a kernel-level Landlock (or
bwrap) confinement ruleset that DENIES writes to every tracked source file in
the repository while preserving the agent's legitimate scratch (/tmp, ~/.claude,
/dev) and HTTP-over-MCP memory operations (network is NOT restricted by Landlock
here — only filesystem access is governed).

This defeats the sub-agent / heredoc bypass that tool-deny lists cannot stop:
Landlock's NO_NEW_PRIVS + restrict_self are inherited by the entire claude
process tree, so a child process that bypasses the tool-layer (e.g. a
``python3 heredoc`` spawned from within the agent) still cannot write to
production source files.

Resolution order mirrors ``orchestrator.agents.sandbox_dispatch``
(Landlock > bwrap) but is **fail-CLOSED** at the terminal case: when neither
backend is available and the caller has opted in to sandboxing, this module
raises ``RemediationSandboxUnavailable`` so that the caller can refuse to launch
an unconfined agent.  An identity wrap (return inner_cmd unchanged) is never
returned — that would silently reopen the hole this module was created to close.

Imports of the orchestrator sandbox backends are LAZY and guarded:
- Runtime resolution works via the shared uv-workspace venv (orchestrator is
  installed in the same venv as fused-memory).
- A hard ``pyproject.toml`` dependency on orchestrator is intentionally absent
  to avoid the memory→orchestrator layering inversion.
- An ``ImportError`` on either backend also raises ``RemediationSandboxUnavailable``
  (fail-closed: if the substrate is unreachable, refuse to run unconfined).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


class RemediationSandboxUnavailable(RuntimeError):
    """Raised when confinement is required but no sandbox backend is available.

    The caller (``cli_stage_runner.run_stage_via_cli``) must NOT launch the
    reconciliation agent unconfined when this exception is raised.  Instead it
    should return a ``StageResult(error=...)`` and log loudly.

    To opt out of confinement on a sandbox-less host, set
    ``reconciliation.sandbox_recon_agents = false`` in config.yaml.
    """


def resolve_recon_sandbox_wrap(
    cwd: Path,
    writable_extras: list[str] | None = None,
) -> Callable[[list[str]], list[str]]:
    """Return a sandbox-wrap callable for a reconciliation stage agent.

    The callable transforms ``inner_cmd`` (the full claude argv) into a
    sandboxed command that confines the entire claude process tree.

    Resolution order: Landlock > bwrap.  Fail-CLOSED when neither backend is
    available (raises ``RemediationSandboxUnavailable`` rather than returning
    an identity/passthrough wrap).

    Args:
        cwd: The working directory that will be passed to ``_run_subprocess``
             (typically ``config.explore_codebase_root``).  Used as the
             ``worktree`` argument to ``build_landlock_command`` /
             ``build_bwrap_command`` — only the gitignored ``.task/``
             subdirectory of this root will be made writable; no module
             source directories are added to the writable set
             (``writable_modules=[]``).
        writable_extras: Optional additional paths to include in the writable
             set (e.g. a uvx/pip cache dir used by a stdio MCP server).
             These are passed through to the underlying backend's
             ``writable_extras`` parameter.

    Returns:
        A ``Callable[[list[str]], list[str]]`` that wraps the inner argv.

    Raises:
        RemediationSandboxUnavailable: When neither Landlock nor bwrap is
            available, OR when the orchestrator backends cannot be imported.
    """
    # ── Landlock branch ───────────────────────────────────────────────────────
    try:
        from orchestrator.agents.landlock import (  # type: ignore[import]
            build_landlock_command,
            is_landlock_available,
        )
    except ImportError as exc:
        raise RemediationSandboxUnavailable(
            f'Cannot import orchestrator.agents.landlock — refusing to run '
            f'reconciliation agent unconfined. '
            f'Install orchestrator in the same venv or set '
            f'reconciliation.sandbox_recon_agents=false to opt out. '
            f'(ImportError: {exc})'
        ) from exc

    if is_landlock_available():
        # writable_modules=[] → no repo source dir is writable; only the
        # built-in scratch (/tmp, ~/.claude, /dev) and the gitignored .task/
        # subdirectory of cwd (added unconditionally by build_landlock_command).
        def _landlock_wrap(cmd: list[str]) -> list[str]:
            return build_landlock_command(cmd, cwd, [], writable_extras=writable_extras)
        return _landlock_wrap

    # ── bwrap fallback ────────────────────────────────────────────────────────
    try:
        from orchestrator.agents.sandbox import (  # type: ignore[import]
            build_bwrap_command,
            is_bwrap_available,
        )
    except ImportError as exc:
        raise RemediationSandboxUnavailable(
            f'Landlock unavailable and cannot import orchestrator.agents.sandbox '
            f'for bwrap fallback — refusing to run reconciliation agent unconfined. '
            f'(ImportError: {exc})'
        ) from exc

    if is_bwrap_available():
        def _bwrap_wrap(cmd: list[str]) -> list[str]:
            return build_bwrap_command(cmd, cwd, [], writable_extras=writable_extras)
        return _bwrap_wrap

    # ── Fail-closed: neither backend available ────────────────────────────────
    raise RemediationSandboxUnavailable(
        'Reconciliation sandboxing is enabled (sandbox_recon_agents=true) but '
        'neither Landlock nor bwrap is available on this host.  Refusing to '
        'launch reconciliation agent unconfined — an unconfined agent can '
        'write to production source files without going through review→verify→merge.  '
        'Options: (1) upgrade to a kernel ≥5.13 with Landlock support, '
        '(2) install bubblewrap (bwrap), or '
        '(3) set reconciliation.sandbox_recon_agents=false to opt out explicitly.'
    )
