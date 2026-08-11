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

OPERATIONAL PRECONDITION — venv install contract
-------------------------------------------------
This module's default-on confinement (``sandbox_recon_agents=True``) depends on
``orchestrator`` being importable at runtime.  The canonical deployment is the
shared uv-workspace venv in which both ``fused-memory`` and ``orchestrator`` are
installed.  **If fused-memory is ever installed in a venv that does NOT include
orchestrator** (e.g. a stripped CI image or a standalone deployment), the
``ImportError`` guard raises ``RemediationSandboxUnavailable``, which
``run_stage_via_cli`` treats as fail-CLOSED: *every* reconciliation stage returns
an error ``StageResult`` and reconciliation halts entirely.  This is the correct
safety posture but will be operationally surprising if the venv requirement is
not met.

Operators deploying fused-memory outside the workspace venv MUST either:
  1. Install orchestrator in the same venv (``uv add orchestrator`` / workspace
     member), OR
  2. Set ``reconciliation.sandbox_recon_agents = false`` in config.yaml to
     explicitly opt out of confinement on that host.

CONFIG-DIR CONTAINMENT — the INV-1 machine check (task 4003)
------------------------------------------------------------
The recon per-run ``CLAUDE_CONFIG_DIR`` is part of the writable set.  The
*policy* lives in the caller (``cli_stage_runner.run_stage_via_cli`` appends
``config_dir.path`` to the writable extras); the *verification* lives here:
``resolve_recon_sandbox_wrap`` refuses to return a wrap — fail-CLOSED, as
everywhere else in this module — unless the config dir it is handed is actually
contained in the writable set that will be built.

That split is what makes the assertion load-bearing.  If a future edit drops the
computed grant, recon refuses to launch instead of silently producing a
transcript-less stage.

Why it exists: task 2744 (2026-07-18) redirected recon stages to a per-run
config dir under ``<data_dir>/recon-config/`` — neither ``/tmp`` nor
``<cwd>/.task``, i.e. outside every writable root either backend grants.  The
CLI's session-JSONL writes were denied, so ``count_transcript_turns`` returned
None forever: the liveness watchdog degraded to inert and every cap-retry
force-freshed instead of resuming.  That ran silently until 2026-08-11.  A
comment claiming the dir was writable existed the entire time; only a check can
hold an invariant a comment cannot.
"""

from __future__ import annotations

import os
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


def _writable_roots(cwd: Path, writable_extras: list[str] | None) -> list[str]:
    """Roots BOTH backends make writable: ``/tmp``, ``<cwd>/.task``, each extra.

    Computed backend-agnostically on purpose — the containment invariant must
    hold whether Landlock or bwrap wins resolution, and both grant exactly these
    (``build_landlock_command`` at landlock.py:69-108, ``build_bwrap_command``
    at sandbox.py:56-101).

    Two details keep the check honest:

    - ``os.path.realpath`` on every root. Landlock resolves its rules by O_PATH
      fd — i.e. by real path — so a symlinked ``/tmp`` or a symlinked data dir
      must not produce a false PASS or a false FAIL here.
    - Roots that do not exist as directories are DROPPED. ``landlock_exec._add_path``
      returns silently for a non-existent path, adding no rule and raising
      nothing, so a grant naming a missing dir is vacuous. Counting it would let
      the check pass while the write still failed at runtime — precisely the
      silent-degrade class this module is meant to end.
    """
    roots = [
        os.path.realpath('/tmp'),
        os.path.realpath(os.path.join(str(Path(cwd).resolve()), '.task')),
    ]
    roots += [os.path.realpath(extra) for extra in (writable_extras or [])]
    return [root for root in roots if os.path.isdir(root)]


def _assert_config_dir_writable(
    config_dir: Path,
    cwd: Path,
    writable_extras: list[str] | None,
) -> None:
    """Raise unless *config_dir* is inside the writable set (task 4003).

    Containment is prefix-WITH-SEPARATOR, never a bare ``str.startswith`` on the
    root: ``/tmp-evil`` is not inside ``/tmp``.
    """
    raw = str(config_dir)
    resolved = os.path.realpath(raw)
    roots = _writable_roots(cwd, writable_extras)

    for root in roots:
        if resolved == root or resolved.startswith(root + os.sep):
            return

    # Name the raw path as well as the resolved one: the operator configured the
    # former, but containment is decided on the latter, and a symlink between
    # them is exactly the case where a bare resolved path reads as a non sequitur.
    shown = raw if resolved == raw else f'{raw} (resolved: {resolved})'
    raise RemediationSandboxUnavailable(
        f'Refusing to launch a reconciliation agent whose CLAUDE_CONFIG_DIR is '
        f'OUTSIDE the sandbox writable set: {shown}. The CLI would be told to '
        f'write its session transcript there and then denied the write by the '
        f'kernel — silently (this is the 2026-07-18 -> 2026-08-11 recon defect, '
        f'task 4003). Writable roots resolved for this invocation: '
        f'{roots or "(none — no writable root exists on disk)"}. '
        f'The per-run config dir is normally granted automatically by '
        f'cli_stage_runner.run_stage_via_cli; if you are seeing this, that '
        f'computed grant was lost. Note that a path listed in '
        f'reconciliation.sandbox_recon_writable_extras is IGNORED here unless the '
        f'directory actually exists — landlock_exec skips a non-existent path '
        f'silently, so such a grant would be vacuous. Do NOT "fix" this by adding '
        f'the config-dir BASE (<data_dir>/recon-config) to '
        f'reconciliation.sandbox_recon_writable_extras: that is the root under '
        f'which every run\'s claude-config-<run_id>/.credentials.json lives, and '
        f'granting it would give every recon stage write access to every other '
        f'run\'s credentials.'
    )


def resolve_recon_sandbox_wrap(
    cwd: Path,
    writable_extras: list[str] | None = None,
    *,
    config_dir: Path | None = None,
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
             set (e.g. a uvx/pip cache dir used by a stdio MCP server, or —
             for recon stages — the per-run ``CLAUDE_CONFIG_DIR``, appended by
             ``cli_stage_runner.run_stage_via_cli``).
             These are passed through to the underlying backend's
             ``writable_extras`` parameter.
        config_dir: Optional per-run ``CLAUDE_CONFIG_DIR`` (task 4003). When
             given, this function VERIFIES that the path is contained in the
             writable set that will be built and raises otherwise — it does not
             grant it; granting is the caller's job (policy in the caller,
             verification here). ``None`` skips the check entirely, keeping the
             generic and non-recon call sites byte-identical.

    Returns:
        A ``Callable[[list[str]], list[str]]`` that wraps the inner argv.

    Raises:
        RemediationSandboxUnavailable: When neither Landlock nor bwrap is
            available, when the orchestrator backends cannot be imported, OR
            when ``config_dir`` is outside the writable set.
    """
    # The containment check runs ONCE, before the backend branch, because the
    # invariant is backend-independent: Landlock and bwrap grant the same roots
    # (/tmp, <cwd>/.task, extras). Checking here rather than inside each branch
    # means a future third backend cannot be added without inheriting it.
    if config_dir is not None:
        _assert_config_dir_writable(config_dir, cwd, writable_extras)

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
