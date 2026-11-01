"""Sandbox backend dispatcher.

Chooses between ``bwrap`` and ``landlock`` at wrap-time, based on a
process-global preference set once at startup from
``SandboxConfig.backend``. Each of the three agent backends
(claude/codex/gemini) calls ``wrap_command()`` at its sandbox point.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, get_args

logger = logging.getLogger(__name__)

Backend = Literal['auto', 'bwrap', 'landlock', 'none']

_preferred: Backend = 'auto'

# Process-global escalation-dedup state for resolve_backend_or_refuse (task
# 2908, INV-4). Keyed on the configured backend string (_preferred): holds the
# backend state for which a fail-closed escalation was last signalled, so the
# escalation fires exactly ONCE across N consecutive refusals and re-arms when
# the backend state changes. Lives here — alongside the process-global
# _preferred — rather than on a per-task TaskWorkflow, because the dedup is
# shared across every workflow in one orchestrator process. Reset (re-armed) on
# ANY non-refusing return of resolve_backend_or_refuse (healthy backend OR the
# 'none' escape hatch).
#
# Re-arm is driven ONLY by backend state, never by escalation lifecycle: an
# operator resolving the filed escalation does NOT re-arm this dedup. So while
# the backend stays unavailable, subsequent refusals remain deduped — they
# still fail-closed (raise SandboxUnavailable) but file NO new escalation —
# until the backend state actually changes (a config edit via sandbox.backend /
# reload, or a landlock/bwrap recovery). This is an observability caveat, not a
# safety gap (every refusal still raises); it is spelled out in the escalation
# detail so an operator who resolves without installing a backend understands
# the guard stays silent until the backend state changes (task 2908 review).
_escalated_backend_state: str | None = None


def set_backend(name: Backend) -> None:
    """Set the preferred sandbox backend. Called once at orchestrator startup.

    Raises TypeError if ``name`` is not a ``str``.
    Raises ValueError if ``name`` is a ``str`` but not in the Backend Literal.
    """
    valid = get_args(Backend)
    if not isinstance(name, str):
        raise TypeError(
            f'set_backend: expected a str, got {type(name).__name__!r}'
        )
    if name not in valid:
        raise ValueError(
            f'set_backend: invalid backend {name!r}; expected one of {valid}'
        )
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
    # On the unavailable branch, return 'none' and let the CALLER decide the
    # outcome — do NOT assert "running unsandboxed" here. resolve_active_backend
    # is shared by a fail-OPEN caller (invoke._invoke_claude_with_sandbox, which
    # runs unsandboxed) AND a fail-CLOSED caller (resolve_backend_or_refuse,
    # which REFUSES). Claiming an unsandboxed run in the log would be false on
    # the fail-closed path (task 2908 review).
    if _preferred == 'landlock':
        from orchestrator.agents.landlock import is_landlock_available
        if is_landlock_available():
            return 'landlock'
        logger.warning(
            'sandbox backend=landlock configured but unavailable — resolving to '
            'none (caller decides: run unsandboxed or refuse fail-closed)',
        )
        return 'none'
    if _preferred == 'bwrap':
        from orchestrator.agents.sandbox import is_bwrap_available
        if is_bwrap_available():
            return 'bwrap'
        logger.warning(
            'sandbox backend=bwrap configured but unavailable — resolving to '
            'none (caller decides: run unsandboxed or refuse fail-closed)',
        )
        return 'none'
    # set_backend's input validator blocks corrupt values from entering via the
    # public API, so reaching this branch means _preferred was mutated directly
    # (attribute assignment) past the validator. Fail loudly rather than
    # silently invoke the agent unsandboxed.
    raise RuntimeError(
        f'sandbox_dispatch._preferred is corrupted: {_preferred!r}',
    )


class SandboxUnavailable(RuntimeError):
    """Raised when confinement is required but no OS sandbox backend resolves.

    Fail-CLOSED guard for the sandboxed-dispatch call-site (task 2908, PRD
    plans/os-sandbox-worktree-containment-prd.md D4 / invariant INV-4). Mirrors
    reconciliation's ``RemediationSandboxUnavailable`` (task 1935,
    fused_memory/reconciliation/sandbox_guard.py): when sandboxing is required
    (``config.sandbox.enabled`` + ``role.sandboxed``) but the configured
    backend resolves to no available OS sandbox (landlock/bwrap), the caller
    must REFUSE to launch the agent unconfined rather than silently fall
    through to an unsandboxed run.

    Attributes:
        should_escalate: The process-global dedup decision — True on the first
            refusal for a given backend state, False on subsequent refusals
            (INV-4: exactly one escalation across N refused invocations).
        backend_state: The configured backend that failed to resolve
            (``_preferred``) — the dedup key and escalation detail.
    """

    def __init__(
        self, message: str, *, should_escalate: bool, backend_state: str,
    ) -> None:
        super().__init__(message)
        self.should_escalate = should_escalate
        self.backend_state = backend_state


def resolve_backend_or_refuse() -> Backend:
    """Fail-CLOSED backend resolution for the sandboxed-dispatch call-site.

    Unlike ``resolve_active_backend()`` (which returns ``'none'`` when the
    configured backend is unavailable, leaving the caller free to run
    unsandboxed), this REFUSES: it raises ``SandboxUnavailable`` when
    confinement is required but no OS backend resolves.

    - ``backend == 'none'`` short-circuits FIRST: the explicit operator escape
      hatch. Returns ``'none'`` with a WARNING and never refuses (same effect
      as ``sandbox.enabled = false``; PRD D4). No availability probe is run.
    - Otherwise resolve via ``resolve_active_backend()``; return the real
      backend when landlock/bwrap is available.
    - RAISE ``SandboxUnavailable`` when the configured backend (auto/landlock/
      bwrap) resolves to nothing.
    """
    global _escalated_backend_state
    if _preferred == 'none':
        logger.warning(
            'sandbox enabled + role sandboxed but backend=none — running '
            'UNSANDBOXED (operator escape hatch)',
        )
        # Non-refusing return -> re-arm the dedup so a later re-break re-fires.
        _escalated_backend_state = None
        return 'none'
    resolved = resolve_active_backend()
    if resolved != 'none':
        # Healthy backend -> re-arm the dedup (recovery re-fires on next break).
        _escalated_backend_state = None
        return resolved
    # Refusal path: signal escalate only when the backend state differs from the
    # one we last escalated for (INV-4: exactly one escalation across N refusals,
    # re-armed on any backend-state change).
    should = _escalated_backend_state != _preferred
    if should:
        _escalated_backend_state = _preferred
    raise SandboxUnavailable(
        f'Sandboxing is required (sandbox.enabled + role.sandboxed) but the '
        f'configured backend={_preferred!r} resolved to no available OS '
        f'sandbox (landlock/bwrap). Refusing to launch the agent unconfined '
        f'(fail-closed, PRD D4). Install/enable landlock or bwrap, or set '
        f'sandbox.backend=none to explicitly run unsandboxed.',
        should_escalate=should,
        backend_state=_preferred,
    )


def reset_escalation_dedupe() -> None:
    """Re-arm the process-global fail-closed escalation dedup (test-only).

    Clears ``_escalated_backend_state`` so the next refusal re-signals escalate.
    Exposed for test fixtures that must isolate the process-global dedup state
    between tests; production re-arms implicitly on any non-refusing return of
    ``resolve_backend_or_refuse``.
    """
    global _escalated_backend_state
    _escalated_backend_state = None


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

    # set_backend's input validator blocks corrupt values from entering via the
    # public API, so reaching this branch means _preferred was mutated directly
    # (attribute assignment) past the validator. Fail loudly rather than
    # silently invoke the agent unsandboxed.
    raise RuntimeError(
        f'sandbox_dispatch._preferred is corrupted: {backend!r}',
    )


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
