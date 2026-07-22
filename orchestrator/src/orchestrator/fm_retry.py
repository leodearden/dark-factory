"""Single-source fm-restart retry window for orchestrator<->fused-memory clients.

Every orchestrator call into fused-memory (MCP session transport, the
scheduler's ``set_task_status``, and ``DeterministicRunner``'s post-deploy
writeback) can transiently fail while fused-memory restarts. Historically each
call site invented its own, much-too-small retry budget (McpSession ~7s,
scheduler ~4.5s, writeback ~62s) — all smaller than fm's own restart window
(systemd ``TimeoutStopSec=90`` + observed ~15s start), so a *successful*
restart could still surface as a client-visible hard failure.

This module is the ONE shared source: :data:`FM_RESTART_RETRY_WINDOW_SECS`
and :func:`fm_retry_backoffs`, imported by ``mcp_lifecycle``, ``scheduler``,
and ``deterministic_runner`` — no per-callsite copies of the number.
"""

from __future__ import annotations

import random
from collections.abc import Callable

# Basis: 90s systemd TimeoutStopSec ceiling + ~15s observed fm start + margin.
# Consumed by McpSession's transport retry loop, Scheduler.set_task_status's
# transient-rejection retry loop, and DeterministicRunner's
# _writeback_deploy_success budget — single source, no per-callsite copies.
FM_RESTART_RETRY_WINDOW_SECS: float = 120.0

# The D8 null sentinel — single-sourced here (below the transport; this module
# is stdlib-only, so evals/profile.py can import it DOWN with no cycle). Eval
# runs set EVAL_PROFILE['fused_memory.url'] to this non-routable address
# (orchestrator/src/orchestrator/evals/profile.py) to neutralize every
# orchestrator-side fused-memory write with an immediate ECONNREFUSED. Because
# 127.0.0.1:1 can NEVER host a *restarting* fused-memory, the fm-restart retry
# window above is inapplicable to it: riding the window out against a
# permanently-dead sentinel just burns ~FM_RESTART_RETRY_WINDOW_SECS per
# logical McpSession call (task 2880). Defined here so the McpSession
# fail-fast seam and EVAL_PROFILE reference exactly one string and cannot drift.
FM_NULL_SENTINEL_URL: str = 'http://127.0.0.1:1'


def is_fm_null_sentinel(url: str | None) -> bool:
    """Return True iff *url* is the D8 null sentinel (:data:`FM_NULL_SENTINEL_URL`).

    Falsy input (``None`` / ``''``) is never the sentinel. Otherwise *url* is
    normalized — a trailing ``'/'`` is stripped, then a trailing ``'/mcp'``
    endpoint suffix (and any ``'/'`` it exposes) — before an exact compare, so
    both the already-``rstrip('/')``-ed ``McpSession.base_url`` and a raw
    ``memory_url + '/mcp'`` endpoint resolve to the sentinel. A real fm
    endpoint (e.g. ``http://localhost:8002``) is never misclassified, so it
    keeps the full fm-restart retry window.
    """
    if not url:
        return False
    normalized = url.rstrip('/')
    if normalized.endswith('/mcp'):
        normalized = normalized[: -len('/mcp')].rstrip('/')
    return normalized == FM_NULL_SENTINEL_URL


# Private tunables for the jittered-exponential schedule below.
_FM_RETRY_BASE_SECS: float = 1.0
_FM_RETRY_CAP_SECS: float = 20.0
_FM_RETRY_MAX_SLEEPS: int = 32


def fm_retry_backoffs(
    window_secs: float = FM_RESTART_RETRY_WINDOW_SECS,
    *,
    base_secs: float = _FM_RETRY_BASE_SECS,
    cap_secs: float = _FM_RETRY_CAP_SECS,
    max_sleeps: int = _FM_RETRY_MAX_SLEEPS,
    rng: Callable[[float, float], float] | None = None,
) -> list[float]:
    """Return a bounded, decorrelated schedule of between-attempt sleeps.

    The schedule is a jittered-exponential series: step *i* draws uniformly
    from ``[exp/2, exp]`` where ``exp = min(cap_secs, base_secs * 2**i)``.
    Jitter decorrelates a multi-unit fleet's retries against a single shared
    fm instance, while the deterministic exponential floor (``exp/2``)
    guarantees the schedule spans ``window_secs`` regardless of the actual
    draws. The final sleep is clamped to exactly the remaining budget, so
    ``sum(schedule) == window_secs`` whenever the window is reachable within
    ``max_sleeps`` steps. ``max_sleeps`` hard-bounds the attempt count for an
    unreachable (too-small max_sleeps / too-large window) configuration.

    ``attempts = len(schedule) + 1`` — the sleeps are *between* attempts, so
    callers never sleep after the final attempt.

    Args:
        window_secs: Total time the schedule should span.
        base_secs: Exponential backoff base (seconds).
        cap_secs: Per-step exponential ceiling (seconds).
        max_sleeps: Hard cap on the number of sleeps returned.
        rng: Injectable ``(lo, hi) -> float`` draw function, defaulting to
            ``random.uniform``. Tests inject a deterministic boundary
            function (e.g. ``lambda lo, hi: hi``) for exact assertions.

    Returns:
        A list of between-attempt sleep durations (seconds).
    """
    uniform = rng or random.uniform
    schedule: list[float] = []
    total = 0.0
    i = 0
    while total < window_secs and len(schedule) < max_sleeps:
        exp = min(cap_secs, base_secs * (2 ** i))
        nxt = uniform(exp / 2, exp)
        nxt = min(nxt, window_secs - total)
        if nxt <= 0:
            break
        schedule.append(nxt)
        total += nxt
        i += 1
    return schedule
