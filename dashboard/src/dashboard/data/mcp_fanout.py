"""Shared MCP fan-out idioms: first-success failover and a short-TTL cache.

Two patterns recur across the dashboard's MCP data-fetchers
(``dashboard.data.memory``, ``dashboard.data.tasks``, and — imminently —
``dashboard.data.scheduler``/``metrics``/``app``/``merge_queue``):

1. **Fan-out-with-failover** (:func:`first_success`) — call an MCP tool on
   each configured fused-memory URL in order, returning the first success;
   on a transport error or a "soft failure" (a malformed/errored MCP result,
   signalled by the caller raising ``ValueError``) invalidate that URL's
   cached session and fall through to the next URL. If every URL fails,
   return a caller-defined offline sentinel built from the collected
   per-URL error strings.

2. **Single-flight short-TTL cache** (:class:`TTLCache`) — memoize an
   expensive async refresh for a few seconds so concurrent/rapid callers
   collapse onto one in-flight refresh instead of hammering MCP.

Both are extracted here from their original call sites so new consumers
do not have to re-derive the failover/caching discipline from scratch.

The extraction is behavior-preserving for ``dashboard.data.memory``, whose
``_first_success`` already invalidated the failing session on both
transport errors and ``ValueError``. For ``dashboard.data.tasks`` it is
behavior-preserving *except* for one intentional normalization: a "soft
failure" (a malformed/errored MCP result) previously fell through to the
next URL with a bare ``continue`` and no session teardown, whereas now it
is signalled by raising ``ValueError`` from within ``call``, which
``first_success`` treats the same as a transport error — including
invalidating that URL's cached session. This is a deliberate unification
with memory.py's pre-existing behavior, not a regression: re-initializing
a session after a soft failure is cheap and strictly more conservative,
and no caller depends on the session surviving one.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Generic, TypeVar

import httpx

logger = logging.getLogger(__name__)

V = TypeVar('V')


# ── per-URL failure log policy (task 3871) ──────────────────────────
#
# A fan-out failure must leave a journal trace (the dashboard runs at the
# default WARNING root level, so DEBUG means a total outage is invisible —
# task 1814's class of bug), but it must NOT become a log flood. first_success
# sits on ~8 hot paths and the UI polls every 2s, so a *sustained* outage at
# one-WARNING-per-failure would emit hundreds of identical lines per minute,
# indefinitely — burying the very signal it is trying to preserve.
#
# Policy: WARNING on TRANSITION (the first failure of a streak, and again on
# recovery), DEBUG for the repeats in between, plus a bounded heartbeat every
# _FANOUT_REWARN_EVERY consecutive failures so an outage that outlives log
# rotation still leaves a trace. Streaks are keyed per (log_label, url), so
# each path/endpoint pair reports independently.
_FANOUT_REWARN_EVERY = 500

_failure_streaks: dict[tuple[str, str], int] = {}


def describe_exc(exc: BaseException) -> str:
    """Render *exc* as ``'Type: message'``, or just ``'Type'`` when empty.

    Several exceptions on this path stringify to ``''`` — most importantly
    ``httpx.PoolTimeout``, which means the shared client's *own* connection
    pool is saturated rather than the server being down. Formatting with a
    bare ``str(exc)`` turns those into content-free log lines ("failed for
    <url>: ") and content-free offline pills; always naming the type keeps
    client-side saturation distinguishable from a genuinely dead endpoint.
    """
    text = str(exc)
    return f'{type(exc).__name__}: {text}' if text else type(exc).__name__


def log_fanout_failure(log_label: str, url: str, exc: BaseException) -> None:
    """Record one per-URL fan-out failure under the transition-only policy.

    WARNING for the first failure of a streak (and every
    ``_FANOUT_REWARN_EVERY``-th thereafter), DEBUG for the repeats. Shared by
    :func:`first_success` and by the hand-rolled per-URL loops in
    ``dashboard.data.memory`` so the whole invisible-failure class is closed
    with one policy rather than three divergent ones.
    """
    key = (log_label, url)
    streak = _failure_streaks.get(key, 0) + 1
    _failure_streaks[key] = streak
    detail = describe_exc(exc)
    if streak == 1:
        logger.warning('%s failed for %s: %s', log_label, url, detail)
    elif streak % _FANOUT_REWARN_EVERY == 0:
        logger.warning(
            '%s still failing for %s (%d consecutive): %s',
            log_label, url, streak, detail,
        )
    else:
        logger.debug(
            '%s failed for %s (%d consecutive): %s', log_label, url, streak, detail,
        )


def note_fanout_success(log_label: str, url: str) -> None:
    """Close an open failure streak for *(log_label, url)*, logging recovery.

    Emits at WARNING — the same level as the streak's opening line — so the
    incident has a visible closing bracket at the operator's default level.
    Bounded by construction: it fires only when a streak was actually open,
    i.e. at most once per opening WARNING.
    """
    streak = _failure_streaks.pop((log_label, url), 0)
    if streak:
        logger.warning(
            '%s recovered for %s after %d consecutive failure(s)',
            log_label, url, streak,
        )


def reset_failure_streaks() -> None:
    """Forget all open failure streaks (test/admin hook).

    Called by ``dashboard.data.memory.reset_sessions`` so the one fixture that
    every session-touching test module already uses also gives each test a
    clean throttling state — otherwise an earlier test's failure would silently
    demote a later test's first WARNING to DEBUG.
    """
    _failure_streaks.clear()


async def first_success(
    urls: Sequence[str],
    call: Callable[[str], Awaitable[V]],
    *,
    log_label: str,
    offline_result: Callable[[list[str]], V],
    log_failures: bool = True,
) -> V:
    """Call *call(url)* for each URL in order; return the first success.

    ``call`` is invoked with one URL at a time and must return an awaitable
    (a single MCP tool call, or a coroutine performing several paired calls
    against that URL). On:

    - ``httpx.ConnectError`` / ``httpx.TimeoutException`` / ``httpx.HTTPStatusError``
      — a transport-level failure;
    - ``ValueError`` — a caller-detected "soft failure" (e.g. a structured
      MCP error dict or an empty/malformed result) that *call* raises to
      signal fall-through;

    the failing URL's cached MCP session is invalidated, the error is
    recorded, and the loop continues to the next URL. Any other exception
    type propagates uncaught.

    If every URL fails, returns ``offline_result(errors)`` where *errors*
    is the list of collected ``f'{url}: {e}'`` strings — letting each
    caller reproduce its own existing offline shape (e.g.
    ``{'offline': True, 'error': '; '.join(errors)}``) while preserving the
    per-URL error detail.

    **Per-URL failures are surfaced at WARNING on transition only** — see
    :func:`log_fanout_failure`. The fall-through is silent to the caller (a
    total outage just yields the offline sentinel), so logging at DEBUG would
    leave *no* journal trace at all at the dashboard's default WARNING root
    level, and the operator would see an "offline" pill with no recorded cause
    (task 1814's class of bug, which this path had regressed into). Logging
    every repeat at WARNING is the opposite failure — this helper is on ~8 hot
    paths behind a 2s UI poll — so a streak reports once at its start, once at
    its end, and DEBUG in between. The message names the exception *type*, so
    an ``httpx.PoolTimeout`` (this client's own pool is saturated) is
    distinguishable from a genuinely unreachable endpoint.

    ``log_failures=False`` suppresses that reporting for the two ``app.py``
    proxies that already emit their own fully-detailed WARNING at the call
    site: the failure is then reported exactly once, by the caller, rather
    than twice at the same level.
    """
    # Local import breaks the memory<->mcp_fanout import cycle: memory.py
    # imports first_success at module top, so invalidate_session (which
    # must stay defined in memory.py) can only be imported here lazily,
    # deferring resolution until call time (after both modules are loaded).
    from dashboard.data.memory import invalidate_session

    errors: list[str] = []
    for url in urls:
        try:
            result = await call(url)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError,
                ValueError) as e:
            if log_failures:
                log_fanout_failure(log_label, url, e)
            errors.append(f'{url}: {describe_exc(e)}')
            invalidate_session(url)
        else:
            if log_failures:
                note_fanout_success(log_label, url)
            return result
    return offline_result(errors)


class TTLCache(Generic[V]):
    """Single-flight, short-TTL cache keyed by an arbitrary string.

    Generalizes scheduler.py's ``_scheduler_cache`` +
    ``_scheduler_refresh_lock`` double-checked-locking pattern: a warm entry
    is served without any lock; a cold or expired entry causes exactly one
    concurrent caller to run ``refresh`` while the rest wait on a per-key
    lock and then re-check freshness (double-checked locking) rather than
    each running their own redundant refresh.

    ``ttl_seconds`` accepts a plain float OR a zero-arg callable, resolved
    at *each* freshness check rather than captured once at construction —
    this is what lets a caller monkeypatch a module-level TTL constant at
    runtime (as ``test_tasks.py`` does for ``_FETCH_TASKS_TTL_SECONDS``) and
    have it take effect immediately.

    ``cache_ok`` (per-call, default always-true) gates whether a given
    refresh result is stored — e.g. an offline/error marker should not pin
    itself in the cache for the TTL window. The cache is value-type-agnostic
    and always returns the raw stored value; copy isolation (if needed) is
    the caller's responsibility.

    **Single-flight only holds for cacheable results.** The "exactly one
    refresh" guarantee above applies while the produced value passes
    ``cache_ok``. If ``cache_ok(value)`` is False, nothing is stored, so the
    next lock-queued waiter's post-lock freshness re-check still misses and
    it runs its own ``refresh`` in turn — concurrent cold callers during an
    outage each perform a full refresh (serialized one-at-a-time by the
    per-key lock, not run in parallel) rather than collapsing onto one. This
    is intentional: a non-cacheable result must not be handed to every other
    waiter when a subsequent attempt might succeed.

    **Unbounded key space is a caller assumption, not a guarantee.** Neither
    ``_store`` nor ``_locks`` ever evict individual entries — only the
    blanket ``clear()`` resets them — so a distinct key accumulates a store
    slot and a ``Lock`` forever once created. This is safe for the current
    callers (each keyed by a small, bounded set of project roots), but a
    future high-cardinality key space (e.g. keyed per-request or per-user)
    would leak locks and stale entries indefinitely; such a caller should
    add its own eviction or key externally.
    """

    def __init__(self, ttl_seconds: float | Callable[[], float]):
        self._ttl_fn: Callable[[], float] = (
            ttl_seconds if callable(ttl_seconds) else (lambda: ttl_seconds)
        )
        self._store: dict[str, tuple[float, V]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def get_fresh(self, key: str) -> V | None:
        """Return the cached value for *key* iff still within TTL, else None."""
        cached = self._store.get(key)
        if cached is not None and (time.monotonic() - cached[0]) < self._ttl_fn():
            return cached[1]
        return None

    async def get_or_refresh(
        self,
        key: str,
        refresh: Callable[[], Awaitable[V]],
        *,
        cache_ok: Callable[[V], bool] = lambda v: True,
    ) -> V:
        """Return a fresh cached value for *key*, refreshing at most once.

        Fast path: a warm entry is returned with no lock. Otherwise acquires
        a per-key lock, re-checks freshness (another waiter may have just
        filled it), and — if still cold — runs ``refresh()`` and stores the
        result iff ``cache_ok(value)``.
        """
        fresh = self.get_fresh(key)
        if fresh is not None:
            return fresh

        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            fresh = self.get_fresh(key)
            if fresh is not None:
                return fresh

            value = await refresh()
            if cache_ok(value):
                self._store[key] = (time.monotonic(), value)
            return value

    def clear(self) -> None:
        """Reset the store and all per-key locks (test/admin hook)."""
        self._store.clear()
        self._locks.clear()
