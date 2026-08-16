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
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
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


class PreformattedFanoutError(ValueError):
    """A fan-out failure whose message is ALREADY a rendered ``'Type: message'``.

    :func:`first_success` renders every caught exception through
    :func:`describe_exc`, which unconditionally prepends
    ``type(exc).__name__``. A call site that has already formatted the *real*
    cause — because it caught it, logged it, and re-raised to signal
    fall-through — therefore reached the operator doubled:
    ``'ValueError: ConnectError: refused'`` in the ``cancel_ticket`` 502
    ``detail`` and in the dashboard's offline pill. Raise this instead of a
    bare ``ValueError`` when the message you pass is final.

    Subclassing ``ValueError`` (rather than adding a new type to
    ``first_success``'s catch tuple) keeps control flow and every caller's own
    ``except ValueError`` unchanged. A marker type is used rather than having
    ``describe_exc`` sniff the message for an existing ``'Type: message'``
    shape because sniffing is a heuristic on operator-visible text: any
    legitimate message whose first token happened to look like an identifier
    followed by ``': '`` would be silently stripped of its real type name,
    with no way for a call site to opt out.
    """


def describe_exc(exc: BaseException) -> str:
    """Render *exc* as ``'Type: message'``, or just ``'Type'`` when empty.

    Several exceptions on this path stringify to ``''`` — most importantly
    ``httpx.PoolTimeout``, which means the shared client's *own* connection
    pool is saturated rather than the server being down. Formatting with a
    bare ``str(exc)`` turns those into content-free log lines ("failed for
    <url>: ") and content-free offline pills; always naming the type keeps
    client-side saturation distinguishable from a genuinely dead endpoint.

    A non-empty :class:`PreformattedFanoutError` is the one exception: its
    message is already a rendered cause, so it is returned verbatim rather
    than gaining a second prefix. An *empty* one still falls through to the
    generic path above, so opting in can never reintroduce the content-free
    line this function exists to prevent.
    """
    text = str(exc)
    if isinstance(exc, PreformattedFanoutError) and text:
        return text
    return f'{type(exc).__name__}: {text}' if text else type(exc).__name__


def project_label(project_root: str | os.PathLike[str]) -> str:
    """Render *project_root* as the short project name the UI labels it with.

    The basename, falling back to the full string for a root with no basename
    (``'/'``) so a label can never degrade to empty. This is the single
    definition of that rule for the fan-out cluster — :func:`fanout_label`
    composes it rather than re-deriving it.

    ``active_tasks._project_label`` and ``redux_api._project_label`` are
    independent hand-rolled copies of the same rule. They are not imported here
    (``active_tasks`` imports from ``tasks``, which imports this module), which
    also means the delegation can only run the other way: those two can
    eventually call *this*, collapsing three copies onto one. That cross-module
    edit is out of task 4133's module lock and is filed as follow-up work; until
    it lands, the three definitions must be kept string-identical by hand.
    """
    root_str = str(project_root)
    return Path(root_str).name or root_str


def fanout_label(base: str, project_root: str | os.PathLike[str]) -> str:
    """Compose a per-project-root fan-out log label, ``'base[project-name]'``.

    **Every fan-out caller parameterized by project_root MUST compose its
    ``log_label`` through this helper.** The contract is not cosmetic — it is
    what makes the transition-only policy above hold at all:

    - the throttle key is ``(log_label, url)``;
    - ONE fused-memory URL serves *every* project_root, so a fixed literal
      label collapses all roots onto a single key;
    - :func:`note_fanout_success` **pops** that key, so a healthy root's
      success in the same UI poll cycle clears a broken root's open streak.
      The broken root's next failure is therefore ``streak == 1`` again,
      re-arming the opening WARNING *and* adding a 'recovered' WARNING —
      every cycle, indefinitely. That is precisely the sustained flood the
      transition-only policy exists to prevent (task 3871), reintroduced
      through the key rather than the level.

    A collapsed key also erases the diagnosis: the message names only the
    shared URL, so the operator cannot tell *which* project_root is down.

    The discriminator is :func:`project_label` — the basename, deliberately
    string-identical to ``active_tasks._project_label`` /
    ``redux_api._project_label`` so operator log labels match the project chips
    the UI already renders. ``mcp_fanout``, the leaf of this cluster, is the
    helper's home (see :func:`project_label`) and this docstring is the single
    place the convention is written down.

    **The guarantee above assumes project-root basenames are distinct.** Two
    configured roots sharing one (``/srv/team-a/app`` and ``/srv/team-b/app``,
    or two checkouts both named ``dark-factory``) collapse back onto a single
    key and reintroduce both failure modes described above. That assumption is
    pre-existing and system-wide rather than introduced here —
    ``scheduler``'s ``label_to_root = {_project_label(r): r for r in ...}``
    already silently drops one of two same-basename roots, and ``redux_api``
    keys its whole per-project payload on the same basename — so this helper
    inherits it deliberately instead of diverging from every other project
    label the dashboard renders. For ``list_tickets`` the unambiguous full root
    is in any case still in the message body (metrics appends
    ``(project_root=...)``).
    """
    return f'{base}[{project_label(project_root)}]'


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
