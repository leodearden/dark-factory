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


# ── bounded per-key lock acquisition (task 4789) ────────────────────
#
# TTLCache.get_or_refresh used to acquire its per-key lock with a bare
# `async with lock:` — unbounded. On 2026-08-27 a refresh that never
# returned (parked forever inside httpcore) held one key's lock for 19.8h
# with 7 waiters queued behind it: because the holder never returned,
# `cache_ok` never ran and nothing was ever stored, so every later caller
# missed the warm fast path and queued on a latch that would only have
# released at process exit.
#
# _LOCK_ACQUIRE_TIMEOUT_SECONDS bounds that wait. It must exceed every
# observed HEALTHY refresh so happy-path single-flight is never disturbed —
# post-restart the incident's own endpoints returned in 0.22s / 0.3s /
# 4.2s, and tasks.fetch_tasks documents a cold-session worst case of
# roughly 3 * DEFAULT_PER_CALL_TIMEOUT (~6s) per URL — while converting an
# UNBOUNDED wedge (19.8h measured) into a bounded one.
#
# This bound is UNREACHABLE from active_tasks.collect_tasks_with_counts's
# per-project path (the Tasks tab): _shape_one_project — which calls
# fetch_tasks/fetch_statuses, i.e. THIS bound's own _fetch_tasks_cache /
# _fetch_statuses_cache — runs under
# ``asyncio.wait_for(..., timeout=min(remaining, _TASKS_PER_PROJECT_BUDGET))``
# with active_tasks._TASKS_PER_PROJECT_BUDGET == 7.0, itself inside
# active_tasks._TASKS_TOTAL_BUDGET == 20.0. A caller on THAT path is
# cancelled at <= 7s, well before this 15s bound could ever fire, so a wedge
# there still surfaces exactly as it did before this module's fix: as a
# per-project "degraded" WARNING (rows/done-count UNKNOWN), never as a
# bypass. That is acceptable — collect_tasks_with_counts already has its own
# adequate degradation story for exactly this case, and this fix does not
# need to duplicate it; the fix is chosen for the callers below instead.
#
# The bound IS reachable from every OTHER live call site, because none of
# them has an enclosing deadline: app._task_cards_cache (which also reaches
# _fetch_tasks_cache, but via fetch_tasks called from _load_task_cards — a
# path with no _TASKS_PER_PROJECT_BUDGET-style wrapper, so the SAME cache
# can be bound-reachable or not depending on which caller reached it),
# app._analytics_cache, app._memory_evals_cache, scheduler._scheduler_cache,
# and merge_queue._task_titles_cache.
_LOCK_ACQUIRE_TIMEOUT_SECONDS = 15.0

# A bypass must leave its own journal trace — reusing the SAME
# transition-only policy as _FANOUT_REWARN_EVERY above rather than
# inventing a second one. The dashboard's UI polls every ~2-3s across ~8
# TTLCache-backed paths, so one WARNING per bypass would emit tens of
# thousands of identical lines over a 19.8h wedge and bury the opening
# (diagnostic) line — the exact flood the fan-out policy above was
# written to prevent.
_LOCK_BYPASS_REWARN_EVERY = 100


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

    **Single-flight is bounded, not guaranteed.** Lock acquisition on the cold
    path waits at most ``_LOCK_ACQUIRE_TIMEOUT_SECONDS``; a refresh that
    outlives the bound is bypassed — run without the lock. Concurrent
    bypasses for the SAME key still collapse onto one shared refresh
    (:meth:`_bypass_refresh`), but joining that shared refresh is ITSELF
    bounded (task 4789 review fix): a caller waits at most
    ``_LOCK_ACQUIRE_TIMEOUT_SECONDS`` again on a bypass it did not start, and
    never on one already known to have outlived that bound — such a caller
    is instead promoted to running its own refresh, superseding the dead one
    for whoever comes next. A caller in that position is in exactly the
    position of a cold caller on a healthy cache: bounded by that refresh's
    own per-HTTP-request budgets (``mcp_tool_call`` threads ``timeout`` into
    every httpx post). The residual is stated explicitly rather than left
    for a reader to over-infer: a refresh whose OWN budgets fail to fire (the
    2026-08-27 httpcore hang) still never returns, and a caller that ends up
    running such a refresh still waits indefinitely — ``get_or_refresh`` is
    total by contract, so there is nothing else it could return.

    What IS guaranteed, and pinned by
    :class:`TestTTLCacheBoundedBypassInheritance`: no caller inherits a
    stall it did not itself cause. At most one TRACKED live bypass exists
    per key at any instant, and a sustained wedge produces new bypasses at
    roughly one per bound-window, never one per caller — an unbounded
    pile-up of one refresh per timed-out caller would itself risk
    exhausting the shared httpx client's connection pool
    (``httpx.PoolTimeout``) for keys the wedge never touched, exactly the
    resource-isolation regression this bound exists to avoid. This remains
    the same duplicate-concurrent-refresh trade the ``cache_ok``-is-False
    paragraph above already accepts, now also reachable when a *cacheable*
    refresh simply runs too long, not only when it fails outright. The
    alternative, a permanent latch, is not acceptable: on 2026-08-27 one
    refresh that never returned held a key's lock for 19.8h with 7 waiters
    queued behind it, leaving 3 of 14 endpoints dead for the duration. A
    bypass — and a re-arm of one already over-age — is never silent, see
    :meth:`get_or_refresh`'s bypass-logging policy: a silent one would hide
    the next occurrence, the same invisibility that let the incident run
    unnoticed for as long as it did.

    **Key space is bounded by disuse, not by cardinality.** A key that stops
    being requested is reclaimed: :meth:`_evict_expired` drops store entries
    older than ``_EVICTION_TTL_MULTIPLE`` times the TTL, along with their
    idle locks, any bypass-streak counter that has outlived both, and any
    now-dead bypass-task entry, and ``get_or_refresh`` runs it on the cold
    path — the same path that mints new keys, so growth and reclamation are
    coupled by construction. An entry past its TTL is already unservable
    (``get_fresh`` enforces the TTL and is the only reader), so eviction
    reclaims memory with NO semantic change; the multiple is margin against a
    callable ``ttl_seconds`` that returns a larger value later. (The
    in-flight bypass-task map introduced below has two independent
    reclamation paths, not one: an entry whose refresh COMPLETES is removed
    by the task's own done-callback the instant it finishes (see
    :meth:`_start_bypass`), while an entry whose refresh never completes is
    removed either by supersession — the next caller to time out on it
    re-arms a fresh one, see :meth:`_bypass_refresh` — or, for a key that
    goes quiet before that happens, by :meth:`_evict_expired` on a later cold
    miss for a DIFFERENT key. Its steady-state size is therefore bounded by
    the same "keys requested within the eviction horizon" rule as ``_store``
    and ``_locks`` above, not by the stronger claim that an entry can never
    outlive the refresh it tracks — outliving it is exactly the case this
    task exists to handle.)

    This makes a HIGH-CARDINALITY key space safe, which it previously was
    not. It used to be true that "neither ``_store`` nor ``_locks`` ever
    evict individual entries", and callers were told to key externally or
    add their own eviction. ``fetch_tasks`` then acquired a key derived from
    a paging offset computed off a live, monotonically-growing task count —
    minting a fresh key on every task completion, each permanently retaining
    a 400-row task list plus a ``Lock`` (task 3857 review). A caller cannot
    add eviction from outside (``_store`` is private), so the fix belongs
    here. Steady-state size is now "keys requested within the eviction
    horizon", regardless of how many distinct keys the caller has ever used.
    """

    # Multiple of the TTL after which an untouched entry is evicted. Entries
    # are unservable past 1x the TTL, so anything >= 1 is semantically free;
    # 4 leaves margin for a callable TTL that grows between calls.
    _EVICTION_TTL_MULTIPLE = 4

    def __init__(self, ttl_seconds: float | Callable[[], float]):
        self._ttl_fn: Callable[[], float] = (
            ttl_seconds if callable(ttl_seconds) else (lambda: ttl_seconds)
        )
        self._store: dict[str, tuple[float, V]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        # Per-INSTANCE (not module-level) consecutive-bypass streaks, keyed
        # by cache key. Per-instance because the key space is per-cache:
        # tasks._fetch_tasks_cache and tasks._fetch_tasks_negative_cache
        # share key strings exactly, and three more live instances key on a
        # bare project_root — a module-level dict would collapse them.
        self._bypass_streaks: dict[str, int] = {}
        # The in-flight bypass task for a key, if any, stamped with the
        # monotonic time it started. Bounds concurrent bypass refreshes to
        # ONE per key (see _bypass_refresh) rather than one per timed-out
        # caller. The timestamp lets a LATER caller decide whether this
        # bypass is still safe to inherit, or has itself outlived
        # _LOCK_ACQUIRE_TIMEOUT_SECONDS -- joining one already over-age
        # would just trade one dead task for another, so such a caller
        # starts a fresh bypass instead, superseding this entry (task 4789
        # review fix). A completed entry is removed instantly by the task's
        # own done-callback; a superseded entry is dropped by the caller
        # that replaces it.
        self._bypass_tasks: dict[str, tuple[float, asyncio.Task[V]]] = {}

    def get_fresh(self, key: str) -> V | None:
        """Return the cached value for *key* iff still within TTL, else None."""
        cached = self._store.get(key)
        if cached is not None and (time.monotonic() - cached[0]) < self._ttl_fn():
            return cached[1]
        return None

    def _evict_expired(self) -> int:
        """Drop store entries past the eviction horizon, their idle locks,
        and any now-dead bypass-streak counter.

        Returns the number of store entries dropped (for tests/observability).

        Safe by construction, in two independent senses:

        * SEMANTICALLY — an evicted entry is older than the TTL, and
          :meth:`get_fresh` (the only reader of ``_store``) already refuses
          to serve those. Eviction therefore changes no observable value,
          only resident memory. Dropping an over-age ``_bypass_tasks`` entry
          changes no observable value either: every joiner's own wait
          deadline IS that entry's deadline (``started_at +
          _LOCK_ACQUIRE_TIMEOUT_SECONDS``, see :meth:`_bypass_refresh`), so
          once an entry is over-age no caller can still be parked on it —
          dropping it strands nobody. Doing so also reclaims the last
          reference this cache holds to a wedged refresh's task, so the
          coroutine can be garbage-collected once it finally resolves or the
          event loop shuts down.
        * CONCURRENTLY — this method is fully synchronous, so it runs to
          completion without yielding to another coroutine, and a lock is
          dropped only when it is neither held nor awaited. A coroutine that
          has taken a lock object from ``_locks`` reaches ``lock.acquire()``
          with no intervening await (see ``get_or_refresh``), so it cannot be
          suspended between ``setdefault`` and entering ``acquire()``. This
          invariant survives the bounded acquisition (task 4789) ON CPYTHON
          >= 3.12: there, ``asyncio.wait_for`` is implemented as ``async with
          timeouts.timeout(...): return await fut``, so
          ``await asyncio.wait_for(lock.acquire(), ...)`` delegates into
          ``lock.acquire()`` as a coroutine, and neither that preamble nor
          ``Timeout.__aenter__`` contains an await — control never returns to
          the event loop before ``acquire()``'s body runs. This package's
          declared floor is lower (``requires-python = '>=3.11,<4'``), and on
          3.11 ``wait_for`` instead unconditionally wraps its argument in a
          ``Task`` via ``ensure_future`` and awaits a *separate* waiter
          future — control CAN return to the loop first, so a concurrent
          sweep could reclaim the still-idle, unwaited lock in that window,
          leaving two callers holding different ``Lock`` objects for one
          key. The blast radius matches the ``getattr``-fallback case just
          below — one duplicate refresh, never corruption — and the
          deployed interpreter is 3.13, where the invariant holds without
          qualification; this note exists so a reader on 3.11 knows the
          degradation is bounded, not silent.

          ``locked()`` alone would NOT be enough, which is why the predicate
          below also checks for waiters. ``asyncio.Lock.release()`` clears
          its ``_locked`` flag and merely SCHEDULES the first waiter's
          future; between that release and the waiter's task actually
          resuming, ``locked()`` reads False while a queued waiter is very
          much still using the lock. A sweep landing in that window — reached
          on any cold miss, and the key is absent from ``_store`` exactly
          during an outage, since ``cache_ok`` stores nothing then — would
          otherwise drop the lock, the next caller would ``setdefault`` a
          fresh one, and the two would refresh concurrently.

          The waiter probe reads ``asyncio.Lock``'s private ``_waiters``
          defensively via ``getattr``: if a future CPython drops the
          attribute, the predicate degrades to the ``locked()``-only test and
          the cost is at most one duplicate refresh in that narrow window —
          never corruption, and never a wrong value (the two refreshes write
          the same key). Dropping a genuinely idle lock is safe regardless:
          the next caller for that key simply creates a new one.
        """
        horizon = self._ttl_fn() * self._EVICTION_TTL_MULTIPLE
        now = time.monotonic()
        stale = [k for k, (stamp, _) in self._store.items() if (now - stamp) >= horizon]
        for key in stale:
            del self._store[key]
        # A lock still guarding a live entry is kept, so single-flight for an
        # actively-used key is never disturbed by a sweep.  ``_waiters`` covers
        # the released-but-not-yet-resumed window in which ``locked()`` lies —
        # see the docstring.
        idle = [
            k for k, lk in self._locks.items()
            if k not in self._store
            and not lk.locked()
            and not getattr(lk, '_waiters', None)
        ]
        for key in idle:
            del self._locks[key]
        # A bypass streak for a key with no live lock and no cached value
        # cannot be reached by anything but a fresh cold miss, so re-arming
        # its opening WARNING on a future bypass is the desired behavior,
        # not a regression (see _note_lock_bypass). Pruned AFTER the lock
        # sweep above so a lock idled out in THIS same pass has its streak
        # reclaimed in the same pass too, rather than lingering one more
        # sweep cycle.
        dead_streaks = [
            k for k in self._bypass_streaks
            if k not in self._store and k not in self._locks
        ]
        for key in dead_streaks:
            del self._bypass_streaks[key]
        # A tracked bypass entry is safe to drop once no caller can still be
        # waiting on it: its task has already finished (belt-and-braces —
        # _start_bypass's done-callback normally removes these itself, but
        # one that fired while the entry was already superseded has nothing
        # left to remove), OR it is over-age. The predicate is deliberately
        # age-based rather than the lock sweep's "not locked() and no
        # _waiters" above: every joiner's own wait deadline IS
        # `started_at + _LOCK_ACQUIRE_TIMEOUT_SECONDS` (see
        # _bypass_refresh), so once an entry passes that instant no caller
        # can still be parked on it and dropping it strands nobody — while a
        # WITHIN-bound entry must survive, since pruning it would silently
        # restore one-refresh-per-caller for every later timed-out caller of
        # this key. The abandoned task is never cancelled (see the design
        # decision on abandon-don't-cancel): it keeps running in the
        # background and may still store a late value.
        dead_bypasses = [
            k for k, (started_at, task) in self._bypass_tasks.items()
            if task.done() or (now - started_at) >= _LOCK_ACQUIRE_TIMEOUT_SECONDS
        ]
        for key in dead_bypasses:
            del self._bypass_tasks[key]
        return len(stale)

    async def _refresh_and_store(
        self,
        key: str,
        refresh: Callable[[], Awaitable[V]],
        cache_ok: Callable[[V], bool],
    ) -> V:
        """Sweep, run ``refresh()``, and store the result iff ``cache_ok`` AND still newest.

        Shared verbatim by the normal (locked) cold path and the bounded-
        acquisition bypass path in :meth:`get_or_refresh`, so there is exactly
        one definition of "sweep, refresh, store iff cache_ok" for both to
        stay in sync with.

        **Concurrent writers are possible for one key** once a bypass can run
        alongside the still-wedged original holder (task 4789): this method
        may be entered twice for the same *key* — once by the locked holder,
        once by :meth:`_bypass_refresh` — before either has stored, with no
        guarantee on which ``refresh()`` resolves first. A write is therefore
        only committed if no FRESHER entry has landed since THIS call started
        (tracked via ``started_at``, captured before ``refresh()`` runs) —
        otherwise a very late-returning wedged refresh could overwrite a
        newer bypass-produced value with a stale one, AND stamp it with a
        fresh ``time.monotonic()``, making the stale data look maximally
        fresh for a full TTL window. This never mattered before the bypass
        path existed: the lock made the two cold-path callers mutually
        exclusive, so only one writer per key was ever possible.
        """
        self._evict_expired()
        started_at = time.monotonic()
        value = await refresh()
        if cache_ok(value):
            current = self._store.get(key)
            if current is None or current[0] <= started_at:
                self._store[key] = (time.monotonic(), value)
        return value

    def _start_bypass(
        self,
        key: str,
        refresh: Callable[[], Awaitable[V]],
        cache_ok: Callable[[V], bool],
    ) -> asyncio.Task[V]:
        """Create, track, and return a new bypass task for *key*.

        Hoisted out of :meth:`_bypass_refresh`'s loop body rather than
        defined inline there: a done-callback defined INSIDE the loop would
        close over the loop's own ``task`` variable, which ``ruff`` (B023)
        correctly flags as a bug generator — MEASURED, the inline form fails
        with "Function definition does not bind loop variable 'task'" while
        this hoisted form is clean (``All checks passed!``).

        The done-callback's identity check is against the CURRENT tuple
        (not just the key), so a superseded task's callback can never delete
        its successor's entry. It also consumes a non-cancelled task's
        exception (``completed.exception()``) so an orphaned bypass — one
        whose only awaiter was itself cancelled while joining, or was
        abandoned by :meth:`_bypass_refresh` after outliving the bound —
        cannot surface as an asyncio "Task exception was never retrieved"
        error at garbage-collection time.
        """
        task = asyncio.create_task(self._refresh_and_store(key, refresh, cache_ok))
        self._bypass_tasks[key] = (time.monotonic(), task)

        def _forget_if_current(completed: asyncio.Task[V]) -> None:
            current = self._bypass_tasks.get(key)
            if current is not None and current[1] is task:
                del self._bypass_tasks[key]
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(_forget_if_current)
        return task

    async def _bypass_refresh(
        self,
        key: str,
        refresh: Callable[[], Awaitable[V]],
        cache_ok: Callable[[V], bool],
    ) -> V:
        """Run the bypass refresh for *key*, bounding both concurrency and inheritance.

        Every timed-out caller for *key* reaches this method (see
        :meth:`get_or_refresh`'s ``except TimeoutError`` branch). At most one
        refresh is ever TRACKED as "the" bypass for a key at a time: a
        caller that finds a live, WITHIN-BOUND one joins it via
        :func:`asyncio.shield`; a caller that finds none, or one already
        known to have outlived ``_LOCK_ACQUIRE_TIMEOUT_SECONDS``, becomes the
        creator of a new one instead (:meth:`_start_bypass`). Without the
        per-key sharing, a genuinely wedged refresh (one that never returns —
        this task's own 2026-08-27 incident) would let every subsequent
        timed-out caller start its own independent unlocked refresh,
        accumulating without bound at the dashboard's ~2-3s poll rate — over
        a multi-hour wedge that is thousands of parked coroutines, each
        pinning a connection on the shared httpx client and risking pool
        saturation (``httpx.PoolTimeout``) for UNRELATED endpoints —
        degrading keys the wedge never touched, the opposite of what
        ``TestTTLCacheBoundedLockAcquisition``'s
        ``test_a_wedged_key_degrades_only_itself`` proves (key isolation,
        not resource isolation).

        Without ALSO bounding INHERITANCE, a second failure mode opens up
        (reviewer finding, task 4789 review fix): if the shared bypass
        itself never returns — the likely case, since the wedge is
        endpoint/session-level and :func:`first_success` invalidates a
        cached MCP session only on an *exception*, never on a hang — every
        caller after the second would join that SAME dead task forever via
        an unbounded ``asyncio.shield``, as ``TestTTLCacheBoundedBypassInheritance``
        reproduces. Bounding the join means a caller instead re-evaluates
        once its own wait expires (or immediately, if the tracked entry was
        ALREADY over-age when it looked) and — finding the entry it would
        have joined now provably over-age — supersedes it with a fresh
        attempt rather than joining a task nobody can prove will ever
        resolve. Every abandonment is logged exactly once via
        :meth:`_note_lock_bypass`'s ``reason='inherited_bypass'``, whether
        detected instantly or only after actually timing out on the join —
        an abandonment is never silent, matching the class docstring's
        bypass-logging guarantee.

        Two properties worth stating because neither is obvious from the
        loop alone. (i) The loop cannot spin: every path either returns
        (a join succeeded, or a fresh bypass was just created and awaited),
        or re-reads the map after establishing the previous entry is
        over-age — and a re-read after an actual await is required, not
        optional, because another caller may have already installed a
        newer entry while this one was waiting; creating a second new
        bypass on top of theirs would break the "at most one tracked live
        bypass per key" bound. (ii) ``asyncio.wait_for`` cancels only the
        *shield* future it was handed, never the shielded task itself, so
        bounding the join preserves exactly the isolation
        :func:`asyncio.shield` exists for here: one joiner timing out never
        disturbs the shared refresh for every other joiner, or for the
        eventual store.

        ``asyncio.shield`` is kept on the CREATOR's own await too, not just
        joiners': a creator that is itself cancelled (e.g. by its own
        caller's outer budget) must not cancel the refresh other joiners may
        already be riding on.

        A caller promoted to running its own refresh here is in exactly the
        position of a cold caller on a healthy cache — see the class
        docstring's "bounded single-flight" paragraph for the residual that
        leaves: a refresh whose own budgets never fire still never returns,
        for whoever ends up running it.
        """
        while True:
            entry = self._bypass_tasks.get(key)
            if entry is None:
                return await asyncio.shield(self._start_bypass(key, refresh, cache_ok))

            started_at, running = entry
            remaining = (started_at + _LOCK_ACQUIRE_TIMEOUT_SECONDS) - time.monotonic()
            if not running.done() and remaining > 0:
                try:
                    return await asyncio.wait_for(asyncio.shield(running), remaining)
                except TimeoutError:
                    pass  # provably over-age now -- fall through to abandon it

            self._note_lock_bypass(key, refresh, reason='inherited_bypass')
            if self._bypass_tasks.get(key) is entry:
                return await asyncio.shield(self._start_bypass(key, refresh, cache_ok))
            # Superseded by another caller while we were waiting above --
            # rejoin against whatever they installed instead of creating a
            # second new bypass for the same key.
            continue

    def _note_lock_bypass(
        self,
        key: str,
        refresh: Callable[[], Awaitable[V]],
        *,
        reason: str = 'lock_timeout',
    ) -> None:
        """Record one lock-acquisition bypass OR bypass re-arm for *key*.

        Mirrors :func:`log_fanout_failure`'s policy: WARNING for the first
        event of a streak (and every ``_LOCK_BYPASS_REWARN_EVERY``-th
        thereafter), DEBUG for the repeats — so a sustained wedge still
        leaves a trace without flooding the journal. Both call sites share
        ONE streak counter per key, so the throttle covers the whole
        incident rather than resetting at whichever cause happens next.

        *reason* names what actually timed out, since the two are NOT the
        same event: ``'lock_timeout'`` (default) is logged from
        :meth:`get_or_refresh` when the ORIGINAL per-key lock acquisition
        times out. ``'inherited_bypass'`` is logged from
        :meth:`_bypass_refresh` when a caller finds the shared bypass it
        would otherwise join has ITSELF outlived the bound — the lock did
        not time out again here, the inherited bypass did — and is about to
        supersede it with a fresh attempt.
        """
        streak = self._bypass_streaks.get(key, 0) + 1
        self._bypass_streaks[key] = streak
        name = getattr(refresh, '__qualname__', repr(refresh))
        if reason == 'inherited_bypass':
            opening = (
                'a shared bypass refresh for key %r that this caller '
                'inherited has itself run longer than the bound (%.2fs) '
                'without completing (refresh=%s); superseding it with a '
                'fresh refresh (%d consecutive)'
            )
            repeat = (
                'inherited bypass refresh for key %r has itself run longer '
                'than the bound (%.2fs) without completing (refresh=%s); '
                'superseding it (%d consecutive)'
            )
        else:
            opening = (
                'lock acquisition for key %r timed out after %.2fs '
                '(refresh=%s); a prior refresh for this key has been running '
                'longer than the bound, so single-flight is being skipped '
                'and this refresh is running WITHOUT the lock '
                '(%d consecutive)'
            )
            repeat = (
                'lock acquisition for key %r timed out after %.2fs '
                '(refresh=%s), running without the lock (%d consecutive)'
            )
        if streak == 1 or streak % _LOCK_BYPASS_REWARN_EVERY == 0:
            logger.warning(opening, key, _LOCK_ACQUIRE_TIMEOUT_SECONDS, name, streak)
        else:
            logger.debug(repeat, key, _LOCK_ACQUIRE_TIMEOUT_SECONDS, name, streak)

    def _note_lock_acquired(self, key: str) -> None:
        """Close an open bypass streak for *key*, logging recovery.

        Emits at WARNING — the same level as the streak's opening line — so
        the incident has a visible closing bracket at the operator's default
        level. Bounded by construction: fires only when a streak was
        actually open, i.e. at most once per opening WARNING. Called on the
        NORMAL acquisition path only — a key that self-heals via a bypassed
        store is instead served from the lock-free warm path on its next
        call, so its recovery line appears on its next COLD miss rather than
        immediately; the bracket is still bounded and still closes.
        """
        streak = self._bypass_streaks.pop(key, 0)
        if streak:
            logger.warning(
                'lock for key %r recovered after %d consecutive bypass(es)',
                key, streak,
            )

    async def get_or_refresh(
        self,
        key: str,
        refresh: Callable[[], Awaitable[V]],
        *,
        cache_ok: Callable[[V], bool] = lambda v: True,
    ) -> V:
        """Return a fresh cached value for *key*, refreshing at most once.

        Fast path: a warm entry is returned with no lock. Otherwise lock
        acquisition is BOUNDED to ``_LOCK_ACQUIRE_TIMEOUT_SECONDS`` (task
        4789). On a NORMAL (within-bound) acquisition, freshness is re-checked
        (another waiter may have just filled it), and — if still cold —
        :meth:`_refresh_and_store` runs ``refresh()`` and stores the result
        iff ``cache_ok(value)``.

        On a TIMED-OUT acquisition, freshness is re-checked FIRST — mirroring
        the post-lock double-check on the normal path, because the timeout
        window is exactly when another caller can have filled the entry, and
        a bypass that skipped this cheap check would spend a duplicate MCP
        round trip (and clobber a newer value with an older one) for no
        benefit — and only if still cold does ``refresh()`` run, WITHOUT the
        lock, rather than raising: a caller that could never previously see a
        ``TimeoutError`` from this method still cannot. That unlocked refresh
        is itself bounded — see :meth:`_bypass_refresh` for both bounds: at
        most one TRACKED live bypass per key at any instant (not one per
        timed-out caller), and INHERITANCE of that bypass is itself bounded
        to ``_LOCK_ACQUIRE_TIMEOUT_SECONDS`` — a caller never waits on one
        already known to have outlived it, and is instead promoted to
        running its own refresh. The bypassed result is stored under the
        same ``cache_ok`` rule as the locked path, which is what lets a
        wedged key self-heal for every later caller — during the 2026-08-27
        incident nothing was ever stored for the wedged key, which is
        exactly why every later caller kept queueing on a latch that would
        only release at process exit. See the class docstring's
        "bounded single-flight" paragraph for the accepted trade.

        Every bypass is logged under the module's transition-only policy —
        WARNING on the first bypass of a streak and every
        ``_LOCK_BYPASS_REWARN_EVERY``-th thereafter, DEBUG in between,
        WARNING again on recovery (see :meth:`_note_lock_bypass` /
        :meth:`_note_lock_acquired`, mirroring :func:`log_fanout_failure` /
        :func:`note_fanout_success`) — so a wedge is never silent even though
        no caller ever sees a raised exception for it.

        Both the locked cold path and the bypass path sweep expired entries
        (:meth:`_evict_expired`, via :meth:`_refresh_and_store`). That is
        deliberately the ONLY sweep site: a cold miss is exactly when a new
        key can be minted, so reclamation runs at the same rate as growth and
        a hot warm-path hit stays lock-free and O(1). The sweep is O(store
        size) but happens only when an ``await refresh()`` — a network round
        trip — is about to run, which dwarfs it. Running it on the bypass
        path too matters because a bypass is the ONLY traffic a wedged key
        sees — skipping the sweep there would create a sweep-starved regime
        during exactly the outage the eviction work (task 3857) was added
        for.
        """
        fresh = self.get_fresh(key)
        if fresh is not None:
            return fresh

        lock = self._locks.setdefault(key, asyncio.Lock())
        try:
            await asyncio.wait_for(lock.acquire(), _LOCK_ACQUIRE_TIMEOUT_SECONDS)
        except TimeoutError:
            # Report the bypass BEFORE the freshness re-check below: the
            # wedge outliving the bound is the thing being reported, not
            # whether this particular caller happens to save a round trip,
            # so a bypass is logged even when the re-check finds a value.
            self._note_lock_bypass(key, refresh)
            # Freshness first, refresh second — never reordered. The timeout
            # window is exactly when another caller can have filled the
            # entry, so a bypass that skipped this cheap check would spend a
            # duplicate MCP round trip (and clobber a newer value with an
            # older one) for no benefit.
            fresh = self.get_fresh(key)
            if fresh is not None:
                return fresh
            return await self._bypass_refresh(key, refresh, cache_ok)
        else:
            self._note_lock_acquired(key)
            try:
                fresh = self.get_fresh(key)
                if fresh is not None:
                    return fresh
                return await self._refresh_and_store(key, refresh, cache_ok)
            finally:
                lock.release()

    def clear(self) -> None:
        """Reset the store, all per-key locks, and open bypass streaks (test/admin hook).

        Does NOT cancel an in-flight bypass task — it simply stops tracking
        it, so a bypass requested after ``clear()`` starts a fresh one rather
        than joining the orphaned one. The orphaned task keeps running to
        completion in the background regardless (it may still store into a
        since-cleared cache), the same pre-existing race ``clear()`` already
        has with any other in-flight refresh.
        """
        self._store.clear()
        self._locks.clear()
        self._bypass_streaks.clear()
        self._bypass_tasks.clear()
