"""Reclaim shared-client pool connections that httpcore itself cannot (task 5319).

THE INVARIANT THIS MODULE ENFORCES. Every connection in the shared client's
pool is either owned by a live pool request, or idle/closed and therefore
reclaimable by httpcore's own sweep. A connection that is NEITHER is an
ORPHAN: nothing will ever complete it, and nothing will ever remove it. It
holds a pool slot for the life of the process, and holds its socket open —
which is CLOSE-WAIT once the peer FINs.

WHY HTTPCORE CANNOT SEE IT.
``httpcore/_async/connection_pool.py::AsyncConnectionPool._assign_requests_to_connections``
is the ONLY place a connection is removed from the pool, and it reclaims on
exactly three predicates: ``is_closed()``, ``has_expired()``, or
``is_idle()``-and-surplus. Against
``httpcore/_async/http11.py::AsyncHTTP11Connection``:

  * ``is_closed()``   is ``_state == CLOSED``
  * ``is_idle()``     is ``_state == IDLE``
  * ``has_expired()`` is ``(_expire_at is not None and now > _expire_at)`` OR
    ``(_state == IDLE and the socket is readable)``

``_expire_at`` is set to ``None`` on entry to ACTIVE and re-armed only in
``_response_closed()``. So a connection parked in **NEW or ACTIVE** satisfies
none of the three, while ``is_available()`` requires IDLE — simultaneously
unusable and unreclaimable.

HOW ONE GETS PARKED THERE. ``AsyncHTTP11Connection.handle_async_request`` sets
``_state = ACTIVE`` OUTSIDE its ``try:``, and the ``_connect_failed`` flag that
``httpcore/_async/connection.py::AsyncHTTPConnection.has_expired`` consults is
read only while ``self._connection is None`` — dead weight once ``_connect()``
has returned. A cancellation delivered in that window, or inside the
shielded ``_response_closed()`` cleanup, leaves the connection parked. The
dashboard cancels in-flight requests as a matter of routine design (the
``active_tasks`` per-project and total budgets, ``task_runtime``'s probe
budget, ``mcp_fanout``'s lock-acquire timeout, abandoned bypass refreshes), so
every one of those is a draw against this race.

RELATIONSHIP TO TASK 3857. 3857's refutation stands, and this extends it
rather than reopening it: 3857 measured IDLE connections, for which
``has_expired()``'s ``server_disconnected`` term does fire and "reaped on the
next pool use" is exactly right. That term is gated on IDLE, so the NEW/ACTIVE
case is invisible to it by construction. Nothing here counts sockets or
CLOSE-WAIT entries — the measure 3857 correctly rejected. The reading is pool
occupancy, taken from the pool's own bookkeeping.

THE PREDICATE, and why it is sound without a timestamp. ``pool._requests``
holds one ``AsyncPoolRequest`` per live request, each carrying ``.connection``,
and every exit path removes the request before its connection is abandoned. So
a pooled connection owned by no live request, and neither idle nor closed, is
provably unreachable. A genuinely in-flight request is never an orphan,
because its ``AsyncPoolRequest`` is still in ``_requests``. No age heuristic,
and so no window in which a slow-but-healthy request looks dead.

VERIFIED AGAINST httpx 0.28.1 / httpcore 1.0.9 — the same discipline, and the
same reason, as ``app.py::_HTTP_KEEPALIVE_EXPIRY_SECONDS``: this module reads
attributes those releases do not promise to keep. RE-CHECK AFTER AN UPGRADE.
:func:`_resolve_pool` is the single place that touches them, and it degrades
loudly rather than silently if they move.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import cast

import httpcore
import httpx

logger = logging.getLogger(__name__)

# The pool attributes and methods this module reads or calls. Named once, here,
# so the shape guard and the failure message it produces cannot drift apart.
_REQUIRED_POOL_ATTRIBUTES = (
    '_connections',
    '_requests',
    '_max_connections',
    '_assign_requests_to_connections',
    '_close_connections',
)


class _LatchedWarning:
    """A WARNING that announces itself once, then repeats only at DEBUG.

    LOUD ONCE, THEN QUIET, and both halves matter. Silence is the worst outcome
    available to this module — a reaper that has quietly become a no-op while
    still sitting in the tree looking like it handles the problem. But a
    WARNING on every sweep is its own kind of silence: on a 60s loop that is
    ~1400 identical lines a day, and the line that opened the diagnosis is
    buried under its own repeats. Demoted, never discarded: the repeat stays at
    DEBUG for whoever turns it on.

    ONE TYPE FOR BOTH of this module's throttled reports, because the
    throttling rule is one rule and stating it twice is how two spellings
    drift apart. The reports differ only in when they RE-ARM, which is
    :meth:`clear`'s caller's business rather than this class's: the shape guard
    never re-arms on its own (a relocated attribute stays relocated), while the
    saturation alarm re-arms as occupancy falls back under the mark.
    """

    def __init__(self) -> None:
        self._fired = False

    def clear(self) -> None:
        """Re-arm, so the next :meth:`fire` is loud again."""
        self._fired = False

    def fire(self, message: str, *args: object) -> None:
        """Log *message* — at WARNING the first time, at DEBUG after that."""
        if self._fired:
            logger.debug(message, *args)
            return
        self._fired = True
        logger.warning(message, *args)


_shape_guard = _LatchedWarning()


def reset_shape_guard() -> None:
    """Re-arm the once-per-process shape WARNING. For tests.

    Part of this module's interface rather than a private global for tests to
    poke — the same role ``memory.reset_sessions()`` plays for the session
    cache. The latch is state this module owns, so clearing it is this
    module's operation to offer, and a test asserting on the WARNING can do so
    without depending on whether some other test ran first.
    """
    _shape_guard.clear()


def _report_unresolved(path: str) -> None:
    """Announce a pool attribute this module can no longer find.

    Latched, for the reason :class:`_LatchedWarning` gives: this is the one
    line that turns "the reaper stopped working" from a 32-hour mystery into a
    grep, so it has to survive weeks of journal, which means it has to be rare.
    """
    _shape_guard.fire(
        'httpx connection pool is not where this module expects it: cannot resolve '
        'client.%s. The orphan reaper is INERT until this is repaired — see '
        'dashboard/src/dashboard/http_pool.py for the attributes it reads and the '
        'httpx/httpcore versions they were verified against.',
        path,
    )


@dataclass(frozen=True)
class PoolCensus:
    """A reading of the shared client's pool occupancy.

    ``orphaned`` is the count this module exists for; ``total`` against
    ``max_connections`` is what predicts a wedge, since a pool at its ceiling
    converts ordinary queueing into ``httpx.PoolTimeout``.
    """

    total: int
    idle: int
    orphaned: int
    max_connections: int


# Occupancy at which the sweep starts reporting, as a fraction of
# `max_connections`.
#
# BELOW THE CEILING, DELIBERATELY. Degradation begins well before the pool is
# full: the 2026-09-11 replication measured `/api/v2/dashboard/tasks` already
# burning its entire 20s budget at 83 of 100 connections. An alarm armed AT the
# ceiling would therefore fire only once the endpoint had been unusable for
# some time, which is the same too-late signal that left the original incident
# invisible for 32 hours.
POOL_HIGH_WATER_FRACTION = 0.8

_saturation_alarm = _LatchedWarning()


def reset_saturation_guard() -> None:
    """Re-arm the saturation alarm. For tests.

    Companion to :func:`reset_shape_guard`, and separate from it on purpose:
    the two latches answer different questions ("can this module still read the
    pool?" and "is the pool filling up?"), so a test that wants one in a known
    state should not have to disturb the other.
    """
    _saturation_alarm.clear()


def _report_occupancy(reading: PoolCensus) -> None:
    """Report pool occupancy at or above :data:`POOL_HIGH_WATER_FRACTION`.

    RE-ARMS ON THE WAY BACK DOWN, which is the only thing distinguishing this
    report from the shape guard's. A pool that saturates, recovers, and
    saturates again has had two incidents, and the second matters at least as
    much as the first; a latch that only ever fell one way would report the
    first episode a process saw and nothing after it.

    The whole census goes in the line, not just the ratio: "80 of 100" does not
    say whether those are healthy in-flight requests or orphans this module
    failed to reclaim, and that is the first question an operator asks.
    """
    if reading.total < reading.max_connections * POOL_HIGH_WATER_FRACTION:
        _saturation_alarm.clear()
        return
    _saturation_alarm.fire(
        'httpx connection pool at or above its high-water mark (%.0f%% of capacity): '
        '%s. Sustained saturation ends in httpx.PoolTimeout, which the dashboard '
        'renders as an "offline" pill on a healthy orchestrator.',
        POOL_HIGH_WATER_FRACTION * 100,
        reading,
    )


def _resolve_pool(client: httpx.AsyncClient) -> httpcore.AsyncConnectionPool | None:
    """Return *client*'s underlying httpcore pool, or ``None`` if unrecognised.

    THE ONLY PLACE private attributes are reached. Both the walk
    (``client._transport`` -> ``_pool``) and the shape check live here, so an
    httpx release that relocates either is a single-function repair rather
    than a hunt — and so the rest of the module stays typed, with no
    ``# type: ignore`` scattered across it.

    The shape check is what makes the :func:`cast` honest: every attribute the
    module goes on to read is confirmed present before the pool is handed
    back, so the cast asserts something just measured rather than something
    assumed.
    """
    transport = getattr(client, '_transport', None)
    if transport is None:
        _report_unresolved('_transport')
        return None
    pool = getattr(transport, '_pool', None)
    if pool is None:
        _report_unresolved('_transport._pool')
        return None
    for attribute in _REQUIRED_POOL_ATTRIBUTES:
        if not hasattr(pool, attribute):
            _report_unresolved(f'_transport._pool.{attribute}')
            return None
    return cast(httpcore.AsyncConnectionPool, pool)


def _orphaned(
    pool: httpcore.AsyncConnectionPool,
) -> list[httpcore.AsyncConnectionInterface]:
    """Pooled connections no live request owns and no httpcore branch reclaims.

    SYNCHRONOUS BY CONSTRUCTION — no ``await`` anywhere in it. That makes the
    reading atomic with respect to the event loop: no request can be assigned
    to one of these connections between observing ``_requests`` and returning,
    so a caller acting on the result is acting on a state that still holds.
    """
    owned = {request.connection for request in pool._requests}
    return [
        connection
        for connection in pool._connections
        if connection not in owned
        and not connection.is_idle()
        and not connection.is_closed()
    ]


def census(client: httpx.AsyncClient) -> PoolCensus | None:
    """Read *client*'s pool occupancy, or ``None`` if its shape is unrecognised.

    ``None`` rather than a zeroed census: an unreadable pool and an empty one
    are different facts, and reporting the first as the second would make a
    broken guard indistinguishable from a healthy system.
    """
    pool = _resolve_pool(client)
    if pool is None:
        return None
    connections = list(pool._connections)
    return PoolCensus(
        total=len(connections),
        idle=sum(1 for connection in connections if connection.is_idle()),
        orphaned=len(_orphaned(pool)),
        max_connections=pool._max_connections,
    )


async def reap_orphaned_connections(client: httpx.AsyncClient) -> int:
    """Close and unpool *client*'s orphaned connections. Returns how many closed.

    FOUR PHASES, AND THE ORDER IS LOAD-BEARING.

    1. SYNCHRONOUS — resolve the pool and decide the doomed set. There is no
       ``await`` between reading ``_requests`` and fixing that set, so httpcore
       cannot assign a queued request to one of these connections in between.
       Without that, the reaper would race live traffic rather than avoid it.

    2. AWAIT — close each doomed connection. ``Exception`` is suppressed per
       connection so one bad close cannot strand the rest; ``BaseException``
       (notably ``CancelledError``) is deliberately allowed to propagate,
       because the only caller that delivers one is shutdown, where the
       process is going away and every remaining connection is either already
       closed or still pooled — both safe.

    3. SYNCHRONOUS — drop the doomed connections from ``_connections``. Phase 4
       below would remove most of them anyway, through httpcore's own
       ``is_closed()`` branch, and that overlap is deliberate rather than
       missed: this phase states the guarantee DIRECTLY — a sweep unpools what
       it doomed — instead of inheriting it from a private reclaim branch the
       shape guard cannot check and a future httpcore is free to narrow. It
       also covers the one case that branch cannot see, a connection whose
       ``aclose()`` was a no-op because it had no transport yet.

    4. HAND THE FREED SLOTS TO WHOEVER IS ALREADY WAITING, and nothing else
       will. ``_assign_requests_to_connections`` is the ONLY place httpcore
       gives a queued ``AsyncPoolRequest`` the connection its
       ``wait_for_connection`` is blocked on, and httpcore runs it only as a
       request enters or leaves the pool. A sweep is neither. So phase 3 alone
       leaves the wedge this module exists to end fully intact in its worst
       case — a pool at ``max_connections``, every slot an orphan, N requests
       parked in ``wait_for_connection``: the slots are freed and the waiters
       are still never dispatched, dying on their own pool timeouts seconds
       after the space they needed appeared. MEASURED on a ``max_connections=1``
       harness: without this phase the queued request burned its entire 2.0s
       pool budget and raised ``httpx.PoolTimeout``; with it the same request
       completed in ~1ms with a 200.

    CLOSING BEFORE REMOVING is what makes a cancellation mid-sweep harmless.
    An already-closed connection still in ``_connections`` satisfies httpcore's
    own ``is_closed()`` branch and is reclaimed by the pool on its next use, so
    the worst case is a delayed slot. Remove-then-close would instead leak the
    file descriptor outright, with nothing left holding a reference to close.

    The ``ValueError`` suppressed in phase 3 is that same benign race seen from
    the other side: across phase 2's awaits, a request completing elsewhere can
    run ``_assign_requests_to_connections``, which removes a connection this
    sweep has just closed. Already gone is the outcome this phase wanted.
    """
    pool = _resolve_pool(client)
    if pool is None:
        return 0

    doomed = _orphaned(pool)
    if not doomed:
        # The overwhelmingly common sweep. Returning here keeps it a pure read:
        # phase 4 below is httpcore's own pool-management pass, and running that
        # every 60s on an untouched pool would make this function's effect
        # something other than what its name says.
        return 0

    closed = 0
    for connection in doomed:
        with contextlib.suppress(Exception):
            await connection.aclose()
            closed += 1

    # Every doomed connection is unpooled, including any whose close raised:
    # ``aclose`` sets CLOSED before releasing the stream, so such a connection
    # is closed as far as the pool is concerned and holding its slot would help
    # nobody. Hence ``closed`` can read lower than the number removed — it
    # counts clean closes, which is the number worth seeing in the log.
    for connection in doomed:
        with contextlib.suppress(ValueError):
            pool._connections.remove(connection)

    # Phase 4, in httpcore's own pairing: assign synchronously, then close what
    # the assignment displaced. ``_close_connections`` shields its awaits from
    # cancellation, which matters because those connections are already out of
    # ``_connections`` and nothing else would ever close them.
    closing = pool._assign_requests_to_connections()
    await pool._close_connections(closing)

    return closed


# How often the background sweep runs, and the arithmetic that sized it.
#
# THE LEAK RATE IS MEASURED, not assumed: ~1.9-2.4 orphans/hour in production,
# against an `app.py::_HTTP_MIN_CONNECTIONS` ceiling of 100. The worst observed
# burst was 6 at once, when an orchestrator restart FINned every ESTAB
# connection on one escalation port.
#
# At 60s, expected steady-state occupancy from orphans is rate x interval =
# 2.4/hour x 1/60 hour = ~0.04 connections — i.e. the pool is essentially never
# holding one — and the worst burst is cleared inside a single tick, four
# orders of magnitude below the ceiling either way.
#
# Sweeping this often costs nothing worth counting: a sweep is a synchronous
# in-memory scan of at most `max_connections` objects with no I/O, and on the
# overwhelmingly common empty result it logs nothing at all.
#
# THE CADENCE LIVES HERE, not in `app.py`. It is this module's concern — it
# follows from this module's predicate and the leak rate that predicate
# addresses — and `app.py` is ~2500 lines already.
REAP_INTERVAL_SECONDS = 60.0


async def reaper_loop(
    client: httpx.AsyncClient, interval: float = REAP_INTERVAL_SECONDS
) -> None:
    """Sweep *client*'s pool for orphans forever, at *interval*.

    Shaped like ``app.py``'s ``_burndown_loop`` and ``_metrics_loop``, and for
    the same reason: a background task that dies on its first bad cycle is a
    background task that silently stopped doing its job. Each cycle's
    ``Exception`` is logged with its traceback and the next tick still runs.

    ``CancelledError`` is NOT caught — it is a ``BaseException``, so the
    ``except Exception`` below lets it through by construction. That is what
    lets ``lifespan``'s ``task.cancel()`` then ``await task`` terminate rather
    than hang.

    Sleeps BEFORE its first sweep: a pool that has served no requests yet can
    hold no orphans, so a sweep at startup could only ever find nothing.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            reaped = await reap_orphaned_connections(client)
            # One reading per sweep, taken AFTER the reap and shared by both
            # reports below: post-reap occupancy is the number that actually
            # predicts a wedge, and two separate readings could disagree.
            reading = census(client)
            if reaped:
                # Loud on every reap, and it can afford to be: reaps run at
                # roughly 2/hour, so one line each is a usable history of the
                # defect rather than a flood. The census makes that one line
                # answer the follow-up question too — whether the pool is
                # actually healthy now, or filling faster than this sweep
                # empties it.
                logger.warning(
                    'Reaped %d orphaned pool connection(s); pool now %s',
                    reaped,
                    reading,
                )
            if reading is not None:
                _report_occupancy(reading)
        except Exception:
            logger.warning('Orphan reaper sweep failed', exc_info=True)
