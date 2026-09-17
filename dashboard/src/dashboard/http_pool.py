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

# The pool attributes this module reads. Named once, here, so the shape guard
# and the failure message it produces cannot drift apart.
_REQUIRED_POOL_ATTRIBUTES = ('_connections', '_requests', '_max_connections')

# Has the shape guard already announced itself in this process? See
# :func:`_report_unresolved` for why announcing once is the requirement.
_shape_guard_warned = False


def reset_shape_guard() -> None:
    """Re-arm the once-per-process shape WARNING. For tests.

    Part of this module's interface rather than a private global for tests to
    poke — the same role ``memory.reset_sessions()`` plays for the session
    cache. The latch is state this module owns, so clearing it is this
    module's operation to offer, and a test asserting on the WARNING can do so
    without depending on whether some other test ran first.
    """
    global _shape_guard_warned
    _shape_guard_warned = False


def _report_unresolved(path: str) -> None:
    """Announce a pool attribute this module can no longer find.

    LOUD ONCE, THEN QUIET, and both halves matter. Silence would be the worst
    outcome available: the reaper would become a permanent no-op while a
    module in the tree still looked like it was handling the problem — the
    original incident again, now harder to find. But a WARNING on every sweep
    is its own kind of silence: on the reaper's loop that is ~1400 identical
    lines a day, and the line that opened the diagnosis is buried under its
    own repeats. Demoted, never discarded: the repeat stays at DEBUG.
    """
    global _shape_guard_warned
    message = (
        'httpx connection pool is not where this module expects it: cannot resolve '
        'client.%s. The orphan reaper is INERT until this is repaired — see '
        'dashboard/src/dashboard/http_pool.py for the attributes it reads and the '
        'httpx/httpcore versions they were verified against.'
    )
    if _shape_guard_warned:
        logger.debug(message, path)
        return
    _shape_guard_warned = True
    logger.warning(message, path)


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

    THREE PHASES, AND THE ORDER IS LOAD-BEARING.

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

    3. SYNCHRONOUS — drop the now-closed connections from ``_connections``, so
       the slot is free immediately instead of at the pool's next use. This is
       the half that actually ends the wedge: a connection marked closed but
       still listed keeps counting against ``max_connections``.

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
                    census(client),
                )
        except Exception:
            logger.warning('Orphan reaper sweep failed', exc_info=True)
