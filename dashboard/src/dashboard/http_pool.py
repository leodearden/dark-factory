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

from dataclasses import dataclass
from typing import cast

import httpcore
import httpx

# The pool attributes this module reads. Named once, here, so the shape guard
# and the failure message it produces cannot drift apart.
_REQUIRED_POOL_ATTRIBUTES = ('_connections', '_requests', '_max_connections')


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
        return None
    pool = getattr(transport, '_pool', None)
    if pool is None:
        return None
    for attribute in _REQUIRED_POOL_ATTRIBUTES:
        if not hasattr(pool, attribute):
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
