"""Tests for ``dashboard.http_pool`` — reclaiming the connections httpcore cannot.

The defect these tests reproduce is stated once, where the fix lives:
``dashboard/src/dashboard/http_pool.py``'s module docstring. In one line: a
connection parked in NEW or ACTIVE state satisfies none of the three
predicates httpcore reclaims on, so it holds a pool slot and a socket for the
life of the process. The dashboard mints those routinely because it cancels
in-flight requests by design (``active_tasks``'s per-project and total
budgets, ``task_runtime``'s probe budget, ``mcp_fanout``'s lock-acquire
timeout).

WHY THE REPRODUCTION IS DETERMINISTIC. The trigger is a cancellation landing
inside a specific window of httpcore's request path, so a timing-based
reproduction (real sockets, real deadlines) reproduces it only probabilistically
and would be flaky in CI. Instead the pool here runs over a fake
``AsyncNetworkBackend`` — no file descriptors, no wall clock — and the
cancellation is delivered after an exact number of event-loop steps. The test
sweeps that number rather than pinning it: see ``_MAX_CANCEL_STEPS``.

RELATIONSHIP TO TASK 3857, which this extends rather than contradicts. 3857
measured IDLE connections and correctly established that the pool reaps them
on its next use — ``has_expired()``'s ``server_disconnected`` term fires for
those. That term is gated on IDLE, so it cannot see the connections here.
``TestHttpcoreCannotReclaimTheOrphan`` pins that distinction directly.

Nothing here counts sockets or asserts a CLOSE-WAIT total — the measure 3857
correctly rejected. Every assertion is about the pool's own occupancy.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Any

import httpcore
import httpx
import pytest

from dashboard import http_pool

# The shipped pool's shape, for a small install where
# ``app.py::_HTTP_MIN_CONNECTIONS`` is the binding term. Copied as literals
# rather than derived from ``_build_http_limits``: this module is testing what
# httpcore does with a pool of this shape, not what the dashboard sizes it to
# (``tests/test_app_http_limits.py`` owns the sizing).
_MAX_CONNECTIONS = 100
_MAX_KEEPALIVE_CONNECTIONS = 20
_KEEPALIVE_EXPIRY = 4.0

_URL = 'http://svc.local/mcp'
_CANNED_RESPONSE = b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}'

# How many event-loop positions the cancellation sweep explores.
#
# SWEEPING, NOT PINNING, and the reason is measured rather than defensive: the
# orphan was minted at k=3 on 2026-09-15 and at k=2 on 2026-09-17 against the
# SAME httpx 0.28.1 / httpcore 1.0.9 install. k is a property of the
# dependency's yield schedule, not of anything in this repo, so a hard-coded
# constant would go red on a dependency bump that changed nothing about the
# defect. 32 is comfortable headroom over both observations.
_MAX_CANCEL_STEPS = 32


class FakeStream(httpcore.AsyncNetworkStream):
    """An in-memory stand-in for a socket: no file descriptor, no wall clock.

    Buffers :data:`_CANNED_RESPONSE` when the request is written and drains it
    on read, so a full round-trip completes without the network. That is what
    lets the cancellation sweep below be deterministic rather than a race.

    ``write`` ASSIGNS rather than appends. httpcore writes request headers and
    body as separate calls, and a connection is written to again on reuse;
    appending would leave a stale response in the buffer for the next request
    to mis-parse.
    """

    def __init__(self) -> None:
        self.pending = b''
        self.closed = False

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self.pending = _CANNED_RESPONSE

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        chunk, self.pending = self.pending[:max_bytes], self.pending[max_bytes:]
        return chunk

    async def aclose(self) -> None:
        self.closed = True

    def get_extra_info(self, info: str) -> Any:
        # None for every key, 'is_readable' included. has_expired()'s
        # server_disconnected term is therefore never satisfied here — which is
        # exactly the condition under which an orphan is unreclaimable, and the
        # one task 3857's IDLE measurements could not observe.
        return None


class FakeBackend(httpcore.AsyncNetworkBackend):
    """Hands out :class:`FakeStream`s and retains every one it handed out.

    Retention is what lets a test assert a stream was CLOSED rather than
    merely dropped — the difference between releasing the fd and leaking it.
    """

    def __init__(self) -> None:
        self.streams: list[FakeStream] = []

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        stream = FakeStream()
        self.streams.append(stream)
        return stream


@dataclass(frozen=True)
class _Harness:
    """One client, and the two objects a test needs to observe it."""

    client: httpx.AsyncClient
    backend: FakeBackend
    pool: httpcore.AsyncConnectionPool


def _build_harness() -> _Harness:
    """An ``httpx.AsyncClient`` over a pool with no sockets behind it."""
    backend = FakeBackend()
    pool = httpcore.AsyncConnectionPool(
        max_connections=_MAX_CONNECTIONS,
        max_keepalive_connections=_MAX_KEEPALIVE_CONNECTIONS,
        keepalive_expiry=_KEEPALIVE_EXPIRY,
        network_backend=backend,
    )
    transport = httpx.AsyncHTTPTransport()
    transport._pool = pool
    return _Harness(httpx.AsyncClient(transport=transport), backend, pool)


def _unreclaimable(pool: httpcore.AsyncConnectionPool) -> list[Any]:
    """Pool connections httpcore's own sweep will never remove.

    Spelled out here, against httpcore's interface, rather than borrowed from
    ``http_pool``: this is the test's independent statement of the condition,
    so a bug in the module under test cannot make its own assertion vacuous.
    """
    return [c for c in pool._connections if not c.is_idle() and not c.is_closed()]


@contextlib.asynccontextmanager
async def _orphan_sweep() -> AsyncIterator[_Harness | None]:
    """Cancel an in-flight POST at each of :data:`_MAX_CANCEL_STEPS` positions.

    Yields the first harness whose pool was left holding an orphan, or ``None``
    if no position produced one. A FRESH client per position, so a connection
    left behind by an earlier position cannot be mistaken for this one's.
    """
    harnesses: list[_Harness] = []
    try:
        minted: _Harness | None = None
        for steps in range(_MAX_CANCEL_STEPS):
            harness = _build_harness()
            harnesses.append(harness)
            task = asyncio.create_task(harness.client.post(_URL, content=b'{}'))
            for _ in range(steps):
                await asyncio.sleep(0)
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
            if _unreclaimable(harness.pool):
                minted = harness
                break
        yield minted
    finally:
        for harness in harnesses:
            with contextlib.suppress(Exception):
                await harness.client.aclose()


class TestCancellationOrphansAConnection:
    """The reproduction: a cancelled request can park a connection forever."""

    async def test_sweep_mints_a_connection_in_a_non_reclaimable_state(self) -> None:
        async with _orphan_sweep() as minted:
            assert minted is not None, (
                f'no cancellation position in range({_MAX_CANCEL_STEPS}) left a '
                'connection in a non-reclaimable state. Either httpcore now cleans '
                'up on cancel — in which case dashboard/src/dashboard/http_pool.py '
                'can be retired — or its yield schedule moved past this sweep.'
            )
            census = http_pool.census(minted.client)
            assert census is not None, 'census could not resolve a pool it just built'
            assert census.orphaned == 1
            assert census.max_connections == _MAX_CONNECTIONS

            # The state is the whole point: NEW or ACTIVE is simultaneously
            # unusable (is_available() requires IDLE) and unreclaimable.
            [connection] = _unreclaimable(minted.pool)
            state = connection.info()
            assert 'IDLE' not in state and 'CLOSED' not in state, state


class TestHttpcoreCannotReclaimTheOrphan:
    """CHARACTERIZATION of httpcore, and the exact boundary of task 3857.

    3857's "fully reaped on the next pool use" is correct — for IDLE
    connections. This one is not IDLE, so no reaping branch in
    ``AsyncConnectionPool._assign_requests_to_connections`` can see it, and
    using the pool does not help.
    """

    async def test_three_further_requests_do_not_reclaim_it(self) -> None:
        async with _orphan_sweep() as minted:
            if minted is None:
                pytest.skip(
                    'no orphan was minted, so there is nothing for httpcore to fail '
                    'to reclaim; TestCancellationOrphansAConnection is the test that '
                    'reports that change loudly'
                )
            before = http_pool.census(minted.client)
            assert before is not None

            for _ in range(3):
                response = await minted.client.post(_URL, content=b'{}')
                assert response.status_code == 200

            after = http_pool.census(minted.client)
            assert after is not None
            assert after.orphaned == before.orphaned, (
                'httpcore reclaimed a non-IDLE connection on pool use — the premise '
                'behind dashboard/src/dashboard/http_pool.py no longer holds'
            )
