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
import logging
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpcore
import httpx
import pytest
from _dashboard_helpers import apply_isolated_env
from fastapi import FastAPI
from starlette.datastructures import State

from dashboard import http_pool
from dashboard.app import lifespan

# The shipped pool's shape, for a small install where
# ``app.py::_HTTP_MIN_CONNECTIONS`` is the binding term. Copied as literals
# rather than derived from ``_build_http_limits``: this module is testing what
# httpcore does with a pool of this shape, not what the dashboard sizes it to
# (``tests/test_app_http_limits.py`` owns the sizing).
_MAX_CONNECTIONS = 100
_MAX_KEEPALIVE_CONNECTIONS = 20
_KEEPALIVE_EXPIRY = 4.0

_LOGGER_NAME = 'dashboard.http_pool'

# Captured before any test patches asyncio.sleep, so the fake cadence below can
# still yield control without recursing into itself.
_REAL_SLEEP = asyncio.sleep
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

# Bound on :func:`_step_until`, a different dimension from the sweep above:
# how long a request may take to reach a named point, rather than where a
# cancellation is delivered. Generous, because overshooting costs one cheap
# no-op iteration while undershooting is a false failure.
_MAX_SETTLE_STEPS = 64

_NO_ORPHAN_MINTED = (
    f'no cancellation position in range({_MAX_CANCEL_STEPS}) left a connection in a '
    'non-reclaimable state. Either httpcore now cleans up on cancel — in which case '
    'dashboard/src/dashboard/http_pool.py can be retired — or its yield schedule '
    'moved past this sweep.'
)


class FakeStream(httpcore.AsyncNetworkStream):
    """An in-memory stand-in for a socket: no file descriptor, no wall clock.

    Buffers :data:`_CANNED_RESPONSE` when the request is written and drains it
    on read, so a full round-trip completes without the network. That is what
    lets the cancellation sweep below be deterministic rather than a race.

    ``write`` ASSIGNS rather than appends. httpcore writes request headers and
    body as separate calls, and a connection is written to again on reuse;
    appending would leave a stale response in the buffer for the next request
    to mis-parse.

    ``read_gate`` parks the response read until a test releases it, which is
    how a request is held GENUINELY in flight — the state a reaper must never
    touch. ``reached_read`` is the observable that lets a test wait for that
    point deterministically instead of guessing a number of loop steps.

    ``close_error`` and ``close_gate`` are the same two controls over the CLOSE
    side, set on an individual stream after a test has minted its orphan. They
    exist because ``reap_orphaned_connections``'s two suppressed-failure paths
    are otherwise unreachable: a close that raises is what separates its
    "closed" count from the set it unpools, and a close that suspends is the
    only window in which a concurrent request can unpool a connection the sweep
    is midway through closing.
    """

    def __init__(self, read_gate: asyncio.Event | None = None) -> None:
        self.pending = b''
        self.closed = False
        self.reached_read = False
        self.reached_close = False
        self.close_error: Exception | None = None
        self.close_gate: asyncio.Event | None = None
        self._read_gate = read_gate

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self.pending = _CANNED_RESPONSE

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        self.reached_read = True
        if self._read_gate is not None:
            await self._read_gate.wait()
        chunk, self.pending = self.pending[:max_bytes], self.pending[max_bytes:]
        return chunk

    async def aclose(self) -> None:
        self.reached_close = True
        if self.close_gate is not None:
            await self.close_gate.wait()
        # Raised BEFORE the flag, so ``closed`` keeps meaning "this stream was
        # actually released" — a failed close released nothing.
        if self.close_error is not None:
            raise self.close_error
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

    def __init__(self, read_gate: asyncio.Event | None = None) -> None:
        self.streams: list[FakeStream] = []
        self._read_gate = read_gate

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        stream = FakeStream(self._read_gate)
        self.streams.append(stream)
        return stream


@dataclass(frozen=True)
class _Harness:
    """One client, and the two objects a test needs to observe it."""

    client: httpx.AsyncClient
    backend: FakeBackend
    pool: httpcore.AsyncConnectionPool


def _build_harness(
    read_gate: asyncio.Event | None = None,
    max_connections: int = _MAX_CONNECTIONS,
) -> _Harness:
    """An ``httpx.AsyncClient`` over a pool with no sockets behind it.

    *max_connections* is overridable only so the saturation tests can reach a
    high-water mark in four requests instead of eighty.
    """
    backend = FakeBackend(read_gate)
    pool = httpcore.AsyncConnectionPool(
        max_connections=max_connections,
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


def _queued(pool: httpcore.AsyncConnectionPool) -> list[Any]:
    """Pool requests still waiting to be handed a connection.

    Stated against httpcore's interface for the same reason as
    :func:`_unreclaimable`: a test that borrowed the module's own notion of
    "waiting" could not catch the module getting it wrong.
    """
    return [r for r in pool._requests if r.is_queued()]


async def _step_until(predicate: Callable[[], bool], *, what: str) -> None:
    """Advance the event loop until *predicate* holds.

    Bounded and clock-free, so a test waits for a named condition rather than
    for a guessed number of steps, and reports what it was waiting for if the
    condition never arrives.
    """
    for _ in range(_MAX_SETTLE_STEPS):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError(f'{what} did not happen within {_MAX_SETTLE_STEPS} event-loop steps')


@contextlib.asynccontextmanager
async def _orphan_sweep(
    max_connections: int = _MAX_CONNECTIONS,
) -> AsyncIterator[_Harness | None]:
    """Cancel an in-flight POST at each of :data:`_MAX_CANCEL_STEPS` positions.

    Yields the first harness whose pool was left holding an orphan, or ``None``
    if no position produced one. A FRESH client per position, so a connection
    left behind by an earlier position cannot be mistaken for this one's.

    *max_connections* is forwarded to :func:`_build_harness`, so a test can
    mint its orphan into a pool that the orphan alone fills.
    """
    harnesses: list[_Harness] = []
    try:
        minted: _Harness | None = None
        for steps in range(_MAX_CANCEL_STEPS):
            harness = _build_harness(max_connections=max_connections)
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
            assert minted is not None, _NO_ORPHAN_MINTED
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


class TestReapingReclaimsTheOrphan:
    """The fix: an orphan's pool slot AND its socket are both released."""

    async def test_reap_frees_the_slot_and_closes_the_stream(self) -> None:
        async with _orphan_sweep() as minted:
            assert minted is not None, _NO_ORPHAN_MINTED
            before = http_pool.census(minted.client)
            assert before is not None
            assert before.orphaned == 1
            assert [s for s in minted.backend.streams if not s.closed], (
                'the orphan holds no open stream, so this test would assert nothing'
            )

            reaped = await http_pool.reap_orphaned_connections(minted.client)

            assert reaped == 1
            after = http_pool.census(minted.client)
            assert after is not None
            assert after.orphaned == 0
            # The SLOT, not just the flag: a connection marked closed but left
            # in _connections still counts against max_connections, so the pool
            # would wedge exactly as before.
            assert after.total == before.total - 1
            # The SOCKET, not just the bookkeeping: dropping the reference
            # without closing leaks the fd, which is the CLOSE-WAIT this task
            # exists to end.
            assert all(s.closed for s in minted.backend.streams), (
                'the orphan was forgotten rather than closed'
            )

    async def test_a_second_reap_changes_nothing(self) -> None:
        """IDEMPOTENCE — the reaper runs on a 60s loop, so most sweeps find nothing."""
        async with _orphan_sweep() as minted:
            assert minted is not None, _NO_ORPHAN_MINTED
            assert await http_pool.reap_orphaned_connections(minted.client) == 1
            settled = http_pool.census(minted.client)

            assert await http_pool.reap_orphaned_connections(minted.client) == 0

            assert http_pool.census(minted.client) == settled

    async def test_a_close_that_fails_still_unpools_the_connection(self) -> None:
        """``closed`` counts clean closes; the SET that gets unpooled is larger.

        Both halves are asserted because both are load-bearing and neither is
        visible elsewhere: a bad close must not be counted as a reclaim (the
        reap WARNING would overstate what it achieved), and it must not leave
        the connection pooled either, since httpcore already marked it CLOSED
        and the slot would be held for nothing. Moving ``closed += 1`` outside
        phase 2's suppression passes every other test in this module.
        """
        async with _orphan_sweep() as minted:
            assert minted is not None, _NO_ORPHAN_MINTED
            before = http_pool.census(minted.client)
            assert before is not None
            assert before.orphaned == 1
            [stream] = [s for s in minted.backend.streams if not s.closed]
            stream.close_error = OSError('close(2) failed on the underlying socket')

            reaped = await http_pool.reap_orphaned_connections(minted.client)

            assert reaped == 0, 'a close that raised is not a clean close and must not count'
            after = http_pool.census(minted.client)
            assert after is not None
            assert after.total == before.total - 1, (
                'a connection whose close failed is CLOSED as far as httpcore is '
                'concerned, so leaving it pooled would hold the slot for nothing'
            )
            assert after.orphaned == 0

    async def test_a_concurrent_unpooling_mid_sweep_is_not_an_error(self) -> None:
        """Phase 3's suppressed ``ValueError``, driven by the real race.

        Across phase 2's awaits the connection is already CLOSED, so any other
        request entering the pool runs ``_assign_requests_to_connections`` and
        removes it — and phase 3 then finds it gone. Already-removed is the
        outcome phase 3 wanted, so the sweep must finish normally rather than
        raise into the loop's error handler.
        """
        async with _orphan_sweep() as minted:
            assert minted is not None, _NO_ORPHAN_MINTED
            [stream] = [s for s in minted.backend.streams if not s.closed]
            gate = asyncio.Event()
            stream.close_gate = gate

            sweep = asyncio.create_task(http_pool.reap_orphaned_connections(minted.client))
            try:
                await _step_until(
                    lambda: stream.reached_close,
                    what='the sweep suspending inside the orphan\'s close',
                )

                # A real request, through the real pool: this is what unpools
                # the connection out from under the sweep.
                other = await minted.client.post('http://other.local/mcp', content=b'{}')
                assert other.status_code == 200
                assert minted.pool._connections and all(
                    c is not stream for c in minted.pool._connections
                )

                gate.set()
                assert await sweep == 1
            finally:
                gate.set()
                sweep.cancel()
                with contextlib.suppress(BaseException):
                    await sweep

            settled = http_pool.census(minted.client)
            assert settled is not None
            assert settled.orphaned == 0

    async def test_a_request_queued_behind_a_full_pool_is_dispatched_by_the_sweep(
        self,
    ) -> None:
        """Freeing the slot is not enough — httpcore has to be TOLD to use it.

        ``_assign_requests_to_connections`` is the only place a queued
        ``AsyncPoolRequest`` is handed the connection its ``wait_for_connection``
        blocks on, and httpcore runs it only as a request enters or leaves the
        pool. A sweep is neither. So this is the wedge the module exists to end,
        in its worst form: every slot an orphan, a request already parked. A
        sweep that only unpools leaves that request to die on its pool timeout
        seconds after the space it needed appeared.
        """
        async with _orphan_sweep(max_connections=1) as minted:
            assert minted is not None, _NO_ORPHAN_MINTED
            before = http_pool.census(minted.client)
            assert before is not None
            assert before.total == before.max_connections == 1, (
                f'wanted a pool whose one and only slot is the orphan, got {before}'
            )

            queued = asyncio.create_task(minted.client.post(_URL, content=b'{}'))
            try:
                await _step_until(
                    lambda: bool(_queued(minted.pool)),
                    what='the second request queueing for the slot the orphan holds',
                )

                assert await http_pool.reap_orphaned_connections(minted.client) == 1

                # CLOCK-FREE, deliberately: a sweep that merely unpooled would
                # exhaust these steps and fail here naming what it was waiting
                # for, rather than sitting out a real pool timeout.
                await _step_until(
                    queued.done,
                    what='the queued request being dispatched into the freed slot',
                )
                assert (await queued).status_code == 200
            finally:
                queued.cancel()
                with contextlib.suppress(BaseException):
                    await queued


class TestReapingSparesLiveTraffic:
    """The one thing standing between this fix and a reaper that kills live traffic.

    A slow-but-healthy request looks exactly like an orphan on every axis
    except one: its ``AsyncPoolRequest`` is still in ``pool._requests``. This
    pins that axis, so no future simplification of the predicate can quietly
    drop it.
    """

    async def test_an_in_flight_request_is_not_reaped_and_completes(self) -> None:
        gate = asyncio.Event()
        harness = _build_harness(read_gate=gate)
        task = asyncio.create_task(harness.client.post(_URL, content=b'{}'))
        try:
            await _step_until(
                lambda: any(s.reached_read for s in harness.backend.streams),
                what='the in-flight request reaching its response read',
            )

            # NON-VACUOUS, and this assertion is what makes it so: the
            # connection is pooled, not idle and not closed — precisely the
            # state an abandoned one is in. Ownership is the ONLY term that
            # separates the two.
            assert _unreclaimable(harness.pool), 'the in-flight connection never reached the pool'
            census = http_pool.census(harness.client)
            assert census is not None
            assert census.orphaned == 0

            assert await http_pool.reap_orphaned_connections(harness.client) == 0
            assert not any(s.closed for s in harness.backend.streams), (
                'the reaper closed a socket a live request was still using'
            )

            gate.set()
            response = await task
            assert response.status_code == 200
        finally:
            gate.set()
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
            with contextlib.suppress(Exception):
                await harness.client.aclose()


def _client_without_pool() -> httpx.AsyncClient:
    """A supported transport that simply is not the pooled one."""
    return httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200))
    )


def _client_with_malformed_pool() -> httpx.AsyncClient:
    """A pool missing one attribute the module reads.

    This is the shape an httpx release that relocated its internals would
    actually present: the walk still lands on a pool object, and only the
    attribute check catches it.
    """
    harness = _build_harness()
    delattr(harness.pool, '_requests')
    return harness.client


# Each case pairs a broken client with the attribute path the WARNING must
# name, so a message that degraded to a generic "could not resolve pool"
# fails here rather than passing as if it were still diagnostic. The two
# parametrize lists below are derived from it rather than written twice: only
# one of the two tests has anything to say about the path.
_UNRESOLVABLE_CASES: dict[str, tuple[Callable[[], httpx.AsyncClient], str]] = {
    'transport-has-no-pool': (_client_without_pool, '_transport._pool'),
    'pool-missing-an-attribute': (_client_with_malformed_pool, '_requests'),
}

_UNRESOLVABLE = [
    pytest.param(build, path, id=case)
    for case, (build, path) in _UNRESOLVABLE_CASES.items()
]
_UNRESOLVABLE_CLIENTS = [
    pytest.param(build, id=case) for case, (build, _) in _UNRESOLVABLE_CASES.items()
]


def _records(caplog: pytest.LogCaptureFixture, level: int) -> list[logging.LogRecord]:
    """Records this module logged at exactly *level*.

    Filtering on ``r.name`` is also the "names the module" half of the
    assertion: a record from anywhere else does not count.
    """
    return [r for r in caplog.records if r.levelno == level and r.name == _LOGGER_NAME]


def _messages(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [r.getMessage() for r in _records(caplog, level)]


def _above_debug(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Everything this module said loudly enough to reach an operator."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == _LOGGER_NAME and r.levelno > logging.DEBUG
    ]


class TestShapeGuardDegradesLoudly:
    """The failure mode that would recreate the original incident with a fix in place.

    If a future httpx moves these private attributes, the reaper must not
    quietly become a permanent no-op — that is the 32-hour-invisible failure
    all over again, but now with a module in the tree that looks like it is
    handling the problem.

    XDIST: ``dashboard/pyproject.toml`` runs ``-n auto --dist loadgroup`` and
    no dashboard test declares an ``xdist_group``, so any two tests here can
    land on different workers in either order. Every test below therefore
    resets the latch in its own body and asserts any first/second-call pair
    WITHIN itself — never relying on a sibling having run, or not run.
    """

    @pytest.mark.parametrize('build_client', _UNRESOLVABLE_CLIENTS)
    async def test_both_entry_points_return_sentinels_rather_than_raising(
        self, build_client: Callable[[], httpx.AsyncClient]
    ) -> None:
        http_pool.reset_shape_guard()
        client = build_client()
        try:
            assert http_pool.census(client) is None
            assert await http_pool.reap_orphaned_connections(client) == 0
        finally:
            with contextlib.suppress(Exception):
                await client.aclose()

    @pytest.mark.parametrize(('build_client', 'expected_path'), _UNRESOLVABLE)
    async def test_the_first_failure_warns_naming_the_attribute_path(
        self,
        build_client: Callable[[], httpx.AsyncClient],
        expected_path: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        http_pool.reset_shape_guard()
        client = build_client()
        try:
            with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
                assert http_pool.census(client) is None

            warnings = _messages(caplog, logging.WARNING)
            assert len(warnings) == 1, f'expected exactly one WARNING, got {warnings}'
            assert expected_path in warnings[0], (
                'the WARNING must name the attribute path that could not be resolved, '
                f'so an upgrade is diagnosable from the journal alone; got {warnings[0]}'
            )
        finally:
            with contextlib.suppress(Exception):
                await client.aclose()

    async def test_the_warning_fires_once_per_process_not_once_per_sweep(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A repeat at WARNING would bury the opening diagnostic under itself.

        The reaper sweeps every 60s, so a permanently broken guard would emit
        ~1400 identical lines a day — the flood ``mcp_fanout``'s
        transition-only WARNING policy exists to prevent. The line has to stay
        findable weeks later, which means it must be rare.
        """
        http_pool.reset_shape_guard()
        client = _client_without_pool()
        try:
            with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
                assert http_pool.census(client) is None
                assert len(_messages(caplog, logging.WARNING)) == 1

                caplog.clear()
                assert http_pool.census(client) is None
                assert await http_pool.reap_orphaned_connections(client) == 0

            assert _messages(caplog, logging.WARNING) == [], (
                'the shape guard warned again on a later sweep; with a 60s loop that '
                'is ~1400 lines a day burying the one line that mattered'
            )
            # Demoted, not discarded: still there for whoever turns DEBUG on.
            assert _messages(caplog, logging.DEBUG), (
                'repeat failures must remain visible at DEBUG'
            )
        finally:
            with contextlib.suppress(Exception):
                await client.aclose()


class FakeSleep:
    """A cadence driver: records each requested interval, waits for nothing.

    Ends the loop after *stop_after* ticks by raising ``CancelledError`` from
    the sleep — the same way the lifespan's ``task.cancel()`` does — so a test
    drives the loop through its real termination path rather than a special
    one built for testing.
    """

    def __init__(self, stop_after: int) -> None:
        self.intervals: list[float] = []
        self._stop_after = stop_after

    async def __call__(self, interval: float) -> None:
        if len(self.intervals) >= self._stop_after:
            raise asyncio.CancelledError
        self.intervals.append(interval)
        await _REAL_SLEEP(0)


class TestReaperLoop:
    """The background sweep, driven by a fake cadence rather than real time.

    Every test here patches ``asyncio.sleep``, so the loop's schedule is
    exercised in microseconds and asserted exactly, instead of being slept
    through and asserted approximately.
    """

    async def test_each_sweep_targets_the_client_it_was_handed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleep = FakeSleep(stop_after=3)
        monkeypatch.setattr(asyncio, 'sleep', sleep)
        swept: list[httpx.AsyncClient] = []

        async def _record(client: httpx.AsyncClient) -> int:
            swept.append(client)
            return 0

        monkeypatch.setattr(http_pool, 'reap_orphaned_connections', _record)
        harness = _build_harness()

        with pytest.raises(asyncio.CancelledError):
            await http_pool.reaper_loop(harness.client)

        # The CLIENT, not merely a client: a correct reaper pointed at the
        # wrong pool fixes nothing (see TestLifespanWiresTheReaper).
        assert swept == [harness.client] * 3
        assert sleep.intervals == [http_pool.REAP_INTERVAL_SECONDS] * 3

    async def test_a_sweep_that_reaps_warns_with_the_count_and_the_census(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One line per reap is diagnostic, not a flood: reaps run ~2/hour."""
        monkeypatch.setattr(asyncio, 'sleep', FakeSleep(stop_after=1))

        async def _reaped_two(client: httpx.AsyncClient) -> int:
            return 2

        monkeypatch.setattr(http_pool, 'reap_orphaned_connections', _reaped_two)
        harness = _build_harness()

        with (
            caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME),
            pytest.raises(asyncio.CancelledError),
        ):
            await http_pool.reaper_loop(harness.client)

        warnings = _messages(caplog, logging.WARNING)
        assert len(warnings) == 1, f'expected exactly one WARNING, got {warnings}'
        assert '2' in warnings[0], f'the WARNING must name the count, got {warnings[0]}'
        assert all(field in warnings[0] for field in ('total=', 'orphaned=', 'max_connections=')), (
            'the WARNING must name the resulting census, so one line answers both '
            f'"did it work" and "is the pool still healthy"; got {warnings[0]}'
        )

    async def test_a_sweep_that_reaps_nothing_says_nothing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The overwhelmingly common case. A line per quiet sweep is 1440/day."""
        monkeypatch.setattr(asyncio, 'sleep', FakeSleep(stop_after=3))

        async def _reaped_none(client: httpx.AsyncClient) -> int:
            return 0

        monkeypatch.setattr(http_pool, 'reap_orphaned_connections', _reaped_none)
        harness = _build_harness()

        with (
            caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME),
            pytest.raises(asyncio.CancelledError),
        ):
            await http_pool.reaper_loop(harness.client)

        assert _above_debug(caplog) == []

    async def test_a_failing_sweep_does_not_end_the_loop(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The house pattern for a background task that must outlive its errors.

        ``_burndown_loop`` and ``_metrics_loop`` both wrap each cycle in
        ``try/except Exception: logger.warning(..., exc_info=True)`` for the
        same reason: a loop that dies on its first bad cycle is a loop that
        silently stopped doing its job.
        """
        monkeypatch.setattr(asyncio, 'sleep', FakeSleep(stop_after=3))
        sweeps: list[httpx.AsyncClient] = []

        async def _fail_the_first_sweep(client: httpx.AsyncClient) -> int:
            sweeps.append(client)
            if len(sweeps) == 1:
                raise RuntimeError('pool went sideways')
            return 0

        monkeypatch.setattr(http_pool, 'reap_orphaned_connections', _fail_the_first_sweep)
        harness = _build_harness()

        with (
            caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME),
            pytest.raises(asyncio.CancelledError),
        ):
            await http_pool.reaper_loop(harness.client)

        assert len(sweeps) == 3, 'a failed sweep must not stop the ticks after it'
        failures = _records(caplog, logging.WARNING)
        assert len(failures) == 1, f'expected one WARNING for the failed sweep, got {failures}'
        assert failures[0].exc_info is not None, (
            'the traceback is the whole diagnostic value of this line; log it with '
            'exc_info=True as _burndown_loop and _metrics_loop do'
        )

    async def test_cancellation_propagates_so_shutdown_terminates(self) -> None:
        """No patched sleep — this is the real lifespan shutdown path.

        ``lifespan`` cancels the task and awaits it under
        ``contextlib.suppress(asyncio.CancelledError)``. A loop that caught
        ``CancelledError`` itself would return normally instead, and the
        shutdown would move on believing it had stopped something it had not.
        """
        harness = _build_harness()
        # An interval long enough that the loop is certainly parked in its
        # sleep when the cancel lands, which is where shutdown finds it.
        task = asyncio.create_task(http_pool.reaper_loop(harness.client, interval=3600.0))
        # THE ONE STEP THAT PARKS IT. A task just created has not run at all,
        # and nothing here observes the loop from outside — this test uses the
        # real asyncio.sleep precisely so it exercises the real shutdown path.
        # So the wait is this yield and nothing else: it hands control to the
        # loop, which runs until it suspends in its own sleep. Deleting it
        # would leave the cancel landing on a task that never started, and the
        # assertion below would still pass.
        await asyncio.sleep(0)

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert task.cancelled(), (
            'the loop swallowed CancelledError; lifespan shutdown would hang waiting '
            'for a task that had already decided to keep running'
        )


# A pool small enough that the high-water mark is four requests away rather
# than eighty. 4/5 is EXACTLY the 0.8 fraction, so the "at" case pins the
# inclusive boundary itself rather than a comfortable overshoot.
_SMALL_MAX_CONNECTIONS = 5
_AT_HIGH_WATER = 4
_BELOW_HIGH_WATER = 3


async def _fill_pool(client: httpx.AsyncClient, connections: int) -> None:
    """Leave *connections* idle connections pooled, one per distinct origin.

    Distinct origins are the point: httpcore reuses a pooled connection for
    the same one, so N requests to a single host would occupy a single slot.
    """
    for port in range(9000, 9000 + connections):
        response = await client.post(f'http://svc.local:{port}/mcp', content=b'{}')
        assert response.status_code == 200


@contextlib.asynccontextmanager
async def _pool_holding(occupancy: int) -> AsyncIterator[httpx.AsyncClient]:
    """A small pool left holding exactly *occupancy* idle connections."""
    harness = _build_harness(max_connections=_SMALL_MAX_CONNECTIONS)
    try:
        await _fill_pool(harness.client, occupancy)
        reading = http_pool.census(harness.client)
        assert reading is not None and reading.total == occupancy, (
            f'wanted a pool holding {occupancy} connections, got {reading}'
        )
        yield harness.client
    finally:
        with contextlib.suppress(Exception):
            await harness.client.aclose()


async def _sweep_once(client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run exactly one reaper sweep, through the real loop."""
    monkeypatch.setattr(asyncio, 'sleep', FakeSleep(stop_after=1))
    with contextlib.suppress(asyncio.CancelledError):
        await http_pool.reaper_loop(client)


class TestPoolSaturationAlarm:
    """The guard against this fix silently ceasing to work.

    The original incident was invisible for 32 hours precisely because nothing
    in the dashboard reported pool occupancy: ``httpx.PoolTimeout`` surfaced
    only as an "offline" pill on a perfectly healthy orchestrator. If orphans
    are ever minted faster than the sweep reclaims them — or by a mechanism
    this module's predicate does not cover — this line is the signal.

    It is an OCCUPANCY reading, taken from the pool's own bookkeeping, so it
    does not reintroduce the socket census task 3857 correctly rejected.

    XDIST: as with the shape guard, every test here re-arms the latch in its
    own body and asserts any first/second-sweep pair within itself.
    """

    def test_the_fixture_occupancies_straddle_the_threshold(self) -> None:
        """Keeps the tests below honest if POOL_HIGH_WATER_FRACTION ever moves.

        Without this, retuning the fraction would quietly turn the "at" case
        into a second "below" case, and both tests would still pass while
        asserting nothing about the boundary.
        """
        assert _AT_HIGH_WATER / _SMALL_MAX_CONNECTIONS >= http_pool.POOL_HIGH_WATER_FRACTION
        assert _BELOW_HIGH_WATER / _SMALL_MAX_CONNECTIONS < http_pool.POOL_HIGH_WATER_FRACTION

    async def test_a_sweep_at_the_high_water_mark_warns_with_the_census(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        http_pool.reset_saturation_guard()
        async with _pool_holding(_AT_HIGH_WATER) as client:
            with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
                await _sweep_once(client, monkeypatch)

        warnings = _messages(caplog, logging.WARNING)
        assert len(warnings) == 1, f'expected exactly one WARNING, got {warnings}'
        assert all(
            field in warnings[0]
            for field in ('total=', 'idle=', 'orphaned=', 'max_connections=')
        ), (
            'the WARNING must carry the whole census: "4 of 5" alone does not say '
            f'whether the pool is busy or wedged; got {warnings[0]}'
        )

    async def test_a_sweep_below_the_high_water_mark_says_nothing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        http_pool.reset_saturation_guard()
        async with _pool_holding(_BELOW_HIGH_WATER) as client:
            with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
                await _sweep_once(client, monkeypatch)

        assert _above_debug(caplog) == [], (
            'an ordinarily busy pool must be silent, or the alarm becomes noise and '
            'stops being read'
        )

    async def test_sustained_saturation_warns_once_then_drops_to_debug(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A wedged pool stays wedged; one line per minute forever is not a signal."""
        http_pool.reset_saturation_guard()
        async with _pool_holding(_AT_HIGH_WATER) as client:
            with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
                await _sweep_once(client, monkeypatch)
                assert len(_messages(caplog, logging.WARNING)) == 1

                caplog.clear()
                await _sweep_once(client, monkeypatch)

        assert _messages(caplog, logging.WARNING) == []
        assert _messages(caplog, logging.DEBUG), 'demoted, not discarded'

    async def test_falling_below_the_mark_re_arms_the_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Otherwise the alarm fires once per process and never again.

        A pool that saturates, recovers, and saturates again has had two
        incidents, and the second one matters at least as much as the first.
        """
        http_pool.reset_saturation_guard()
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            async with _pool_holding(_AT_HIGH_WATER) as client:
                await _sweep_once(client, monkeypatch)
            assert len(_messages(caplog, logging.WARNING)) == 1

            caplog.clear()
            async with _pool_holding(_BELOW_HIGH_WATER) as client:
                await _sweep_once(client, monkeypatch)
            assert _above_debug(caplog) == [], 'recovery itself must be silent'

            async with _pool_holding(_AT_HIGH_WATER) as client:
                await _sweep_once(client, monkeypatch)

        assert len(_messages(caplog, logging.WARNING)) == 1, (
            'saturating again after a recovery must warn again; a latch that never '
            're-arms reports only the first incident a process ever sees'
        )


# ---------------------------------------------------------------------------
# Lifespan wiring
#
# Driven through `lifespan` directly rather than a TestClient — the recipe from
# tests/test_app_http_limits.py::TestLifespanWiresTheLimits. The conftest
# `client` fixture is function-scoped while ~15 modules hold module-scoped
# TestClient(app) fixtures that clobber app.state, so a TestClient here would
# be asserting against whichever app.state won the race.
# ---------------------------------------------------------------------------


class _StateThatNeverReturnsTheClient(State):
    """An ``app.state`` whose ``http_client`` READS hand back a sentinel.

    This is what makes task 3771's split-binding invariant falsifiable rather
    than merely asserted. ``app.state`` is one mutable namespace shared by
    every overlapping lifespan over an app, so a handle read back from it can
    belong to someone else — which is why handles must bind to ARGUMENTS.
    Against this state, a call site that passed ``app.state.http_client``
    captures the sentinel and fails; one that passed its local captures the
    real client and passes.

    Writes still land normally, so the rest of ``lifespan`` is untouched. Only
    request handlers read this key back, and none of them run here.
    """

    def __getattr__(self, key: str) -> Any:
        if key == 'http_client':
            return _NOT_THE_LIFESPANS_CLIENT
        return super().__getattr__(key)


_NOT_THE_LIFESPANS_CLIENT = object()


def _task_state(task: asyncio.Task | None) -> dict[str, bool]:
    """What was true of *task* at this instant."""
    return {
        'exists': task is not None,
        'done': task is not None and task.done(),
        'cancelled': task is not None and task.cancelled(),
    }


@dataclass
class _LifespanRun:
    """What one full lifespan did with its reaper."""

    app: FastAPI
    constructed: list[httpx.AsyncClient] = field(default_factory=list)
    reaper_client: object = None
    reaper_task: asyncio.Task | None = None
    at_detached_reap: dict[str, bool] = field(default_factory=dict)
    at_aclose: dict[str, bool] = field(default_factory=dict)


async def _run_lifespan_with_a_recording_reaper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, app: FastAPI | None = None
) -> _LifespanRun:
    """Run one full lifespan, observing the reaper task at each teardown step.

    The two ``at_*`` readings are OBSERVATIONS of the running system rather
    than readings of ``lifespan``'s source, so the ordering claims they support
    survive a later refactor of it — the technique
    ``tests/test_app_lifespan_reap.py`` uses for the same reason.
    """
    apply_isolated_env(monkeypatch, tmp_path)
    run = _LifespanRun(app=app if app is not None else FastAPI(lifespan=lifespan))
    real_async_client = httpx.AsyncClient

    def _recording_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        # A REAL client, so the pool and aclose() behave normally; only the
        # moment of closing is instrumented.
        client = real_async_client(*args, **kwargs)
        run.constructed.append(client)
        real_aclose = client.aclose

        async def _observing_aclose() -> None:
            run.at_aclose = _task_state(run.reaper_task)
            return await real_aclose()

        client.aclose = _observing_aclose
        return client

    async def _recording_reaper_loop(client: object, *args: Any, **kwargs: Any) -> None:
        run.reaper_client = client
        run.reaper_task = asyncio.current_task()
        await asyncio.Event().wait()  # park until shutdown cancels it

    async def _observing_reap_detached() -> None:
        run.at_detached_reap = _task_state(run.reaper_task)

    with (
        patch('dashboard.app.httpx.AsyncClient', _recording_async_client),
        patch('dashboard.app.reaper_loop', _recording_reaper_loop),
        patch('dashboard.app.reap_detached_refreshes', _observing_reap_detached),
        patch('dashboard.app.collect_snapshot', new=AsyncMock(return_value=None)),
        patch('dashboard.app.collect_metrics_snapshot', new=AsyncMock(return_value=None)),
    ):
        async with lifespan(run.app):
            await _step_until(
                lambda: run.reaper_task is not None, what='the reaper task starting'
            )
    return run


class TestLifespanWiresTheReaper:
    """A reaper nobody starts, or one pointed at the wrong pool, fixes nothing.

    The module above proves the reaper WORKS; this proves it is WIRED — the
    same pairing ``test_app_http_limits.py`` makes between the pure sizing
    helper and the lifespan that must actually pass its result.
    """

    async def test_the_reaper_runs_against_the_client_the_lifespan_constructed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = await _run_lifespan_with_a_recording_reaper(tmp_path, monkeypatch)

        assert len(run.constructed) == 1, (
            f'lifespan must construct exactly one shared client, got {run.constructed}'
        )
        assert run.reaper_client is run.constructed[0], (
            'the reaper must sweep the pool the rest of the app is using; against a '
            'client of its own it would run forever and reclaim nothing'
        )
        assert run.reaper_client is run.app.state.http_client

    async def test_the_reaper_binds_to_the_argument_not_to_app_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Task 3771's split-binding invariant: handles bind to ARGUMENTS.

        Stated in full in ``_metrics_loop``'s docstring and pinned for the
        other two loops by ``tests/test_lifespan_resource_binding.py``. An
        ``app.state`` read-back would hand the reaper an overlapping
        lifespan's client — closed, in this suite — and the sweep would fail
        silently for the rest of the process.
        """
        app = FastAPI(lifespan=lifespan)
        app.state = _StateThatNeverReturnsTheClient()

        run = await _run_lifespan_with_a_recording_reaper(tmp_path, monkeypatch, app=app)

        assert run.reaper_client is run.constructed[0], (
            'the reaper was handed something other than the local http_client — an '
            'app.state read-back binds it to whichever lifespan wrote there last'
        )

    async def test_shutdown_leaves_no_reaper_task_pending(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = await _run_lifespan_with_a_recording_reaper(tmp_path, monkeypatch)

        assert run.reaper_task is not None
        assert run.reaper_task.cancelled(), (
            'the reaper must be cancelled AND awaited by the lifespan that started '
            'it; a task left pending outlives its client and, since every '
            'TestClient(app) runs its own loop, outlives the loop it was bound to'
        )

    async def test_the_reaper_stops_before_the_detached_cache_reap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same reason the existing two cancels sit above it.

        ``reap_detached_refreshes`` makes one pass over a snapshot, not a
        barrier. A poller still running behind it can start a fresh bypass the
        reap has already walked past.
        """
        run = await _run_lifespan_with_a_recording_reaper(tmp_path, monkeypatch)

        assert run.at_detached_reap.get('cancelled'), (
            'the reaper must already be cancelled when reap_detached_refreshes() '
            f'runs; observed at that instant: {run.at_detached_reap}'
        )

    async def test_the_reaper_stops_before_the_shared_client_is_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A sweep unwinding against a closed pool is the failure this prevents."""
        run = await _run_lifespan_with_a_recording_reaper(tmp_path, monkeypatch)

        assert run.at_aclose.get('cancelled'), (
            'the reaper must already be cancelled when http_client.aclose() is '
            f'entered; observed at that instant: {run.at_aclose}'
        )


class TestWiringTheReaperKeepsTheExitPathGuarantee:
    """Adding a third task must not strand what the lifespan opened.

    ``tests/test_app_lifespan_reap.py::TestLifespanClosesItsResourcesEvenIfTheReapFails``
    pins this for the teardown as it stood. The reaper's cancel goes inside the
    same ``try:``, so a raise from anywhere above must still reach
    ``finally: _close_each(...)`` — otherwise the hook added here would be
    capable of causing the very ``RuntimeError: Event loop is closed`` that
    ``lifespan``'s docstring records for task 3466.
    """

    _BOOM = 'the detached reap itself blew up'

    async def test_a_failing_teardown_step_still_closes_the_shared_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        apply_isolated_env(monkeypatch, tmp_path)
        # Held, not inlined: app.state is the observable and must outlive the
        # context.
        app = FastAPI(lifespan=lifespan)

        with (
            patch(
                'dashboard.app.reap_detached_refreshes',
                new=AsyncMock(side_effect=RuntimeError(self._BOOM)),
            ),
            patch('dashboard.app.collect_snapshot', new=AsyncMock(return_value=None)),
            patch(
                'dashboard.app.collect_metrics_snapshot',
                new=AsyncMock(return_value=None),
            ),
            pytest.raises(RuntimeError, match=self._BOOM),
        ):
            async with lifespan(app):
                pass

        assert app.state.http_client.is_closed, (
            'aclose() is LAST in _close_each, so this one observable stands for the '
            'burndown store, the metrics store and the DB pool too'
        )
