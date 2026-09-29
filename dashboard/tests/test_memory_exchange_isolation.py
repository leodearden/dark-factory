"""A caller that gives up on an MCP call must not take the HTTP exchange with it.

The dashboard's budgets are native asyncio cancellations (``wait_for``,
``asyncio.timeout``, gather). httpcore shields its release path with anyio
scopes, which a native cancel passes straight through, so a cancel landing
there strands the connection ACTIVE under a pool request no task holds
(``dashboard/src/dashboard/http_pool.py`` states the class). The contract
pinned here is therefore stated at ``memory.py::mcp_tool_call``, the choke
point every dashboard HTTP call goes through: the caller's deadline still
fires on time, and the exchange finishes on its own and hands its connection
back to the pool.

Every assertion reads the pool through ``http_pool.census`` or through what
the server saw, never through httpcore's bookkeeping.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpcore
import httpx
import pytest
from _dashboard_helpers import apply_isolated_env
from fastapi import FastAPI

from dashboard.app import lifespan
from dashboard.data.memory import mcp_tool_call, reset_sessions
from dashboard.http_pool import PoolCensus, census

_TOOL_RESULT = {'ok': True}
_QUIESCENCE_SECONDS = 5.0


def _mcp_body(request_id: int) -> bytes:
    """One JSON-RPC result that satisfies initialize, the notify and tools/call."""
    return json.dumps({
        'jsonrpc': '2.0',
        'id': request_id,
        'result': {'content': [{'type': 'text', 'text': json.dumps(_TOOL_RESULT)}]},
    }).encode()


def _http_response(body: bytes) -> bytes:
    return (
        b'HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n'
        b'content-length: %d\r\n\r\n' % len(body)
    ) + body


class _HeldToolCallServer:
    """A keep-alive MCP server on a real socket that can hold one tools/call.

    ``connections_accepted`` is how a test tells a reused connection from a
    fresh one without looking inside the pool.
    """

    def __init__(self) -> None:
        self.connections_accepted = 0
        self.handlers: set[asyncio.Task[Any]] = set()
        self.hold_next_tool_call = True
        self.tool_call_held = asyncio.Event()
        self.release = asyncio.Event()
        self.url = ''
        self._server: asyncio.Server | None = None

    async def __aenter__(self) -> _HeldToolCallServer:
        self._server = await asyncio.start_server(self._serve, '127.0.0.1', 0)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f'http://127.0.0.1:{port}'
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        assert self._server is not None
        self._server.close()

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections_accepted += 1
        handler = asyncio.current_task()
        assert handler is not None
        self.handlers.add(handler)
        try:
            while True:
                try:
                    head = await reader.readuntil(b'\r\n\r\n')
                except asyncio.IncompleteReadError:
                    return
                length = next(
                    int(line.split(b':', 1)[1])
                    for line in head.split(b'\r\n')
                    if line.lower().startswith(b'content-length:')
                )
                request = json.loads(await reader.readexactly(length))
                if request['method'] == 'tools/call' and self.hold_next_tool_call:
                    self.hold_next_tool_call = False
                    self.tool_call_held.set()
                    await self.release.wait()
                writer.write(_http_response(_mcp_body(request.get('id', 0))))
                await writer.drain()
        finally:
            writer.close()


async def _census_once_quiet(client: httpx.AsyncClient) -> PoolCensus | None:
    """The pool's census once every connection is idle, or the last one read."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _QUIESCENCE_SECONDS
    reading = census(client)
    while loop.time() < deadline:
        reading = census(client)
        if reading is not None and reading.total and reading.total == reading.idle:
            return reading
        await asyncio.sleep(0.01)
    return reading


def _pool_is_empty(client: httpx.AsyncClient) -> bool:
    reading = census(client)
    return reading is not None and reading.total == 0


@pytest.fixture(autouse=True)
def _fresh_sessions() -> Iterable[None]:
    reset_sessions()
    yield
    reset_sessions()


async def test_a_caller_that_gives_up_leaves_its_connection_reusable() -> None:
    async with _HeldToolCallServer() as server, httpx.AsyncClient() as client:
        call = asyncio.create_task(mcp_tool_call(client, server.url, 'get_status', {}))
        await server.tool_call_held.wait()

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(call, timeout=0)
        server.release.set()

        quiet = await _census_once_quiet(client)
        assert quiet is not None and quiet.total == quiet.idle == 1, (
            f'the abandoned exchange did not hand its connection back: {quiet}'
        )
        assert await mcp_tool_call(client, server.url, 'get_status', {}) == _TOOL_RESULT
        assert server.connections_accepted == 1


async def test_an_abandoned_exchange_that_fails_is_settled_silently(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Nobody awaits an abandoned exchange, so nobody else can retrieve its error."""
    async with _HeldToolCallServer() as server, httpx.AsyncClient() as client:
        call = asyncio.create_task(
            mcp_tool_call(client, server.url, 'get_status', {}, timeout=0.3),
        )
        await server.tool_call_held.wait()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(call, timeout=0)

        with caplog.at_level(logging.ERROR, logger='asyncio'):
            loop = asyncio.get_running_loop()
            deadline = loop.time() + _QUIESCENCE_SECONDS
            while not _pool_is_empty(client) and loop.time() < deadline:
                await asyncio.sleep(0.01)
            gc.collect()
            await asyncio.sleep(0)
        server.release.set()

    assert _pool_is_empty(client), f'the exchange never failed: {census(client)}'
    assert not [r for r in caplog.records if 'never retrieved' in r.getMessage()]


async def test_an_abandoned_exchange_does_not_outlive_the_apps_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lifespan ends every detached exchange before it closes the client.

    Observed at the instant ``aclose`` is entered, as
    ``test_app_lifespan_reap.py`` observes its bypass reap.
    """
    apply_isolated_env(monkeypatch, tmp_path)
    before = asyncio.all_tasks()
    still_running_at_aclose: list[set[asyncio.Task[Any]]] = []
    real_async_client = httpx.AsyncClient

    def _observing_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        client = real_async_client(*args, **kwargs)
        real_aclose = client.aclose

        async def _observing_aclose() -> None:
            still_running_at_aclose.append({
                task for task in asyncio.all_tasks() - before - server.handlers
                if not task.done()
            })
            await real_aclose()

        client.aclose = _observing_aclose
        return client

    async with _HeldToolCallServer() as server:
        with (
            patch('dashboard.app.httpx.AsyncClient', _observing_async_client),
            patch('dashboard.loops.collect_snapshot', new=AsyncMock(return_value=None)),
            patch('dashboard.loops.collect_metrics_snapshot', new=AsyncMock(return_value=None)),
        ):
            app = FastAPI(lifespan=lifespan)
            async with lifespan(app):
                call = asyncio.create_task(
                    mcp_tool_call(app.state.http_client, server.url, 'get_status', {}),
                )
                await server.tool_call_held.wait()
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(call, timeout=0)
        server.release.set()

    assert still_running_at_aclose == [set()], (
        f'exchanges still running when the client closed: {still_running_at_aclose}'
    )


# ── the stranded-owned class, reproduced deterministically ─────────────
#
# A real socket makes WHERE a cancellation lands a matter of timing, so this
# half runs the pool over an in-memory network backend and interrupts the call
# after each of a range of exact event-loop steps. One of those positions is
# inside the release path the module docstring names. Two interruptions: the
# caller giving up, and the exchange's own time bound expiring.

_FAKE_URL = 'http://svc.local'
_MAX_STEPS = 64
_SETTLE_STEPS = 64


class _CannedStream(httpcore.AsyncNetworkStream):
    def __init__(self) -> None:
        self._pending = b''

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self._pending = _http_response(_mcp_body(1))

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        chunk, self._pending = self._pending[:max_bytes], self._pending[max_bytes:]
        return chunk

    async def aclose(self) -> None:
        self._pending = b''

    def get_extra_info(self, info: str) -> Any:
        return None


class _CannedBackend(httpcore.AsyncNetworkBackend):
    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        return _CannedStream()


def _client_without_sockets() -> httpx.AsyncClient:
    transport = httpx.AsyncHTTPTransport()
    transport._pool = httpcore.AsyncConnectionPool(network_backend=_CannedBackend())
    return httpx.AsyncClient(transport=transport)


async def _steps(count: int) -> None:
    for _ in range(count):
        await asyncio.sleep(0)


class _ManualClock:
    """Stands in for the loop's clock, so a time bound expires on a chosen step."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._now = loop.time()

    def time(self) -> float:
        return self._now

    def pass_every_deadline(self, _call: asyncio.Task[Any]) -> None:
        self._now += 1e9


def _cancel(call: asyncio.Task[Any]) -> None:
    call.cancel()


async def _interrupt_at(
    steps: int, interrupt: Callable[[asyncio.Task[Any]], None],
) -> tuple[bool, PoolCensus | None]:
    """Interrupt a cold-session call after *steps* loop steps.

    Returns whether the call had already finished by then, and the census once
    everything it started has settled.
    """
    reset_sessions()
    async with _client_without_sockets() as client:
        call = asyncio.create_task(mcp_tool_call(client, _FAKE_URL, 'get_status', {}))
        await _steps(steps)
        finished_first = call.done()
        interrupt(call)
        await asyncio.gather(call, return_exceptions=True)
        await _steps(_SETTLE_STEPS)
        return finished_first, census(client)


async def _stranded_by(
    interrupt: Callable[[asyncio.Task[Any]], None],
) -> list[tuple[int, PoolCensus | None]]:
    sweep = [(steps, *await _interrupt_at(steps, interrupt)) for steps in range(_MAX_STEPS)]
    assert sweep[-1][1], 'the sweep ended before the call did, so it missed positions'
    return [
        (steps, reading)
        for steps, _finished, reading in sweep
        if reading is None or reading.total != reading.idle
    ]


async def test_no_cancellation_point_strands_a_connection() -> None:
    stranded = await _stranded_by(_cancel)
    assert stranded == [], (
        'cancelling the caller at these event-loop steps left a connection that is '
        f'neither idle nor released: {stranded}'
    )


async def test_no_expiry_point_of_the_exchange_bound_strands_a_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bound must be one httpcore's shields honour, or it strands as a caller would."""
    loop = asyncio.get_running_loop()
    clock = _ManualClock(loop)
    monkeypatch.setattr(loop, 'time', clock.time)
    stranded = await _stranded_by(clock.pass_every_deadline)
    assert stranded == [], (
        "the exchange's own bound expiring at these event-loop steps left a "
        f'connection that is neither idle nor released: {stranded}'
    )
