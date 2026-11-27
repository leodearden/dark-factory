"""Child-process driver: one SIGTERM stops two embedded uvicorn servers composed from main.py's own seams.

Run as ``python -I <this file>`` from a clean cwd; the stop's log lines are the
output, on stderr in main.py's LOG_FORMAT. Asserted by
``fused-memory/tests/test_operator_signal_handler.py::TestOneStopSignalStopsBothServersCleanly``.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket

import uvicorn
from starlette.applications import Starlette

import fused_memory.server.main as main
from fused_memory.server.embedded_uvicorn import EmbeddedUvicornServer


class _ReadySignallingServer(EmbeddedUvicornServer):
    """Sets ``ready`` once uvicorn's startup has run."""

    def __init__(self, config: uvicorn.Config) -> None:
        super().__init__(config)
        self.ready = asyncio.Event()

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets=sockets)
        self.ready.set()


def _server() -> _ReadySignallingServer:
    return _ReadySignallingServer(
        main._build_uvicorn_config(Starlette(), host='127.0.0.1', port=0, graceful_shutdown_timeout=5),
    )


async def _until_started(server: _ReadySignallingServer, serving: asyncio.Task[None]) -> None:
    ready = asyncio.ensure_future(server.ready.wait())
    await asyncio.wait({ready, serving}, return_when=asyncio.FIRST_COMPLETED)
    ready.cancel()
    if not server.started:
        raise RuntimeError(f'server did not start: {serving.exception() if serving.done() else None!r}')


async def _stop_with_one_sigterm() -> None:
    servers = (_server(), _server())
    main._install_operator_stop_handler(main._make_operator_stop_callback(*servers))
    signals_drained = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGUSR2, signals_drained.set)

    serving = [asyncio.ensure_future(server.serve()) for server in servers]
    for server, task in zip(servers, serving, strict=True):
        await _until_started(server, task)

    os.kill(os.getpid(), signal.SIGTERM)
    await asyncio.gather(*serving)

    # Signals reach the loop in arrival order through its wakeup fd, so any
    # SIGTERM a server re-raised as serve() returned is handled before this one.
    os.kill(os.getpid(), signal.SIGUSR2)
    await signals_drained.wait()


if __name__ == '__main__':
    asyncio.run(_stop_with_one_sigterm())
