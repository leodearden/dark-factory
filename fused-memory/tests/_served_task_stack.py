"""The hermetic fused-memory task stack, served over real HTTP behind production's dispatch guards.

``served_task_stack(tmp_path)`` enters ``_hermetic_task_stack.hermetic_task_stack`` over a
fresh store (that module's docstring says why its curator is off), installs
``fused_memory.server.main._install_tool_dispatch_guards`` on the server, markup guard
outermost, and serves ``streamable_http_app()`` through uvicorn on a 127.0.0.1 listener
bound before the serving thread starts. It yields the base URL, with no ``/mcp`` suffix,
and the store's resolved root, the ``project_root`` every call must name.

Readiness is ``uvicorn.Server.startup`` returning, signalled through an Event that the
serving thread also sets when it exits; there is no sleep and no readiness deadline, so a
true hang is bounded by the per-test pytest timeout. The shape follows
``escalation/tests/conftest.py::serve_escalation_mcp``: listener first, loop created by the
caller, pending tasks drained before the loop closes. The stack is entered and exited on
that same loop, from the caller's thread, before the serving thread starts and after it
joins, so a store that fails to open raises in the caller.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import uvicorn
from _hermetic_task_stack import hermetic_task_stack
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette

from fused_memory.config.schema import ServerConfig
from fused_memory.server.main import _install_tool_dispatch_guards
from fused_memory.server.markup_residue_spool import MarkupResidueSpool

HOST: Final = '127.0.0.1'
GRACEFUL_SHUTDOWN_SECONDS: Final = 5


@dataclass(frozen=True)
class ServedTaskStack:
    """Where the served stack answers, and the project root its store keeps."""

    url: str
    root: Path


class ServedTaskStackDidNotStart(RuntimeError):
    """uvicorn never finished starting the served stack; *cause* is what the serving thread raised, if anything."""

    def __init__(self, cause: BaseException | None) -> None:
        self.cause = cause
        super().__init__(f'the served task stack did not start (serving thread raised: {cause!r})')


class _SignallingServer(uvicorn.Server):
    """A uvicorn server that sets *settled* once its startup has run: listener live, lifespan up, or ``started`` left False."""

    def __init__(self, config: uvicorn.Config, settled: threading.Event) -> None:
        super().__init__(config)
        self._settled = settled

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets=sockets)
        self._settled.set()


class _ServingThread(threading.Thread):
    """Runs *server* on *loop* over *listener*; sets *settled* on exit too, so a startup that never returned cannot hang the caller."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        server: uvicorn.Server,
        listener: socket.socket,
        settled: threading.Event,
    ) -> None:
        super().__init__(name=f'served-task-stack-{listener.getsockname()[1]}', daemon=True)
        self._loop = loop
        self._server = server
        self._listener = listener
        self._settled = settled
        self.failure: BaseException | None = None

    def run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._server.serve(sockets=[self._listener]))
        except BaseException as exc:
            self.failure = exc
        finally:
            self._settled.set()


def _listener() -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((HOST, 0))
    listener.listen()
    return listener


def _drain_pending_tasks(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel and await every task still pending on *loop*, as ``asyncio.run`` does before it closes one."""
    pending = asyncio.all_tasks(loop)
    if not pending:
        return
    for task in pending:
        task.cancel()
    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))


def _guarded_app(server: FastMCP, residue_dir: Path) -> Starlette:
    """*server*'s HTTP app behind production's dispatch guards and HTTP settings."""
    _install_tool_dispatch_guards(
        server, known_projects=None, prefix_resolver=None, residue_spool=MarkupResidueSpool(residue_dir)
    )
    production = ServerConfig()
    server.settings.stateless_http = production.stateless_http
    server.settings.json_response = production.json_response
    return server.streamable_http_app()


@contextlib.contextmanager
def _serving(loop: asyncio.AbstractEventLoop, app: Starlette, listener: socket.socket) -> Iterator[None]:
    """Serve *app* on *loop* in a daemon thread until the block exits; ServedTaskStackDidNotStart if uvicorn never started."""
    settled = threading.Event()
    config = uvicorn.Config(
        app,
        log_level='warning',
        ws='none',
        lifespan='on',
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
    )
    server = _SignallingServer(config, settled)
    thread = _ServingThread(loop, server, listener, settled)
    thread.start()
    settled.wait()
    try:
        if not server.started:
            thread.join()
            raise ServedTaskStackDidNotStart(thread.failure)
        yield
    finally:
        server.should_exit = True
        thread.join()


@contextlib.contextmanager
def served_task_stack(tmp_path: Path) -> Iterator[ServedTaskStack]:
    """Serve a fresh hermetic task stack rooted under *tmp_path* for the block's lifetime."""
    root = (tmp_path / 'store').resolve()
    root.mkdir()
    loop = asyncio.new_event_loop()
    entered = contextlib.AsyncExitStack()
    try:
        with contextlib.closing(_listener()) as listener:
            stack = loop.run_until_complete(entered.enter_async_context(hermetic_task_stack(root)))
            try:
                app = _guarded_app(stack.server, tmp_path / 'markup-residue')
                with _serving(loop, app, listener):
                    yield ServedTaskStack(url=f'http://{HOST}:{listener.getsockname()[1]}', root=root)
            finally:
                loop.run_until_complete(entered.aclose())
    finally:
        _drain_pending_tasks(loop)
        loop.close()
