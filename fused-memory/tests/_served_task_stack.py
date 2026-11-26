"""The hermetic fused-memory task stack, served over real HTTP behind production's dispatch guards.

``served_task_stack(tmp_path)`` serves ``_hermetic_task_stack.hermetic_task_stack`` over a
fresh store and yields where it answers: the base URL (no ``/mcp`` suffix), the
``project_root`` every call must name, and the residue spool the markup guard preserves a
refused payload in. ``serving(loop, app)`` is the transport alone: any ASGI app, served by
uvicorn on *loop* from a daemon thread, with a typed error for each way that goes wrong.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypeAlias

import uvicorn
from _hermetic_task_stack import hermetic_task_stack
from mcp.server.fastmcp import FastMCP

from fused_memory.config.schema import ServerConfig
from fused_memory.server.main import (
    _add_json_http_error_handler,
    _ASGIExceptionShield,
    _install_tool_dispatch_guards,
)
from fused_memory.server.markup_residue_spool import MarkupResidueSpool

HOST: Final = '127.0.0.1'
GRACEFUL_SHUTDOWN_SECONDS: Final = 5
TEARDOWN_BOUND_SECONDS: Final = 30

AsgiApp: TypeAlias = Callable[..., Awaitable[Any]]


@dataclass(frozen=True)
class ServedTaskStack:
    """Where the served stack answers, the project root its store keeps, and its markup guard's residue spool."""

    url: str
    root: Path
    residue_spool: MarkupResidueSpool


class ServedTaskStackDidNotStart(RuntimeError):
    """uvicorn never finished starting; *cause* is what the serving thread raised, if anything."""

    def __init__(self, cause: BaseException | None) -> None:
        self.cause = cause
        super().__init__(f'the served task stack did not start (serving thread raised: {cause!r})')


class ServedTaskStackCrashed(RuntimeError):
    """uvicorn's serve() raised after a good startup; *cause* is what it raised."""

    def __init__(self, cause: BaseException) -> None:
        self.cause = cause
        super().__init__(f'the served task stack crashed after it started: {cause!r}')


class ServedTaskStackHung(RuntimeError):
    """The serving thread outlived ``should_exit`` by the teardown bound, and *loop_still_running* if ``loop.stop`` did not free it either."""

    def __init__(self, thread_name: str, *, loop_still_running: bool) -> None:
        self.loop_still_running = loop_still_running
        after_stop = 'nor after loop.stop, so its loop is left running' if loop_still_running else 'and loop.stop freed it'
        super().__init__(
            f'{thread_name} did not stop within {TEARDOWN_BOUND_SECONDS}s of should_exit, {after_stop}'
        )


class _SignallingServer(uvicorn.Server):
    """A uvicorn server that sets *settled* once its startup has run: listener live, lifespan up, or ``started`` left False."""

    def __init__(self, config: uvicorn.Config, settled: threading.Event) -> None:
        super().__init__(config)
        self._settled = settled

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets=sockets)
        self._settled.set()


class _ServingThread(threading.Thread):
    """Runs *server* on *loop* over *listener*, keeping what serve() raised; sets *settled* on exit too, so a startup that never returned cannot hang the caller."""

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
    """A socket bound to a free 127.0.0.1 port and held from here on; uvicorn's ``create_server`` listens on it."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind((HOST, 0))
    return listener


def _drain_pending_tasks(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel and await every task still pending on *loop*, as ``asyncio.run`` does before it closes one."""
    pending = asyncio.all_tasks(loop)
    if not pending:
        return
    for task in pending:
        task.cancel()
    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))


def _stop(loop: asyncio.AbstractEventLoop, server: uvicorn.Server, thread: threading.Thread) -> None:
    """Ask uvicorn to exit; if the thread outlives the bound, stop *loop* under it and raise ServedTaskStackHung."""
    server.should_exit = True
    thread.join(timeout=TEARDOWN_BOUND_SECONDS)
    if not thread.is_alive():
        return
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=TEARDOWN_BOUND_SECONDS)
    raise ServedTaskStackHung(thread.name, loop_still_running=thread.is_alive())


@contextlib.contextmanager
def serving(loop: asyncio.AbstractEventLoop, app: AsgiApp) -> Iterator[str]:
    """Serve *app* on *loop* from a daemon thread for the block's lifetime, yielding its base URL.

    ServedTaskStackDidNotStart if uvicorn's startup did not complete, ServedTaskStackCrashed
    if serve() raised after it did, ServedTaskStackHung if the thread outlived teardown.
    """
    settled = threading.Event()
    config = uvicorn.Config(
        app,
        log_level='warning',
        ws='none',
        lifespan='on',
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
    )
    server = _SignallingServer(config, settled)
    started = False
    with contextlib.closing(_listener()) as listener:
        thread = _ServingThread(loop, server, listener, settled)
        thread.start()
        try:
            settled.wait()
            started = server.started
            if started:
                yield f'http://{HOST}:{listener.getsockname()[1]}'
        finally:
            _stop(loop, server, thread)
    if not started:
        raise ServedTaskStackDidNotStart(thread.failure)
    if thread.failure is not None:
        raise ServedTaskStackCrashed(thread.failure)


def _guarded_app(server: FastMCP, residue_spool: MarkupResidueSpool) -> AsgiApp:
    """*server*'s HTTP app assembled as ``fused_memory.server.main``'s http transport assembles it."""
    _install_tool_dispatch_guards(server, known_projects=None, prefix_resolver=None, residue_spool=residue_spool)
    production = ServerConfig()
    server.settings.stateless_http = production.stateless_http
    server.settings.json_response = production.json_response
    app = server.streamable_http_app()
    _add_json_http_error_handler(app)
    return _ASGIExceptionShield(app)


def _close_unless_held(loop: asyncio.AbstractEventLoop, entered: contextlib.AsyncExitStack) -> None:
    """Exit *entered* on *loop*, then drain and close it; a loop a hung serving thread still runs is left alone."""
    if loop.is_running():
        return
    try:
        loop.run_until_complete(entered.aclose())
    finally:
        _drain_pending_tasks(loop)
        loop.close()


@contextlib.contextmanager
def served_task_stack(tmp_path: Path) -> Iterator[ServedTaskStack]:
    """Serve a fresh hermetic task stack rooted under *tmp_path* for the block's lifetime."""
    root = (tmp_path / 'store').resolve()
    root.mkdir()
    residue_spool = MarkupResidueSpool(tmp_path / 'markup-residue')
    loop = asyncio.new_event_loop()
    entered = contextlib.AsyncExitStack()
    try:
        stack = loop.run_until_complete(entered.enter_async_context(hermetic_task_stack(root)))
        with serving(loop, _guarded_app(stack.server, residue_spool)) as url:
            yield ServedTaskStack(url=url, root=root, residue_spool=residue_spool)
    finally:
        _close_unless_held(loop, entered)
