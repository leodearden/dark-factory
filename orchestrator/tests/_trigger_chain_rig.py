"""Serve one fixture project to a real orchestrator and to the factory's own script.

A live fused-memory MCP server (production's ``streamable_http_app`` under
uvicorn) runs over a real SqliteTaskBackend + TaskInterceptor in a daemon
thread with its own event loop, rooted at a seeded git repo that has no
``scripts/``.  The orchestrator side is a real Scheduler whose fused-memory URL
points at that server, so every tool call takes the production HTTP path.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
from collections.abc import AsyncIterator, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar
from unittest.mock import AsyncMock

import uvicorn
from _git_fixtures import seed_repo
from _trigger_chain_doc_shapes import concretize, snippet_submit_calls
from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import FusedMemoryConfig as FusedMemoryServerConfig
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.server.tools import create_mcp_server
from shared.mcp_envelope import parse_tool_result

from orchestrator.config import FusedMemoryConfig, OrchestratorConfig, PsiAdmissionConfig
from orchestrator.scheduler import Scheduler

PROJECT_ID = 'lambda_fixture'
FIRST_CHECK_SECS = 86400

_READY_TIMEOUT_S = 60.0
_READY_POLL_S = 0.05
_TOOL_TIMEOUT_S = 30.0

_T = TypeVar('_T')


class FusedMemoryNotReady(RuntimeError):
    """The live server never answered a tool call; the cause is chained when one was seen."""


@dataclass(frozen=True)
class _Stack:
    backend: SqliteTaskBackend
    event_buffer: EventBuffer
    interceptor: TaskInterceptor

    async def close(self) -> None:
        await self.interceptor.close()
        await self.event_buffer.close()
        await self.backend.close()


async def _open_stack(root: Path, state_dir: Path, server_config: FusedMemoryServerConfig) -> _Stack:
    backend = SqliteTaskBackend(TaskmasterConfig(project_root=str(root)))
    await backend.start()
    event_buffer = EventBuffer(db_path=state_dir / 'events.db', buffer_size_threshold=100)
    await event_buffer.initialize()
    ticket_store = TicketStore(state_dir / 'tickets.db')
    await ticket_store.initialize()
    interceptor = TaskInterceptor(
        backend, None, event_buffer, server_config, ticket_store=ticket_store
    )
    return _Stack(backend, event_buffer, interceptor)


def _memory_service_stand_in() -> AsyncMock:
    memory = AsyncMock()
    memory.get_status.return_value = {}
    return memory


def _drain_pending_tasks(loop: asyncio.AbstractEventLoop) -> None:
    pending = asyncio.all_tasks(loop)
    for task in pending:
        task.cancel()
    if pending:
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))


class LiveFusedMemory:
    """A fused-memory MCP server for *root*, served from a thread that owns its loop.

    Every store touch goes through :meth:`run`, so aiosqlite only ever sees the
    server loop.
    """

    def __init__(self, root: Path, state_dir: Path) -> None:
        self._root = root
        self._state_dir = state_dir
        self._config = FusedMemoryServerConfig()
        self._config.taskmaster = None
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.bind(('127.0.0.1', 0))
        self.url = f'http://127.0.0.1:{self._listener.getsockname()[1]}'
        self._loop = asyncio.new_event_loop()
        self._stack: _Stack | None = None
        self._server: uvicorn.Server | None = None
        self._stopping = False
        self._serve_error: BaseException | None = None
        self._thread = threading.Thread(target=self._serve, name=f'fused-memory-{self.url}', daemon=True)

    @property
    def backend(self) -> SqliteTaskBackend:
        if self._stack is None:
            raise RuntimeError('the live fused-memory server has not started')
        return self._stack.backend

    def _serve(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except BaseException as exc:  # noqa: BLE001 - surfaced by start()
            self._serve_error = exc
        finally:
            _drain_pending_tasks(self._loop)
            self._loop.close()

    async def _main(self) -> None:
        self._stack = await _open_stack(self._root, self._state_dir, self._config)
        try:
            mcp = create_mcp_server(_memory_service_stand_in(), task_interceptor=self._stack.interceptor)
            mcp.settings.stateless_http = self._config.server.stateless_http
            mcp.settings.json_response = self._config.server.json_response
            self._server = uvicorn.Server(
                uvicorn.Config(
                    mcp.streamable_http_app(),
                    log_level='error',
                    ws='none',
                    timeout_graceful_shutdown=self._config.server.graceful_shutdown_timeout,
                )
            )
            self._server.should_exit = self._stopping
            await self._server.serve(sockets=[self._listener])
        finally:
            await self._stack.close()

    async def run(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Await *coro* on the server loop."""
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, self._loop))

    async def start(self, probe: Any) -> None:
        """Serve, and return once *probe* (an async no-arg callable) completes a real tool call."""
        self._thread.start()
        deadline = asyncio.get_running_loop().time() + _READY_TIMEOUT_S
        last_error: BaseException | None = None
        while self._thread.is_alive() and (remaining := deadline - asyncio.get_running_loop().time()) > 0:
            try:
                async with asyncio.timeout(remaining):
                    await probe()
                return
            except Exception as exc:  # noqa: BLE001 - any failure means not ready yet
                last_error = exc
            await asyncio.sleep(_READY_POLL_S)
        raise FusedMemoryNotReady(
            f'fused-memory on {self.url} answered no tool call '
            f'(server thread alive={self._thread.is_alive()}, raised={self._serve_error!r})'
        ) from last_error

    def _request_exit(self) -> None:
        self._stopping = True
        if self._server is not None:
            self._server.should_exit = True

    def stop(self) -> None:
        """Ask the server loop to exit, and wait (bounded) for its thread."""
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._request_exit)
        self._thread.join(timeout=self._config.server.graceful_shutdown_timeout + 10)
        self._listener.close()
        assert not self._thread.is_alive(), f'fused-memory serving thread for {self.url} did not stop'


class FakeClock:
    """One mutable aware-UTC instant, read by the Scheduler and the runner alike."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def set(self, now: datetime) -> None:
        self.now = now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)

    def timestamp(self) -> float:
        return self.now.timestamp()


@dataclass(frozen=True)
class TriggerChainRig:
    root: Path
    factory_root: Path
    server: LiveFusedMemory
    scheduler: Scheduler
    clock: FakeClock
    project_id: str = PROJECT_ID

    @property
    def url(self) -> str:
        return self.server.url

    async def call(self, tool: str, **arguments: Any) -> dict[str, Any]:
        """*tool*'s payload, called through the Scheduler's own transport."""
        reply = await self.scheduler.dispatch_tool(tool, arguments, timeout=_TOOL_TIMEOUT_S)
        payload, err = parse_tool_result(reply, None, dict)
        if not isinstance(payload, dict):
            raise AssertionError(f'{tool} returned no payload: {err}')
        return payload

    def snippet_call(self, role: str, run_id: str, skill: str = 'review') -> dict[str, Any]:
        """The snippet's ``submit_task`` call for *role*, concretized for this project."""
        placeholders = {
            'skill': skill,
            'run_id': run_id,
            'project_root': str(self.root),
            'project_id': self.project_id,
            'now + 86400 s, ISO-8601 UTC': (self.clock() + timedelta(seconds=FIRST_CHECK_SECS)).isoformat(),
        }
        calls = [concretize(call, placeholders) for call in snippet_submit_calls(self.factory_root)]
        (call,) = [call for call in calls if call['metadata']['trigger_chain']['role'] == role]
        return call

    async def submit(self, call: dict[str, Any]) -> str:
        """File a concretized snippet call; its task id."""
        reply = await self.call('submit_task', **call)
        if 'task_id' not in reply:
            raise AssertionError(f'submit_task refused {call["title"]!r}: {reply}')
        return str(reply['task_id'])


def _scheduler(root: Path, url: str, clock: FakeClock) -> Scheduler:
    config = OrchestratorConfig(
        project_root=root,
        fused_memory=FusedMemoryConfig(url=url, project_id=PROJECT_ID),
        max_per_module=1,
        psi_admission=PsiAdmissionConfig(enabled=False),
    )
    scheduler = Scheduler(config, wall_time_source=clock, time_source=clock.timestamp)
    scheduler.finish_startup()
    return scheduler


@contextlib.asynccontextmanager
async def open_rig(tmp_path: Path, factory_root: Path) -> AsyncIterator[TriggerChainRig]:
    root = seed_repo(tmp_path / 'project').resolve()
    state_dir = tmp_path / 'fused-memory-state'
    state_dir.mkdir()
    server = LiveFusedMemory(root, state_dir)
    clock = FakeClock(datetime.now(UTC).replace(microsecond=0))
    rig = TriggerChainRig(
        root=root,
        factory_root=factory_root,
        server=server,
        scheduler=_scheduler(root, server.url, clock),
        clock=clock,
    )
    try:
        await server.start(lambda: rig.call('get_tasks', project_root=str(root)))
        yield rig
    finally:
        server.stop()
