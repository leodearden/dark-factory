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
import json
import os
import socket
import subprocess
import sys
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar
from unittest.mock import AsyncMock

import uvicorn
from _git_fixtures import seed_repo
from _trigger_chain_doc_shapes import concretize, snippet_placeholders, snippet_submit_calls
from escalation.action_effects import effect_for
from escalation.models import Escalation, ReviewDueState
from escalation.queue import EscalationQueue
from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import FusedMemoryConfig as FusedMemoryServerConfig
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.server.tools import create_mcp_server
from shared.mcp_envelope import parse_tool_result
from shared.task_statuses import TERMINAL

from orchestrator.config import (
    PREDICATE_RECHECK_STALL_CAP,
    FusedMemoryConfig,
    OrchestratorConfig,
    PsiAdmissionConfig,
)
from orchestrator.deterministic_runner import DETERMINISTIC_AGENT_ROLE, DeterministicRunner
from orchestrator.event_store import EventStore, EventType
from orchestrator.scheduler import Scheduler, TaskAssignment
from orchestrator.workflow import WorkflowOutcome

PROJECT_ID = 'lambda_fixture'
INTERIM_DELAY_SECS = 259200

_READY_TIMEOUT_S = 60.0
_READY_POLL_S = 0.05
_TOOL_TIMEOUT_S = 30.0
_SCRIPT_TIMEOUT_S = 120.0
_SCHEDULER_CLOSE_TIMEOUT_S = 5.0
CHECK_SCRIPT_RELPATH = Path('scripts/check_run_completion.py')

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
        except BaseException as exc:
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

    async def start(self, probe: Callable[[], Awaitable[object]]) -> None:
        """Serve, and return once *probe* completes a real tool call."""
        self._thread.start()
        deadline = asyncio.get_running_loop().time() + _READY_TIMEOUT_S
        last_error: BaseException | None = None
        while self._thread.is_alive() and (remaining := deadline - asyncio.get_running_loop().time()) > 0:
            try:
                async with asyncio.timeout(remaining):
                    await probe()
                return
            except Exception as exc:
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
        """Ask the server loop to exit, wait (bounded) for its thread, and raise whatever ended it."""
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._request_exit)
        self._thread.join(timeout=self._config.server.graceful_shutdown_timeout + 10)
        self._listener.close()
        assert not self._thread.is_alive(), f'fused-memory serving thread for {self.url} did not stop'
        if self._serve_error is not None:
            raise self._serve_error


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
class ScriptRun:
    rc: int
    lines: tuple[str, ...]
    stderr: str


class StatusWriteFailing:
    """*scheduler*, except that each ``set_task_status`` to *status* raises as an exhausted transport retry does."""

    def __init__(self, scheduler: Scheduler, status: str) -> None:
        self._scheduler = scheduler
        self._status = status
        self.refused: list[tuple[str, str]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._scheduler, name)

    async def set_task_status(self, task_id: str, status: str, **kwargs: Any) -> None:
        if status == self._status:
            self.refused.append((task_id, status))
            raise RuntimeError('connection severed')
        await self._scheduler.set_task_status(task_id, status, **kwargs)


@dataclass(frozen=True)
class Chain:
    completion_id: str
    human_id: str


@dataclass(frozen=True)
class TriggerChainRig:
    root: Path
    factory_root: Path
    server: LiveFusedMemory
    scheduler: Scheduler
    queue: EscalationQueue
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
        placeholders = snippet_placeholders(
            skill=skill, run_id=run_id, project_root=self.root, project_id=self.project_id, now=self.clock()
        )
        calls = [concretize(call, placeholders) for call in snippet_submit_calls(self.factory_root)]
        (call,) = [call for call in calls if call['metadata']['trigger_chain']['role'] == role]
        return call

    async def submit(self, call: dict[str, Any]) -> str:
        """File a concretized snippet call; its task id."""
        reply = await self.call('submit_task', **call)
        if 'task_id' not in reply:
            raise AssertionError(f'submit_task refused {call["title"]!r}: {reply}')
        return str(reply['task_id'])

    async def call_ok(self, tool: str, **arguments: Any) -> dict[str, Any]:
        """*tool*'s payload, which must not be a refusal."""
        reply = await self.call(tool, **arguments)
        if 'error' in reply:
            raise AssertionError(f'{tool} refused: {reply}')
        return reply

    async def task(self, task_id: str) -> dict[str, Any]:
        return await self.call_ok('get_task', id=task_id, project_root=str(self.root))

    async def file_final_chain(self, run_id: str, skill: str = 'review') -> Chain:
        """The snippet's §3 final form: both gates, the dependency, then one commit."""
        completion_id = await self.submit(self.snippet_call('completion_gate', run_id, skill))
        human_id = await self.submit(self.snippet_call('human_gate', run_id, skill))
        await self.call_ok('add_dependency', id=human_id, depends_on=completion_id, project_root=str(self.root))
        await self.call_ok('commit_planning', project_root=str(self.root), task_ids=f'{completion_id},{human_id}')
        return Chain(completion_id, human_id)

    async def file_interim_gate(self, run_id: str, deps: tuple[str, ...], skill: str = 'review') -> str:
        """The snippet's §4 interim form: the human gate, delayed, held until its dependencies exist."""
        call = self.snippet_call('human_gate', run_id, skill)
        call['metadata']['milestone'] = {'mode': 'delayed', 'after_secs': INTERIM_DELAY_SECS}
        human_id = await self.submit(call)
        for dep in deps:
            await self.call_ok('add_dependency', id=human_id, depends_on=dep, project_root=str(self.root))
        await self.call_ok('commit_planning', project_root=str(self.root), task_ids=human_id)
        return human_id

    async def seed_run(self, run_id: str, *findings: tuple[str, str]) -> tuple[str, ...]:
        """File *run_id*'s findings as (priority, status) pairs, the way the lanes that own them write."""
        return tuple(
            [await self.server.run(self._seed_finding(run_id, n, *finding)) for n, finding in enumerate(findings)]
        )

    async def _seed_finding(self, run_id: str, n: int, priority: str, status: str) -> str:
        backend = self.server.backend
        added = await backend.add_task(
            project_root=str(self.root),
            title=f'{run_id} finding {n}',
            priority=priority,
            metadata=json.dumps({'x_finding_run': run_id}),
        )
        if status != 'pending':
            await backend.set_task_status(added['id'], status, str(self.root))
        return str(added['id'])

    async def set_finding_status(self, task_id: str, status: str) -> None:
        await self.server.run(self.server.backend.set_task_status(task_id, status, str(self.root)))

    def runner(
        self,
        *,
        stall_cap: int = PREDICATE_RECHECK_STALL_CAP,
        scheduler: Scheduler | StatusWriteFailing | None = None,
    ) -> DeterministicRunner:
        """A runner as the harness builds one per slot, on the fake clock, running the real script."""
        return DeterministicRunner(
            scheduler=self.scheduler if scheduler is None else scheduler,
            escalation_queue=self.queue,
            predicate_recheck_stall_cap=stall_cap,
            clock=self.clock,
        )

    async def dispatch(
        self, *, runner: DeterministicRunner | None = None
    ) -> tuple[TaskAssignment | None, WorkflowOutcome | None]:
        """One tick; a deterministic assignment runs and is released with requeued=False, as Harness._run_slot's deterministic return leaves it."""
        assignment = await self.scheduler.acquire_next()
        if assignment is None:
            return None, None
        try:
            if not Scheduler.is_deterministic(assignment.task):
                raise AssertionError(
                    f'task {assignment.task_id} is not deterministic, so this rig cannot run it; '
                    'seed an open finding as one another lane holds (in-progress), not pending'
                )
            outcome = await (runner or self.runner()).run(assignment)
        finally:
            self.scheduler.release(assignment.task_id)
        return assignment, outcome

    async def resolve(self, escalation: Escalation, action: str, resolution: str) -> None:
        """Resolve *escalation* with *action*, then apply its Table B status effect as the harness does."""
        effect = effect_for(action, escalation.level, escalation.category)
        if effect is None:
            raise AssertionError(f'{action!r} is not a resolve action for {escalation.category}')
        self.queue.resolve(escalation.id, resolution, resolution_action=action, resolved_by='lambda-e2e')
        if effect.target_status is not None:
            await self.scheduler.set_task_status(escalation.task_id, effect.target_status)

    async def set_status(self, task_id: str, status: str) -> None:
        await self.call_ok('set_task_status', id=task_id, status=status, project_root=str(self.root))

    async def supersede(self, old: Chain, new_run_id: str, *, launched_from: str | None) -> None:
        """Retire *old* for run *new_run_id* in the snippet's §5 order: mark both, the human gate, then the completion gate."""
        for task_id in (old.human_id, old.completion_id):
            chain = (await self.task(task_id))['metadata']['trigger_chain']
            await self.call_ok(
                'update_task',
                id=task_id,
                project_root=str(self.root),
                metadata={'trigger_chain': {**chain, 'superseded_by': new_run_id}},
            )
        await self._retire_human_gate(old.human_id, new_run_id, launched_from)
        await self._retire_completion_gate(old.completion_id, new_run_id)

    async def _retire_human_gate(self, task_id: str, new_run_id: str, launched_from: str | None) -> None:
        status = (await self.task(task_id))['status']
        if status in ('pending', 'deferred'):
            await self.set_status(task_id, 'cancelled')
            return
        if status != 'blocked':
            raise AssertionError(f'human gate {task_id} is {status!r}; §5 retires only pending, deferred or blocked')
        escalation = self._one_pending(task_id)
        answered = escalation.id == launched_from
        await self.resolve(
            escalation, 'resume', f'ran as {new_run_id}' if answered else f'superseded by {new_run_id}'
        )
        assignment, outcome = await self.dispatch()
        if assignment is None or assignment.task_id != task_id or outcome != WorkflowOutcome.DONE:
            raise AssertionError(f'resuming human gate {task_id} dispatched {assignment!r} -> {outcome!r}')

    async def _retire_completion_gate(self, task_id: str, new_run_id: str) -> None:
        status = (await self.task(task_id))['status']
        if status == 'pending':
            await self.set_status(task_id, 'cancelled')
        elif status == 'blocked':
            await self.resolve(self._one_pending(task_id), 'abandon', f'superseded by {new_run_id}')
        elif status not in TERMINAL:
            raise AssertionError(f'completion gate {task_id} is {status!r}; §5 re-reads it later')

    def _one_pending(self, task_id: str) -> Escalation:
        (escalation,) = self.own_escalations(task_id, status='pending')
        return escalation

    def pending_report(self, *extra_args: str) -> ScriptRun:
        """The factory script's ``--pending`` report on this project, read from the live server."""
        done = subprocess.run(
            [
                sys.executable,
                '-s',
                str(self.factory_root / CHECK_SCRIPT_RELPATH),
                '--pending',
                '--project-root',
                str(self.root),
                *extra_args,
            ],
            env={**os.environ, 'FUSED_MEMORY_MCP_URL': self.url},
            capture_output=True,
            text=True,
            timeout=_SCRIPT_TIMEOUT_S,
        )
        return ScriptRun(done.returncode, tuple(done.stdout.splitlines()), done.stderr)

    def status_write_failing(self, status: str) -> StatusWriteFailing:
        """This rig's Scheduler, except that its writes of *status* are lost."""
        return StatusWriteFailing(self.scheduler, status)

    def own_escalations(self, task_id: str, status: str | None = None) -> list[Escalation]:
        return self.queue.get_by_task(task_id, status=status, agent_role=DETERMINISTIC_AGENT_ROLE)

    def events(self, event_type: EventType) -> list[dict]:
        event_store = self.scheduler.event_store
        assert event_store is not None
        return event_store.fetch_events_by_type(event_type)


def review_due(escalation: Escalation) -> ReviewDueState:
    """A ``review_due`` escalation's payload; any other escalation fails the test."""
    if escalation.review_due is None:
        raise AssertionError(f'{escalation.id} ({escalation.category}) carries no review_due payload')
    return escalation.review_due


def _scheduler(root: Path, url: str, clock: FakeClock, state_dir: Path) -> Scheduler:
    config = OrchestratorConfig(
        project_root=root,
        fused_memory=FusedMemoryConfig(url=url, project_id=PROJECT_ID),
        max_per_module=1,
        psi_admission=PsiAdmissionConfig(enabled=False),
    )
    scheduler = Scheduler(
        config,
        event_store=EventStore(state_dir / 'orchestrator-events.db', run_id='lambda-e2e'),
        wall_time_source=clock,
        time_source=clock.timestamp,
    )
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
        scheduler=_scheduler(root, server.url, clock, state_dir),
        queue=EscalationQueue(tmp_path / 'escalations'),
        clock=clock,
    )
    try:
        await server.start(lambda: rig.call('get_tasks', project_root=str(root)))
        yield rig
    finally:
        try:
            await asyncio.wait_for(rig.scheduler.aclose(), timeout=_SCHEDULER_CLOSE_TIMEOUT_S)
        finally:
            server.stop()
