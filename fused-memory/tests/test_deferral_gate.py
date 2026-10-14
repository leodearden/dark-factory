"""Boundary rows for metadata.deferral at the status choke point (task 6525).

The rows are plans/deferral-flip-condition-prd.md §6. Each one drives the public MCP
tools against a real SQLite store, a real write journal, a real event buffer and real
processes; nothing private is patched.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.models.reconciliation import ReconciliationEvent
from fused_memory.models.scope import resolve_project_id
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.targeted import TargetedReconciler
from fused_memory.server.tools import create_mcp_server
from fused_memory.services.write_journal import WriteJournal

_EVENT_PEEK_LIMIT = 1000


class RecordingReconciler(TargetedReconciler):
    """A targeted reconciler that records each request instead of reconciling."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def reconcile_task(
        self,
        task_id: str,
        transition: str,
        project_id: str,
        task_before: dict,
        project_root: str,
        reopen_reason: str | None = None,
    ) -> dict:
        self.calls.append({'task_id': task_id, 'transition': transition})
        return {}


@dataclass
class Stack:
    root: str
    backend: SqliteTaskBackend
    interceptor: TaskInterceptor
    journal: WriteJournal
    event_buffer: EventBuffer
    reconciler: RecordingReconciler
    server: Any
    ticket_store: TicketStore

    async def call(self, tool: str, **args: Any) -> Any:
        return await self.server._tool_manager.call_tool(
            tool, {'project_root': self.root, **args},
        )

    async def seed(self, status: str, title: str) -> str:
        created = await self.backend.add_task(
            project_root=self.root, title=title, status=status,
        )
        return str(created['id'])

    async def read(self, task_id: str) -> dict:
        return await self.interceptor.get_task(task_id, self.root)

    async def journal_rows(self, operation: str) -> list[dict]:
        return [
            row for row in await self.journal.get_ops_since('1970-01-01', limit=10_000)
            if row['operation'] == operation
        ]

    async def events(self, event_type: str) -> list[ReconciliationEvent]:
        buffered = await self.event_buffer.peek_buffered(
            resolve_project_id(self.root), _EVENT_PEEK_LIMIT,
        )
        return [event for event in buffered if event.type == event_type]


def _git(root: Path, *args: str) -> None:
    subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True)


def _git_main_checkout(root: Path) -> str:
    """A real main checkout, so the tool layer's project_root resolution runs unpatched."""
    root.mkdir()
    _git(root, 'init', '-q', '-b', 'main')
    _git(root, 'config', 'user.email', 'deferral-gate@test')
    _git(root, 'config', 'user.name', 'deferral-gate')
    (root / 'README').write_text('deferral gate fixture\n')
    _git(root, 'add', 'README')
    _git(root, 'commit', '-q', '-m', 'seed')
    return str(root.resolve())


@pytest_asyncio.fixture
async def stack(tmp_path: Path) -> AsyncIterator[Stack]:
    root = _git_main_checkout(tmp_path / 'proj')
    backend = SqliteTaskBackend(TaskmasterConfig(project_root=root))
    await backend.start()
    event_buffer = EventBuffer(db_path=tmp_path / 'events.db', buffer_size_threshold=100)
    await event_buffer.initialize()
    journal = WriteJournal(tmp_path / 'wj')
    await journal.initialize()
    ticket_store = TicketStore(tmp_path / 'tickets.db')
    await ticket_store.initialize()
    reconciler = RecordingReconciler()
    interceptor = TaskInterceptor(backend, reconciler, event_buffer, ticket_store=ticket_store)
    interceptor.set_write_journal(journal)
    server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)
    built = Stack(
        root=root,
        backend=backend,
        interceptor=interceptor,
        journal=journal,
        event_buffer=event_buffer,
        reconciler=reconciler,
        server=server,
        ticket_store=ticket_store,
    )
    try:
        yield built
    finally:
        await interceptor.close()
        await journal.close()
        await event_buffer.close()
        await backend.close()


def _spawn(argv: list[str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _reap(process: subprocess.Popen[bytes]) -> None:
    process.kill()
    process.wait()


def _sleep_binary() -> str:
    found = shutil.which('sleep')
    assert found is not None, 'the holder fixtures need a sleep binary on PATH'
    return found


@pytest.fixture
def claude_holder(tmp_path: Path) -> Iterator[int]:
    """A live process whose /proc/<pid>/comm reads 'claude'."""
    binary = tmp_path / 'claude'
    shutil.copy(_sleep_binary(), binary)
    binary.chmod(0o755)
    process = _spawn([str(binary), '600'])
    try:
        yield process.pid
    finally:
        _reap(process)


@pytest.fixture
def other_process() -> Iterator[int]:
    """A live process that is not the Claude CLI."""
    process = _spawn([_sleep_binary(), '600'])
    try:
        yield process.pid
    finally:
        _reap(process)


@pytest.fixture
def dead_pid() -> int:
    """The pid of a process that has exited and been reaped."""
    true_binary = shutil.which('true')
    assert true_binary is not None
    process = _spawn([true_binary])
    process.wait()
    return process.pid
