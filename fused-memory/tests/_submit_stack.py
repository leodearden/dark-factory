"""A real submit stack driven through the public MCP tools: SQLite store, ticket store, interceptor.

Import the fixtures in the explicit re-export form so pytest registers them in the importing module
(the autouse ``passthrough_main_checkout`` included):

    from _submit_stack import Stack
    from _submit_stack import curator_creates as curator_creates
    from _submit_stack import passthrough_main_checkout as passthrough_main_checkout
    from _submit_stack import stack as stack
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import CuratorConfig, FusedMemoryConfig, TaskmasterConfig
from fused_memory.middleware.curator_types import CuratorDecision, PreparedCandidate
from fused_memory.middleware.task_curator import TaskCurator
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.server.tools import create_mcp_server


@pytest.fixture(autouse=True)
def passthrough_main_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stack's project roots are tmp_path directories, not git checkouts."""
    monkeypatch.setattr('fused_memory.server.tools.resolve_main_checkout', lambda p: str(p))


@dataclass
class Stack:
    root: str
    backend: SqliteTaskBackend
    interceptor: TaskInterceptor
    server: Any

    async def call(self, tool: str, **args: Any) -> Any:
        return await self.server._tool_manager.call_tool(tool, {'project_root': self.root, **args})

    async def task_count(self) -> int:
        return len((await self.call('get_tasks'))['tasks'])

    async def metadata(self, task_id: str) -> dict[str, Any]:
        result = await self.call('get_task', id=task_id)
        assert 'error' not in result, result
        return result['metadata']

    async def seed(self, title: str, metadata: dict[str, Any]) -> str:
        """A row written straight to the store, past every submit-boundary guard."""
        created = await self.backend.add_task(
            project_root=self.root, title=title, metadata=json.dumps(metadata),
        )
        return str(created['id'])

    async def file_normal(self, **args: Any) -> str:
        submitted = await self.call('submit_task', **args)
        assert 'ticket' in submitted, submitted
        resolved = await self.call('resolve_ticket', ticket=submitted['ticket'], timeout_seconds=30)
        assert resolved.get('status') == 'created', resolved
        return str(resolved['task_id'])

    async def file_planned(self, **args: Any) -> str:
        result = await self.call('submit_task', planning_mode=True, **args)
        assert 'error' not in result, result
        return str(result['task_id'])


@pytest_asyncio.fixture
async def stack(tmp_path: Path) -> AsyncIterator[Stack]:
    backend = SqliteTaskBackend(TaskmasterConfig(project_root=str(tmp_path)), task_metadata_enforce=True)
    await backend.start()
    event_buffer = EventBuffer(db_path=tmp_path / 'events.db', buffer_size_threshold=100)
    await event_buffer.initialize()
    ticket_store = TicketStore(tmp_path / 'tickets.db')
    await ticket_store.initialize()
    config = FusedMemoryConfig()
    config.taskmaster = None
    config.curator = CuratorConfig()
    interceptor = TaskInterceptor(backend, None, event_buffer, config, ticket_store=ticket_store)
    server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)
    try:
        yield Stack(root=str(tmp_path / 'proj'), backend=backend, interceptor=interceptor, server=server)
    finally:
        await interceptor.close()
        await event_buffer.close()
        await backend.close()


@pytest.fixture
def curator_creates(monkeypatch: pytest.MonkeyPatch) -> list[PreparedCandidate]:
    """Every curated candidate is created as filed; the candidates the curator was handed are kept."""
    handed: list[PreparedCandidate] = []

    async def _create_each(self, prepared, project_id, project_root):
        handed.extend(prepared)
        return [CuratorDecision(action='create', justification='submit-stack test') for _ in prepared]

    monkeypatch.setattr(TaskCurator, 'curate_prepared', _create_each)
    return handed
