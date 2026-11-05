"""Boundary rows for metadata.declared_scope at the submit boundary (task 8413).

The rows are plans/follow-up-filing-shape-prd.md §5 rows 6 and 20 and §3 decision 7. Each one
drives the public MCP tools against a real SQLite store, ticket store and interceptor.
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
from shared.declared_scope import derive_declared_scope

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import CuratorConfig, FusedMemoryConfig, TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.server.tools import create_mcp_server


@pytest.fixture(autouse=True)
def passthrough_main_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    """This suite's project roots are tmp_path directories, not git checkouts."""
    monkeypatch.setattr('fused_memory.server.tools.resolve_main_checkout', lambda p: str(p))


@dataclass
class Stack:
    root: str
    backend: SqliteTaskBackend
    server: Any

    async def call(self, tool: str, **args: Any) -> Any:
        return await self.server._tool_manager.call_tool(tool, {'project_root': self.root, **args})

    async def task_count(self) -> int:
        return len((await self.call('get_tasks'))['tasks'])


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
        yield Stack(root=str(tmp_path / 'proj'), backend=backend, server=server)
    finally:
        await interceptor.close()
        await event_buffer.close()
        await backend.close()


_FORGED = derive_declared_scope(['pkg/a.py'], 'forged', '', basis='submit').model_dump()


def _assert_write_authority(result: Any, task_id: str | None) -> None:
    assert result['success'] is False, result
    assert result['error'] == 'declared_scope_write_authority', result
    assert result['error_type'] == 'DeclaredScopeWriteAuthorityError', result
    assert result['path'] == 'metadata.declared_scope', result
    assert result['task_id'] == task_id, result


# Row 6: a submission that supplies declared_scope is refused and nothing is filed.


@pytest.mark.asyncio
@pytest.mark.parametrize('planning_mode', [False, True])
@pytest.mark.parametrize('as_json', [False, True], ids=['object', 'json-string'])
async def test_submit_task_refuses_a_caller_declared_scope(
    stack: Stack, planning_mode: bool, as_json: bool,
) -> None:
    metadata: dict[str, Any] = {'declared_scope': _FORGED}
    before = await stack.task_count()

    result = await stack.call(
        'submit_task', title='row 6 forged scope', description='forged declared_scope',
        planning_mode=planning_mode, metadata=json.dumps(metadata) if as_json else metadata,
    )

    _assert_write_authority(result, None)
    assert 'ticket' not in result
    assert await stack.task_count() == before
