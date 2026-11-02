"""A hermetic fused-memory task stack behind the MCP tool surface, for boundary tests.

SqliteTaskBackend + EventBuffer + TicketStore + a real TaskInterceptor, served
by ``create_mcp_server``, so a test drives ``server._tool_manager.call_tool``
against persisted rows. Import it as ``from _hermetic_task_stack import
hermetic_task_stack``.

``config.taskmaster`` is cleared so the curator's one-shot corpus backfill
never fires, which keeps the stack hermetic whatever the ambient
fused-memory/config/config.yaml says. That clearing is what sets it apart from
``test_task_tools.py::real_task_stack``, a fixture that passes no config.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import FusedMemoryConfig, TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.server.tools import create_mcp_server


@contextlib.asynccontextmanager
async def hermetic_task_stack(tmp_path: Path) -> AsyncIterator[tuple[Any, TaskInterceptor]]:
    """Yield ``(server, interceptor)`` over a fresh store rooted at *tmp_path*."""
    backend = SqliteTaskBackend(TaskmasterConfig(project_root=str(tmp_path)))
    await backend.start()
    event_buffer = EventBuffer(db_path=tmp_path / 'stack_eb.db', buffer_size_threshold=100)
    await event_buffer.initialize()
    ticket_store = TicketStore(tmp_path / 'stack_tickets.db')
    await ticket_store.initialize()
    config = FusedMemoryConfig()
    config.taskmaster = None
    interceptor = TaskInterceptor(backend, None, event_buffer, config, ticket_store=ticket_store)
    try:
        yield create_mcp_server(AsyncMock(), task_interceptor=interceptor), interceptor
    finally:
        await interceptor.close()
        await event_buffer.close()
        await backend.close()
