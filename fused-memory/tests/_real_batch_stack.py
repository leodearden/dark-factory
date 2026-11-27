"""A real batch-curation stack: a real TaskCurator over a real SQLite store, ticket store and event buffer.

The curator answers through the injected *invoke*. Its candidate preparation is stubbed to
``_curator_helpers.prepared`` (no embedding) and its corpus writes (``record_task``,
``reembed_task``) are AsyncMocks; everything from the batch call on is the production path. Drain
several submissions as one worker batch with ``_curator_helpers.submit_as_one_batch``:

    async with real_batch_stack(tmp_path, invoke) as stack:
        tickets = await submit_as_one_batch(
            stack.interceptor, stack.curator, filings, project_root=stack.root,
        )

The store's project root is *tmp_path*. Every component is closed when the block exits, so read
the store inside it.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

from _curator_helpers import escalator_double, make_config, prepared

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.curator_types import CandidateTask, PreparedCandidate
from fused_memory.middleware.task_curator import TaskCurator
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer


@dataclass(frozen=True)
class RealBatchStack:
    root: str
    backend: SqliteTaskBackend
    curator: TaskCurator
    interceptor: TaskInterceptor


@contextlib.asynccontextmanager
async def real_batch_stack(tmp_path: Path, invoke: AsyncMock) -> AsyncIterator[RealBatchStack]:
    """Yield a fresh stack rooted at *tmp_path* whose curator answers through *invoke*."""
    async with contextlib.AsyncExitStack() as cleanup:
        backend = SqliteTaskBackend(TaskmasterConfig(project_root=str(tmp_path)))
        await backend.start()
        cleanup.push_async_callback(backend.close)
        event_buffer = EventBuffer(db_path=tmp_path / 'events.db', buffer_size_threshold=100)
        await event_buffer.initialize()
        cleanup.push_async_callback(event_buffer.close)
        ticket_store = TicketStore(tmp_path / 'tickets.db')
        await ticket_store.initialize()
        cleanup.push_async_callback(ticket_store.close)
        curator = TaskCurator(
            config=make_config(), taskmaster=backend, escalator=escalator_double(), invoke=invoke,
        )
        curator.prepare_candidate = AsyncMock(side_effect=_bundle_for)
        curator.record_task = AsyncMock()
        curator.reembed_task = AsyncMock()
        interceptor = TaskInterceptor(
            backend, None, event_buffer, ticket_store=ticket_store, curator=curator,
        )
        cleanup.push_async_callback(interceptor.close)
        yield RealBatchStack(
            root=str(tmp_path), backend=backend, curator=curator, interceptor=interceptor,
        )


async def _bundle_for(candidate: CandidateTask, *_args: Any) -> PreparedCandidate:
    return prepared(candidate.title, candidate=candidate)
