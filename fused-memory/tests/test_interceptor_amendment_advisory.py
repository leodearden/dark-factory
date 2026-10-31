"""The write-boundary amendment advisory and its task_text_amended event (task 4034).

PRD plans/task-amendment-delivery-prd.md §10 row 13 and its negatives. Each row drives a
public seam (the MCP update_task tool, or submit_task with an injected curator) over a
real SQLite store, event buffer, write journal and ticket store; only the LLM is faked.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from _fm_helpers import _init_git_repo
from shared.task_text_fingerprint import task_text_fingerprint

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.models.reconciliation import ReconciliationEvent
from fused_memory.models.scope import resolve_project_id
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.server.tools import create_mcp_server
from fused_memory.services.write_journal import WriteJournal

ORIG = 'Add a retry loop around the flush.'
NEW = 'Add a bounded retry loop around the flush, capped at three attempts.'
DETAILS = 'Touch only writer.py.'
NEW_DETAILS = 'Touch writer.py and its test.'
AUTHORED_AT = '2026-10-01T00:00:00+00:00'
STAMP = task_text_fingerprint(ORIG, DETAILS)
AGENT = 'claude-test'


@dataclass
class Stack:
    root: str
    backend: SqliteTaskBackend
    interceptor: TaskInterceptor
    event_buffer: EventBuffer
    server: Any

    async def call(self, tool: str, **args: Any) -> Any:
        return await self.server._tool_manager.call_tool(
            tool, {'project_root': self.root, **args},
        )

    async def seed(
        self,
        *,
        title: str = 'Planned task',
        description: str = ORIG,
        status: str = 'pending',
        stamped: bool = True,
    ) -> str:
        metadata = (
            json.dumps({'plan_text_fp': STAMP, 'plan_authored_at': AUTHORED_AT})
            if stamped else None
        )
        created = await self.backend.add_task(
            project_root=self.root,
            title=title,
            description=description,
            details=DETAILS,
            status=status,
            metadata=metadata,
        )
        return str(created['id'])

    async def amended_events(self) -> list[ReconciliationEvent]:
        buffered = await self.event_buffer.peek_buffered(resolve_project_id(self.root), 1000)
        return [event for event in buffered if event.type == 'task_text_amended']


BuildStack = Callable[..., Awaitable[Stack]]


@pytest_asyncio.fixture
async def build_stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[BuildStack]:
    monkeypatch.setenv('DARK_FACTORY_DATA_DIR', str(tmp_path / 'data'))
    closers: list[Callable[[], Awaitable[object]]] = []

    async def _build(*, curator: Any = None, event_queue: Any = None) -> Stack:
        _init_git_repo(tmp_path / 'proj')
        root = str((tmp_path / 'proj').resolve())
        backend = SqliteTaskBackend(TaskmasterConfig(project_root=root))
        await backend.start()
        closers.append(backend.close)
        event_buffer = EventBuffer(db_path=tmp_path / 'events.db', buffer_size_threshold=100)
        await event_buffer.initialize()
        closers.append(event_buffer.close)
        journal = WriteJournal(tmp_path / 'wj')
        await journal.initialize()
        closers.append(journal.close)
        ticket_store = TicketStore(tmp_path / 'tickets.db')
        await ticket_store.initialize()
        closers.append(ticket_store.close)
        interceptor = TaskInterceptor(
            backend, None, event_buffer,
            ticket_store=ticket_store, curator=curator, event_queue=event_queue,
        )
        interceptor.set_write_journal(journal)
        closers.append(interceptor.close)
        server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)
        return Stack(
            root=root,
            backend=backend,
            interceptor=interceptor,
            event_buffer=event_buffer,
            server=server,
        )

    yield _build
    for close in reversed(closers):
        await close()


def advisory_for(
    task_id: str,
    *,
    fields_changed: list[str],
    description: str = NEW,
    details: str = DETAILS,
    will_replan: bool = True,
) -> dict[str, Any]:
    return {
        'task_id': task_id,
        'plan_authored_at': AUTHORED_AT,
        'fields_changed': fields_changed,
        'will_replan': will_replan,
        'plan_text_fp_before': STAMP,
        'task_text_fp_after': task_text_fingerprint(description, details),
    }


def payload_of(event: ReconciliationEvent) -> dict[str, Any]:
    return {key: value for key, value in event.payload.items() if key != '_project_root'}


@pytest.mark.asyncio
async def test_a_description_amendment_is_advised_and_journalled(build_stack: BuildStack) -> None:
    stack = await build_stack()
    task_id = await stack.seed()

    result = await stack.call('update_task', id=task_id, description=NEW, agent_id=AGENT)

    expected = advisory_for(task_id, fields_changed=['description'])
    assert result['amendment_advisory'] == expected
    assert result['updated_task']['description'] == NEW
    [event] = await stack.amended_events()
    assert payload_of(event) == {**expected, 'writer_source': 'update_task', 'agent_id': AGENT}


@pytest.mark.asyncio
async def test_description_and_details_together_are_both_reported(build_stack: BuildStack) -> None:
    stack = await build_stack()
    task_id = await stack.seed()

    result = await stack.call(
        'update_task', id=task_id, description=NEW, details=NEW_DETAILS, agent_id=AGENT,
    )

    assert result['amendment_advisory'] == advisory_for(
        task_id, fields_changed=['description', 'details'], details=NEW_DETAILS,
    )


@pytest.mark.asyncio
async def test_a_metadata_write_on_an_already_drifted_task_is_not_advised(
    build_stack: BuildStack,
) -> None:
    stack = await build_stack()
    task_id = await stack.seed(description=NEW)

    result = await stack.call(
        'update_task', id=task_id, metadata={'x_note': 'metadata only'}, agent_id=AGENT,
    )

    assert result['updated'] is True, result
    assert 'amendment_advisory' not in result
    assert await stack.amended_events() == []


@pytest.mark.asyncio
async def test_an_unstamped_task_is_not_advised(build_stack: BuildStack) -> None:
    stack = await build_stack()
    task_id = await stack.seed(stamped=False)

    result = await stack.call('update_task', id=task_id, description=NEW, agent_id=AGENT)

    assert result['updated_task']['description'] == NEW
    assert 'amendment_advisory' not in result
    assert await stack.amended_events() == []


@pytest.mark.asyncio
async def test_a_whitespace_only_reflow_is_not_advised(build_stack: BuildStack) -> None:
    stack = await build_stack()
    task_id = await stack.seed()
    reflowed = '  Add a retry\n loop around   the flush.  '

    result = await stack.call('update_task', id=task_id, description=reflowed, agent_id=AGENT)

    assert result['updated_task']['description'] == reflowed
    assert 'amendment_advisory' not in result
    assert await stack.amended_events() == []


@pytest.mark.asyncio
async def test_amending_a_done_task_says_it_will_not_replan(build_stack: BuildStack) -> None:
    stack = await build_stack()
    task_id = await stack.seed(status='done')

    result = await stack.call('update_task', id=task_id, description=NEW, agent_id=AGENT)

    assert result['amendment_advisory'] == advisory_for(
        task_id, fields_changed=['description'], will_replan=False,
    )


@pytest.mark.asyncio
async def test_a_replayed_write_returns_the_advisory_without_re_journalling(
    build_stack: BuildStack,
) -> None:
    stack = await build_stack()
    task_id = await stack.seed()

    first = await stack.call(
        'update_task', id=task_id, description=NEW, agent_id=AGENT, client_op_id='op-1',
    )
    replayed = await stack.call(
        'update_task', id=task_id, description=NEW, agent_id=AGENT, client_op_id='op-1',
    )

    expected = advisory_for(task_id, fields_changed=['description'])
    assert first['amendment_advisory'] == expected
    assert replayed['amendment_advisory'] == expected
    assert len(await stack.amended_events()) == 1
