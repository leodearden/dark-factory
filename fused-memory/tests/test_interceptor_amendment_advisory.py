"""The write-boundary amendment advisory and its task_text_amended event (task 4034).

PRD plans/task-amendment-delivery-prd.md §10 row 13 and its negatives. Each row drives a
public seam (the MCP update_task tool, or submit_task with an injected curator) over a
real SQLite store, event buffer, write journal and ticket store; only the LLM is faked.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from _curator_helpers import answer_each_with, curator_returning
from _fm_git_repos import _init_git_repo
from _fm_helpers import submit_and_resolve
from shared.task_text_fingerprint import task_text_fingerprint

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.curator_types import CuratorDecision, RewrittenTask
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
RECON_AGENT = 'recon-stage-task_knowledge_sync'
AUDIT_TRAIL = '\n\n'.join(
    [ORIG, *(f'Cycle {n:02d} relay.\n' + 'evidence unchanged; ' * 60 for n in range(20))],
)


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
async def build_stack(tmp_path: Path) -> AsyncIterator[BuildStack]:
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


class VerbatimArchive:
    """An audit-trail archive that reads back exactly what it stored."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def write(self, *, project_id: str, content: str, metadata: dict[str, Any]) -> str:
        memory_id = str(uuid.uuid4())
        self.store[memory_id] = content
        return memory_id

    async def read(self, *, project_id: str, memory_id: str) -> str | None:
        return self.store.get(memory_id)

    async def discard(self, *, project_id: str, memory_id: str) -> None:
        self.store.pop(memory_id, None)


@pytest.mark.asyncio
async def test_a_recon_metadata_write_whose_rotation_rewrote_the_description_is_advised(
    build_stack: BuildStack,
) -> None:
    stack = await build_stack()
    stack.interceptor.set_audit_trail_archive(VerbatimArchive())
    task_id = await stack.seed(description=AUDIT_TRAIL)

    result = await stack.call(
        'update_task', id=task_id, metadata={'x_note': 'cycle 21'}, agent_id=RECON_AGENT,
    )

    assert result['audit_trail_rotation']['description_rewritten'] is True, result
    committed = await stack.backend.get_task(task_id, stack.root)
    expected = advisory_for(
        task_id, fields_changed=['description'], description=committed['description'],
    )
    assert result['amendment_advisory'] == expected
    [event] = await stack.amended_events()
    assert payload_of(event) == {
        **expected, 'writer_source': 'update_task', 'agent_id': RECON_AGENT,
    }


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


COMBINED_TITLE = 'Planned task, folded with its duplicate'


def combine_into(target_id: str) -> CuratorDecision:
    return CuratorDecision(
        action='combine',
        target_id=target_id,
        target_fingerprint='Planned task',
        rewritten_task=RewrittenTask(
            title=COMBINED_TITLE,
            description=NEW,
            details='Merged with the duplicate submission.',
            files_to_modify=['a.py'],
            priority='medium',
        ),
        justification='duplicate of the planned task',
    )


async def combine_a_duplicate_into_a_new_stack(
    build_stack: BuildStack, *, stamped: bool, event_queue: Any = None,
) -> tuple[Stack, str, dict[str, Any]]:
    curator = curator_returning()
    stack = await build_stack(curator=curator, event_queue=event_queue)
    target_id = await stack.seed(stamped=stamped)
    answer_each_with(curator, AsyncMock(return_value=combine_into(target_id)))
    result = await submit_and_resolve(
        stack.interceptor, stack.root,
        title='Duplicate of the planned task', description='candidate description',
    )
    return stack, target_id, result


@pytest.mark.asyncio
async def test_a_curator_combine_into_a_stamped_task_is_journalled(
    build_stack: BuildStack,
) -> None:
    stack, target_id, result = await combine_a_duplicate_into_a_new_stack(
        build_stack, stamped=True,
    )

    assert result['action'] == 'combine', result
    stored = await stack.backend.get_task(target_id, stack.root)
    [event] = await stack.amended_events()
    assert payload_of(event) == {
        'task_id': target_id,
        'plan_authored_at': AUTHORED_AT,
        'fields_changed': ['title', 'description', 'details'],
        'will_replan': True,
        'plan_text_fp_before': STAMP,
        'task_text_fp_after': task_text_fingerprint(stored['description'], stored['details']),
        'writer_source': 'curator_combine',
        'agent_id': None,
    }


@pytest.mark.asyncio
async def test_a_curator_combine_into_an_unstamped_task_is_not_journalled(
    build_stack: BuildStack,
) -> None:
    stack, _target_id, result = await combine_a_duplicate_into_a_new_stack(
        build_stack, stamped=False,
    )

    assert result['action'] == 'combine', result
    assert await stack.amended_events() == []


class AmendmentJournalOutage:
    """An event queue whose task_text_amended enqueue raises; every other event is accepted."""

    def __init__(self) -> None:
        self.events: list[ReconciliationEvent] = []

    def enqueue(self, event: ReconciliationEvent) -> bool:
        self.events.append(event)
        if event.type == 'task_text_amended':
            raise RuntimeError('event queue unavailable')
        return True


async def amend_through_update_task(build_stack: BuildStack, queue: AmendmentJournalOutage) -> str:
    stack = await build_stack(event_queue=queue)
    task_id = await stack.seed()

    result = await stack.call('update_task', id=task_id, description=NEW, agent_id=AGENT)

    assert result['amendment_advisory'] == advisory_for(task_id, fields_changed=['description'])
    stored = await stack.backend.get_task(task_id, stack.root)
    assert stored['description'] == NEW
    return task_id


async def amend_through_a_combine(build_stack: BuildStack, queue: AmendmentJournalOutage) -> str:
    stack, target_id, result = await combine_a_duplicate_into_a_new_stack(
        build_stack, stamped=True, event_queue=queue,
    )

    assert result['action'] == 'combine', result
    stored = await stack.backend.get_task(target_id, stack.root)
    assert (stored['title'], stored['description']) == (COMBINED_TITLE, NEW)
    listing = await stack.backend.get_tasks(project_root=stack.root)
    assert [str(task['id']) for task in listing['tasks']] == [target_id]
    return target_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('amend', 'writer_source'),
    [(amend_through_update_task, 'update_task'), (amend_through_a_combine, 'curator_combine')],
    ids=['update_task', 'curator_combine'],
)
async def test_a_failed_amendment_journal_does_not_fail_the_landed_write(
    build_stack: BuildStack,
    caplog: pytest.LogCaptureFixture,
    amend: Callable[[BuildStack, AmendmentJournalOutage], Awaitable[str]],
    writer_source: str,
) -> None:
    queue = AmendmentJournalOutage()

    with caplog.at_level(logging.WARNING):
        task_id = await amend(build_stack, queue)

    assert [event.type for event in queue.events if event.type == 'task_text_amended'] == [
        'task_text_amended',
    ]
    warnings = [
        record.getMessage() for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert any(task_id in message and writer_source in message for message in warnings), warnings
