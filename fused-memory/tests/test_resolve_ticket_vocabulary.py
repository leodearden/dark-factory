"""The ticket-status vocabulary and the answers resolve_ticket builds from it (task 6705)."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from _fastmcp_tool_result import tool_result_dict
from _fm_git_repos import _init_git_repo
from _fused_memory_tool_surface import FUSED_MEMORY_TOOL_PREFIX, registered_fused_memory_tools
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import CuratorConfig, TaskmasterConfig
from fused_memory.middleware.curation_deferral import CurationRetryPolicy, DeferReason
from fused_memory.middleware.curator_types import (
    CandidateTask,
    CuratorDecision,
    PreparedCandidate,
)
from fused_memory.middleware.task_curator import TaskCurator
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import (
    OPEN_TICKET_STATUSES,
    TERMINAL_TICKET_STATUSES,
    ResolveReason,
    TicketStatus,
    TicketStore,
)
from fused_memory.reconciliation import recon_self_model
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.prompts.stage2 import build_stage2_system_prompt
from fused_memory.server.tools import create_mcp_server

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

WAIT_SECONDS = 30.0


@pytest_asyncio.fixture
async def store(tmp_path):
    ticket_store = TicketStore(tmp_path / 'tickets.db')
    await ticket_store.initialize()
    yield ticket_store
    await ticket_store.close()


class TestTicketStatusVocabulary:
    def test_status_values(self):
        assert {s.value for s in TicketStatus} == {
            'pending', 'awaiting_curation', 'created', 'combined', 'refused', 'failed', 'cancelled',
        }

    def test_open_and_terminal_partition_the_statuses(self):
        assert frozenset({TicketStatus.PENDING, TicketStatus.AWAITING_CURATION}) == OPEN_TICKET_STATUSES
        assert isinstance(OPEN_TICKET_STATUSES, frozenset)
        assert isinstance(TERMINAL_TICKET_STATUSES, frozenset)
        assert frozenset(TicketStatus) - OPEN_TICKET_STATUSES == TERMINAL_TICKET_STATUSES
        assert OPEN_TICKET_STATUSES.isdisjoint(TERMINAL_TICKET_STATUSES)

    def test_resolve_reason_values(self):
        assert {r.value for r in ResolveReason} == {
            'wait_timeout', 'wait_interrupted', 'store_unavailable', 'unknown_ticket',
            'ticket_store not configured',
        }

    @pytest.mark.asyncio
    async def test_a_stored_row_status_parses_into_the_enum(self, store):
        ticket = await store.submit('dark_factory', '{}')
        assert TicketStatus((await store.get(ticket))['status']) is TicketStatus.PENDING

        await store.mark_awaiting_curation(
            ticket,
            reason=DeferReason.ALL_ACCOUNTS_CAPPED,
            policy=CurationRetryPolicy.from_config(CuratorConfig()),
        )
        assert TicketStatus((await store.get(ticket))['status']) is TicketStatus.AWAITING_CURATION


class _HeldCurator:
    """A curator double: ``prepare_candidate`` waits on ``release``, ``close`` on ``close_gate``."""

    def __init__(self, decision: CuratorDecision | None = None) -> None:
        self.decision = decision or CuratorDecision(action='create', justification='new')
        self.release: asyncio.Event | None = None
        self.close_gate: asyncio.Event | None = None

    async def prepare_candidate(
        self, candidate: CandidateTask, project_id: str, project_root: str,
    ) -> PreparedCandidate:
        if self.release is not None:
            await self.release.wait()
        return PreparedCandidate(candidate=candidate, pool=[], pool_sizes={}, prompt_tokens=0)

    async def curate_prepared(
        self, prepared: list[PreparedCandidate], project_id: str, project_root: str,
    ) -> list[CuratorDecision]:
        return [self.decision for _ in prepared]

    def note_created(self, project_id: str, candidate: CandidateTask, task_id: str) -> None:
        return None

    async def record_task(self, task_id: str, candidate: CandidateTask, project_id: str) -> None:
        return None

    async def reembed_task(self, task_id: str, candidate: CandidateTask, project_id: str) -> None:
        return None

    async def close(self) -> None:
        if self.close_gate is not None:
            await self.close_gate.wait()


@dataclass
class _Stack:
    server: Any
    interceptor: TaskInterceptor
    store: TicketStore
    backend: SqliteTaskBackend
    event_buffer: EventBuffer
    root: str

    async def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return tool_result_dict(await self.server._tool_manager.call_tool(tool, arguments))

    async def submit(self, title: str) -> str:
        answer = await self.call(
            'submit_task',
            {'project_root': self.root, 'title': title, 'description': f'{title} description'},
        )
        assert 'ticket' in answer, answer
        return answer['ticket']

    async def resolve(self, ticket: str, *, timeout_seconds: float, deadline: float) -> dict:
        return await asyncio.wait_for(
            self.call(
                'resolve_ticket',
                {'ticket': ticket, 'project_root': self.root, 'timeout_seconds': timeout_seconds},
            ),
            timeout=deadline,
        )


@pytest_asyncio.fixture
async def make_stack(tmp_path):
    root = tmp_path / 'project'
    _init_git_repo(root)
    backend = SqliteTaskBackend(TaskmasterConfig(project_root=str(root)))
    await backend.start()
    event_buffer = EventBuffer(db_path=tmp_path / 'events.db', buffer_size_threshold=100)
    await event_buffer.initialize()
    built: list[TaskInterceptor] = []

    async def _make(curator: _HeldCurator | None) -> _Stack:
        store = TicketStore(tmp_path / f'tickets_{len(built)}.db')
        await store.initialize()
        interceptor = TaskInterceptor(
            backend, None, event_buffer,
            ticket_store=store, curator=cast(TaskCurator | None, curator),
        )
        built.append(interceptor)
        server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)
        return _Stack(server, interceptor, store, backend, event_buffer, str(root))

    yield _make
    for interceptor in built:
        await interceptor.close()
    await backend.close()
    await event_buffer.close()


def _held_curator() -> _HeldCurator:
    curator = _HeldCurator()
    curator.release = asyncio.Event()
    return curator


class TestResolveTicketReportsTheRow:
    @pytest.mark.asyncio
    async def test_b1_an_expired_wait_answers_pending(self, make_stack):
        stack = await make_stack(_held_curator())
        ticket = await stack.submit('Held work')

        answer = await stack.resolve(ticket, timeout_seconds=0.05, deadline=5)

        assert answer == {'status': 'pending', 'ticket': ticket, 'reason': 'wait_timeout'}
        assert 'task_id' not in answer
        assert (await stack.store.get(ticket))['status'] == 'pending'

    @pytest.mark.asyncio
    async def test_b2_resolving_the_same_ticket_again_gets_the_one_task(self, make_stack):
        curator = _held_curator()
        stack = await make_stack(curator)
        ticket = await stack.submit('Held work')
        assert (await stack.resolve(ticket, timeout_seconds=0.05, deadline=5))['status'] == 'pending'

        assert curator.release is not None
        curator.release.set()
        answer = await stack.resolve(ticket, timeout_seconds=WAIT_SECONDS, deadline=WAIT_SECONDS + 5)

        assert answer['status'] == 'created'
        assert 'ticket' not in answer
        tasks = (await stack.backend.get_tasks(project_root=stack.root))['tasks']
        assert [str(task['id']) for task in tasks] == [answer['task_id']]

    @pytest.mark.asyncio
    async def test_b3_a_parked_row_is_answered_at_once_with_its_reason(self, make_stack):
        stack = await make_stack(_HeldCurator(CuratorDecision.deferred(DeferReason.ALL_ACCOUNTS_CAPPED)))
        ticket = await stack.submit('Capped work')
        parked = {
            'status': 'awaiting_curation', 'ticket': ticket,
            'reason': 'curation-deferred: all-accounts-capped',
        }

        first = await stack.resolve(ticket, timeout_seconds=WAIT_SECONDS, deadline=WAIT_SECONDS + 5)
        second = await stack.resolve(ticket, timeout_seconds=WAIT_SECONDS, deadline=5)

        assert first == parked
        assert second == parked
        assert 'task_id' not in first and 'task_id' not in second

    @pytest.mark.asyncio
    async def test_b4_a_waiter_woken_while_still_open_answers_pending(self, make_stack):
        curator = _held_curator()
        curator.close_gate = asyncio.Event()
        stack = await make_stack(curator)
        ticket = await stack.submit('Held work')

        waiter = asyncio.create_task(
            stack.interceptor.resolve_ticket(ticket, stack.root, timeout_seconds=WAIT_SECONDS),
        )
        await asyncio.sleep(0)
        # close() closes the curator before the ticket store, so holding the
        # curator's close keeps the store readable while the woken waiter re-reads.
        closing = asyncio.create_task(stack.interceptor.close())
        try:
            answer = await asyncio.wait_for(waiter, timeout=5)
            row = await stack.store.get(ticket)
        finally:
            curator.close_gate.set()
            await closing

        assert answer == {'status': 'pending', 'ticket': ticket, 'reason': 'wait_interrupted'}
        assert 'task_id' not in answer
        assert row is not None and row['status'] == 'pending'

    @pytest.mark.asyncio
    async def test_a_closed_store_answers_pending_store_unavailable(self, make_stack):
        stack = await make_stack(None)
        ticket = await stack.store.submit('dark_factory', '{}')
        await stack.store.close()

        answer = await stack.interceptor.resolve_ticket(ticket, stack.root, timeout_seconds=1)

        assert answer == {'status': 'pending', 'ticket': ticket, 'reason': 'store_unavailable'}
        assert 'task_id' not in answer

    @pytest.mark.asyncio
    async def test_an_unknown_ticket_fails_without_a_task_id_key(self, make_stack):
        stack = await make_stack(None)

        answer = await stack.interceptor.resolve_ticket('tkt_' + '0' * 26, stack.root, timeout_seconds=1)

        assert answer == {'status': 'failed', 'reason': 'unknown_ticket'}

    @pytest.mark.asyncio
    async def test_no_ticket_store_fails_without_a_task_id_key(self, make_stack):
        stack = await make_stack(None)
        bare = TaskInterceptor(stack.backend, None, stack.event_buffer)

        try:
            answer = await bare.resolve_ticket('tkt_' + '0' * 26, stack.root, timeout_seconds=1)
        finally:
            await bare.close()

        assert answer == {'status': 'failed', 'reason': 'ticket_store not configured'}


def _unnamed(text: str, values: Iterable[str]) -> list[str]:
    """The *values* *text* does not name as a quoted token (backtick, single or double quote)."""
    return sorted(
        str(value) for value in values
        if not re.search(rf"[`'\"]{re.escape(value)}[`'\"]", text)
    )


def _tool_description(name: str) -> str:
    return registered_fused_memory_tools()[f'{FUSED_MEMORY_TOOL_PREFIX}{name}'].description or ''


def _stage2_slice(start: str, end: str) -> str:
    prompt = build_stage2_system_prompt('dark_factory')
    begin = prompt.index(start)
    return prompt[begin:prompt.index(end, begin + len(start))]


ALL_STATUSES = frozenset(TicketStatus)
ALL_RESOLVE_REASONS = frozenset(ResolveReason)

RUNTIME_SITES: dict[str, tuple[Callable[[], str], frozenset[str]]] = {
    'mcp-resolve_ticket': (lambda: _tool_description('resolve_ticket'), ALL_STATUSES | ALL_RESOLVE_REASONS),
    'mcp-list_tickets': (lambda: _tool_description('list_tickets'), ALL_STATUSES),
    'stage2-status-list': (lambda: _stage2_slice('Interpreting the status:', '\n\n'), ALL_STATUSES),
    'stage2-verifying-rule': (
        lambda: _stage2_slice('## Verifying Task Operations', '\n## '), frozenset(OPEN_TICKET_STATUSES),
    ),
    'recon-self-model': (lambda: recon_self_model.MCP_CALL_SIGNATURES['resolve_ticket'], ALL_STATUSES),
}


class TestVocabularyParityRuntimeSites:
    """Each artifact fused-memory serves names the live vocabulary it must (B5)."""

    @pytest.mark.parametrize('site', sorted(RUNTIME_SITES))
    def test_site_names_the_vocabulary(self, site):
        text_of, required = RUNTIME_SITES[site]
        assert _unnamed(text_of(), required) == [], f'{site} does not name these values'


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_CALLER_DOC = 'skills/_shared/ticket-failure-handling.md'
CALLER_SNIPPETS = (
    'skills/escalation-watcher/SKILL.md',
    'skills/recon-escalation-watcher/SKILL.md',
    'skills/review/references/phase3-triage.md',
    'skills/unblock/SKILL.md',
    'skills/orchestrate/SKILL.md',
    'skills/escalation-watcher-auto/SKILL.md',
)


class TestVocabularyParityCallerDocs:
    """The caller rule is stated once; every caller snippet branches on every open status (B5)."""

    def test_the_canonical_doc_names_the_whole_vocabulary(self):
        text = (REPO_ROOT / CANONICAL_CALLER_DOC).read_text()
        assert _unnamed(text, ALL_STATUSES | ALL_RESOLVE_REASONS) == [], (
            f'{CANONICAL_CALLER_DOC} does not name these values'
        )

    @pytest.mark.parametrize('snippet', CALLER_SNIPPETS)
    def test_a_caller_snippet_has_one_open_branch_and_points_at_the_rule(self, snippet):
        text = (REPO_ROOT / snippet).read_text()
        open_branches = [
            line for line in text.splitlines() if not _unnamed(line, OPEN_TICKET_STATUSES)
        ]
        assert open_branches, f'{snippet} has no line naming every open status'
        assert 'ticket-failure-handling.md' in text, f'{snippet} does not point at the caller rule'
