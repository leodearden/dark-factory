"""Routed review suggestions land as curator tickets (task 7165).

The REAL orchestrator router posts over a real httpx client into the REAL
fused-memory FastMCP app, served in-process through ``httpx.ASGITransport``:
JSON-RPC over HTTP, FastMCP's argument validation, production's dispatch
guards, the interceptor and the ticket store.  No POST is mocked.  The
assertions read the ticket ROWS that ``submit_task`` wrote, never the
router's account of itself, because the router's self-report is what was
wrong: every submission was refused by FastMCP while the escalation said
``[suggestions → curator]``.

This lives in fused-memory/tests because only this package's pytest path
carries both ``fused_memory`` and ``orchestrator``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from _fm_helpers import _init_git_repo
from orchestrator.artifacts import ReviewAggregation
from orchestrator.config import OrchestratorConfig
from orchestrator.review_suggestions.disposition import SuggestionDisposition
from orchestrator.workflow import TaskWorkflow

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import ServerConfig, TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.models.scope import resolve_project_id
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.server.main import _install_tool_dispatch_guards
from fused_memory.server.tools import create_mcp_server

TASK_ID = '7165'
#: Localhost-like on purpose: FastMCP's transport security answers any other
#: Host header with a 421.
MCP_BASE_URL = 'http://127.0.0.1:8002'

#: Captured before any patch is active, so the factory below builds a real client.
_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _suggestion(tag: str) -> dict:
    return {
        'reviewer': f'reviewer-{tag}',
        'severity': 'suggestion',
        'location': f'src/{tag}.py:{len(tag)}',
        'category': f'category-{tag}',
        'description': f'description-{tag}',
        'suggested_fix': f'fix-{tag}',
    }


def _reviews(suggestions: list[dict]) -> ReviewAggregation:
    return ReviewAggregation(
        has_blocking_issues=False,
        blocking_issues=[],
        suggestions=suggestions,
        reviews={},
    )


def _workflow(project_root: Path) -> TaskWorkflow:
    assignment = MagicMock()
    assignment.task_id = TASK_ID
    assignment.task = {'id': TASK_ID, 'title': 'Reviewed task', 'description': 'desc'}
    mcp = MagicMock()
    mcp.url = MCP_BASE_URL
    return TaskWorkflow(
        assignment=assignment,
        config=OrchestratorConfig(project_root=project_root),
        git_ops=MagicMock(),
        scheduler=MagicMock(),
        briefing=MagicMock(),
        mcp=mcp,
    )


@asynccontextmanager
async def _served_fused_memory(
    tmp_path: Path, *, with_ticket_store: bool,
) -> AsyncIterator[TicketStore | None]:
    """Serve the real submit_task stack to every ``httpx.AsyncClient`` built inside."""
    backend = SqliteTaskBackend(TaskmasterConfig(project_root=str(tmp_path)))
    await backend.start()
    event_buffer = EventBuffer(db_path=tmp_path / 'e2e_eb.db', buffer_size_threshold=100)
    await event_buffer.initialize()
    ticket_store = None
    if with_ticket_store:
        ticket_store = TicketStore(tmp_path / 'e2e_tickets.db')
        await ticket_store.initialize()
    interceptor = TaskInterceptor(backend, None, event_buffer, ticket_store=ticket_store)
    server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)
    _install_tool_dispatch_guards(server, known_projects={})
    server_config = ServerConfig()
    server.settings.stateless_http = server_config.stateless_http
    server.settings.json_response = server_config.json_response
    app = server.streamable_http_app()

    def client_over_asgi(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs['transport'] = httpx.ASGITransport(app=app)
        return _REAL_ASYNC_CLIENT(*args, **kwargs)

    try:
        async with server.session_manager.run():
            with patch('httpx.AsyncClient', client_over_asgi):
                yield ticket_store
    finally:
        await interceptor.close()
        await event_buffer.close()
        await backend.close()


async def _review_suggestion_tickets(
    ticket_store: TicketStore, project_root: Path,
) -> list[dict]:
    rows = await ticket_store.list_tickets(resolve_project_id(str(project_root)))
    candidates = [json.loads(row['candidate_json']) for row in rows]
    return [
        candidate for candidate in candidates
        if candidate.get('metadata', {}).get('spawn_context') == 'review_suggestions'
    ]


@pytest.mark.asyncio
async def test_routed_suggestions_land_as_ticket_rows(tmp_path: Path) -> None:
    project_root = tmp_path / 'project'
    project_root.mkdir()
    _init_git_repo(project_root)
    suggestions = [_suggestion('alpha'), _suggestion('beta')]

    async with _served_fused_memory(tmp_path, with_ticket_store=True) as ticket_store:
        assert ticket_store is not None
        disposition = await _workflow(project_root)._route_review_suggestions_to_curator(
            _reviews(suggestions),
        )
        tickets = await _review_suggestion_tickets(ticket_store, project_root)

    assert disposition == SuggestionDisposition.CURATOR
    assert len(tickets) == 2
    assert {t['metadata']['escalation_id'] for t in tickets} == {
        f'review-suggestions-{TASK_ID}',
    }
    assert len({t['metadata']['suggestion_hash'] for t in tickets}) == 2
    assert all(isinstance(t['kwargs']['details'], str) for t in tickets)


@pytest.mark.asyncio
async def test_refused_submissions_report_undelivered_loudly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    project_root = tmp_path / 'project'
    project_root.mkdir()
    _init_git_repo(project_root)

    async with _served_fused_memory(tmp_path, with_ticket_store=False):
        with caplog.at_level(logging.WARNING):
            disposition = await _workflow(project_root)._route_review_suggestions_to_curator(
                _reviews([_suggestion('alpha')]),
            )

    assert disposition == SuggestionDisposition.UNDELIVERED
    assert any(
        record.levelno == logging.WARNING and 'ConfigError' in record.getMessage()
        for record in caplog.records
    )
