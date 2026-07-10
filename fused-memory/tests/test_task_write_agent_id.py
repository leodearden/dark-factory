"""Tests for threading the caller agent_id onto the task-write path (W5-ε, task 2223).

Covers the two write paths whose interceptor plumbing task 2223 completes on top
of task 2175/ρ1b (which already threaded set_task_status → _apply_status_transition):

- interceptor.update_task (agent_id captured out of **kwargs; not leaked to tm.update_task)
- interceptor.submit_task (agent_id captured out of **kwargs; not serialized into the
  ticket blob)
- MCP handlers set_task_status / update_task / submit_task (agent_id resolved from
  ctx via _resolve_identity and forwarded to the interceptor)

The interceptor-internal set_task_status → _apply_status_transition threading (and its
actor-class transition gate) is owned and tested by task 2175/ρ1b; this module only
asserts the set_task_status *handler* forwards agent_id, for symmetry across the three
write paths.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from mcp.server.fastmcp import Context

from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.server.tools import create_mcp_server

AGENT_ID = 'recon-stage-task_knowledge_sync'


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_ctx(
    client_name: str | None = 'claude-code',
    session_header: str | None = 'abc-123-session',
    *,
    has_client_params: bool = True,
    has_request: bool = True,
) -> Context[Any, Any, Any]:
    """Build a minimal Context-shaped object. Copied from test_identity_resolution.py."""
    ctx = SimpleNamespace()

    if has_client_params and client_name is not None:
        ctx.session = SimpleNamespace(
            client_params=SimpleNamespace(
                clientInfo=SimpleNamespace(name=client_name)
            )
        )
    elif not has_client_params:
        # Stateless HTTP — no client_params at all
        ctx.session = SimpleNamespace(client_params=None)
    else:
        ctx.session = SimpleNamespace(
            client_params=SimpleNamespace(clientInfo=SimpleNamespace(name=None))
        )

    if has_request and session_header is not None:
        headers = {'mcp-session-id': session_header}
        ctx.request_context = SimpleNamespace(
            request=SimpleNamespace(headers=MagicMock(get=lambda k, d=None: headers.get(k, d)))
        )
    elif not has_request:
        # stdio transport — no request_context
        ctx.request_context = None
    else:
        ctx.request_context = SimpleNamespace(
            request=SimpleNamespace(headers=MagicMock(get=lambda k, d=None: None))
        )

    return cast(Context[Any, Any, Any], ctx)


def _raw(mcp_server, name: str):
    """Return the raw (undecorated) tool closure so a fake ``ctx`` can be injected.

    FastMCP treats a ``ctx: Context`` parameter as the context_kwarg and excludes
    it from the tool's JSON arg schema, so a fake ctx cannot be threaded through
    ``_tool_manager.call_tool``'s arguments dict — the closure must be invoked
    directly.
    """
    return mcp_server._tool_manager.get_tool(name).fn


# ---------------------------------------------------------------------------
# Interceptor-level fixtures (mirrors test_task_interceptor.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def taskmaster():
    tm = AsyncMock()
    tm.get_task = AsyncMock(return_value={'id': '1', 'status': 'pending', 'title': 'Test Task'})
    tm.set_task_status = AsyncMock(return_value={'success': True})
    tm.get_tasks = AsyncMock(return_value={'tasks': []})
    tm.add_task = AsyncMock(return_value={'id': '2', 'title': 'New Task'})
    tm.update_task = AsyncMock(return_value={'success': True})
    tm.remove_tasks = AsyncMock(return_value={'success': True})
    tm.add_dependency = AsyncMock(return_value={'success': True})
    tm.remove_dependency = AsyncMock(return_value={'success': True})
    return tm


@pytest.fixture
def reconciler():
    r = AsyncMock()
    r.reconcile_task = AsyncMock(return_value={'actions': [{'type': 'knowledge_captured'}]})
    return r


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    buf = EventBuffer(db_path=tmp_path / 'interceptor_eb.db', buffer_size_threshold=100)
    await buf.initialize()
    yield buf
    await buf.close()


@pytest.fixture
def interceptor(taskmaster, reconciler, event_buffer):
    return TaskInterceptor(taskmaster, reconciler, event_buffer)


# ---------------------------------------------------------------------------
# MCP handler-level fixtures (mirrors test_scheduler_overrides_tools.py)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def passthrough_main_checkout(monkeypatch):
    """Stub resolve_main_checkout to pass its argument through unchanged.

    These tests use synthetic project_root values that aren't real git
    working trees; the real resolver would reject them.
    """
    monkeypatch.setattr(
        'fused_memory.server.tools.resolve_main_checkout', lambda p: str(p),
    )


@pytest.fixture
def memory_service():
    """Mocked MemoryService with all methods as AsyncMocks."""
    svc = AsyncMock()
    svc.add_memory = AsyncMock(return_value=None)
    return svc


@pytest.fixture
def task_interceptor():
    """Mocked task interceptor with task-write methods returning success by default."""
    ti = AsyncMock()
    ti.set_task_status = AsyncMock(return_value={'success': True})
    ti.update_task = AsyncMock(return_value={'success': True})
    ti.submit_task = AsyncMock(return_value={'ticket': 'tkt_x'})
    return ti


@pytest.fixture
def mcp_server(memory_service, task_interceptor):
    """MCP server with a mocked MemoryService and a mocked task interceptor."""
    return create_mcp_server(memory_service, task_interceptor=task_interceptor)


# ---------------------------------------------------------------------------
# set_task_status handler forwards agent_id (interceptor path owned by task 2175)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_task_status_handler_resolves_ctx_and_forwards_agent_id(
    mcp_server, task_interceptor,
):
    """The handler resolves agent_id from ctx and forwards it to the interceptor."""
    handler = _raw(mcp_server, 'set_task_status')

    await handler(
        id='1', status='done', project_root='/project', ctx=_make_ctx(AGENT_ID),
    )

    task_interceptor.set_task_status.assert_awaited_once()
    assert task_interceptor.set_task_status.await_args.kwargs.get('agent_id') == AGENT_ID


@pytest.mark.asyncio
async def test_set_task_status_handler_explicit_agent_id_takes_precedence_over_ctx(
    mcp_server, task_interceptor,
):
    """_resolve_identity contract: 'Explicit caller values always take precedence.'

    Passing agent_id explicitly alongside a ctx that would resolve a
    *different* identity must forward the explicit value, not the
    ctx-derived one.
    """
    handler = _raw(mcp_server, 'set_task_status')

    await handler(
        id='1',
        status='done',
        project_root='/project',
        agent_id='explicit',
        ctx=_make_ctx('recon-stage-task_knowledge_sync'),
    )

    task_interceptor.set_task_status.assert_awaited_once()
    assert task_interceptor.set_task_status.await_args.kwargs.get('agent_id') == 'explicit'


# ---------------------------------------------------------------------------
# update_task: interceptor captures agent_id (no backend leak) + handler forwards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interceptor_update_task_captures_agent_id_without_leaking_to_backend(
    interceptor, taskmaster,
):
    """agent_id is captured out of **kwargs — it must not ride into tm.update_task."""
    await interceptor.update_task('1', '/project', title='x', agent_id=AGENT_ID)

    taskmaster.update_task.assert_awaited_once()
    assert 'agent_id' not in taskmaster.update_task.call_args.kwargs


@pytest.mark.asyncio
async def test_update_task_handler_resolves_ctx_and_forwards_agent_id(
    mcp_server, task_interceptor,
):
    """The handler resolves agent_id from ctx and forwards it to the interceptor.

    Passes ``title`` (not ``status``) — ``_reject_status_in_update_task``
    rejects status writes before the interceptor is ever reached.
    """
    handler = _raw(mcp_server, 'update_task')

    await handler(
        id='1', project_root='/project', title='x', ctx=_make_ctx(AGENT_ID),
    )

    task_interceptor.update_task.assert_awaited_once()
    assert task_interceptor.update_task.await_args.kwargs.get('agent_id') == AGENT_ID


# ---------------------------------------------------------------------------
# submit_task: interceptor captures agent_id (no blob leak) + handler forwards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interceptor_submit_task_captures_agent_id_without_serializing_into_ticket_blob(
    taskmaster, reconciler, event_buffer, monkeypatch,
):
    """agent_id is captured out of **kwargs — it must not be serialized into the ticket blob."""
    ticket_store = SimpleNamespace(submit=AsyncMock(return_value='tkt_x'))
    ti = TaskInterceptor(
        taskmaster, reconciler, event_buffer, ticket_store=cast(TicketStore, ticket_store),
    )
    monkeypatch.setattr(ti, '_start_worker_if_needed', lambda project_id: None)

    result = await ti.submit_task('/project', title='x', agent_id=AGENT_ID)

    assert result == {'ticket': 'tkt_x'}
    ticket_store.submit.assert_awaited_once()
    candidate_json = ticket_store.submit.await_args.kwargs['candidate_json']
    assert 'agent_id' not in json.loads(candidate_json)['kwargs']


@pytest.mark.asyncio
async def test_submit_task_handler_resolves_ctx_and_forwards_agent_id(
    mcp_server, task_interceptor,
):
    """The handler resolves agent_id from ctx and forwards it to the interceptor."""
    handler = _raw(mcp_server, 'submit_task')

    await handler(
        project_root='/project', title='x', ctx=_make_ctx(AGENT_ID),
    )

    task_interceptor.submit_task.assert_awaited_once()
    assert task_interceptor.submit_task.await_args.kwargs.get('agent_id') == AGENT_ID
