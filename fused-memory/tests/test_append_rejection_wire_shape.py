"""The WIRE shape of the task-4039 append rejection, end to end (task 4039).

Two agent-facing surfaces promise callers one specific reply for
``update_task(description=..., append=True)``:
``server/tools.py::update_task``'s ``append``/``description`` docstring and the
recon Stage-2 system prompt (``reconciliation/prompts/stage2.py``) both tell
agents the write surfaces as ``TASKMASTER_TOOL_ERROR`` with
``error_type='AppendUnsupportedFieldError'`` naming the offending field.

That shape is NOT produced by the backend guard alone. It is the product of two
further hops the backend tests cannot see:

1. ``TaskInterceptor.update_task`` must let the exception propagate RAW. Its
   only ``except`` catches the two write-authority errors and converts them via
   ``to_error_dict()``; adding ``AppendUnsupportedFieldError`` to that clause
   would turn the raise into a dict and silently falsify both surfaces above
   with the whole backend suite still green.
2. ``@mcp_tool_errors()`` must be the thing that converts it, yielding
   ``{'error': str(e), 'error_type': type(e).__name__}`` — the tool body itself
   wraps the interceptor call in no ``try``.

So this module drives a REAL ``SqliteTaskBackend`` behind a REAL
``TaskInterceptor``, and composes the real ``mcp_tool_errors`` decorator over a
handler that calls it exactly the way ``server/tools.py::update_task`` does
(``return await task_interceptor.update_task(...)``, no interposed ``except``).

Separate file, deliberately: ``tests/test_sqlite_task_backend.py`` owns the
guard's own contract and every one of its cases stops at the backend function.
The subject here is the seam between three modules, not the backend.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.backends.task_backend_errors import AppendUnsupportedFieldError
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.server.tool_errors import mcp_tool_errors

_ORIGINAL_DESCRIPTION = (
    'ORIGINAL defect report. ' * 40
    + 'The multi-KB authored prose the four recorded live repros destroyed.'
)
_ADDENDUM = '\n\n--- PRD ADOPTION ---\nthis was meant to be appended'


@pytest_asyncio.fixture
async def stack(tmp_path):
    """A real ``(interceptor, backend, project_root)`` stack, curator OFF.

    ``config=None`` short-circuits ``TaskInterceptor._get_curator()`` so no
    ticket worker runs; this suite only ever drives ``update_task``.
    """
    cfg = TaskmasterConfig(project_root=str(tmp_path))
    backend = SqliteTaskBackend(cfg)
    await backend.start()

    unique = uuid.uuid4().hex[:8]
    event_buffer = EventBuffer(
        db_path=tmp_path / f'events_{unique}.db', buffer_size_threshold=100,
    )
    await event_buffer.initialize()
    store = TicketStore(tmp_path / f'tickets_{unique}.db')
    await store.initialize()

    interceptor = TaskInterceptor(
        backend, None, event_buffer, config=None, ticket_store=store,
    )
    await backend.add_task(
        project_root=str(tmp_path), title='t', description=_ORIGINAL_DESCRIPTION,
    )
    try:
        yield interceptor, backend, str(tmp_path)
    finally:
        await interceptor.close()
        await backend.close()
        await event_buffer.close()


@pytest.mark.asyncio
async def test_rejection_propagates_raw_through_the_interceptor(stack):
    """Hop 1: the interceptor must NOT convert this raise into a dict.

    Catching it in the write-authority ``except`` clause would return a
    rejection dict instead, and ``mcp_tool_errors`` would never see the
    exception — so the ``error_type`` both agent-facing surfaces advertise
    would silently disappear while every backend test stayed green.
    """
    interceptor, backend, project_root = stack

    with pytest.raises(AppendUnsupportedFieldError) as exc:
        await interceptor.update_task(
            '1', project_root, description=_ADDENDUM, append=True,
        )

    assert exc.value.fields == ('description',), (
        f'the raised error must name the offending field; got {exc.value.fields!r}'
    )
    task = await backend.get_task('1', project_root=project_root)
    assert task['description'] == _ORIGINAL_DESCRIPTION, (
        f'original description must survive the rejected write; got: '
        f'{task["description"]!r}'
    )


@pytest.mark.asyncio
async def test_wire_reply_is_the_shape_agents_are_told_to_expect(stack):
    """Hop 2: the composed handler returns exactly the advertised reply.

    ``{'error': 'TASKMASTER_TOOL_ERROR: …description…',
       'error_type': 'AppendUnsupportedFieldError'}`` — the shape
    ``server/tools.py::update_task`` and the Stage-2 recon prompt both name.
    """
    interceptor, backend, project_root = stack

    @mcp_tool_errors('update_task')
    async def update_task_tool(**kwargs):
        # Mirrors the real tool body: the interceptor call is the tail of the
        # handler, wrapped in no try/except of its own.
        return await interceptor.update_task(**kwargs)

    result = await update_task_tool(
        task_id='1', project_root=project_root,
        description=_ADDENDUM, append=True,
    )

    assert result['error_type'] == 'AppendUnsupportedFieldError', (
        f'agents are told to branch on this error_type; got {result!r}'
    )
    assert result['error'].startswith('TASKMASTER_TOOL_ERROR: '), (
        f'the code must survive into the wire message; got {result["error"]!r}'
    )
    assert 'description' in result['error'], (
        f'the wire message must name the offending field; got {result["error"]!r}'
    )
    task = await backend.get_task('1', project_root=project_root)
    assert task['description'] == _ORIGINAL_DESCRIPTION, (
        'a rejection reply must mean nothing was written'
    )
