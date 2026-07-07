"""Tests for the mcp_tool_errors decorator (fused_memory.server.tool_errors).

Covers the core contract (exception -> structured error dict; success
passthrough; ERROR-level exception logging; CancelledError/KeyboardInterrupt/
SystemExit propagation; functools.wraps metadata preservation), the
__mcp_tool_errors__ marker + operation-label override, and integration over
the real MCP server proving the 4 gap handlers (submit_task, resolve_ticket,
list_tickets, cancel_ticket) now carry the marker and produce a
shape-identical error response.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from unittest.mock import AsyncMock

import pytest

from fused_memory.server.tool_errors import mcp_tool_errors
from fused_memory.server.tools import create_mcp_server

# ---------------------------------------------------------------------------
# Core contract (step-1 / step-2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exception_becomes_structured_error_dict():
    """A wrapped handler raising RuntimeError returns {'error', 'error_type'}."""

    @mcp_tool_errors()
    async def h():
        raise RuntimeError('boom')

    result = await h()

    assert result == {'error': 'boom', 'error_type': 'RuntimeError'}


@pytest.mark.asyncio
async def test_success_passthrough_forwards_args_and_kwargs():
    """A successful handler's return value flows through unchanged, and the
    wrapper forwards *args/**kwargs to the underlying handler.
    """

    @mcp_tool_errors()
    async def h(a, b, *, c):
        return {'a': a, 'b': b, 'c': c}

    result = await h(1, 2, c=3)

    assert result == {'a': 1, 'b': 2, 'c': 3}


@pytest.mark.asyncio
async def test_exception_logs_at_error_level_with_traceback(caplog):
    """The RuntimeError case emits exactly one ERROR record with exc_info set
    and a message prefixed with the handler's name (operation defaults to
    fn.__name__).
    """

    @mcp_tool_errors()
    async def h():
        raise RuntimeError('boom')

    with caplog.at_level(logging.ERROR, logger='fused_memory.server.tool_errors'):
        await h()

    matched = [r for r in caplog.records if r.name == 'fused_memory.server.tool_errors']
    assert len(matched) == 1
    record = matched[0]
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None
    assert record.getMessage().startswith('h error:')


@pytest.mark.asyncio
@pytest.mark.parametrize('exc_cls', [asyncio.CancelledError, KeyboardInterrupt, SystemExit])
async def test_cancellation_family_propagates(exc_cls):
    """CancelledError, KeyboardInterrupt, and SystemExit are not converted to
    an error dict — they propagate out of the wrapper unchanged.
    """

    @mcp_tool_errors()
    async def h():
        raise exc_cls

    with pytest.raises(exc_cls):
        await h()


@pytest.mark.asyncio
async def test_wraps_preserves_metadata():
    """functools.wraps preserves __name__, __doc__, __wrapped__, and signature."""

    async def h(a: int, b: str = 'x') -> dict:
        """h's docstring."""
        return {}

    wrapped = mcp_tool_errors()(h)

    assert wrapped.__name__ == 'h'
    assert wrapped.__doc__ == "h's docstring."
    # __wrapped__ exists at runtime (set by functools.wraps) but pyright types
    # mcp_tool_errors()(h)'s result as a plain function, which doesn't declare
    # it -- see test_project_scope.py's __supertype__ tests for the same
    # documented pyright limitation.
    assert wrapped.__wrapped__ is h  # pyright: ignore[reportFunctionMemberAccess]
    assert inspect.signature(wrapped) == inspect.signature(h)


# ---------------------------------------------------------------------------
# Marker + operation-label override (step-3 / step-4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_marker_is_stamped_on_wrapper():
    """The wrapper carries __mcp_tool_errors__ = True so a registry walker
    (task ε) can confirm every handler is decorated.
    """

    @mcp_tool_errors()
    async def h():
        return {}

    assert getattr(h, '__mcp_tool_errors__', False) is True


@pytest.mark.asyncio
async def test_operation_override_relabels_log_message(caplog):
    """mcp_tool_errors(operation=...) relabels the log message prefix while
    leaving the returned error-dict shape unchanged.
    """

    @mcp_tool_errors(operation='custom_op')
    async def h():
        raise RuntimeError('boom')

    with caplog.at_level(logging.ERROR, logger='fused_memory.server.tool_errors'):
        result = await h()

    assert result == {'error': 'boom', 'error_type': 'RuntimeError'}
    matched = [r for r in caplog.records if r.name == 'fused_memory.server.tool_errors']
    assert len(matched) == 1
    assert matched[0].getMessage().startswith('custom_op error:')


# ---------------------------------------------------------------------------
# Integration over the real MCP server (step-5 / step-6)
#
# Proves the decorator is actually applied to the 4 gap handlers (submit_task,
# resolve_ticket, list_tickets, cancel_ticket) and that the observable error
# shape is unchanged from the pre-refactor hand-written tails.
# ---------------------------------------------------------------------------

_GAP_HANDLERS = ('submit_task', 'resolve_ticket', 'list_tickets', 'cancel_ticket')


@pytest.fixture(autouse=True)
def passthrough_main_checkout(monkeypatch):
    """Stub resolve_main_checkout to pass its argument through unchanged.

    These tests use synthetic project_root values like ``/project`` that
    aren't real git working trees.
    """
    monkeypatch.setattr(
        'fused_memory.server.tools.resolve_main_checkout', lambda p: str(p),
    )


@pytest.fixture
def task_interceptor():
    ti = AsyncMock()
    ti.submit_task = AsyncMock(side_effect=RuntimeError('backend down'))
    ti.resolve_ticket = AsyncMock(side_effect=RuntimeError('backend down'))
    ti.list_tickets = AsyncMock(side_effect=RuntimeError('backend down'))
    ti.cancel_ticket = AsyncMock(side_effect=RuntimeError('backend down'))
    return ti


@pytest.fixture
def mcp_server(task_interceptor):
    """MCP server with a mocked task interceptor whose 4 gap methods all fail."""
    mock_service = AsyncMock()
    return create_mcp_server(mock_service, task_interceptor=task_interceptor)


@pytest.mark.asyncio
async def test_gap_handlers_carry_mcp_tool_errors_marker(mcp_server):
    """Each of the 4 gap handlers is registered with the decorator applied."""

    for name in _GAP_HANDLERS:
        tool = mcp_server._tool_manager.get_tool(name)
        assert tool.fn.__mcp_tool_errors__ is True, (
            f'{name} is not decorated with @mcp_tool_errors()'
        )


@pytest.mark.asyncio
async def test_gap_handlers_return_shape_identical_error_on_failure(mcp_server):
    """An interceptor RuntimeError surfaces as {'error', 'error_type'} —
    byte-identical to the pre-change hand-written tail's shape — for all 4
    gap handlers, including list_tickets whose specialized ValueError branch
    and post-call summarisation must survive the refactor untouched.
    """
    expected = {'error': 'backend down', 'error_type': 'RuntimeError'}

    # submit_task's minimal {'project_root': ...} arg set is expected to clear
    # every pre-interceptor validation gate (project_root format via the
    # passthrough_main_checkout fixture, the lock-charter directory guard on
    # metadata.files, and the deterministic-task-kind invariants) untouched,
    # since metadata=None and task_kind defaults to 'normal'. That is what
    # lets this call reach task_interceptor.submit_task and exercise the
    # decorator's passthrough. If validation ordering later moves a
    # required-field check ahead of the interceptor call, this assertion
    # would silently start checking that ValidationError dict instead —
    # revisit the arg set here if that happens.
    submit_result = await mcp_server._tool_manager.call_tool(
        'submit_task', {'project_root': '/project'},
    )
    assert submit_result == expected

    resolve_result = await mcp_server._tool_manager.call_tool(
        'resolve_ticket',
        {'ticket': 'tkt_ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'project_root': '/project'},
    )
    assert resolve_result == expected

    list_result = await mcp_server._tool_manager.call_tool(
        'list_tickets', {'project_root': '/project'},
    )
    assert list_result == expected

    cancel_result = await mcp_server._tool_manager.call_tool(
        'cancel_ticket', {'ticket_id': 'tkt_X'},
    )
    assert cancel_result == expected
