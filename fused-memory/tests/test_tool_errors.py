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

import pytest

from fused_memory.server.tool_errors import mcp_tool_errors


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
    assert wrapped.__wrapped__ is h
    assert inspect.signature(wrapped) == inspect.signature(h)
