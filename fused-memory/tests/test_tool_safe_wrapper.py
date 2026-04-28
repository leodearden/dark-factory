"""Tests for the _install_safe_tool_wrapper helper in fused_memory.server.main.

Fix C: defence-in-depth wrapper at FastMCP's central tool-dispatch chokepoint.

Background — what the wrapper actually defends against:

FastMCP's :class:`Tool.run` already catches :class:`Exception` and re-raises as
:class:`ToolError`, so RuntimeError / ValueError / etc. never escape the SDK.
The cascade trigger is :class:`BaseException` subclasses that FastMCP does NOT
catch: :class:`SystemExit`, :class:`KeyboardInterrupt`, and the asyncio /
anyio :class:`BaseExceptionGroup` family. A bare :class:`asyncio.CancelledError`
must still propagate (cancellation semantics).

Tests organized so each failure mode that wedged StreamableHTTPSessionManager's
shared task group on 2026-04-28 00:32:15 is reproduced before the wrapper, and
contained after.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock

import pytest

from fused_memory.server.main import _install_safe_tool_wrapper
from fused_memory.server.tools import create_mcp_server


def _build_server_with_tool(handler):
    """Construct a real FastMCP server, register a fake tool, install the wrapper."""
    mock_service = AsyncMock()
    server = create_mcp_server(mock_service)
    server._tool_manager.add_tool(handler, name='_fake_tool')
    _install_safe_tool_wrapper(server)
    return server


class TestSafeWrapperReraisesCancelledError:
    """Bare CancelledError MUST propagate — required for asyncio cancellation
    semantics. Swallowing it would leak tasks and break structured concurrency."""

    @pytest.mark.asyncio
    async def test_cancellederror_propagates(self):
        async def cancels():
            raise asyncio.CancelledError

        server = _build_server_with_tool(cancels)
        with pytest.raises(asyncio.CancelledError):
            await server._tool_manager.call_tool('_fake_tool', {})


class TestSafeWrapperContainsRuntimeError:
    """RuntimeError gets wrapped by FastMCP as ToolError before it reaches the
    wrapper, so the wrapper sees a ToolError. Either way, the contract is the
    same: structured error dict, no propagation, log at ERROR."""

    @pytest.mark.asyncio
    async def test_runtimeerror_returns_structured_error(self, caplog):
        async def raises():
            raise RuntimeError('boom')

        server = _build_server_with_tool(raises)
        with caplog.at_level(logging.ERROR, logger='fused_memory.server.main'):
            result = await server._tool_manager.call_tool('_fake_tool', {})

        assert isinstance(result, dict)
        assert 'error' in result
        assert 'error_type' in result
        assert 'boom' in result['error']
        # ERROR-level log includes tool_name in the structured extra.
        matched = [
            r for r in caplog.records
            if 'Tool handler escaped exception' in r.getMessage()
        ]
        assert matched, 'wrapper did not log at ERROR'
        assert getattr(matched[0], 'tool_name', None) == '_fake_tool'


class TestSafeWrapperContainsSystemExit:
    """SystemExit is a BaseException, NOT an Exception. FastMCP's
    ``except Exception`` does NOT catch it — so without the wrapper it
    propagates and tears down the SDK's shared task group. This is one of
    the BaseException paths that motivated the wrapper."""

    @pytest.mark.asyncio
    async def test_systemexit_does_not_propagate(self):
        async def exits():
            raise SystemExit(2)

        server = _build_server_with_tool(exits)
        result = await server._tool_manager.call_tool('_fake_tool', {})

        assert isinstance(result, dict)
        assert result['error_type'] == 'SystemExit'


class TestSafeWrapperContainsBaseExceptionGroup:
    """A true BaseExceptionGroup (containing at least one BaseException) is what
    asyncio.timeout / anyio task groups raise during cancellation. FastMCP's
    ``except Exception`` does NOT catch BaseExceptionGroup."""

    @pytest.mark.asyncio
    async def test_base_exception_group_does_not_propagate(self):
        # CancelledError is a BaseException (since 3.8), so a group containing
        # one is a true BaseExceptionGroup — not an ExceptionGroup. This is the
        # exact poison shape from the journal trace at 00:32:15.
        async def group():
            raise BaseExceptionGroup(
                'task group failure',
                [RuntimeError('inner'), asyncio.CancelledError()],
            )

        server = _build_server_with_tool(group)
        result = await server._tool_manager.call_tool('_fake_tool', {})

        assert isinstance(result, dict)
        assert 'BaseExceptionGroup' in result['error_type'] or \
               'ExceptionGroup' in result['error_type']

    @pytest.mark.asyncio
    async def test_subprocess_cancel_surrogate_does_not_propagate(self):
        """Surrogate for the shared.cli_invoke pattern at journal 00:32:15:
        a subprocess cleanup raises a BaseExceptionGroup mixing CancelledError
        with a regular cleanup error. Without the wrapper, this escapes
        FastMCP and kills the session manager's task group."""

        async def wrapped_cancel():
            try:
                async with asyncio.timeout(0.001):
                    await asyncio.sleep(10)
            except asyncio.CancelledError:
                # Mirrors the cli_invoke pattern: re-raise the cleanup-failure
                # collection as a BaseExceptionGroup.
                raise BaseExceptionGroup(
                    'subprocess cleanup',
                    [RuntimeError('cleanup failed'), asyncio.CancelledError()],
                ) from None

        server = _build_server_with_tool(wrapped_cancel)
        result = await server._tool_manager.call_tool('_fake_tool', {})

        assert isinstance(result, dict)
        assert 'error_type' in result


class TestSafeWrapperHappyPath:
    @pytest.mark.asyncio
    async def test_successful_tool_returns_unchanged_result(self):
        async def ok():
            return {'ok': True, 'value': 42}

        server = _build_server_with_tool(ok)
        result = await server._tool_manager.call_tool('_fake_tool', {})

        assert result == {'ok': True, 'value': 42}


class TestSafeWrapperIdempotent:
    """Calling the installer twice must not double-wrap (which would silently
    suppress the structured error returned by the inner wrapper)."""

    @pytest.mark.asyncio
    async def test_double_install_does_not_double_wrap(self):
        async def raises():
            raise RuntimeError('boom')

        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)
        server._tool_manager.add_tool(raises, name='_fake_tool')

        _install_safe_tool_wrapper(server)
        first_wrapped = server._tool_manager.call_tool

        _install_safe_tool_wrapper(server)
        assert server._tool_manager.call_tool is first_wrapped, (
            'second install replaced the wrapper instead of being a no-op'
        )

        result = await server._tool_manager.call_tool('_fake_tool', {})
        assert isinstance(result, dict) and 'error' in result


class TestSafeWrapperReproductionWithoutWrapperBehavior:
    """Reproduction tests proving that *without* the wrapper, the BaseException
    paths escape FastMCP's own ``except Exception``. These are not strictly
    needed for regression — they exist to anchor the contract that the wrapper
    is what closes the gap.
    """

    @pytest.mark.asyncio
    async def test_systemexit_escapes_fastmcp_without_wrapper(self):
        """Without the wrapper, FastMCP lets SystemExit escape — proving the
        gap that the wrapper closes."""
        async def exits():
            raise SystemExit(2)

        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)
        server._tool_manager.add_tool(exits, name='_fake_tool')
        # NOTE: no _install_safe_tool_wrapper call.

        with pytest.raises(SystemExit):
            await server._tool_manager.call_tool('_fake_tool', {})

    @pytest.mark.asyncio
    async def test_base_exception_group_escapes_fastmcp_without_wrapper(self):
        async def group():
            raise BaseExceptionGroup(
                'task group failure',
                [RuntimeError('inner'), asyncio.CancelledError()],
            )

        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)
        server._tool_manager.add_tool(group, name='_fake_tool')
        # NOTE: no _install_safe_tool_wrapper call.

        with pytest.raises(BaseExceptionGroup):
            await server._tool_manager.call_tool('_fake_tool', {})
