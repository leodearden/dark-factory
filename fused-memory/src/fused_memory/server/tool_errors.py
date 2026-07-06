"""Shared error-handling decorator for FastMCP tool handlers.

Centralizes the hand-written cancellation-guard + structured-error-dict
convention that most handlers in :mod:`fused_memory.server.tools` already
follow by hand (see e.g. ``set_task_status`` and ``search_tasks``):

    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.exception(f'{op} error: {e}')
        return {'error': str(e), 'error_type': type(e).__name__}

``_install_safe_tool_wrapper`` (fused_memory.server.main) remains as a
central defence-in-depth backstop beneath this decorator; this module does
not replace it.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)


def mcp_tool_errors(
    operation: str | None = None,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Return a decorator that gives an async MCP tool handler the standard
    cancellation-guard + structured-error-dict convention.

    Args:
        operation: Label used in the ERROR log message (``f'{operation} error: {e}'``).
            Defaults to the wrapped function's ``__name__``.
    """

    def decorator(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        op = operation or fn.__name__

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                logger.exception(f'{op} error: {e}')
                return {'error': str(e), 'error_type': type(e).__name__}

        # Set after functools.wraps: update_wrapper copies fn.__dict__ onto
        # wrapper, which would otherwise clobber this marker.
        wrapper.__mcp_tool_errors__ = True
        return wrapper

    return decorator
