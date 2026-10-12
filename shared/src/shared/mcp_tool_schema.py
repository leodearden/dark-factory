"""The invoked tool's LIVE JSON Schema, read from inside FastMCP middleware.

Measured against fastmcp 3.2.2: ``FastMCP.get_tool`` is a coroutine and must be
awaited, it answers ``None`` for an unknown name, and ``tool.parameters`` is the
full JSON Schema, so parameter nodes sit under ``'properties'`` and required
names under ``'required'``. Read live rather than captured at registration, so a
guard checks the tool as it actually is.

:func:`live_tool_parameters` is the one walk from a middleware context to that
schema. It answers ``None`` when there is no usable schema, after logging a
WARNING, and each caller picks its own fail-safe from there. Not re-exported
from ``shared/__init__``, so ``import shared`` does not pull in fastmcp.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from fastmcp.server.middleware import MiddlewareContext

logger = logging.getLogger(__name__)


async def live_tool_parameters(
    context: MiddlewareContext[Any], name: str
) -> Mapping[str, Any] | None:
    fastmcp_context = context.fastmcp_context
    if fastmcp_context is None:
        logger.warning('no FastMCP context to resolve the schema of tool %r from', name)
        return None
    try:
        tool = await fastmcp_context.fastmcp.get_tool(name)
    except Exception:
        logger.warning('could not resolve the live schema of tool %r', name, exc_info=True)
        return None
    parameters = getattr(tool, 'parameters', None)
    if not isinstance(parameters, dict):
        logger.warning(
            'tool %r has no usable live schema (got %s)', name, type(parameters).__name__
        )
        return None
    return parameters
