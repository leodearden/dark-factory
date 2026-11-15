"""Run a fastmcp ``Middleware.on_call_tool`` at the BUNDLED FastMCP's dispatch chokepoint.

fused-memory's servers are :class:`mcp.server.fastmcp.FastMCP` — the FastMCP
BUNDLED inside the ``mcp`` SDK — not the standalone ``fastmcp`` distribution.
Probed in fused-memory's own venv:

* bundled ``FastMCP``: ``add_middleware`` ABSENT, ``get_tool`` ABSENT. Its tool
  surface is ``{add_tool, call_tool, list_tools, remove_tool, tool}``.
* standalone ``fastmcp`` 3.2.2 ``FastMCP``: both present.

So a fastmcp middleware cannot be attached the documented way. This module
attaches one at the chokepoint the bundled class does have —
``ToolManager.call_tool`` — driving the middleware UNMODIFIED through a
stand-in context. Two further substrate deltas, both measured:

* ``ToolManager.get_tool(name)`` is SYNC here and returns ``Tool | None``; on
  the standalone class it is a coroutine. :class:`_ShimFastMCP` wraps it in an
  ``async def``, so ``shared.mcp_tool_schema.live_tool_parameters``, which
  awaits it, needs no adaptation. ``Tool.parameters`` is the same full JSON
  Schema dict on both classes.
* Argument validation lives INSIDE ``Tool.run``
  (``fn_metadata.call_fn_with_arg_validation``), strictly AFTER this
  chokepoint, so a middleware here sees the arguments exactly as sent.

A middleware rejects by raising :class:`fastmcp.exceptions.ToolError`, and must
be installed outside ``main._install_safe_tool_wrapper``;
``fused_memory/server/main.py::_install_tool_dispatch_guards`` owns that order
and says why.

Being outside that wrapper also puts the middleware's OWN defects beyond its
containment, so :func:`install_tool_middleware` carries its own: anything but
``CancelledError`` and the middleware's ``ToolError`` is logged and the call is
re-dispatched to the contained inner wrapper (fail OPEN — a check that crashed
checked nothing), except after the call was already forwarded, where
re-dispatching would double a write.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, cast

from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware

logger = logging.getLogger(__name__)


@dataclass
class _ShimMessage:
    """The ``context.message`` shape ``on_call_tool`` reads.

    NOT frozen, and ``arguments`` is a mutable dict a middleware is free to
    edit in place (the markup guard's override path deletes ``metadata`` here):
    ``call_next`` re-reads this attribute so the edit lands on the forwarded
    call.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


class _ShimFastMCP:
    """``context.fastmcp_context.fastmcp``, with ``get_tool`` made awaitable.

    The ONE adaptation the substrate needs, absorbed here rather than branched
    on inside a shared middleware.
    """

    def __init__(self, tool_manager: Any) -> None:
        self._tool_manager = tool_manager

    async def get_tool(self, name: str) -> Any:
        return self._tool_manager.get_tool(name)


@dataclass
class _ShimFastMCPContext:
    """``context.fastmcp_context`` — a carrier for ``.fastmcp`` and nothing else."""

    fastmcp: _ShimFastMCP


@dataclass
class _ShimContext:
    """The middleware-facing context surface, which is exactly three paths.

    ``context.message.name``, ``context.message.arguments`` and
    ``context.fastmcp_context.fastmcp.get_tool``. A middleware driven through
    this bridge must read nothing else.
    """

    message: _ShimMessage
    fastmcp_context: _ShimFastMCPContext


def install_tool_middleware(
    mcp: Any, middleware: Middleware, *, sentinel: str, label: str
) -> None:
    """Wrap ``mcp._tool_manager.call_tool`` with *middleware*'s ``on_call_tool``.

    Wrapping ``call_tool`` rather than each tool covers every registered tool
    and any added later, in one place. Each call wraps whatever ``call_tool``
    it finds, so the install order decides the nesting.

    *sentinel* is the tool-manager attribute that marks this middleware as
    installed; a second install with the same sentinel is a no-op, so re-entry
    never double-wraps. *label* names the middleware in the log lines.
    """
    tool_manager = mcp._tool_manager
    if getattr(tool_manager, sentinel, False):
        return

    original_call_tool = tool_manager.call_tool
    shim_fastmcp = _ShimFastMCPContext(_ShimFastMCP(tool_manager))

    async def _call_tool_through_middleware(
        name: str, arguments: dict[str, Any] | None, *args: Any, **kwargs: Any
    ) -> Any:
        # A COPY: a middleware may edit ``arguments`` in place, and a wrapper
        # has no licence to edit the caller's own dict.
        context = _ShimContext(_ShimMessage(name, dict(arguments or {})), shim_fastmcp)
        forwarded = False

        async def call_next(ctx: _ShimContext) -> Any:
            # Re-read off the context so an in-place edit reaches the tool.
            # ``*args``/``**kwargs`` — the bundled signature's ``context`` and
            # ``convert_result`` — pass through untouched.
            nonlocal forwarded
            forwarded = True
            return await original_call_tool(
                ctx.message.name, ctx.message.arguments, *args, **kwargs
            )

        try:
            # ``cast`` at this ONE site is the whole nominal cost of the
            # adaptation: ``on_call_tool`` is annotated against fastmcp's
            # ``MiddlewareContext``, which the bundled SDK cannot construct.
            return await middleware.on_call_tool(cast(Any, context), cast(Any, call_next))
        except asyncio.CancelledError:
            raise
        except ToolError:
            raise
        except BaseException:
            if forwarded:
                logger.exception(
                    '%s raised AFTER forwarding %s; not re-dispatching '
                    '(a second call would double the write)', label, name,
                )
                raise
            logger.exception(
                '%s raised while inspecting %s; failing OPEN to the contained '
                'dispatcher — this call was NOT scanned', label, name,
            )
            return await original_call_tool(name, arguments, *args, **kwargs)

    tool_manager.call_tool = _call_tool_through_middleware
    setattr(tool_manager, sentinel, True)
