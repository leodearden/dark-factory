"""Refuse a tool call that names an argument its tool does not declare (task 4867).

The bundled FastMCP's argument model ignores extras, so without this guard a
misnamed argument is dropped and the call runs without it.
:class:`UnknownArgumentMiddleware` reads the invoked tool's LIVE schema on
every call and raises the refusal
:func:`shared.mcp_unknown_arguments.unknown_argument_refusal` builds. Nothing is
special-cased by tool name: the declared set is the schema's ``properties``,
which also leaves out the framework-injected ``ctx``, so a client that sends
``ctx`` is refused as well.

The guard fails OPEN — the call goes through unchecked — whenever there is no
usable schema to compare against: an unknown tool, a schema that is not a dict,
or one without a ``properties`` mapping. Refusing a valid call is worse than
the defect it guards against.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware
from shared.mcp_tool_schema import live_tool_parameters
from shared.mcp_unknown_arguments import unknown_argument_refusal

from fused_memory.server.tool_middleware_bridge import install_tool_middleware

logger = logging.getLogger(__name__)


class UnknownArgumentMiddleware(Middleware):
    async def on_call_tool(self, context, call_next):
        name = context.message.name
        parameters = await live_tool_parameters(context, name)
        properties = parameters.get('properties') if parameters is not None else None
        if not isinstance(properties, dict):
            if parameters is not None:
                logger.warning(
                    'tool %r has no properties in its live schema; its arguments are unchecked',
                    name,
                )
            return await call_next(context)
        refusal = unknown_argument_refusal(name, list(properties), context.message.arguments or {})
        if refusal is not None:
            logger.warning(
                'refused %s: undeclared argument(s) %s',
                name, {entry['name']: entry['closest_declared'] for entry in refusal['unknown']},
            )
            raise ToolError(json.dumps(refusal, default=repr))
        return await call_next(context)


def install_unknown_argument_guard(mcp: Any) -> None:
    install_tool_middleware(
        mcp,
        UnknownArgumentMiddleware(),
        sentinel='_fused_memory_unknown_argument_guarded',
        label='unknown-argument guard',
    )
