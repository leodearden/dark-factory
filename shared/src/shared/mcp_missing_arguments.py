"""FastMCP middleware that turns a missing-required-argument failure into a refusal.

FastMCP 3.2.2 validates tool arguments in
``fastmcp/tools/function_tool.py::FunctionTool.run``, after the middleware
chain and BEFORE the tool body. A call that omits a required argument fails
there with pydantic's raw text ("1 validation error for call[<tool>] ...
Missing required argument"), which names the field but not what the tool
expects of it, and never says whether anything ran. That was the fact the
caller needed most in task 5979 (reify #7913), after a timeout on the same
call; ``plans/confusion-reduction-prd.md`` is the wider motivation.

:class:`MissingArgumentMiddleware` rewrites exactly that failure into a
structured JSON refusal keyed by :data:`MISSING_ARGUMENT_CODE`. Every other
error passes through unchanged. It is not re-exported from ``shared/__init__``,
so ``import shared`` does not pull in fastmcp.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import mcp.types as mt
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.base import ToolResult
from pydantic import ValidationError

logger = logging.getLogger(__name__)

MISSING_ARGUMENT_CODE: str = 'missing_required_argument'

_MISSING_ERROR_TYPE = 'missing_argument'

_HINT = (
    'The call was rejected at argument validation, BEFORE the tool ran: nothing '
    'was executed, enqueued or written, so resubmitting is safe. Re-issue it '
    'with every argument already sent plus the missing one(s), whose declared '
    'schema is under `missing`.'
)


class MissingArgumentMiddleware(Middleware):
    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        try:
            return await call_next(context)
        except ValidationError as exc:
            parameters = await self._live_parameters(context)
            if parameters is None:
                raise
            arguments = context.message.arguments or {}
            missing = _missing_required(parameters.get('required') or [], arguments)
            if not missing:
                raise
            refusal = _refusal(
                context.message.name,
                parameters.get('properties') or {},
                missing,
                arguments,
                exc.errors(include_input=False, include_url=False),
            )
            raise ToolError(json.dumps(refusal)) from exc

    @staticmethod
    async def _live_parameters(
        context: MiddlewareContext[mt.CallToolRequestParams],
    ) -> Mapping[str, Any] | None:
        name = context.message.name
        fastmcp_context = context.fastmcp_context
        tool = None
        try:
            if fastmcp_context is not None:
                tool = await fastmcp_context.fastmcp.get_tool(name)
        except Exception:
            logger.warning(
                'missing-argument guard could not resolve the schema for %r; '
                'passing the original validation error through',
                name,
                exc_info=True,
            )
            return None
        parameters = tool.parameters if tool is not None else None
        if not isinstance(parameters, dict):
            logger.warning(
                'missing-argument guard found no usable schema for %r (got %s); '
                'passing the original validation error through',
                name,
                type(parameters).__name__,
            )
            return None
        return parameters


def _missing_required(required: Sequence[str], arguments: Mapping[str, Any]) -> list[str]:
    return [name for name in required if name not in arguments]


def _refusal(
    tool: str,
    properties: Mapping[str, Any],
    missing: Sequence[str],
    arguments: Mapping[str, Any],
    errors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered_missing = [name for name in properties if name in missing]
    ordered_missing += [name for name in missing if name not in properties]
    return {
        'error': f'{tool} was called without required argument(s): {", ".join(ordered_missing)}',
        'code': MISSING_ARGUMENT_CODE,
        'tool': tool,
        'missing': [
            {'name': name, 'schema': dict(properties.get(name, {}))} for name in ordered_missing
        ],
        'provided': list(arguments),
        'example_call': _example_call(tool, properties, missing, arguments),
        'hint': _HINT,
        'other_errors': [
            {'type': error['type'], 'loc': list(error['loc']), 'msg': error['msg']}
            for error in errors
            if error['type'] != _MISSING_ERROR_TYPE
        ],
    }


def _example_call(
    tool: str,
    properties: Mapping[str, Any],
    missing: Sequence[str],
    arguments: Mapping[str, Any],
) -> str:
    rendered = []
    for name, node in properties.items():
        if name in arguments:
            rendered.append(f'{name}=<as sent>')
        elif name in missing:
            declared_type = node.get('type') if isinstance(node, Mapping) else None
            rendered.append(f'{name}=<{declared_type if isinstance(declared_type, str) else "value"}>')
    return f'{tool}({", ".join(rendered)})'
