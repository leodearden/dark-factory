"""A FastMCP tool call's result, as the dict the tool returned.

Tests that drive a fused-memory tool through ``server._tool_manager.call_tool``
get back either the tool's own dict or a list of MCP content blocks whose first
carries that dict as JSON text. Reading it through here keeps that return shape
in one place, so a FastMCP change to it is absorbed once rather than per file.

Lives outside conftest.py, under a module name unique across subprojects, for
the reason ``_ast_guard`` records.
"""

import json
from typing import Any


def tool_result_dict(result: Any) -> dict[str, Any]:
    """*result* of a ``call_tool`` as the tool's dict, whichever shape it arrived in."""
    if isinstance(result, list):
        first = result[0]
        return json.loads(first.text if hasattr(first, 'text') else str(first))
    return result
