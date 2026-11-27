"""Journal every stored-markup fact a tool response discloses (task 11978).

The read-time repair (``plan_markup_repair.read_plan_repaired``) reports what it
found in damage already on disk only on the response, under
:data:`~orchestrator.mcp.plan_markup_repair.MARKUP_REPAIRS_KEY`, and on a
per-process stderr. That left task 3382's unrepairable decision with no durable
record anywhere.

This observer sits at the RESPONSE layer rather than inside the repair for
three reasons. The repair runs under ``TaskArtifacts.plan_mutex``, which must
not be held across the journal's ``git rev-parse``. The host's one journal is
async and already built by the server. And journalling exactly what the caller
was shown keeps the response the single source of what was disclosed.

Each line is the fact as disclosed, tagged :data:`STORED_MARKUP_FACT`, which
says "damage already stored was found by this read", never "this call leaked".
The fact's ``tool`` is the field's writer, so the read is named apart:
:data:`READ_BY_TOOL_KEY` is the invoked tool and :data:`READ_BY_ROLE_KEY` the
role the host serves, ``None`` when it was not told. What a line may carry is
``orchestrator/src/orchestrator/mcp/markup_journal.py``'s rule.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

import mcp.types as mt
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.base import ToolResult

from orchestrator.mcp.plan_markup_repair import MARKUP_REPAIRS_KEY

logger = logging.getLogger(__name__)

STORED_MARKUP_FACT = 'stored_markup_detected'
READ_BY_TOOL_KEY = 'read_by_tool'
READ_BY_ROLE_KEY = 'read_by_role'


class StoredMarkupJournalMiddleware(Middleware):
    def __init__(
        self,
        journal: Callable[[dict[str, Any]], Awaitable[object]],
        reader_role: str | None = None,
    ) -> None:
        self._journal = journal
        self._reader_role = reader_role

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        result = await call_next(context)
        reader = {
            READ_BY_TOOL_KEY: context.message.name,
            READ_BY_ROLE_KEY: self._reader_role,
        }
        for fact in _disclosed_facts(result.structured_content):
            try:
                await self._journal({**fact, 'fact': STORED_MARKUP_FACT, **reader})
            except Exception:
                logger.exception(
                    'stored markup journal: could not journal the fact for %s.%s',
                    fact.get('collection'), fact.get('field'),
                )
        return result


def _disclosed_facts(structured: object) -> list[dict[str, Any]]:
    if not isinstance(structured, dict):
        return []
    disclosed = structured.get(MARKUP_REPAIRS_KEY)
    if not isinstance(disclosed, list):
        return []
    return [fact for fact in disclosed if isinstance(fact, dict)]
