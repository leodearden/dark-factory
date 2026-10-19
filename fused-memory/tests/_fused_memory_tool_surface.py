"""The fused-memory MCP server's live tool surface, as recon-stage tests see it.

One home for the live enumerator and the reviewed-safe set, shared by the
classification guard
(``tests/test_stages.py::TestDisallowedToolLists::test_every_fused_memory_tool_is_classified``),
the per-tool stage-scoping pins
(``tests/test_recon_stage_fused_memory_tool_classification.py``) and the tests
that hold a stage prompt's advertised tools to the registered surface. The live
surface is enumerated, never a hand-kept list: a constant-only test pins what
the list says, not what the server exposes.

Lives outside conftest.py, under a module name unique across subprojects, for
the reason ``_ast_guard`` records.
"""

import asyncio
import functools
import re
from collections.abc import Mapping
from types import MappingProxyType
from unittest.mock import AsyncMock

from mcp.types import Tool

from fused_memory.server.tools import create_mcp_server

FUSED_MEMORY_TOOL_PREFIX = 'mcp__fused-memory__'

# The reviewed-safe half of the classification: tools that only READ, so a
# stage — Stage 3 included — may hold them. Adding a name here is a decision
# that the tool mutates nothing; it is not a formality.
REVIEWED_FUSED_MEMORY_STAGE_SAFE = frozenset(
    f'{FUSED_MEMORY_TOOL_PREFIX}{name}'
    for name in (
        'search',
        'count_memories_by_metadata',
        'get_cycle_summary_presence',
        'scan_memory_content',
        'get_memories_by_metadata',
        'get_memory_by_id',
        'get_entity',
        'get_entity_by_uuid',
        'get_episodes',
        'get_status',
        'get_queue_stats',
        'get_curator_state',
        'get_wal_status',
        'get_dead_letters',
        'get_tasks',
        'get_statuses',
        'get_external_statuses',
        'get_task',
        'list_tickets',
        'search_tasks',
        'find_tasks_by_metadata',
        'get_pin_queue',
        'get_scheduler_state',
        'get_scheduler_events',
    )
)


@functools.cache
def registered_fused_memory_tools() -> Mapping[str, Tool]:
    """Every registered tool, keyed by the id a stage sees: ``mcp__fused-memory__<name>``.

    Each value is the listed ``Tool``, so ``inputSchema`` carries the
    parameter schema an agent is shown. Call from a synchronous test: the
    listing runs on its own event loop.
    """
    svc = AsyncMock()
    svc.durable_queue = None
    server = create_mcp_server(svc)
    return MappingProxyType({
        f'{FUSED_MEMORY_TOOL_PREFIX}{tool.name}': tool
        for tool in asyncio.run(server.list_tools())
    })


def fused_memory_names(tools) -> frozenset[str]:
    """The fused-memory ids among *tools* (a stage disallow list mixes servers)."""
    return frozenset(name for name in tools if name.startswith(FUSED_MEMORY_TOOL_PREFIX))


_PREFIXED_ID_RE = re.compile(re.escape(FUSED_MEMORY_TOOL_PREFIX) + r'([a-z_]+)')
_BACKTICKED_NAME_RE = re.compile(r'`([a-z_]+)[`(]')


def fused_memory_tool_names_mentioned(text: str) -> frozenset[str]:
    """Bare tool names *text* mentions: prefixed ids, plus backticked bare names.

    Over-collects on purpose (a backticked parameter name is collected too):
    callers intersect the result with tool names, so an extra token is inert.
    """
    return frozenset(_PREFIXED_ID_RE.findall(text)) | frozenset(_BACKTICKED_NAME_RE.findall(text))


def denied_fused_memory_tool_names(*stage_lists) -> frozenset[str]:
    """Bare names of the fused-memory tools any of *stage_lists* denies."""
    return frozenset(
        name.removeprefix(FUSED_MEMORY_TOOL_PREFIX)
        for stage_list in stage_lists
        for name in fused_memory_names(stage_list)
    )
