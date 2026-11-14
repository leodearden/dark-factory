"""fused-memory's MCP boundary guards: one ordered chain, one wrapping (task 4643).

Every call to the primary server passes through these members, outermost
first, installed as ONE wrapping of ``ToolManager.call_tool`` by
:func:`fused_memory.server.tool_middleware_bridge.install_tool_middleware`:

1. the markup guard (:func:`fused_memory.server.markup_guard.build_markup_guard`),
   OUTERMOST. Its override, ``metadata={'allow_mcp_markup': True}``, is legal
   on every tool, and on a tool that declares no ``metadata`` the guard strips
   it before forwarding, so every member inside it must see the post-strip
   call. Any parameter it repairs is likewise what the members inside it see;
2. the unknown-argument guard
   (:class:`fused_memory.server.unknown_argument_guard.UnknownArgumentMiddleware`),
   which refuses an argument the tool does not declare;
3. the uuid-prefix guard, innermost of the three, per the order
   ``plans/review-all-s15-fused-memory-mcp-server-boundary-prd.md`` D2 fixes.

The safe wrapper must already be installed INSIDE this chain;
``fused_memory/server/main.py::_install_tool_dispatch_guards`` owns that
order and says why.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from fastmcp.server.middleware import Middleware
from shared.mcp_markup_middleware import RepairPolicy
from shared.uuid_prefix_guard import PrefixResolver

from fused_memory.server.markup_guard import build_markup_guard
from fused_memory.server.tool_middleware_bridge import install_tool_middleware
from fused_memory.server.unknown_argument_guard import UnknownArgumentMiddleware

logger = logging.getLogger(__name__)

_BOUNDARY_GUARDED_SENTINEL = '_fused_memory_boundary_guarded'


def boundary_guard_chain(
    *,
    known_projects: Mapping[str, str] | None,
    prefix_resolver: PrefixResolver | None,
) -> tuple[Middleware, ...]:
    """The chain's members, outermost first.

    REJECT_WITH_REPAIR is the PRD's declared markup tier for fused-memory
    (``plans/toolcall-markup-containment-prd.md`` section 4, C2), declared HERE
    at the interception point rather than inferred per tool (INV-1). Whether
    fused-memory should forward repairs instead is that PRD's open question 3.
    """
    return (
        build_markup_guard(policy=RepairPolicy.REJECT_WITH_REPAIR, known_projects=known_projects),
        UnknownArgumentMiddleware(),
    )


def install_boundary_guards(
    mcp: Any,
    *,
    known_projects: Mapping[str, str] | None,
    prefix_resolver: PrefixResolver | None,
) -> None:
    """Install :func:`boundary_guard_chain` on *mcp* as one wrapping; idempotent.

    Both arguments are keyword-only with no default, so every caller declares
    them at the interception point. Without *prefix_resolver* the uuid-prefix
    member is absent, and the first install says so once.
    """
    if getattr(mcp._tool_manager, _BOUNDARY_GUARDED_SENTINEL, False):
        return
    if prefix_resolver is None:
        logger.warning(
            'boundary guards: no prefix resolver is wired, so the uuid-prefix '
            'member is absent and bare uuid prefixes reach the tools unresolved'
        )
    install_tool_middleware(
        mcp,
        boundary_guard_chain(known_projects=known_projects, prefix_resolver=prefix_resolver),
        sentinel=_BOUNDARY_GUARDED_SENTINEL,
        label='boundary guards',
    )
