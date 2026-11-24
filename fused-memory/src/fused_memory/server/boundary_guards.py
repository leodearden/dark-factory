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
3. the uuid-prefix guard (:class:`shared.uuid_prefix_guard.UuidPrefixGuardMiddleware`),
   innermost of the three, per the order
   ``plans/review-all-s15-fused-memory-mcp-server-boundary-prd.md`` D2 fixes.
   Present exactly when a prefix resolver is wired, and declared never to
   repair :data:`UUID_PREFIX_EXEMPT_PATHS`.

The safe wrapper must already be installed INSIDE this chain;
``fused_memory/server/main.py::_install_tool_dispatch_guards`` owns that
order and says why.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from fastmcp.server.middleware import Middleware
from shared.mcp_markup_middleware import RepairPolicy
from shared.uuid_prefix_guard import (
    FACT_UUID_PREFIX_DETECTED,
    STORM_COUNTED_OUTCOMES,
    UUID_PREFIX_REPAIR_META_KEY,
    PrefixResolver,
    UuidPrefixGuardMiddleware,
    project_from_arguments,
)

from fused_memory.backends.sqlite_task_backend import PASSTHROUGH_ONLY_METADATA_KEYS
from fused_memory.server.markup_guard import ResidueFloor, build_markup_guard
from fused_memory.server.tool_middleware_bridge import install_tool_middleware
from fused_memory.server.unknown_argument_guard import UnknownArgumentMiddleware

logger = logging.getLogger(__name__)

_BOUNDARY_GUARDED_SENTINEL = '_fused_memory_boundary_guarded'

#: Tools the uuid-prefix guard never resolves on (PRD D10): an expansion the
#: guard chose and an id the author chose are indistinguishable once a delete,
#: merge or edge rewrite has run. The PRD also names ``remove_edge``, which
#: does not exist (capability manifest, erratum 5).
UUID_PREFIX_EXEMPT_TOOLS: frozenset[str] = frozenset({
    'delete_memory',  # id-addressed and irreversible
    'delete_episode',  # id-addressed and irreversible
    'delete_entity',  # id-addressed and irreversible
    'update_edge',  # rewrites an edge by id, and soft-deletes edges
    'reassign_edge',  # rewrites an edge's endpoints by id
    'merge_entities',  # folds one node into another by id
    'consolidate_memories',  # deletes the superseded ids it names
    'scan_memory_content',  # its needles legitimately carry bare hex
})

#: Tools that forward an ambiguous prefix unchanged instead of rejecting it
#: (PRD D3): losing the write is worse than the defect.
UUID_PREFIX_FORWARD_ON_AMBIGUITY_TOOLS: frozenset[str] = frozenset({'update_task'})

#: update_task accepts these values only verbatim
#: (``sqlite_task_backend.py::_assert_passthrough_only_keys``), so repairing a
#: stored value inside them turns an exact echo into a refusal (task 11331).
UUID_PREFIX_EXEMPT_PATHS: Mapping[str, frozenset[tuple[str, ...]]] = MappingProxyType({
    'update_task': frozenset(('metadata', key) for key in PASSTHROUGH_ONLY_METADATA_KEYS),
})

#: Also folded into the result proper, because the agent's client drops
#: ``_meta`` (``plans/uuid-prefix-resolution-prd.md`` §11 Q1).
_ECHOED_META_KEYS: frozenset[str] = frozenset({UUID_PREFIX_REPAIR_META_KEY})


def _log_uuid_prefix_fact(fact: dict[str, Any]) -> None:
    """One ``uuid_prefix_detected`` line, as ``escalation/server.py::_emit_uuid_prefix_fact`` writes it.

    WARNING for the guard's fail-soft outcomes, INFO otherwise, so one grep
    reads both servers' operator streams. A lock-step duplicate of that
    emitter; task 9988 hoists one public emitter into ``shared.uuid_prefix_guard``.
    """
    level = logging.WARNING if fact.get('outcome') in STORM_COUNTED_OUTCOMES else logging.INFO
    logger.log(level, '%s %s', FACT_UUID_PREFIX_DETECTED, json.dumps(fact, sort_keys=True))


def boundary_guard_chain(
    *,
    known_projects: Mapping[str, str] | None,
    prefix_resolver: PrefixResolver | None,
    residue_floor: ResidueFloor | None = None,
) -> tuple[Middleware, ...]:
    """The chain's members, outermost first.

    *residue_floor* is the markup guard's floor under its residue queue
    (``fused_memory/server/markup_guard.py::build_markup_guard`` says what it
    takes and returns); a refused payload the queue cannot hold goes there.

    REJECT_WITH_REPAIR is the PRD's declared markup tier for fused-memory
    (``plans/toolcall-markup-containment-prd.md`` section 4, C2), declared HERE
    at the interception point rather than inferred per tool (INV-1). Whether
    fused-memory should forward repairs instead is that PRD's open question 3.
    """
    members: tuple[Middleware, ...] = (
        build_markup_guard(
            policy=RepairPolicy.REJECT_WITH_REPAIR,
            known_projects=known_projects,
            residue_floor=residue_floor,
        ),
        UnknownArgumentMiddleware(),
    )
    if prefix_resolver is None:
        return members
    return (
        *members,
        UuidPrefixGuardMiddleware(
            prefix_resolver,
            project_from_arguments(known_projects or {}),
            exempt_tools=UUID_PREFIX_EXEMPT_TOOLS,
            forward_on_ambiguity_tools=UUID_PREFIX_FORWARD_ON_AMBIGUITY_TOOLS,
            exempt_paths=UUID_PREFIX_EXEMPT_PATHS,
            # escalation_sink is unwired here: a burst still logs the storm
            # escape's ERROR line and rides on the caller's response. Wiring it
            # into a queue is task 9988.
            fact_sink=_log_uuid_prefix_fact,
        ),
    )


def install_boundary_guards(
    mcp: Any,
    *,
    known_projects: Mapping[str, str] | None,
    prefix_resolver: PrefixResolver | None,
    residue_floor: ResidueFloor | None = None,
) -> None:
    """Install :func:`boundary_guard_chain` on *mcp* as one wrapping; idempotent.

    *known_projects* and *prefix_resolver* are keyword-only with no default, so
    every caller declares them at the interception point. Without
    *prefix_resolver* the uuid-prefix member is absent, and the first install
    says so once. *residue_floor* reaches the markup guard unchanged;
    ``fused_memory/server/main.py::_install_tool_dispatch_guards`` requires one.
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
        boundary_guard_chain(
            known_projects=known_projects,
            prefix_resolver=prefix_resolver,
            residue_floor=residue_floor,
        ),
        sentinel=_BOUNDARY_GUARDED_SENTINEL,
        label='boundary guards',
        echoed_meta_keys=_ECHOED_META_KEYS,
    )
