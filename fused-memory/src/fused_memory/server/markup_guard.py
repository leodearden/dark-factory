"""Attach beta's markup guard to fused-memory's BUNDLED FastMCP (task 4458).

PRD ``plans/toolcall-markup-containment-prd.md``, leaf gamma-3. Task 3689
(beta) delivered :class:`shared.mcp_markup_middleware.MarkupGuardMiddleware` as
a :class:`fastmcp.server.middleware.Middleware`, attached with
``mcp.add_middleware(...)``. fused-memory does not run that FastMCP.

## The measured substrate difference

fused-memory's server is :class:`mcp.server.fastmcp.FastMCP` — the FastMCP
BUNDLED inside the ``mcp`` SDK — not the standalone ``fastmcp`` distribution.
Probed in fused-memory's own venv:

* bundled ``FastMCP``: ``add_middleware`` ABSENT, ``get_tool`` ABSENT. Its tool
  surface is ``{add_tool, call_tool, list_tools, remove_tool, tool}``.
* standalone ``fastmcp`` 3.2.2 ``FastMCP``: both present.

So the documented attachment point does not exist here. This module bridges the
gap at the one chokepoint the bundled class does have — ``ToolManager.call_tool``
— reusing the middleware class UNMODIFIED (INV-5: no second implementation of
detection, repair, policy or payload shape lives here, and nothing is imported
from it but its public names).

Two further substrate deltas, both measured rather than assumed:

* ``ToolManager.get_tool(name)`` is SYNC here and returns ``Tool | None``; on
  the standalone class it is a coroutine. :class:`_ShimFastMCP` wraps it in an
  ``async def`` so ``MarkupGuardMiddleware._schema_params``, which awaits it,
  needs no adaptation. ``Tool.parameters`` is the same full JSON Schema dict on
  both classes, so the parameter names are under ``'properties'`` either way.
* Argument validation lives INSIDE ``Tool.run``
  (``fn_metadata.call_fn_with_arg_validation``), i.e. strictly AFTER this
  chokepoint. That is what preserves the guard's headline benefit here: a leak
  that swallowed a REQUIRED parameter is repaired before pydantic ever sees the
  call. Unguarded, that same call fails with a bare ``Field required``; behind
  the guard it is rejected with the repair attached.

## Why this RAISES, and why installation order matters

The guard lets :class:`fastmcp.exceptions.ToolError` propagate, exactly as
``MarkupGuardMiddleware._reject`` raises it. Measured: returning the rejection
as a bare dict from this chokepoint works at the ``_tool_manager`` layer and for
tools annotated ``-> dict``, but a tool annotated ``-> str`` fails with "Output
validation error: 'result' is a required property" and the rejection payload is
DESTROYED. Raising is safe for every output schema and keeps fused-memory's
caller-facing contract byte-identical to the three standalone servers.

That forces the installation ORDER, which is asserted by a test rather than
merely described here: this guard must be installed AFTER — and therefore
OUTSIDE — ``main._install_safe_tool_wrapper``. That wrapper catches
``BaseException`` and the bundled ``Tool.run`` wraps EVERY tool-body exception
into ``ToolError``, so teaching it to re-raise ``ToolError`` would gut the
containment ``tests/test_tool_safe_wrapper.py`` pins. As the OUTER wrapper this
guard's ``ToolError`` reaches the lowlevel server intact while the
defence-in-depth wrapper still contains everything the tool bodies raise.

Consequence for tests: the guard is installed by ``main.py``, NOT by
``create_mcp_server``, so a test must install it explicitly — the same shape
``_install_safe_tool_wrapper`` already has.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, cast

from shared.mcp_markup_middleware import MarkupGuardMiddleware, RepairPolicy

from fused_memory.server.markup_tripwire import (
    _MARKUP_STORM_THRESHOLD,
    _MARKUP_STORM_WINDOW_SECONDS,
    emit_markup_storm_escalation,
)

logger = logging.getLogger(__name__)


#: Tools DECLARED to carry envelope literals as ordinary DATA, and therefore not
#: scanned (INV-1: a declared, machine-checkable opt-out, never a heuristic).
#:
#: Names are BARE in-server names. ``context.message.name`` is
#: ``scan_memory_content``, not the agent-facing
#: ``mcp__fused-memory__scan_memory_content`` that the specimen corpus records,
#: so an exemption declared with the prefixed spelling would silently never
#: match — and a machine-checked declaration that fails open is worse than none.
#:
#: One name, with the reason it is here:
#:
#: * ``scan_memory_content`` — PRD boundary row B7. Its whole job is scanning the
#:   corpus for exactly these substrings, so guarding it would make the
#:   retroactive-sweep tool unable to look for the thing it was built to find.
#:   It is the sanctioned tool for that lookup precisely because the fragments
#:   carry almost no semantic signal (a live 2026-07-26 semantic probe for the
#:   known corrupted records returned ZERO), so there is no other path to the
#:   same answer. Load-bearing on a MEASURED shape, not a precaution: ``needles``
#:   is declared ``list[str] | None`` and the bundled dispatcher accepts it as a
#:   JSON STRING and coerces it, which makes it a scanned string argument — and
#:   without this exemption a sweep for the envelope literals is refused with
#:   ``mcp_markup_unrepairable``.
#:
#: Surveyed for other legitimate carriers and deliberately NOT exempted:
#: ``redact_episode_content``, whose ``new_content`` is the CLEAN replacement
#: text for a corrupted episode — markup there is a defect, so rejecting it is
#: correct; and ``search``, whose ``query`` is not a sanctioned carrier because
#: ``scan_memory_content`` is explicitly the tool for literal substring lookup.
EXEMPT_TOOLS: frozenset[str] = frozenset({'scan_memory_content'})


#: The anchor this guard files bursts under. Deliberately NOT markup_tripwire's
#: ``markup-tripwire``: the L1 escalation watcher files its own cluster records
#: under that id and SQUATS it — measured, the tripwire filed nothing
#: 2026-08-16..2026-08-19 while 41 rejections occurred, and all 17 records sat at
#: dedupe_count 0. Deduping against a squatted anchor is suppression that reads
#: as calm, so this guard needs an anchor of its own.
_STORM_ANCHOR_TASK_ID = 'markup-guard'

#: Where an operator should take a recurrence. DF 3083 delivered the root cause
#: and the corpus sweep but is DONE and CLOSED to appends, so a reader sent there
#: reports it nowhere. This text carries forward the correction commit e0ea6e3fe9
#: made to the ERROR line inside tools.py's ``_markup_gate`` closure, which is
#: deleted when the in-line gates retire — so it has to live here to survive.
_STORM_ROUTING = (
    'report the recurrence against plans/toolcall-markup-containment-prd.md '
    '(DF 3083 is done and closed to appends, so not against 3083)'
)


def _resolve_project_root(project: Any, known_projects: dict[str, str] | None) -> str | None:
    """Turn the middleware's ``project`` label into a filesystem project_root.

    ``MarkupGuardMiddleware._identity`` resolves ``project_root`` FIRST then
    ``project_id``, so the label arrives in one of two shapes depending on which
    tool leaked: the task tools carry a ``project_root`` and yield a PATH, while
    the memory tools carry only a ``project_id`` and yield a BARE ID like
    ``dark_factory``. Filing against a bare id would open a queue at a relative
    directory named after the project.

    So an absolute path passes through unchanged, and anything else is looked up
    in *known_projects* — the same ``_kp.get(project_id)`` translation the
    write-time gate does at its four call sites.

    An unresolvable project yields ``None`` and files NOTHING. An escalation
    filed against a guessed default is worse than one an operator has to place
    by hand.
    """
    if not isinstance(project, str) or not project:
        return None
    if project.startswith('/'):
        return project
    if known_projects:
        return known_projects.get(project)
    return None


@dataclass
class _ShimMessage:
    """The ``context.message`` shape ``on_call_tool`` reads.

    NOT frozen, and ``arguments`` is a mutable dict the guard is free to edit
    in place: ``_apply_override`` deletes or rewrites ``metadata`` here, and
    :func:`install_markup_guard`'s ``call_next`` re-reads this attribute so the
    edit lands on the forwarded call.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


class _ShimFastMCP:
    """``context.fastmcp_context.fastmcp``, with ``get_tool`` made awaitable.

    The ONE adaptation the substrate needs. The bundled ``ToolManager.get_tool``
    is synchronous; ``MarkupGuardMiddleware._schema_params`` awaits it. Rather
    than branch inside the middleware (which would put substrate knowledge in
    the shared layer), the difference is absorbed here.
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
    """The whole middleware-facing context surface, which is exactly three paths.

    Verified exhaustively against the middleware source: it touches
    ``context.message.name``, ``context.message.arguments`` and
    ``context.fastmcp_context.fastmcp.get_tool`` — nothing else. Anything the
    real fastmcp context carries beyond those is unused here, so a faithful
    stand-in is this small rather than a partial mock of a large object.
    """

    message: _ShimMessage
    fastmcp_context: _ShimFastMCPContext


def install_markup_guard(
    mcp: Any,
    *,
    policy: RepairPolicy,
    known_projects: dict[str, str] | None = None,
    exempt_tools: frozenset[str] = EXEMPT_TOOLS,
) -> None:
    """Wrap ``mcp._tool_manager.call_tool`` with the shared markup guard.

    *policy* is REQUIRED and keyword-only: the declared tier for this server
    belongs at the interception point, never inferred per call from the shape of
    the damage or from a tool's name (INV-1).

    *known_projects* is the ``project_id -> project_root`` registry the storm
    escalation sink uses to place a burst in the right queue. Accepted here and
    threaded to the sink; the guard's rejection behaviour does not depend on it.

    *exempt_tools* names the tools allowed to carry envelope literals as data.

    Wrapping ``call_tool`` rather than each tool covers every currently
    registered tool and any added later, in one place — the same reasoning
    ``main._install_safe_tool_wrapper`` records for choosing this chokepoint.

    Idempotent: re-entry (which tests do) leaves the existing wrapping in place
    rather than double-wrapping, mirroring that wrapper's
    ``_fused_memory_safe_wrapped`` sentinel.

    Raises:
        ValueError: for any *policy* but ``REJECT_WITH_REPAIR``. Not a
            preference — ``MarkupGuardMiddleware._forward`` returns a fastmcp
            ``ToolResult`` that the bundled dispatcher does not understand, so
            ``FORWARD_REPAIR`` is genuinely unsupported at this site. A
            declarable tier that cannot work must fail loudly at wiring time
            rather than silently at the first repair.
    """
    if policy is not RepairPolicy.REJECT_WITH_REPAIR:
        raise ValueError(
            f'markup guard policy {policy!r} is not supported on the bundled '
            f'FastMCP: MarkupGuardMiddleware._forward returns a fastmcp '
            f'ToolResult that mcp.server.fastmcp\'s dispatcher cannot consume. '
            f'Only {RepairPolicy.REJECT_WITH_REPAIR!r} is installable here.'
        )

    tool_manager = mcp._tool_manager
    if getattr(tool_manager, '_fused_memory_markup_guarded', False):
        return

    def _escalation_sink(record: dict[str, Any]) -> None:
        """File a fired burst, and never change the rejection that fired it.

        Receives the middleware's
        ``{'error_type': 'mcp_markup_storm', count, threshold, window_seconds,
        outcome, project}`` record. The middleware delegates dedup here on
        purpose — whether an open escalation already exists is queue knowledge
        the shared layer does not have and must not guess at.
        """
        project_root = _resolve_project_root(record.get('project'), known_projects)
        if project_root is None:
            logger.error(
                'markup_guard_storm: %s %s outcome(s) in %ss but project=%r could '
                'not be resolved to a project_root, so NO escalation was filed — '
                'the serialization leak is ACTIVE; %s',
                record.get('count'), record.get('outcome'),
                record.get('window_seconds'), record.get('project'), _STORM_ROUTING,
            )
            return

        # ONE greppable operator-facing ERROR line. The storm summary folded
        # into the tool response reaches only the leaking caller — the one party
        # that already knows — so the operator's copy cannot ride on it. Naming
        # the count here is also the visible-counter half of the remedy, without
        # an escalation-schema change.
        logger.error(
            'markup_guard_storm: %s %s outcome(s) in %ss for project_root=%r '
            '(threshold=%s) — the serialization leak is ACTIVE; %s',
            record.get('count'), record.get('outcome'), record.get('window_seconds'),
            project_root, record.get('threshold'), _STORM_ROUTING,
        )

        # Purely additive: the rejection is already decided, so a queue failure
        # costs the operator a heads-up and nothing else. Same reasoning the
        # retiring _markup_gate applies to its own emitter — and
        # emit_markup_storm_escalation is itself built never to raise, so this
        # guards only against a defect in it.
        try:
            emit_markup_storm_escalation(
                project_root, record, anchor_task_id=_STORM_ANCHOR_TASK_ID
            )
        except Exception:
            logger.exception(
                'markup_guard: emit_markup_storm_escalation raised for '
                'project_root=%r; storm %r not escalated',
                project_root, record,
            )

    original_call_tool = tool_manager.call_tool
    guard = MarkupGuardMiddleware(
        policy,
        exempt_tools,
        escalation_sink=_escalation_sink,
        # ONE source for the tuning across both guards, so the in-line gate and
        # the boundary cannot drift to different thresholds while both are live.
        storm_threshold=_MARKUP_STORM_THRESHOLD,
        storm_window_seconds=_MARKUP_STORM_WINDOW_SECONDS,
        # fact_sink is deliberately left unwired: the middleware already emits
        # the structured fact unconditionally as a WARNING (INV-2), and
        # fused-memory has no queue-backed CONSUMER for the stream. This task's
        # consumers are the calling agent (served by repaired_call) and the
        # operator (served by the storm escalation above). Left explicit so a
        # future consumer wires it deliberately rather than inheriting a guess.
        fact_sink=None,
    )
    shim_fastmcp = _ShimFastMCPContext(_ShimFastMCP(tool_manager))

    async def _guarded_call_tool(
        name: str, arguments: dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        # A COPY: the guard mutates ``arguments`` in place on the override
        # path, and a wrapper has no licence to edit the caller's own dict.
        context = _ShimContext(_ShimMessage(name, dict(arguments or {})), shim_fastmcp)

        async def call_next(ctx: _ShimContext) -> Any:
            # Re-read name and arguments off the context rather than closing
            # over the originals, so an in-place repair reaches the tool.
            # ``*args``/``**kwargs`` — the bundled signature's ``context`` and
            # ``convert_result`` — pass through untouched.
            return await original_call_tool(
                ctx.message.name, ctx.message.arguments, *args, **kwargs
            )

        # ToolError is deliberately NOT caught: see the module docstring.
        #
        # ``cast`` at this ONE site, and nowhere else, is the whole nominal cost
        # of the adaptation. ``on_call_tool`` is annotated against fastmcp's
        # ``MiddlewareContext[CallToolRequestParams]``, a class the bundled SDK
        # does not have and cannot construct; the shim satisfies the three
        # attribute paths the method actually reads (verified exhaustively
        # against the middleware source) but is not a nominal subtype. Keeping
        # the cast here rather than typing the shims as ``Any`` preserves real
        # type checking inside the shim classes themselves.
        return await guard.on_call_tool(cast(Any, context), cast(Any, call_next))

    tool_manager.call_tool = _guarded_call_tool
    tool_manager._fused_memory_markup_guarded = True
    logger.info(
        'markup boundary guard installed at ToolManager.call_tool '
        '(policy=%s, exempt=%s)',
        policy, sorted(exempt_tools),
    )
