"""Run a chain of fastmcp ``Middleware.on_call_tool`` at the BUNDLED FastMCP's dispatch chokepoint.

fused-memory's servers are :class:`mcp.server.fastmcp.FastMCP` — the FastMCP
BUNDLED inside the ``mcp`` SDK — not the standalone ``fastmcp`` distribution.
Probed in fused-memory's own venv:

* bundled ``FastMCP``: ``add_middleware`` ABSENT, ``get_tool`` ABSENT. Its tool
  surface is ``{add_tool, call_tool, list_tools, remove_tool, tool}``.
* standalone ``fastmcp`` 3.2.2 ``FastMCP``: both present.

So a fastmcp middleware cannot be attached the documented way. This module
attaches an ordered chain of them at the chokepoint the bundled class does
have — ``ToolManager.call_tool`` — as ONE wrapping, driving each member
UNMODIFIED through a stand-in context. Two further substrate deltas, both
measured:

* ``ToolManager.get_tool(name)`` is SYNC here and returns ``Tool | None``; on
  the standalone class it is a coroutine. :class:`_ShimFastMCP` wraps it in an
  ``async def``, so ``shared.mcp_tool_schema.live_tool_parameters``, which
  awaits it, needs no adaptation. ``Tool.parameters`` is the same full JSON
  Schema dict on both classes.
* Argument validation lives INSIDE ``Tool.run``
  (``fn_metadata.call_fn_with_arg_validation``), strictly AFTER this
  chokepoint, so a member here sees the arguments exactly as sent.

## What a member is handed, and what the caller gets

The innermost ``call_next`` hands the chain a :class:`_DispatchedResult`: the
dispatcher's raw value seen through the three public ``ToolResult`` names a
member reads, ``content``, ``structured_content`` and ``meta``. When a view
comes back out of the chain the raw value is returned untouched, so a chain
whose members only pass through or reject is byte-identical to no chain.

A member with something to say returns its own fastmcp ``ToolResult``, and only
that object's public fields are read. Under ``convert_result=True`` (the
low-level handler's path) it becomes a ``types.CallToolResult`` carrying
``_meta``, which that handler passes straight to the client. Under
``convert_result=False`` (in-process callers, which expect the tool's dict) the
echoed keys below land in that dict.

Keys the installer declares in *echoed_meta_keys* are also folded into the
result proper, because the agent's client renders ``structuredContent`` and
drops ``_meta`` (``plans/uuid-prefix-resolution-prd.md`` §11 Q1). A key lands
only where it is absent, and only when the tool's ``outputSchema`` still
accepts the folded ``structuredContent``: the mcp ``ClientSession`` raises on
one that fails it. The JSON text block gets exactly the keys
``structuredContent`` accepted, so the two stay one serialisation; a tool with
no structured content has only the text block to fold into.

## Order and containment

A member rejects by raising :class:`fastmcp.exceptions.ToolError`, and the
chain must be installed outside ``main._install_safe_tool_wrapper``;
``fused_memory/server/main.py::_install_tool_dispatch_guards`` owns that order
and says why.

Being outside that wrapper also puts each member's OWN defects beyond its
containment, so every member is contained here: anything but
``CancelledError`` and ``ToolError`` raised before the member forwarded is
logged, the member's argument edits are rolled back, and the call continues to
the REST of the chain (fail OPEN — a check that crashed checked nothing, but
the checks inside it still run). After the member forwarded, the exception is
re-raised, because re-dispatching would double a write. Only the link whose
member raised logs it: an exception that came up out of a member's
``call_next`` passes the outer links silently.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import jsonschema
import pydantic_core
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware
from fastmcp.tools.base import ToolResult
from mcp import types

logger = logging.getLogger(__name__)


@dataclass
class _ShimMessage:
    """The ``context.message`` shape ``on_call_tool`` reads.

    NOT frozen, and ``arguments`` is a mutable dict a member is free to edit in
    place (the markup guard's override path deletes ``metadata`` here):
    ``call_next`` re-reads this attribute so the edit lands on the forwarded
    call, and on every member inside the one that made it.
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


@dataclass(frozen=True)
class _DispatchedResult:
    """The dispatcher's raw value, exposed as ``content`` / ``structured_content`` / ``meta``.

    A cheap view rather than a real ``ToolResult``, whose constructor would
    serialise the payload on every call's clean path.

    The raw shapes are measured: under ``convert_result=True`` a tool with an
    output schema yields a ``(content, structured)`` tuple, one without yields a
    bare content list, and a tool may return a ``CallToolResult`` itself.
    Anything else — an unconverted dispatch, or the safe wrapper's error dict —
    has nowhere to put meta, which ``carries_tool_output`` records.
    """

    raw: Any
    content: list[Any]
    structured_content: dict[str, Any] | None
    meta: dict[str, Any] | None
    carries_tool_output: bool
    is_error: bool = False

    @classmethod
    def of(cls, raw: Any, *, convert_result: bool) -> _DispatchedResult:
        if convert_result:
            if isinstance(raw, types.CallToolResult):
                return cls(raw, list(raw.content), raw.structuredContent, raw.meta, True, raw.isError)
            if isinstance(raw, tuple) and len(raw) == 2:
                return cls(raw, list(raw[0]), raw[1], None, True)
            if isinstance(raw, list):
                return cls(raw, list(raw), None, None, True)
        return cls(raw, [], None, None, False)


_CallNext = Callable[[_ShimContext], Awaitable[Any]]


def _contained(member: Middleware, call_next: _CallNext, *, label: str) -> _CallNext:
    """One chain link: *member*'s ``on_call_tool``, with its own defects contained."""

    async def link(context: _ShimContext) -> Any:
        snapshot = dict(context.message.arguments)
        forwarded = False
        escaped_from_inside: BaseException | None = None

        async def forward(forwarded_context: _ShimContext) -> Any:
            nonlocal forwarded, escaped_from_inside
            forwarded = True
            try:
                return await call_next(forwarded_context)
            except BaseException as exc:
                escaped_from_inside = exc
                raise

        try:
            # ``cast`` here is the whole nominal cost of the adaptation:
            # ``on_call_tool`` is annotated against fastmcp's
            # ``MiddlewareContext``, which the bundled SDK cannot construct.
            return await member.on_call_tool(cast(Any, context), cast(Any, forward))
        except (asyncio.CancelledError, ToolError):
            raise
        except BaseException as exc:
            if exc is escaped_from_inside:
                raise
            name = context.message.name
            if forwarded:
                logger.exception(
                    '%s: %s raised AFTER forwarding %s; not re-dispatching '
                    '(a second call would double the write)',
                    label, type(member).__name__, name,
                )
                raise
            logger.exception(
                '%s: %s raised while inspecting %s; failing OPEN to the rest of '
                'the chain — this call was NOT scanned by it',
                label, type(member).__name__, name,
            )
            context.message.arguments = snapshot
            return await call_next(context)

    return link


def _accepts(schema: dict[str, Any] | None, instance: dict[str, Any]) -> bool:
    if schema is None:
        return True
    try:
        jsonschema.validate(instance=instance, schema=schema)
    except jsonschema.ValidationError:
        return False
    return True


def _accepted_echoes(
    structured: dict[str, Any] | None,
    echoes: dict[str, Any],
    output_schema: dict[str, Any] | None,
) -> dict[str, Any]:
    """The *echoes* that may join the result: all of them without *structured*, else the absent keys its schema still accepts."""
    if structured is None:
        return dict(echoes)
    accepted: dict[str, Any] = {}
    for key, value in echoes.items():
        if key not in structured and _accepts(output_schema, {**structured, **accepted, key: value}):
            accepted[key] = value
    return accepted


def _fold_text(content: list[Any], echoes: dict[str, Any]) -> list[Any]:
    """Fold *echoes* into a lone JSON-object text block, as the bundled formatter writes it."""
    if not echoes or len(content) != 1 or not isinstance(content[0], types.TextContent):
        return content
    try:
        payload = json.loads(content[0].text)
    except ValueError:
        return content
    if not isinstance(payload, dict):
        return content
    absent = {key: value for key, value in echoes.items() if key not in payload}
    if not absent:
        return content
    text = pydantic_core.to_json({**payload, **absent}, fallback=str, indent=2).decode()
    return [content[0].model_copy(update={'text': text})]


def _deliver_member_result(
    result: ToolResult,
    dispatched: _DispatchedResult | None,
    *,
    convert_result: bool,
    echoed_meta_keys: frozenset[str],
    output_schema: dict[str, Any] | None,
    label: str,
    name: str,
) -> Any:
    """Turn a member-authored ``ToolResult`` into what the bundled dispatcher's caller expects."""
    meta = dict(result.meta or {})
    echoes = {key: meta[key] for key in sorted(echoed_meta_keys) if key in meta}
    if not convert_result:
        if dispatched is None:
            return result
        raw = dispatched.raw
        if isinstance(raw, dict):
            return {**raw, **{key: value for key, value in echoes.items() if key not in raw}}
        return raw
    if dispatched is not None and not dispatched.carries_tool_output:
        logger.warning(
            '%s: a member attached meta %s to %s, but the dispatcher returned a '
            '%s, which cannot carry it; the result is delivered without it',
            label, sorted(meta), name, type(dispatched.raw).__name__,
        )
        return dispatched.raw
    structured = result.structured_content
    accepted = _accepted_echoes(structured, echoes, output_schema)
    return types.CallToolResult(
        content=_fold_text(list(result.content), accepted),
        structuredContent=None if structured is None else {**structured, **accepted},
        isError=dispatched.is_error if dispatched is not None else False,
        _meta=meta or None,
    )


def install_tool_middleware(
    mcp: Any,
    middlewares: Sequence[Middleware],
    *,
    sentinel: str,
    label: str,
    echoed_meta_keys: frozenset[str] = frozenset(),
) -> None:
    """Wrap ``mcp._tool_manager.call_tool`` ONCE with *middlewares*, outermost first.

    Wrapping ``call_tool`` rather than each tool covers every registered tool
    and any added later, in one place. Every member shares one context per
    call, which holds a COPY of the caller's arguments, so an in-place edit by
    one member is what every member inside it sees.

    *sentinel* is the tool-manager attribute that marks this chain as
    installed; a second install with the same sentinel is a no-op, so re-entry
    never double-wraps. *label* names the chain in the log lines.
    *echoed_meta_keys* names the member meta keys also folded into the result
    proper (module docstring).
    """
    tool_manager = mcp._tool_manager
    if getattr(tool_manager, sentinel, False):
        return

    members = tuple(middlewares)
    original_call_tool = tool_manager.call_tool
    shim_fastmcp = _ShimFastMCPContext(_ShimFastMCP(tool_manager))

    async def _call_tool_through_chain(
        name: str,
        arguments: dict[str, Any] | None,
        context: Any = None,
        convert_result: bool = False,
    ) -> Any:
        dispatches: list[_DispatchedResult] = []

        async def dispatch(ctx: _ShimContext) -> _DispatchedResult:
            raw = await original_call_tool(
                ctx.message.name, ctx.message.arguments,
                context=context, convert_result=convert_result,
            )
            dispatched = _DispatchedResult.of(raw, convert_result=convert_result)
            dispatches.append(dispatched)
            return dispatched

        call_next: _CallNext = dispatch
        for member in reversed(members):
            call_next = _contained(member, call_next, label=label)

        # A COPY: a member may edit ``arguments`` in place, and a wrapper has
        # no licence to edit the caller's own dict.
        returned = await call_next(
            _ShimContext(_ShimMessage(name, dict(arguments or {})), shim_fastmcp)
        )
        if isinstance(returned, _DispatchedResult):
            return returned.raw
        if not isinstance(returned, ToolResult):
            return returned
        tool = tool_manager.get_tool(name)
        return _deliver_member_result(
            returned,
            dispatches[-1] if dispatches else None,
            convert_result=convert_result,
            echoed_meta_keys=echoed_meta_keys,
            output_schema=tool.output_schema if tool is not None else None,
            label=label,
            name=name,
        )

    tool_manager.call_tool = _call_tool_through_chain
    setattr(tool_manager, sentinel, True)
