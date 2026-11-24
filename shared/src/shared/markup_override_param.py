"""Declaring the markup guard's deliberate-quoting ``metadata`` parameter.

The boundary guard,
``shared/src/shared/mcp_markup_middleware.py::MarkupGuardMiddleware``, honours
``metadata={MARKUP_OVERRIDE_KEY: True}`` as a caller's statement that envelope
markup in its arguments is quoted deliberately. A tool that does not DECLARE
``metadata`` leaves that remediation outside its own advertised contract, so
:func:`accepts_markup_override` declares it on a tool once, describes it in
the schema, and consumes the map before the tool body runs.

This module needs nothing from the middleware: it reads the flag through
:func:`shared.toolcall_markup.strip_markup_override` and spells it from
:data:`shared.toolcall_markup.MARKUP_OVERRIDE_KEY`, the same two names the
middleware uses, so both halves of the hatch name one flag (INV-5).

Like the middleware, it is not re-exported from ``shared/__init__``; import it
fully qualified.
"""
from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable
from typing import Annotated, Any

from pydantic import Field

from shared.toolcall_markup import MARKUP_OVERRIDE_KEY, strip_markup_override

logger = logging.getLogger(__name__)



#: What the declared parameter SAYS IT IS FOR, in the schema itself.
#:
#: A bare ``metadata: object|null`` on twenty agent-facing tools is an
#: invitation to the wrong call: ``create_plan`` or ``mark_step_done``
#: advertising an undescribed ``metadata`` reads exactly like somewhere to
#: attach metadata to the plan, and a caller that believes it has its payload
#: dropped by :func:`_consume_override`. Before the declaration existed
#: pydantic bounced that caller — a bad response, but a response — so the
#: description is what replaces it: the mistake is forestalled in the contract
#: rather than reported after the fact. The alternative considered and NOT
#: taken was handing the leftover key names back to the caller; both channels
#: that could carry them are closed here. Raising re-creates the bounce D12
#: deliberately removed (the flag beside another key used to be a hard error),
#: and widening a tool's RESULT to carry a guard diagnostic would rewrite the
#: structured-content contract of all twenty.
#:
#: Spelled from :data:`~shared.toolcall_markup.MARKUP_OVERRIDE_KEY` rather than
#: quoting it, so the schema and the middleware's bounce hint
#: (``shared/src/shared/mcp_markup_middleware.py::_OVERRIDE_SENTENCE``) cannot
#: name different flags (INV-5).
_OVERRIDE_DESCRIPTION = (
    "Write-time-only control for this server's tool-call markup guard, and "
    "NOT payload. Send {'" + MARKUP_OVERRIDE_KEY + "': True} to declare that "
    'envelope markup in these arguments is quoted DELIBERATELY, so the guard '
    'lets the call through instead of bouncing it. The tool persists nothing '
    'from this map — the flag is consumed before dispatch and any other key '
    'is discarded, so this is not a place to attach metadata to whatever the '
    'tool writes.'
)

#: The parameter a decorated tool advertises. Keyword-only and defaulted, so it
#: lands in ``properties`` and never in ``required``; ``dict | None`` so the
#: schema accepts the flag map or nothing at all; ``Annotated`` with a pydantic
#: ``Field`` so it also SAYS what it is for. Measured against fastmcp 3.2.2:
#: the description survives the ``__signature__`` route and lands on the
#: advertised property beside ``anyOf`` and ``default``.
_OVERRIDE_PARAMETER = inspect.Parameter(
    'metadata',
    inspect.Parameter.KEYWORD_ONLY,
    default=None,
    annotation=Annotated[
        dict[str, Any] | None, Field(description=_OVERRIDE_DESCRIPTION)
    ],
)


def _consume_override(tool: str, metadata: Any) -> None:
    """End the override map's lifecycle: drop the flag, REPORT what is left.

    The flag is write-time-only control and never payload, so a decorated tool
    body never sees it and nothing here forwards it. That leaves one question:
    what about anything the caller sent BESIDES the flag?

    Before the parameter was declared, that residue was the caller's own bug
    AND its own diagnostic — the middleware's drop branch
    (``shared/src/shared/mcp_markup_middleware.py::MarkupGuardMiddleware._apply_override``)
    left it in
    place and pydantic bounced the call with ``Unexpected keyword argument``.
    Declaring ``metadata`` removes the bounce, which is the point; silently
    eating the residue with it would not be. So the response goes and the
    REPORT stays.

    The report is a server-side log, which the party holding the bug cannot
    read. That gap is closed at the DECLARATION rather than here: the schema
    carries :data:`_OVERRIDE_DESCRIPTION`, so a caller is told what this
    parameter is and is not for before it sends the wrong thing. The two
    return channels that could carry the key names back — raising, or widening
    the tool's own result — are argued and rejected there.

    KEYS ONLY, never values — the same names-only convention the middleware's
    ``MarkupGuardMiddleware._emit_fact`` and ``MarkupGuardMiddleware._forward``
    (``shared/src/shared/mcp_markup_middleware.py``) already hold to. A record that copies the caller's payload is a second
    copy of data this boundary has no business retaining.

    The strip itself is :func:`~shared.toolcall_markup.strip_markup_override`,
    the shared helper the tripwire and the middleware already use — "what is
    the flag" is answered in one place (INV-5).

    Total for any input. The declared type makes ``dict | None`` the only
    shape pydantic lets through, so a non-dict here means something upstream
    changed; it is reported by TYPE rather than by content, and never raised.
    """
    if metadata is None:
        return
    residue = strip_markup_override(metadata)
    if isinstance(residue, dict):
        if not residue:
            # Only the flag: the expected shape, and the silent one.
            return
        leftover = sorted(residue)
    else:
        leftover = [type(residue).__name__]
    logger.warning(
        'markup override: %s was sent metadata it does not own and will not '
        'persist — leftover keys %r (names only; values are not logged)',
        tool, leftover,
    )


def _swallowing_wrapper(fn: Callable[..., Any]) -> Callable[..., Any]:
    """*fn* plus a swallowed ``metadata`` keyword. The sync half of the pair."""
    @functools.wraps(fn)
    def wrapper(*args: Any, metadata: Any = None, **kwargs: Any) -> Any:
        _consume_override(getattr(fn, '__name__', repr(fn)), metadata)
        return fn(*args, **kwargs)

    return wrapper


def _async_swallowing_wrapper(fn: Callable[..., Any]) -> Callable[..., Any]:
    """The same for a coroutine function, which must be AWAITED, not returned.

    A sync wrapper here would hand FastMCP an un-awaited coroutine as the
    tool's result — a silent wrong value rather than a failure.
    """
    @functools.wraps(fn)
    async def wrapper(*args: Any, metadata: Any = None, **kwargs: Any) -> Any:
        _consume_override(getattr(fn, '__name__', repr(fn)), metadata)
        return await fn(*args, **kwargs)

    return wrapper


def accepts_markup_override(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Declare ``metadata`` on *fn* so the deliberate-quoting hatch is LEGAL.

    The middleware's bounce hint
    (``shared/src/shared/mcp_markup_middleware.py::_OVERRIDE_SENTENCE``) tells a
    bounced caller to resubmit with
    ``metadata={'`` + :data:`~shared.toolcall_markup.MARKUP_OVERRIDE_KEY` +
    ``': True}``. A tool that does not DECLARE ``metadata`` advertises
    ``additionalProperties: false`` over exactly its own parameters, so that
    remediation is not part of its contract at all — it worked only because
    ``claude`` CLI 2.1.250 transmits the undeclared argument anyway (measured
    2026-08-28 on a transparent stdio JSON-RPC tee proxy, task 4817 /
    esc-4817-1). A stricter client is entitled to refuse to send it, which
    would leave a caller bounced with a hint it cannot act on. Decorating is
    how a server makes the hatch a contract instead of a client accident.

    ONE DECLARATION, ONE CONSUMPTION POINT. The parameter is spelled here and
    only here, rather than edited into each tool's signature — twenty copies of
    a declaration nobody's body reads is the lock-step duplication INV-5 exists
    to end, and it would leave twenty places for a future tool to be born
    without it. The argument is swallowed HERE too, so no tool body can
    persist it: "the flag is never written" is structural rather than a promise
    repeated at every registration site. :func:`_consume_override` is that one
    consumption point, and the one place a caller's non-flag residue can still
    be reported now that pydantic no longer bounces it.

    SUBSTRATE, MEASURED IN THIS WORKTREE (fastmcp 3.2.2, CPython 3.13.9) — the
    PRD's section 6 table has been wrong about this library before, so this is
    recorded the way :mod:`shared.mcp_tool_schema` records its own substrate
    facts:

    1. FastMCP builds ``inputSchema`` from ``__signature__``, so appending a
       parameter there is enough — the function's own parameter list is never
       re-read. A ``tools/list`` round-trip through an in-process ``Client``
       advertises ``"metadata": {"anyOf": [{"additionalProperties": true,
       "type": "object"}, {"type": "null"}], "default": null}``, with the
       tool's own ``required`` list and its ``additionalProperties: false``
       both preserved. An ``Annotated[..., Field(description=...)]`` annotation
       survives the same route and lands as that property's ``description``,
       which is what carries :data:`_OVERRIDE_DESCRIPTION` to the caller.
    2. Calls WITH and WITHOUT the argument both succeed, and the wrapped body
       never sees it.
    3. An ``async def`` tool is supported: FastMCP awaits the coroutine the
       async wrapper returns, exactly as it awaits the undecorated tool. The
       branch matters — a sync wrapper around a coroutine function would hand
       FastMCP an un-awaited coroutine as the tool's result.
    """
    wrapper = (
        _async_swallowing_wrapper(fn)
        if inspect.iscoroutinefunction(fn)
        else _swallowing_wrapper(fn)
    )
    signature = inspect.signature(fn)
    # A NEW annotations dict, never a mutation: ``functools.wraps`` copies the
    # wrapped function's own ``__annotations__`` BY REFERENCE, so updating it
    # in place would edit the undecorated function too.
    wrapper.__annotations__ = {
        **getattr(fn, '__annotations__', {}),
        'metadata': _OVERRIDE_PARAMETER.annotation,
    }
    wrapper.__signature__ = signature.replace(  # type: ignore[attr-defined]
        parameters=[*signature.parameters.values(), _OVERRIDE_PARAMETER],
    )
    return wrapper
