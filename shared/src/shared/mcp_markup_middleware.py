"""FastMCP boundary guard for leaked tool-call envelope markup (task 3689).

PRD ``plans/toolcall-markup-containment-prd.md`` task beta, contract C2. This
is the filed consumer of task 3688's staged :func:`shared.toolcall_markup.detect`
/ :func:`~shared.toolcall_markup.repair` API: a
:class:`fastmcp.server.middleware.Middleware` that inspects every string
argument of every tool call, repairs a leaked envelope against the invoked
tool's LIVE parameter schema, and applies a registration-declared policy.

It is a pure BOUNDARY layer. It contributes policy, structured facts and the
storm escape; it contributes NO parsing of its own. Every question about what
counts as envelope markup, what a repair may recover, and when a repair must be
refused is answered by ``shared.toolcall_markup``, which owns the single
enumeration (INV-5).

## The two tiers (INV-1: DECLARED, never inferred)

:class:`RepairPolicy` is supplied at REGISTRATION time, per server, by the site
that knows what that server's tools do — not guessed per call from the shape of
the damage. The set of declarable tiers is therefore closed and machine
checkable; a tier nobody declared cannot exist.

* :attr:`RepairPolicy.REJECT_WITH_REPAIR` — write nothing, bounce the caller
  with the repaired argument map so the retry is mechanical.
* :attr:`RepairPolicy.FORWARD_REPAIR` — repair in place and let the call
  through, warning the caller that its arguments were altered. For tools where
  bouncing costs more than proceeding (a strand-risk escalation that would
  otherwise be lost).

Both tiers share ONE refusal: when :func:`~shared.toolcall_markup.repair`
returns ``None`` the input is UNREPAIRABLE, and nothing is guessed, forwarded
or partially written under either tier.

## Why the escalation and fact channels are injected callables

``shared`` is the base layer every other package imports. ``escalation``
already depends on ``dark-factory-shared``, so importing it here would be a
package cycle; ``fused_memory`` is likewise off-limits. fused-memory's
``markup_tripwire`` dodges this with a defensive ``try/except ImportError``,
but it can afford to because nothing depends on IT. So both channels arrive as
constructor-injected callables and the registration site (task 3690) wires the
concrete emitters, each server against its own queue and project attribution.
A side benefit: INV-2 and INV-7 become directly assertable against a fake sink
instead of by filesystem archaeology.

## Substrate facts, measured against fastmcp 3.2.2 — do not re-derive these

1. ``FastMCP.get_tool`` is a COROUTINE and MUST be awaited. (PRD section 6
   writes it as a sync call; that is wrong for this version.)
2. ``tool.parameters`` is a full JSON Schema dict, so the parameter NAMES are
   ``tool.parameters['properties'].keys()`` — not the object itself.
3. Middleware is BYPASSED by ``tool.fn(...)``, ``await tool.run({...})`` and
   ``server._tool_manager.call_tool(...)`` — the repo's two established
   in-process idioms. Only a ``Client``-driven call exercises this module. See
   ``shared/tests/test_mcp_markup_middleware.py``.

## Sentinel-literal hazard — DO NOT "helpfully" un-escape these

Any envelope literal in this file is spelled with the ``\\x3c`` escape for
``<``, for the reason ``shared/src/shared/toolcall_markup.py`` records: writing
``<`` verbatim would force an agent editing this file to emit that literal
inside its own tool-call envelope, reproducing the exact defect this module
contains.

This module is deliberately NOT re-exported from ``shared/__init__``, following
the ``mcp_envelope`` / ``proc_group`` / ``config_dir`` convention — import it
fully qualified. That also keeps ``import shared`` from pulling in fastmcp for
the consumers that never touch the middleware.
"""
from __future__ import annotations

import enum
import json
import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware

from shared.storm_counter import StormCounter
from shared.toolcall_markup import (
    detect,
    markup_override_requested,
    repair,
    strip_markup_override,
)

logger = logging.getLogger(__name__)

#: The three legal values of a fact's ``outcome``. ONE source for both INV-2's
#: fact vocabulary and INV-4's storm key, so the two cannot drift apart.
_OUTCOME_REPAIRED = 'repaired'
_OUTCOME_REJECTED = 'rejected'
_OUTCOME_UNREPAIRABLE = 'unrepairable'
OUTCOMES: tuple[str, ...] = (_OUTCOME_REPAIRED, _OUTCOME_REJECTED, _OUTCOME_UNREPAIRABLE)

#: Surfaced at the point of rejection, mirroring markup_tripwire's
#: ``_MARKUP_HINT``: the remediation must be discoverable where the caller is
#: bounced, not only in documentation it may never have read.
_REJECT_HINT = (
    'Resubmit repaired_call verbatim — it is the COMPLETE argument map with the '
    'leaked envelope fragment removed and the parameters it swallowed restored '
    '(see recovered_params). Do NOT reword the payload around the fragment: the '
    'leak is a serialization defect in your own tool-call envelope, not a '
    'content problem. If the markup is quoted DELIBERATELY (e.g. documenting '
    "the leak itself), override with metadata={'allow_mcp_markup': True}."
)


class RepairPolicy(enum.StrEnum):
    """What a server does with a tool call carrying leaked envelope markup.

    A ``StrEnum`` so a declaration round-trips through config and serializes
    legibly into a JSON fact record without a conversion step.
    """

    #: Write nothing; bounce the caller with the repaired argument map.
    REJECT_WITH_REPAIR = 'reject_with_repair'

    #: Repair in place, forward, and warn that the arguments were altered.
    FORWARD_REPAIR = 'forward_repair'


class MarkupGuardMiddleware(Middleware):
    """Detect, repair and police leaked envelope markup at the FastMCP boundary.

    *policy* is the declared tier for THIS server (INV-1). *exempt_tools* names
    the tools that are allowed to carry envelope literals as ordinary data —
    ``scan_memory_content``'s whole job is to search the corpus for exactly
    these substrings, and guarding it would make the retroactive-sweep tool
    unable to look for the thing it was built to find. Nothing is exempt by
    default: a registration site opts a tool OUT explicitly.

    Names are matched BARE. ``context.message.name`` is the in-server name
    (``submit_task``), not the agent-facing ``mcp__fused-memory__submit_task``
    that the specimen corpus records, so an exemption declared with the prefixed
    spelling would silently never match — and a machine-checked declaration that
    fails open is worse than none (INV-1).

    *escalation_sink* and *fact_sink* are the injected channels described in the
    module docstring. Both are invoked DEFENSIVELY: the call's outcome is
    already decided by the time either runs, so a sink that raises is logged and
    never changes what the caller sees.

    The storm counter is held PER INSTANCE (like ``MarkupStormCounter``), so no
    burst state bleeds between servers, or between tests in one process.
    """

    def __init__(
        self,
        policy: RepairPolicy,
        exempt_tools: frozenset[str] | set[str] = frozenset(),
        *,
        escalation_sink: Callable[[dict[str, Any]], Any] | None = None,
        fact_sink: Callable[[dict[str, Any]], Any] | None = None,
        storm_threshold: int = 3,
        storm_window_seconds: float = 3600.0,
        time_provider: Callable[[], float] = time.time,
    ) -> None:
        self.policy = policy
        self.exempt_tools = frozenset(exempt_tools)
        self._escalation_sink = escalation_sink
        self._fact_sink = fact_sink
        # Stored, then passed PER record() call — StormCounter's reload-safety
        # contract, so a consumer whose threshold comes from a green-tier config
        # leaf can read it live rather than capturing it at construction.
        self._storm_threshold = storm_threshold
        self._storm_window_seconds = storm_window_seconds
        self._storm = StormCounter(time_provider=time_provider)

    # -- the hook ---------------------------------------------------------

    async def on_call_tool(self, context, call_next):
        """Guard one tool call.

        Ordered so the overwhelmingly common path is cheapest. The corruption
        rate measured in PRD section 2.3 is 0.26%, and this sits on EVERY tool
        call on the server, so a clean call costs one substring scan and
        nothing else — in particular it never pays the awaited ``get_tool``
        round-trip, which is deferred behind the :func:`detect` gate.
        """
        name = context.message.name
        arguments = context.message.arguments or {}

        # B7 — a declared exemption. No scan, no fact: an exemption is a
        # declaration that this is not a leak, so recording it would pollute
        # the fact stream with a non-event.
        if name in self.exempt_tools:
            return await call_next(context)

        # B6 — the author is quoting the markup deliberately. Strip the
        # write-time-only control flag before dispatch (it must never reach the
        # tool or be persisted) and forward, again with no fact: a declared
        # quote is not a detection.
        if markup_override_requested(arguments.get('metadata')):
            arguments['metadata'] = strip_markup_override(arguments['metadata'])
            return await call_next(context)

        hit = self._first_markup_argument(arguments)
        if hit is None:
            return await call_next(context)

        param, value = hit
        return await self._handle_markup(context, call_next, name, arguments, param, value)

    # -- detection --------------------------------------------------------

    @staticmethod
    def _first_markup_argument(arguments: Mapping[str, Any]) -> tuple[str, str] | None:
        """Return the first ``(param, value)`` whose value carries envelope markup.

        Insertion order, which for a JSON-RPC argument map is the order the
        caller sent. Non-string values are skipped: the leak is a serialization
        defect in the caller's own text emission, so it can only ever land in a
        string.

        First hit wins rather than collecting all of them, because the repair
        is defined against ONE absorbing parameter — the argument whose value
        swallowed the rest of the envelope. A second hit downstream of it is
        either the same leak seen twice or a value that repair() will refuse
        anyway.
        """
        for param, value in arguments.items():
            if isinstance(value, str) and detect(value) is not None:
                return param, value
        return None

    # -- schema resolution ------------------------------------------------

    @staticmethod
    async def _schema_params(context, name: str) -> tuple[str, ...]:
        """The invoked tool's LIVE parameter names.

        Two measured substrate facts, both of which the PRD's section 6 table
        gets wrong for fastmcp 3.2.2: ``get_tool`` is a COROUTINE and must be
        awaited, and ``tool.parameters`` is a full JSON Schema dict — so the
        names are under ``'properties'``, not the object itself.

        LIVE rather than declared at registration: what a recovery is allowed
        to name has to be checked against the tool as it actually is, or the
        guard would validate against a schema that has since drifted.

        Any failure to resolve one yields an EMPTY set, which makes repair()
        refuse rather than recover against a phantom schema. That is the same
        fail-safe direction ``_as_name_set`` already takes: with no schema,
        every recovered name is out-of-schema, so nothing can be fabricated.
        """
        try:
            tool = await context.fastmcp_context.fastmcp.get_tool(name)
        except Exception:
            logger.exception('markup guard could not resolve the schema for %r', name)
            return ()
        parameters = getattr(tool, 'parameters', None) or {}
        properties = parameters.get('properties') if isinstance(parameters, dict) else None
        if not isinstance(properties, dict):
            return ()
        return tuple(properties)

    # -- policy -----------------------------------------------------------

    async def _handle_markup(self, context, call_next, name, arguments, param, value):
        """Apply the declared policy to a detected leak.

        The schema read happens HERE and not on the fast path, so the awaited
        ``get_tool`` round-trip is paid only by the 0.26% of calls that
        actually carry a leak.
        """
        fix = repair(value, param, await self._schema_params(context, name), tuple(arguments))

        if fix is None:
            # UNREPAIRABLE — one refusal shared by both tiers. Lands in a later
            # step of this task, together with the residue escalation that
            # preserves the caller's payload. Refusing is already the right
            # direction and must stay so.
            raise ToolError(
                'Tool call carries leaked MCP envelope markup in '
                f'{param!r} (matched {detect(value)!r}) that could not be repaired.'
            )

        if self.policy is RepairPolicy.REJECT_WITH_REPAIR:
            return self._reject(name, arguments, param, fix)

        raise ToolError(
            'Tool call carries leaked MCP envelope markup in '
            f'{param!r} (matched {detect(value)!r}).'
        )

    def _reject(self, name, arguments, param, fix) -> None:
        """Write nothing; bounce the caller with the repaired argument map.

        RAISES rather than short-circuiting with a middleware-authored
        ``ToolResult``. Measured: a tool with a return annotation carries an
        output schema, and a ToolResult that does not satisfy it fails with
        "Output validation error: 'result' is a required property". This
        middleware is generic over every tool on the server, so it cannot know
        how to satisfy an arbitrary output schema. Raising is verified to
        prevent the tool body from running at all — which is what makes
        "nothing written" TRUE rather than merely intended — and to round-trip
        a ``json.dumps`` payload to the caller byte-intact, so ``repaired_call``
        survives the boundary.

        (The repo's usual "return a dict, don't raise" convention comes from the
        ``@mcp_tool_errors`` decorator INSIDE tool bodies. Middleware sits above
        that layer and has no such wrapper.)

        ``repaired_call`` is the COMPLETE argument map, not just the repaired
        field, so the retry is mechanical: resubmit it verbatim. A caller
        reassembling the rest by hand is how the swallowed arguments get lost
        for good.

        The flat key vocabulary — error / error_type / field / matched_pattern
        / hint — is markup_tripwire's ``build_markup_block``, so an agent
        bounced by this guard sees the same diagnostic shape as one bounced by
        the write-time tripwire: one mechanism, one error contract. It is
        EXTENDED with misclose, recovered_params and repaired_call, which the
        old single-literal guard could not produce.
        """
        repaired_call = {**arguments, param: fix.clean_value, **fix.recovered}
        raise ToolError(json.dumps({
            'error': (
                'Tool call rejected: it carries raw MCP envelope markup in '
                f'{param!r}, which means the caller serialized part of its own '
                'tool-call envelope into the payload.'
            ),
            'error_type': 'mcp_markup_detected',
            'outcome': _OUTCOME_REJECTED,
            'tool': name,
            'field': param,
            'matched_pattern': fix.pattern,
            'misclose': fix.misclose,
            'recovered_params': sorted(fix.recovered),
            'repaired_call': repaired_call,
            'hint': _REJECT_HINT,
        }))
