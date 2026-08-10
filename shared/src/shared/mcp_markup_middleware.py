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
from typing import Any, NoReturn

from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware

# From ``fastmcp.tools.base``, which is where ``Middleware.on_call_tool``'s own
# return annotation imports it from. NOT ``fastmcp.tools.tool``: that path is a
# runtime-only backward-compat shim (``tool.py`` was renamed to ``base.py``
# precisely to stop pyright resolving ``from fastmcp.tools import tool`` as the
# submodule instead of the decorator), so it imports fine but type-checks as
# reportMissingImports. Same class either way — measured identical.
from fastmcp.tools.base import ToolResult

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

#: Surfaced when the value's boundary is a GUESS. It deliberately does NOT
#: offer a mechanical retry — there is no repaired call to resend — so it points
#: at the two things the caller can actually do.
_UNREPAIRABLE_HINT = (
    'This value is UNREPAIRABLE: its own boundary cannot be determined, so no '
    'repair was attempted and nothing was written. Guessing would silently '
    'drop whatever arguments hide in the residue. Your full payload is '
    'preserved verbatim in the escalation named above — resend it from your '
    'own copy once your tool-call envelope stops leaking, rather than '
    'reconstructing it from this error. If the markup is quoted DELIBERATELY, '
    "override with metadata={'allow_mcp_markup': True}."
)

#: The residue escalation's category, and the machine-readable owner that will
#: exit the hold unprompted (INV-7). This is a queue-backed handoff, so the
#: bound is the L2 watcher's standing age surfacing rather than a deadline of
#: this record's own — which is what ``_ESCALATION_LEVEL`` buys.
_ESCALATION_CATEGORY = 'mcp_markup_residue'
_ESCALATION_OWNER = 'l2-escalation-watcher'
_ESCALATION_LEVEL = 2

#: Surfaced on the FORWARDED call, whose caller is NOT bounced and would
#: otherwise have no way to learn that its arguments were altered.
_FORWARD_HINT = (
    'This call carried raw MCP envelope markup and was REPAIRED in place '
    'before dispatch: the leaked fragment was stripped from `field` and the '
    'parameters it had swallowed (see recovered_params) were restored. The '
    'call went through, but your tool-call envelope is leaking into your own '
    'payloads — fix the emission rather than relying on this repair.'
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

        param, value, pattern = hit
        return await self._handle_markup(
            context, call_next, name, arguments, param, value, pattern
        )

    # -- detection --------------------------------------------------------

    @staticmethod
    def _first_markup_argument(arguments: Mapping[str, Any]) -> tuple[str, str, str] | None:
        """Return the first ``(param, value, pattern)`` carrying envelope markup.

        The matched *pattern* is carried out of the scan rather than re-derived
        later: it is needed on all three outcome paths, and only here is it
        known to be non-``None``.

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
            if isinstance(value, str):
                pattern = detect(value)
                if pattern is not None:
                    return param, value, pattern
        return None

    # -- identity ---------------------------------------------------------

    @staticmethod
    def _identity(arguments: Mapping[str, Any]) -> tuple[str | None, str | None]:
        """The ``(agent_id, project)`` this call attributes itself to.

        A DECLARED, deliberately narrow precedence, not an oversight: middleware
        sits above the tool bodies and has no equivalent of
        ``fused_memory.server.tools._resolve_identity(ctx)``, which can consult
        the request context and the server's own config. All this layer has is
        the call's own arguments.

        * ``agent_id`` — from the ``agent_id`` argument, or ``None``.
        * ``project`` — from ``project_root`` FIRST, then ``project_id``.

        ``project_root`` wins because it is the label
        ``MarkupStormCounter`` already keys on and the one an escalation queue
        is addressed by, so a burst attributed here lands in the same queue
        attribution the write-time tripwire would have chosen. A record whose
        project cannot be resolved carries ``None`` rather than a guessed
        default: an escalation filed against the wrong project is worse than
        one an operator has to place by hand.

        Non-string values yield ``None``. These fields are identity, and a
        caller that sent a dict where a name belongs has not identified itself.
        """
        def _text(key: str) -> str | None:
            value = arguments.get(key)
            return value if isinstance(value, str) else None

        return _text('agent_id'), _text('project_root') or _text('project_id')

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

    async def _handle_markup(self, context, call_next, name, arguments, param, value, pattern):
        """Apply the declared policy to a detected leak.

        The schema read happens HERE and not on the fast path, so the awaited
        ``get_tool`` round-trip is paid only by the 0.26% of calls that
        actually carry a leak.
        """
        fix = repair(value, param, await self._schema_params(context, name), tuple(arguments))

        if fix is None:
            self._refuse_unrepairable(name, arguments, param, value, pattern)

        if self.policy is RepairPolicy.REJECT_WITH_REPAIR:
            return self._reject(name, arguments, param, fix)

        return await self._forward(context, call_next, arguments, param, fix)

    def _refuse_unrepairable(self, name, arguments, param, value, pattern) -> NoReturn:
        """ONE refusal, shared by both tiers, when the boundary is a guess.

        The tiers disagree about what to do with a REPAIR. They do not disagree
        about this: :attr:`RepairPolicy.FORWARD_REPAIR` forwards a repair, and a
        tier that forwarded unparsed residue would deliver a call the caller
        never made while permanently dropping whatever arguments hide inside it
        — the silent fail-soft this whole guard exists to end.

        No separate handling is needed for PRD boundary rows B8 (a recovered
        name outside the tool's schema) and B9 (a recovered name the caller
        already supplied). Task 3688's :func:`repair` already refuses both at
        its accept-time conditions, so they arrive here as an ordinary ``None``.
        That is the point of keeping every "may I?" question in one place: this
        layer never has to re-implement a rule it can only get subtly wrong.

        The residue escalation is what makes refusing non-destructive. Without
        it a refusal DESTROYS the caller's payload whenever the caller does not
        retry — which is the common case for a leak that is, by construction,
        an agent emitting text it cannot re-emit identically.
        """
        agent_id, project = self._identity(arguments)
        escalation_id = self._file_residue_escalation({
            'error_type': 'mcp_markup_unrepairable',
            'category': _ESCALATION_CATEGORY,
            # INV-7: a machine-readable owner plus a bound. This hold is a
            # queue-backed handoff to a supervised consumer, so the bound is
            # that consumer's standing age surfacing rather than a deadline of
            # this record's own.
            'owner': _ESCALATION_OWNER,
            'level': _ESCALATION_LEVEL,
            'tool': name,
            'field': param,
            'matched_pattern': pattern,
            'agent_id': agent_id,
            'project': project,
            # FULL and VERBATIM, deliberately unlike build_markup_block's
            # 200-char content_excerpt. That excerpt is a diagnostic sitting
            # beside a payload the caller still holds; this is the only
            # surviving copy of data that may never be resent.
            'raw_value': value,
            'summary': (
                f'Unrepairable MCP envelope markup in {name}.{param} — the '
                'call was refused and its payload preserved here'
            ),
            'suggested_action': (
                'Recover the raw_value for the caller if it is still needed, '
                'then chase the harness serialization leak that produced it.'
            ),
            # No 'detail' blob: raw_value plus the flat fields above ARE the
            # detail, and prose duplicating them would carry the caller's
            # payload twice in one record.
        })

        logger.warning(
            'markup guard: UNREPAIRABLE envelope markup in %s.%s (matched %r) '
            'from agent_id=%r project=%r; refused under policy=%s, residue '
            'escalation=%r',
            name, param, pattern, agent_id, project, self.policy.value, escalation_id,
        )
        raise ToolError(json.dumps({
            'error': (
                f'Tool call refused: {param!r} carries raw MCP envelope markup '
                'whose own boundary cannot be determined, so no repair was '
                'attempted.'
            ),
            'error_type': 'mcp_markup_unrepairable',
            'outcome': _OUTCOME_UNREPAIRABLE,
            'tool': name,
            'field': param,
            'matched_pattern': pattern,
            'escalation_id': escalation_id,
            # NO 'repaired_call'. There is no repair, and offering one would
            # invite a retry that re-sends a guess.
            'hint': _UNREPAIRABLE_HINT,
        }))

    def _file_residue_escalation(self, record: dict[str, Any]) -> str | None:
        """Hand the residue to the injected sink; never change the outcome.

        The refusal is already decided by the time this runs, so escalation is
        purely ADDITIVE — the same never-raises contract
        ``markup_tripwire.emit_markup_storm_escalation`` keeps, and for the same
        reason: a queue I/O failure should cost the operator a heads-up, not
        turn a clean refusal into an opaque middleware crash.

        Returns the id the sink reports, so the refusal payload can name the
        record holding the caller's data. A sink that reports something other
        than a string reports no id — the caller is better told nothing than
        pointed at a value it cannot look up.
        """
        if self._escalation_sink is None:
            logger.warning(
                'markup guard: no escalation_sink is wired, so the residue of '
                '%r will not be preserved anywhere', record.get('tool'),
            )
            return None
        try:
            escalation_id = self._escalation_sink(record)
        except Exception:
            logger.exception(
                'markup guard: the escalation sink failed for %r; the refusal '
                'stands but the residue was not preserved', record.get('tool'),
            )
            return None
        return escalation_id if isinstance(escalation_id, str) else None

    def _reject(self, name, arguments, param, fix) -> NoReturn:
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

    async def _forward(self, context, call_next, arguments, param, fix):
        """Repair in place, let the call through, and say so.

        The mutation is IN PLACE on ``context.message.arguments``, which is
        verified to reach the tool (probe: the tool body observed the mutated
        value). That is what makes the recovered parameter LAND rather than
        merely be reported — the point of this tier.

        The warning rides on ``meta`` and NEVER on ``structured_content`` or an
        extra content block. Both alternatives are ruled out by measurement,
        not preference: a middleware-authored ``structured_content`` fails the
        tool's own output schema, and an appended content block would corrupt
        any caller that indexes ``content[0]``. ``meta`` is verified to survive
        to the client untouched.

        ``call_next``'s result already carries ``{'fastmcp': {'wrap_result':
        True}}``, so the warning is FOLDED into whatever meta came back rather
        than replacing it — discarding FastMCP's own signalling to deliver our
        own would be a poor trade.
        """
        arguments[param] = fix.clean_value
        arguments.update(fix.recovered)

        result = await call_next(context)

        meta = dict(result.meta or {})
        meta['markup_repair'] = {
            'outcome': _OUTCOME_REPAIRED,
            'field': param,
            'matched_pattern': fix.pattern,
            'misclose': fix.misclose,
            # NAMES only, never values: a forwarding caller needs to know WHICH
            # of its arguments the guard altered, and the warning must not
            # become a second copy of its payload.
            'recovered_params': sorted(fix.recovered),
            'hint': _FORWARD_HINT,
        }
        return ToolResult(
            content=result.content,
            structured_content=result.structured_content,
            meta=meta,
        )
