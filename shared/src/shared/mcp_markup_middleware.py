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
   in-process idioms. Only a ``Client``-driven call exercises this module, so a
   test written either of the established ways would pass while running none of
   the code below.
4. ``on_call_tool`` runs STRICTLY BEFORE the tool's pydantic argument
   validation, so a repair can recover an absorbed parameter the tool declares
   as REQUIRED — this guard's whole benefit for the loudest leak shape, which
   the write-time tripwire cannot see at all. Chain: ``server.py:1112-1132``
   runs ``_run_middleware`` FIRST and its ``call_next`` re-reads the arguments
   back off the context at ``server.py:1127`` (which is why ``_forward``'s
   in-place mutation lands at all); execution follows at ``server.py:1162`` ->
   ``tools/base.py:373``; validation is ``type_adapter.validate_python(
   arguments)`` at ``tools/function_tool.py:249/253/273/276``, which is where
   ``Missing required argument`` comes from. Pinned by PRD boundary row B14 in
   ``shared/tests/test_mcp_markup_middleware.py``.
5. That ordering holds ONLY while ``strict_input_validation`` is off. With it
   on, ``mixins/mcp_operations.py:77`` registers the SDK handler with
   ``validate_input=True`` and ``mcp/server/lowlevel/server.py:528-532``
   jsonschema-validates against ``tool.inputSchema`` and returns an error
   result BEFORE FastMCP's handler runs — the middleware chain is never
   entered, no ``markup_detected`` fact is emitted, no storm is counted, and
   every required-parameter leak becomes silently unrepairable. It is off by
   default (``fastmcp/settings.py:297-311``) and no site in this repo sets it,
   but it is settable by constructor kwarg, by the global setting, and by
   ``FASTMCP_STRICT_INPUT_VALIDATION`` — so the registration sites must not
   turn it on. The negative half of B14 pins it.

Both test modules therefore drive a Client over a toy server:
``test_mcp_markup_middleware.py`` pins the hand-authored boundary rows B1-B10
(where a human decided the right answer), and
``test_mcp_markup_middleware_corpus.py`` replays the committed corpus of real
leaked calls through this same boundary (where the expectations are
machine-generated, so it pins regression-stability and the non-circular
guarantee that nothing still tripping :func:`detect` ever escapes).

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
import inspect
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
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
    MARKUP_OVERRIDE_KEY,
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

#: The fact names itself, so a multiplexed sink does not have to infer a
#: record's type from which callable it arrived on.
FACT_MARKUP_DETECTED = 'markup_detected'

#: The escape hatch, spelled ONCE and INTERPOLATED from the constant — exactly
#: as ``markup_tripwire._MARKUP_HINT`` builds its own. Both guards can bounce a
#: caller for the same reason, so a rename of the key must not leave either one
#: instructing callers to set a flag that no longer exists: that is the same
#: lockstep duplication (INV-5) promoting the key into ``shared.toolcall_markup``
#: was meant to end.
#:
#: The claim it makes is true on EVERY tool, including the many that declare no
#: ``metadata`` parameter at all — see :meth:`MarkupGuardMiddleware._apply_override`
#: for why the key is forwarded to a tool that takes ``metadata`` and dropped
#: from one that does not.
_OVERRIDE_SENTENCE = (
    "override with metadata={'" + MARKUP_OVERRIDE_KEY + "': True}. On a tool "
    'that takes no metadata the flag is dropped before dispatch rather than '
    'forwarded as an unexpected argument, so it works there too.'
)

#: Surfaced at the point of rejection, mirroring markup_tripwire's
#: ``_MARKUP_HINT``: the remediation must be discoverable where the caller is
#: bounced, not only in documentation it may never have read.
_REJECT_HINT = (
    'Resubmit repaired_call verbatim — it is the COMPLETE argument map with the '
    'leaked envelope fragment removed and the parameters it swallowed restored '
    '(see recovered_params). Do NOT reword the payload around the fragment: the '
    'leak is a serialization defect in your own tool-call envelope, not a '
    'content problem. If the markup is quoted DELIBERATELY (e.g. documenting '
    'the leak itself), ' + _OVERRIDE_SENTENCE
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
    + _OVERRIDE_SENTENCE
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
    module docstring. Either may be a plain function OR an ``async def`` — the
    machinery a registration site will wire them to is largely async in this
    repo — and both are invoked DEFENSIVELY: the call's outcome is already
    decided by the time either runs, so a sink that raises is logged and never
    changes what the caller sees. See :meth:`_call_sink`.

    The storm counter is held PER INSTANCE (like ``MarkupStormCounter``), so no
    burst state bleeds between servers, or between tests in one process.
    """

    def __init__(
        self,
        policy: RepairPolicy,
        exempt_tools: frozenset[str] | set[str] = frozenset(),
        *,
        # ``Awaitable`` spelled out in the annotation rather than left to
        # ``Any``: a sync-looking type is exactly what invited the sink to be
        # called without awaiting it. See :meth:`_call_sink`.
        escalation_sink: Callable[[dict[str, Any]], Any | Awaitable[Any]] | None = None,
        fact_sink: Callable[[dict[str, Any]], Any | Awaitable[Any]] | None = None,
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
        self._storm_time_provider = time_provider
        # ONE COUNTER PER KEY, not one counter with a composed label.
        # MEASURED: StormCounter holds a single deque and a single
        # _last_fire_ts, so its count spans EVERY event in the window
        # regardless of label — a label buys per-key ATTRIBUTION, never a
        # per-key THRESHOLD. Fed ['p|repaired', 'p|repaired', 'p|rejected',
        # 'p|rejected'] at threshold 3, one instance fires on the third event.
        # That is exactly the pooling boundary row B10 forbids, so the keying
        # has to live out here. This is the established shape for it, not an
        # invention: MemoryService keys per agent_id the same way, and
        # StormCounter.prune() exists as that consumer's sweep hook. The class
        # itself is untouched and its body still has no fourth copy (INV-5).
        self._storms: dict[str, StormCounter] = {}

    # -- the hook ---------------------------------------------------------

    async def on_call_tool(self, context, call_next):
        """Guard one tool call.

        Ordered so the overwhelmingly common path is cheapest. The corruption
        rate measured in PRD section 2.3 is 0.26%, and this sits on EVERY tool
        call on the server, so a clean call costs ONE scan per string argument
        and nothing else: :func:`detect` searches the whole literal set in a
        single compiled pass (not one pass per literal), and the awaited
        ``get_tool`` round-trip is deferred behind that gate.
        """
        name = context.message.name
        arguments = context.message.arguments or {}

        # B7 — a declared exemption. No scan, no fact: an exemption is a
        # declaration that this is not a leak, so recording it would pollute
        # the fact stream with a non-event.
        if name in self.exempt_tools:
            return await call_next(context)

        # B6 — the author is quoting the markup deliberately. Forward, with no
        # fact: a declared quote is not a detection. Whether the flag itself
        # travels with the call is decided by the invoked tool's own schema —
        # see :meth:`_apply_override`.
        if markup_override_requested(arguments.get('metadata')):
            await self._apply_override(context, name, arguments)
            return await call_next(context)

        hit = self._first_markup_argument(arguments)
        if hit is None:
            return await call_next(context)

        param, value, pattern = hit
        return await self._handle_markup(
            context, call_next, name, arguments, param, value, pattern
        )

    # -- the deliberate-quoting override (B6) ------------------------------

    @classmethod
    async def _apply_override(cls, context, name: str, arguments: dict[str, Any]) -> None:
        """Decide, FROM THE TOOL'S OWN SCHEMA, whether the flag travels onward.

        ONE override, ONE lifecycle. There are two guards on this hatch, not
        one: this boundary middleware, and fused-memory's write-time tripwire
        (``tools.py``'s ``_markup_gate``, which returns ``None`` for a caller
        that set the flag and then strips it at 2703/2933/6565/7118). They key
        off the SAME ``metadata`` map. So whether the flag reaches the tool body
        is not a detail — it decides whether the second guard can see the
        declaration the first one already honoured.

        * The tool DECLARES ``metadata`` — forward it UNCHANGED, flag included.
          The tool body owns the rest of the lifecycle: it is the guard that
          still has to be told, and it is already the party that strips the key
          before persistence. Consuming the flag here would leave a caller doing
          exactly what :data:`_OVERRIDE_SENTENCE` tells it to — quoting envelope
          literals in ``add_memory.content`` or ``submit_task.description`` with
          the flag set — past this guard and REJECTED by the next one, with a
          hint telling it to set a flag it had already set.
        * The tool declares NO ``metadata`` — strip the flag (it is
          write-time-only control, never payload) and drop the key when that
          empties it. MEASURED: most tools this
          guard sits in front of take none — ``escalate_info``,
          ``escalate_blocker``, ``add_reuse_item`` and most of the
          escalation/orchestrator surface. Forwarding ``metadata`` to one of
          those turns the documented remediation into ``ToolError: Unexpected
          keyword argument``: an agent quoting envelope markup in an escalation
          summary — a very likely topic, given DF 3083 — would land in a second,
          more confusing error with no working way out. That failure, and only
          that failure, is what the drop exists to fix, so anything the caller
          sent BESIDES the flag is left in place, in its original shape: on a
          tool with no ``metadata`` parameter that residue is the caller's own
          bug, and an unknown-parameter error on it is visible where silently
          discarding it would not be.

        The awaited ``get_tool`` round-trip is affordable here for the same
        reason it is on the repair path: an explicit override is rarer even than
        the 0.26% leak rate, and the clean fast path never reaches this method.

        Fail-safe: an unresolvable schema yields ``()`` (see
        :meth:`_schema_params`), which takes the DROP branch — the branch that
        cannot raise ``Unexpected keyword argument``. A tool that does take
        metadata then loses the flag and its own gate bounces the write with an
        actionable hint, which is strictly better than a call the tool cannot
        accept at all.
        """
        if 'metadata' in await cls._schema_params(context, name):
            return
        stripped = strip_markup_override(arguments['metadata'])
        # Both shapes of "nothing left": ``{}`` from a dict, and the ``'{}'``
        # ``json.dumps`` emits for the JSON-string form the real
        # submit_task/update_task also accept.
        if stripped == {} or stripped == '{}':
            del arguments['metadata']
        else:
            arguments['metadata'] = stripped

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

        # All three emissions sit SIDE BY SIDE, and every one of them runs
        # BEFORE its outcome is delivered. Two reasons. The detection is a fact
        # whether or not the tool then succeeds, so emitting after ``call_next``
        # would lose it to any tool that raises. And keeping the three calls in
        # one function is what makes "the eight keys cannot drift between
        # paths" visible rather than merely intended.
        if fix is None:
            # Identity from the RAW arguments: no recovery validated, so
            # nothing from the tail may be believed.
            identity = self._identity(arguments)
            await self._emit_fact(
                name, identity, param, pattern,
                outcome=_OUTCOME_UNREPAIRABLE, misclose=None, recovered=(),
            )
            storm = await self._record_storm(_OUTCOME_UNREPAIRABLE, identity[1])
            await self._refuse_unrepairable(name, identity, param, value, pattern, storm)

        # Identity from the REPAIRED view. If the leak ate the caller's own
        # ``agent_id`` — PRD section 2.1's first specimen does exactly that —
        # then reporting the pre-repair ``None`` would lose the one thing the
        # attribution exists for. Reading the recovery cannot overwrite a
        # supplied value: repair() only validates when the recovered names are
        # DISJOINT from the supplied ones (boundary row B9).
        identity = self._identity({**arguments, **fix.recovered})

        if self.policy is RepairPolicy.REJECT_WITH_REPAIR:
            await self._emit_fact(
                name, identity, param, pattern,
                outcome=_OUTCOME_REJECTED, misclose=fix.misclose, recovered=fix.recovered,
            )
            storm = await self._record_storm(_OUTCOME_REJECTED, identity[1])
            return self._reject(name, arguments, param, fix, storm)

        await self._emit_fact(
            name, identity, param, pattern,
            outcome=_OUTCOME_REPAIRED, misclose=fix.misclose, recovered=fix.recovered,
        )
        storm = await self._record_storm(_OUTCOME_REPAIRED, identity[1])
        return await self._forward(context, call_next, arguments, param, fix, storm)

    # -- the injected channels --------------------------------------------

    async def _call_sink(
        self,
        sink: Callable[[dict[str, Any]], Any | Awaitable[Any]],
        record: dict[str, Any],
        channel: str,
    ) -> Any:
        """Invoke one injected sink, AWAITING an async emitter, and never raise.

        The whole point of injecting these is that a registration site (task
        3690) wires the concrete emitters, and the queue/escalation machinery
        they will wire to is largely async in this repo — so an ``async def``
        emitter is a legitimate thing to be handed. Calling one without awaiting
        it queues NOTHING while handing back a coroutine that looks like a
        result: the residue escalation would report no id, the refusal payload
        would name none, the caller's only surviving copy of an unrepairable
        payload would be destroyed, and the sole trace would be a bare
        ``coroutine was never awaited`` RuntimeWarning. That is precisely the
        silent fail-soft this module exists to end, committed by the module
        itself, so an awaitable is awaited rather than trusted to be a value.

        Never raises. The call's outcome is already decided by the time either
        sink runs, so both channels are purely ADDITIVE: a sink outage costs an
        operator visibility rather than turning a working guard into an outage of
        its own. Same never-raises contract
        ``markup_tripwire.emit_markup_storm_escalation`` keeps, and for the same
        reason. It is logged via ``logger.exception``, never swallowed.
        """
        try:
            result = sink(record)
            if inspect.isawaitable(result):
                result = await result
        except Exception:
            logger.exception(
                'markup guard: the %s sink failed for %r; the outcome stands',
                channel, record.get('error_type') or record.get('fact'),
            )
            return None
        return result

    # -- facts (INV-2) ----------------------------------------------------

    async def _emit_fact(
        self,
        tool: str,
        identity: tuple[str | None, str | None],
        param: str,
        pattern: str,
        *,
        outcome: str,
        misclose: str | None,
        recovered: Any,
    ) -> None:
        """Emit one ``markup_detected`` (INV-2), and never change an outcome.

        ONE builder for all three paths, so the contracted key set cannot drift
        between them. The record is COMPLETE even where the answer is ``None``:
        the unrepairable path has no accepted mis-close — that is precisely why
        it is unrepairable — so ``misclose`` is present and null rather than
        absent, and a consumer never has to tell "no mis-close" apart from "that
        emitter forgot the key".

        ``recovered_params`` carries NAMES ONLY. Names are what a consumer needs
        to see which parameters a leak is eating; the values are the caller's
        data, and this stream is not where it survives. That is the residue
        escalation's job, on the one path where it would otherwise be lost.

        ``pattern`` and ``misclose`` are kept APART on purpose. ``pattern`` is
        the envelope literal :func:`detect` matched; ``misclose`` is the tag
        that actually drifted, verbatim, including the blended dialect's stray
        quote — which is not a literal at all. PRD section 2.2: collapsing them
        is what left the old guard blaming whatever happened to follow a
        mis-closed ``description``.

        The log line carries the same fields, mirroring ``markup_tripwire``'s
        split: the structured record goes to the operator-facing channel while
        the caller-facing block reaches only the leaking caller.
        """
        agent_id, project = identity
        fact = {
            'fact': FACT_MARKUP_DETECTED,
            'tool': tool,
            'param': param,
            'pattern': pattern,
            'misclose': misclose,
            'outcome': outcome,
            'recovered_params': sorted(recovered),
            'agent_id': agent_id,
            'project': project,
        }

        logger.warning(
            'markup guard: %s tool=%s param=%s pattern=%r misclose=%r '
            'recovered_params=%r agent_id=%r project=%r',
            outcome, tool, param, pattern, misclose,
            fact['recovered_params'], agent_id, project,
        )

        if self._fact_sink is None:
            return
        await self._call_sink(self._fact_sink, fact, 'fact')

    # -- the storm escape (INV-4) -----------------------------------------

    async def _record_storm(self, outcome: str, project: str | None) -> dict[str, Any] | None:
        """Count this outcome; escalate and return a summary iff a burst fired.

        Repair is a FAIL-SOFT path, so it owes the rate/streak escalation. One
        corrupted call is routine — the measured rate is 0.26% — but a BURST
        means the upstream serialization leak is running right now.

        Keyed by ``(project, outcome)``, which is the generalisation this task
        owns. ``MarkupStormCounter`` keys by project alone, which sufficed while
        "rejected" was the only outcome. With two declared tiers there are
        three, and a burst of REPAIRS is exactly as urgent as a burst of
        rejections while being invisible to every caller — those calls all
        succeeded. Pooling the outcomes would fire an alarm naming one that
        never burst, and an operator sent chasing a burst that did not happen
        learns to ignore the alarm.

        Threshold and window are passed PER CALL, honouring ``StormCounter``'s
        reload-safety contract, so a registration site backed by a green-tier
        config leaf can read them live rather than capturing them here.
        """
        key = f'{project}\x1f{outcome}'
        counter = self._storms.get(key)
        if counter is None:
            counter = StormCounter(time_provider=self._storm_time_provider)
            self._storms[key] = counter

        summary = counter.record(
            threshold=self._storm_threshold,
            window_seconds=self._storm_window_seconds,
            label=key,
        )

        # One counter per key means one object per key ever seen, and `project`
        # is caller-supplied — so sweep the dormant ones, exactly as the
        # MemoryService consumer StormCounter.prune() was written for.
        for other, dormant in list(self._storms.items()):
            if other != key and dormant.prune(self._storm_window_seconds) == 0:
                del self._storms[other]

        if summary is None:
            return None

        storm = {
            'count': summary['count'],
            'threshold': summary['threshold'],
            'window_seconds': summary['window_seconds'],
            'outcome': outcome,
            'project': project,
        }
        # ERROR, and greppable: markup_tripwire's split again — the summary
        # folded into the response reaches ONLY the leaking caller, which is
        # the one party that already knows, so the operator-facing half cannot
        # ride on it.
        logger.error(
            'markup_guard_storm: %d %s outcome(s) in %ss for project=%r — the '
            'serialization leak is ACTIVE (see DF 3083)',
            storm['count'], outcome, storm['window_seconds'], project,
        )
        await self._file_storm_escalation(storm)
        return storm

    async def _file_storm_escalation(self, storm: dict[str, Any]) -> None:
        """Hand the burst to the injected sink; never change an outcome.

        No dedup here. ``emit_markup_storm_escalation`` dedups against an OPEN
        escalation in the target queue, which is knowledge this layer does not
        have and must not guess at — the sink owns it, exactly as it does
        today.
        """
        if self._escalation_sink is None:
            return
        await self._call_sink(
            self._escalation_sink,
            {'error_type': 'mcp_markup_storm', **storm},
            'escalation',
        )

    @staticmethod
    def _with_storm(payload: dict[str, Any], storm: dict[str, Any] | None) -> dict[str, Any]:
        """Fold a FIRED burst into a caller-facing payload, and only then.

        ``build_markup_block``'s convention: the key is OMITTED entirely when
        no burst fired, rather than set to ``None``, so its presence is an
        unambiguous signal instead of a value the caller has to inspect.
        """
        if storm is not None:
            payload['storm'] = storm
        return payload

    async def _refuse_unrepairable(
        self, name, identity, param, value, pattern, storm
    ) -> NoReturn:
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
        # The SAME identity the fact carries, threaded in rather than
        # re-resolved, so the two records cannot disagree about which agent is
        # leaking — an operator reading one and acting on the other would
        # otherwise be chasing two different callers.
        agent_id, project = identity
        escalation_id = await self._file_residue_escalation({
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

        raise ToolError(json.dumps(self._with_storm({
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
        }, storm)))

    async def _file_residue_escalation(self, record: dict[str, Any]) -> str | None:
        """Hand the residue to the injected sink; never change the outcome.

        The refusal is already decided by the time this runs, so escalation is
        purely ADDITIVE — see :meth:`_call_sink` for the never-raises contract
        and for why an ``async def`` emitter is awaited rather than left as an
        unawaited coroutine, which on THIS path would destroy the only surviving
        copy of the caller's payload.

        Returns the id the sink reports, so the refusal payload can name the
        record holding the caller's data. A sink that reports something other
        than a string reports no id — the caller is better told nothing than
        pointed at a value it cannot look up. A sink that FAILED reports ``None``
        for the same reason.
        """
        if self._escalation_sink is None:
            logger.warning(
                'markup guard: no escalation_sink is wired, so the residue of '
                '%r will not be preserved anywhere', record.get('tool'),
            )
            return None
        escalation_id = await self._call_sink(self._escalation_sink, record, 'escalation')
        if not isinstance(escalation_id, str):
            return None
        # Naming the queued record, exactly as emit_markup_storm_escalation
        # signs off — the operator's entry point to the preserved payload.
        logger.warning(
            'markup guard: queued %s holding the unrepairable payload of '
            '%s.%s (%d chars)',
            escalation_id, record.get('tool'), record.get('field'),
            len(record.get('raw_value') or ''),
        )
        return escalation_id

    def _reject(self, name, arguments, param, fix, storm) -> NoReturn:
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
        raise ToolError(json.dumps(self._with_storm({
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
        }, storm)))

    async def _forward(self, context, call_next, arguments, param, fix, storm):
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

        A FIRED burst rides along under the same ``storm`` key and the same
        omit-when-absent convention both refusal payloads use. This tier is the
        one whose callers cannot learn of the burst any other way — their calls
        all SUCCEEDED — and :meth:`_record_storm` argues a burst of repairs is
        exactly as urgent as a burst of rejections, so the one tier that is
        invisible must not also be the one tier that is never told.
        """
        arguments[param] = fix.clean_value
        arguments.update(fix.recovered)

        result = await call_next(context)

        meta = dict(result.meta or {})
        meta['markup_repair'] = self._with_storm({
            'outcome': _OUTCOME_REPAIRED,
            'field': param,
            'matched_pattern': fix.pattern,
            'misclose': fix.misclose,
            # NAMES only, never values: a forwarding caller needs to know WHICH
            # of its arguments the guard altered, and the warning must not
            # become a second copy of its payload.
            'recovered_params': sorted(fix.recovered),
            'hint': _FORWARD_HINT,
        }, storm)
        return ToolResult(
            content=result.content,
            structured_content=result.structured_content,
            meta=meta,
        )
