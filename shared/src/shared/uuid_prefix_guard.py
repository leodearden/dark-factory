"""Boundary policy for truncated-uuid prefixes (PRD contract C3).

``plans/uuid-prefix-resolution-prd.md`` §4-C3's table is the SOURCE of the
policy in this module. :data:`POLICY_MATRIX` transcribes it, and the matrix IS
the policy: the middleware body contains no second if/else expressing the same
decisions, because two expressions of one policy drift and the declared one is
the checkable one (INV-1). Adding an outcome or a tool class without declaring
its cells fails a test, not a production call.

THREE ORTHOGONAL AXES, kept apart on purpose (heuristic 3). What to detect is
``shared.uuid_prefix``'s job. What to do about it is this matrix. Which tools
are in which class is a per-registration DECLARATION handed to the constructor
— never inferred from a tool's name, which is the failure mode where a rename
silently changes a security posture.

WHY THE RESOLVER IS INJECTED. ``shared`` is the base layer every other package
imports, and it may not import ``fused_memory`` or ``escalation`` — the same
argument ``mcp_markup_middleware.py`` records for its own sinks. So the guard
declares the PORT it depends on (an async resolver callable, plus the sinks)
and each registration site supplies it. That inversion is also why
:class:`Candidate`, :class:`Resolution` and :class:`ResolverUnavailable` are
declared HERE rather than in the resolver: the PRD writes them under C2, whose
home is fused-memory, but a guard typed against a port it cannot import is not
typed at all. Leaf β's ``fused_memory.services.id_resolver`` imports these
three from here rather than defining its own — a second copy is exactly the
lockstep duplication INV-5 forbids.

They live in this module rather than in ``uuid_prefix`` so the detector stays
stdlib-only and dependency-light for consumers — β's resolver, γ's
``tools.py`` — that want the detector or the override key without pulling
fastmcp in through a middleware import.

This module is deliberately NOT re-exported from ``shared/__init__``, the
``mcp_envelope`` / ``storm_counter`` / ``mcp_markup_middleware`` convention.
Here it is load-bearing rather than stylistic: a re-export would make a plain
``import shared`` pull fastmcp for every consumer that never touches
middleware.
"""

from __future__ import annotations

import enum
import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from types import MappingProxyType
from typing import Any, Literal, NamedTuple, get_args

from fastmcp.server.middleware import Middleware

from shared.uuid_prefix import PrefixToken, find_prefix_tokens

__all__ = [
    'FACT_OUTCOMES',
    'FACT_UUID_PREFIX_DETECTED',
    'NAMESPACES',
    'POLICY_MATRIX',
    'RESOLUTION_OUTCOMES',
    'STORM_COUNTED_OUTCOMES',
    'Candidate',
    'FactOutcome',
    'Namespace',
    'PrefixAction',
    'Resolution',
    'ResolutionOutcome',
    'ResolverUnavailable',
    'ToolClass',
    'UuidPrefixGuardMiddleware',
]

logger = logging.getLogger(__name__)


#: Where a candidate id lives. Both stores are always consulted (D4), so one
#: Mem0 match plus one Graphiti match is `ambiguous`, not two separate hits.
Namespace = Literal['mem0', 'graphiti_node', 'graphiti_edge']

#: DERIVED from :data:`Namespace` rather than written out a second time, so the
#: type and the constant cannot drift apart — the :data:`storm_counter.
#: FIRE_MODES` convention. Public so a consumer names a namespace against the
#: constant instead of retyping the literal.
NAMESPACES: tuple[Namespace, ...] = get_args(Namespace)

#: What the resolver concluded. Deliberately does NOT include an "unavailable"
#: member: an outage is an EXCEPTION (:class:`ResolverUnavailable`), never a
#: value the resolver may return, so no caller can accidentally treat a store
#: it could not reach as a store that held nothing (INV-11).
ResolutionOutcome = Literal['unique', 'ambiguous', 'none']

RESOLUTION_OUTCOMES: tuple[ResolutionOutcome, ...] = get_args(ResolutionOutcome)

#: The `outcome` of a ``uuid_prefix_detected`` fact. A superset of
#: :data:`ResolutionOutcome`'s useful members in a different vocabulary,
#: because the fact stream reports what the GUARD did, not what the resolver
#: found: `none` never produces a fact at all (33,076 such tokens in the
#: corpus — a fact stream that is 75% noise teaches readers to skip it), and
#: `resolver_unavailable` has no resolution behind it.
FactOutcome = Literal['expanded', 'rejected', 'forwarded_ambiguous', 'resolver_unavailable']

FACT_OUTCOMES: tuple[FactOutcome, ...] = get_args(FactOutcome)


class Candidate(NamedTuple):
    """One id a prefix could refer to, with enough context to choose between them."""

    namespace: Namespace
    id: str
    preview: str


class Resolution(NamedTuple):
    """What the resolver found for one prefix.

    ``candidates`` is exactly 1 for `unique`, >=2 for `ambiguous`, and empty
    for `none`.
    """

    outcome: ResolutionOutcome
    candidates: tuple[Candidate, ...]


class ResolverUnavailable(Exception):
    """A store could not be reached, so the prefix was never actually checked.

    Carries WHICH store failed, because "the resolver is down" is not
    actionable and "mem0 is down" is. The distinction this type exists to
    protect is the one INV-11 turns on: a store that cannot be reached must
    never be reported as a store that held nothing, or a live id silently
    becomes a `none` and the guard goes quietly inert on a real citation.
    """

    def __init__(self, store: str) -> None:
        super().__init__(f'prefix resolution unavailable: the {store} store could not be reached')
        self.store = store


class ToolClass(enum.StrEnum):
    """Which policy column a tool sits in, DECLARED per registration site.

    Never inferred from a tool's name. A tool is in a class because a
    registration site said so, so a rename cannot silently move it.
    """

    #: Reject on ambiguity: a retry is cheap and the author must choose.
    DEFAULT = 'default'
    #: Forward on ambiguity: losing the write is worse than the defect (D3).
    FORWARD_ON_AMBIGUITY = 'forward_on_ambiguity'
    #: Not a repair site at all. Destructive and id-addressed mutation tools
    #: stay refuse-only (D10) — a prefix must never be expanded before a
    #: delete, merge or edge rewrite. Short-circuits ahead of resolution, so
    #: it has no row in :data:`POLICY_MATRIX`.
    EXEMPT = 'exempt'


class PrefixAction(enum.StrEnum):
    """What the guard does with one call, once outcome and tool class are known."""

    SUBSTITUTE_AND_FORWARD = 'substitute_and_forward'
    REJECT_WITH_CANDIDATES = 'reject_with_candidates'
    FORWARD_WITH_CANDIDATES = 'forward_with_candidates'
    FORWARD_UNCHANGED = 'forward_unchanged'
    INERT = 'inert'


#: PRD §4-C3's table, transcribed. TOTAL over
#: :data:`RESOLUTION_OUTCOMES` x the two non-exempt :class:`ToolClass` members,
#: so no cell is ever reached by falling off the end of a lookup.
#:
#: ``EXEMPT`` has no row by design, and ``FORWARD_UNCHANGED`` appears in none
#: of them: both are decided BEFORE a resolution exists — the first by
#: declaration, the second by a :class:`ResolverUnavailable` or an
#: undeterminable project. A row for either would imply the resolver had
#: answered when it had not.
#:
#: A ``MappingProxyType`` rather than a plain dict because the matrix is the
#: policy: it is read at every call and must not be editable at runtime.
POLICY_MATRIX: Mapping[tuple[ResolutionOutcome, ToolClass], PrefixAction] = MappingProxyType(
    {
        ('unique', ToolClass.DEFAULT): PrefixAction.SUBSTITUTE_AND_FORWARD,
        ('unique', ToolClass.FORWARD_ON_AMBIGUITY): PrefixAction.SUBSTITUTE_AND_FORWARD,
        ('ambiguous', ToolClass.DEFAULT): PrefixAction.REJECT_WITH_CANDIDATES,
        ('ambiguous', ToolClass.FORWARD_ON_AMBIGUITY): PrefixAction.FORWARD_WITH_CANDIDATES,
        ('none', ToolClass.DEFAULT): PrefixAction.INERT,
        ('none', ToolClass.FORWARD_ON_AMBIGUITY): PrefixAction.INERT,
    }
)


#: The two FAIL-SOFT outcomes, and the only ones a storm counter ever sees
#: (INV-4). Declared as DATA so no future edit can start counting a third by
#: flipping a threshold somewhere: `expanded` is never handed to a counter at
#: all.
#:
#: `expanded` is excluded because it is the DESIGNED SUCCESS PATH — at ~10% of
#: 124 writes/day it would fire the 3/3600 thresholds continuously and be
#: ignored, which is how a storm escape stops being an escape. `rejected` is
#: excluded for a different reason: it is not fail-soft at all. The caller is
#: told and can act, so there is no silent degradation for an escape to
#: surface.
STORM_COUNTED_OUTCOMES: frozenset[str] = frozenset({'forwarded_ambiguous', 'resolver_unavailable'})


#: The fact names itself, so a multiplexed sink does not have to infer a
#: record's type from which callable it arrived on — ``FACT_MARKUP_DETECTED``'s
#: convention, and the two streams are read side by side.
FACT_UUID_PREFIX_DETECTED = 'uuid_prefix_detected'


#: The C2 port, declared HERE because the guard is the party that depends on
#: it. ``resolve_prefix(project_id, prefix)``: a project scope and one token,
#: answered from the stores or raising :class:`ResolverUnavailable`.
PrefixResolver = Callable[[str, str], Awaitable[Resolution]]

#: Which project a call is scoped to. Its own contract, and not the markup
#: guard's ``_identity``: this value feeds the RESOLVER, whose universe is a
#: Mem0 collection and a graph ``group_id`` — both keyed by project_id — where
#: an escalation queue is addressed by project_root. Same two argument names,
#: opposite answers.
ProjectFor = Callable[[Mapping[str, Any]], str | None]

#: Either channel may be a plain function OR an ``async def``: the machinery a
#: registration site wires them to is largely async in this repo.
Sink = Callable[[dict[str, Any]], Any | Awaitable[Any]]


class UuidPrefixGuardMiddleware(Middleware):
    """Resolve and police truncated-uuid prefixes at the FastMCP boundary.

    *resolver* and *project_for* are the two injected ports (see the module
    docstring for why they are injected rather than imported). *exempt_tools*
    and *forward_on_ambiguity_tools* are the DECLARED tool classes: a tool is
    in one because a registration site said so, never because its name looked
    a certain way, so a rename cannot silently move a security posture.

    Names are matched BARE. ``context.message.name`` is the in-server name
    (``update_task``), not the agent-facing ``mcp__fused-memory__update_task``,
    so a declaration written with the prefixed spelling would silently never
    match — and a machine-checked declaration that fails open is worse than
    none (INV-1).

    *escalation_sink* and *fact_sink* are injected for the same reason the
    resolver is, and both are invoked DEFENSIVELY: the call's outcome is
    already decided by the time either runs, so a sink that raises is logged
    and never changes what the caller sees. See :meth:`_call_sink`.

    The storm counters are held PER INSTANCE, so no burst state bleeds between
    servers, or between tests in one process.
    """

    def __init__(
        self,
        resolver: PrefixResolver,
        project_for: ProjectFor,
        *,
        # Keyword-only, and deliberately so: these are two same-typed frozensets
        # whose meanings are opposite, and a positional swap would silently
        # exempt the tools meant to forward and vice versa.
        exempt_tools: frozenset[str] | set[str] = frozenset(),
        forward_on_ambiguity_tools: frozenset[str] | set[str] = frozenset(),
        escalation_sink: Sink | None = None,
        fact_sink: Sink | None = None,
        storm_threshold: int = 3,
        storm_window_seconds: float = 3600.0,
        time_provider: Callable[[], float] = time.time,
    ) -> None:
        self._resolver = resolver
        self._project_for = project_for
        self.exempt_tools = frozenset(exempt_tools)
        self.forward_on_ambiguity_tools = frozenset(forward_on_ambiguity_tools)
        self._escalation_sink = escalation_sink
        self._fact_sink = fact_sink
        # Stored, then passed PER record() call — StormCounter's reload-safety
        # contract, so a consumer whose threshold comes from a green-tier config
        # leaf can read it live rather than capturing it at construction.
        self._storm_threshold = storm_threshold
        self._storm_window_seconds = storm_window_seconds
        self._storm_time_provider = time_provider

    # -- the hook ---------------------------------------------------------

    async def on_call_tool(self, context, call_next):
        """Guard one tool call.

        Ordered so the overwhelmingly common path is cheapest. This sits on
        EVERY tool call on the server, so a call carrying no prefix at all
        costs ONE linear scan of its strings and nothing else — in particular
        no awaited resolver round-trip, which would put a store read on the
        event-loop thread for every call the factory makes. The INV-8 bound on
        that scan is stated in ``shared.uuid_prefix``'s module docstring.
        """
        arguments = context.message.arguments or {}

        tokens = find_prefix_tokens(arguments)
        if not tokens:
            return await call_next(context)

        project = self._project_for(arguments)
        if project is None:
            # A prefix is only meaningful inside a project scope — the
            # resolver's universe IS one Mem0 collection and one graph. With no
            # scope there is nothing to resolve against, so the call is
            # forwarded UNCHANGED rather than guessed at against a default
            # project, which is how a citation gets expanded to another
            # project's id.
            return await call_next(context)

        resolutions = await self._resolve(project, tokens)
        actions = self._actions(context.message.name, resolutions)

        # INERT. Read off the MATRIX rather than tested as ``outcome ==
        # 'none'``: a second expression of the policy in the body is exactly
        # what INV-1 forbids, because the two drift and only the declared one
        # is checkable.
        if all(action is PrefixAction.INERT for action in actions.values()):
            return await call_next(context)

        raise NotImplementedError(
            'the substitute / reject / forward arms land with their own steps'
        )

    # -- resolution -------------------------------------------------------

    async def _resolve(
        self, project: str, tokens: tuple[PrefixToken, ...]
    ) -> dict[str, Resolution]:
        """Resolve each DISTINCT token exactly once, in document order.

        C1 states the invariant and this is where it is kept. A call that
        cites one id three times is one store read, not three, and each of
        those reads is a full collection walk — so per-occurrence resolution
        would multiply the cost of exactly the calls that discuss one record at
        length.

        ALL of them, before ANY of them is applied. Three of C3's four outcomes
        state a guarantee about what the tool receives — an ambiguous call on a
        default-class tool writes NOTHING, an unavailable resolver forwards
        UNCHANGED — and one call can carry tokens with different outcomes.
        Resolving and applying interleaved would make those guarantees
        incidental rather than structural: a call whose first token expanded
        and whose second was unresolvable would forward a half-applied
        argument map while reporting that it had changed nothing.
        """
        resolutions: dict[str, Resolution] = {}
        for token in tokens:
            if token.token not in resolutions:
                resolutions[token.token] = await self._resolver(project, token.token)
        return resolutions

    # -- policy -----------------------------------------------------------

    def _tool_class(self, name: str) -> ToolClass:
        """Which policy column *name* sits in, per this registration's declaration.

        :attr:`ToolClass.EXEMPT` is never returned. An exemption short-circuits
        ahead of resolution, which is precisely why it has no row in
        :data:`POLICY_MATRIX` — a cell for it would imply the resolver had been
        consulted.
        """
        if name in self.forward_on_ambiguity_tools:
            return ToolClass.FORWARD_ON_AMBIGUITY
        return ToolClass.DEFAULT

    def _actions(
        self, name: str, resolutions: Mapping[str, Resolution]
    ) -> dict[str, PrefixAction]:
        """The declared action for each resolved token, straight off the matrix.

        Per TOKEN, not per call: one call can cite one id that resolves
        uniquely and another that is ambiguous, and the policy has something
        different to say about each.
        """
        tool_class = self._tool_class(name)
        return {
            token: POLICY_MATRIX[(resolution.outcome, tool_class)]
            for token, resolution in resolutions.items()
        }

    # -- the injected channels --------------------------------------------

    async def _call_sink(self, sink: Sink, record: dict[str, Any], channel: str) -> Any:
        """Invoke one injected sink, AWAITING an async emitter, and never raise.

        A registration site wires the concrete emitters, and the queue and
        escalation machinery they will wire to is largely async in this repo —
        so an ``async def`` emitter is a legitimate thing to be handed. Calling
        one without awaiting it queues NOTHING while handing back a coroutine
        that looks like a result, and the sole trace would be a bare
        ``coroutine was never awaited`` RuntimeWarning. So an awaitable is
        awaited rather than trusted to be a value.

        Never raises. The call's outcome is already decided by the time either
        sink runs, so both channels are purely ADDITIVE: a sink outage costs an
        operator visibility rather than turning a working guard into an outage
        of its own. Logged via ``logger.exception``, never swallowed.
        """
        try:
            result = sink(record)
            if inspect.isawaitable(result):
                result = await result
        except Exception:
            logger.exception(
                'uuid prefix guard: the %s sink failed for %r; the outcome stands',
                channel, record.get('error_type') or record.get('fact'),
            )
            return None
        return result
