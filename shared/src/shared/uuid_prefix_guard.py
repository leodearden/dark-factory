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

WHAT ONE GUARDED CALL COSTS, stated because a middleware on every tool call
owes its reader a bound. A call carrying no prefix pays ONE linear scan of its
strings and nothing else — the INV-8 occupancy bound on that scan is
``shared.uuid_prefix``'s to state. A call that DOES carry prefixes pays at most
:data:`MAX_DISTINCT_PREFIXES_PER_CALL` resolver walks, overlapped
:data:`MAX_CONCURRENT_RESOLUTIONS` at a time; beyond the ceiling it is
forwarded unchanged with the reason on ``meta``. At the PRD's measured ~0.2s
per walk that is ~0.4s of added latency in the worst case rather than the
unbounded N x 0.2s a serial, uncapped resolve would have cost.

The burst escape and the injected-sink discipline are NOT here: they are
:mod:`shared.boundary_storm_escape`, composed. Mechanism that is the same for
every boundary guard belongs in one place (INV-5); what stays here is what is
this guard's own — WHICH outcomes are counted, and what a burst is called.

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

import asyncio
import enum
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from types import MappingProxyType
from typing import Any, Literal, NamedTuple, NoReturn, get_args

from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware

# From ``fastmcp.tools.base``, which is where ``Middleware.on_call_tool``'s own
# return annotation imports it from. NOT ``fastmcp.tools.tool``: that path is a
# runtime-only backward-compat shim, so it imports fine but type-checks as
# reportMissingImports.
from fastmcp.tools.base import ToolResult

from shared.boundary_storm_escape import BoundaryStormEscape, Sink, call_sink
from shared.uuid_prefix import (
    PrefixToken,
    find_prefix_tokens,
    strip_uuid_prefix_override,
    substitute,
    uuid_prefix_override_requested,
)

__all__ = [
    'FACT_OUTCOMES',
    'FACT_UUID_PREFIX_DETECTED',
    'MAX_CONCURRENT_RESOLUTIONS',
    'MAX_DISTINCT_PREFIXES_PER_CALL',
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
    'UUID_PREFIX_STORM_ERROR_TYPE',
    'UuidPrefixGuardMiddleware',
    'project_from_arguments',
]

logger = logging.getLogger(__name__)

#: How this guard names ITSELF in the log lines the shared
#: :mod:`shared.boundary_storm_escape` collaborator writes on its behalf. "the
#: fact sink failed" is not actionable in a repo with two such guards; "uuid
#: prefix guard: the fact sink failed" is.
_OWNER = 'uuid prefix guard'


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


#: The most DISTINCT prefixes one call will have resolved, and the guard's
#: per-call cost bound — the companion to the INV-8 occupancy bound
#: ``shared.uuid_prefix`` states for the DETECTOR.
#:
#: MEASURED (PRD §6): one resolver walk is ~0.2s (34,731 ids in 0.19-0.24s),
#: and §4-C1's corpus says 33,076 of 47,209 prefix-shaped occurrences resolve
#: to `none` — so the overwhelming majority of tokens cost a full two-store
#: walk to learn nothing. Without a ceiling, N is bounded only by the size of
#: the argument map: a session summary quoting thirty hashes would put thirty
#: store walks on one tool call.
#:
#: A call carrying more forwards UNCHANGED with the reason on ``meta`` rather
#: than resolving a prefix of them. Resolving the first sixteen would make WHICH
#: citations got checked depend on where they sat in the document, which is
#: worse than a stated, uniform "not checked".
#:
#: DATA, not a constant buried in a body, for :data:`POLICY_MATRIX`'s reason:
#: the number is a policy an operator may want to read off the module.
MAX_DISTINCT_PREFIXES_PER_CALL = 16

#: How many of those resolutions are in flight at once. The resolutions run
#: CONCURRENTLY — serial awaits would make the per-call bound 16 x ~0.2s ≈ 3.2s
#: of latency on a single write — but not UNBOUNDED: an unlimited fan-out turns
#: one verbose call into sixteen simultaneous two-store walks, which is a way
#: to brown out the stores the guard depends on.
MAX_CONCURRENT_RESOLUTIONS = 8


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

#: The two argument names a guarded tool spells its project scope with. Data,
#: because which one a tool declares is a fact about the surface rather than a
#: rule: ``add_memory`` takes ``project_id``, ``update_task`` and
#: ``submit_task`` take ``project_root``, and the ORDER here is the precedence.
_PROJECT_ARGUMENTS = ('project_id', 'project_root')


def project_from_arguments(known_projects: Mapping[str, str]) -> ProjectFor:
    """Build the shipped :data:`ProjectFor` over a ``{project_id: project_root}`` registry.

    ERRATUM 7 is what this exists for. C3's letter says every guarded tool
    declares ``project_id``; it does not. ``update_task`` and ``submit_task``
    declare ``project_root``, and D3 puts ``update_task`` in the
    forward-on-ambiguity class — so a ``project_for`` written to the PRD's
    letter would return ``None`` for the one tool that ruling most depends on,
    and the guard would go inert there without anything saying so.

    ``project_id`` FIRST, then ``project_root`` translated through the inverse
    registry. That is the opposite precedence to
    ``MarkupGuardMiddleware._identity``, which prefers the root — and the
    reason the two are not one helper. ``_identity``'s value addresses an
    ESCALATION QUEUE, which lives at a filesystem root. This value feeds the
    RESOLVER, whose universe is one Mem0 collection and one graph ``group_id``,
    both keyed by project_id. A root must therefore be TRANSLATED here, never
    preferred, and sharing a helper would couple two answers that must diverge.

    Exact membership only, mirroring
    ``fused_memory/server/markup_guard.py::_resolve_project_root``'s refusal to
    accept an arbitrary path the registry does not vouch for. A root the
    registry has never heard of yields ``None``, as do a non-string and an
    empty value: resolving a citation against a GUESSED project is the one
    outcome worse than leaving the prefix alone, because it expands to another
    project's id and reads as a repair.

    A ``project_id`` is returned without a registry lookup — it already IS the
    resolver's vocabulary, so there is nothing to translate and no reason for
    an unregistered project's citations to go unchecked.

    The inverse map is built ONCE here, at registration, and a duplicate root
    raises immediately, naming both ids: a registry mapping two projects to one
    directory is a startup wiring bug, and picking one of them silently would
    resolve a whole project's citations against the other's ids with nothing
    downstream able to tell. Deterministic and at construction — the
    ``StormCounter`` ``fire_mode`` precedent.

    The signature is what lets leaf γ hand it ``run_server``'s existing
    ``_known_projects_map`` (from ``build_known_projects_map``) with no new
    plumbing, and lets leaf δ pass a registration-time constant instead, since
    no escalation tool declares a project at all.
    """
    by_root: dict[str, str] = {}
    for project_id, project_root in known_projects.items():
        claimed = by_root.setdefault(project_root, project_id)
        if claimed != project_id:
            raise ValueError(
                f'known_projects maps both {claimed!r} and {project_id!r} to '
                f'{project_root!r}; a project_root must identify exactly one '
                'project, or a citation resolves against the wrong one'
            )

    def project_for(arguments: Mapping[str, Any]) -> str | None:
        project_id, project_root = (arguments.get(key) for key in _PROJECT_ARGUMENTS)
        if isinstance(project_id, str) and project_id:
            return project_id
        if isinstance(project_root, str) and project_root:
            return by_root.get(project_root)
        return None

    return project_for


#: Which fact outcome each action reports. A DECLARED table for the same
#: reason :data:`POLICY_MATRIX` is one: the guard has two vocabularies — what
#: it DID (an action) and what it tells an operator (a fact outcome) — and the
#: map between them is policy rather than a body detail.
#:
#: :attr:`PrefixAction.INERT` has no entry, which is the mechanical reason a
#: `none` token never produces a fact: there is no outcome to report.
_FACT_OUTCOME: Mapping[PrefixAction, FactOutcome] = MappingProxyType(
    {
        PrefixAction.SUBSTITUTE_AND_FORWARD: 'expanded',
        PrefixAction.REJECT_WITH_CANDIDATES: 'rejected',
        PrefixAction.FORWARD_WITH_CANDIDATES: 'forwarded_ambiguous',
        PrefixAction.FORWARD_UNCHANGED: 'resolver_unavailable',
    }
)


#: How a fired burst names itself to the escalation sink. The sink owns dedup
#: against an open escalation in the target queue — knowledge this layer does
#: not have and must not guess at — so this is a name, not a decision.
UUID_PREFIX_STORM_ERROR_TYPE = 'uuid_prefix_boundary_storm'


#: The escape from a rejection, and the ONLY action available to the caller.
#: It must not offer to choose: "exactly one candidate across both namespaces,
#: or nothing is changed" is the whole design, and a hint that implied
#: otherwise would be a promise this guard cannot keep.
_REJECT_HINT = (
    'More than one live id in this project starts with that prefix, so there is no '
    'single correct expansion and the choice is yours. Pick the intended id from '
    '`candidates`, put the FULL uuid in place of the prefix in `original_call`, and '
    'resubmit that call verbatim.'
)


class _Outage(Exception):
    """Internal: a :class:`ResolverUnavailable`, tagged with the site that hit it.

    A private wrapper rather than a field added to ``ResolverUnavailable``:
    that type is the RESOLVER's vocabulary — leaf β raises it and knows
    nothing about argument paths — while the site is the GUARD's. Keeping them
    apart is what lets the resolver stay unaware of the boundary it is called
    from.
    """

    def __init__(self, token: PrefixToken, cause: ResolverUnavailable) -> None:
        super().__init__(str(cause))
        self.token = token
        self.store = cause.store


class _Planned(NamedTuple):
    """One token SITE, with its resolution and the action the matrix chose.

    The unit of the guard's second phase. Resolution happens per distinct
    token because a store walk is expensive; everything after it happens per
    site, because an edit and its fact describe a place in the argument map.
    """

    token: PrefixToken
    resolution: Resolution
    action: PrefixAction


def _distinct_sites(tokens: tuple[PrefixToken, ...]) -> dict[str, PrefixToken]:
    """Each DISTINCT token once, in document order, paired with its FIRST site.

    Both halves are load-bearing and computing them together is what keeps them
    consistent. Distinctness is C1's "one store read per cited id, not one per
    occurrence"; document order is what makes the resolver's call sequence, the
    fact stream and the token an outage is attributed to deterministic rather
    than dict-iteration order. The first site is the one an outage names,
    because the earliest occurrence is the one a reader finds first.

    A dict rather than a tuple of pairs: insertion-ordered, so ``tuple(sites)``
    IS the ordered distinct token list and no second structure can disagree
    with it.
    """
    sites: dict[str, PrefixToken] = {}
    for token in tokens:
        sites.setdefault(token.token, token)
    return sites


def _applicable(prefix: str, resolution: Resolution) -> Resolution:
    """Demote a `unique` answer this guard could not actually apply to a `none`.

    The RESOLVER's contract, checked where its answer ARRIVES. C1's
    substitution is MONOTONE — the token must be a literal prefix of its
    replacement, so a wrong expansion stays visible and reversible — and
    ``substitute`` enforces that by raising ``ValueError``.

    That raise would arrive at the worst possible moment. The substitution fold
    runs at the END of delivery, after the `expanded` facts have already been
    published, so an injected resolver that normalised dashes out before
    matching (the natural way to make a 12-char token like ``ccf73ca48240``
    match the canonical ``ccf73ca4-8240-...``) would take the whole call down
    with an unhandled ValueError while the fact stream already claimed the
    substitution had happened. Both halves of that are outcomes this design
    rules out: the call is LOST, and the record of it is a lie. The guard is
    typed against a PORT, not against the one resolver that happens to use
    ``STARTS WITH`` today, and it is the guard that pays.

    A violating answer is a resolver bug, logged at ERROR rather than tolerated
    quietly, and the citation is left exactly as its author wrote it — the same
    thing the guard does for a prefix that matches nothing, which is the honest
    reading of an answer it cannot use.

    Only `unique` can violate this: `ambiguous` never substitutes, and `none`
    carries no candidate at all. An empty ``candidates`` under `unique` is the
    same class of violation and takes the same exit, so no arm downstream has
    to defend against indexing it.
    """
    if resolution.outcome != 'unique':
        return resolution
    candidate = resolution.candidates[0] if resolution.candidates else None
    if candidate is not None and candidate.id.startswith(prefix):
        return resolution
    logger.error(
        'uuid prefix guard: RESOLVER CONTRACT VIOLATION — a unique answer of %r for '
        'prefix %r does not extend it, so it cannot be applied; the citation is left '
        'unexpanded and no expansion is reported',
        candidate.id if candidate is not None else None, prefix,
    )
    return Resolution('none', ())


def _field(token: PrefixToken) -> str:
    """The top-level argument name a token sits under.

    The FLAT vocabulary C3's fact table names, and the same key the markup
    guard's stream calls ``param``, so the two stay queryable the same way.
    Depth is carried by the structured ``path`` beside it, never by encoding
    it into this string.
    """
    return str(token.path[0])


def _agent_id(arguments: Mapping[str, Any]) -> str | None:
    """The agent this call attributes itself to, or ``None``. Never guessed.

    Middleware sits above the tool bodies and has no equivalent of
    fused-memory's ``_resolve_identity(ctx)``; all this layer has is the
    call's own arguments. A non-string value yields ``None``: a caller that
    sent a dict where a name belongs has not identified itself.
    """
    value = arguments.get('agent_id')
    return value if isinstance(value, str) else None


def _substitution_record(planned: _Planned) -> dict[str, Any]:
    """One entry of ``meta.uuid_prefix_repair.substitutions``.

    Carries BOTH ``field`` and ``path``: the flat name because C3's fact
    vocabulary names it, and the structured list because that is what a
    consumer needs to locate a nested edit without parsing anything
    (heuristic 12).
    """
    candidate = planned.resolution.candidates[0]
    return {
        'field': _field(planned.token),
        'path': list(planned.token.path),
        'from': planned.token.token,
        'to': candidate.id,
        'namespace': candidate.namespace,
    }


def _ambiguity_record(planned: _Planned) -> dict[str, Any]:
    """One entry of ``meta.uuid_prefix_repair.ambiguous``.

    The same ``field``/``path`` pair a substitution carries, so a consumer
    reads one location vocabulary across both keys — but ``from``/``to`` are
    absent because nothing moved, and the candidates are here instead so the
    author can still see what the citation could have meant.
    """
    return {
        'field': _field(planned.token),
        'path': list(planned.token.path),
        'token': planned.token.token,
        'candidates': [candidate._asdict() for candidate in planned.resolution.candidates],
    }


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
    and never changes what the caller sees. See
    :func:`shared.boundary_storm_escape.call_sink`.

    The storm counters are held PER INSTANCE — one
    :class:`~shared.boundary_storm_escape.BoundaryStormEscape` per middleware —
    so no burst state bleeds between servers, or between tests in one process.
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
        self._fact_sink = fact_sink
        # The burst escape is a COLLABORATOR, not a set of members on this
        # class: the counting, the dormant sweep and the summary shape are the
        # same for every boundary guard that absorbs a defect, and the version
        # that lived here was a near-verbatim copy of the markup guard's
        # (INV-5). What stays HERE is the part that is this guard's own —
        # WHICH outcomes are counted (:data:`STORM_COUNTED_OUTCOMES`) and what
        # a burst is called (:data:`UUID_PREFIX_STORM_ERROR_TYPE`).
        self._storms = BoundaryStormEscape(
            owner=_OWNER,
            error_type=UUID_PREFIX_STORM_ERROR_TYPE,
            log_event='uuid_prefix_guard_storm',
            escalation_sink=escalation_sink,
            threshold=storm_threshold,
            window_seconds=storm_window_seconds,
            time_provider=time_provider,
        )

    # -- the hook ---------------------------------------------------------

    async def on_call_tool(self, context, call_next):
        """Guard one tool call.

        Ordered so the overwhelmingly common path is cheapest. This sits on
        EVERY tool call on the server, so a call carrying no prefix at all
        costs ONE linear scan of its strings and nothing else — in particular
        no awaited resolver round-trip, which would put a store read on the
        event-loop thread for every call the factory makes. The INV-8 bound on
        that scan is stated in ``shared.uuid_prefix``'s module docstring, and
        the bound on the resolution a call that DOES carry prefixes pays is
        :data:`MAX_DISTINCT_PREFIXES_PER_CALL` — enforced here, three lines
        before the resolver is reached.

        The two DECLARED bypasses come first, both ahead of the scan, which is
        also ``MarkupGuardMiddleware.on_call_tool``'s ordering. Neither is a
        detection, so neither may cost a scan, a resolution, a ``meta`` key or
        a fact.
        """
        name = context.message.name
        # The SAME dict object the tool will be called with, which is what
        # makes the in-place write-back in :meth:`_expand` — and in
        # :meth:`_apply_override` — reach the tool. The ``or {}`` fallback can
        # only produce a fresh dict for an empty argument map, and such a map
        # has neither a ``metadata`` key nor a token, so nothing is ever
        # written back into a detached dict.
        arguments = context.message.arguments or {}

        # B9 — a declared exemption. An id-addressed destructive tool is not a
        # repair site: an expansion the guard chose and an id the author chose
        # are indistinguishable once the delete has run, and only one of them
        # is recoverable. Recording the exemption would additionally make the
        # fact stream measure declarations rather than citations.
        if name in self.exempt_tools:
            return await call_next(context)

        # B17 — the author means the bare token. Forward it, with no fact: a
        # declared intent is not a detection. Whether the flag itself travels
        # onward is the invoked tool's own schema's business — see
        # :meth:`_apply_override`.
        if uuid_prefix_override_requested(arguments.get('metadata')):
            await self._apply_override(context, name, arguments)
            return await call_next(context)

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
            #
            # SAID rather than left to be inferred (INV-11): an absent meta
            # block means the guard saw nothing, and this call is the opposite
            # — it saw a citation and could not scope it. No fact, because
            # nothing was detected in the fact stream's sense; no storm,
            # because an unscoped tool is a wiring fact and not a burst.
            return await self._forward(context, call_next, {'resolver': 'no_project'})

        sites = _distinct_sites(tokens)
        if len(sites) > MAX_DISTINCT_PREFIXES_PER_CALL:
            return await self._forward_over_ceiling(context, call_next, name, len(sites))

        try:
            resolutions = await self._resolve(project, sites)
        except _Outage as outage:
            return await self._forward_outage(
                context, call_next, name, arguments, project, outage
            )

        plan = self._plan(name, tokens, resolutions)
        if not plan:
            return await call_next(context)

        return await self._deliver(context, call_next, name, arguments, project, plan)

    # -- the call whose resolution is not attempted ------------------------

    async def _forward_over_ceiling(self, context, call_next, name: str, distinct: int):
        """Too many distinct prefixes to resolve: forward unchanged, and say so.

        The ``no_project`` arm's shape, for the same reason — the call is
        forwarded byte-identical, ``meta`` states why its citations went
        unchecked, and there is no fact and no storm.

        NOT a fact, because nothing was detected in the fact stream's sense: no
        resolution was attempted, so there is no `expanded` / `ambiguous` /
        `unavailable` to report and the stream would have to invent a fifth
        outcome for a non-event. NOT a storm either, and the line is the one
        ``no_project`` already draws: a storm escape exists for absorption the
        CALLER CANNOT SEE, and this caller is told on its own ``meta``. A run of
        these is a fact about how a batch of writes was authored — someone
        pasted a hash dump — not about the boundary degrading.

        WARNING, so it is not operator-invisible either: a citation that went
        unchecked is worth a log line even when the caller was told.
        """
        logger.warning(
            'uuid prefix guard: %r carries %d distinct prefix-shaped tokens, over the '
            'ceiling of %d; it is forwarded unchanged with its citations unresolved',
            name, distinct, MAX_DISTINCT_PREFIXES_PER_CALL,
        )
        return await self._forward(
            context,
            call_next,
            {
                'resolver': 'too_many_prefixes',
                'distinct_prefixes': distinct,
                'ceiling': MAX_DISTINCT_PREFIXES_PER_CALL,
            },
        )

    # -- the author's override (B17 / D10) ---------------------------------

    @classmethod
    async def _apply_override(cls, context, name: str, arguments: dict[str, Any]) -> None:
        """Decide, FROM THE TOOL'S OWN SCHEMA, whether the flag travels onward.

        ONE override, ONE lifecycle. This boundary is not the only party that
        honours ``allow_uuid_prefix``: leaf γ's write-time layer in
        fused-memory's ``tools.py`` keys off the SAME ``metadata`` map and
        strips the key there, exactly as ``allow_mcp_markup`` is stripped at
        both layers. So whether the flag reaches the tool body decides whether
        the SECOND guard can see the declaration the first one already
        honoured.

        A DELIBERATE DEVIATION from C3's literal wording, recorded here because
        the plan text stays frozen. §4-C3's Override paragraph says the flag is
        "stripped before dispatch"; this strips it only where the tool declares
        no ``metadata``. Consuming it on a tool that DOES would leave an author
        who did exactly what the rejection hint told them to bounced by the next
        guard, with a hint telling them to set a flag they had already set — so
        the paragraph's own purpose is served by forwarding it and its letter is
        not. Filed as an ``escalate_info`` on this task.

        THE OBLIGATION THAT DEVIATION HANDS LEAF γ, spelled out per tool rather
        than left as "the write-time layer": ``add_memory``, ``submit_task`` AND
        ``update_task`` each receive the flag intact, and the two task tools
        PERSIST their ``metadata`` into the task metadata vocabulary. So γ must
        call ``strip_uuid_prefix_override`` on all three and not only on the
        memory write. A miss on either task tool lands ``allow_uuid_prefix`` in
        stored task metadata, where nothing downstream can tell it was
        call-time control — which is precisely what that helper's own docstring
        says must never happen.

        * The tool DECLARES ``metadata`` — forward it UNCHANGED, flag
          included, in whichever shape the caller chose (a dict or the JSON
          string ``submit_task``/``update_task`` also accept, which γ's own
          ``uuid_prefix_override_requested`` parses either way). Consuming the
          flag here would leave an author who did exactly what the rejection
          hint told them to bounced by the next guard, with a hint telling them
          to set a flag they had already set.
        * The tool declares NO ``metadata`` — strip the flag (it is call-time
          control, never payload) and drop the key when that empties the map.
          MEASURED on the sibling guard: most tools a boundary middleware sits
          in front of take no ``metadata`` at all, and forwarding the parameter
          to one of those turns the documented escape into ``ToolError:
          Unexpected keyword argument``. Anything the caller sent BESIDES the
          flag is left in place: on a tool with no ``metadata`` parameter that
          residue is the caller's own bug, and an unknown-parameter error on it
          is visible where a silent discard would not be.

        The awaited ``get_tool`` round-trip is affordable because the clean
        path never reaches this method — an explicit override is rarer than a
        prefix, which is itself rare, and the INV-8 bound is stated against a
        call that carries neither.

        Fail-safe: an unresolvable schema yields ``()`` (see
        :meth:`_schema_params`), which takes the DROP branch — the only branch
        that cannot raise ``Unexpected keyword argument``. A tool that does
        take metadata then loses the flag and its own write-time gate answers
        with an actionable hint, which beats a call the tool cannot accept.
        """
        if 'metadata' in await cls._schema_params(context, name):
            return
        stripped = strip_uuid_prefix_override(arguments['metadata'])
        # Both shapes of "nothing left": ``{}`` from a dict, and the ``'{}'``
        # ``json.dumps`` emits for the JSON-string form.
        if stripped == {} or stripped == '{}':
            del arguments['metadata']
        else:
            arguments['metadata'] = stripped

    @staticmethod
    async def _schema_params(context, name: str) -> tuple[str, ...]:
        """The invoked tool's LIVE parameter names.

        Two measured substrate facts, both of which the PRD's section 6 table
        gets wrong for fastmcp 3.2.2: ``get_tool`` is a COROUTINE and must be
        awaited, and ``tool.parameters`` is a full JSON Schema dict — so the
        names live under ``'properties'``, not on the object itself.

        LIVE rather than declared at registration, because what travels to a
        tool has to be checked against the tool as it actually is; a schema
        captured at wiring time would drift.

        Any failure to resolve one yields an EMPTY tuple, which routes
        :meth:`_apply_override` to the drop branch. That is the fail-safe
        direction: the flag is lost and the write-time gate answers, where the
        other direction is a call fastmcp refuses outright.

        THE RESIDUAL DUPLICATION, named in full so a reader is not told this
        file is INV-5-clean when part of it is not. This method and
        :meth:`_apply_override` are near-twins of
        ``MarkupGuardMiddleware._schema_properties`` / ``_schema_params`` and
        ``._apply_override``: same two substrate facts, same fail-safe
        direction, same strip-or-forward decision keyed off ``metadata``. They
        are NOT extracted, and the storm/sink members that used to sit in this
        file WERE (see :mod:`shared.boundary_storm_escape`), for a reason that
        is not arbitrary — an extraction that leaves both call sites intact can
        be done from this leaf, while unifying these two would mean editing
        that middleware, outside this leaf's declared file scope. Filed rather
        than dropped, as this task's ``cleanup_needed`` note records.
        """
        try:
            tool = await context.fastmcp_context.fastmcp.get_tool(name)
        except Exception:
            logger.exception('uuid-prefix guard could not resolve the schema for %r', name)
            return ()
        parameters = getattr(tool, 'parameters', None) or {}
        properties = parameters.get('properties') if isinstance(parameters, dict) else None
        if not isinstance(properties, dict):
            return ()
        return tuple(properties)

    # -- resolution -------------------------------------------------------

    async def _resolve(
        self, project: str, sites: Mapping[str, PrefixToken]
    ) -> dict[str, Resolution]:
        """Resolve each DISTINCT token exactly once, CONCURRENTLY.

        C1 states the once-per-distinct-token invariant and this is where it is
        kept. A call that cites one id three times is one store read, not
        three, and each of those reads is a full collection walk — so
        per-occurrence resolution would multiply the cost of exactly the calls
        that discuss one record at length.

        CONCURRENT, under a semaphore. Serially, a call citing sixteen distinct
        prefixes would add sixteen ~0.2s store walks to one tool call, and the
        corpus says the overwhelming majority of them cost that walk only to
        arrive at `none`. So the walks overlap, up to
        :data:`MAX_CONCURRENT_RESOLUTIONS` — bounded, because an unlimited
        fan-out browns out the stores the guard depends on. Nothing here is
        order-sensitive in its EFFECT: the answers are keyed by token and the
        document order that matters downstream is *sites*', not completion
        order.

        The FIRST outage in DOCUMENT order is the one raised, not the first to
        arrive, so the fact and the ``meta`` block a caller sees for an
        unreachable store do not depend on which walk lost the race. Every
        sibling walk is awaited to completion before that happens
        (``return_exceptions=True``), so an outage leaves no orphaned task
        writing to a store after the call has been answered.

        ALL of them, before ANY of them is applied. Three of C3's four outcomes
        state a guarantee about what the tool receives — an ambiguous call on a
        default-class tool writes NOTHING, an unavailable resolver forwards
        UNCHANGED — and one call can carry tokens with different outcomes.
        Resolving and applying interleaved would make those guarantees
        incidental rather than structural: a call whose first token expanded
        and whose second was unresolvable would forward a half-applied
        argument map while reporting that it had changed nothing.
        """
        limit = asyncio.Semaphore(MAX_CONCURRENT_RESOLUTIONS)

        async def resolve_one(prefix: str) -> Resolution:
            async with limit:
                return _applicable(prefix, await self._resolver(project, prefix))

        answers = await asyncio.gather(
            *(resolve_one(prefix) for prefix in sites), return_exceptions=True
        )

        resolutions: dict[str, Resolution] = {}
        for prefix, answer in zip(sites, answers, strict=True):
            if isinstance(answer, ResolverUnavailable):
                raise _Outage(sites[prefix], answer) from answer
            if isinstance(answer, BaseException):
                # Not this guard's to interpret — a cancellation, or a bug in
                # the injected resolver. Re-raised rather than folded into the
                # degraded path, which is reserved for the one failure C2
                # DECLARES (INV-11): a store that could not be reached.
                raise answer
            resolutions[prefix] = answer
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

    def _plan(
        self,
        name: str,
        tokens: tuple[PrefixToken, ...],
        resolutions: Mapping[str, Resolution],
    ) -> tuple[_Planned, ...]:
        """Pair each token SITE with its resolution and the action the matrix chose.

        Per site and in document order, not per call: one call can cite an id
        that resolves uniquely and another that is ambiguous, and the policy
        has something different to say about each.

        `none` sites are dropped HERE, and this is the only place the guard
        discards a token — so every arm downstream operates on citations that
        actually resolved and none of them has to re-check for the inert case.
        An empty plan IS the inert outcome. Note that it is read off the
        MATRIX rather than tested as ``outcome == 'none'``: a second
        expression of the policy in the body is exactly what INV-1 forbids,
        because the two drift and only the declared one is checkable.
        """
        tool_class = self._tool_class(name)
        planned = []
        for token in tokens:
            resolution = resolutions[token.token]
            action = POLICY_MATRIX[(resolution.outcome, tool_class)]
            if action is not PrefixAction.INERT:
                planned.append(_Planned(token, resolution, action))
        return tuple(planned)

    # -- delivery ---------------------------------------------------------

    async def _deliver(self, context, call_next, name, arguments, project, plan):
        """Perform the actions the matrix chose. It decides; this executes.

        Facts are emitted BEFORE the outcome is delivered, so a tool body that
        raises cannot take the record of what the guard did down with it.
        """
        rejections = tuple(p for p in plan if p.action is PrefixAction.REJECT_WITH_CANDIDATES)
        if rejections:
            # A rejection is a WHOLE-CALL outcome: nothing is written, so no
            # other site in this plan was acted on and none of them has
            # anything to report. Emitting an `expanded` fact for a token this
            # call never expanded would put a lie in the stream.
            await self._emit_facts(name, rejections, arguments, project)
            self._reject(name, arguments, rejections[0])

        await self._emit_facts(name, plan, arguments, project)

        report: dict[str, Any] = {}
        expansions = tuple(p for p in plan if p.action is PrefixAction.SUBSTITUTE_AND_FORWARD)
        if expansions:
            report['substitutions'] = [_substitution_record(p) for p in expansions]

        ambiguities = tuple(p for p in plan if p.action is PrefixAction.FORWARD_WITH_CANDIDATES)
        if ambiguities:
            report['ambiguous'] = [_ambiguity_record(p) for p in ambiguities]

        await self._record_storms(plan, project, report)

        self._expand(arguments, expansions)
        return await self._forward(context, call_next, report)

    def _reject(self, name: str, arguments: Mapping[str, Any], planned: _Planned) -> NoReturn:
        """Write nothing; bounce the caller with the complete original call.

        RAISES rather than short-circuiting with a middleware-authored
        ``ToolResult``. Measured on the markup guard: a tool with a return
        annotation carries an output schema, and a ToolResult that does not
        satisfy it fails with "Output validation error: 'result' is a required
        property". This middleware is generic over every tool on the server, so
        it cannot know how to satisfy an arbitrary output schema. Raising is
        verified to prevent the tool body from running at all — which is what
        makes "nothing written" TRUE rather than merely intended — and to
        round-trip a ``json.dumps`` payload to the caller byte-intact.

        ``original_call`` is the COMPLETE argument map, and it is the caller's
        own: the resolve phase finishes before any substitution is applied, so
        a call carrying one unique and one ambiguous token bounces with
        nothing expanded rather than half-repaired.

        There is deliberately NO ``repaired_call``, which is where this refusal
        diverges from the markup guard's. A mis-closed tag has one correct
        repair; an ambiguous citation has two correct answers and no way to
        tell them apart, so a repaired call would be the guard making the
        author's choice and presenting it as a fix.

        The first ambiguity in document order is the one named, the markup
        guard's first-hit-wins convention: a refusal is a whole-call outcome
        and the caller resubmits the whole call, so listing the rest would add
        length without adding an action.

        The flat key vocabulary — error / error_type / outcome / tool / field /
        token / hint — is the markup guard's refusal shape, so an agent bounced
        by either guard reads ONE diagnostic contract.
        """
        field = _field(planned.token)
        raise ToolError(
            json.dumps(
                {
                    'error': (
                        f'Tool call rejected: {planned.token.token!r} in {field!r} is a '
                        'truncated uuid that matches more than one live id in this project.'
                    ),
                    'error_type': 'ambiguous_uuid_prefix',
                    'outcome': _FACT_OUTCOME[PrefixAction.REJECT_WITH_CANDIDATES],
                    'tool': name,
                    'field': field,
                    'token': planned.token.token,
                    'candidates': [
                        candidate._asdict() for candidate in planned.resolution.candidates
                    ],
                    'original_call': dict(arguments),
                    'hint': _REJECT_HINT,
                }
            )
        )

    @staticmethod
    def _expand(arguments: dict[str, Any], expansions: tuple[_Planned, ...]) -> None:
        """Apply every substitution, then write the result back IN PLACE.

        REVERSE document order. ``substitute`` is span-exact against offsets
        that an earlier replacement in the same string would have invalidated,
        so folding from the back leaves every remaining span untouched. The
        alternative — re-scanning after each replacement — costs a second
        linear pass and can see different tokens once an expansion has landed.

        The write-back is a clear-and-update on the SAME dict object rather
        than a rebind. MEASURED (``mcp_markup_middleware``): ``call_next``
        re-reads the arguments off the context, so mutating that object is
        what actually reaches the tool; assigning a new dict to a local would
        report a substitution that never happened.
        """
        if not expansions:
            return
        repaired: Any = arguments
        for planned in reversed(expansions):
            repaired = substitute(repaired, planned.token, planned.resolution.candidates[0].id)
        arguments.clear()
        arguments.update(repaired)

    @staticmethod
    async def _forward(context, call_next, report: dict[str, Any]):
        """Let the call through and fold the report into ``meta``.

        The report rides on ``meta`` and NEVER on ``structured_content`` or an
        extra content block. Both alternatives are ruled out by measurement:
        a middleware-authored ``structured_content`` fails the tool's own
        output schema, and an appended content block corrupts any caller that
        indexes ``content[0]``.

        FOLDED into whatever meta came back rather than replacing it —
        ``call_next``'s result already carries ``{'fastmcp': {'wrap_result':
        True}}``, and discarding FastMCP's own signalling to deliver ours
        would be a poor trade.
        """
        result = await call_next(context)
        meta = dict(result.meta or {})
        meta['uuid_prefix_repair'] = report
        return ToolResult(
            content=result.content,
            structured_content=result.structured_content,
            meta=meta,
        )

    # -- facts (INV-2) ----------------------------------------------------

    async def _emit_facts(
        self,
        tool: str,
        sites: tuple[_Planned, ...],
        arguments: Mapping[str, Any],
        project: str,
    ) -> None:
        """Emit one fact per acted-on site, BEFORE the outcome is delivered.

        Identity is read here, ahead of any substitution, so a fact reports the
        arguments the caller actually sent.
        """
        agent_id = _agent_id(arguments)
        for planned in sites:
            await self._emit_fact(
                tool, planned.token, planned.resolution.candidates,
                outcome=_FACT_OUTCOME[planned.action], agent_id=agent_id, project=project,
            )

    async def _emit_fact(
        self,
        tool: str,
        token: PrefixToken,
        candidates: tuple[Candidate, ...],
        *,
        outcome: FactOutcome,
        agent_id: str | None,
        project: str,
    ) -> None:
        """Emit one ``uuid_prefix_detected``, and never change an outcome.

        ONE builder for every arm, so the contracted key set cannot drift
        between them. The record is COMPLETE even where the answer is
        ``None``: ``namespace`` is present and null for a cross-namespace
        ambiguity and for an unreachable store, so a consumer never has to
        tell "no single namespace" apart from "that arm forgot the key".

        Logged at INFO, not WARNING. `expanded` is the DESIGNED success path —
        the same measurement that keeps it out of :data:`STORM_COUNTED_OUTCOMES`
        — and a warning per success teaches a reader to ignore the channel.
        The two fail-soft arms log their own WARNING beside this record.
        """
        namespaces = {candidate.namespace for candidate in candidates}
        fact = {
            'fact': FACT_UUID_PREFIX_DETECTED,
            'tool': tool,
            'field': _field(token),
            'token': token.token,
            'outcome': outcome,
            'candidate_ids': [candidate.id for candidate in candidates],
            'namespace': next(iter(namespaces)) if len(namespaces) == 1 else None,
            'agent_id': agent_id,
            'project': project,
        }
        logger.info(
            'uuid prefix guard: %s tool=%s field=%s token=%s candidates=%r project=%r',
            outcome, tool, fact['field'], fact['token'], fact['candidate_ids'], project,
        )
        if self._fact_sink is None:
            return
        await call_sink(self._fact_sink, fact, 'fact', owner=_OWNER)

    # -- the one degraded path (INV-11) -----------------------------------

    async def _forward_outage(self, context, call_next, name, arguments, project, outage):
        """A store could not be reached: forward unchanged, and say so.

        This is the guard's ONLY degraded path, and the ruling behind it is
        not negotiable: a boundary guard that turns a store outage into lost
        writes is worse than the defect it exists to fix. So an outage is never
        a refusal — not even on a default-class tool, whose ambiguity cell IS a
        refusal, because the guard never learned whether the citation was
        ambiguous at all.

        Nor is it silence. ``resolver`` is on ``meta`` and the store is NAMED,
        so a caller can see that its citations went unchecked rather than
        having to infer it from their absence — and "mem0 is down" is
        actionable where "the resolver is down" is not.

        The whole plan is abandoned, including any expansion already resolved
        earlier in this call. That is what the resolve-then-apply split buys:
        the arguments that reach the tool are byte-identical to the ones
        submitted, so "forwarded unchanged" is structurally true rather than
        very nearly true.
        """
        await self._emit_fact(
            name, outage.token, (),
            outcome=_FACT_OUTCOME[PrefixAction.FORWARD_UNCHANGED],
            agent_id=_agent_id(arguments), project=project,
        )
        logger.warning(
            'uuid prefix guard: the %s store could not be reached; %r forwarded '
            'unchanged with its prefixes unresolved',
            outage.store, name,
        )
        report: dict[str, Any] = {'resolver': 'unavailable', 'store': outage.store}
        storm = await self._storms.record(
            _FACT_OUTCOME[PrefixAction.FORWARD_UNCHANGED], project
        )
        if storm is not None:
            report['storm'] = storm
        return await self._forward(context, call_next, report)

    # -- the storm escape (INV-4) -----------------------------------------

    async def _record_storms(
        self, plan: tuple[_Planned, ...], project: str, report: dict[str, Any]
    ) -> None:
        """Count the fail-soft outcomes this call absorbed, once each.

        Per CALL, not per site. A storm counts WRITES the boundary absorbed,
        and one call citing three ambiguous ids is one authoring event —
        counting citations would let a single verbose write trip the alarm on
        its own and teach an operator to ignore it.

        Only :data:`STORM_COUNTED_OUTCOMES` is counted, and `expanded` is never
        handed to a counter AT ALL rather than being given a high threshold. A
        threshold is a number some future edit can move; not passing the value
        is a shape that edit would have to invent.

        At most one storm-counted outcome can be present in one call — a
        resolver outage aborts the whole resolve phase, so `resolver_unavailable`
        and `forwarded_ambiguous` cannot co-occur — which is why ``report``
        carries a single ``storm`` key rather than a list.
        """
        absorbed = {_FACT_OUTCOME[planned.action] for planned in plan} & STORM_COUNTED_OUTCOMES
        for outcome in sorted(absorbed):
            storm = await self._storms.record(outcome, project)
            if storm is not None:
                report['storm'] = storm
