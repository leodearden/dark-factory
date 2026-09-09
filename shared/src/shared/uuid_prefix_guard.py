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
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, NamedTuple, get_args

__all__ = [
    'FACT_OUTCOMES',
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
]


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
