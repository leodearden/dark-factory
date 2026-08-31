"""Write-time gate for a caller's DECLARED referents (task 3669, PRD leaf delta).

``utils/referent_resolution`` (leaf gamma) is MECHANISM: it resolves which
referents a write is about and REPORTS what the prose contradicts, but it never
rejects.  This module is the POLICY half of that split — it decides that a
declaration the content contradicts blocks the write.  Gamma says so in its own
words: "Whether a conflict rejects the write is leaf delta's gate to decide."

A pure, server-state-free function rather than a ``create_mcp_server`` closure
like ``_backlog_gate`` / ``_known_project_gate``.  Those two are closures ONLY
because they close over server state; this gate closes over nothing — its whole
input is ``(entities, content, group_id, agent_id)`` — so it follows the closer
precedent of ``server/near_duplicate_guard.py``, ``server/markup_guard.py`` and
``services/completion_claim_gate.py``: module-level hint constants, a
``build_*_block`` factory, and a pure predicate that needs no FastMCP server to
test.

THE POLICY, in one line: reject on CONFLICT, never on ABSENCE (PRD resolved
decision 3).  Rejecting on absence would lose memories from every agent that
does not retry — ``/reflect`` at session end is the named case — so the
undeclared majority flows through untouched and pays for no scan at all.

This module counts nothing.  PRD leaf iota owns the declaration-rate telemetry
surface end to end (``write_ops.params`` plus ``_referent_source_counts``), and
standing up a second counter here would fork that surface before iota has
chosen its shape — the same INV-5 pressure that produced beta and gamma.  The
structured block returned to the caller already satisfies INV-2
(structured-facts-at-failure); the ``logger.warning`` beside it is the interim
operator-greppable record until iota lands.  No kill-switch either: unlike the
near-duplicate guard (whose switch exists because it costs an embedding
round-trip on every write) this gate is a pure in-memory scan, and rejecting on
conflict IS the leaf's entire user-observable signal — a switch could only turn
the feature off.
"""

from __future__ import annotations

import logging
from typing import Any

from fused_memory.utils.canonical_labels import scan_content
from fused_memory.utils.referent_resolution import ReferentSet, resolve_referents
from fused_memory.utils.validation import InputValidationError

logger = logging.getLogger(__name__)

#: Remediation for a declaration the content contradicts.  Names BOTH
#: legitimate remedies, and says explicitly that omitting the parameter is
#: always safe — an agent that reads "your referents were rejected" and cannot
#: tell whether silence is also rejected will start guessing, and a guessed
#: declaration is the misattribution this PRD exists to prevent.
_ENTITIES_CONFLICT_HINT = (
    'The referents you declared in `entities` are contradicted by your own '
    'content (see conflicts/content_referents). Either correct `entities` to '
    'name what the content is actually about, or drop `entities` entirely and '
    'let the content scan derive the set for you. This gate rejects only a '
    'CONFLICT, never an absence: omitting `entities` — or passing [] to record '
    'that you considered referents and none applied — always succeeds.'
)


def entities_gate(
    entities: Any,
    *,
    content: str,
    group_id: str,
    agent_id: str | None = None,
) -> dict[str, Any] | None:
    """Verdict on a caller's declared referents: a block dict, or ``None``.

    Returns ``None`` when the write may proceed.  Returns a structured
    write-blocking dict in exactly two cases — a MALFORMED declaration
    (``error_type='ValidationError'``) and a declaration the content
    CONTRADICTS (``error_type='DeclaredReferentConflictRejected'``).

    Args:
        entities: The caller's ``entities`` argument, VERBATIM and unparsed.
            TRI-STATE (PRD resolved decision 2): ``None`` = never considered,
            ``[]`` = considered and none apply, ``[...]`` = declared.  Typed
            ``Any`` rather than ``list[dict] | None`` on purpose: this is raw
            MCP input, so every wrong shape is reachable and the point of the
            gate is to REJECT them rather than to assume them away. That
            reachability is not free — it holds only because BOTH write tools
            annotate ``entities`` as ``Any`` too (same precedent as
            ``get_tasks(statuses: Any = None)``). Narrow either annotation and
            pydantic rejects the commonest mistakes ahead of this function,
            with a raw ToolError carrying none of the remediation gamma folded
            into its message — so the hint would reach an agent or not
            depending on WHICH way it got the shape wrong.
            ``tests/server/test_entities_gate_ingestion.py::TestEveryMalformedShapeReachesTheGate``
            fails if that happens.
        content: The verbatim write body, scanned for the referents the prose
            actually names.
        group_id: The local project id.  Must be the CANONICAL one — the
            tool's prologue has already run ``_canonicalize_project_id_arg``,
            and ``Scope.graphiti_group_id`` IS ``project_id``, so the gate and
            ``MemoryService`` classify local-vs-foreign referents identically.
        agent_id: Echoed into the block, matching every sibling guard.

    ``metadata`` is deliberately NOT a parameter and ``metadata=None`` is
    passed to the resolver.  ``ReferentResolution.conflicts`` is populated on
    the DECLARED path ONLY — ``resolve_referents`` returns from that branch
    before the metadata bridge is ever consulted — so metadata provably cannot
    change this gate's verdict.  Accepting it would imply a dependency that
    does not exist and invite a future reader to "fix" a divergence between the
    tool's raw metadata and the service's ``cleaned_meta`` (post
    ``_extract_causation`` / ``strip_markup_override`` /
    ``_normalize_task_id_metadata``) that has no observable effect.  The
    metadata bridge stays exactly where leaf epsilon put it: inside
    ``MemoryService``, on the write that actually enqueues.

    Only ``entities is None`` short-circuits — never ``not entities``.  Two
    distinct reasons, both load-bearing:

    * ``[]`` is a REAL declaration in the PRD's tri-state ("considered, none
      apply") and must flow through the one resolver so the write stamps
      ``source='declared'``.  Collapsing it onto ``None`` would silently
      downgrade the write to ``'derived'``/``'none'`` and corrupt the very
      counter leaf iota reads.
    * A falsy NON-list (``''``, ``{}``, ``0``) is a wiring bug, not an
      absence, and must reach ``_declared_referents`` to be REJECTED.

    Placing the ``None`` arm first is what makes "never on absence" STRUCTURAL
    rather than merely tested: with no declaration there is nothing that COULD
    conflict, so no code path exists that could reject an undeclared write.
    """
    if entities is None:
        # Tri-state arm 1: never considered. The undeclared majority pays for
        # no scan, and no reachable path can reject it.
        return None

    try:
        resolution = resolve_referents(
            declared=entities,
            metadata=None,
            content=content,
            group_id=group_id,
        )
    except InputValidationError as exc:
        # Gamma documents its InputValidationError contract as TOTAL — every
        # malformed shape (wrong container, non-dict element, unknown key, bad
        # id/kind/project_id type, unregistered kind, path-shaped or
        # task-vocabulary project_id) leaves `_declared_referents` as exactly
        # this type — written so THIS handler catches all of them.
        return build_entities_malformed_block(agent_id, content, exc)

    if resolution.conflicts:
        # A SECOND CALL of the one scanner, on the REJECTION PATH ONLY.
        #
        # `ReferentResolution` deliberately publishes `.conflicts` (the
        # DECLARED side) and `.ambiguous` but withholds the scan's `.refs`, so
        # the content side of the disagreement is not derivable from the
        # resolution — and INV-2 requires a rejection to name both sides.  This
        # is a second CALL, not a second COPY: the label vocabulary stays at
        # its one normative site in canonical_labels, so it is not the INV-5
        # lockstep duplication beta exists to prevent, and leaf zeta already
        # sets this precedent by re-deriving the producer's ambiguity set from
        # `content`.  Scoping it inside this branch keeps the ACCEPTED path at
        # exactly one scan.
        scan = scan_content(content, group_id=group_id)
        block = build_entities_conflict_block(
            agent_id,
            content,
            declared=resolution.referents,
            conflicts=resolution.conflicts,
            # `refs + ambiguous`, never `refs` alone. `_conflicting_referents`
            # tests membership against `refs | ambiguous`, so a scan that found
            # ONLY an ambiguous referent still satisfies its "the scan saw this
            # kind" precondition and CAN produce a conflict. Reporting `refs`
            # there would emit "the content cites nothing" beside a rejection
            # for contradicting the content — a guard malfunction to read, not
            # a decision.
            content_referents=scan.refs + scan.ambiguous,
        )
        # The interim operator-greppable record until leaf iota's telemetry
        # surface lands (INV-4: a rejection nobody can count is a rejection
        # nobody can see). Carries the same fields as the block, so grepping
        # the log and reading the agent's error tell the same story.
        logger.warning(
            'entities_gate: rejecting write — declared referents contradict the '
            'content. agent_id=%r group_id=%r declared=%r conflicts=%r '
            'content_referents=%r',
            agent_id,
            group_id,
            block['declared'],
            block['conflicts'],
            block['content_referents'],
        )
        return block

    return None


def build_entities_malformed_block(
    agent_id: str | None,
    content: str,
    exc: InputValidationError,
) -> dict[str, Any]:
    """Build the write-blocking dict for a STRUCTURALLY invalid declaration.

    The flat house shape ``_canonicalize_project_id_arg`` already returns
    (``{'error': <message>, 'error_type': 'ValidationError'}``), plus the
    echoed ``agent_id`` / ``content_excerpt`` every sibling pre-service guard
    in ``server/tools.py`` carries.

    NO ``hint`` key, on purpose.  Gamma states that an exception has no room
    for a separate structured key and therefore folds
    ``_DECLARED_REFERENT_HINT`` — the accepted entry shape and the remediation
    — into the raised message itself.  A second copy at this boundary would be
    exactly the lockstep duplication this batch keeps guarding against, and the
    two spellings would drift the first time the accepted shape widened.
    """
    return {
        'error': str(exc),
        'error_type': 'ValidationError',
        'agent_id': agent_id,
        'content_excerpt': content[:200],
    }


def build_entities_conflict_block(
    agent_id: str | None,
    content: str,
    *,
    declared: ReferentSet,
    conflicts: ReferentSet,
    content_referents: ReferentSet,
) -> dict[str, Any]:
    """Build the write-blocking dict for a declaration the content refutes.

    A DISTINCT ``error_type`` from the malformed arm, following the same
    precedent as ``ProceduralKnowledgeNearDuplicateWriteRejected`` versus
    ``ProceduralKnowledgeKnownTopicClusterWriteRejected``.  The two failures
    need opposite remedies — one is "your list is the wrong shape", the other
    is "your list disagrees with your own prose" — and merging them behind one
    type would leave leaf iota unable to separate structural rejections from
    the semantic misattribution signal this PRD exists to produce.

    Args:
        declared: The FULL declared set, so the agent sees what it sent beside
            what was refused.
        conflicts: The subset the content contradicts — a per-referent verdict,
            not a set-level one, which is what catches ``[3127, 3129]`` where
            3129 is an adjacent-number typo and 3127 corroborates.
        content_referents: What the scan actually found, ``refs + ambiguous``.

    All three are spelled as ``Referent.node_name`` ('Task 3127' own-project,
    'reify:132' foreign) because that is the spelling the agent and the graph
    both use, and the one leaf zeta's reason strings already emit.
    """
    return {
        'error': 'declared_referent_conflict',
        'error_type': 'DeclaredReferentConflictRejected',
        'agent_id': agent_id,
        'content_excerpt': content[:200],
        'declared': _node_names(declared),
        'conflicts': _node_names(conflicts),
        'content_referents': _node_names(content_referents),
        'hint': _ENTITIES_CONFLICT_HINT,
    }


def _node_names(referents: ReferentSet) -> list[str]:
    """Spell a referent set the way the agent and the graph both spell it.

    ``Referent.node_name`` rather than the wire dict, and a plain ``list`` so
    the block stays JSON-safe without a caller having to coerce it.
    """
    return [referent.node_name for referent in referents]

