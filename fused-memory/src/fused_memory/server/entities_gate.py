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

from fused_memory.utils.referent_resolution import resolve_referents
from fused_memory.utils.validation import InputValidationError

logger = logging.getLogger(__name__)


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
            gate is to REJECT them rather than to assume them away.
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
        # Called for its VALIDATION verdict only, for now — the `.conflicts`
        # this resolution also carries is read by the conflict arm.
        resolve_referents(
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

