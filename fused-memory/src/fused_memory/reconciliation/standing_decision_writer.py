"""Entity-standing-decision WRITE path + evidence gate (task 2895 β).

Builds on α (task 2894, ``standing_decision_constants`` + ``recon_ledger``'s
``upsert_entity_standing_decision``). This module owns:

* the two-armed **evidence gate** (:func:`evaluate_evidence_gate`) that decides
  whether an entity standing decision is authorized to be written, and
* the writer helper (:func:`write_entity_standing_decision`) that samples the
  decision-time edge-count fingerprint, writes the active ledger row through
  α's ``upsert_entity_standing_decision``, and best-effort mirrors it to mem0
  for ε's advisory check — mirroring ``flag_dedup.write_suppression_record``.

Placement (design decision): a NEW sibling module rather than an addition to
``flag_dedup.py`` — it keeps β's writer/gate next to α's constants, imports only
``recon_ledger`` + ``standing_decision_constants`` (no circular import), and is
imported by ``server/recon_report.py`` via a local import inside the state
method (matching that file's FastMCP local-import style).

**Reads never consult mem0 for the standing-decision RECORD** (the flag_dedup
precedent): Hook A/B (γ/δ) read the SQLite ledger only. The gate's mem0 reads
here are EVIDENCE reads — a separate concern from record reads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from fused_memory.reconciliation.standing_decision_constants import (
    MEM0_KIND_INVESTIGATION_OUTCOME,
    RECORD_KIND_ENTITY_STANDING_DECISION,
    STANDING_DECISION_TTL_DAYS,
    STATE_ACTIVE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Evidence-ref vocabulary
# ---------------------------------------------------------------------------

# A cited evidence ref carries a ``type`` naming its store/kind. Only mem0 refs
# are locally resolvable (via memory_service.get_memory_by_id) and thus eligible
# to count toward the gate; every other type (escalation ids, task refs, …) is
# foreign — recorded verbatim as provenance but NEVER counted (PRD §Authorization
# gate: orchestrator escalation ids etc. are context, not authorization).
EVIDENCE_TYPE_MEM0 = 'mem0'

# ---------------------------------------------------------------------------
# Arm 1 — human-authorship predicate (PRD Open Question 1)
# ---------------------------------------------------------------------------

# An evidence mem0 record counts toward arm 1 only if its ``metadata.agent_id``
# starts with one of these prefixes. ``claude-interactive`` is the documented
# interactive/operator agent id (CLAUDE.md write-tagging convention). Agent/stage
# ids (``reconciliation-stage-*``, ``claude-task-*``) are deliberately NOT
# human-touched, preserving the under-suppression bias (PRD decision 10). A
# prefix tuple keeps the allowlist single-source and extensible.
HUMAN_AUTHORED_AGENT_ID_PREFIXES: tuple[str, ...] = ('claude-interactive',)

# ---------------------------------------------------------------------------
# Arm 2 — independent-investigation predicate (PRD Open Question 2)
# ---------------------------------------------------------------------------

# Arm 2 requires ≥ this many DISTINCT non-empty investigation_outcome run_ids
# for the entity (distinct run_ids model independent investigations). Qdrant's
# scroll can filter on the metadata fields but cannot dedup by run_id, so the
# distinct count is computed Python-side after the scroll.
ARM2_MIN_DISTINCT_RUNS = 3

# ---------------------------------------------------------------------------
# Rejection payload vocabulary (single-source; INV-1 ValidationError+hint)
# ---------------------------------------------------------------------------

# Structured rejection returned to the Stage-2 LLM when neither arm is met, so
# the agent can self-correct mid-cycle rather than persisting a poisoned row.
STANDING_DECISION_INSUFFICIENT_EVIDENCE = 'insufficient_evidence'
STANDING_DECISION_INSUFFICIENT_EVIDENCE_ERROR_TYPE = 'StandingDecisionEvidenceInsufficient'

# Arm identifiers + their remedies — the single source both ``unmet_arms`` and
# the human-readable ``hint`` derive from (no lockstep-duplicated wording).
ARM1_KEY = 'human_authored_mem0_evidence'
ARM2_KEY = 'independent_investigation_outcomes'
_ARM1_REMEDY = (
    'cite at least one locally-resolvable, human-authored mem0 evidence record '
    '(agent_id starting with ' + ', '.join(HUMAN_AUTHORED_AGENT_ID_PREFIXES) + ')'
)
_ARM2_REMEDY = (
    f'accumulate at least {ARM2_MIN_DISTINCT_RUNS} investigation_outcome mem0 records '
    'for this entity_uuid with actionable=false and distinct run_ids'
)


def _build_rejection(*, arm1_satisfied: bool, arm2_satisfied: bool, arm2_distinct_run_count: int) -> dict[str, Any]:
    """Assemble the structured ``insufficient_evidence`` rejection dict.

    Lists every UNMET arm and its remedy, and embeds the observed distinct-run
    count so the caller sees exactly how far arm 2 fell short.
    """
    unmet_arms: list[dict[str, Any]] = []
    if not arm1_satisfied:
        unmet_arms.append({'arm': ARM1_KEY, 'needs': _ARM1_REMEDY})
    if not arm2_satisfied:
        unmet_arms.append(
            {
                'arm': ARM2_KEY,
                'needs': _ARM2_REMEDY,
                'observed_distinct_run_count': arm2_distinct_run_count,
            }
        )
    hint = (
        'Insufficient evidence to authorize an entity standing decision — satisfy at '
        f'least ONE arm. Arm 1: {_ARM1_REMEDY}. Arm 2: {_ARM2_REMEDY} '
        f'(found {arm2_distinct_run_count} distinct run_id(s) so far).'
    )
    return {
        'error': STANDING_DECISION_INSUFFICIENT_EVIDENCE,
        'error_type': STANDING_DECISION_INSUFFICIENT_EVIDENCE_ERROR_TYPE,
        'unmet_arms': unmet_arms,
        'hint': hint,
    }


def _is_human_authored(agent_id: Any) -> bool:
    """True iff *agent_id* is a string starting with a human-authored prefix."""
    if not isinstance(agent_id, str) or not agent_id:
        return False
    return any(agent_id.startswith(prefix) for prefix in HUMAN_AUTHORED_AGENT_ID_PREFIXES)


async def resolve_evidence_refs(
    memory_service: Any,
    project_id: str,
    evidence: Any,
) -> list[dict[str, Any]]:
    """Stamp each cited evidence ref with ``locally_resolved`` (+ author).

    Each ref is a dict carrying at least a ``type`` and ``id``. mem0 refs are
    resolved via ``memory_service.get_memory_by_id(project_id, id)``: found ⇒
    ``locally_resolved=True`` and ``agent_id`` copied from the record's
    ``metadata``; not found ⇒ ``locally_resolved=False``. Every non-mem0
    (foreign) ref is stamped ``locally_resolved=False`` verbatim WITHOUT a mem0
    lookup. The returned list is the single stamped copy consumed by BOTH the
    gate (arm 1) and the writer (row-payload provenance) — evidence is resolved
    exactly once.
    """
    resolved: list[dict[str, Any]] = []
    for ref in evidence or []:
        stamped: dict[str, Any] = dict(ref)
        if stamped.get('type') == EVIDENCE_TYPE_MEM0:
            ref_id = stamped.get('id')
            record = (
                await memory_service.get_memory_by_id(project_id, ref_id)
                if ref_id
                else None
            )
            if record:
                stamped['locally_resolved'] = True
                metadata = record.get('metadata') or {}
                stamped['agent_id'] = metadata.get('agent_id')
            else:
                stamped['locally_resolved'] = False
        else:
            stamped['locally_resolved'] = False
        resolved.append(stamped)
    return resolved


@dataclass
class EvidenceGateResult:
    """Outcome of :func:`evaluate_evidence_gate`.

    ``satisfied`` is the OR of the two independent authorization arms. On a
    rejection (neither arm), ``rejection`` carries the structured
    ``_ERR_*``-style dict the Stage-2 tool returns to the LLM; it is ``None``
    when satisfied.
    """

    satisfied: bool
    arm1_satisfied: bool
    arm2_satisfied: bool
    arm2_distinct_run_count: int
    resolved_evidence: list[dict[str, Any]] = field(default_factory=list)
    rejection: dict[str, Any] | None = None


async def evaluate_evidence_gate(
    memory_service: Any,
    *,
    project_id: str,
    entity_uuid: str,
    evidence: Any,
) -> EvidenceGateResult:
    """Evaluate the two-armed evidence gate for an entity standing decision.

    * **Arm 1** — ≥1 cited evidence ref that is a mem0 ref, locally resolvable,
      AND authored by a human (:func:`_is_human_authored`).
    * **Arm 2** — ≥``ARM2_MIN_DISTINCT_RUNS`` DISTINCT investigation_outcome
      run_ids for this entity (``actionable=False``), counted Python-side after
      the metadata scroll.

    The gate is satisfied when EITHER arm holds.
    """
    resolved = await resolve_evidence_refs(memory_service, project_id, evidence)
    arm1_satisfied = any(
        ref.get('type') == EVIDENCE_TYPE_MEM0
        and ref.get('locally_resolved')
        and _is_human_authored(ref.get('agent_id'))
        for ref in resolved
    )

    # Arm 2 — count DISTINCT non-empty run_ids among the entity's dismissed
    # (actionable=False) investigation_outcome records. Qdrant filters the three
    # metadata fields; run_id dedup is Python-side. A record lacking a run_id
    # cannot establish independence and is not counted.
    outcomes = await memory_service.get_memories_by_metadata(
        project_id,
        {
            'kind': MEM0_KIND_INVESTIGATION_OUTCOME,
            'entity_uuid': entity_uuid,
            'actionable': False,
        },
    )
    distinct_run_ids: set[str] = set()
    for record in outcomes or []:
        run_id = (record.get('metadata') or {}).get('run_id')
        if isinstance(run_id, str) and run_id:
            distinct_run_ids.add(run_id)
    arm2_distinct_run_count = len(distinct_run_ids)
    arm2_satisfied = arm2_distinct_run_count >= ARM2_MIN_DISTINCT_RUNS

    satisfied = arm1_satisfied or arm2_satisfied
    rejection = (
        None
        if satisfied
        else _build_rejection(
            arm1_satisfied=arm1_satisfied,
            arm2_satisfied=arm2_satisfied,
            arm2_distinct_run_count=arm2_distinct_run_count,
        )
    )
    return EvidenceGateResult(
        satisfied=satisfied,
        arm1_satisfied=arm1_satisfied,
        arm2_satisfied=arm2_satisfied,
        arm2_distinct_run_count=arm2_distinct_run_count,
        resolved_evidence=resolved,
        rejection=rejection,
    )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

# Evidence-payload marker for an operator/backfill (η) authorization bypass.
EVIDENCE_TYPE_OPERATOR_AUTHORIZATION = 'operator_authorization'


class EvidenceGateRejected(Exception):
    """Raised by :func:`write_entity_standing_decision` when the evidence gate is
    not satisfied and no ``authorized_by`` override was supplied.

    Carries the structured ``.rejection`` dict (the same
    ``insufficient_evidence`` payload :func:`evaluate_evidence_gate` builds) so
    the Stage-2 state method can catch it and return the dict to the LLM
    verbatim — matching the recon-report cite_* returned-error convention.
    """

    def __init__(self, rejection: dict[str, Any]) -> None:
        self.rejection = rejection
        message = (
            rejection.get('hint')
            if isinstance(rejection, dict)
            else str(rejection)
        )
        super().__init__(message or 'insufficient_evidence')


async def write_entity_standing_decision(
    memory_service: Any,
    *,
    project_id: str,
    entity_uuid: str,
    grounds: str,
    evidence: Any,
    authorized_by: str | None = None,
) -> dict[str, Any]:
    """Write one ACTIVE entity standing decision to α's ledger (gated).

    Mirrors ``flag_dedup.write_suppression_record``: authoritative SQLite-ledger
    write + a best-effort mem0 mirror (reads never consult mem0 for the RECORD).

    Flow:

    1. Resolve the cited evidence refs once (payload provenance). With an
       ``authorized_by`` override (operator/interactive/backfill η), append an
       operator-authorization provenance entry and SKIP the gate. Otherwise run
       :func:`evaluate_evidence_gate` and raise :class:`EvidenceGateRejected`
       (carrying the structured rejection) when neither arm is satisfied — no
       row is written.
    2. Sample the decision-time edge-count fingerprint via graphiti; a sampling
       failure PROPAGATES (loud-over-silent, INV-1) rather than persisting a row
       with a bogus count that would corrupt ζ's growth sweep.
    3. Compute ``decided_at = now(UTC)`` and ``expires_at = decided_at +
       STANDING_DECISION_TTL_DAYS``; upsert the ACTIVE row through α's
       ``upsert_entity_standing_decision`` (which validates
       ``grounds ∈ GROUNDS_ENUM`` and never-None ``expires_at``, raising
       ``ValueError`` on violation).
    4. Best-effort mirror to mem0 for ε's advisory check — its metadata carries
       the record kind + entity_uuid + grounds. A mirror failure is swallowed
       and never blocks the authoritative ledger write.

    Returns ``{status:'written', entity_uuid, grounds, edge_count_at_decision,
    expires_at, decided_at}``.
    """
    if authorized_by:
        resolved = await resolve_evidence_refs(memory_service, project_id, evidence)
        resolved.append(
            {
                'type': EVIDENCE_TYPE_OPERATOR_AUTHORIZATION,
                'authorized_by': authorized_by,
                'locally_resolved': False,
            }
        )
    else:
        gate = await evaluate_evidence_gate(
            memory_service,
            project_id=project_id,
            entity_uuid=entity_uuid,
            evidence=evidence,
        )
        if not gate.satisfied:
            raise EvidenceGateRejected(gate.rejection or {})
        resolved = gate.resolved_evidence

    # Decision-time edge-count fingerprint — loud-fail (no row) on a sampling
    # error rather than persisting a poisoned count.
    edges = await memory_service.graphiti.get_valid_edges_for_node(
        entity_uuid, group_id=project_id
    )
    edge_count_at_decision = len(edges)

    decided_dt = datetime.now(UTC)
    decided_at = decided_dt.isoformat()
    expires_at = (decided_dt + timedelta(days=STANDING_DECISION_TTL_DAYS)).isoformat()

    await memory_service.recon_ledger.upsert_entity_standing_decision(
        project_id=project_id,
        entity_uuid=entity_uuid,
        grounds=grounds,
        decided_at=decided_at,
        expires_at=expires_at,
        edge_count_at_decision=edge_count_at_decision,
        evidence=resolved,
        state=STATE_ACTIVE,
    )

    # Best-effort mem0 mirror (ε advisory check) — never blocks the ledger write.
    try:
        await memory_service.add_memory(
            content=(
                f'ENTITY STANDING DECISION entity_uuid={entity_uuid} grounds={grounds}'
            ),
            category='observations_and_summaries',
            metadata={
                'kind': RECORD_KIND_ENTITY_STANDING_DECISION,
                'entity_uuid': entity_uuid,
                'grounds': grounds,
                'edge_count_at_decision': edge_count_at_decision,
                'decided_at': decided_at,
                'expires_at': expires_at,
            },
            project_id=project_id,
            _source=RECORD_KIND_ENTITY_STANDING_DECISION,
        )
    except Exception:
        logger.debug(
            'write_entity_standing_decision: mem0 mirror failed for '
            'entity_uuid=%s grounds=%s project_id=%s (ledger row already committed)',
            entity_uuid,
            grounds,
            project_id,
            exc_info=True,
        )

    return {
        'status': 'written',
        'entity_uuid': entity_uuid,
        'grounds': grounds,
        'edge_count_at_decision': edge_count_at_decision,
        'expires_at': expires_at,
        'decided_at': decided_at,
    }
