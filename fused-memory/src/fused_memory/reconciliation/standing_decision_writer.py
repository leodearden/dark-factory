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

from dataclasses import dataclass, field
from typing import Any

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
    * **Arm 2** — ≥3 DISTINCT investigation_outcome run_ids for this entity
      (implemented in a later step; stubbed to 0 for now).

    The gate is satisfied when EITHER arm holds.
    """
    resolved = await resolve_evidence_refs(memory_service, project_id, evidence)
    arm1_satisfied = any(
        ref.get('type') == EVIDENCE_TYPE_MEM0
        and ref.get('locally_resolved')
        and _is_human_authored(ref.get('agent_id'))
        for ref in resolved
    )

    # Arm 2 is implemented in a later step; stub it to unsatisfied/0 for now.
    arm2_distinct_run_count = 0
    arm2_satisfied = False

    satisfied = arm1_satisfied or arm2_satisfied
    return EvidenceGateResult(
        satisfied=satisfied,
        arm1_satisfied=arm1_satisfied,
        arm2_satisfied=arm2_satisfied,
        arm2_distinct_run_count=arm2_distinct_run_count,
        resolved_evidence=resolved,
        rejection=None,
    )
