"""Post-assembly citation verification for Stage-1 reconciliation (task 2978).

``verify_cited_memories`` walks each finding's ``cited_memories`` list and
re-resolves every cited Mem0 id against the live store, so a finding's claim
can never be silently backed by an id that does not (or no longer) exist.

It is a small, single-purpose async helper in the ``reconciliation/``
convention (cf. ``flag_dedup``/``task_filter``): it takes the memory service
and post-processes ``report.items_flagged`` in place, returning ``stage1_*``
stats that ``MemoryConsolidator.run()`` merges into ``report.stats``.
"""

from __future__ import annotations

from typing import Any


async def verify_cited_memories(
    findings: list[dict[str, Any]],
    memory_service: Any,
    project_id: str,
) -> dict[str, int]:
    """Verify each finding's cited Mem0 memories still resolve; drop phantoms.

    For every ``cited_memories`` entry, resolve its ``memory_id`` via
    ``memory_service.get_memory_by_id(project_id, memory_id)`` (the
    no-silent-fail raw Qdrant point read):

    - resolves (truthy record) -> KEEP the citation and count it verified;
    - genuine not-found (``None``) -> DROP it from ``cited_memories`` and append
      a ``{memory_id, store, reason: 'memory_not_found'}`` marker to the
      finding's ``citation_failures`` list, so the phantom claim is surfaced
      rather than silently retained.

    Mirrors ``standing_decision_writer.resolve_evidence_refs``'s found/None
    branching. Returns ``stage1_*`` stats for ``report.stats``.
    """
    stats = {
        'stage1_phantom_citations_dropped': 0,
        'stage1_citations_verified': 0,
        'stage1_citation_verification_errors': 0,
    }
    for finding in findings:
        cited = finding.get('cited_memories') or []
        kept: list[Any] = []
        for entry in cited:
            memory_id = entry.get('memory_id')
            store = entry.get('store')
            record = await memory_service.get_memory_by_id(project_id, memory_id)
            if record:
                kept.append(entry)
                stats['stage1_citations_verified'] += 1
            else:
                finding.setdefault('citation_failures', []).append(
                    {'memory_id': memory_id, 'store': store, 'reason': 'memory_not_found'},
                )
                stats['stage1_phantom_citations_dropped'] += 1
        finding['cited_memories'] = kept
    return stats
