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

    Only ``store == 'mem0'`` entries carrying a truthy ``memory_id`` are
    resolved. ``store == 'graphiti'`` citations (and any malformed/id-less
    entries) are left UNTOUCHED and never looked up: ``get_memory_by_id`` is a
    Mem0/Qdrant-only point read, so graphiti-store verification is intentionally
    out of scope (it would need a different graph primitive and would otherwise
    false-flag every graphiti citation as a phantom).

    Mirrors ``standing_decision_writer.resolve_evidence_refs``'s found/None
    branching. Returns ``stage1_*`` stats for ``report.stats``.
    """
    # Why re-verify at all, at report-assembly time? Two root causes this pass
    # closes that a cite-time check cannot:
    #   (1) The LLM stage hallucinating/typo-ing an id: the structured-output
    #       JSON fallback (BaseStage.run builds items_flagged straight from the
    #       model's flagged_items) bypasses recon_report.cite_memory's existence
    #       check entirely, so a fabricated id reaches cited_memories unchecked.
    #   (2) Citing an id whose queued add_memory write later FAILED: a TOCTOU
    #       the cite-time-only check structurally cannot catch (it validates at
    #       cite-time, not at report-assembly-time). This run()-time
    #       re-verification is the only check that closes it.
    stats = {
        'stage1_phantom_citations_dropped': 0,
        'stage1_citations_verified': 0,
        'stage1_citation_verification_errors': 0,
    }
    for finding in findings:
        cited = finding.get('cited_memories') or []
        if not cited:
            # Nothing to verify — leave the finding ENTIRELY untouched. In
            # particular, do NOT add an empty ``cited_memories`` key to a finding
            # that never carried one: this pass runs over every flagged finding,
            # and mutating citation-less findings would surprise unrelated
            # consumers (and their tests).
            continue
        kept: list[Any] = []
        for entry in cited:
            # Skip anything we cannot — or must not — resolve, preserving it
            # verbatim and never counting it verified/dropped/errored:
            #   * a non-dict entry (malformed);
            #   * a dict with no truthy memory_id (nothing to look up);
            #   * a non-mem0-store citation. get_memory_by_id is Mem0/Qdrant-only
            #     (a raw point-id read), so resolving a graphiti edge uuid through
            #     it would return not-found for EVERY graphiti citation and
            #     false-flag legitimate graph evidence as a phantom.
            if (
                not isinstance(entry, dict)
                or entry.get('store') != 'mem0'
                or not entry.get('memory_id')
            ):
                kept.append(entry)
                continue
            memory_id = entry.get('memory_id')
            store = entry.get('store')
            try:
                record = await memory_service.get_memory_by_id(project_id, memory_id)
            except Exception as exc:
                # A raised backend error is 'unknown', not 'absent': dropping the
                # citation here would itself be a silent-fail (the exact
                # anti-pattern this fix forbids). KEEP it, surface the
                # uncertainty via a marker, and never propagate — the stage must
                # not crash on a check error.
                kept.append(entry)
                finding.setdefault('citation_failures', []).append(
                    {
                        'memory_id': memory_id,
                        'store': store,
                        'reason': 'verification_error',
                        'error_type': type(exc).__name__,
                    },
                )
                stats['stage1_citation_verification_errors'] += 1
                continue
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
