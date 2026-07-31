"""Citation integrity for Stage-1 reconciliation (tasks 2978, 3108).

This module owns one invariant end-to-end: **a cited memory id must resolve.**
It covers both halves of that invariant, so there is one owner rather than two
mechanisms that can drift.

Half 1 — recon-report citations (task 2978). ``verify_cited_memories`` walks
each finding's ``cited_memories`` list and re-resolves every cited Mem0 id
against the live store, so a finding's claim can never be silently backed by an
id that does not (or no longer) exist.

Half 2 — task-metadata citations (task 3108). ``find_citation_occurrences`` /
``repoint_metadata`` / ``repoint_task_citations`` find and rewrite live
pointers to a memory id that is about to be deleted, so a consolidation delete
repoints citations BEFORE the irreversible destruction rather than leaving
dangling pointers behind it.

These are small, single-purpose helpers in the ``reconciliation/`` convention
(cf. ``flag_dedup``/``task_filter``), returning ``stage1_*`` stats that
``MemoryConsolidator.run()`` merges into ``report.stats``.
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


# --------------------------------------------------------------------------- #
# Task-metadata citation scanning (task 3108)
# --------------------------------------------------------------------------- #


def find_citation_occurrences(metadata: Any, memory_id: str) -> list[str]:
    """Return a dotted/indexed path for EVERY occurrence of ``memory_id``.

    A pure, side-effect-free recursive walk over ``metadata``:

    - **dicts** are descended by key (``memory_hints`` -> ``memory_hints.x``);
    - **lists/tuples** are descended by index (``queries`` -> ``queries[0]``);
    - **strings** match on SUBSTRING containment, so an id embedded in free
      prose (``'see canonical entry <uuid> for ...'``) is found, not just a
      scalar whose whole value is the id.

    Never mutates its input, and never raises on malformed input: a ``None``,
    a bare string, an int or a top-level list all return ``[]``.

    **Why this is a mechanical all-keys scan and not a known-field lookup.**
    Incident failure mode (1) was exactly a known-field/known-task
    enumeration: a hand-written pass over the citation-bearing tasks found 3
    of 8, and the 5 it missed included the pending/dispatchable ones — i.e.
    the ones that actually mattered, because they were still going to be
    dispatched against a dead pointer. The reflex fix ("just list the
    citation-bearing keys") is worse than it looks here: the known citation
    key names from that incident (``mem0_canonical_entry``,
    ``mem0_cluster_entries``, ``x_memory_write_caution``) are *reify-project*
    task-DB keys with ZERO occurrences in this repo, so an allowlist built
    from them would be empty-by-construction and would silently pass every
    delete. Metadata is ``extra='allow'`` (``shared/task_metadata.py:474``)
    with a wide-open Tier-C ``x_`` namespace, so the set of keys that may
    carry a citation is not knowable in advance. Scan everything instead.

    ``memory_id`` is matched in full. A truncated 8-char prefix therefore
    matches nothing, which is deliberate: two distinct UUIDs can share a
    prefix (the hazard ``prompts/stage1.py:99-113`` warns about), and a
    prefix match would repoint an unrelated entry.
    """
    if not memory_id or not isinstance(memory_id, str):
        return []
    if not isinstance(metadata, dict):
        # A non-dict blob carries no addressable top-level keys. Returning []
        # (rather than raising) keeps the scan callable on whatever the task
        # backend hands back, including a missing/NULL metadata column.
        return []

    paths: list[str] = []

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, str):
            if memory_id in node:
                paths.append(path)
            return
        if isinstance(node, dict):
            for key, value in node.items():
                child = f'{path}.{key}' if path else str(key)
                _walk(value, child)
            return
        if isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                _walk(value, f'{path}[{index}]')
            return
        # Any other scalar (int/float/bool/None) cannot contain a uuid.

    _walk(metadata, '')
    return paths


def repoint_metadata(
    metadata: Any,
    old_id: str,
    new_id: str,
) -> tuple[Any, int]:
    """Rewrite every occurrence of ``old_id`` to ``new_id``; return ``(blob, count)``.

    Pure: the input is deep-copied, never mutated, so a caller whose
    subsequent write fails is left holding its original object rather than a
    half-rewritten one.

    Rewrite rules mirror :func:`find_citation_occurrences`'s match rules
    exactly, and the count is computed on the SAME traversal, so the two
    functions cannot drift apart:

    - a string equal to ``old_id`` becomes ``new_id``;
    - any other string gets ``str.replace(old_id, new_id)``, so an id embedded
      in free prose is repointed with the surrounding text preserved verbatim;
    - dicts and lists are descended; every other scalar is returned as-is.

    A string containing ``old_id`` more than once counts as ONE occurrence, to
    stay path-for-path consistent with the scanner (which reports one path per
    string, not one per byte offset).
    """
    count = 0

    def _rewrite(node: Any) -> Any:
        nonlocal count
        if isinstance(node, str):
            if old_id in node:
                count += 1
                return node.replace(old_id, new_id)
            return node
        if isinstance(node, dict):
            return {key: _rewrite(value) for key, value in node.items()}
        if isinstance(node, list):
            return [_rewrite(value) for value in node]
        if isinstance(node, tuple):
            return tuple(_rewrite(value) for value in node)
        return node

    if not old_id or not isinstance(old_id, str) or not isinstance(metadata, dict):
        # Nothing addressable to rewrite. Still deep-copy dict input so the
        # return value is never an alias of the caller's object.
        return (_rewrite(metadata) if isinstance(metadata, dict) else metadata), 0

    return _rewrite(metadata), count
