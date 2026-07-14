"""Write-time near-duplicate guard for procedural_knowledge add_memory writes.

Recurring near-duplicate ``procedural_knowledge`` writes on well-known
gotchas (e.g. the ``.task/`` gitignore git-add gotcha, the ``write_queue.db``
path-mismatch note) consumed repeated reconciliation Stage-1 cleanup cycles
because dedup only ran reactively, after the duplicate had already landed.
This module moves that check to write time: :func:`find_near_duplicate_memory`
is a pure, synchronous selector over already-fetched search results, and
:func:`build_near_duplicate_block` builds the structured soft-block dict the
``add_memory`` MCP tool returns instead of calling the service when a match
is found (see ``server/tools.py``).

This module intentionally does no I/O — the caller is responsible for
fetching candidate ``MemoryResult`` objects (typically via
``MemoryService.search``) and for fail-open behaviour on search errors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fused_memory.models.enums import MemoryCategory, SourceStore

if TYPE_CHECKING:
    from fused_memory.models.memory import MemoryResult

# Default similarity threshold when config is absent/partial/non-numeric.
# Mirrors Mem0's own cited ~0.92 cosine dedup threshold.
_DEFAULT_NEAR_DUP_THRESHOLD = 0.92

# Default enable flag when config is absent/partial/non-numeric.
_DEFAULT_NEAR_DUP_GUARD_ENABLED = True

# Surfaced in the soft-block dict, the add_memory tool docstring, and
# FUSED_MEMORY_INSTRUCTIONS so the override is discoverable at the point of
# rejection, not just in documentation the agent may not have read.
_NEAR_DUPLICATE_HINT = (
    'This procedural_knowledge content is highly similar to an existing memory '
    '(see matched_memory_id/matched_excerpt). Search first and update or skip '
    'instead of writing a duplicate. If the content is genuinely distinct, '
    "override with metadata={'allow_near_duplicate': True}."
)


def find_near_duplicate_memory(
    results: list[MemoryResult],
    threshold: float,
    *,
    category: MemoryCategory = MemoryCategory.procedural_knowledge,
    source_store: SourceStore = SourceStore.mem0,
) -> MemoryResult | None:
    """Select the best near-duplicate candidate from already-fetched *results*.

    Returns the highest ``relevance_score`` result that matches *category*
    and *source_store* and whose ``relevance_score >= threshold``, or
    ``None`` if no result qualifies. Defensively filters out results with a
    mismatched category or source_store even when their score is high —
    callers may pass unfiltered/mixed search results.

    Pure and synchronous: does no I/O and raises nothing on empty input.
    """
    qualifying = [
        r
        for r in results
        if r.category == category
        and r.source_store == source_store
        and r.relevance_score >= threshold
    ]
    if not qualifying:
        return None
    return max(qualifying, key=lambda r: r.relevance_score)


def resolve_near_dup_threshold(memory_service: Any) -> float:
    """Read the near-dup similarity threshold from *memory_service*'s config.

    Navigates ``memory_service.config.reconciliation.procedural_knowledge_near_dup_threshold``
    defensively via ``getattr`` at each hop, returning the module default
    (:data:`_DEFAULT_NEAR_DUP_THRESHOLD`) whenever any hop is missing, ``None``,
    or the leaf value is not a real ``float``/``int`` (this also excludes
    ``bool``, which is an ``int`` subclass, and any Mock attribute an
    unspecced test double might auto-generate).
    """
    value = _reconciliation_attr(memory_service, 'procedural_knowledge_near_dup_threshold')
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return _DEFAULT_NEAR_DUP_THRESHOLD


def resolve_near_dup_guard_enabled(memory_service: Any) -> bool:
    """Read the near-dup guard enable flag from *memory_service*'s config.

    Same defensive ``getattr`` navigation as :func:`resolve_near_dup_threshold`,
    falling back to :data:`_DEFAULT_NEAR_DUP_GUARD_ENABLED` unless the leaf
    value is a real ``bool``.
    """
    value = _reconciliation_attr(memory_service, 'procedural_knowledge_near_dup_guard_enabled')
    if isinstance(value, bool):
        return value
    return _DEFAULT_NEAR_DUP_GUARD_ENABLED


def _reconciliation_attr(memory_service: Any, attr: str) -> Any:
    config = getattr(memory_service, 'config', None)
    reconciliation = getattr(config, 'reconciliation', None)
    return getattr(reconciliation, attr, None)


def build_near_duplicate_block(
    agent_id: str | None,
    content: str,
    match: MemoryResult,
    threshold: float,
) -> dict[str, Any]:
    """Build the structured soft-block dict returned by the add_memory tool.

    Mirrors the shape of the existing pre-service guards in ``server/tools.py``
    (``count_snapshot_write_blocked`` et al.): a flat dict with ``error``/
    ``error_type``, echoed ``agent_id``/``content_excerpt``, plus fields
    identifying the matched memory and a remediation ``hint``.
    """
    return {
        'error': 'procedural_knowledge_near_duplicate_write_blocked',
        'error_type': 'ProceduralKnowledgeNearDuplicateWriteRejected',
        'agent_id': agent_id,
        'content_excerpt': content[:200],
        'matched_memory_id': match.id,
        'similarity': match.relevance_score,
        'threshold': threshold,
        'matched_excerpt': match.content[:200],
        'hint': _NEAR_DUPLICATE_HINT,
    }
