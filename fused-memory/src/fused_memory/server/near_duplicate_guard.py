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
    from fused_memory.config.schema import ProceduralTopicCluster
    from fused_memory.models.memory import MemoryResult

# Default similarity threshold when config is absent/partial/non-numeric.
# Mirrors Mem0's own cited ~0.92 cosine dedup threshold.
_DEFAULT_NEAR_DUP_THRESHOLD = 0.92

# Default enable flag when config is absent/partial/non-numeric.
_DEFAULT_NEAR_DUP_GUARD_ENABLED = True

# Surfaced in the soft-block dict, the add_memory tool docstring, and
# FUSED_MEMORY_INSTRUCTIONS so the override is discoverable at the point of
# rejection, not just in documentation the agent may not have read.
#
# "Update" names a real tool as of task 3088: update_memory amends a Mem0
# record in place, preserving its point id. Before that shipped, this hint
# instructed agents to do something the API could not do — the only ways to
# "update" were delete-then-re-add (which mints a new id and breaks every
# reference) or nothing at all. Name the tool and its arguments so the
# instruction is executable at the point of rejection.
_NEAR_DUPLICATE_HINT = (
    'This procedural_knowledge content is highly similar to an existing memory '
    '(see matched_memory_id/matched_excerpt). Either skip this write, or amend '
    'that record in place with update_memory(memory_id=<matched_memory_id>, '
    "store='mem0', project_id=..., content=<merged text>, reason=...) — which "
    'preserves its id, unlike delete-then-re-add. If the content is genuinely '
    "distinct, override with metadata={'allow_near_duplicate': True}."
)

# Fallback hint for a topic-cluster soft-block when the matched cluster carries
# no per-cluster ``hint``. Surfaced in the soft-block dict (and the add_memory
# docstring / FUSED_MEMORY_INSTRUCTIONS) so the remediation is discoverable at
# the point of rejection.
_TOPIC_CLUSTER_DEFAULT_HINT = (
    'This procedural_knowledge content matches a KNOWN-contradictory topic '
    'cluster (see topic_id/matched_phrases) that is gated to a human review '
    'task. Do NOT add another entry — search first, then consolidate the '
    'existing entries for this topic by amending one in place with '
    "update_memory(memory_id=..., store='mem0', project_id=..., content=..., "
    'reason=...) and deleting the ones it subsumes, or add context to the '
    'referenced human gate task, instead of accumulating another paraphrased '
    'restatement. If the content is genuinely distinct, override with '
    "metadata={'allow_near_duplicate': True}."
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


def find_matching_topic_cluster(
    content: str,
    clusters: list[ProceduralTopicCluster],
) -> tuple[ProceduralTopicCluster, list[str]] | None:
    """Select the first topic cluster *content* matches, by phrase count OR sufficiency.

    For each cluster in order, counts the DISTINCT ``phrases`` (compared
    case-insensitively) that appear as substrings of *content*, and returns
    ``(cluster, sorted matched phrases)`` for the first cluster that qualifies.
    A cluster qualifies when EITHER:

    * its distinct hit count is ``>= cluster.min_phrase_hits`` (the original
      rule), OR
    * at least one hit phrase is declared in ``cluster.sufficient_phrases``
      -- phrases so distinctive they cannot appear incidentally, which
      therefore qualify the cluster alone (task 3054).

    The second arm exists because hits do NOT aggregate across clusters: a
    write touching one distinctive phrase in each of several clusters used to
    be blocked by none, even though each phrase unambiguously identified its
    own topic. Note the consequence for callers -- on a sufficient-phrase
    match the returned ``matched_phrases`` may contain FEWER entries than the
    cluster's ``min_phrase_hits``. That is correct, not a guard bug: it
    reports exactly what fired.

    A phrase repeated in *content* counts once (distinct-phrase membership,
    not occurrence count); an empty phrase is ignored (it would otherwise
    substring-match everything). Returns ``None`` when *content* is empty,
    *clusters* is empty, or no cluster qualifies.

    Pure and synchronous — mirrors :func:`find_near_duplicate_memory`: it does
    no I/O and no embedding round-trip, and raises nothing on empty input. This
    deterministic matcher catches same-topic paraphrases that score below the
    cosine near-dup threshold (task 2845).
    """
    if not content:
        return None
    content_lower = content.lower()
    for cluster in clusters:
        matched = {
            phrase.lower()
            for phrase in cluster.phrases
            if phrase and phrase.lower() in content_lower
        }
        # Read defensively, matching this module's posture for every other
        # config leaf (`_reconciliation_attr`, `resolve_topic_guard_clusters`):
        # an operator-supplied or older cluster object without the field
        # degrades to today's count-only behaviour rather than raising
        # AttributeError inside the add_memory hot path.
        sufficient = {
            phrase.lower()
            for phrase in (getattr(cluster, 'sufficient_phrases', None) or [])
            if phrase
        }
        if matched and (len(matched) >= cluster.min_phrase_hits or matched & sufficient):
            return cluster, sorted(matched)
    return None


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


def resolve_topic_guard_clusters(memory_service: Any) -> list:
    """Read the configured topic-guard clusters from *memory_service*'s config.

    Same defensive ``getattr`` navigation as :func:`resolve_near_dup_threshold`
    (via :func:`_reconciliation_attr`), returning the configured
    ``procedural_knowledge_topic_guard_clusters`` list iff the leaf is a real
    ``list`` — otherwise an empty list. The ``isinstance(value, list)`` guard
    excludes a missing/``None`` config hop and any Mock attribute an unspecced
    test double might auto-generate, so an empty return reliably means "topic
    guard inert" (task 2845).
    """
    value = _reconciliation_attr(memory_service, 'procedural_knowledge_topic_guard_clusters')
    if isinstance(value, list):
        return value
    return []


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


def build_topic_cluster_block(
    agent_id: str | None,
    content: str,
    cluster: ProceduralTopicCluster,
    matched_phrases: list[str],
) -> dict[str, Any]:
    """Build the structured soft-block dict for a known-topic-cluster write.

    Mirrors :func:`build_near_duplicate_block`'s flat-dict shape (``error`` /
    ``error_type``, echoed ``agent_id`` / ``content_excerpt``, plus a
    remediation ``hint``), but carries topic-guard-specific fields
    (``topic_id`` / ``matched_phrases``) and a distinct ``error_type`` so the
    two guards' agent-facing diagnostics stay cleanly separable. Uses the
    matched cluster's ``hint`` when set, else the module default
    (:data:`_TOPIC_CLUSTER_DEFAULT_HINT`).

    ``matched_phrases`` may contain FEWER entries than the matched cluster's
    ``min_phrase_hits`` -- that is a correct sufficient-phrase block, not a
    guard bug. :func:`find_matching_topic_cluster` also qualifies a cluster on
    a single phrase declared in its ``sufficient_phrases`` (task 3054), and
    this field then reports exactly the one phrase that fired, which is what
    makes the routing unambiguous.
    """
    return {
        'error': 'procedural_knowledge_known_topic_cluster_write_blocked',
        'error_type': 'ProceduralKnowledgeKnownTopicClusterWriteRejected',
        'agent_id': agent_id,
        'content_excerpt': content[:200],
        'topic_id': cluster.topic_id,
        'matched_phrases': matched_phrases,
        'hint': cluster.hint or _TOPIC_CLUSTER_DEFAULT_HINT,
    }
