#!/usr/bin/env python3
"""One-shot + periodic sweep: find near-duplicate ``procedural_knowledge``
Mem0 memories for a project and report (or delete) the losers.

Motivation: Stage-1 reconciliation (finding 2cf1b99f) observed the
worktree-local-venv-vs-shared-checkout-venv gotcha rewritten as a
near-duplicate ``procedural_knowledge`` memory >=13 times -- task-worker
agents write the gotcha ad hoc without first ``search()``-ing Mem0 for
existing coverage, so it recurs faster than any single consolidation pass
absorbs it. This script is the automated backstop: it enumerates a project's
``procedural_knowledge`` memories, clusters near-duplicates by CONTENT
similarity (``difflib.SequenceMatcher.ratio()`` + union-find transitive
closure), picks a survivor per cluster, and reports (or deletes) the rest.

Structural parallel to ``scripts/audit_duplicate_tasks.py`` (union-find
near-duplicate clustering + pick_survivor + dry-run/apply split) and the Mem0
sweep-script family (``scripts/prune_recon_cycle_summaries.py``,
``scripts/sweep_orphan_flag_markers.py``): enumerate via
``memory.mem0.scroll_by_metadata``, delete losers via
``memory.delete_memory`` best-effort.

Safety carve-outs:
  - Dry-run report is the default; deletion only under explicit ``--apply``.
  - Only near-duplicate CLUSTERS above a high similarity threshold are
    actioned; a survivor (canonical-flagged, else oldest) is always retained
    per cluster.
  - ``--apply`` refuses to run when the scan looks truncated
    (``len(records) >= scan_limit``) so a truncated scan never silently
    reaches deletions.
  - Missing/unextractable content degrades to ``''``, which never clusters
    and is never deleted.

Usage
-----
  # Dry run (default): print JSON report, change nothing.
  python scripts/audit_duplicate_memories.py --project-id dark_factory

  # Commit the deletions.
  python scripts/audit_duplicate_memories.py --project-id dark_factory --apply

  # Tune near-duplicate threshold (default 0.85).
  python scripts/audit_duplicate_memories.py --project-id dark_factory \\
      --threshold 0.80
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import logging
import sys
from collections.abc import Iterable, Iterator
from datetime import datetime
from typing import Any

logger = logging.getLogger('audit_duplicate_memories')


# ---------------------------------------------------------------------------
# Pure-function core (no I/O — fully testable without a live Mem0)
# ---------------------------------------------------------------------------

def _sort_groups_deterministically(groups: list[list[dict]]) -> list[list[dict]]:
    """Return a new list of groups in deterministic order without mutating *groups*.

    Members within each returned group are sorted by ``str(id)``; the list of
    groups is then sorted by the minimum id (as a string) in each group.
    Unlike Taskmaster task ids (mostly-numeric, handled via ``_id_as_int`` in
    ``audit_duplicate_tasks.py``), Mem0 memory ids are typically UUIDs, so a
    plain string sort key is used instead.
    """
    sorted_groups = [sorted(g, key=lambda m: str(m.get('id', ''))) for g in groups]
    sorted_groups.sort(key=lambda g: str(g[0].get('id', '')))
    return sorted_groups


def cluster_memories_by_pairs(
    memories: list[dict],
    pairs: Iterable[tuple[int, int]],
) -> list[list[dict]]:
    """Transitive closure over candidate *pairs* → deterministic memory groups.

    The shared clustering core. Candidate GENERATION is pluggable — the
    lexical ``difflib`` scan in ``find_near_duplicate_memory_groups`` and the
    ANN neighbour scan both emit ``(i, j)`` index pairs into this one
    function — but the closure itself has a single implementation. Two
    union-finds that must agree is a defect waiting to happen; this is the
    one site.

    Args:
        memories: Memory dicts. Indices in *pairs* address this list.
        pairs: Any iterable (list, generator, ...) of ``(i, j)`` index pairs
            deemed near-duplicates by some candidate generator. Duplicated
            and mirrored pairs are harmless — union is idempotent.

    Returns:
        List of groups (each a list of >= 2 memories) formed by transitive
        closure of *pairs*; singletons are dropped. Groups are sorted by the
        minimum id within the group so output is deterministic. Members are
        the caller's own dict objects (passed through by identity, not
        copied). Does not mutate *memories*.
    """
    n = len(memories)
    if n < 2:
        return []

    # Union-find (path-compressed) over memory indices.
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, j in pairs:
        union(i, j)

    # Materialise groups: collect memory lists per root, drop singletons.
    groups: dict[int, list[dict]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(memories[i])

    result = [g for g in groups.values() if len(g) >= 2]
    return _sort_groups_deterministically(result)


def find_near_duplicate_memory_groups(
    memories: list[dict],
    threshold: float = 0.85,
) -> list[list[dict]]:
    """Find groups of memories with near-duplicate content using SequenceMatcher.

    Args:
        memories: Memory dicts (each with at least an ``'id'`` and
            ``'content'`` key). No category filtering is performed here —
            the caller (``build_sweep_plan``) is responsible for that.
        threshold: Minimum ``difflib.SequenceMatcher.ratio()`` to flag a pair.

    Returns:
        List of groups (each a list of >= 2 memories) formed by transitive
        closure of all pairs whose content similarity >= threshold. Groups
        are sorted by the minimum id within the group so output is
        deterministic. Does not mutate *memories*.

    Complexity:
        O(n^2) pairs x O(L) per ``SequenceMatcher.ratio()`` call (L = content
        length) — mirrors ``audit_duplicate_tasks.find_near_duplicate_groups``.
        Each pair is pre-filtered through ``quick_ratio()``, a documented
        upper bound on ``ratio()`` that is much cheaper to compute, so
        obviously-dissimilar pairs skip the expensive exact comparison
        without changing which pairs end up >= threshold.
    """
    n = len(memories)
    if n < 2:
        return []

    normalized = [(m.get('content') or '').strip().lower() for m in memories]
    return cluster_memories_by_pairs(
        memories, _lexical_pairs(normalized, threshold),
    )


def _lexical_pairs(
    normalized: list[str],
    threshold: float,
) -> Iterator[tuple[int, int]]:
    """Yield index pairs whose normalised contents are >= *threshold* similar.

    The lexical candidate GENERATOR, split out from the clustering core it
    feeds (``cluster_memories_by_pairs``). Behaviour is unchanged from when
    this loop was inline: same O(n^2) scan, same ``quick_ratio`` pre-filter,
    same empty-content guard.
    """
    n = len(normalized)
    for i in range(n):
        # Empty/blank content never clusters and is never deleted (safe
        # degradation). Guard here because SequenceMatcher(None, '', '').ratio()
        # returns 1.0, which would otherwise union two empty-content records
        # and mark one for deletion — a real memory whose content simply could
        # not be extracted. See the docstring safety carve-out.
        if not normalized[i]:
            continue
        # Reuse one SequenceMatcher across the inner loop instead of
        # constructing a fresh one per pair (set_seq1 is fixed per i;
        # set_seq2 varies per j — identical argument order to the original
        # difflib.SequenceMatcher(None, normalized[i], normalized[j])).
        matcher = difflib.SequenceMatcher()
        matcher.set_seq1(normalized[i])
        for j in range(i + 1, n):
            if not normalized[j]:
                continue
            matcher.set_seq2(normalized[j])
            # quick_ratio()/real_quick_ratio() are documented upper bounds on
            # ratio(), so skipping pairs below threshold here never changes
            # which pairs end up unioned — only cheaper on obvious misses.
            if matcher.real_quick_ratio() < threshold or matcher.quick_ratio() < threshold:
                continue
            if matcher.ratio() >= threshold:
                yield (i, j)


# The three Mem0-backed categories, enumerated ONCE. Graphiti-backed
# categories (entities_and_relations, temporal_facts, decisions_and_rationale)
# are deliberately absent: this detector reads the Mem0/Qdrant collection, and
# a Graphiti edge has no vector here to cluster on.
_ALL_CATEGORIES: tuple[str, ...] = (
    'procedural_knowledge',
    'preferences_and_norms',
    'observations_and_summaries',
)


# Every way the ANN candidate path can lose a candidate, enumerated ONCE so
# the zero-filled disclosure dict has a single definition. The metrics layer
# emits one metric per key; an absent key downstream is indistinguishable
# from "not measured", so these are always all present, always integers,
# always zero-filled on a clean run.
_ANN_DISCLOSURE_KEYS: tuple[str, ...] = (
    'top_k_saturated',
    'below_threshold_dropped',
    'unknown_neighbor_dropped',
    'missing_vector',
)


def ann_pairs_from_neighbors(
    memories: list[dict],
    neighbors_by_id: dict[Any, list[dict]],
    threshold: float,
    *,
    top_k: int | None = None,
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    """ANN neighbour lists → candidate index pairs + a full loss disclosure.

    The ANN sibling of ``_lexical_pairs``: both emit ``(i, j)`` index pairs
    into ``cluster_memories_by_pairs``. This one catches the paraphrase class
    the lexical scan structurally cannot — same meaning, almost no shared
    wording — but unlike a full O(n^2) scan it can lose candidates, so every
    loss is counted rather than absorbed.

    Pure: no I/O. The caller supplies the already-fetched neighbour lists
    (see ``fetch_ann_neighbors``), which keeps this fully testable without a
    live Qdrant.

    Args:
        memories: Scanned records, each optionally carrying ``'vector'``.
            Pair indices address this list.
        neighbors_by_id: ``{memory_id: [{'id': ..., 'score': ...}, ...]}``.
            A record absent from this mapping simply contributes no pairs.
        threshold: Minimum score for a hit to become a candidate pair.
            INCLUSIVE — a hit exactly at *threshold* is kept.
        top_k: The per-record neighbour cap the query was issued with. When
            given, a neighbour list filled to the cap is counted as
            ``top_k_saturated``: further matches may have been cut off.

    Returns:
        ``(pairs, disclosure)``. *pairs* are canonical ``(low, high)`` index
        tuples, deduped and sorted so the output is deterministic.
        *disclosure* maps every key in ``_ANN_DISCLOSURE_KEYS`` to an int:

          - ``top_k_saturated`` — records whose neighbour list hit the cap.
            A property of the QUERY, so counted regardless of how many hits
            then survived the threshold.
          - ``below_threshold_dropped`` — hits scored under *threshold*.
          - ``unknown_neighbor_dropped`` — hits naming a record outside the
            scanned set (excluded by the category filter, or written between
            the scroll and the query); it cannot be clustered, so it is lost.
          - ``missing_vector`` — records with no stored vector to query with,
            so they were never used as a query point.

        A self-hit is dropped silently: a record is always its own nearest
        neighbour, which is expected rather than a loss.

    Does not mutate *memories*.
    """
    disclosure = dict.fromkeys(_ANN_DISCLOSURE_KEYS, 0)
    index_by_id: dict[Any, int] = {m.get('id'): i for i, m in enumerate(memories)}
    pairs: set[tuple[int, int]] = set()

    for i, memory in enumerate(memories):
        if memory.get('vector') is None:
            # Never queried — with_vectors was off, or Qdrant returned the
            # point without a vector. Either way its neighbourhood is unknown.
            disclosure['missing_vector'] += 1
            continue

        hits = neighbors_by_id.get(memory.get('id')) or []
        if top_k is not None and len(hits) >= top_k:
            disclosure['top_k_saturated'] += 1

        for hit in hits:
            hit_id = hit.get('id')
            if hit_id == memory.get('id'):
                continue  # self-hit: expected, not a loss
            if (hit.get('score') or 0.0) < threshold:
                disclosure['below_threshold_dropped'] += 1
                continue
            j = index_by_id.get(hit_id)
            if j is None:
                disclosure['unknown_neighbor_dropped'] += 1
                continue
            pairs.add((i, j) if i < j else (j, i))

    return sorted(pairs), disclosure


def _created_at_sort_key(created_at: Any) -> tuple[int, float]:
    """Sort key placing parseable ``created_at`` oldest-first, unparseable last.

    Returns ``(0, timestamp)`` for a parseable ISO datetime string (so the
    oldest instant sorts first), or ``(1, 0.0)`` for ``None``/unparseable
    values — always after every parseable entry, so a record with no usable
    timestamp is never mistakenly picked as "the oldest".
    """
    if not isinstance(created_at, str) or not created_at:
        return (1, 0.0)
    try:
        return (0, datetime.fromisoformat(created_at).timestamp())
    except (ValueError, TypeError):
        return (1, 0.0)


def pick_survivor(group: list[dict]) -> tuple[dict, list[dict]]:
    """Pick the survivor from a near-duplicate memory group.

    Survivor selection (in order):
      1. A member explicitly flagged canonical (``metadata.get('canonical')``
         truthy) wins, regardless of age.
      2. Otherwise, the oldest member by ``created_at`` (ISO string) wins.
         Records with a missing/unparseable ``created_at`` sort last (see
         ``_created_at_sort_key``) so they are never chosen as "oldest"
         unless every member in the group lacks a usable timestamp.
      3. Ties (equal or absent ``created_at``) are broken by the lowest
         ``str(id)``.

    Raises ``ValueError`` for groups with < 2 memories.

    Returns ``(survivor, losers)`` with ``losers`` = all non-survivor members.
    """
    if len(group) < 2:
        raise ValueError(f'pick_survivor requires a group of >= 2 memories, got {len(group)}')

    def _sort_key(m: dict) -> tuple[bool, int, float, str]:
        canonical = bool((m.get('metadata') or {}).get('canonical'))
        bucket, ts = _created_at_sort_key(m.get('created_at'))
        # `not canonical` so canonical=True sorts first (False < True);
        # bucket/ts ascending so the oldest parseable timestamp sorts first;
        # id ascending so the lowest id wins remaining ties.
        return (not canonical, bucket, ts, str(m.get('id', '')))

    ordered = sorted(group, key=_sort_key)
    survivor = ordered[0]
    losers = [m for m in group if m is not survivor]
    return survivor, losers


def _shared_topic(group: list[dict]) -> str | None:
    """The one non-empty ``metadata.topic`` every member shares, else None."""
    topics = {str((m.get('metadata') or {}).get('topic') or '') for m in group}
    if len(topics) != 1:
        return None
    topic = topics.pop()
    return topic or None


def partition_topic_carveout(
    groups: list[list[dict]],
) -> tuple[list[list[dict]], list[list[dict]]]:
    """Split clusters into ``(actionable, carved_out)`` by shared topic.

    OWNERSHIP SEAM — this rule belongs to task 3136, not to this detector.
    3136's amended Option-C carve-out says a cluster whose members all share
    the same non-empty ``metadata.topic`` is an EXPECTED shape: those records
    are deliberately co-topic'd, so deleting the "losers" would destroy
    intended structure rather than remove accidental duplication. It is
    honored here as INPUT FILTERING only — applied between clustering and
    survivor selection, so carved-out clusters never reach ``pick_survivor``,
    whose contract is untouched. Do NOT re-derive this rule elsewhere.

    The boundary is deliberately narrow: only a FULLY shared, non-empty topic
    carves out. Differing topics, a partially-missing topic, or an
    empty-string topic all stay actionable — a cluster is only exempt when
    every member was deliberately tagged the same way.

    Note ``metadata.topic`` has zero corpus footprint until 3195/3201 land,
    so this is inert-but-correct today by design: implemented now so it is
    not reinvented later, and load-bearing the moment topics appear.
    """
    actionable: list[list[dict]] = []
    carved_out: list[list[dict]] = []
    for group in groups:
        (carved_out if _shared_topic(group) else actionable).append(group)
    return actionable, carved_out


def _max_lexical_ratio(normalized: list[str], member_indices: list[int]) -> float | None:
    """Highest pairwise ``ratio()`` among a cluster's members, ignoring thresholds.

    Reported even when the lexical path LOST (scored below threshold and
    contributed no pair). An operator looking at an ANN-only cluster needs to
    see how the other detector scored it — "ANN 0.905, lexical 0.095" is the
    auditable form of a path disagreement; omitting the loser's score would
    hide exactly the number that explains why the paths differed.
    """
    best: float | None = None
    for a in range(len(member_indices)):
        for b in range(a + 1, len(member_indices)):
            left, right = normalized[member_indices[a]], normalized[member_indices[b]]
            if not left or not right:
                continue
            ratio = difflib.SequenceMatcher(None, left, right).ratio()
            best = ratio if best is None else max(best, ratio)
    return best


def build_sweep_plan(
    memories: list[dict],
    threshold: float = 0.85,
    *,
    categories: Iterable[str] = _ALL_CATEGORIES,
    ann_pairs: Iterable[tuple[int, int]] | None = None,
    ann_scores: dict[tuple[int, int], float] | None = None,
    ann_disclosure: dict[str, int] | None = None,
    ann_threshold: float | None = None,
) -> dict[str, Any]:
    """Orchestrate filtering -> dual-path clustering -> survivor selection -> report.

    Runs BOTH candidate generators and reports both verdicts. The lexical
    ``difflib`` path is kept, not swapped out for ANN: it is high-precision
    on the near-verbatim rewrites that dominate this corpus, while ANN
    catches the paraphrase class lexical structurally cannot. Silently
    replacing one with the other would lose signal, so the actioned plan
    clusters over the UNION of both pair sets and each emitted cluster
    records which path(s) found it.

    Args:
        memories: Raw memory list (any/all categories).
        threshold: Near-duplicate similarity threshold for the LEXICAL path
            only. The ANN cutoff is applied upstream, when the pairs are
            generated (see ``ann_pairs_from_neighbors``).
        categories: Categories to sweep (default: all three Mem0 categories).
            Clustering runs once PER category, so a cross-category union is
            structurally impossible; records outside the set are ignored.
        ann_pairs: Optional ``(i, j)`` index pairs from the ANN path.
            Indices address *memories* as passed in — they are remapped
            across this function's category filter, and a pair whose
            endpoint was filtered out is dropped.
        ann_scores: Optional ``{(i, j): score}`` for those pairs, in the same
            index space, surfaced per cluster as ``ann_max_score``.
        ann_disclosure: Optional counter dict from the ANN path, echoed into
            the report so cap/loss counts travel with the plan they shaped.
        ann_threshold: The ANN cutoff actually in effect, echoed so a reader
            of the report never has to guess which number produced it.

    Returns:
        JSON-serialisable plan dict. Pre-existing keys keep their exact
        meaning and shape (``clusters_total``, ``near_duplicate_groups``
        with the survivor's id/content and full member id list,
        ``delete_candidates`` flattened across clusters), so existing
        consumers are unaffected. Each group additionally carries
        ``found_by`` (``['ann']``, ``['lexical']`` or both),
        ``lexical_clustered``, ``ann_max_score`` and ``lexical_max_ratio``;
        the plan additionally carries ``threshold``, ``ann_threshold``,
        ``ann_disclosure``, ``topic_carveout_clusters`` /
        ``topic_carveout_groups`` (see ``partition_topic_carveout``) and
        ``path_verdicts``
        (``lexical_only_clusters`` / ``ann_only_clusters`` /
        ``both_paths_clusters``).
    """
    near_duplicate_groups: list[dict[str, Any]] = []
    # Mem0/Qdrant point ids can be int or str (ExtendedPointId), so this is
    # not narrowed to list[str].
    delete_candidates: list[Any] = []
    lexical_only = ann_only = both_paths = 0
    clusters_total = 0
    topic_carveout_groups: list[dict[str, Any]] = []

    # Cluster PER CATEGORY. Widening the filter must not merge the corpus: a
    # preference and a procedure that happen to read alike are different
    # kinds of knowledge, and unioning them would be cross-store data loss
    # rather than deduplication. Running the closure once per category makes
    # a cross-category union structurally impossible instead of relying on a
    # downstream check to catch it.
    for category in categories:
        candidate_indices = [
            i for i, m in enumerate(memories) if m.get('category') == category
        ]
        if not candidate_indices:
            continue
        candidates = [memories[i] for i in candidate_indices]
        # Original index -> candidate index, so ANN pairs generated against
        # the unfiltered scan still address the right records after filtering.
        candidate_index = {orig: new for new, orig in enumerate(candidate_indices)}

        normalized = [(m.get('content') or '').strip().lower() for m in candidates]
        lexical_pairs = set(_lexical_pairs(normalized, threshold))

        remapped_ann: set[tuple[int, int]] = set()
        ann_score_by_pair: dict[tuple[int, int], float] = {}
        for pair in ann_pairs or []:
            left, right = candidate_index.get(pair[0]), candidate_index.get(pair[1])
            if left is None or right is None:
                # Endpoint outside this category — either filtered out
                # entirely, or in a DIFFERENT category, in which case the
                # pair is dropped here and cannot union across categories.
                continue
            key = (left, right) if left < right else (right, left)
            remapped_ann.add(key)
            score = (ann_scores or {}).get(pair)
            if score is not None:
                ann_score_by_pair[key] = max(ann_score_by_pair.get(key, score), score)

        clustered = cluster_memories_by_pairs(candidates, lexical_pairs | remapped_ann)
        # Task 3136's Option-C carve-out, applied BEFORE survivor selection so
        # a deliberately co-topic'd cluster can never reach pick_survivor and
        # therefore can never produce a delete candidate.
        groups, carved_out = partition_topic_carveout(clustered)
        for carved in carved_out:
            topic_carveout_groups.append({
                'member_ids': [m.get('id') for m in carved],
                'category': category,
                'topic': _shared_topic(carved),
            })
        clusters_total += len(groups)
        # cluster_memories_by_pairs passes member dicts through by identity,
        # so this recovers each cluster's indices for per-path attribution.
        index_by_identity = {id(m): i for i, m in enumerate(candidates)}

        for group in groups:
            member_indices = [index_by_identity[id(m)] for m in group]
            member_set = set(member_indices)
            in_group = [
                (a, b)
                for a in member_set
                for b in member_set
                if a < b
            ]
            group_lexical = [p for p in in_group if p in lexical_pairs]
            group_ann = [p for p in in_group if p in remapped_ann]

            found_by = []
            if group_ann:
                found_by.append('ann')
            if group_lexical:
                found_by.append('lexical')
            if group_ann and group_lexical:
                both_paths += 1
            elif group_ann:
                ann_only += 1
            else:
                lexical_only += 1

            scores = [ann_score_by_pair[p] for p in group_ann if p in ann_score_by_pair]

            survivor, losers = pick_survivor(group)
            near_duplicate_groups.append({
                'survivor_id': survivor.get('id'),
                'survivor_content': survivor.get('content'),
                'member_ids': [m.get('id') for m in group],
                'category': category,
                'found_by': found_by,
                'lexical_clustered': bool(group_lexical),
                'ann_max_score': max(scores) if scores else None,
                'lexical_max_ratio': _max_lexical_ratio(normalized, member_indices),
            })
            delete_candidates.extend(m.get('id') for m in losers)

    return {
        'clusters_total': clusters_total,
        'near_duplicate_groups': near_duplicate_groups,
        'delete_candidates': delete_candidates,
        'threshold': threshold,
        'ann_threshold': ann_threshold,
        'ann_disclosure': dict(ann_disclosure) if ann_disclosure is not None else None,
        'topic_carveout_clusters': len(topic_carveout_groups),
        'topic_carveout_groups': topic_carveout_groups,
        'path_verdicts': {
            'lexical_only_clusters': lexical_only,
            'ann_only_clusters': ann_only,
            'both_paths_clusters': both_paths,
        },
    }


# ---------------------------------------------------------------------------
# Cluster metrics + the M1 metric series (pure — no I/O)
# ---------------------------------------------------------------------------

_EVAL_ID = 'e6-corpus-health'
"""This detector's eval id — the artifact directory name under the M1 root."""

_GLOBAL_SCOPE = 'all'
"""metric_id suffix for a run-global counter, where no single category owns it.

Every metric_id is ``<name>.<scope>`` so the suffix is always meaningful and
always present. The ANN disclosures are properties of ONE query pass over the
whole scanned set, not of a category, so inventing a per-category split for
them would be a fabricated attribution.
"""

_SECONDS_PER_DAY = 86400.0

# Metrics whose full shape (histogram, per-topic table, per-event table) is
# too wide for a single scalar and lives in the companion details file.
_DETAILED_METRICS: tuple[str, ...] = (
    'cluster_size_max',
    'cluster_size_mean',
    'topic_accretion_rate_mean',
    'consolidation_net_delta',
)


def _metric_id(name: str, scope: str) -> str:
    return f'{name}.{scope}'


def split_plan_by_category(
    plan: dict[str, Any],
    categories: Iterable[str],
) -> dict[str, dict[str, list[dict]]]:
    """Project a sweep plan into per-category slices.

    ``build_sweep_plan`` already clusters ONCE PER CATEGORY and tags every
    emitted group with the category it was clustered under, so this is a
    projection of what the plan states — not a second categorisation that
    could disagree with the one the plan actually used.

    Every requested category gets a slice, empty ones included: an absent key
    downstream would read as "not measured" rather than "measured, found
    nothing".
    """
    slices: dict[str, dict[str, list[dict]]] = {
        category: {'near_duplicate_groups': [], 'topic_carveout_groups': []}
        for category in categories
    }
    for key in ('near_duplicate_groups', 'topic_carveout_groups'):
        for group in plan.get(key) or []:
            slot = slices.get(group.get('category'))
            if slot is not None:
                slot[key].append(group)
    return slices


def _parsed_timestamp(created_at: Any) -> float | None:
    """Epoch seconds for a parseable ``created_at``, else None.

    Delegates to ``_created_at_sort_key``'s tolerance so "what counts as a
    usable timestamp" has one definition shared with survivor selection.
    """
    bucket, ts = _created_at_sort_key(created_at)
    return None if bucket else ts


def _topic_of(record: dict) -> str | None:
    topic = (record.get('metadata') or {}).get('topic')
    return str(topic) if topic else None


def _absorbed_count(metadata: dict) -> int:
    """How many predecessors a consolidation event RECORDS having absorbed.

    Read strictly from ``supersedes``: a list contributes its length, a
    non-empty scalar contributes 1. A record flagged ``canonical`` with no
    ``supersedes`` names no predecessors, so it absorbed 0 *recorded* ones —
    guessing a number for it would put an invented figure into a metric.
    """
    supersedes = metadata.get('supersedes')
    if isinstance(supersedes, str):
        return 1 if supersedes else 0
    if isinstance(supersedes, list | tuple | set):
        return len(supersedes)
    return 1 if supersedes else 0


def _topic_accretion_table(records: list[dict]) -> list[dict[str, Any]]:
    """Per-topic ``members / span_days`` accretion, topics sorted for determinism.

    Only records whose ``created_at`` parses can be placed on the span, so
    ``members`` counts those and unparseable ones are counted separately (see
    the ``unparseable_created_at`` metric) rather than silently inflating a
    rate over a span they did not contribute to.

    A topic spanning zero elapsed time has NO defined rate — ``rate_per_day``
    is None. A rate needs elapsed time; dividing by a fudged epsilon would
    manufacture an enormous rate out of a single record.
    """
    by_topic: dict[str, list[float]] = {}
    for record in records:
        topic = _topic_of(record)
        if topic is None:
            continue
        ts = _parsed_timestamp(record.get('created_at'))
        if ts is not None:
            by_topic.setdefault(topic, []).append(ts)
        else:
            by_topic.setdefault(topic, [])

    table: list[dict[str, Any]] = []
    for topic in sorted(by_topic):
        stamps = sorted(by_topic[topic])
        span_days = (stamps[-1] - stamps[0]) / _SECONDS_PER_DAY if stamps else 0.0
        rate = len(stamps) / span_days if span_days > 0 else None
        table.append({
            'topic': topic,
            'members': len(stamps),
            'span_days': span_days,
            'rate_per_day': rate,
        })
    return table


def _consolidation_events(
    records: list[dict],
    groups: list[dict],
) -> list[dict[str, Any]]:
    """Per-consolidation net delta: what accreted after it minus what it absorbed.

    A consolidation event is a record carrying ``canonical: true`` and/or a
    non-empty ``supersedes``. Its peer set — the population that can accrete
    "after" it — is its shared ``metadata.topic`` when it has one, else the
    near-duplicate cluster it belongs to. An event with neither has no
    identifiable peers, so nothing is attributed to it (0), rather than
    charging it with every later record in the category.

    A negative delta means consolidation is winning; a positive one means the
    topic is re-accreting faster than the consolidation absorbed.
    """
    cluster_by_id: dict[Any, frozenset] = {}
    for group in groups:
        members = frozenset(group.get('member_ids') or ())
        for member_id in members:
            cluster_by_id[member_id] = members

    events: list[dict[str, Any]] = []
    for record in records:
        metadata = record.get('metadata') or {}
        if not (metadata.get('canonical') or metadata.get('supersedes')):
            continue
        topic = _topic_of(record)
        event_ts = _parsed_timestamp(record.get('created_at'))
        peers = cluster_by_id.get(record.get('id'))

        accreted = 0
        if event_ts is not None:
            for other in records:
                if other is record:
                    continue
                if topic is not None:
                    if _topic_of(other) != topic:
                        continue
                elif peers is None or other.get('id') not in peers:
                    continue
                other_ts = _parsed_timestamp(other.get('created_at'))
                if other_ts is not None and other_ts > event_ts:
                    accreted += 1

        absorbed = _absorbed_count(metadata)
        events.append({
            'memory_id': record.get('id'),
            'topic': topic,
            'absorbed': absorbed,
            'accreted_after': accreted,
            'net_delta': accreted - absorbed,
        })
    return events


def compute_cluster_metrics(
    records_by_category: dict[str, list[dict]],
    plan_by_category: dict[str, dict[str, list[dict]]],
    disclosures: dict[str, Any],
    *,
    details_path: str | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Corpus-health metrics for one run, plus the companion details payload.

    Pure: takes the already-fetched records and the already-built plan, so the
    whole metric surface is testable without a live Mem0.

    Kind selection follows M2's rule vocabulary rather than taste. A ``count``
    or ``proportion`` ARMS an exact statistical test, so it is used only where
    a regression is well-defined and directional; everything whose movement is
    not a regression is a ``scalar``, which is reported but never alarmed. In
    particular the topic carve-out count is a scalar: more carve-outs is the
    EXPECTED Option-C shape once 3195/3201 land, so arming a rule on it would
    alarm on the feature working.

    Args:
        records_by_category: ``{category: [record, ...]}`` — the scanned
            corpus. Its per-category lengths are the metric denominators.
        plan_by_category: ``{category: {'near_duplicate_groups': [...],
            'topic_carveout_groups': [...]}}`` (see
            :func:`split_plan_by_category`).
        disclosures: ``{counter_name: int | {category: int}}``. An int is a
            run-global counter (emitted as ``<name>.all``); a mapping is
            per-category (one metric per category, e.g. ``scan_truncated``).
            EVERY key is emitted, zeros included — an absent metric is
            indistinguishable from "not measured", and a cap that fired must
            be a countable, alarmable number rather than invisible recall loss.
        details_path: Filename of the companion details artifact, stamped onto
            the metrics whose full shape lives there.

    Returns:
        ``(metrics, details)``. *metrics* is a list of
        ``shared.memory_eval_metrics.Metric``, sorted by ``metric_id`` so two
        identical runs serialise byte-identically. *details* carries the
        per-category histogram, per-topic accretion table and per-event
        consolidation table.

    Note the clustered-member share is WITHHELD for an empty category and
    named in ``details['categories'][cat]['undefined_metrics']``. M1 forbids a
    zero-trial proportion, and emitting 0.0 would assert "0% of this corpus is
    duplicated" about a corpus that contains nothing to measure.
    """
    from shared.memory_eval_metrics import Metric  # noqa: PLC0415

    metrics: list[Any] = []
    details: dict[str, Any] = {'categories': {}}
    total_corpus = sum(len(v) for v in records_by_category.values())

    for category in sorted(records_by_category):
        records = records_by_category[category]
        slice_ = plan_by_category.get(category) or {}
        groups = slice_.get('near_duplicate_groups') or []
        carveouts = slice_.get('topic_carveout_groups') or []
        corpus_size = len(records)

        sizes = [len(g.get('member_ids') or ()) for g in groups]
        clustered_members = sum(sizes)
        histogram: dict[str, int] = {}
        for size in sizes:
            histogram[str(size)] = histogram.get(str(size), 0) + 1

        accretion = _topic_accretion_table(records)
        rates = [r['rate_per_day'] for r in accretion if r['rate_per_day'] is not None]
        events = _consolidation_events(records, groups)
        unparseable = sum(
            1 for r in records if _parsed_timestamp(r.get('created_at')) is None
        )
        undefined: list[str] = []

        metrics.append(Metric(
            metric_id=_metric_id('near_duplicate_clusters', category),
            kind='count', value=float(len(groups)), n=corpus_size,
            direction='higher_is_worse',
        ))
        if corpus_size > 0:
            metrics.append(Metric(
                metric_id=_metric_id('clustered_member_share', category),
                kind='proportion', value=clustered_members / corpus_size,
                n=corpus_size, denominator=corpus_size,
                direction='higher_is_worse',
            ))
        else:
            undefined.append(_metric_id('clustered_member_share', category))
        metrics.append(Metric(
            metric_id=_metric_id('cluster_size_max', category),
            kind='scalar', value=float(max(sizes)) if sizes else 0.0,
            n=len(groups), details_path=details_path,
        ))
        metrics.append(Metric(
            metric_id=_metric_id('cluster_size_mean', category),
            kind='scalar', value=(sum(sizes) / len(sizes)) if sizes else 0.0,
            n=len(groups), details_path=details_path,
        ))
        metrics.append(Metric(
            metric_id=_metric_id('topic_carveout_clusters', category),
            kind='scalar', value=float(len(carveouts)), n=len(carveouts),
        ))
        metrics.append(Metric(
            metric_id=_metric_id('topic_accretion_rate_mean', category),
            kind='scalar', value=(sum(rates) / len(rates)) if rates else 0.0,
            n=len(rates), details_path=details_path,
        ))
        metrics.append(Metric(
            metric_id=_metric_id('consolidation_net_delta', category),
            kind='scalar', value=float(sum(e['net_delta'] for e in events)),
            n=len(events), details_path=details_path,
        ))
        metrics.append(Metric(
            metric_id=_metric_id('unparseable_created_at', category),
            kind='count', value=float(unparseable), n=corpus_size,
            direction='higher_is_worse',
        ))

        details['categories'][category] = {
            'corpus_size': corpus_size,
            'clustered_members': clustered_members,
            'cluster_size_histogram': histogram,
            'topic_accretion': accretion,
            'consolidation_events': events,
            'undefined_metrics': undefined,
        }

    for name in sorted(disclosures):
        value = disclosures[name]
        if isinstance(value, dict):
            for category in sorted(value):
                metrics.append(Metric(
                    metric_id=_metric_id(name, category), kind='count',
                    value=float(value[category]),
                    n=len(records_by_category.get(category) or ()),
                    direction='higher_is_worse',
                ))
        else:
            metrics.append(Metric(
                metric_id=_metric_id(name, _GLOBAL_SCOPE), kind='count',
                value=float(value), n=total_corpus, direction='higher_is_worse',
            ))

    metrics.sort(key=lambda m: m.metric_id)
    return metrics, details


def build_metric_series(
    project_id: str,
    metrics: Iterable[Any],
    *,
    corpus_counts: dict[str, int],
    eval_id: str = _EVAL_ID,
    run_stamp: str | None = None,
) -> Any:
    """Assemble one run's :class:`MetricSeries`, validating it at BUILD time.

    Everything is routed through ``validate_metric_series`` on the raw payload
    rather than relying on model construction, so a malformed metric — however
    it was produced, ``Metric`` object or hand-built dict — surfaces as one
    ``MetricSchemaError`` here in the runner that computed it. That is M1's
    "rejected at emit time, not read time": by the time the dashboard reads the
    artifact there is nobody left to tell.

    The schema itself lives in ``shared.memory_eval_metrics`` and is never
    restated here (INV-5) — this function only fills it in.

    Args:
        project_id: Scanned project; becomes ``Corpus.project_id``.
        metrics: ``Metric`` models and/or raw metric dicts.
        corpus_counts: Per-category corpus sizes — the denominators behind the
            denominators.
        eval_id: Artifact directory name (default :data:`_EVAL_ID`).
        run_stamp: Run stamp; defaults to ``shared.memory_eval_metrics``'
            helper, which honours ``MEMORY_EVAL_RUN_STAMP``.

    Raises:
        MetricSchemaError: The series is not emittable.
    """
    from shared.memory_eval_metrics import (  # noqa: PLC0415
        SCHEMA_VERSION,
        Metric,
        MetricSeries,
        validate_metric_series,
    )
    from shared.memory_eval_metrics import run_stamp as _run_stamp

    payload = {
        'schema_version': SCHEMA_VERSION,
        'eval_id': eval_id,
        'run_stamp': run_stamp if run_stamp is not None else _run_stamp(),
        'corpus': {'project_id': project_id, 'counts': dict(corpus_counts)},
        'metrics': [
            m.model_dump(mode='json', exclude_none=True) if isinstance(m, Metric) else dict(m)
            for m in metrics
        ],
    }
    # Raises MetricSchemaError (never a bare pydantic error) on any malformed
    # metric, unknown field, or duplicate metric_id. Parsing afterwards cannot
    # fail: the same payload just validated.
    validate_metric_series(payload)
    return MetricSeries.model_validate(payload)


# ---------------------------------------------------------------------------
# I/O layer: fetch (thin, mock-tested)
# ---------------------------------------------------------------------------

# Payload keys tried in order when extracting a Mem0 memory's text content
# from its scroll_by_metadata() 'metadata' payload dict. 'data' is the
# canonical scroll-payload key (mirrors prune_recon_cycle_summaries.py, which
# reads only metadata.get('data')) so it is tried first; 'memory' is a
# search-result-layer key (memory_service.py's MemoryResult.content, built
# from Mem0's search API, not scroll) that can appear stale on a scroll
# payload, so it is only a fallback; 'content' is a defensive third fallback.
_CONTENT_KEYS: tuple[str, ...] = ('data', 'memory', 'content')


async def fetch_memories(
    memory: Any,
    project_id: str,
    *,
    categories: Iterable[str] = _ALL_CATEGORIES,
    scan_limit: int = 5000,
    with_vectors: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Enumerate a project's memories across *categories*, with scan stats.

    Issues ONE ``scroll_by_metadata`` per category. That is not a missed
    optimisation: the primitive builds an AND-equality filter and has no OR,
    so a single widened call cannot express "any of these three categories".

    Each raw record (``{'id', 'created_at', 'metadata'[, 'vector']}``) is
    normalised to ``{'id', 'content', 'category', 'created_at', 'metadata'}``
    (plus ``'vector'`` when *with_vectors*). The top-level ``'category'`` is
    lifted out of the payload because ``build_sweep_plan`` filters on it —
    without the lift every record would read as category ``None`` and the
    sweep would be a silent no-op. Content is extracted by trying
    ``_CONTENT_KEYS`` in order, falling back to ``''``; a record with no
    extractable content never clusters and is never deleted.

    Args:
        memory: Live (or mock) MemoryService instance.
        project_id: Project scope to scan.
        categories: Categories to enumerate (default: all three Mem0 ones).
        scan_limit: Max points PER CATEGORY.
        with_vectors: Fetch each record's stored vector, for ANN candidate
            generation. See ``Mem0Backend.scroll_by_metadata``.

    Returns:
        ``(records, scan_stats)``. *scan_stats* maps each requested category
        to ``{'scanned': n, 'truncated': 0|1}``, counted PER CATEGORY so a cap
        firing on one is never reported as a clean scan of the whole corpus.
        ``truncated`` is an int rather than a bool so it can be summed and
        emitted as a metric directly.

    Raises:
        TimeoutError: A Qdrant read timeout propagates rather than degrading
            to an empty list (``scroll_by_metadata`` no longer swallows it),
            so a timed-out scan is never mistaken for an empty corpus.
    """
    from fused_memory.models.scope import Scope  # noqa: PLC0415

    scope = Scope(project_id=project_id)
    records: list[dict[str, Any]] = []
    scan_stats: dict[str, dict[str, int]] = {}

    for category in categories:
        raw_records = await memory.mem0.scroll_by_metadata(
            scope, {'category': category}, limit=scan_limit, with_vectors=with_vectors,
        ) or []
        scan_stats[category] = {
            'scanned': len(raw_records),
            'truncated': int(len(raw_records) >= scan_limit),
        }
        for record in raw_records:
            payload = record.get('metadata') or {}
            content = ''
            for key in _CONTENT_KEYS:
                value = payload.get(key)
                if isinstance(value, str) and value:
                    content = value
                    break
            normalized: dict[str, Any] = {
                'id': record.get('id'),
                'content': content,
                'category': payload.get('category'),
                'created_at': record.get('created_at'),
                'metadata': payload,
            }
            if with_vectors:
                normalized['vector'] = record.get('vector')
            records.append(normalized)

    return records, scan_stats


async def fetch_procedural_memories(
    memory: Any,
    project_id: str,
    scan_limit: int = 5000,
) -> list[dict[str, Any]]:
    """Single-category ``procedural_knowledge`` fetch — retained for back-compat.

    A thin alias over :func:`fetch_memories`, kept because task 3136 schedules
    this detector and existing callers bind this name. It returns a BARE LIST
    (not the ``(records, scan_stats)`` tuple) and defaults to payload-only, so
    every pre-existing caller is unaffected by the widening to three
    categories and to optional vectors.

    New callers should use :func:`fetch_memories`, which reports per-category
    scan stats the truncation guard and the metrics series both need.
    """
    records, _scan_stats = await fetch_memories(
        memory, project_id,
        categories=('procedural_knowledge',),
        scan_limit=scan_limit,
    )
    return records


# ---------------------------------------------------------------------------
# I/O layer: ANN candidate generation (thin, mock-tested)
# ---------------------------------------------------------------------------

def resolve_ann_threshold(
    memory_service: Any,
    override: float | None = None,
) -> tuple[float | None, dict[str, int]]:
    """Resolve the ANN cosine cutoff. READ, never invented.

    The cutoff comes from ``config.write_triage.t_high`` — a CALIBRATION
    OUTPUT derived by ``scripts/calibrate_write_triage.py`` from measured
    similarity distributions over the curator-labeled corpus, with
    ``calibration_report_path`` recording its provenance. This detector reads
    that single home; it never restates the number and never supplies a
    fallback of its own.

    Follows ``near_duplicate_guard.resolve_near_dup_threshold``'s defensive
    getattr-at-every-hop shape and its ``isinstance(int | float) and not
    bool`` leaf check (which also rejects the auto-generated attributes an
    unspecced Mock would hand back) — but returns ``None`` instead of a
    module default, because substituting an uncalibrated stand-in here is
    precisely the invented threshold that is forbidden.

    Args:
        memory_service: Live (or mock) MemoryService.
        override: Operator-supplied ``--ann-threshold``. Wins when given,
            including over an uncalibrated config.

    Returns:
        ``(threshold, disclosure)``. *threshold* is ``None`` when the config
        is uncalibrated and no override was given — the caller must then
        DISABLE the ANN path rather than guess. *disclosure* carries
        ``ann_disabled_uncalibrated`` (0 or 1) so a disabled path is a
        counted, alarmable metric rather than a silently missing detector.
    """
    if isinstance(override, int | float) and not isinstance(override, bool):
        return float(override), {'ann_disabled_uncalibrated': 0}

    value = getattr(getattr(memory_service, 'config', None), 'write_triage', None)
    t_high = getattr(value, 't_high', None)
    if isinstance(t_high, int | float) and not isinstance(t_high, bool):
        return float(t_high), {'ann_disabled_uncalibrated': 0}

    logger.error(
        'ANN path DISABLED: config.write_triage.t_high is not calibrated '
        '(got %r) and no --ann-threshold override was given. Refusing to '
        'invent a similarity cutoff — run scripts/calibrate_write_triage.py, '
        'or pass --ann-threshold explicitly. The lexical path still runs.',
        t_high,
    )
    return None, {'ann_disabled_uncalibrated': 1}


async def fetch_ann_neighbors(
    memory: Any,
    project_id: str,
    records: list[dict[str, Any]],
    *,
    top_k: int,
    score_threshold: float,
) -> tuple[dict[Any, list[dict[str, Any]]], dict[str, int]]:
    """ANN neighbours for each record, queried with the record's OWN vector.

    Mirrors ``task_curator.search_corpus``'s ``query_points`` call, retargeted
    from the curator's own collection to the Mem0 collection. The query vector
    is the embedding Mem0 already stored (fetched by
    ``scroll_by_metadata(..., with_vectors=True)``), NOT a freshly-embedded
    string: the detector therefore makes zero embedding API calls at runtime
    and cannot drift into a second metric space.

    Args:
        memory: Live (or mock) MemoryService instance.
        project_id: Project scope to query.
        records: Normalised records, each optionally carrying ``'vector'``.
        top_k: Neighbours wanted per record. The query asks for ``top_k + 1``
            so the record's own guaranteed self-hit does not consume a slot.
        score_threshold: Cutoff pushed down into Qdrant.

    Returns:
        ``({memory_id: [{'id', 'score'}, ...]}, disclosure)``. A record with
        no vector is skipped (``ann_pairs_from_neighbors`` counts it as
        ``missing_vector``). *disclosure* carries ``ann_query_errors``: a
        failure on one record is logged and counted, and the sweep continues,
        so one bad point never costs the whole scan.
    """
    from fused_memory.models.scope import Scope  # noqa: PLC0415

    neighbors: dict[Any, list[dict[str, Any]]] = {}
    disclosure = {'ann_query_errors': 0}
    if not records:
        return neighbors, disclosure

    scope = Scope(project_id=project_id)
    collection_name = scope.mem0_collection_name(memory.config.mem0.collection_prefix)
    client = await memory.mem0._get_async_qdrant()  # noqa: SLF001

    for record in records:
        vector = record.get('vector')
        if vector is None:
            continue  # never queried; counted as missing_vector downstream
        record_id = record.get('id')
        try:
            result = await client.query_points(
                collection_name=collection_name,
                query=vector,
                limit=top_k + 1,  # +1 absorbs the guaranteed self-hit
                score_threshold=score_threshold,
                with_payload=False,
            )
        except Exception as exc:
            logger.error('ANN query failed for memory %s: %s', record_id, exc)
            disclosure['ann_query_errors'] += 1
            continue
        neighbors[record_id] = [
            {'id': point.id, 'score': point.score}
            for point in result.points
            if point.id != record_id
        ]

    return neighbors, disclosure


# ---------------------------------------------------------------------------
# I/O layer: apply (thin, mock-tested)
# ---------------------------------------------------------------------------

async def apply_deletions(
    memory: Any,
    project_id: str,
    plan: dict[str, Any],
    *,
    dry_run: bool,
) -> dict[str, int]:
    """Delete the plan's ``delete_candidates`` from Mem0 (best-effort).

    Dry-run performs no ``delete_memory`` calls and returns zero counts.
    Otherwise, each candidate is deleted individually in a try/except so a
    single failure does not abort the remaining deletes (best-effort partial
    progress) — mirrors ``audit_duplicate_tasks.apply_changes``.

    Returns:
        Dict with ``deleted`` and ``delete_errors`` counts.
    """
    if dry_run:
        return {'deleted': 0, 'delete_errors': 0}

    deleted = 0
    delete_errors = 0
    for memory_id in plan.get('delete_candidates', []):
        try:
            await memory.delete_memory(memory_id, store='mem0', project_id=project_id)
            logger.info('Deleted near-duplicate memory %s', memory_id)
            deleted += 1
        except Exception as exc:
            logger.error('Failed to delete memory %s: %s', memory_id, exc)
            delete_errors += 1

    return {'deleted': deleted, 'delete_errors': delete_errors}


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

async def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    )

    import os  # noqa: PLC0415

    from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415
    from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)

    config = FusedMemoryConfig()
    memory = MemoryService(config)
    await memory.initialize()
    try:
        records = await fetch_procedural_memories(
            memory, args.project_id, scan_limit=args.scan_limit,
        )
        logger.info('Fetched %d procedural_knowledge memory/memories', len(records))

        plan = build_sweep_plan(records, threshold=args.threshold)
        print(json.dumps(plan, indent=2, default=str))

        if not args.apply:
            logger.info('Dry run — nothing was modified. Use --apply to commit.')
            return 0

        # Irreversible-deletion guards: never apply against an empty or
        # suspected-truncated scan (mirrors prune_recon_cycle_summaries.py's
        # scan-completeness guard).
        if not records:
            logger.error('ABORT: scan returned 0 records — refusing to apply on an empty scan.')
            return 1
        if len(records) >= args.scan_limit:
            logger.error(
                'ABORT: scan returned %d records >= --scan-limit=%d — scan looks '
                'truncated; refusing to apply. Re-run with a higher --scan-limit.',
                len(records), args.scan_limit,
            )
            return 1

        result = await apply_deletions(memory, args.project_id, plan, dry_run=False)
        logger.info(
            'Applied: deleted %d/%d memory/memories; %d error(s)',
            result['deleted'], len(plan['delete_candidates']), result['delete_errors'],
        )
        return 1 if result['delete_errors'] > 0 else 0
    finally:
        await memory.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--project-id', dest='project_id', required=True,
        help='Project id to scan for near-duplicate procedural_knowledge memories',
    )
    parser.add_argument(
        '--threshold', type=float, default=0.85,
        help='Near-duplicate similarity threshold (default: 0.85)',
    )
    parser.add_argument(
        '--scan-limit', dest='scan_limit', type=int, default=5000,
        help='Maximum number of procedural_knowledge memories to scan (default: 5000)',
    )
    parser.add_argument(
        '--apply', action='store_true',
        help='Commit deletions (default: dry-run, report only)',
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to fused-memory config file (sets CONFIG_PATH env var)',
    )
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == '__main__':
    sys.exit(main())
