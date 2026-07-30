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
        ``ann_disclosure`` and ``path_verdicts``
        (``lexical_only_clusters`` / ``ann_only_clusters`` /
        ``both_paths_clusters``).
    """
    near_duplicate_groups: list[dict[str, Any]] = []
    # Mem0/Qdrant point ids can be int or str (ExtendedPointId), so this is
    # not narrowed to list[str].
    delete_candidates: list[Any] = []
    lexical_only = ann_only = both_paths = 0
    clusters_total = 0

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

        groups = cluster_memories_by_pairs(candidates, lexical_pairs | remapped_ann)
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
        'path_verdicts': {
            'lexical_only_clusters': lexical_only,
            'ann_only_clusters': ann_only,
            'both_paths_clusters': both_paths,
        },
    }


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
