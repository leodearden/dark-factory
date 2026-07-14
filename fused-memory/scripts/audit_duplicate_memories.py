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

import difflib
import logging
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
    """
    n = len(memories)
    if n < 2:
        return []

    normalized = [(m.get('content') or '').strip().lower() for m in memories]

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

    for i in range(n):
        for j in range(i + 1, n):
            ratio = difflib.SequenceMatcher(None, normalized[i], normalized[j]).ratio()
            if ratio >= threshold:
                union(i, j)

    # Materialise groups: collect memory lists per root, drop singletons.
    groups: dict[int, list[dict]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(memories[i])

    result = [g for g in groups.values() if len(g) >= 2]
    return _sort_groups_deterministically(result)


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
