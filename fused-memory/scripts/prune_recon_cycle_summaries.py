#!/usr/bin/env python3
"""One-shot prune: collapse the per-cycle reconciliation summary pools (Stage 1
``memory_consolidator`` and legacy pre-1657 Stage 2 ``task_knowledge_sync``)
down to the N most-recent entries per project, preserving any older entry
that carries real remediation history.

Background
----------
Task 1942 adds going-forward pool-cap enforcement for Stage 1 cycle summaries
(mirroring the Stage 2 fix from task 1657/1831) by tagging every NEW per-cycle
summary with ``recon_pool`` metadata and trimming to a small cap on each
cycle.  That fixes going-forward growth, but two pre-existing piles are
invisible to it because they predate the tag:

  1. Stage 1 ``memory_consolidator`` cycle summaries written before this
     task (know_live: ~224 entries), none of which carry ``recon_pool``.
  2. Legacy pre-1657 Stage 2 ``task_knowledge_sync`` cycle summaries that
     predate the Stage 2 pool-cap fix (know_live: ~41 entries) and likewise
     lack the tag.

This script is the one-shot cleanup for both piles.

Delete, not retag
-----------------
Mem0/Qdrant exposes ``delete_memory`` but no in-place payload-update
primitive on this path — the same constraint documented in
``scripts/sweep_orphan_flag_markers.py`` (task-1659), where pre-existing
orphans were deleted rather than re-tagged.  Re-tagging via delete+re-add
would also change ``created_at`` and lose provenance.  So the effective
operation here is PRUNE-to-N: keep the ``--keep-recent`` most-recent
summaries per project x pool, plus any older summary that carries real
remediation history (see ``carries_remediation_history``), and delete the
rest.

Two-phase model
----------------
**Phase 1 — Scan + report (default, --dry-run)**: enumerate every known
project's Stage 1 and legacy Stage 2 cycle-summary pools, classify each
member as keep/delete, and print a structured JSON report + human-readable
summary table.  No writes.

**Phase 2 — Apply (--apply)**: delete the classified delete-set via
``memory.delete_memory`` (best-effort; a failed delete is logged and
excluded, other deletes still proceed).

Usage
-----
  # Dry run (default): print JSON report, touch nothing.
  python scripts/prune_recon_cycle_summaries.py

  # Commit the deletions.
  python scripts/prune_recon_cycle_summaries.py --apply

  # Limit to a single project.
  python scripts/prune_recon_cycle_summaries.py --project-id dark_factory

  # Keep more than the default 2 most-recent summaries per project x pool.
  python scripts/prune_recon_cycle_summaries.py --apply --keep-recent 5

  # Override the per-project scan limit (safety cap).
  python scripts/prune_recon_cycle_summaries.py --apply --limit-per-project 2000

  # Bypass the safety cap (required when a project has > limit entries).
  python scripts/prune_recon_cycle_summaries.py --apply --yes-i-am-sure
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Pure core: carries_remediation_history
# ---------------------------------------------------------------------------

# Markers indicating a cycle produced no mutations (candidates for deletion,
# absent any remediation signal below).
_QUIESCENT_MARKERS: tuple[str, ...] = (
    '0 new episodes',
    '0 mutations',
    'no mutations',
    'quiescent cycle',
)

# Action keywords indicating the cycle actually did something worth
# preserving, even if it also happens to mention a quiescent marker above
# (e.g. "0 new episodes, but 1 flag processed").
_REMEDIATION_KEYWORDS: tuple[str, ...] = (
    'deleted',
    'invalidated',
    'merged',
    'flag processed',
    'edge correct',
    'refreshed entity',
)

# A non-zero mutation/deletion count is a remediation signal even without
# one of the keywords above (e.g. "3 mutations applied").
_NONZERO_MUTATION_RE = re.compile(r'\b[1-9]\d*\s+(?:mutations?|deletions?)\b')


def carries_remediation_history(content: str) -> bool:
    """Return True if *content* shows evidence of real remediation history.

    Fail-safe classifier used by ``classify_pool`` to decide whether an
    older-than-keep-recent cycle summary must be preserved even though it
    predates the ``--keep-recent`` cutoff.  Deletion is irreversible, so this
    function defaults to True (preserve) for anything that is not CLEARLY
    pure-quiescent boilerplate.

    Returns False ONLY when a quiescent marker is present (one of "0 new
    episodes", "0 mutations", "no mutations", "quiescent cycle") AND no
    remediation signal is found.  A remediation signal is either one of the
    action keywords (``deleted``, ``invalidated``, ``merged``,
    ``flag processed``, ``edge correct``, ``refreshed entity``) or a
    non-zero mutation/deletion count (e.g. "3 mutations", "2 deletions").

    Empty, whitespace-only, or otherwise ambiguous content (no quiescent
    marker present at all) also returns True — there is nothing to safely
    classify as boilerplate, so the fail-safe default applies.

    Args:
        content: The cycle-summary memory's raw text content.

    Returns:
        True to preserve (real remediation, or ambiguous/empty content);
        False only for clearly pure-quiescent boilerplate.
    """
    if not content or not content.strip():
        return True

    text = content.lower()
    is_quiescent = any(marker in text for marker in _QUIESCENT_MARKERS)
    has_remediation = (
        any(keyword in text for keyword in _REMEDIATION_KEYWORDS)
        or bool(_NONZERO_MUTATION_RE.search(text))
    )
    if is_quiescent and not has_remediation:
        return False
    return True
