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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

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


# ---------------------------------------------------------------------------
# Pure core: classify_pool
# ---------------------------------------------------------------------------

def _assume_utc(dt: datetime) -> datetime:
    """Return *dt* with UTC timezone attached if it is naive; return *dt* unchanged otherwise.

    Local copy of the "naive datetimes from our journal/Mem0 are UTC"
    convention shared with ``reconciliation.summary_pool._assume_utc`` — this
    script deliberately has no dependency on the fused_memory package's
    reconciliation internals so it stays a self-contained, load-in-isolation
    ops script (mirrors the sibling copy already in task_knowledge_sync.py).
    """
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def _newest_first_sort_key(item: dict) -> tuple:
    """Sort key placing parseable created_at newest-first, unparseable last.

    Mirrors the oldest-first missing-sorts-last convention used by
    ``reconciliation.summary_pool.enforce_summary_pool_cap``, inverted for a
    newest-first ordering: parseable dates sort by descending timestamp
    (leading tuple element 0), missing/unparseable dates sort after every
    parseable one (leading tuple element 1) — i.e. treated as the oldest.
    """
    raw = item.get('created_at')
    if raw is None:
        return (1, 0.0)
    try:
        dt = _assume_utc(datetime.fromisoformat(raw))
        return (0, -dt.timestamp())
    except (ValueError, TypeError):
        return (1, 0.0)


@dataclass
class PruneDecision:
    """Classification result for one project x pool's cycle-summary members."""
    keep_ids: list[str] = field(default_factory=list)
    delete_ids: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)  # id -> reason


def classify_pool(summaries: list[dict], keep_recent_n: int) -> PruneDecision:
    """Classify a pool's cycle-summary members into keep/delete sets.

    Sorts *summaries* newest-first by ``created_at`` (missing/unparseable
    dates sort last — see :func:`_newest_first_sort_key`). The
    ``keep_recent_n`` newest members are always kept (reason ``'recent'``).
    Among the remainder, any member whose ``content`` carries real
    remediation history (:func:`carries_remediation_history`) is kept
    (reason ``'remediation'``); the rest are marked for deletion (reason
    ``'quiescent_boilerplate'``).

    Args:
        summaries: List of normalized ``{id, created_at, content, metadata}``
            dicts.
        keep_recent_n: Number of most-recent members to unconditionally keep.

    Returns:
        A :class:`PruneDecision` with ``keep_ids``, ``delete_ids``, and a
        per-id ``reasons`` mapping. Pure function — no I/O.
    """
    ordered = sorted(summaries, key=_newest_first_sort_key)

    keep_ids: list[str] = []
    delete_ids: list[str] = []
    reasons: dict[str, str] = {}

    for i, item in enumerate(ordered):
        item_id = item['id']
        if i < keep_recent_n:
            keep_ids.append(item_id)
            reasons[item_id] = 'recent'
        elif carries_remediation_history(item.get('content') or ''):
            keep_ids.append(item_id)
            reasons[item_id] = 'remediation'
        else:
            delete_ids.append(item_id)
            reasons[item_id] = 'quiescent_boilerplate'

    return PruneDecision(keep_ids=keep_ids, delete_ids=delete_ids, reasons=reasons)


# ---------------------------------------------------------------------------
# Pure core: audit report + summary table
# ---------------------------------------------------------------------------

def build_prune_report(
    decisions_by_project_pool: dict[tuple[str, str], PruneDecision],
    applied_ids: set[str],
    dry_run: bool,
    generated_at: str,
) -> dict[str, Any]:
    """Assemble the structured JSON-serialisable prune report.

    Args:
        decisions_by_project_pool: ``{(project_id, pool): PruneDecision}`` —
            one entry per project x pool that was scanned.
        applied_ids: Set of memory ids actually deleted (empty on dry-run).
        dry_run: True when no writes were made.
        generated_at: ISO timestamp string.

    Returns:
        Dict with keys: dry_run, generated_at, projects, pools, totals,
        deletions, preserved_remediation. ``projects`` nests
        ``{project_id: {pool: {scanned, deletable, deleted,
        preserved_remediation}}}``. ``deletions`` is a deterministically
        ordered list of ``{project_id, pool, id, deleted}`` dicts.
        ``preserved_remediation`` lists ``{project_id, pool, id}`` dicts for
        every kept member whose reason was 'remediation' (not just 'recent').
    """
    projects: dict[str, dict[str, Any]] = {}
    pools_seen: set[str] = set()
    deletions: list[dict[str, Any]] = []
    preserved_remediation: list[dict[str, Any]] = []

    for (project_id, pool), decision in sorted(decisions_by_project_pool.items()):
        pools_seen.add(pool)
        scanned = len(decision.keep_ids) + len(decision.delete_ids)
        deletable = len(decision.delete_ids)
        deleted = sum(1 for mid in decision.delete_ids if mid in applied_ids)
        preserved = sum(
            1 for mid in decision.keep_ids if decision.reasons.get(mid) == 'remediation'
        )

        projects.setdefault(project_id, {})[pool] = {
            'scanned': scanned,
            'deletable': deletable,
            'deleted': deleted,
            'preserved_remediation': preserved,
        }

        for mid in sorted(decision.delete_ids):
            deletions.append({
                'project_id': project_id,
                'pool': pool,
                'id': mid,
                'deleted': mid in applied_ids,
            })
        for mid in sorted(decision.keep_ids):
            if decision.reasons.get(mid) == 'remediation':
                preserved_remediation.append({
                    'project_id': project_id,
                    'pool': pool,
                    'id': mid,
                })

    totals: dict[str, Any] = {
        'scanned': sum(p['scanned'] for proj in projects.values() for p in proj.values()),
        'deletable': sum(p['deletable'] for proj in projects.values() for p in proj.values()),
        'deleted': sum(p['deleted'] for proj in projects.values() for p in proj.values()),
        'preserved_remediation': sum(
            p['preserved_remediation'] for proj in projects.values() for p in proj.values()
        ),
    }

    return {
        'dry_run': dry_run,
        'generated_at': generated_at,
        'projects': projects,
        'pools': sorted(pools_seen),
        'totals': totals,
        'deletions': deletions,
        'preserved_remediation': preserved_remediation,
    }


def format_summary_table(report: dict[str, Any]) -> str:
    """Render a human-readable per-(project, pool) summary table from a prune report.

    Produces one row per (project, pool) plus a TOTAL row, mirroring
    ``cleanup_count_snapshots.format_summary_table``.
    """
    projects = report.get('projects', {})
    totals = report.get('totals', {})

    header = (
        f"{'Project':<24} {'Pool':<22} {'Scanned':>8} "
        f"{'Deletable':>10} {'Deleted':>8} {'Preserved':>10}"
    )
    sep = '-' * len(header)
    rows = [header, sep]

    for pid in sorted(projects.keys()):
        for pool in sorted(projects[pid].keys()):
            p = projects[pid][pool]
            rows.append(
                f"{pid:<24} {pool:<22} {p.get('scanned', 0):>8} "
                f"{p.get('deletable', 0):>10} {p.get('deleted', 0):>8} "
                f"{p.get('preserved_remediation', 0):>10}"
            )

    rows.append(sep)
    rows.append(
        f"{'TOTAL':<24} {'':<22} {totals.get('scanned', 0):>8} "
        f"{totals.get('deletable', 0):>10} {totals.get('deleted', 0):>8} "
        f"{totals.get('preserved_remediation', 0):>10}"
    )

    dry_tag = ' [DRY RUN]' if report.get('dry_run') else ''
    rows.insert(0, f"Recon cycle-summary prune report — {report.get('generated_at', '')}{dry_tag}")
    return '\n'.join(rows)


# ---------------------------------------------------------------------------
# Pure helpers: project selection + safety cap
# ---------------------------------------------------------------------------

def select_projects(
    known_map: dict[str, str],
    project_id_filter: str | None,
) -> list[str]:
    """Return the sorted list of project_ids to process.

    Parameters
    ----------
    known_map:
        ``{project_id: project_root}`` from ``build_known_projects_map``.
    project_id_filter:
        When given, restrict to this single project_id. Raises ValueError
        with the list of known ids if the filter is not recognised.

    Returns
    -------
    Sorted list of project_ids.
    """
    if project_id_filter is None:
        return sorted(known_map.keys())
    if project_id_filter not in known_map:
        known_ids = sorted(known_map.keys())
        raise ValueError(
            f'Unknown project_id {project_id_filter!r}. '
            f'Known project ids: {known_ids}'
        )
    return [project_id_filter]


def check_limit_cap(
    per_project_delete_counts: dict[str, int],
    limit: int,
    yes_i_am_sure: bool,
) -> tuple[list[str], bool]:
    """Check whether any project's classified deletions exceed the safety cap.

    Adapted from ``cleanup_count_snapshots.check_limit_cap`` — here the
    counted quantity is deletions about to happen (irreversible), not scan
    volume, so the cap protects against an unexpectedly large one-shot prune.

    Parameters
    ----------
    per_project_delete_counts:
        ``{project_id: deletable_count}`` — total across all pools for that
        project.
    limit:
        Maximum allowed deletable count before aborting.
    yes_i_am_sure:
        When True, override the abort even if projects exceed the limit.

    Returns
    -------
    ``(exceeding_projects, abort)`` where ``exceeding_projects`` lists
    project_ids whose count > limit and ``abort`` is True when any project
    exceeds AND ``yes_i_am_sure`` is False.
    """
    exceeding = [pid for pid, count in per_project_delete_counts.items() if count > limit]
    abort = bool(exceeding) and not yes_i_am_sure
    return exceeding, abort
